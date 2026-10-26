"""Shared request handling for the API runners (EXPERIMENT.md 6.6, 9).

One retry and timing policy for all API methods: retry on RETRY_STATUSES and transport errors,
honor retry-after, otherwise wait min(2**attempt, 30) s with jitter, at most MAX_ATTEMPTS.
"""

import argparse
import asyncio
import random
import time
import tomllib
from collections import Counter
from collections.abc import Awaitable, Callable
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd

from common import CONFIG, connect, delete_errors, done_eans, eval_items, load, now, save_predictions, save_run

RETRY_STATUSES = {408, 409, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 6
TIMEOUT_S = 60.0
CONCURRENCY = 16
COMMIT_EVERY = 100


class Retryable(Exception):
    def __init__(self, message: str, status: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class Failed(Exception):
    """A request error that is not retried. `raw` is the response body, if any."""

    def __init__(self, message: str, raw: str | None = None):
        super().__init__(message)
        self.raw = raw


def parse_retry_after(headers) -> float | None:
    """Seconds to wait from retry-after-ms or retry-after (seconds or HTTP date)."""
    if (ms := headers.get("retry-after-ms")) is not None:
        try:
            return max(0.0, float(ms) / 1000)
        except ValueError:
            pass
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


Call = Callable[[object], Awaitable[dict]]


async def with_retries(call: Call, item, stats: Counter) -> dict:
    """Run call(item) under the shared retry policy. Returns a prediction row (without keys).

    latency_ms is the time of the successful attempt only.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        t0 = time.perf_counter()
        try:
            row = await call(item)
        except Retryable as e:
            stats[f"retry:{e.status or 'transport'}"] += 1
            if attempt == MAX_ATTEMPTS:
                return dict(status="error", label="", attempts=attempt, error=str(e))
            delay = e.retry_after if e.retry_after is not None else min(2**attempt, 30) * random.uniform(0.5, 1.0)
            await asyncio.sleep(delay)
            continue
        except Failed as e:
            return dict(status="error", label="", attempts=attempt, error=str(e), raw=e.raw)
        except Exception as e:  # a bug in a runner must not stop the other items
            return dict(status="error", label="", attempts=attempt, error=f"{type(e).__name__}: {e}")
        row.update(latency_ms=(time.perf_counter() - t0) * 1000, attempts=attempt)
        return row
    raise AssertionError("unreachable")


def summarize(rows: list[dict], labels: dict[str, str]):
    df = pd.DataFrame(rows)
    print("status:", df.status.value_counts().to_dict())
    ok = df[df.status != "error"]
    if ok.empty:
        return
    correct = (ok.label == ok.ean.map(labels)).sum()
    print(f"accuracy on these items (non-answers wrong): {correct}/{len(ok)} = {correct / len(ok):.3f}")
    first = ok[ok.attempts == 1].latency_ms
    if len(first):
        p50, p90, p99 = np.percentile(first, [50, 90, 99])
        print(f"latency (first-attempt successes, n={len(first)}): "
              f"p50={p50:.0f} ms p90={p90:.0f} ms p99={p99:.0f} ms")
    for col in ["input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "cost_usd"]:
        if col in ok and ok[col].notna().any():
            print(f"{col}: mean={ok[col].mean():.6g} min={ok[col].min():.6g} max={ok[col].max():.6g}")
    print("model:", ok.model.value_counts().to_dict(), "provider:", ok.provider.value_counts().to_dict())


async def run(method: str, variant: str, items: pd.DataFrame, call: Call, *, limit: int | None,
              retry_errors: bool, concurrency: int, notes: dict):
    """Resumable run: skip items stored without a request error, commit every COMMIT_EVERY rows."""
    db = connect()
    if retry_errors:
        print(f"deleted {delete_errors(db, method, variant)} error rows")
    todo = items[~items.ean.isin(done_eans(db, method, variant))]
    if limit is not None:
        todo = todo.head(limit)
    print(f"{method}/{variant}: {len(todo)} items to run ({len(items)} selected)")
    if todo.empty:
        return

    sem = asyncio.Semaphore(concurrency)
    stats, buffer, rows = Counter(), [], []

    async def one(item):
        async with sem:
            row = await with_retries(call, item, stats)
        row.update(ean=item.ean, method=method, variant=variant, created_at=now())
        buffer.append(row)
        rows.append(row)
        if len(buffer) >= COMMIT_EVERY:
            save_predictions(db, buffer[:])
            buffer.clear()
        if len(rows) % 500 == 0:
            print(f"  {len(rows)}/{len(todo)}")

    # Open the connections before the timed run: first requests on a new connection take seconds
    # (TLS handshake), and section 8.4 leaves them out of the latency. Results are not stored.
    first = next(todo.itertuples())
    await asyncio.gather(*(with_retries(call, first, Counter()) for _ in range(concurrency)))

    started_at = now()
    t0 = time.perf_counter()
    await asyncio.gather(*(one(item) for item in todo.itertuples()))
    wall_s = time.perf_counter() - t0
    save_predictions(db, buffer)
    save_run(db, method=method, variant=variant, phase="infer", device="api", batch_size=1,
             n_items=len(todo), wall_s=wall_s,
             notes={**notes, "concurrency": concurrency, "warmup_requests": concurrency,
                    "started_at": started_at, "retries": dict(stats)})
    print(f"wall {wall_s:.1f} s, {len(todo) / wall_s:.1f} items/s, retries: {dict(stats)}")
    summarize(rows, dict(zip(items.ean, items.category)))


def load_config(name: str) -> dict:
    with open(CONFIG / f"{name}.toml", "rb") as f:
        return tomllib.load(f)


def parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--items", choices=["tune", "eval"], required=True,
                    help="tune: the 500 tune items; eval: all 11,087 evaluation items")
    ap.add_argument("--prompt", help="prompt variant from the config (default: the frozen `main`)")
    ap.add_argument("--variant", help="results variant (default: tune-<prompt> or main)")
    ap.add_argument("--limit", type=int, help="run at most N items (smoke test)")
    ap.add_argument("--retry-errors", action="store_true", help="delete error rows and run them again")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    return ap


def select(args, main: str) -> tuple[pd.DataFrame, str, str]:
    """Items, prompt variant and results variant for the command-line arguments."""
    prompt = args.prompt or main
    if not prompt:
        raise SystemExit("no --prompt given and no frozen `main` prompt in the config")
    df = load()
    if args.items == "tune":
        return df[df.tune], prompt, args.variant or f"tune-{prompt}"
    if not main:
        raise SystemExit("freeze the config (set `main`) before a run on the evaluation items")
    if prompt != main and not args.variant:
        raise SystemExit("a non-frozen prompt on the evaluation items needs an explicit --variant")
    return eval_items(df), prompt, args.variant or "main"
