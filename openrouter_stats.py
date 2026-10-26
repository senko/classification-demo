"""Estimate how much OpenRouter adds to the Haiku and Qwen latency (EXPERIMENT.md 8.4).

For a random sample of first-attempt successes of the main run, fetch
GET /api/v1/generation?id=<id> and store OpenRouter's `latency` (time to first token) and
`generation_time` (ms) next to the client-side latency, in table `openrouter_generation`.
Client latency - generation_time estimates the share of OpenRouter and the network.
Only these numbers are stored; the generation record also has account fields.
"""

import argparse
import asyncio
import json
import os

import httpx

from api import CONCURRENCY, TIMEOUT_S
from common import connect, load_env

SAMPLE = 200

SCHEMA = """
CREATE TABLE IF NOT EXISTS openrouter_generation (
    ean TEXT NOT NULL,
    method TEXT NOT NULL,
    variant TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    client_latency_ms REAL,
    or_latency_ms REAL,
    or_generation_time_ms REAL,
    provider TEXT,
    PRIMARY KEY (ean, method, variant)
);
"""


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="main")
    ap.add_argument("--sample", type=int, default=SAMPLE)
    args = ap.parse_args()
    load_env()
    db = connect()
    db.executescript(SCHEMA)
    sem = asyncio.Semaphore(CONCURRENCY // 2)

    async with httpx.AsyncClient(base_url="https://openrouter.ai/api/v1", timeout=TIMEOUT_S,
                                 headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}) as client:

        async def fetch(ean, method, raw, latency_ms):
            gid = json.loads(raw)["id"]
            for attempt in range(5):
                async with sem:
                    r = await client.get("/generation", params={"id": gid})
                if r.status_code not in (408, 429) and r.status_code < 500:
                    break
                await asyncio.sleep(2 ** attempt)
            if r.is_error:
                print(f"skipped {gid}: HTTP {r.status_code}")
                return None
            d = r.json()["data"]
            return (ean, method, args.variant, gid, latency_ms, d.get("latency"), d.get("generation_time"),
                    d.get("provider_name"))

        for method in ["haiku", "qwen"]:
            rows = db.execute(
                "SELECT ean, method, raw, latency_ms FROM predictions WHERE method = ? AND variant = ? "
                "AND status = 'ok' AND attempts = 1 ORDER BY ean", (method, args.variant)).fetchall()
            if not rows:
                print(f"{method}: no {args.variant} rows")
                continue
            # Deterministic sample: every k-th item in ean order.
            step = max(1, len(rows) // args.sample)
            sample = rows[::step][:args.sample]
            stats = [s for s in await asyncio.gather(*(fetch(*r) for r in sample)) if s]
            db.executemany("INSERT OR REPLACE INTO openrouter_generation VALUES (?, ?, ?, ?, ?, ?, ?, ?)", stats)
            db.commit()
            print(f"{method}: stored {len(stats)} generation records")


if __name__ == "__main__":
    asyncio.run(main())
