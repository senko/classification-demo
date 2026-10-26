"""Shared data loading, input formatting and results database (EXPERIMENT.md sections 3, 4, 9)."""

import json
import os
import resource
import sqlite3
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CONFIG = ROOT / "config"
# Working directory of a run (gitignored): results.db, pricing snapshots, models, report.
# The committed reference results are in results/ and only `report.py --publish` writes there.
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR") or ROOT / "out")
RESULTS_DB = RESULTS_DIR / "results.db"
PUBLISHED = ROOT / "results"


def read_csv(path: Path) -> pd.DataFrame:
    # Strings only: barcodes keep leading zeros, empty brands stay "" (not NaN).
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_categories() -> pd.DataFrame:
    return read_csv(DATA / "categories.csv")


CATEGORIES: list[str] = load_categories().id.tolist()


def load_products() -> pd.DataFrame:
    p = read_csv(DATA / "products.csv")
    l = read_csv(DATA / "labelled-products.csv")
    return p.merge(l, left_on="barcode", right_on="ean", validate="one_to_one")


def load() -> pd.DataFrame:
    """Products with labels and the committed split columns (split, split_brand, tune)."""
    s = read_csv(DATA / "split.csv")
    df = load_products().merge(s, on="ean", validate="one_to_one")
    df["tune"] = df.tune == "1"
    return df


def eval_items(df: pd.DataFrame) -> pd.DataFrame:
    """Union of all evaluation items for the zero-shot methods (section 7)."""
    return df[(df.split == "test") | (df.split_brand == "test") | df.tune]


# Model inputs (section 3). name, quantity and unit go in as they are in the CSV.
# `row` is a row from df.itertuples(). Do not pass a row from df.apply(axis=1): there,
# row.name is the index label, not the name column.

def text_input(row) -> str:
    """TF-IDF and BERTić."""
    return f"{row.name} {row.quantity} {row.unit}"


def prompt_input(row) -> str:
    """Luna, Haiku, Qwen."""
    return f"Product: {row.name}\nQuantity: {row.quantity} {row.unit}"


def jev_state(row) -> dict:
    return {"product": row.name, "quantity": f"{row.quantity} {row.unit}"}


# Local resources (section 8.6)

def peak_rss_mb() -> float:
    # Peak of the whole process so far (Linux reports KiB).
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class GpuProcessMemory:
    """Peak GPU memory of this process from nvidia-smi, sampled every 0.5 s while in the `with`
    block. Includes the CUDA context, which torch.cuda.max_memory_reserved() does not count."""

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.peak_mb = 0.0
        self._stop = threading.Event()

    def _sample(self):
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        ).stdout
        for line in out.splitlines():
            pid, mb = (x.strip() for x in line.split(","))
            if pid == str(os.getpid()):
                self.peak_mb = max(self.peak_mb, float(mb))

    def _loop(self):
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sample()


def summary(variant: str, test: pd.DataFrame, pred: list[str]):
    """Print accuracy, macro-F1 and error count of predictions against the labels."""
    from sklearn.metrics import f1_score

    errors = sum(t != p for t, p in zip(test.category, pred))
    f1 = f1_score(test.category, pred, average="macro", labels=CATEGORIES)
    print(f"{variant:14s} n={len(test):5d} acc={1 - errors / len(test):.3f} "
          f"macro-F1={f1:.3f} errors={errors}")


def load_env():
    """Put KEY=VALUE lines from .env into os.environ without overriding the environment."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip().removeprefix("export "), value.strip().strip("\"'"))


# Results database (section 9)

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    ean TEXT NOT NULL,
    method TEXT NOT NULL,
    variant TEXT NOT NULL,
    status TEXT NOT NULL,          -- ok, refusal, unparsable, error
    label TEXT NOT NULL,           -- empty for a non-answer
    probs TEXT,                    -- JSON {category: p}, NULL for the LLMs
    confidence REAL,
    input_tokens INTEGER,
    cached_tokens INTEGER,
    output_tokens INTEGER,
    reasoning_tokens INTEGER,
    cost_usd REAL,                 -- only where the API reports it
    latency_ms REAL,
    attempts INTEGER,
    model TEXT,
    provider TEXT,
    error TEXT NOT NULL DEFAULT '',
    raw TEXT,                      -- response body, never headers
    created_at TEXT NOT NULL,
    PRIMARY KEY (ean, method, variant)
);
CREATE TABLE IF NOT EXISTS runs (
    method TEXT NOT NULL,
    variant TEXT NOT NULL,
    phase TEXT NOT NULL,
    device TEXT,
    batch_size INTEGER,
    n_items INTEGER,
    wall_s REAL,
    peak_rss_mb REAL,
    peak_vram_alloc_mb REAL,
    peak_vram_reserved_mb REAL,
    peak_vram_process_mb REAL,
    notes TEXT,
    created_at TEXT NOT NULL
);
"""

PREDICTION_COLUMNS = [
    "ean", "method", "variant", "status", "label", "probs", "confidence",
    "input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "cost_usd",
    "latency_ms", "attempts", "model", "provider", "error", "raw", "created_at",
]


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def connect(path: Path = RESULTS_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    return db


def save_predictions(db: sqlite3.Connection, rows: list[dict]):
    """Insert or replace prediction rows and commit. Missing columns are NULL."""
    for r in rows:
        r.setdefault("error", "")
        r.setdefault("created_at", now())
        if isinstance(r.get("probs"), dict):
            r["probs"] = json.dumps(r["probs"])
    placeholders = ", ".join(f":{c}" for c in PREDICTION_COLUMNS)
    db.executemany(
        f"INSERT OR REPLACE INTO predictions ({', '.join(PREDICTION_COLUMNS)}) VALUES ({placeholders})",
        [{c: r.get(c) for c in PREDICTION_COLUMNS} for r in rows],
    )
    db.commit()


def save_run(db: sqlite3.Connection, **fields):
    fields.setdefault("created_at", now())
    if isinstance(fields.get("notes"), dict):
        fields["notes"] = json.dumps(fields["notes"])
    cols = ", ".join(fields)
    db.execute(f"INSERT INTO runs ({cols}) VALUES ({', '.join('?' * len(fields))})", list(fields.values()))
    db.commit()


def done_eans(db: sqlite3.Connection, method: str, variant: str) -> set[str]:
    """Items stored without a request error (resume skips these)."""
    cur = db.execute(
        "SELECT ean FROM predictions WHERE method = ? AND variant = ? AND status != 'error'",
        (method, variant),
    )
    return {r[0] for r in cur}


def delete_errors(db: sqlite3.Connection, method: str, variant: str) -> int:
    cur = db.execute(
        "DELETE FROM predictions WHERE method = ? AND variant = ? AND status = 'error'",
        (method, variant),
    )
    db.commit()
    return cur.rowcount
