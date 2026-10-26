"""Your results next to the reference results: main test set quality, median latency, cost.

Usage: uv run compare.py [REPORT_DIR]   (default: $RESULTS_DIR/report, normally out/report)
Run report.py first. The reference is results/report/.
"""

import sys
from pathlib import Path

import pandas as pd

from common import PUBLISHED, RESULTS_DIR


def numbers(report: Path) -> pd.DataFrame:
    quality = pd.read_csv(report / "main-quality.csv").set_index("method")[["accuracy", "macro-F1", "errors"]]
    speed = pd.read_csv(report / "speed.csv").drop_duplicates("method").set_index("method")
    quality["p50 ms"] = speed["latency p50 / p90 / p99 (ms)"].str.split(" / ").str[0].astype(float)
    if (report / "cost.csv").exists():
        quality["$ / 1K"] = pd.read_csv(report / "cost.csv").set_index("method")["cost per 1,000 items (USD)"]
    return quality


def main():
    yours_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR / "report"
    if not (yours_dir / "main-quality.csv").exists():
        raise SystemExit(f"no report in {yours_dir}: run report.py first")
    ref, yours = numbers(PUBLISHED / "report"), numbers(yours_dir)
    table = ref.join(yours, how="outer", lsuffix=" (ref)", rsuffix=" (yours)")
    table = table.reindex(ref.index.union(yours.index, sort=False))
    table = table[[f"{c} ({s})" for c in ref.columns for s in ["ref", "yours"] if f"{c} ({s})" in table]]
    print(f"reference: {PUBLISHED / 'report'}   yours: {yours_dir}\n")
    print(table.to_string(na_rep="–", float_format=lambda v: f"{v:.4g}"))
    print("\nTF-IDF and BERTić on the same hardware and software should match exactly; Jev and Luna match while "
          "the vendors serve the same model; Haiku and Qwen vary by a few answers. Latency and cost depend on "
          "your hardware, network and the current prices.")


if __name__ == "__main__":
    main()
