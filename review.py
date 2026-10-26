"""Blind label review (EXPERIMENT.md 8.3).

  items    Write the review items for the review page: every main-split test item where at
           least one method's main prediction differs from the label. Items already written
           by an earlier call (out/review/items-*.json) are left out, so the call can be
           repeated as more methods finish. Each new item gets a fixed random order key.
  collect  Turn the page's verdict documents (JSON files from the page's database) into
           data/test-review.csv.

The reviewer sees the input, the brand, the label and the set of predicted categories, never
which method predicted what.
"""

import argparse
import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from common import CATEGORIES, DATA, RESULTS_DB, RESULTS_DIR, load

REVIEW_DIR = RESULTS_DIR / "review"
CHUNK = 200
VERDICTS = ["label_correct", "label_wrong", "spec_conflict", "ambiguous"]


def order_key(ean: str) -> int:
    # Fixed pseudo-random order: a reviewer who stops early has reviewed a random sample.
    return int(hashlib.sha256(f"review-{ean}".encode()).hexdigest()[:12], 16)


def existing_items() -> dict[str, dict]:
    items = {}
    for f in sorted(REVIEW_DIR.glob("items-*.json")):
        for item in json.loads(f.read_text())["items"]:
            items[item["ean"]] = item
    return items


def cmd_items(batch: str):
    df = load()
    test = df[df.split == "test"].set_index("ean")
    p = pd.read_sql("SELECT ean, method, label FROM predictions WHERE variant = 'main' AND status = 'ok'",
                    sqlite3.connect(RESULTS_DB))
    p = p[p.ean.isin(test.index)]
    print("methods with main predictions:", p.groupby("method").size().to_dict())
    preds = p.groupby("ean").label.apply(set)
    have = existing_items()
    new = []
    for ean, labels in preds.items():
        row = test.loc[ean]
        others = labels - {row.category}
        if not others:
            continue
        candidates = [c for c in CATEGORIES if c in others]
        if ean in have:
            if set(have[ean]["candidates"]) != set(candidates):
                print(f"note: {ean} has new predicted categories {candidates}; earlier item kept")
            continue
        new.append({"ean": ean, "key": order_key(ean), "product": row["name"],
                    "quantity": f"{row.quantity} {row.unit}", "brand": row.brand, "label": row.category,
                    "candidates": candidates, "batch": batch})
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(new), CHUNK):
        path = REVIEW_DIR / f"items-{batch}-{i // CHUNK}.json"
        path.write_text(json.dumps({"items": new[i:i + CHUNK]}, ensure_ascii=False))
        print(f"wrote {path.relative_to(RESULTS_DIR.parent)} ({len(new[i:i + CHUNK])} items)")
    print(f"{len(new)} new items, {len(have) + len(new)} in total")


def verdict(label: str, choice: str, convention: bool) -> tuple[str, str]:
    if choice == "ambiguous":
        return "ambiguous", ""
    if choice == label:
        return "label_correct", label
    return ("spec_conflict" if convention else "label_wrong"), choice


def cmd_collect(verdicts_dir: Path):
    items = existing_items()
    rows = []
    for f in sorted(verdicts_dir.rglob("*.json")):
        d = json.loads(f.read_text())
        d = d.get("data", d)  # tolerate a wrapper around the document body
        ean = d.get("ean") or f.stem
        if ean not in items or d.get("choice") not in CATEGORIES + ["ambiguous"]:
            print(f"skipped {f.name}")
            continue
        it = items[ean]
        v, correct = verdict(it["label"], d["choice"], bool(d.get("convention")))
        rows.append({"ean": ean, "product": it["product"], "quantity": it["quantity"], "label": it["label"],
                     "predicted": " ".join(it["candidates"]), "verdict": v, "correct_category": correct,
                     "note": d.get("note", ""), "reviewed_at": d.get("updated_at", "")})
    out = pd.DataFrame(rows).sort_values("ean")
    out.to_csv(DATA / "test-review.csv", index=False)
    print(f"wrote data/test-review.csv: {len(out)} of {len(items)} items reviewed")
    print(out.verdict.value_counts().reindex(VERDICTS, fill_value=0).to_dict())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("items")
    a.add_argument("--batch", default=datetime.now(UTC).strftime("b%H%M"), help="name of this batch of items")
    c = sub.add_parser("collect")
    c.add_argument("verdicts_dir", type=Path, help="directory with the verdict JSON files")
    args = ap.parse_args()
    if args.cmd == "items":
        cmd_items(args.batch)
    else:
        cmd_collect(args.verdicts_dir)


if __name__ == "__main__":
    main()
