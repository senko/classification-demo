"""TF-IDF + logistic regression: training, prediction, timing (EXPERIMENT.md 6.1, 8.4, 8.6).

Variants:
  main               train on split=train, predict split=test; timed (batch 1 and bulk)
  brand              train on split_brand=train, predict split_brand=test
  lc-<size>-s<seed>  train on a learning-curve subset, predict split=test
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import (CATEGORIES, DATA, connect, load, peak_rss_mb, save_predictions, save_run, summary,
                    text_input)

METHOD = "tfidf"
MODEL = f"sklearn-{sklearn.__version__}"


def texts(df: pd.DataFrame) -> list[str]:
    return [text_input(r) for r in df.itertuples()]


def fit(train: pd.DataFrame):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
    clf = LogisticRegression(C=10, max_iter=3000)
    clf.fit(vec.fit_transform(texts(train)), train.category)
    return vec, clf


def to_rows(test: pd.DataFrame, variant: str, probs: np.ndarray, classes, latency_ms=None) -> list[dict]:
    # Full distribution over all 10 categories; a category missing from training gets 0.
    idx = {c: i for i, c in enumerate(classes)}
    rows = []
    for i, ean in enumerate(test.ean):
        p = {c: float(probs[i, idx[c]]) if c in idx else 0.0 for c in CATEGORIES}
        label = max(p, key=p.get)
        rows.append(dict(
            ean=ean, method=METHOD, variant=variant, status="ok", label=label, probs=p,
            confidence=p[label], latency_ms=None if latency_ms is None else latency_ms[i],
            attempts=1, model=MODEL, provider="local",
        ))
    return rows


def run_main(db, df: pd.DataFrame):
    train, test = df[df.split == "train"], df[df.split == "test"]
    notes = {"cpu_threads": os.cpu_count()}

    t0 = time.perf_counter()
    vec, clf = fit(train)
    save_run(db, method=METHOD, variant="main", phase="train", device="cpu", n_items=len(train),
             wall_s=time.perf_counter() - t0, peak_rss_mb=peak_rss_mb(), notes=notes)

    # Batch size 1, items one after another, model loaded.
    test_texts = texts(test)
    latency_ms = []
    t0 = time.perf_counter()
    for t in test_texts:
        t1 = time.perf_counter()
        clf.predict_proba(vec.transform([t]))
        latency_ms.append((time.perf_counter() - t1) * 1000)
    save_run(db, method=METHOD, variant="main", phase="infer", device="cpu", batch_size=1,
             n_items=len(test), wall_s=time.perf_counter() - t0, peak_rss_mb=peak_rss_mb(),
             notes=notes)

    # Whole test set in one batch.
    t0 = time.perf_counter()
    probs = clf.predict_proba(vec.transform(test_texts))
    save_run(db, method=METHOD, variant="main", phase="infer", device="cpu",
             batch_size=len(test), n_items=len(test), wall_s=time.perf_counter() - t0,
             peak_rss_mb=peak_rss_mb(), notes=notes)

    rows = to_rows(test, "main", probs, clf.classes_, latency_ms)
    save_predictions(db, rows)
    summary("main", test, [r["label"] for r in rows])
    print(f"{'':14s} latency p50={np.percentile(latency_ms, 50):.2f} ms "
          f"p99={np.percentile(latency_ms, 99):.2f} ms")


def run_variant(db, variant: str, train: pd.DataFrame, test: pd.DataFrame):
    t0 = time.perf_counter()
    vec, clf = fit(train)
    save_run(db, method=METHOD, variant=variant, phase="train", device="cpu", n_items=len(train),
             wall_s=time.perf_counter() - t0, peak_rss_mb=peak_rss_mb())
    rows = to_rows(test, variant, clf.predict_proba(vec.transform(texts(test))), clf.classes_)
    save_predictions(db, rows)
    summary(variant, test, [r["label"] for r in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="*", choices=["main", "brand", "lc"], help="default: all")
    args = ap.parse_args()
    args.parts = args.parts or ["main", "brand", "lc"]

    df = load()
    db = connect()
    if "main" in args.parts:
        run_main(db, df)
    if "brand" in args.parts:
        run_variant(db, "brand", df[df.split_brand == "train"], df[df.split_brand == "test"])
    if "lc" in args.parts:
        test = df[df.split == "test"]
        subsets = pd.read_csv(DATA / "train-subsets.csv", dtype={"ean": str})
        for (size, seed), g in subsets.groupby(["size", "seed"]):
            run_variant(db, f"lc-{size}-s{seed}", df[df.ean.isin(g.ean)], test)


if __name__ == "__main__":
    main()
