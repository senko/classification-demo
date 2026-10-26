"""Metrics, tables and charts from $RESULTS_DIR/results.db (EXPERIMENT.md section 8).

Writes report.md, CSV tables and SVG charts to $RESULTS_DIR/report/. Runs without API keys.
A method appears in a scenario only when it has a prediction for every item of the scenario.

--publish copies the report, the price snapshots, the predictions without response bodies
(predictions.csv.gz) and results.db (if < 50 MB) to results/.
"""

import argparse
import itertools
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import tomllib

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402

from common import CATEGORIES, CONFIG, DATA, PUBLISHED, RESULTS_DB, RESULTS_DIR, load  # noqa: E402

METHODS = {"tfidf": "TF-IDF + LR", "bertic": "BERTić", "jev": "Jev", "luna": "Luna",
           "haiku": "Haiku 5.5", "qwen": "Qwen3.8 27B"}
TRAINED = ["tfidf", "bertic"]
ZERO_SHOT = ["jev", "luna", "haiku", "qwen"]
N_BOOT = 10_000
THRESHOLDS = [0.5, 0.7, 0.9, 0.95]
MAX_PUBLISHED_DB_MB = 50
# predictions.csv.gz: no response bodies and no per-category probabilities (they make the file 6x larger)
PUBLISHED_COLUMNS = ["ean", "method", "variant", "status", "label", "confidence", "input_tokens", "cached_tokens",
                     "output_tokens", "reasoning_tokens", "cost_usd", "latency_ms", "attempts", "model", "provider"]

# Chart colors: categorical slots 1-6 of the dataviz reference palette, light mode, in fixed order.
COLORS = dict(zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


# Data

def hardware() -> str:
    """CPU, RAM and GPU model of this machine (Linux; other systems give less)."""
    cpu = platform.processor() or "unknown CPU"
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    parts = [f"{cpu} ({os.cpu_count()} threads)"]
    try:
        parts.append(f"{os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30:.0f} GiB RAM")
    except (ValueError, OSError, AttributeError):
        pass
    try:
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
        parts += [g.replace(", ", " ") for g in gpus.splitlines()]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ", ".join(parts)


def load_predictions(db) -> pd.DataFrame:
    p = pd.read_sql(
        "SELECT ean, method, variant, status, label, probs, confidence, input_tokens, cached_tokens, "
        "output_tokens, reasoning_tokens, cost_usd, latency_ms, attempts, model, provider "
        "FROM predictions WHERE variant != 'smoke'", db)
    p["label"] = p.label.fillna("").where(p.status == "ok", "")
    return p


def scenario(preds: pd.DataFrame, items: pd.DataFrame, variants: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Predictions per method, indexed and ordered like `items`; methods with missing items are left out."""
    out = {}
    for m, v in variants.items():
        g = preds[(preds.method == m) & (preds.variant == v)].set_index("ean").reindex(items.ean)
        if len(g) and g.status.notna().all():
            out[m] = g
    return out


def conf_array(g: pd.DataFrame) -> np.ndarray:
    # Non-answers get -1: below every real confidence, so they are never accepted.
    return g.confidence.where(g.status == "ok").fillna(-1.0).to_numpy(float)


def probs_matrix(g: pd.DataFrame) -> np.ndarray | None:
    if g.probs.isna().all():
        return None
    rows = [json.loads(p) if isinstance(p, str) else {} for p in g.probs]
    return np.array([[r.get(c, 0.0) for c in CATEGORIES] for r in rows])


# Metrics

def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return precision_recall_fscore_support(y, pred, labels=CATEGORIES, average="macro", zero_division=0)[2]


def bootstrap(y: np.ndarray, preds: dict[str, np.ndarray], seed: int = 0):
    """Paired bootstrap over items (the same resamples for every method).
    Returns per resample and method: error count and macro-F1."""
    n, k = len(y), len(CATEGORIES)
    blocks = []
    for p in preds.values():
        cols = [p != y]
        cols += [(p == c) & (y == c) for c in CATEGORIES]
        cols += [(p == c) & (y != c) for c in CATEGORIES]
        cols += [(y == c) & (p != c) for c in CATEGORIES]
        blocks.append(np.stack(cols, axis=1))
    m = np.concatenate(blocks, axis=1).astype(np.float32)
    rng = np.random.default_rng(seed)
    errors, f1 = [], []
    for start in range(0, N_BOOT, 500):
        w = rng.multinomial(n, np.full(n, 1 / n), size=min(500, N_BOOT - start)).astype(np.float32)
        s = (w @ m).reshape(len(w), len(preds), 1 + 3 * k)
        tp, fp, fn = s[..., 1:1 + k], s[..., 1 + k:1 + 2 * k], s[..., 1 + 2 * k:]
        denom = 2 * tp + fp + fn
        f1.append(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0).mean(axis=-1))
        errors.append(s[..., 0])
    return np.concatenate(errors), np.concatenate(f1)


def coverage_at(conf: np.ndarray, correct: np.ndarray, max_error: float = 0.01) -> float:
    """Largest share of items, accepted in order of confidence, with error rate <= max_error.
    Items with the same confidence are accepted or rejected together."""
    order = np.argsort(-conf, kind="stable")
    c, ok = conf[order], correct[order]
    ends = np.r_[np.nonzero(np.diff(c))[0], len(c) - 1]
    n = ends + 1
    valid = np.cumsum(~ok)[ends] / n <= max_error
    return n[valid].max() / len(c) if valid.any() else 0.0


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over answered items (confidence in [0, 1])."""
    answered = conf >= 0
    c, ok = np.clip(conf[answered], 0, 1), correct[answered]
    idx = np.minimum((c * bins).astype(int), bins - 1)
    return sum(abs(c[idx == b].mean() - ok[idx == b].mean()) * (idx == b).mean()
               for b in range(bins) if (idx == b).any())


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def quality(items: pd.DataFrame, sc: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Main quality table and pairwise comparison table for one scenario."""
    y = items.category.to_numpy()
    preds = {m: g.label.to_numpy() for m, g in sc.items()}
    errs, f1s = bootstrap(y, preds)
    rows = []
    for j, (m, g) in enumerate(sc.items()):
        correct = preds[m] == y
        conf = conf_array(g)
        probs = probs_matrix(g)
        top2 = None
        if probs is not None:
            best2 = np.argsort(-probs, axis=1)[:, :2]
            top2 = np.mean([(CATEGORIES.index(t) in b) and ok for t, b, ok in
                            zip(y, best2, g.status == "ok")])
        lo, hi = np.percentile(errs[:, j], [2.5, 97.5])
        f_lo, f_hi = np.percentile(f1s[:, j], [2.5, 97.5])
        rows.append({
            "method": METHODS[m], "n": len(y), "errors": int((~correct).sum()),
            "errors 95% CI": f"{lo:.0f}-{hi:.0f}", "error rate": f"{(~correct).mean():.2%}",
            "accuracy": round(correct.mean(), 4), "macro-F1": round(macro_f1(y, preds[m]), 4),
            "macro-F1 95% CI": f"{f_lo:.3f}-{f_hi:.3f}", "non-answers": int((g.status != "ok").sum()),
            "coverage at <=1% error": f"{coverage_at(conf, correct):.1%}",
            "top-2 accuracy": "–" if top2 is None else round(top2, 4),
            "ECE": round(ece(conf, correct), 4),
        })
    pairs, pvalues = [], []
    methods = list(sc)
    for a, b in itertools.combinations(range(len(methods)), 2):
        ca, cb = preds[methods[a]] == y, preds[methods[b]] == y
        only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
        p = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
        diff = errs[:, a] - errs[:, b]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        pairs.append({"A": METHODS[methods[a]], "B": METHODS[methods[b]],
                      "errors A - B": int((~ca).sum() - (~cb).sum()), "95% CI": f"{lo:.0f} to {hi:.0f}",
                      "only A right": only_a, "only B right": only_b, "McNemar p": p})
        pvalues.append(p)
    for row, adj in zip(pairs, holm(pvalues)):
        row["Holm p"] = round(adj, 4)
        row["McNemar p"] = round(row["McNemar p"], 4)
        row["verdict"] = "not distinguishable" if adj >= 0.05 else \
            (f"{row['A']} better" if row["errors A - B"] < 0 else f"{row['B']} better")
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def confidence_table(items: pd.DataFrame, sc: dict[str, pd.DataFrame]) -> pd.DataFrame:
    y = items.category.to_numpy()
    rows = []
    for m, g in sc.items():
        conf, correct = conf_array(g), g.label.to_numpy() == y
        for t in THRESHOLDS:
            covered = conf >= t
            rows.append({"method": METHODS[m], "confidence >=": t, "coverage": f"{covered.mean():.1%}",
                         "error among covered": f"{(~correct[covered]).mean():.2%}" if covered.any() else "–",
                         "routed to review": f"{1 - covered.mean():.1%}"})
    return pd.DataFrame(rows)


def per_class(items: pd.DataFrame, sc: dict[str, pd.DataFrame]) -> pd.DataFrame:
    y = items.category.to_numpy()
    rows = []
    for m, g in sc.items():
        p, r, f, s = precision_recall_fscore_support(y, g.label.to_numpy(), labels=CATEGORIES, zero_division=0)
        tp = [int(((g.label.to_numpy() == c) & (y == c)).sum()) for c in CATEGORIES]
        for i, c in enumerate(CATEGORIES):
            rows.append({"method": METHODS[m], "category": c, "support": int(s[i]), "correct": tp[i],
                         "precision": round(p[i], 3), "recall": round(r[i], 3), "F1": round(f[i], 3)})
    return pd.DataFrame(rows)


def agreement(items: pd.DataFrame, sc: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pairwise agreement; agreed items against the label (likely label errors); on disagreement,
    which side the label takes."""
    y = items.category.to_numpy()
    rows = []
    for a, b in itertools.combinations(sc, 2):
        pa, pb = sc[a].label.to_numpy(), sc[b].label.to_numpy()
        agree = (pa == pb) & (pa != "")
        dis = ~agree
        rows.append({"A": METHODS[a], "B": METHODS[b], "agreement": f"{agree.mean():.1%}",
                     "agreed, label differs": int((agree & (pa != y)).sum()),
                     "disagreed": int(dis.sum()), "label = A": int((dis & (pa == y)).sum()),
                     "label = B": int((dis & (pb == y)).sum()),
                     "label = neither": int((dis & (pa != y) & (pb != y)).sum())})
    return pd.DataFrame(rows)


def write_confusions(items: pd.DataFrame, sc: dict[str, pd.DataFrame], out, prefix: str):
    y = items.category.to_numpy()
    for m, g in sc.items():
        cm = confusion_matrix(y, g.label.to_numpy(), labels=CATEGORIES + [""])
        pd.DataFrame(cm[:-1], index=CATEGORIES, columns=CATEGORIES + ["(none)"]).rename_axis("label").to_csv(
            out / f"{prefix}-confusion-{m}.csv")


# Learning curve

def learning_curve(preds: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    test = df[df.split == "test"]
    sizes = pd.read_csv(DATA / "train-subsets.csv", dtype={"ean": str}).groupby(["size", "seed"]).size()
    rows = []
    variants = {(size, seed): f"lc-{size}-s{seed}" for size, seed in sizes.index}
    for m in TRAINED:
        for (size, seed), v in variants.items():
            g = scenario(preds, test, {m: v}).get(m)
            if g is not None:
                rows.append({"method": m, "size": size, "seed": seed, "train items": int(sizes[size, seed]),
                             "accuracy": (g.label.to_numpy() == test.category.to_numpy()).mean(),
                             "macro-F1": macro_f1(test.category.to_numpy(), g.label.to_numpy())})
        g = scenario(preds, test, {m: "main"}).get(m)
        if g is not None:
            rows.append({"method": m, "size": "all", "seed": 42, "train items": int((df.split == "train").sum()),
                         "accuracy": (g.label.to_numpy() == test.category.to_numpy()).mean(),
                         "macro-F1": macro_f1(test.category.to_numpy(), g.label.to_numpy())})
    return pd.DataFrame(rows)


def learning_curve_chart(lc: pd.DataFrame, zero_shot: dict[str, float], metric: str, path):
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 11})
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for m in TRAINED:
        d = lc[lc.method == m]
        if d.empty:
            continue
        ax.scatter(d["train items"], d[metric], s=18, color=COLORS[m], alpha=0.45, linewidths=0, zorder=3)
        mean = d.groupby("train items", sort=True)[metric].mean()
        ax.plot(mean.index, mean.values, color=COLORS[m], lw=2, marker="o", ms=8, mec=SURFACE, mew=2,
                label=f"{METHODS[m]} (trained; dots: 5 seeds)", zorder=4)
    for m, v in zero_shot.items():
        ax.axhline(v, color=COLORS[m], lw=2, ls=(0, (5, 3)), label=f"{METHODS[m]} (zero-shot, 0 labels)", zorder=2)
        ax.annotate(f"{METHODS[m]} {v:.3f}", xy=(1.0, v), xycoords=("axes fraction", "data"), xytext=(4, 0),
                    textcoords="offset points", va="center", color=TEXT_2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Training labels (log scale)", color=TEXT_2)
    ax.set_ylabel(f"{metric} on the test set", color=TEXT_2)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=TEXT_2)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, loc="lower right", labelcolor=TEXT, fontsize=9)
    ax.set_title(f"Learning curve: {metric} by number of training labels", color=TEXT, loc="left")
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


# Speed, cost, resources

def speed_table(db, preds: pd.DataFrame) -> pd.DataFrame:
    runs = pd.read_sql("SELECT * FROM runs WHERE variant = 'main' AND phase = 'infer'", db)
    rows = []
    for m in ZERO_SHOT:
        g = preds[(preds.method == m) & (preds.variant == "main") & (preds.status == "ok") & (preds.attempts == 1)]
        r = runs[runs.method == m].sort_values("n_items").tail(1)
        if g.empty or r.empty:
            continue
        p50, p90, p99 = np.percentile(g.latency_ms, [50, 90, 99])
        notes = json.loads(r.notes.iloc[0])
        rows.append({"method": METHODS[m], "device": "API", "latency p50 / p90 / p99 (ms)":
                     f"{p50:.0f} / {p90:.0f} / {p99:.0f}", "throughput (items/s)":
                     round(r.n_items.iloc[0] / r.wall_s.iloc[0], 1),
                     "notes": f"concurrency {notes['concurrency']}, retries {notes['retries']}"})
    for m in TRAINED:
        r = runs[runs.method == m]
        for device in ["cuda", "cpu"]:
            b1, bulk = r[(r.device == device) & (r.batch_size == 1)].tail(1), r[(r.device == device) & (r.batch_size > 1)].tail(1)
            if b1.empty or bulk.empty:
                continue
            notes = json.loads(b1.notes.iloc[0] or "{}")
            if "p50_ms" in notes:
                lat = f"{notes['p50_ms']:.2f} / {notes['p90_ms']:.2f} / {notes['p99_ms']:.2f}"
            else:
                g = preds[(preds.method == m) & (preds.variant == "main")].latency_ms.dropna()
                lat = "{:.2f} / {:.2f} / {:.2f}".format(*np.percentile(g, [50, 90, 99]))
            rows.append({"method": METHODS[m], "device": device.upper().replace("CUDA", "GPU"),
                         "latency p50 / p90 / p99 (ms)": lat,
                         "throughput (items/s)": int(bulk.n_items.iloc[0] / bulk.wall_s.iloc[0]),
                         "notes": f"bulk batch {int(bulk.batch_size.iloc[0])}"})
    return pd.DataFrame(rows)


def openrouter_table(db) -> pd.DataFrame:
    try:
        g = pd.read_sql("SELECT * FROM openrouter_generation WHERE variant = 'main'", db)
    except pd.errors.DatabaseError:
        return pd.DataFrame()
    rows = []
    for m, d in g.groupby("method", sort=False):
        overhead = d.client_latency_ms - d.or_generation_time_ms
        rows.append({"method": METHODS[m], "n": len(d), "client latency p50 (ms)": round(d.client_latency_ms.median()),
                     "provider generation time p50 (ms)": round(d.or_generation_time_ms.median()),
                     "time to first token p50 (ms)": round(d.or_latency_ms.median()),
                     "client - generation time p50 / p90 (ms)":
                         f"{overhead.median():.0f} / {overhead.quantile(0.9):.0f}"})
    return pd.DataFrame(rows)


def list_price_cost(g: pd.DataFrame, prices: dict) -> pd.Series:
    return (g.input_tokens.fillna(0) * prices["input_per_mtok"] + g.output_tokens.fillna(0) * prices["output_per_mtok"]) / 1e6


def cost_table(preds: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    with open(CONFIG / "prices.toml", "rb") as f:
        prices = tomllib.load(f)
    rows = []
    for m in ZERO_SHOT:
        main = preds[(preds.method == m) & (preds.variant == "main") & preds.ean.isin(test.ean)]
        tune = preds[(preds.method == m) & preds.variant.str.startswith("tune-")]
        if main.empty and tune.empty:
            continue
        cost = (lambda g: list_price_cost(g, prices[m])) if m in prices else (lambda g: g.cost_usd.fillna(0))
        rows.append({
            "method": METHODS[m],
            "input tokens / item": round(main.input_tokens.mean(), 1) if len(main) else "–",
            "output tokens / item": round(main.output_tokens.mean(), 1) if len(main) else "–",
            "cost per 1,000 items (USD)": round(1000 * cost(main).mean(), 4) if len(main) else "–",
            "cost source": "tokens × list price" if m in prices else "usage.cost",
            "tuning spend (USD)": round(cost(tune).sum(), 4),
            "tuning variants": tune.variant.nunique(),
        })
    return pd.DataFrame(rows)


def tune_table(preds: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    labels = df.set_index("ean").category
    t = preds[preds.variant.str.startswith("tune-")].copy()
    t["correct"] = t.label == t.ean.map(labels)
    table = t.pivot_table(index="variant", columns="method", values="correct", aggfunc="sum")
    table = table[[m for m in ZERO_SHOT if m in table]].rename(columns=METHODS)
    return table.reset_index().astype({c: "Int64" for c in table.columns})


def resources_table(db) -> pd.DataFrame:
    r = pd.read_sql("SELECT method, variant, phase, device, batch_size, n_items, wall_s, peak_rss_mb, "
                    "peak_vram_alloc_mb, peak_vram_reserved_mb, peak_vram_process_mb FROM runs "
                    "WHERE method IN ('tfidf', 'bertic') AND variant IN ('main', 'brand')", db)
    r = r.drop_duplicates(["method", "variant", "phase", "device", "batch_size"], keep="last")
    r["method"] = r.method.map(METHODS)
    return r.round(1).astype({"batch_size": "Int64"})


# Output

def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data yet._\n"
    numeric = [pd.api.types.is_numeric_dtype(df[c]) for c in df.columns]
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---:" if n else "---" for n in numeric) + " |"
    body = ["| " + " | ".join("–" if pd.isna(v) else str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body]) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--publish", action="store_true", help="copy the report of this run directory to results/")
    args = ap.parse_args()

    out = RESULTS_DIR / "report"
    out.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(RESULTS_DB)
    df = load()
    preds = load_predictions(db)
    test, btest = df[df.split == "test"], df[df.split_brand == "test"]
    train_brands = set(df.brand.str.lower()[df.split == "train"])
    unseen = test[~test.brand.str.lower().isin(train_brands)]

    sections = ["# Results\n", f"Generated by `report.py` from `{RESULTS_DB.name}`. Test items: {len(test)} "
                f"(main split), {len(btest)} (brand split). Non-answers count as wrong.\n"]

    def add(title: str, table: pd.DataFrame, name: str | None = None, note: str = ""):
        sections.append(f"## {title}\n\n{note}{md_table(table)}")
        if name and not table.empty:
            table.to_csv(out / f"{name}.csv", index=False)

    main_sc = scenario(preds, test, {m: "main" for m in METHODS})
    if main_sc:
        q, pairs = quality(test, main_sc)
        add("1. Main test set", q, "main-quality")
        add("Pairwise comparison (main test set)", pairs, "main-pairwise",
            "Exact McNemar test on the items where one method is right and the other wrong; Holm "
            "correction over all pairs. 95% CI: paired bootstrap of the error-count difference.\n\n")
        add("Confidence thresholds (main test set)", confidence_table(test, main_sc), "main-confidence")
        add("Per class (main test set)", per_class(test, main_sc), "main-per-class")
        add("Agreement (main test set)", agreement(test, main_sc), "main-agreement",
            "Items where two methods agree with each other and not with the label are often label "
            "errors (section 8.3).\n\n")
        write_confusions(test, main_sc, out, "main")

    lc = learning_curve(preds, df)
    zero = {m: (g.label.to_numpy() == test.category.to_numpy()).mean() for m, g in main_sc.items() if m in ZERO_SHOT}
    if not lc.empty:
        summary = lc.groupby(["method", "size"], sort=False).agg(
            **{"train items": ("train items", "mean"), "seeds": ("seed", "count"),
               "accuracy mean": ("accuracy", "mean"), "accuracy min": ("accuracy", "min"),
               "accuracy max": ("accuracy", "max"), "macro-F1 mean": ("macro-F1", "mean"),
               "macro-F1 min": ("macro-F1", "min"), "macro-F1 max": ("macro-F1", "max")}).reset_index()
        summary["method"] = summary.method.map(METHODS)
        summary["train items"] = summary["train items"].round().astype(int)
        lc.to_csv(out / "learning-curve-runs.csv", index=False)
        for metric in ["accuracy", "macro-F1"]:
            zero_metric = zero if metric == "accuracy" else \
                {m: macro_f1(test.category.to_numpy(), g.label.to_numpy()) for m, g in main_sc.items() if m in ZERO_SHOT}
            learning_curve_chart(lc, zero_metric, metric, out / f"learning-curve-{metric.lower()}.svg")
        add("2. Learning curve", summary.round(4), "learning-curve",
            "![accuracy](learning-curve-accuracy.svg)\n\n![macro-F1](learning-curve-macro-f1.svg)\n\n")

    brand_sc = scenario(preds, btest, {m: "brand" if m in TRAINED else "main" for m in METHODS})
    if brand_sc:
        q, pairs = quality(btest, brand_sc)
        add("3. Unseen brands: brand split", q, "brand-quality",
            "Trained methods are trained on `split_brand = train`; no test brand occurs in training.\n\n")
        add("Pairwise comparison (brand split)", pairs, "brand-pairwise")
    unseen_sc = {m: g.loc[unseen.ean] for m, g in main_sc.items()}
    if unseen_sc:
        add(f"Unseen-brand slice of the main test set ({len(unseen)} items)", quality(unseen, unseen_sc)[0],
            "unseen-slice-quality")
    if "tfidf" in main_sc:
        tfidf_wrong = test.ean[main_sc["tfidf"].label.to_numpy() != test.category.to_numpy()]
        hard = test[test.ean.isin(set(tfidf_wrong) | set(unseen.ean))]
        hard_sc = {m: g.loc[hard.ean] for m, g in main_sc.items()}
        add(f"5. Hard set ({len(hard)} items)", quality(hard, hard_sc)[0], "hard-quality",
            "TF-IDF errors on the main test set plus the unseen-brand items. TF-IDF scores low here by "
            "construction.\n\n")

    add("Speed", speed_table(db, preds), "speed",
        "Batch size 1, warm. API latency is from first-attempt successes of the main run and includes the "
        "network (client in the EU). Throughput: API at the stated concurrency, local methods with a large "
        "batch.\n\n")
    add("OpenRouter share of the latency (Haiku, Qwen)", openrouter_table(db), "openrouter-latency",
        "From `GET /api/v1/generation` for a sample of main-run requests. Client latency minus the "
        "provider's generation time estimates what OpenRouter and the network add.\n\n")
    add("Cost", cost_table(preds, test), "cost",
        "Cost per 1,000 items: mean over the main test items. TF-IDF and BERTić: no dollar figure, see "
        "Resources.\n\n")
    add("Resources (TF-IDF, BERTić)", resources_table(db), "resources",
        f"Hardware of the machine that made this report (normally the one that ran the timings): "
        f"{hardware()}. peak_rss_mb is the process peak so far.\n\n")
    add("Prompt tuning (500 tune items, correct answers)", tune_table(preds, df), "tuning")

    (out / "report.md").write_text("\n".join(sections))
    print(f"wrote {out}")

    if args.publish:
        dest = PUBLISHED
        shutil.rmtree(dest / "report", ignore_errors=True)
        shutil.copytree(out, dest / "report")
        for f in RESULTS_DIR.glob("pricing-*.json"):
            shutil.copy2(f, dest)
        # mtime 0: the same predictions give the same file, so git sees no change
        pd.read_sql(f"SELECT {', '.join(PUBLISHED_COLUMNS)} FROM predictions ORDER BY method, variant, ean", db).to_csv(
            dest / "predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        size_mb = RESULTS_DB.stat().st_size / 2**20
        if size_mb < MAX_PUBLISHED_DB_MB:
            shutil.copy2(RESULTS_DB, dest / "results.db")
        print(f"published to {dest}" + ("" if size_mb < MAX_PUBLISHED_DB_MB else
                                         f" (results.db not copied: {size_mb:.0f} MB)"))


if __name__ == "__main__":
    main()
