"""Write data/split.csv and data/train-subsets.csv (EXPERIMENT.md section 4).

The files are committed. If they exist, this script recomputes them and checks that they are
identical, without writing. --force overwrites them.
"""

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from common import CATEGORIES, DATA, load_products

SEED = 42
SUBSET_SIZES = [50, 200, 1000, 5000]
SUBSET_SEEDS = range(5)
MIN_PER_CLASS = 2


def name_group(name: pd.Series) -> pd.Series:
    # lowercase, remove numbers and number punctuation, sorted set of words
    return (name.str.lower()
            .str.replace(r"[\d.,/%]+", " ", regex=True)
            .str.split().map(lambda w: " ".join(sorted(set(w)))))


def assign(df: pd.DataFrame, groups: pd.Series, seed: int = SEED) -> np.ndarray:
    folds = [te for _, te in StratifiedGroupKFold(10, shuffle=True, random_state=seed)
             .split(df, df.category, groups)]
    split = np.full(len(df), "train", dtype=object)
    split[np.concatenate(folds[:2])] = "test"
    split[folds[2]] = "dev"
    return split


def train_subsets(train: pd.DataFrame) -> pd.DataFrame:
    """Nested, stratified subsets: per seed and class, shuffle once and take the first
    max(2, round(size * class share)) items."""
    share = train.category.value_counts(normalize=True)
    rows = []
    for seed in SUBSET_SEEDS:
        rng = np.random.default_rng(seed)
        for cat in CATEGORIES:
            eans = train.ean[train.category == cat].to_numpy()
            eans = eans[rng.permutation(len(eans))]
            for size in SUBSET_SIZES:
                n = max(MIN_PER_CLASS, round(size * share[cat]))
                rows += [(e, seed, size) for e in eans[:n]]
    return pd.DataFrame(rows, columns=["ean", "seed", "size"])


def check(df: pd.DataFrame, subsets: pd.DataFrame):
    groups = name_group(df.name)
    text = df.name + " " + df.quantity + " " + df.unit
    train, test = df.split == "train", df.split == "test"
    assert not set(groups[train]) & set(groups[test]), "name group in train and test"
    assert not set(text[train]) & set(text[test]), "input string in train and test"
    btrain, btest = df.split_brand == "train", df.split_brand == "test"
    brand = df.brand.str.lower()
    assert not set(brand[btrain]) & set(brand[btest]), "brand in train and test (brand split)"
    assert df.tune.sum() == 500 and (df.split[df.tune == 1] == "dev").all()

    print("main split:", df.split.value_counts().to_dict())
    print("brand split:", df.split_brand.value_counts().to_dict())
    print("unseen-brand items in main test:", (test & ~brand.isin(set(brand[train]))).sum())
    print("evaluation items (test | brand test | tune):", (test | btest | (df.tune == 1)).sum())
    print(pd.crosstab(df.category, df.split).loc[CATEGORIES, ["train", "dev", "test"]])

    # Nested within a seed: each subset contains all smaller subsets of the same seed.
    for seed, g in subsets.groupby("seed"):
        sets = [set(g.ean[g["size"] == s]) for s in SUBSET_SIZES]
        assert all(a <= b for a, b in zip(sets, sets[1:])), f"subsets of seed {seed} not nested"
    assert subsets.ean.isin(df.ean[train]).all()
    sizes = subsets.groupby(["size", "seed"]).size().unstack()
    print("real subset sizes (rows: nominal size, columns: seed):")
    print(sizes)


def write_or_compare(df: pd.DataFrame, path: Path, force: bool) -> bool:
    """Write df to path, or compare it with the existing file. Returns False on a difference."""
    if force or not path.exists():
        df.to_csv(path, index=False)
        print(f"wrote {path.relative_to(DATA.parent)}")
        return True
    new = pd.read_csv(io.StringIO(df.to_csv(index=False)), dtype=str, keep_default_na=False)
    old = pd.read_csv(path, dtype=str, keep_default_na=False)
    if new.equals(old):
        print(f"{path.relative_to(DATA.parent)}: identical to the existing file")
        return True
    detail = f"{(new != old).any(axis=1).sum()} rows differ" if new.shape == old.shape else \
        f"shape {new.shape} vs {old.shape}"
    print(f"{path.relative_to(DATA.parent)}: DIFFERS from the existing file ({detail}). "
          "Check the scikit-learn version against uv.lock. Not overwritten (use --force).")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="overwrite the existing files")
    args = ap.parse_args()

    df = load_products()
    df["split"] = assign(df, name_group(df.name))
    df["split_brand"] = assign(df, df.brand.str.lower())
    dev = df.index[df.split == "dev"]
    tune, _ = train_test_split(dev, train_size=500, stratify=df.category[dev], random_state=SEED)
    df["tune"] = df.index.isin(tune).astype(int)
    subsets = train_subsets(df[df.split == "train"])

    check(df, subsets)
    same = [write_or_compare(df[["ean", "split", "split_brand", "tune"]], DATA / "split.csv", args.force),
            write_or_compare(subsets, DATA / "train-subsets.csv", args.force)]
    if not all(same):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
