"""BERTić fine-tuning (EXPERIMENT.md 6.2). Run with `uv run --group bert`.

Variants:
  main               train on split=train, best checkpoint by dev macro-F1, predict split=test;
                     the final model is saved to $RESULTS_DIR/model/main for bertic_run.py
  brand              the same on split_brand (train/dev/test)
  lc-<size>-s<seed>  train on a learning-curve subset with a fixed schedule (no dev selection,
                     last checkpoint), predict split=test

Each variant predicts its test set after training (GPU, large batch). A variant whose test
predictions are already stored is skipped unless --force.
"""

import argparse
import math
import shutil
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

from common import (CATEGORIES, DATA, RESULTS_DIR, GpuProcessMemory, connect, load, peak_rss_mb,
                    save_predictions, save_run, summary, text_input)

METHOD = "bertic"
BASE_MODEL = "classla/bcms-bertic"
REVISION = "5db9755d6152ec6403c0201223e4848bd1b98a48"  # main on Hugging Face, 2026-10-26
MODEL_DIR = RESULTS_DIR / "model"
MAX_LENGTH = 64
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 10  # 5 in the first run; dev macro-F1 was still rising at epoch 5 (EXPERIMENT.md 12)
SEED = 42
# Learning curve: fixed schedule, set before the runs.
LC_EPOCHS = 10
LC_MIN_STEPS = 200
PREDICT_BATCH_SIZE = 256
LABEL2ID = {c: i for i, c in enumerate(CATEGORIES)}


def texts(df: pd.DataFrame) -> list[str]:
    return [text_input(r) for r in df.itertuples()]


def encode(tokenizer, df: pd.DataFrame) -> Dataset:
    ds = Dataset.from_dict({"text": texts(df), "label": [LABEL2ID[c] for c in df.category]})
    # padding=False here; DataCollatorWithPadding pads each batch.
    return ds.map(lambda b: tokenizer(b["text"], max_length=MAX_LENGTH, truncation=True),
                  batched=True, remove_columns=["text"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", labels=range(len(CATEGORIES)), zero_division=0),
    }


@torch.no_grad()
def predict(model, tokenizer, texts: list[str], batch_size: int, device: str) -> np.ndarray:
    """softmax(logits) for all classes, in CATEGORIES order. The model must be on `device`."""
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], max_length=MAX_LENGTH, truncation=True, padding=True,
                        return_tensors="pt").to(device)
        out.append(torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def to_rows(test: pd.DataFrame, variant: str, probs: np.ndarray) -> list[dict]:
    rows = []
    for ean, p in zip(test.ean, probs):
        p = {c: float(x) for c, x in zip(CATEGORIES, p)}
        label = max(p, key=p.get)
        rows.append(dict(ean=ean, method=METHOD, variant=variant, status="ok", label=label, probs=p,
                         confidence=p[label], attempts=1, model=f"{BASE_MODEL}@{REVISION[:7]}",
                         provider="local"))
    return rows


def train_variant(db, variant: str, train: pd.DataFrame, dev: pd.DataFrame | None, test: pd.DataFrame,
                  seed: int, save: bool = False):
    set_seed(seed)  # before from_pretrained: the classification head is initialized there
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, revision=REVISION, num_labels=len(CATEGORIES),
        id2label=dict(enumerate(CATEGORIES)), label2id=LABEL2ID,
    )
    work_dir = MODEL_DIR / f"{variant}-checkpoints"
    if dev is not None:
        # Not load_best_model_at_end: transformers 5.4 saves the LayerNorm weights under the
        # original names (gamma/beta), and the Trainer reloads the best checkpoint without
        # renaming them, so they stay at the last epoch's values. from_pretrained renames them;
        # the best checkpoint is loaded with it below. Model-only checkpoints, all kept until the end.
        schedule = dict(num_train_epochs=EPOCHS, eval_strategy="epoch", save_strategy="epoch",
                        metric_for_best_model="macro_f1", greater_is_better=True, save_only_model=True)
    else:
        steps = max(LC_MIN_STEPS, LC_EPOCHS * math.ceil(len(train) / BATCH_SIZE))
        schedule = dict(max_steps=steps, eval_strategy="no", save_strategy="no")
    args = TrainingArguments(
        output_dir=str(work_dir), per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=PREDICT_BATCH_SIZE, gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE, warmup_steps=0.1, weight_decay=0.01, fp16=False, seed=seed,
        logging_steps=50, logging_first_step=True, disable_tqdm=True, report_to="none", **schedule,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=encode(tokenizer, train),
        eval_dataset=None if dev is None else encode(tokenizer, dev),
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    torch.cuda.reset_peak_memory_stats()
    with GpuProcessMemory() as gpu:
        t0 = time.perf_counter()
        trainer.train()
        wall_s = time.perf_counter() - t0
    log = trainer.state.log_history
    notes = {
        "steps": trainer.state.global_step, "train_items": len(train), "seed": seed,
        "first_loss": next((e["loss"] for e in log if "loss" in e), None),
        "dev_macro_f1": [round(e["eval_macro_f1"], 4) for e in log if "eval_macro_f1" in e],
        "best_checkpoint": trainer.state.best_model_checkpoint and
                           trainer.state.best_model_checkpoint.rsplit("/", 1)[-1],
    }
    save_run(db, method=METHOD, variant=variant, phase="train", device="cuda", batch_size=BATCH_SIZE,
             n_items=len(train), wall_s=wall_s, peak_rss_mb=peak_rss_mb(),
             peak_vram_alloc_mb=torch.cuda.max_memory_allocated() / 2**20,
             peak_vram_reserved_mb=torch.cuda.max_memory_reserved() / 2**20,
             peak_vram_process_mb=gpu.peak_mb, notes=notes)
    print(f"{variant}: trained in {wall_s:.0f} s, {notes}")

    if trainer.state.best_model_checkpoint:
        model = AutoModelForSequenceClassification.from_pretrained(trainer.state.best_model_checkpoint).to("cuda")
    if save:
        model.save_pretrained(str(MODEL_DIR / variant))
        tokenizer.save_pretrained(str(MODEL_DIR / variant))
    shutil.rmtree(work_dir, ignore_errors=True)

    rows = to_rows(test, variant, predict(model, tokenizer, texts(test), PREDICT_BATCH_SIZE, "cuda"))
    save_predictions(db, rows)
    summary(variant, test, [r["label"] for r in rows])
    del trainer, model
    torch.cuda.empty_cache()


def stored(db, variant: str) -> int:
    return db.execute("SELECT COUNT(*) FROM predictions WHERE method = ? AND variant = ?",
                      (METHOD, variant)).fetchone()[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="*", choices=["main", "brand", "lc"], help="default: all")
    ap.add_argument("--force", action="store_true", help="train variants that are already stored")
    args = ap.parse_args()
    args.parts = args.parts or ["main", "brand", "lc"]

    df = load()
    db = connect()
    test = df[df.split == "test"]
    jobs = []
    if "main" in args.parts:
        jobs.append(("main", df[df.split == "train"], df[df.split == "dev"], test, SEED, True))
    if "brand" in args.parts:
        jobs.append(("brand", df[df.split_brand == "train"], df[df.split_brand == "dev"],
                     df[df.split_brand == "test"], SEED, False))
    if "lc" in args.parts:
        subsets = pd.read_csv(DATA / "train-subsets.csv", dtype={"ean": str})
        for (size, seed), g in subsets.groupby(["size", "seed"]):
            jobs.append((f"lc-{size}-s{seed}", df[df.ean.isin(g.ean)], None, test, seed, False))

    for variant, train, dev, tst, seed, save in jobs:
        if not args.force and stored(db, variant) == len(tst):
            print(f"{variant}: already stored, skipped")
            continue
        train_variant(db, variant, train, dev, tst, seed, save)


if __name__ == "__main__":
    main()
