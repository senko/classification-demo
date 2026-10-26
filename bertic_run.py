"""BERTić inference: speed and resources of the main model (EXPERIMENT.md 8.4, 8.6).
Run with `uv run --group bert`, after `bertic_train.py main`.

Phases on split=test, model loaded once:
  GPU, large batch    throughput; checks the predictions against the stored ones
  GPU, batch size 1   latency per item, stored in predictions.latency_ms (variant main)
  CPU, large batch    throughput, all torch threads
  CPU, batch size 1   latency per item (percentiles in runs.notes)
"""

import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bertic_train import CATEGORIES, METHOD, MODEL_DIR, predict, texts
from common import GpuProcessMemory, connect, load, peak_rss_mb, save_run

BULK_BATCH_SIZE = 512
WARMUP_BATCHES = 5


def percentiles(latency_ms: list[float]) -> dict:
    p50, p90, p99 = np.percentile(latency_ms, [50, 90, 99])
    return {"p50_ms": round(float(p50), 3), "p90_ms": round(float(p90), 3), "p99_ms": round(float(p99), 3)}


def batch1(model, tokenizer, items: list[str], device: str) -> list[float]:
    latency_ms = []
    for t in items:
        t0 = time.perf_counter()
        predict(model, tokenizer, [t], 1, device)  # includes tokenization and the copy to CPU
        latency_ms.append((time.perf_counter() - t0) * 1000)
    return latency_ms


def main():
    if not (MODEL_DIR / "main").is_dir():
        # from_pretrained would read a missing local path as a Hub repo id
        raise SystemExit(f"no trained model in {MODEL_DIR / 'main'}: run `bertic_train.py main` first")
    df = load()
    test = df[df.split == "test"]
    test_texts = texts(test)
    db = connect()

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "main")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / "main")
    load_s = time.perf_counter() - t0
    common_notes = {"load_s": round(load_s, 2), "torch_threads": torch.get_num_threads()}

    for device in ["cuda", "cpu"]:
        model.to(device)
        predict(model, tokenizer, test_texts[:WARMUP_BATCHES * 32], 32, device)  # warm-up, not timed
        vram = {}

        # Large batch
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with GpuProcessMemory() as gpu:
            t0 = time.perf_counter()
            probs = predict(model, tokenizer, test_texts, BULK_BATCH_SIZE, device)
            wall_s = time.perf_counter() - t0
        if device == "cuda":
            vram = dict(peak_vram_alloc_mb=torch.cuda.max_memory_allocated() / 2**20,
                        peak_vram_reserved_mb=torch.cuda.max_memory_reserved() / 2**20,
                        peak_vram_process_mb=gpu.peak_mb)
        stored = dict(db.execute("SELECT ean, label FROM predictions WHERE method = ? AND variant = 'main'",
                                 (METHOD,)).fetchall())
        changed = sum(stored.get(e) != CATEGORIES[i] for e, i in zip(test.ean, probs.argmax(1)))
        save_run(db, method=METHOD, variant="main", phase="infer", device=device,
                 batch_size=BULK_BATCH_SIZE, n_items=len(test), wall_s=wall_s, peak_rss_mb=peak_rss_mb(),
                 notes={**common_notes, "labels_differing_from_stored": changed}, **vram)
        print(f"{device} batch {BULK_BATCH_SIZE}: {wall_s:.2f} s, {len(test) / wall_s:.0f} items/s, "
              f"{changed} labels differ from the stored predictions")

        # Batch size 1
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with GpuProcessMemory() as gpu:
            t0 = time.perf_counter()
            latency_ms = batch1(model, tokenizer, test_texts, device)
            wall_s = time.perf_counter() - t0
        if device == "cuda":
            vram = dict(peak_vram_alloc_mb=torch.cuda.max_memory_allocated() / 2**20,
                        peak_vram_reserved_mb=torch.cuda.max_memory_reserved() / 2**20,
                        peak_vram_process_mb=gpu.peak_mb)
            db.executemany("UPDATE predictions SET latency_ms = ? WHERE method = ? AND variant = 'main' AND ean = ?",
                           [(lat, METHOD, e) for e, lat in zip(test.ean, latency_ms)])
            db.commit()
        save_run(db, method=METHOD, variant="main", phase="infer", device=device, batch_size=1,
                 n_items=len(test), wall_s=wall_s, peak_rss_mb=peak_rss_mb(),
                 notes={**common_notes, **percentiles(latency_ms)}, **vram)
        print(f"{device} batch 1: {wall_s:.1f} s, {percentiles(latency_ms)}")


if __name__ == "__main__":
    main()
