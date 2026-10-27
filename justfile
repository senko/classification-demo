# Commands for the full experiment and for the live demo. `just` lists them.
#
# Full experiment: results go to out/. Live demo: results go to out/demo, so the full run stays
# as it is. Use another demo directory with: RESULTS_DIR=... just <recipe>

export RESULTS_DIR := env("RESULTS_DIR", "out/demo")

# Each demo API run gets its own results variant, so a rerun sends all requests again.
variant := "demo-$(date +%Y%m%d-%H%M%S)"

default:
    @just --list --unsorted

# Install the dependencies, including torch for BERTić (several GB) and Jupyter for the notebook
setup:
    uv sync --group bert --group notebook

# Full experiment into out/: all runs, then the report (~1.5 h, ~$2.30 in API calls; resumable)
reproduce: setup
    RESULTS_DIR=out uv run --no-sync prepare.py
    RESULTS_DIR=out uv run --no-sync tfidf.py
    RESULTS_DIR=out uv run --no-sync --group bert bertic_train.py
    RESULTS_DIR=out uv run --no-sync --group bert bertic_run.py
    RESULTS_DIR=out uv run --no-sync run_jev.py --items eval
    RESULTS_DIR=out uv run --no-sync run_openai.py --items eval
    RESULTS_DIR=out uv run --no-sync run_llm.py haiku --items eval
    RESULTS_DIR=out uv run --no-sync run_llm.py qwen --items eval
    RESULTS_DIR=out uv run --no-sync openrouter_stats.py
    RESULTS_DIR=out uv run --no-sync report.py

# Your full results (out/report) next to the reference results (results/report)
compare dir="out/report":
    uv run --no-sync compare.py "{{ dir }}"

# Demo: train BERTić on the train split, pick the best epoch on dev, score the test split (~10 min on GPU)
train-bertic:
    uv run --no-sync --group bert bertic_train.py main --force

# Demo: speed of the trained BERTić model on the test split, GPU and CPU (~3 min)
test-bertic:
    uv run --no-sync --group bert bertic_run.py

# Demo: train TF-IDF + LR on the train split and score the test split (~20 s)
tf-idf:
    uv run --no-sync tfidf.py main

# Demo: Jev on the 500 tune items
jev: _dir
    uv run --no-sync run_jev.py --items tune --variant "{{ variant }}"

# Demo: OpenAI Decisions (GPT-6 Luna) on the 500 tune items
luna: _dir
    uv run --no-sync run_openai.py --items tune --variant "{{ variant }}"

# Demo: Claude Haiku via OpenRouter on the 500 tune items
haiku: _dir
    uv run --no-sync run_llm.py haiku --items tune --variant "{{ variant }}"

# Demo: Qwen3.8 27B via OpenRouter on the 500 tune items
qwen: _dir
    uv run --no-sync run_llm.py qwen --items tune --variant "{{ variant }}"

# Demo: the code notebook (demo.ipynb) in JupyterLab on port 8888, open on all interfaces
notebook:
    uv run --no-sync --group bert --group notebook jupyter lab --no-browser --ip=0.0.0.0 demo.ipynb

# Report of the full run (the reference results), in the terminal
results:
    glow -t results/report/report.md

# Delete the demo run directory (also the trained BERTić model in it) and the notebook's out/notebook
clean:
    #!/bin/sh
    dir=$(realpath -m "$RESULTS_DIR")
    if [ "$dir" = "$(realpath -m out)" ] || [ "$dir" = "$(realpath -m .)" ]; then
        echo "refusing to delete $RESULTS_DIR: not a demo directory" >&2
        exit 1
    fi
    rm -rf "$RESULTS_DIR" out/notebook

[private]
_dir:
    @mkdir -p "$RESULTS_DIR"
