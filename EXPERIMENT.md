# Experiment design

This document is the plan and the rule set for the comparison. Every method uses the same data
split, the same input and the same metric code. Read it before you change a script or a prompt.

Status: finished 2026-10-26. All runs are done; the reference results are in `results/`
(`results/report/report.md`), the decisions and results log in section 12. Numbers
marked **(preliminary)** come from checks made on 2026-10-26 before any experiment code existed.
Statements marked **(not tested)** are general practice, not results.

## 1. Questions

1. How accurate is each method on retail product names, against the labels and after a label
   review?
2. How many labeled examples does a trained model need to match a zero-shot method?
3. How well does each method handle brands it has not seen?
4. What does each method cost per 1,000 items, how fast is it, and what does it need to run?

All methods are expected to reach more than 90% accuracy on the main test set (TF-IDF gets
96.7%, section 12). The differences will be small. For this reason the plan uses a large test
set (20%), paired statistics, a label review and scenarios where the methods separate (few
labels, unseen brands).

## 2. Methods

| # | method | type | needs labels | runs on |
| --- | --- | --- | --- | --- |
| 0 | TF-IDF + logistic regression | classical machine learning (control) | yes | local CPU |
| 1 | BERTić, fine-tuned | fine-tuned transformer | yes | local GPU |
| 2 | Jev, Choice question | zero-shot decision model | no | TypeSafe API (first party) |
| 3 | GPT-6 Luna, `choice` question | zero-shot decision model | no | OpenAI Decisions API (first party) |
| 4 | Claude Haiku 5.5 (`anthropic/claude-haiku-5.5`) | zero-shot LLM prompt | no | OpenRouter chat completions |
| 5 | Qwen3.8 27B (`qwen/qwen3.8-27b`) | zero-shot open-weights LLM prompt | no | OpenRouter chat completions |

- API keys come from the environment or from `.env`: `TYPESAFE_API_KEY`, `OPENAI_API_KEY`,
  `OPENROUTER_API_KEY`.
- Qwen3.8 27B represents the option "run an open-weights model yourself". It does not fit on
  the local GPU (12 GB), so it runs through OpenRouter. Its OpenRouter price is the cost figure
  for that option.
- Terminology for the presentation: TF-IDF is a word-weighting scheme. It does not classify by
  itself. The classifier is logistic regression, trained on labeled examples. Use "classical
  machine learning (TF-IDF + logistic regression)", "fine-tuned transformer (BERTić)" and
  "zero-shot (decision models, LLMs)". This keeps the main axis visible: does the method need
  labeled data, and how much.
- TF-IDF is the control. Without it the audience cannot tell whether a result is good.

## 3. Data

### Files

| file | columns | rows |
| --- | --- | ---: |
| `data/products.csv` | `barcode, brand, name, unit, quantity` | 29,684 |
| `data/labelled-products.csv` | `ean, category` | 29,684 |
| `data/categories.csv` | `id, name_hr, description` | 10 |

- Join on `barcode = ean`. The join is 1:1, with no duplicate barcodes.
- Read every file as strings:
  `pd.read_csv(path, dtype=str, keep_default_na=False)`. With default settings, barcodes lose
  leading zeros and the 51 empty brands become `NaN`.
- Barcode lengths: 13 digits 28,911; 8: 539; 11: 115; 12: 62; 4: 26.
- Never put the barcode into a model input. The GS1 prefix identifies the manufacturer.

### Categories

| category | items | share |
| --- | ---: | ---: |
| food | 14,204 | 47.9% |
| cosmetics | 4,443 | 15.0% |
| alcohol | 3,082 | 10.4% |
| drinks | 2,563 | 8.6% |
| cleaning | 2,053 | 6.9% |
| household | 1,269 | 4.3% |
| baby | 739 | 2.5% |
| pets | 649 | 2.2% |
| toiletries | 540 | 1.8% |
| health | 142 | 0.5% |

- `food` alone gives 47.9% accuracy. Report macro-F1 next to accuracy.
- `health` has 29 test items. One error moves its recall by 3.4 points. Report counts next to
  percentages.

### Model input

Every method gets the same information: the product name, the quantity and the unit. The
`brand` column is not part of the input.

| method | input |
| --- | --- |
| TF-IDF, BERTić | `f"{name} {quantity} {unit}"`, for example `Nektar Jabuka 2.00 L` |
| Jev | `state = {"product": name, "quantity": f"{quantity} {unit}"}` |
| Luna | `input = f"Product: {name}\nQuantity: {quantity} {unit}"` |
| Haiku, Qwen | user message `f"Product: {name}\nQuantity: {quantity} {unit}"` |

- Use `name`, `quantity` and `unit` as they are in the CSV. Do not normalize units (`L` and
  `l` both occur), decimal formats (`1.000`, `0.75`) or case.
- Do not remove the brand from `name`. 12,632 names (43%) contain the brand string, for
  example `Pfanner Ledeni Čaj`. That is how the products are listed.
- Why no brand field: the brand alone predicts the category too well for brands seen in
  training. On a random split, "most frequent category of this brand" gives 0.930 accuracy.
- Why quantity and unit: products with the same name in different pack sizes are different
  products. Quantity and unit also carry a weak signal (`kg` vs `L` vs `kom`). With quantity
  and unit, TF-IDF makes 197 errors on the main test set; with the name only, 250
  (preliminary, section 12).
- Quantity and unit are noisy. Units: `L` 9,938, `kg` 8,864, `g` 6,559, `kom` 3,777, `ml` 404,
  `l` 118, `m` 23, `par` 1. Some values are wrong, for example toothpaste with unit `L` and
  quantity 0.125. All methods see the same noise.

### Duplicates and short names

- The 29,684 items have 28,745 different input strings. 7 input strings (17 items) have more
  than one label. No method can get all of these right.
- 23,875 groups by normalized name (section 4). The largest group has 31 items.
- 3,390 brands. 267 brands have items in more than one category (for example Nivea:
  `cosmetics` and `baby`).
- 561 names have one word, for example `Piškote`, `Grašak`, `Pelinkovac`, `Aperol`.

### Labels and the specification

- About two thirds of the labels come from the retail chains' own categories. The rest come
  from comparing and cross-checking. No label was made by zero-shot classification or by an
  LLM, so no method has the advantage of agreeing with its own labels.
- **The category descriptions in `data/categories.csv` are the specification** for the
  zero-shot methods. The trained methods learn from the labels. Where the labels follow a
  convention that the descriptions do not state, the two disagree.
- Example of such a convention: kitchen paper towels (Perfex, Violeta Papirnati ručnik) have the
  label `toiletries`, but the `toiletries` description lists only toilet paper, tissues, wet
  wipes, pads, tampons and cotton. A fine-tuned model learns the convention. A zero-shot
  model follows the description. Say this in the presentation.
- Products with conflicting labels (same brand and name, different barcodes):

  | product | labels |
  | --- | --- |
  | Alpro Napitak od badema bez šećera | drinks, food |
  | Z bregov Ledena kava | drinks, food |
  | Vindija Vindi cocktail mandarina naranča | alcohol, drinks |
  | Nivea Sun Kids sprej SPF 50 | baby, cosmetics |
  | Becutan vlažne maramice | toiletries, baby, cosmetics |
  | Dove sapun | toiletries, cosmetics |
  | Vileda spužva Glitzi | household, cleaning |
  | Kühne Dressing American Caesar | food, pets (clear error) |

- Boundary pairs to watch in the confusion matrix: drinks/food (plant drinks, iced coffee),
  alcohol/drinks (cocktails), baby/cosmetics (children's cosmetics, baby wipes),
  toiletries/cosmetics (soap), household/cleaning (sponges, cloths).
- Accuracy against the labels is a lower bound on the true accuracy. A manual read of 71
  TF-IDF test errors (an earlier input format, with the brand) found about 20 label errors or
  spec/label conflicts, 13 ambiguous items and 38 real misses. Most real misses were names
  without a category word that need world knowledge: Glenfiddich 15Y, Ouzo 12, Absinthe Green,
  Staropramen Praha, Valdobbiadene (all predicted `food`).

## 4. Split

`prepare.py` writes `data/split.csv` once. The file is committed. Every script reads it. No
script makes its own split or shuffle. When the files exist, `prepare.py` recomputes them and
reports whether they are identical, without writing (`--force` overwrites). A different
scikit-learn version than in `uv.lock` can change the folds.

`data/split.csv` columns:

| column | values | meaning |
| --- | --- | --- |
| `ean` | barcode | item key |
| `split` | `train`, `dev`, `test` | main split |
| `split_brand` | `train`, `dev`, `test` | split by brand (scenario 3) |
| `tune` | `0`, `1` | 1 = one of the 500 `dev` items for zero-shot prompt tuning |

### Main split

Stratified by category, grouped by normalized name, 70/10/20:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

def load() -> pd.DataFrame:
    p = pd.read_csv("data/products.csv", dtype=str, keep_default_na=False)
    l = pd.read_csv("data/labelled-products.csv", dtype=str, keep_default_na=False)
    return p.merge(l, left_on="barcode", right_on="ean", validate="one_to_one")

def name_group(name: pd.Series) -> pd.Series:
    # lowercase, remove numbers and number punctuation, sorted set of words
    return (name.str.lower()
            .str.replace(r"[\d.,/%]+", " ", regex=True)
            .str.split().map(lambda w: " ".join(sorted(set(w)))))

def assign(df: pd.DataFrame, groups: pd.Series, seed: int = 42) -> np.ndarray:
    folds = [te for _, te in StratifiedGroupKFold(10, shuffle=True, random_state=seed)
             .split(df, df.category, groups)]
    split = np.full(len(df), "train", dtype=object)
    split[np.concatenate(folds[:2])] = "test"
    split[folds[2]] = "dev"
    return split

df = load()
df["split"] = assign(df, name_group(df.name))
df["split_brand"] = assign(df, df.brand.str.lower())
dev = df.index[df.split == "dev"]
tune, _ = train_test_split(dev, train_size=500, stratify=df.category[dev], random_state=42)
df["tune"] = df.index.isin(tune).astype(int)
```

Verified result (preliminary check with the same procedure): train 20,777, dev 2,969, test
5,938. No name group occurs in both train and test. No test input string occurs in train. 343
test items have a brand that does not occur in train.

| category | train | dev | test |
| --- | ---: | ---: | ---: |
| food | 9,942 | 1,421 | 2,841 |
| cosmetics | 3,111 | 444 | 888 |
| alcohol | 2,157 | 309 | 616 |
| drinks | 1,793 | 257 | 513 |
| cleaning | 1,437 | 205 | 411 |
| household | 888 | 127 | 254 |
| baby | 517 | 74 | 148 |
| pets | 455 | 64 | 130 |
| toiletries | 378 | 54 | 108 |
| health | 99 | 14 | 29 |

- **Why group by normalized name:** the group key removes numbers, so size variants of a
  product stay in the same part of the split. If the split were grouped by exact input
  instead, 1,091 of 5,938 test items (18%) would have an item with the same name and a
  different size in train. The model would only have to recall the name.
- **Why 20% test:** at 0.967 accuracy, the 95% interval is ±0.45 points with 5,938 items and
  ±0.64 with 2,969. `health` gets 29 test items instead of 14. Losing 3,000 training items has
  almost no effect on the trained models at this size.

### Brand split (`split_brand`)

The same procedure, grouped by lowercased brand. No test brand occurs in train. Train 20,789,
dev 2,961, test 5,934.

- 719 test items (12%) have a normalized name that occurs in train under a different brand,
  and 189 have the same input string. Most are generic names. Keep them: generic products of a
  new brand are part of the real case. Also report the slice without them.
- Grouping by brand and name together does not work: the connected components of shared
  brands and names put 55% of all items into one group.

### Use of each part

| part | used by |
| --- | --- |
| `train` | TF-IDF and BERTić training |
| `dev` | BERTić checkpoint selection |
| `dev` with `tune = 1` (500 items) | prompt and question tuning for the zero-shot methods |
| `test` | the numbers in the presentation. Nothing is tuned on it. |

### Learning-curve subsets

`prepare.py` also writes `data/train-subsets.csv` (`ean, seed, size`):

- Nominal sizes 50, 200, 1,000 and 5,000 items from `train`. The full size is all of `train`.
- 5 seeds (0-4).
- Stratified by category, with at least 2 items per class. Without the minimum, `health`
  (0.5%) has no item at size 50.
- Nested within a seed: per seed and class, shuffle the class once and take the first
  `max(2, round(size × class share))` items. Each subset then contains all smaller subsets of
  the same seed. The real size can differ from the nominal size by a few items. Store the
  nominal size and report the real one.

## 5. Fairness rules

1. Same input for every method (section 3). No method gets a field that the others do not get.
2. The zero-shot methods get the 10 descriptions from `categories.csv` without changes.
   Tuning changes only the instruction wording and the question form, never the meaning of a
   category.
3. The same tuning budget for every zero-shot method: at most 5 instruction variants, each
   evaluated on the 500 `tune` items. Log every variant and its result.
4. Freeze before the test run: for each method, commit the prompt, the question, the model id,
   the provider and all request settings to `config/`. Then run on test once per
   configuration.
5. Pin versions: dated model ids where they exist, pinned providers (section 6), and store the
   `model` and `provider` that each response reports.
6. A non-answer counts as wrong: refusal, unparsable output, or an error after all retries.
   Report non-answers in a separate column.
7. If a category description has a gap (for example kitchen towels), fixing it changes the
   specification. Do it only before the test run, based on `dev` only, and record it in
   section 13.
8. Record how many labels each method used: training labels, dev labels for checkpoint
   selection, and the 500 tune labels. The fair claim for a zero-shot method is "0 training
   labels, 500 labels to check the prompt".

## 6. Method setup

### 6.1 TF-IDF + logistic regression

```python
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
clf = LogisticRegression(C=10, max_iter=3000)
clf.fit(vec.fit_transform(train_texts), train_labels)
probs = clf.predict_proba(vec.transform(texts))
```

- Character n-grams of length 2-5 inside word boundaries, lowercased (sklearn default), only
  n-grams that occur in at least 2 training items.
- Weight per n-gram: tf = 1 + ln(count), idf = ln((1 + n) / (1 + df)) + 1, then L2
  normalization per row. Vocabulary and idf come from the training items only.
- Multinomial logistic regression (lbfgs, L2, C=10), no class weights. C is fixed, not tuned.
  The same settings are used at every learning-curve size.
- Store the full probability vector for every item, like every other method.

### 6.2 BERTić

**Model:** <https://huggingface.co/classla/bcms-bertic> (Apache-2.0). Paper: N. Ljubešić,
D. Lauc, "BERTić – The Transformer Language Model for Bosnian, Croatian, Montenegrin and
Serbian", BSNLP 2021, <https://aclanthology.org/2021.bsnlp-1.5>.

- ELECTRA-base discriminator: 12 layers, hidden size 768, 110M parameters, cased WordPiece
  vocabulary of 32,000. The tokenizer keeps diacritics and produces 0 `[UNK]` tokens on this
  dataset.
- Use the base model, not the task-specific fine-tunes (`-ner`, `-frenk-hate`).
- The first `from_pretrained("classla/bcms-bertic")` downloads ~423 MB. transformers also
  fetches `model.safetensors` (423 MB) from the Hugging Face auto-conversion branch, so the
  cache entry is ~845 MB. There is no `tokenizer.json`; `AutoTokenizer` builds the fast
  tokenizer from `vocab.txt`.
- `tokenizer_config.json` uses the old key `max_len`. Always pass `max_length=` to the
  tokenizer.
- On load, a warning says that `classifier.*` weights are newly initialized and
  `discriminator_predictions.*` is unused. This is expected.

**Environment that worked** for BERTić fine-tuning on this machine (Python 3.13):
torch 2.11.0 (default PyPI wheel, CUDA 13), transformers 5.4.0, datasets 4.8.4,
accelerate 1.13.0, scikit-learn 1.8.0.

**Training settings** (Hugging Face `Trainer`, `AutoModelForSequenceClassification`):

| setting | value | note |
| --- | --- | --- |
| labels | 10, in `categories.csv` order; `id2label`/`label2id` saved in `config.json` | |
| `MAX_LENGTH` | 64 | inputs with brand, name, quantity and unit were at most 37 tokens |
| batch size | 32, gradient accumulation 1 | |
| learning rate | 2e-5, warmup ratio 0.1, weight decay 0.01 | standard for ELECTRA |
| epochs | 5, evaluate and save every epoch | (not tested) check that dev macro-F1 stops rising |
| checkpoint | `load_best_model_at_end=True`, `metric_for_best_model="macro_f1"`, `save_total_limit=2` | |
| fp16 | off | (not tested) ELECTRA in fp16 |
| padding | `padding=False` in `map`, dynamic padding in the collator | |
| seed | 42 | |
| class weights | none at first | (not tested) add weighted loss only if dev recall of `health` or `toiletries` is poor |

- `compute_metrics` must use `f1_score(..., average="macro")`. The default `average="binary"`
  raises `ValueError` on 10 classes.
- Sanity check: the first logged training loss must be near ln 10 = 2.30.
- The input string must be the same at training and inference time (section 3).
- Save the tokenizer next to the model (`trainer.save_model(path)`,
  `tokenizer.save_pretrained(path)`) in `$RESULTS_DIR/model/` (section 9, not committed). A
  checkpoint with optimizer state is ~1.3 GB, the final model ~423 MB.
- Inference: load the model once, run batches, store `softmax(logits)` for all classes.
- Checkpoint selection uses `dev`, so dev numbers are optimistic. Report test only.
- Implementation (`bertic_train.py`, `bertic_run.py`, run with `uv run --group bert`): base
  model pinned to revision `5db9755`. transformers 5.4 deprecates `warmup_ratio`;
  `warmup_steps=0.1` sets the same 10% warmup. Only the `main` model is saved; the other
  variants predict their test set right after training. `bertic_run.py` measures the `main`
  model on GPU and CPU, batch size 1 and 512, after a warm-up.
- Learning curve: a fixed training schedule per size, set before the runs, and the last
  checkpoint. No dev-based selection for subsets: the full dev set (2,969 labels) is larger
  than most subsets. Schedule: max(10 epochs, 200 optimizer steps), so 200 steps for the 53-
  and 202-item subsets, 320 for 1,001 and 1,570 for 5,000. The training seed is the subset
  seed, so the spread over the 5 seeds covers both the sample and the initialization. BERTić
  is expected to be unstable with 50-200 examples (not tested).

### 6.3 Jev (TypeSafe API)

Python SDK `typesafe-sdk`, API key in `TYPESAFE_API_KEY`:

```python
from typesafe_sdk import AsyncTypeSafeClient, Choice

QUESTIONS = {
    "category": Choice(
        instructions="Which store category does this product belong to?",
        criteria={c.id: c.description for c in categories},  # from categories.csv
    ),
}

async with AsyncTypeSafeClient(model=JEV_MODEL, timeout=60.0) as client:
    resp = await client.system_one(
        state={"product": name, "quantity": f"{quantity} {unit}"}, questions=QUESTIONS)
    ans = resp.choices["category"]
    ans.choice          # top label
    ans.probabilities   # {label: p}, sums to 1
    ans.confidence      # derived from the spread of the probabilities
    resp.usage.input_tokens
    resp.model
```

- Checked with SDK 0.7.5 (2026-10-26): the usage above is unchanged. The SDK uses `httpx2`
  internally. `resp.raw_http_response.text` is the response body. `client.models.list()` lists
  only the aliases `jev-latest` and `jev-preview`.
- Model: pinned to **`jev-1.14.0`** in `config/jev.toml`. `jev-latest` resolved to it in the
  smoke test on 2026-10-26, and the API accepts the version name. Store `resp.model` with every
  result.
- The SDK default timeout is 10 s; use 60 s. The SDK retries by itself (default
  `RetryPolicy(max_retries=2)`, exponential backoff from 0.5 s to 5 s). `run_jev.py` passes
  `RetryPolicy(max_retries=0)` and uses the shared retry rules (6.6), so that attempt counts and
  latencies are comparable with the other methods.
- Price: $0.042 per 1M input tokens, output free. The cost is mostly a fixed overhead per
  request (about 500 input tokens for one short question) plus the 10 category descriptions.
  Measure the tokens on 50 items first.
- Documented rate limit: 1,200 requests per minute. In earlier use, concurrency 16 reached
  55 requests/s without errors. Report the achieved rate and the documented limit.
- Tuning options (inside the budget of section 5): the instruction text, and each criterion
  as a plain string or as an object (for example `{"covers": ...}`) with the same content. In
  earlier use, extra criteria text on a yes/no question made Jev slightly worse; keep the
  criteria short.
- On short inputs, Jev gave more answers in the uncertain range than Luna in earlier use.

### 6.4 Luna (OpenAI Decisions API)

Luna is called only through the Decisions API, which returns a probability for each option.
Do not call `gpt-6-luna` through the chat completions or Responses API: that would make it one
more LLM prompt, and the comparison is about the decision API.

`POST https://api.openai.com/v1/decisions`, header `Authorization: Bearer $OPENAI_API_KEY`:

```json
{
  "model": "gpt-6-luna",
  "input": "Product: Nektar Jabuka\nQuantity: 2.00 L",
  "questions": [{
    "name": "category",
    "type": "choice",
    "instructions": "Which store category does this product belong to?",
    "choices": [
      {"value": "food", "description": "Fruit and vegetables, meat, fish, dairy, ..."},
      {"value": "drinks", "description": "Water, juices, soft drinks, ..."}
    ]
  }]
}
```

`choices` has one entry per category, with the description from `categories.csv`. Response:
`answers` is a list. An answer with `"type": "choice"` has `choice`, `confidence` and
`probabilities` (a list of `{"value", "probability"}`); an answer with `"type": "refusal"` is a
non-answer. Store `usage.input_tokens` and the `model` value of the response.

- Use `choice` with a description for every option. This was the most accurate question form
  in earlier use; a bare yes/no predicate was too permissive.
- Price: $0.10 per 1M input tokens, no output charge. The OpenAI pricing page also lists a
  $0.05 Batch/Flex rate for gpt-6-luna, but did not say whether it applies to
  `/v1/decisions`.
- Behavior seen in earlier use, to check again here:
  - Probabilities are rounded to 2 decimals and are mostly exactly 0 or 1. The confidence
    ranking is coarse.
  - Repeated requests return the same probabilities.
  - Questions in one request are independent of each other.
  - Refusals are rare, but they occurred on short, vague inputs. Expect some on product names.
  - The first requests on a new connection took 1.9-2.9 s. Reuse connections
    (`httpx.Limits(max_connections=16, max_keepalive_connections=16)`).
- The openai Python SDK supports `/v1/decisions` from version 3.26.0. The plan uses `httpx`
  with the JSON above, so the retry and timing code is the same as for OpenRouter.

### 6.5 LLMs (Haiku, Qwen) through OpenRouter chat completions

`POST https://openrouter.ai/api/v1/chat/completions`, header `Authorization: Bearer
$OPENROUTER_API_KEY`. Both models get the same system prompt, the same user message format and
the same output schema.

```json
{
  "model": "qwen/qwen3.8-27b",
  "provider": {"order": ["near-ai"], "quantizations": ["fp8"],
               "allow_fallbacks": false, "require_parameters": true},
  "messages": [
    {"role": "system", "content": "<instructions + the 10 categories with descriptions>"},
    {"role": "user", "content": "Product: Nektar Jabuka\nQuantity: 2.00 L"}
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "classification",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "category": {"type": "string", "enum": ["food", "drinks", "alcohol", "cosmetics", "toiletries",
                                                    "cleaning", "household", "baby", "health", "pets"]},
          "confidence": {"type": "number", "description": "Probability from 0 to 1 that the category is correct"}
        },
        "required": ["category", "confidence"],
        "additionalProperties": false
      }
    }
  },
  "reasoning": {"effort": "none"},
  "max_tokens": 200
}
```

- **Confidence** is the confidence that the model states in its output. It is probably poorly
  calibrated. The coverage and calibration tables (section 8) measure it in the same way as the
  probabilities of the other methods.
- **Reasoning off** for the main run. Check on the tune set that
  `usage.completion_tokens_details.reasoning_tokens` is 0. Reasoning tokens are billed as
  output tokens.
- **Haiku 5.5:** `"provider": {"order": ["anthropic"], "allow_fallbacks": false,
  "require_parameters": true}`. Price $0.10 / $0.50 per 1M input / output tokens. The model
  thinks by default; turning thinking off is allowed at effort `high` or lower. Ablation on the
  tune set: adaptive thinking at effort `low`. If it is clearly better, run both
  configurations on test and report the cost difference. Haiku 5.5 rejects `temperature`,
  `top_p` and `top_k`, so greedy decoding is not available. Its safety classifiers can refuse a
  request; count refusals as non-answers. Record `cached_tokens`; do not pad the prompt to
  reach a minimum length for caching.
- **Qwen3.8 27B:** OpenRouter lists 20 providers for this model, with quantization from fp4 to
  fp16 and input prices from $0.035 to $0.99 per 1M tokens (2026-10-26). Use 8-bit (fp8)
  weights: that is the usual way to serve a model of this size, and the setup that the cost
  figure stands for. Pinned endpoint: **Near AI fp8** (`near-ai/fp8`), $0.04 / $1.35 per 1M
  input / output tokens, 99.8% uptime over the last day, structured outputs, reasoning control
  and `logprobs` supported. Alternatives with the same features: Chutes fp8 ($0.039 / $1.25),
  Reka fp8 ($0.04 / $2.20), Parasail fp8 ($0.24 / $2.20). If the 50-item smoke test shows
  errors or high latency, switch to one of these before the tune runs, not later. Report the
  fp8 price range next to the cost.
- **Qwen reasoning:** the model reasons by default, and its default effort is `xhigh`. The
  request must set `"reasoning": {"effort": "none"}`. If reasoning stays on by mistake, output
  tokens, cost and latency grow by one to two orders of magnitude (estimate).
- **Qwen sampling:** the OpenRouter defaults are `temperature` 1, `top_p` 0.95, `top_k` 20. The
  main run keeps them, like Haiku, which takes no sampling parameters. `temperature` 0 can be
  one of the tune-set variants (section 5, rule 3).
- One product per request in the main run. Ablation (optional): several products per request
  on the tune set. Give each item an id, check that every id comes back once, and compare the
  answers with the one-per-request run.
- OpenRouter lists no `logprobs` support for Haiku 5.5. The main run uses stated confidence for
  both LLMs. Optional ablation for Qwen (the Near AI endpoint supports `logprobs`): confidence
  from the token probabilities of the category value, compared with the stated confidence.
- Model ids are pinned to the dated versions `anthropic/claude-haiku-5.5-20261007` and
  `qwen/qwen3.8-27b-20260814` (`config/llm.toml`). OpenRouter accepts them; the response
  `model` field still reports the undated name, so `runs.notes` stores the requested id.
- Store from the response: the parsed JSON, `id`, `model`, `provider`, and from `usage`:
  `prompt_tokens`, `prompt_tokens_details.cached_tokens`, `completion_tokens`,
  `completion_tokens_details.reasoning_tokens`, `cost` (USD, always present).

### 6.6 Request handling (all API methods)

- `httpx` async client (OpenAI, OpenRouter) or the TypeSafe SDK client (Jev), connection
  reuse, `asyncio.Semaphore(16)`.
- Retry on 408, 409, 429, 500, 502, 503, 504 and transport errors. Honor `retry-after`,
  otherwise wait `min(2**attempt, 30)` s with jitter. At most 6 attempts. Store the attempt
  count.
- Timeout 60 s per request.
- Warm-up: before the timed run, the runner sends one request per connection (16) and does not
  store the results. In the Luna smoke test without warm-up, the first 16 requests took
  1.7-2.0 s and the next ones 110-400 ms.
- Store the response body (for Jev, the raw HTTP response body), never the response headers.
- The shared code is in `api.py`.

## 7. Scenarios

1. **Main test.** All methods on the 5,938 `test` items.
2. **Learning curve (main chart).** TF-IDF and BERTić trained on each subset of
   `data/train-subsets.csv` and on all of `train`, evaluated on `test`. The zero-shot methods
   are horizontal lines (0 training labels). Where a trained curve crosses a zero-shot line is
   the number of labels that the zero-shot method saves. Show the spread over the 5 seeds.
3. **Unseen brands.** TF-IDF and BERTić trained on `split_brand = train` (BERTić selects its
   checkpoint on `split_brand = dev`), evaluated on `split_brand = test`. Also report the
   unseen-brand slice of the main test set (343 items).
4. **Few labels and unseen brands** (a new catalogue with little labeled data): learning-curve
   subsets drawn from `split_brand = train`, evaluated on `split_brand = test` (optional).
5. **Hard set:** the TF-IDF errors on the main test set plus the unseen-brand items. It shows
   where world knowledge matters. TF-IDF scores low on this set by construction; say so.
6. **Taxonomy change** (optional): split one category (for example `household` into garden,
   kitchenware and other). The zero-shot methods need a new description. The trained methods
   need new labels and retraining.

The zero-shot methods do not train, so they run once on the union of all evaluation items
(main `test`, `split_brand = test` and the tune items: 11,087 items). Each scenario reads the
stored predictions.

## 8. Measurement

### 8.1 Quality

On `test`, per method:

| metric | reason |
| --- | --- |
| error rate (1 − accuracy) and error count, with 95% interval | at 97%, "197 vs 150 errors" is clearer than "96.7% vs 97.5%" |
| macro-F1 | `food` is 48% of items |
| coverage at ≤1% error | the share of items a method can accept automatically if 1% errors are acceptable; comparable across methods with differently scaled confidence |
| per-class precision, recall, F1 with counts | rare classes |
| confusion matrix (10 × 10) | boundary pairs |
| top-2 accuracy | for methods with a full distribution (not the LLMs) |
| non-answers | refusals, parse failures, errors |

- **Intervals:** paired bootstrap over test items (10,000 resamples), for each metric and for
  each difference between two methods.
- **Pairwise tests:** exact McNemar test on the items where two methods disagree in
  correctness, for all 15 pairs, Holm-corrected. Where the interval of a difference includes 0,
  write "not distinguishable", not a ranking.
- **Slices:** unseen brands, each boundary pair, single-word names.

### 8.2 Confidence

Confidence = the maximum class probability (TF-IDF, BERTić, Jev, Luna) or the stated confidence
(Haiku, Qwen).

- For thresholds 0.5, 0.7, 0.9 and 0.95: coverage (share with confidence ≥ t), error rate among
  the covered items, share routed to review.
- Risk-coverage curve: sort by confidence, descending, and plot the error rate against coverage.
  Coverage at ≤1% error is read from this curve. Items with the same confidence are accepted or
  rejected together (Luna's rounded probabilities give many ties).
- Calibration: ECE with 10 bins on the confidence.
- These curves are measured on test, so the "coverage at ≤1% error" is optimistic for every
  method in the same way. State it.

### 8.3 Agreement and label review

- Agreement table: for each pair of methods, % agreement; for the agreed items, how many
  disagree with the label; for the disagreed items, how often the label sides with each
  method.
- Items where two strong methods agree with each other and not with the label are usually
  label errors.
- **Blind label review:** take every test item where at least one method disagrees with the
  label. Show the reviewer the input, the label and the set of predicted categories, but not
  which method predicted what. The reviewer decides against the descriptions: label correct,
  label wrong (with the correct category), ambiguous, or spec/label conflict. Store the result
  in `data/test-review.csv`. If there are more than ~500 items, review a random sample and
  report an estimate with an interval.
- Report raw and reviewed numbers side by side. Never change `data/labelled-products.csv`.
  Never use the review for prompt tuning.
- A review of only the items where two methods agree against the label would favor the methods
  that agree with the majority. For this reason the review covers all disagreements.

### 8.4 Speed

Two numbers per method. Do not mix them.

| | API methods | TF-IDF, BERTić |
| --- | --- | --- |
| latency p50 / p90 / p99, one item per call, warm | time per request in the main test run at concurrency 16, from the first-attempt successes only | batch size 1, items one after another, model already loaded; CPU and GPU (BERTić) |
| throughput (items/s), whole test set | wall-clock time at concurrency 16 | largest batch: CPU with all threads, GPU (BERTić) |

- Report separately and leave out of the latency: training time, model load time, first
  requests on new connections.
- API latency includes the network. Record the client location (EU) with the results.
- Haiku and Qwen also pass through OpenRouter; Jev and Luna do not. To estimate the OpenRouter
  share, fetch `GET https://openrouter.ai/api/v1/generation?id=<id>` for a sample of 200
  requests per model and compare its `latency` and `generation_time` (ms) with the latency
  measured by the client. Report the client-side latency as the main number and the estimate
  next to it.
- API throughput depends on the client concurrency and on the rate limits. It is not a
  property of the model. Report the concurrency and the number of 429 responses.
- Run the API measurements from one machine in one session. Latency changes with the load of
  the provider.

### 8.5 Cost

- Store per request: the token counts that the API reports (input, cached input, output,
  reasoning), and `usage.cost` where the API reports it (OpenRouter). Store no computed dollar
  amounts; `report.py` computes them.
- Jev and OpenAI report tokens, not cost. Their list prices go into `config/prices.toml`, with
  the source URL and the date checked. Check the prices again before the test run.
- At each OpenRouter run, save the endpoint price list of each model
  (`GET https://openrouter.ai/api/v1/models/{model}/endpoints`) to
  `$RESULTS_DIR/pricing-<date>-<model>.json`.
- **Cost per 1,000 items** = 1,000 × mean cost per request over the main test run: tokens ×
  list price for Jev and Luna, `usage.cost` for Haiku and Qwen. As a check, also compute
  tokens × list price for Haiku and Qwen and compare with `usage.cost`.
- **Setup cost:** labels used (section 5, rule 8), GPU training time, and the API spend for
  prompt tuning (sum of `usage.cost` over the tune runs).
- **Offline price** (optional): OpenRouter lists a batch variant of Haiku 5.5 at half price
  (`anthropic/claude-haiku-5.5:batch`); it lists none for Qwen3.8 27B (2026-10-26). OpenAI
  lists $0.05 Batch/Flex for gpt-6-luna, not confirmed for `/v1/decisions`. Compute batch
  prices from the same token counts and label them as computed. Do not run them.
- **TF-IDF and BERTić:** no dollar figure. Report the hardware and the resources (8.6).
- Expected size (estimate; measure on 50 items before the full run): Jev, Luna and Haiku about
  $0.04-0.10 per 1,000 items, Qwen (Near AI fp8) about $0.05. All API runs of the experiment
  together should cost less than $10.

### 8.6 Resources (TF-IDF, BERTić)

Hardware: Intel Core i5-13500 (14 cores, 20 threads), 125 GiB RAM, NVIDIA GeForce RTX 3060
12 GB (the desktop uses ~3 GB of it), driver 610.43.02, CUDA 13.3.

Per phase (training; inference with batch size 1; inference with a large batch), on CPU and on
GPU where it applies:

- wall-clock time and items/s;
- peak RAM of the process (`resource.getrusage(...).ru_maxrss`);
- peak VRAM: `torch.cuda.max_memory_allocated()` and `torch.cuda.max_memory_reserved()`, and the
  process memory from `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`, sampled
  every 0.5 s (this includes the CUDA context, which PyTorch does not count).

## 9. Results storage and runners

### Run directory and published results

Every script writes into a run directory: `out/` by default, or the directory in the
`RESULTS_DIR` environment variable (for example, to keep a reproduction next to an earlier
run). Run directories are not committed.

```
out/
  results.db                   all predictions and run measurements (SQLite)
  pricing-<date>-<model>.json  OpenRouter price snapshots (8.5)
  model/                       BERTić checkpoints and final model
  report/                      tables (Markdown, CSV) and charts (SVG) from report.py
```

The reference results of this experiment are committed in `results/`, with the same layout
without `model/`. Only `report.py --publish` writes there. It copies `report/` and the price
snapshots, and also `results.db` if the file is smaller than 50 MB (GitHub warns above 50 MB
and refuses files above 100 MB). On 2026-10-26, with TF-IDF and the smoke tests only, the
database was 65 MB, mostly TF-IDF probability vectors for the learning curve. The final
database is 187 MB, so it is not committed. Instead, `--publish` writes every prediction
without the response body and the per-category probabilities to `results/predictions.csv.gz`
(5 MB).

A reproduction therefore never changes a committed file: `prepare.py` only checks the
committed split (section 4), the runners and `report.py` write into the run directory, and the
result is compared with `results/report/`.

The runners are resumable against the database of the run directory. A fresh clone has an
empty run directory, so every method runs in full.

### Database

Table `predictions`, primary key `(ean, method, variant)`:

| column | content |
| --- | --- |
| `ean`, `method`, `variant` | item, method, configuration (for example `main`, `lc-1000-s3`, `brand`, `tune-v2`, `smoke`) |
| `status` | `ok`, `refusal`, `unparsable` (non-answers from a successful request) or `error` (request failed after all retries) |
| `label` | predicted category, or empty for a non-answer |
| `probs` | JSON with all 10 categories (empty for the LLMs) |
| `confidence` | max probability or stated confidence |
| `input_tokens`, `cached_tokens`, `output_tokens`, `reasoning_tokens`, `cost_usd` | from `usage`; `cost_usd` only where the API reports it |
| `latency_ms`, `attempts` | timing of the successful attempt, number of attempts |
| `model`, `provider` | as reported in the response |
| `error` | error text, empty on success |
| `raw` | response body (JSON) |
| `created_at` | UTC timestamp |

Table `runs`: `method, variant, phase, device, batch_size, n_items, wall_s, peak_rss_mb,
peak_vram_alloc_mb, peak_vram_reserved_mb, peak_vram_process_mb, notes, created_at`.

### Runner requirements

- Resumable: skip rows that are stored without an error (`status != 'error'`). Refusals and
  unparsable answers are final; they are not run again.
- `--retry-errors` deletes the error rows of the selection and runs them again.
- `--items tune|eval` is required. `eval` (the 11,087 evaluation items) runs only after the
  config has a frozen `main` prompt.
- `--limit N` for smoke tests.
- Commit to the database every 100 results.
- `variant` keeps ablations separate from the main run.

## 10. Repository layout and publication

```
data/              source CSVs, split.csv, train-subsets.csv, test-review.csv
config/            frozen prompts, questions and request settings per method; prices.toml
common.py          data loading, input formatting, results database
api.py             shared retry, timing and resumable run loop for the API runners
prepare.py         writes data/split.csv and data/train-subsets.csv
tfidf.py           TF-IDF + logistic regression: training, prediction, timing
bertic_train.py    fine-tuning (writes $RESULTS_DIR/model/)
bertic_run.py      inference, timing, VRAM
run_jev.py         Jev through the TypeSafe API
run_openai.py      Luna through the OpenAI Decisions API
run_llm.py         Haiku and Qwen through OpenRouter chat completions
openrouter_stats.py  OpenRouter generation timing for a sample of Haiku and Qwen requests
report.py          metrics, tables and charts from $RESULTS_DIR/results.db; --publish copies to results/
compare.py         a run's report next to results/report/
review.py          label review: item export and verdict collection
justfile           shortcuts: setup, reproduce, compare, live demo
out/               default run directory (gitignored, section 9)
results/           published reference results: report/, predictions.csv.gz, price snapshots,
                   results.db if < 50 MB
```

The repository will be public. Rules:

- Commit: code, `data/`, `config/`, `results/`, `uv.lock`.
- Not committed (`.gitignore`): `.env`, `out/` and any other run directory, `model/`, `.venv/`,
  Hugging Face cache directories, raw HTTP logs.
- The API keys are read from the environment or from `.env`. They never go into a file that
  is committed.
- Store response bodies only, not headers. Before `results.db` is published, check one stored
  response of each method for account identifiers. (Checked on the smoke test: none.)
- In reports, name hardware by model only. No host names, user names or home paths.
- `report.py` runs from a `results.db` without an API key.

## 11. Environment

- Python 3.13, managed with `uv` (`pyproject.toml` and `uv.lock`, both committed).
- Base dependencies: pandas, scikit-learn, httpx, typesafe-sdk, matplotlib. BERTić dependencies (torch,
  transformers, datasets, accelerate) go into a separate dependency group (`uv sync --group
  bert`), so that the API and report scripts do not need torch.
- The development machine has `exclude-newer = "7 days"` in the user uv configuration. uv does
  not resolve a package version released in the last 7 days. If a resolution fails, check the
  release date of the version you ask for before you change anything else. The committed
  `uv.lock` gives other users the same versions.
- Jev uses `typesafe-sdk`. OpenAI and OpenRouter calls use `httpx` directly, with no provider
  SDK. On 2026-10-26, typesafe-sdk 0.7.5 (released 2026-10-14) is outside the 7-day window.

## 12. Preliminary numbers

TF-IDF + logistic regression (section 6.1), measured on 2026-10-26 with the split procedure of
section 4 (single run, one seed, no tuning):

| setup | accuracy | macro-F1 | errors |
| --- | ---: | ---: | ---: |
| main split, name + quantity + unit | 0.967 | 0.916 | 197 / 5,938 |
| main split, name only | 0.958 | 0.906 | 250 / 5,938 |
| main split, unseen-brand slice (name + quantity + unit) | 0.927 | – | 25 / 343 |
| brand split, name + quantity + unit | 0.927 | 0.815 | 431 / 5,934 |

Confidence on the main split, name + quantity + unit:

| max p ≥ | coverage | error among covered |
| ---: | ---: | ---: |
| 0.5 | 98.1% | 2.58% |
| 0.7 | 94.7% | 1.56% |
| 0.9 | 86.4% | 0.80% |
| 0.95 | 77.9% | 0.54% |

- Weakest classes (recall): `toiletries` 0.750 (81 of 108), `health` 0.759 (22 of 29), `baby`
  0.804 (119 of 148).
- Training took 18 s, prediction of 5,938 items 175 ms (CPU, i5-13500). Peak RAM of the whole
  script (with pandas) was 390 MB.

### TF-IDF with `tfidf.py` (2026-10-26)

`prepare.py` reproduces every count in section 4 (scikit-learn 1.9.1). `tfidf.py` reproduces
the preliminary numbers exactly: main 0.967 / 0.916 / 197 errors, brand split 0.927 / 0.815 /
431 errors. Main split: training 17.5 s, batch size 1 latency p50 0.30 ms and p99 0.43 ms,
whole test set in one batch 0.14 s, peak RSS 350 MB.

Learning curve on `test` (5 seeds, range over seeds):

| nominal size (real) | accuracy | macro-F1 |
| --- | ---: | ---: |
| 50 (53) | 0.573-0.630 | 0.218-0.334 |
| 200 (202) | 0.738-0.761 | 0.421-0.514 |
| 1,000 (1,001) | 0.875-0.885 | 0.702-0.728 |
| 5,000 (5,000) | 0.941-0.946 | 0.851-0.876 |
| all train (20,777) | 0.967 | 0.916 |

### BERTić `main`, first run (2026-10-26)

Settings of 6.2 (5 epochs, no class weights). Training 282 s on the RTX 3060 (11.5 steps/s),
first logged loss 2.296.

| epoch | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev accuracy | 0.887 | 0.952 | 0.961 | 0.969 | 0.968 |
| dev macro-F1 | 0.568 | 0.808 | 0.828 | 0.904 | 0.911 |
| dev loss | 0.413 | 0.220 | 0.168 | 0.153 | 0.158 |

- Best checkpoint by dev macro-F1: epoch 5. Dev macro-F1 was still rising (6.2 says to check
  this). Dev recall: `health` 0.571 (8 of 14), `toiletries` 0.907, `baby` 0.797.
- Test: accuracy 0.964, macro-F1 0.887, 215 errors (TF-IDF: 0.967, 0.916, 197).
- Inference (`bertic_run.py`): GPU batch size 1 p50 / p90 / p99 2.97 / 3.09 / 3.52 ms, batch 512
  1,219 items/s; CPU batch size 1 18.1 / 19.4 / 21.0 ms, batch 512 110 items/s.
- transformers 5.4 issue: checkpoints store the LayerNorm weights as `gamma`/`beta` (the names of
  the original BERTić checkpoint). `from_pretrained` renames them; the Trainer's
  `load_best_model_at_end` does not, so the LayerNorm weights keep the last epoch's values. In
  this run the best epoch was the last one, so the model is correct. `bertic_train.py` now
  loads the best checkpoint with `from_pretrained`.

### API smoke test (2026-10-26)

First 50 `tune` items, prompt `v1`, results variant `smoke`. All four methods answered all 50
items with the settings of section 6. Haiku and Qwen reported 0 reasoning tokens. Haiku ran
on Anthropic, Qwen on Near AI.

| method | correct | input tokens | output tokens | cost per 1,000 items | latency p50 / p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Jev (`jev-1.14.0`) | 49 / 50 | 654 | 98 (free) | $0.027 (tokens × list price) | 164 / 189 ms |
| Luna | 50 / 50 | 376 | 0 | $0.038 (tokens × list price) | 256 / 395 ms |
| Haiku 5.5 | 48 / 50 | 849 | 17 | $0.094 (`usage.cost`) | 1,222 / 2,321 ms |
| Qwen3.8 27B | 48 / 50 | 338 | 21 | $0.041 (`usage.cost`) | 668 / 948 ms |

- For Haiku and Qwen, `usage.cost` agrees with tokens × list price.
- Jev and Luna ran without the warm-up (6.6); their latency leaves out the first 16 requests.
  With n = 34-50, the latency is only an indication.
- Haiku reported 0 cached tokens (the prompt is shorter than its cache minimum). Qwen reported
  cached tokens on some requests.
- Estimated API cost from these numbers: main run on 11,087 items about $2.20 for all four
  methods; 5 tune variants × 500 items about $0.50.

### Prompt tuning log (500 `tune` items)

Prompt texts are in `config/`. v2 and v3 use the same instruction text for all four methods.

| variant | change | Luna | Jev | Haiku | Qwen |
| --- | --- | ---: | ---: | ---: | ---: |
| v1 | "Which store category does this product belong to?" | 481 | 466 | 465 | 451 |
| v2 | v1 + the input is a Croatian store listing, in Croatian, can contain the brand | 479 | 468 | 472 | 464 |
| v3 | v2 + decide by what the product is, not by flavour, scent or ingredient words | 482 | 467 | 471 | 460 |

Correct answers out of 500; no non-answers, no retries. Exact McNemar, v1 → v2: Haiku 8 fixed /
1 broken (p = 0.04), Qwen 23 / 10 (p = 0.04), Luna and Jev p > 0.6. v2 → v3: p > 0.3 for all.

- Qwen changes many answers between variants (33 of 500 from v1 to v2). It samples at
  temperature 1 (6.5).
- 14 items are wrong for all four methods. Most look like label errors or labels that need
  information outside the name, for example "Mladi špinat" (spinach) and "Gel za tuširanje"
  (shower gel) labeled `baby`, and cleaning devices ("Rovus Multi Cleaner", "čistač podova
  1 kom") labeled `household`.
- **Frozen 2026-10-26: v2 for all four methods** (decision after v1-v3). Variants v4 and v5 and
  the Haiku thinking ablation (6.5) were not run. Total tuning spend: $0.31.

### BERTić decision after the first run (2026-10-26)

The two dev checks of 6.2 fired: dev macro-F1 still rising at epoch 5, and dev recall of
`health` 8 of 14. Decision: train `main` and `brand` again with 10 epochs, best checkpoint by
dev macro-F1, no class weights. Add class weights only if dev `health` recall is still below
0.75 (fewer than 11 of 14). The learning-curve schedule does not change. The first run printed
its test score (above) before this decision; the decision uses the dev numbers only.

Result (10 epochs): `main` dev macro-F1 by epoch 0.522, 0.794, 0.842, 0.899, 0.904, 0.907,
0.917, 0.924, 0.923, 0.928, best epoch 10; dev `health` recall 11 of 14, so no class weights.
Test: accuracy 0.965, macro-F1 0.913, 209 errors. `brand`: best epoch 4 (dev macro-F1 0.853),
test 0.920 / 0.802 / 472 errors. Training 574 s and 562 s.

### Label review (2026-10-26)

77 of the 568 disagreement items (first batch: TF-IDF, BERTić 5 epochs, Jev, Luna, part of
Haiku), in the page's random order, so a random sample of that set. Verdicts: 63 label correct,
14 label wrong (the reviewer noted these could also be read as correct), 0 convention, 0
ambiguous. Label wrong: 18% (95% CI 11-28%) of the disagreement items, about 100 items or 1.7%
of the test set. Among the reviewed items where a method disagrees with the label, the method
is right in 12-22% of them, with no clear difference between methods. Review stopped at 77
(reviewer's time); the report uses raw accuracy and states this estimate.

## 13. Change log

Record here every change to the specification (category descriptions), to the input format or to
the split, with the date and the reason.

| date | change | reason |
| --- | --- | --- |
| 2026-10-26 | Plan created | |
