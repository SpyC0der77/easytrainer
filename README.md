# EasyTrain

EasyTrain is [`transformers.pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines) for **training**.

You pick a task, a pretrained model, a dataset, an output directory, and how many epochs. EasyTrain selects the Hugging Face / PyTorch stack — model class, tokenizer, collator, `TrainingArguments`, `Trainer`, metrics, checkpoints, save/push, and the fastest **stable** acceleration for the machine that is actually present.

Students still learn datasets, loss, epochs, batches, learning rate, backpropagation, fine-tuning, and evaluation. They should not have to reimplement that machinery for every project.

```python
from easytrain import train

result = train(
    type="text-classification",
    model="distilbert/distilbert-base-uncased",
    dataset="my-dataset",
    output="my-model",
    epochs=3,
)
```

After `train()`, `transformers.pipeline(type, output)` works on the saved folder.

## v1

v1 ships **two** encoder tasks on **one** Trainer core, so the library is modular from day one:

| `type` | Model | Columns | Collator | Metrics |
|---|---|---|---|---|
| `text-classification` | `AutoModelForSequenceClassification` | `text` + `label` (or `sentence1`/`sentence2`) | `DataCollatorWithPadding` | accuracy / F1 |
| `token-classification` | `AutoModelForTokenClassification` | `tokens` + `ner_tags` | `DataCollatorForTokenClassification` | seqeval |

Later (same `train()` router): `causal-language-modeling` (TRL `SFTTrainer`), `seq2seq`, `image-classification`, `image-to-text`, `text-to-image`.

## Install

```bash
pip install easytrain
```

Install a PyTorch build that matches your machine first (`cpu`, `cu124`, …). EasyTrain does not invent an optimizer or a training loop; it uses Transformers, Datasets, Accelerate, PEFT, evaluate, and seqeval.

Tested lower bounds: Python 3.10+, `torch>=2.2`, `transformers>=4.44`, `datasets>=2.21`, `accelerate>=0.34`, `peft>=0.12`, `evaluate>=0.4.3`, `seqeval>=1.2.2`.

## API

**Required:** `type`, `model`, `dataset`, `output`, `epochs`.

**Optional:** `batch_size="auto"`, `learning_rate=None` (task default), `peft="auto"`, `eval=True`, `push_to_hub=False`, `seed=42`, `speed="auto"`.

**Escape hatches:** `**training_args` forwarded to `TrainingArguments`. `explain=True` / `dry_run=True` prints the resolved plan (model class, columns, collator, metrics, dtype, attention, PEFT, batch, estimated VRAM) and writes `train_config.json` plus an equivalent `Trainer` snippet so you can graduate out.

```python
train(
    type="token-classification",
    model="distilbert/distilbert-base-uncased",
    dataset="conll2003",
    output="my-ner-model",
    epochs=3,
    explain=True,   # print the plan, then train
)

train(..., dry_run=True)  # print the plan, do not train
```

Sklearn-style wrapper, same router:

```python
from easytrain import EasyTrainer

EasyTrainer(
    type="text-classification",
    model="distilbert/distilbert-base-uncased",
    dataset="my-dataset",
    output="my-model",
    epochs=3,
).fit().evaluate().save()
```

`train()` returns a small result: metrics, `output_dir`, model id, underlying `Trainer`, and the resolved plan.

### Speed

`speed` is `"auto"` (default), `"stable"` (bf16/fp16, no compile, no fp8), or `"max"` (try `torch.compile`; fall back on failure).

v1 encoder autodetection: fit a batch in VRAM, SDPA attention, BF16 on Ampere+, FP16 on older CUDA, fused AdamW on GPU, `tf32` where it exists. No Unsloth, no FP8, no DeepSpeed in v1 — those are LLM-later. DistilBERT-sized classification is always full fine-tune unless you pass `peft="lora"`. You never set `target_modules`.

### Data

Strict column conventions. EasyTrain **does not guess**. Failures include a mapping example:

```python
train(
    type="text-classification",
    model="distilbert/distilbert-base-uncased",
    dataset={"path": "reviews.csv", "text": "review", "label": "sentiment"},
    output="my-model",
    epochs=3,
)
```

`dataset` may be a Hub id (`"imdb"`, `"glue:sst2"`), a local CSV/JSON/Parquet file or directory, a `datasets.Dataset` / `DatasetDict`, or a mapping dict.

Token classification documents subword label alignment (`word_ids`, `-100` on specials and continuations) in `explain` output. That concept stays visible.

## What EasyTrain is not

- Not a no-code UI.
- Not a new training algorithm.
- Not an LLM-only factory (Axolotl / LLaMA-Factory / Unsloth).
- Not a huge task zoo. Unknown `type=` is rejected.

Hugging Face AutoTrain Advanced did a version of this (unmaintained). Simple Transformers is the closest living Python API (NLP-only, dated). EasyTrain’s niche is a tiny, educational, task-first `train()` with auto hardware speed.
