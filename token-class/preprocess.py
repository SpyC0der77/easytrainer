"""
THIS FILE IS SPECIFICALLY FOR THE GOEMOTIONS BIO DATASET.

It loads sdeakin/GoEmotions-Projected-BIO-Emotions, converts emotion
spans into per-token BIO tags (B-Joy, I-Sadness, O, ...), splits
train/val, and tokenizes. Swap this file (and config.json) for any
other token-classification dataset.
"""

# --- imports ---
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import progress  # noqa: F401  # configure tqdm before datasets starts bars
from datasets import load_dataset
from transformers import AutoTokenizer

# --- config ---
cfg = json.loads((Path(__file__).parent / "config.json").read_text())


# --- BIO tags ---
def spans_to_bio(tokens, spans):
    tags = ["O"] * len(tokens)
    n = len(tokens)
    for span in spans:
        emotion = span.get("subtype") or span.get("type")
        start, end = span.get("start"), span.get("end")
        if not emotion or start is None or end is None or start < 0 or start >= n:
            continue
        emotion = emotion.replace(" ", "_").replace("-", "_")
        tags[start] = f"B-{emotion}"
        for i in range(start + 1, min(end, n - 1) + 1):
            tags[i] = f"I-{emotion}"
    return tags


def tag_spans(tags):
    spans, i = [], 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            emotion, j = tags[i][2:], i + 1
            while j < len(tags) and tags[j] == f"I-{emotion}":
                j += 1
            spans.append((i, j, emotion))
            i = j
        else:
            i += 1
    return spans


def to_example(row):
    tokens = list(row["data"]["tokens"])
    tags = spans_to_bio(tokens, row["data"].get("spans") or [])
    while tokens and tokens[-1] == "":
        tokens.pop()
        tags.pop()
    return {
        "text": row["text"],
        "tokens": tokens,
        "bio_tags": tags,
    }


# --- load + split ---
raw = load_dataset("json", data_files=cfg["data_url"], split="train")
ds = raw.map(to_example, remove_columns=raw.column_names)
ds = ds.filter(lambda row: len(row["tokens"]))
split = ds.train_test_split(test_size=cfg["test_size"], seed=cfg["seed"])
train_ds, eval_ds = split["train"], split["test"]

# --- labels ---
labels = sorted({tag for tags in ds["bio_tags"] for tag in tags})
labels.remove("O")
labels = ["O"] + labels
label2id = {name: i for i, name in enumerate(labels)}
id2label = {i: name for name, i in label2id.items()}

# --- tokenize ---
tokenizer = AutoTokenizer.from_pretrained(cfg["model"])


def tokenize(batch):
    encoded = tokenizer(
        batch["tokens"], is_split_into_words=True, truncation=True, max_length=cfg["max_length"]
    )
    aligned = []
    for i, tags in enumerate(batch["bio_tags"]):
        word_ids = encoded.word_ids(batch_index=i)
        ids, prev = [], None
        for word_id in word_ids:
            if word_id is None:
                ids.append(-100)
            elif word_id != prev:
                ids.append(label2id[tags[word_id]])
            else:
                ids.append(-100)
            prev = word_id
        aligned.append(ids)
    encoded["labels"] = aligned
    return encoded


tokenized_train = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
tokenized_eval = eval_ds.map(tokenize, batched=True, remove_columns=eval_ds.column_names)
