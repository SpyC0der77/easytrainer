"""
THIS FILE IS SPECIFICALLY FOR THE GOEMOTIONS BIO DATASET.

It loads sdeakin/GoEmotions-Projected-BIO-Emotions, converts emotion
spans into per-token BIO tags (B-Joy, I-Sadness, O, ...), and splits
train/val. Swap this file (and config.json) for any other
token-classification dataset.
"""

import json
from pathlib import Path

import progress  # noqa: F401  # configure tqdm before datasets starts bars
from datasets import load_dataset

cfg = json.loads((Path(__file__).parent / "config.json").read_text())


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


def bio_to_spans(tokens, tags):
    return [(emo, " ".join(tokens[s:e])) for s, e, emo in tag_spans(tags)]


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


raw = load_dataset("json", data_files=cfg["data_url"], split="train")
ds = raw.map(to_example, remove_columns=raw.column_names)
ds = ds.filter(lambda row: len(row["tokens"]))
split = ds.train_test_split(test_size=cfg["test_size"], seed=cfg["seed"])
train_ds, eval_ds = split["train"], split["test"]

labels = sorted({tag for tags in ds["bio_tags"] for tag in tags})
labels.remove("O")
labels = ["O"] + labels
label2id = {name: i for i, name in enumerate(labels)}
id2label = {i: name for name, i in label2id.items()}
