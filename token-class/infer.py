"""Run a trained token classifier. Example: emotion-span BIO tags."""

import json
from pathlib import Path

import progress  # noqa: F401
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from preprocess import bio_to_spans, eval_ds, id2label

root = Path(__file__).parent
cfg = json.loads((root / "config.json").read_text())
model_dir = str(root / cfg["output_dir"] / "best_model")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)
device = next(model.parameters()).device
model.eval()


def predict(tokens):
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=cfg["max_length"],
        return_tensors="pt",
    )
    word_ids = encoded.word_ids()
    inputs = {k: encoded[k].to(device) for k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        pred_ids = model(**inputs).logits[0].argmax(-1).tolist()
    tags = ["O"] * len(tokens)
    seen = set()
    for word_id, pred_id in zip(word_ids, pred_ids):
        if word_id is None or word_id in seen:
            continue
        tags[word_id] = id2label[pred_id]
        seen.add(word_id)
    return tags


shown = 0
for row in eval_ds:
    if not any(tag.startswith("B-") for tag in row["bio_tags"]):
        continue
    pred = predict(row["tokens"])
    gold_spans = bio_to_spans(row["tokens"], row["bio_tags"])
    pred_spans = bio_to_spans(row["tokens"], pred)
    print("\n" + row["text"])
    print("  gold:", " | ".join(f"[{e}] {t}" for e, t in gold_spans) or "(none)")
    print("  pred:", " | ".join(f"[{e}] {t}" for e, t in pred_spans) or "(none)")
    for token, gold, tag in zip(row["tokens"], row["bio_tags"], pred):
        mark = " " if gold == tag else "!"
        print(f"  {mark} {token:16} {gold:18} {tag}")
    shown += 1
    if shown == cfg["show_examples"]:
        break
