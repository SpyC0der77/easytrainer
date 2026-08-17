"""Run a trained token classifier. Example: emotion-span BIO tags."""

# --- imports ---
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import progress  # noqa: F401
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from evaluate import bio_to_spans, describe_example, predict_tags
from preprocess import eval_ds

# --- config ---
root = Path(__file__).parent
cfg = json.loads((root / "config.json").read_text())
model_dir = str(root / cfg["output_dir"] / "best_model")

# --- model ---
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)
model.eval()

# --- infer ---
shown = 0
for row in eval_ds:
    pred = predict_tags(model, tokenizer, row["tokens"])
    gold_spans = bio_to_spans(row["tokens"], row["bio_tags"])
    pred_spans = bio_to_spans(row["tokens"], pred)
    if not gold_spans and not pred_spans:
        continue
    print()
    print(describe_example(row["text"], gold_spans, pred_spans))
    shown += 1
    if shown == cfg["show_examples"]:
        break
