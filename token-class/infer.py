"""Run a trained token classifier. Example: emotion-span BIO tags."""

# --- imports ---
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import progress  # noqa: F401
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from preprocess import eval_ds, id2label, tag_spans

# --- config ---
root = Path(__file__).parent
cfg = json.loads((root / "config.json").read_text())


# --- predict ---
def predict_tags(model, tokenizer, tokens):
    tags = ["O"] * len(tokens)
    device = next(model.parameters()).device
    start = 0
    while start < len(tokens):
        encoded = tokenizer(
            tokens[start:],
            is_split_into_words=True,
            truncation=True,
            max_length=cfg["max_length"],
            return_tensors="pt",
        )
        word_ids = encoded.word_ids()
        inputs = {k: encoded[k].to(device) for k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            pred_ids = model(**inputs).logits[0].argmax(-1).tolist()
        seen = set()
        last = -1
        for word_id, pred_id in zip(word_ids, pred_ids):
            if word_id is None or word_id in seen:
                continue
            tags[start + word_id] = id2label[pred_id]
            seen.add(word_id)
            last = word_id
        if last < 0:
            break
        start += last + 1
    return tags


def bio_to_spans(tokens, tags):
    return [(emo, " ".join(tokens[s:e])) for s, e, emo in tag_spans(tags)]


def describe_spans(spans):
    if not spans:
        return "no emotion"
    return "; ".join(f"{emo} in “{text}”" for emo, text in spans)


def describe_example(text, gold, pred):
    lines = [f"Comment: {text.strip()}"]
    if gold == pred:
        if gold:
            lines.append(f"Match: {describe_spans(gold)}.")
        else:
            lines.append("Match: neither the labels nor the model marked an emotion.")
        return "\n".join(lines)
    lines.append(f"Labeled: {describe_spans(gold)}.")
    lines.append(f"Model:   {describe_spans(pred)}.")
    return "\n".join(lines)


def print_examples(model, tokenizer, n):
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
        if shown == n:
            break


# --- infer ---
if __name__ == "__main__":
    model_dir = str(root / cfg["output_dir"] / "best_model")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    print_examples(model, tokenizer, cfg["show_examples"])
