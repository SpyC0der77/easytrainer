"""Score a trained token classifier. Example: emotion-span BIO tags."""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import progress
import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from preprocess import bio_to_spans, eval_ds, id2label, label2id, tag_spans

root = Path(__file__).parent
cfg = json.loads((root / "config.json").read_text())


def make_tokenize(tokenizer):
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

    return tokenize


def compute_metrics(eval_pred):
    logits, label_ids = eval_pred
    pred_ids = np.argmax(logits, axis=-1)
    gold_spans, pred_spans = [], []
    n_ok = n = 0
    offset = 0
    for pred_row, gold_row in zip(pred_ids, label_ids):
        gold, pred = [], []
        for p, g in zip(pred_row, gold_row):
            if g == -100:
                continue
            gold.append(id2label[int(g)])
            pred.append(id2label[int(p)])
            n_ok += int(p == g)
            n += 1
        gold_spans += [(s + offset, e + offset, emo) for s, e, emo in tag_spans(gold)]
        pred_spans += [(s + offset, e + offset, emo) for s, e, emo in tag_spans(pred)]
        offset += len(gold) + 1
    tp = len(set(gold_spans) & set(pred_spans))
    precision = tp / len(pred_spans) if pred_spans else 0.0
    recall = tp / len(gold_spans) if gold_spans else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "token_accuracy": n_ok / n,
        "span_precision": precision,
        "span_recall": recall,
        "span_f1": f1,
        "n_gold_spans": len(gold_spans),
        "n_pred_spans": len(pred_spans),
        "n_exact_span_hits": tp,
    }


def pct(x):
    return f"{100 * x:.1f}%"


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


def print_metric_report(metrics):
    acc = metrics["eval_token_accuracy"]
    precision = metrics["eval_span_precision"]
    recall = metrics["eval_span_recall"]
    f1 = metrics["eval_span_f1"]
    n_gold = int(metrics["eval_n_gold_spans"])
    n_pred = int(metrics["eval_n_pred_spans"])
    n_hit = int(metrics["eval_n_exact_span_hits"])
    print("Held-out comments (validation split)")
    print(
        f"  Word tags: {pct(acc)} of words got the same BIO tag as the labels. "
        "Most words are ordinary (O), so this number runs high even when emotion phrases are shaky."
    )
    print(
        f"  Emotion phrases: the labels have {n_gold}, the model marked {n_pred}, "
        f"and {n_hit} match exactly (same words and same emotion)."
    )
    print(
        f"  When the model marks a phrase, it is exactly right {pct(precision)} of the time (precision)."
    )
    print(f"  It finds {pct(recall)} of the labeled phrases (recall).")
    print(
        f"  Combined score (span F1): {pct(f1)}. "
        "A hit has to get both the phrase boundaries and the emotion name right."
    )


def print_examples(predict_tags, n):
    print("\nA few comments, in plain English")
    shown = 0
    for row in eval_ds:
        pred = predict_tags(row["tokens"])
        gold_spans = bio_to_spans(row["tokens"], row["bio_tags"])
        pred_spans = bio_to_spans(row["tokens"], pred)
        if not gold_spans and not pred_spans:
            continue
        print()
        print(describe_example(row["text"], gold_spans, pred_spans))
        shown += 1
        if shown == n:
            break


def predict_tags(model, tokenizer, tokens):
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=cfg["max_length"],
        return_tensors="pt",
    )
    word_ids = encoded.word_ids()
    device = next(model.parameters()).device
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


if __name__ == "__main__":
    model_dir = str(root / cfg["output_dir"] / "best_model")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    tokenized_eval = eval_ds.map(
        make_tokenize(tokenizer), batched=True, remove_columns=eval_ds.column_names
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(root / cfg["output_dir"]),
            per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
            fp16=torch.cuda.is_available(),
            report_to="none",
            disable_tqdm=progress.disable_tqdm(),
        ),
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    print_metric_report(trainer.evaluate())
    print_examples(lambda tokens: predict_tags(model, tokenizer, tokens), cfg["show_examples"])
