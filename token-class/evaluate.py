"""Score a trained token classifier. Example: emotion-span BIO tags."""

# --- imports ---
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

from infer import print_examples
from preprocess import id2label, tag_spans, tokenized_eval

# --- config ---
root = Path(__file__).parent
cfg = json.loads((root / "config.json").read_text())


# --- metrics ---
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


# --- plain-English report ---
def pct(x):
    return f"{100 * x:.1f}%"


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


# --- score saved model ---
if __name__ == "__main__":
    model_dir = str(root / cfg["output_dir"] / "best_model")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
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
    output = trainer.predict(tokenized_eval, metric_key_prefix="eval")
    print_metric_report(output.metrics)
    print("\nA few comments, in plain English")
    print_examples(model, tokenizer, cfg["show_examples"])
