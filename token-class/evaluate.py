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

from preprocess import eval_ds, id2label, label2id, tag_spans

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
    }


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
    print(trainer.evaluate())
