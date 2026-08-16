"""Train a token classifier. Example labels: emotion-span BIO tags."""

import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from preprocess import eval_ds, id2label, label2id, labels, tag_spans, train_ds

root = Path(__file__).parent
cfg = json.loads((root / "config.json").read_text())
output_dir = str(root / cfg["output_dir"])

os.environ["WANDB_DISABLED"] = "true"
set_seed(cfg["seed"])

print(len(train_ds), "train,", len(eval_ds), "val,", len(labels), "labels")

tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
model = AutoModelForTokenClassification.from_pretrained(
    cfg["model"], num_labels=len(labels), id2label=id2label, label2id=label2id
)


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


trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        eval_strategy=cfg["eval_strategy"],
        save_strategy=cfg["save_strategy"],
        load_best_model_at_end=True,
        metric_for_best_model=cfg["metric_for_best_model"],
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=cfg["seed"],
    ),
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    processing_class=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)
trainer.train()
print(trainer.evaluate())
trainer.save_model(str(Path(output_dir) / "best_model"))
