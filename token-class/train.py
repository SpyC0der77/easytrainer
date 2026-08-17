"""Train a token classifier. Example labels: emotion-span BIO tags."""

# --- imports ---
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import progress
import torch
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from evaluate import compute_metrics
from preprocess import id2label, label2id, labels, tokenized_eval, tokenized_train, tokenizer

# --- config ---
root = Path(__file__).parent
cfg = json.loads((root / "config.json").read_text())
output_dir = str(root / cfg["output_dir"])

os.environ["WANDB_DISABLED"] = "true"
set_seed(cfg["seed"])

print(len(tokenized_train), "train,", len(tokenized_eval), "val,", len(labels), "labels")

# --- model ---
model = AutoModelForTokenClassification.from_pretrained(
    cfg["model"], num_labels=len(labels), id2label=id2label, label2id=label2id
)

# --- train ---
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
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model=cfg["metric_for_best_model"],
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=cfg["seed"],
        disable_tqdm=progress.disable_tqdm(),
        logging_steps=cfg["logging_steps"],
        logging_first_step=True,
    ),
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    processing_class=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])],
)
trainer.train()
trainer.save_model(str(Path(output_dir) / "best_model"))
