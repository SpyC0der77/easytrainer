"""Train a token classifier. Example labels: emotion-span BIO tags."""

import json
import math
import os
import sys
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import progress
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0",
    category=UserWarning,
)

from evaluate import compute_metrics, make_tokenize
from preprocess import eval_ds, id2label, label2id, labels, train_ds

root = Path(__file__).parent
cfg = json.loads((root / "config.json").read_text())
output_dir = str(root / cfg["output_dir"])

os.environ["WANDB_DISABLED"] = "true"
set_seed(cfg["seed"])

print(len(train_ds), "train,", len(eval_ds), "val,", len(labels), "labels")

tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
model = AutoModelForTokenClassification.from_pretrained(
    cfg["model"],
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)


tokenize = make_tokenize(tokenizer)
tokenized_train = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
tokenized_eval = eval_ds.map(tokenize, batched=True, remove_columns=eval_ds.column_names)

n_devices = max(torch.cuda.device_count(), 1)
batch_size = cfg["per_device_train_batch_size"] * n_devices
steps_per_epoch = max(1, math.ceil(len(tokenized_train) / batch_size))
warmup_steps = int(steps_per_epoch * cfg["num_train_epochs"] * cfg["warmup_ratio"])

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=warmup_steps,
        eval_strategy=cfg["eval_strategy"],
        save_strategy=cfg["save_strategy"],
        save_total_limit=cfg.get("save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model=cfg["metric_for_best_model"],
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=cfg["seed"],
        disable_tqdm=progress.disable_tqdm(),
        logging_steps=cfg.get("logging_steps", 50),
        logging_first_step=True,
    ),
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    processing_class=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model(str(Path(output_dir) / "best_model"))
