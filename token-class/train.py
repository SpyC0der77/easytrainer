"""Train a token classifier. Example labels: emotion-span BIO tags."""

import json
import math
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import progress
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from evaluate import compute_metrics, make_tokenize
from preprocess import eval_ds, id2label, label2id, labels, train_ds

root = Path(__file__).parent
cfg = json.loads((root / "config.json").read_text())
output_dir = str(root / cfg["output_dir"])

os.environ["WANDB_DISABLED"] = "true"
set_seed(cfg["seed"])

print(len(train_ds), "train,", len(eval_ds), "val,", len(labels), "labels")


def add_layernorm_aliases(state_dict):
    """DistilBERT checkpoints mix LayerNorm.weight/bias with gamma/beta names."""
    extra = {}
    for key, value in state_dict.items():
        if key.endswith("LayerNorm.gamma"):
            extra[f"{key[:-5]}weight"] = value
        elif key.endswith("LayerNorm.beta"):
            extra[f"{key[:-4]}bias"] = value
        elif key.endswith("LayerNorm.weight"):
            extra[f"{key[:-6]}gamma"] = value
        elif key.endswith("LayerNorm.bias"):
            extra[f"{key[:-4]}beta"] = value
    state_dict.update(extra)
    return state_dict


def alias_saved_model(model_dir):
    model_dir = Path(model_dir)
    safetensors_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file, save_file

        save_file(add_layernorm_aliases(dict(load_file(safetensors_path))), str(safetensors_path))
    elif bin_path.exists():
        torch.save(add_layernorm_aliases(torch.load(bin_path, map_location="cpu")), bin_path)


class LayerNormAliasCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        alias_saved_model(Path(args.output_dir) / f"checkpoint-{state.global_step}")


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

callbacks = [LayerNormAliasCallback()]
patience = cfg.get("early_stopping_patience")
if patience:
    callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
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
    callbacks=callbacks,
)
progress.attach_aligned_logging(trainer)
train_loader = trainer.get_train_dataloader()
grad_accum = max(trainer.args.gradient_accumulation_steps, 1)
updates_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
total_updates = math.ceil(trainer.args.num_train_epochs * updates_per_epoch)
trainer.args.warmup_steps = math.ceil(total_updates * cfg["warmup_ratio"])
trainer.train()
best_dir = Path(output_dir) / "best_model"
trainer.save_model(str(best_dir))
alias_saved_model(best_dir)
