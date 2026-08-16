from __future__ import annotations

from typing import Any

from easytrain.constants import __version__
from easytrain.result import TrainingPlan


def format_plan(plan: TrainingPlan) -> str:
    labels = ", ".join(f"{i}:{name}" for i, name in plan.labels.id2label.items())
    lines = [
        "EasyTrain plan",
        "==============",
        f"task:            {plan.task_type}",
        f"model:           {plan.model}",
        f"model class:     {plan.model_class}",
        f"dataset:         {plan.dataset_source}",
        f"columns:         {', '.join(plan.columns)}",
        f"schema:          {plan.schema_mode}",
        f"labels:          {plan.labels.num_labels}  [{labels}]",
        f"collator:        {plan.collator}",
        f"metrics:         {', '.join(plan.metrics)}",
        f"tokenizer:       {plan.tokenizer_class or '(unresolved)'}",
        f"preprocess:      {plan.preprocess}",
        f"dtype:           {plan.speed.precision}",
        f"attention:       {plan.speed.attn_implementation or 'default'}",
        f"optim:           {plan.speed.optim}",
        f"peft:            {plan.peft}",
        f"batch:           {plan.speed.batch_size} per device",
        f"tf32:            {plan.speed.tf32}",
        f"workers:         {plan.speed.num_workers}",
        f"pin_memory:      {plan.speed.pin_memory}",
        f"compile:         {plan.speed.torch_compile}",
        f"hardware:        {plan.hardware.device} ({plan.hardware.name})",
        f"eval:            {plan.eval_enabled}  split={plan.eval_split or 'none'}",
        f"epochs:          {plan.epochs}",
        f"learning_rate:   {plan.learning_rate}",
    ]
    if plan.estimated_vram_gb is not None:
        lines.append(f"estimated VRAM:  ~{plan.estimated_vram_gb:.2f} GB")
    lines.append("")
    lines.append(f"Why it's fast: {plan.why_fast}")
    if plan.speed.fallbacks:
        lines.append("Fallbacks: " + "; ".join(plan.speed.fallbacks))
    if plan.notes:
        lines.append("")
        lines.append("Notes")
        lines.append("-----")
        for note in plan.notes:
            lines.append(f"- {note}")
    if plan.alignment_example:
        lines.append("")
        lines.append("Subword label alignment (students: this is the NER trick)")
        lines.append("-------------------------------------------------------")
        lines.append(plan.alignment_example)
    lines.append("")
    lines.append("These are the same Hugging Face objects you would wire by hand:")
    lines.append(f"  {plan.model_class}, {plan.collator}, Trainer, TrainingArguments.")
    return "\n".join(lines) + "\n"


def plan_to_config(plan: TrainingPlan) -> dict[str, Any]:
    return {
        "easytrain": __version__,
        "type": plan.task_type,
        "model": plan.model,
        "dataset": plan.dataset_source,
        "output": plan.training_arguments.get("output_dir"),
        "epochs": plan.epochs,
        "resolved": {
            "model_class": plan.model_class,
            "collator": plan.collator,
            "metrics": list(plan.metrics),
            "tokenizer_class": plan.tokenizer_class,
            "columns": plan.columns,
            "schema_mode": plan.schema_mode,
            "num_labels": plan.labels.num_labels,
            "id2label": {str(k): v for k, v in plan.labels.id2label.items()},
            "peft": plan.peft,
            "dtype": plan.speed.precision,
            "attention": plan.speed.attn_implementation,
            "optim": plan.speed.optim,
            "batch_size": plan.speed.batch_size,
            "hardware": {
                "device": plan.hardware.device,
                "name": plan.hardware.name,
                "capability": list(plan.hardware.capability) if plan.hardware.capability else None,
                "vram_gb": plan.hardware.vram_gb,
            },
            "why_fast": plan.why_fast,
            "estimated_vram_gb": plan.estimated_vram_gb,
        },
        "training_arguments": plan.training_arguments,
    }


def trainer_snippet(plan: TrainingPlan) -> str:
    id2label = plan.labels.id2label
    label2id = plan.labels.label2id
    is_token = plan.task_type == "token-classification"
    is_pair = plan.schema_mode == "pair"
    if is_token:
        tokenize_block = f"""
label2id = {label2id!r}

def encode_label(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return label2id[value]

def tokenize_and_align(batch):
    encoded = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True)
    aligned = []
    for i, tags in enumerate(batch["ner_tags"]):
        word_ids = encoded.word_ids(batch_index=i)
        previous = None
        ids = []
        for word_id in word_ids:
            if word_id is None:
                ids.append(-100)
            elif word_id != previous:
                ids.append(encode_label(tags[word_id]))
            else:
                ids.append(-100)
            previous = word_id
        aligned.append(ids)
    encoded["labels"] = aligned
    return encoded
""".strip()
    elif is_pair:
        tokenize_block = f"""
label2id = {label2id!r}

def encode_label(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return label2id[value]

def tokenize(batch):
    encoded = tokenizer(batch["sentence1"], batch["sentence2"], truncation=True)
    encoded["labels"] = [encode_label(v) for v in batch["label"]]
    return encoded
""".strip()
    else:
        tokenize_block = f"""
label2id = {label2id!r}

def encode_label(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return label2id[value]

def tokenize(batch):
    encoded = tokenizer(batch["text"], truncation=True)
    encoded["labels"] = [encode_label(v) for v in batch["label"]]
    return encoded
""".strip()
    collator_import = plan.collator
    args = plan.training_arguments
    lr = args.get("learning_rate", plan.learning_rate)
    batch = args.get("per_device_train_batch_size", plan.speed.batch_size)
    bf16 = args.get("bf16", plan.speed.precision == "bf16")
    fp16 = args.get("fp16", plan.speed.precision == "fp16")
    optim = args.get("optim", plan.speed.optim)
    eval_strategy = args.get("eval_strategy", args.get("evaluation_strategy", "epoch"))
    return f"""# Equivalent Hugging Face Trainer script generated by EasyTrain {__version__}
# Task: {plan.task_type}

from transformers import (
    AutoTokenizer,
    {plan.model_class},
    {collator_import},
    Trainer,
    TrainingArguments,
)

tokenizer = AutoTokenizer.from_pretrained({plan.model!r})
model = {plan.model_class}.from_pretrained(
    {plan.model!r},
    num_labels={plan.labels.num_labels},
    id2label={id2label!r},
    label2id={label2id!r},
    ignore_mismatched_sizes=True,
)

{tokenize_block}

# tokenized = dataset.map({"tokenize_and_align" if is_token else "tokenize"}, batched=True)

args = TrainingArguments(
    output_dir={args.get("output_dir", "./output")!r},
    num_train_epochs={plan.epochs},
    per_device_train_batch_size={batch},
    learning_rate={lr},
    eval_strategy={eval_strategy!r},
    bf16={bf16},
    fp16={fp16},
    optim={optim!r},
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator={collator_import}(tokenizer=tokenizer),
    # train_dataset=tokenized["train"],
    # eval_dataset=tokenized.get("validation"),
    # compute_metrics=...,
)
# trainer.train()
# trainer.save_model()
# tokenizer.save_pretrained(args.output_dir)
"""
