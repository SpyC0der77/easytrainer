from __future__ import annotations

from typing import Any

from datasets import Dataset

from easytrain.core.labels import encode_label_value, infer_label_info
from easytrain.errors import SchemaError
from easytrain.result import LabelInfo, SchemaInfo

type = "text-classification"
pipeline_task = "text-classification"
default_learning_rate = 2e-5
metrics_names = ("accuracy", "f1")
model_class_name = "AutoModelForSequenceClassification"
collator_class_name = "DataCollatorWithPadding"
peft_task_type = "SEQ_CLS"
metric_for_best_model = "eval_f1"
preprocess_summary = (
    "Tokenize the whole sequence (or sentence pair) with truncation. "
    "The `label` column becomes `labels` for CrossEntropyLoss."
)
mapping_example = """train(
    type="text-classification",
    model="distilbert/distilbert-base-uncased",
    dataset={"path": "reviews.csv", "text": "review", "label": "sentiment"},
    output="my-model",
    epochs=3,
)"""
stratify_column = "label"


def get_model_class() -> type:
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification


def get_collator(tokenizer: Any) -> Any:
    from transformers import DataCollatorWithPadding

    return DataCollatorWithPadding(tokenizer=tokenizer)


def validate_schema(dataset: Dataset) -> SchemaInfo:
    columns = list(dataset.column_names)
    has_text = "text" in columns and "label" in columns
    has_pair = "sentence1" in columns and "sentence2" in columns and "label" in columns
    if has_text:
        return SchemaInfo(columns=columns, mode="single", notes=["Using columns text + label."])
    if has_pair:
        return SchemaInfo(
            columns=columns,
            mode="pair",
            notes=["Using sentence-pair columns sentence1 + sentence2 + label."],
        )
    found = ", ".join(columns) if columns else "(none)"
    raise SchemaError(
        f"""Dataset is missing required columns for text-classification.

Found columns: {found}
Expected one of:
  - text, label
  - sentence1, sentence2, label   (sentence-pair classification)

EasyTrain does not guess column names. Rename them, or pass a mapping:

    {mapping_example}

For pairs, map sentence1/sentence2 instead of text.
"""
    )


def infer_labels(dataset: Dataset) -> LabelInfo:
    return infer_label_info(dataset, "label", kind="class")


def preprocess(dataset: Dataset, tokenizer: Any, labels: LabelInfo, max_length: int) -> Dataset:
    mode = "pair" if "sentence1" in dataset.column_names and "sentence2" in dataset.column_names else "single"

    def tokenize_batch(batch):
        if mode == "pair":
            encoded = tokenizer(
                batch["sentence1"],
                batch["sentence2"],
                truncation=True,
                max_length=max_length,
            )
        else:
            encoded = tokenizer(batch["text"], truncation=True, max_length=max_length)
        encoded["labels"] = [encode_label_value(value, labels) for value in batch["label"]]
        return encoded

    return dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)


def compute_metrics(labels: LabelInfo):
    from sklearn.metrics import accuracy_score, f1_score

    def _compute(eval_pred):
        logits, gold = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": float(accuracy_score(gold, preds)),
            "f1": float(f1_score(gold, preds, average="weighted", zero_division=0)),
        }

    return _compute


def explain_notes(labels: LabelInfo) -> list[str]:
    return [
        f"num_labels={labels.num_labels} inferred from the dataset; the classification head is resized.",
        f"id2label={labels.id2label}",
        "Loss is cross-entropy over the sequence. Accuracy and weighted F1 are reported if eval runs.",
    ]
