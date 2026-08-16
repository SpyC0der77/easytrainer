from __future__ import annotations

from typing import Any

from datasets import Dataset

from easytrain.core.labels import encode_label_value, infer_label_info
from easytrain.errors import ConfigError, SchemaError
from easytrain.result import LabelInfo, SchemaInfo

type = "token-classification"
pipeline_task = "token-classification"
default_learning_rate = 5e-5
metrics_names = ("precision", "recall", "f1", "accuracy")
model_class_name = "AutoModelForTokenClassification"
collator_class_name = "DataCollatorForTokenClassification"
peft_task_type = "TOKEN_CLS"
metric_for_best_model = "eval_f1"
preprocess_summary = (
    "Tokenize pre-split words (`is_split_into_words=True`) and align word-level "
    "ner_tags onto subword tokens. Special tokens and continuation subwords get -100 "
    "so CrossEntropyLoss ignores them."
)
mapping_example = '''train(
    type="token-classification",
    model="distilbert/distilbert-base-uncased",
    dataset={"path": "ner.jsonl", "tokens": "words", "ner_tags": "tags"},
    output="my-ner-model",
    epochs=3,
)'''
stratify_column = None


def get_model_class() -> type:
    from transformers import AutoModelForTokenClassification

    return AutoModelForTokenClassification


def get_collator(tokenizer: Any) -> Any:
    from transformers import DataCollatorForTokenClassification

    return DataCollatorForTokenClassification(tokenizer=tokenizer)


def validate_schema(dataset: Dataset) -> SchemaInfo:
    columns = list(dataset.column_names)
    if "tokens" in columns and "ner_tags" in columns:
        return SchemaInfo(
            columns=columns,
            mode="tokens",
            notes=["Using word-level columns tokens + ner_tags (BIO or similar)."],
        )
    found = ", ".join(columns) if columns else "(none)"
    raise SchemaError(
        f"""Dataset is missing required columns for token-classification.

Found columns: {found}
Expected:
  - tokens   (list of words)
  - ner_tags (list of BIO tags or tag ids, same length as tokens)

EasyTrain does not guess column names. Rename them, or pass a mapping:

    {mapping_example}
"""
    )


def infer_labels(dataset: Dataset) -> LabelInfo:
    return infer_label_info(dataset, "ner_tags", kind="bio")


def align_labels_with_tokens(labels: list[int], word_ids: list[int | None]) -> list[int]:
    """Align word-level NER tags to subword tokens.

    Special tokens (`word_id is None`) and continuation subwords are set to -100
    so they are ignored by the loss. Only the first subword of each word keeps
    the label. This is the standard Hugging Face token-classification alignment.
    """
    aligned: list[int] = []
    previous_word: int | None = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)
        elif word_id != previous_word:
            aligned.append(labels[word_id])
        else:
            aligned.append(-100)
        previous_word = word_id
    return aligned


def format_alignment_example(example: dict[str, Any], tokenizer: Any, labels: LabelInfo, max_length: int) -> str:
    tokens = list(example["tokens"])
    raw_tags = list(example["ner_tags"])
    tag_ids = [encode_label_value(tag, labels) for tag in raw_tags]
    encoded = tokenizer(
        tokens,
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
        return_tensors=None,
    )
    word_ids = encoded.word_ids()
    aligned = align_labels_with_tokens(tag_ids, word_ids)
    pieces = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

    def cell(value: Any) -> str:
        text = str(value)
        return text if text != "None" else "-"

    rows = [
        "  tokens:   " + " ".join(tokens),
        "  ner_tags: " + " ".join(str(t) for t in raw_tags),
        "  subwords: " + " ".join(pieces),
        "  word_ids: " + " ".join(cell(w) for w in word_ids),
        "  labels:   " + " ".join(str(v) for v in aligned),
        "  (-100 = ignored by loss: specials and continuation subwords)",
    ]
    return "\n".join(rows)


def preprocess(dataset: Dataset, tokenizer: Any, labels: LabelInfo, max_length: int) -> Dataset:
    if not getattr(tokenizer, "is_fast", False):
        raise ConfigError(
            "token-classification needs a fast tokenizer so word_ids() can align "
            "subword tokens to ner_tags. Use a model that ships a fast tokenizer."
        )

    def tokenize_and_align(examples):
        encoded = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
        )
        aligned_batch = []
        for i, tags in enumerate(examples["ner_tags"]):
            tag_ids = [encode_label_value(tag, labels) for tag in tags]
            word_ids = encoded.word_ids(batch_index=i)
            aligned_batch.append(align_labels_with_tokens(tag_ids, word_ids))
        encoded["labels"] = aligned_batch
        return encoded

    return dataset.map(tokenize_and_align, batched=True, remove_columns=dataset.column_names)


def compute_metrics(labels: LabelInfo):
    from seqeval.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    id2label = labels.id2label

    def _compute(eval_pred):
        logits, gold = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = logits.argmax(axis=-1)
        true_predictions = []
        true_labels = []
        for pred_row, gold_row in zip(preds, gold, strict=False):
            pred_tags = []
            gold_tags = []
            for pred_id, gold_id in zip(pred_row, gold_row, strict=False):
                if gold_id == -100:
                    continue
                pred_tags.append(id2label[int(pred_id)])
                gold_tags.append(id2label[int(gold_id)])
            true_predictions.append(pred_tags)
            true_labels.append(gold_tags)
        try:
            return {
                "precision": float(precision_score(true_labels, true_predictions)),
                "recall": float(recall_score(true_labels, true_predictions)),
                "f1": float(f1_score(true_labels, true_predictions)),
                "accuracy": float(accuracy_score(true_labels, true_predictions)),
            }
        except Exception:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    return _compute


def explain_notes(labels: LabelInfo) -> list[str]:
    return [
        "Word-level ner_tags are aligned to subword tokens before training.",
        "word_ids() from the fast tokenizer marks which word each subword came from.",
        "-100 on specials ([CLS]/[SEP]/pad) and on continuation subwords (e.g. '##is').",
        "seqeval scores entity-level precision/recall/F1, not just token accuracy.",
        f"label list: {labels.names}",
    ]
