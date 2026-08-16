"""Task plugin contract.

A third task may join v1's core only if it is: schema + model class + collator +
metrics + preprocess. If that is not enough, the core is too fat.
"""

from __future__ import annotations

from typing import Any, Protocol

from datasets import Dataset

from easytrain.result import LabelInfo, SchemaInfo


class TaskPlugin(Protocol):
    type: str
    pipeline_task: str
    default_learning_rate: float
    metrics_names: tuple[str, ...]
    model_class_name: str
    collator_class_name: str
    peft_task_type: str
    metric_for_best_model: str
    preprocess_summary: str
    mapping_example: str

    def get_model_class(self) -> type: ...

    def get_collator(self, tokenizer: Any) -> Any: ...

    def validate_schema(self, dataset: Dataset) -> SchemaInfo: ...

    def infer_labels(self, dataset: Dataset) -> LabelInfo: ...

    def preprocess(self, dataset: Dataset, tokenizer: Any, labels: LabelInfo, max_length: int) -> Dataset: ...

    def compute_metrics(self, labels: LabelInfo): ...

    def explain_notes(self, labels: LabelInfo) -> list[str]: ...
