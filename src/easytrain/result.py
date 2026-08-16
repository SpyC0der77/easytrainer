from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from easytrain.core.speed import Hardware, SpeedPlan


@dataclass
class LabelInfo:
    num_labels: int
    id2label: dict[int, str]
    label2id: dict[str, int]
    label_column: str
    names: list[str]


@dataclass
class SchemaInfo:
    columns: list[str]
    mode: str
    notes: list[str] = field(default_factory=list)


@dataclass
class TrainRequest:
    task_type: str
    model: str
    dataset: Any
    output: str
    epochs: float
    batch_size: int | str = "auto"
    learning_rate: float | None = None
    peft: bool | str = "auto"
    eval: bool = True
    push_to_hub: bool = False
    seed: int = 42
    speed: str = "auto"
    explain: bool = False
    dry_run: bool = False
    training_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingPlan:
    task_type: str
    model: str
    model_class: str
    dataset_source: str
    columns: list[str]
    schema_mode: str
    collator: str
    metrics: tuple[str, ...]
    tokenizer_class: str | None
    labels: LabelInfo
    peft: str
    hardware: Hardware
    speed: SpeedPlan
    learning_rate: float
    epochs: float
    eval_enabled: bool
    eval_split: str | None
    why_fast: str
    preprocess: str
    notes: list[str] = field(default_factory=list)
    alignment_example: str | None = None
    estimated_vram_gb: float | None = None
    training_arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    """Return value of `train()` and `EasyTrainer.fit()`."""

    metrics: dict[str, float]
    output_dir: str
    model_id: str
    trainer: Any | None
    plan: TrainingPlan

    def __repr__(self) -> str:
        metric_bits = ", ".join(f"{k}={v:.4f}" for k, v in list(self.metrics.items())[:4])
        metrics = metric_bits or "(none)"
        return (
            f"TrainResult(task={self.plan.task_type!r}, output={self.output_dir!r}, "
            f"metrics={{{metrics}}})"
        )
