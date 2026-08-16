from __future__ import annotations

from typing import Any

from easytrain.core.loop import run_training
from easytrain.errors import ConfigError
from easytrain.result import TrainRequest, TrainResult


def train(
    type: str,
    model: str,
    dataset: Any,
    output: str,
    epochs: float,
    batch_size: int | str = "auto",
    learning_rate: float | None = None,
    peft: bool | str = "auto",
    eval: bool = True,
    push_to_hub: bool = False,
    seed: int = 42,
    speed: str = "auto",
    explain: bool = False,
    dry_run: bool = False,
    **training_args: Any,
) -> TrainResult:
    """Train a model for a task. EasyTrain selects the Hugging Face stack.

    Required: type, model, dataset, output, epochs.
    Optional: batch_size, learning_rate, peft, eval, push_to_hub, seed, speed.
    Escape hatches: explain, dry_run, and **training_args (forwarded to TrainingArguments).
    """
    request = _build_request(
        type=type,
        model=model,
        dataset=dataset,
        output=output,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        peft=peft,
        eval=eval,
        push_to_hub=push_to_hub,
        seed=seed,
        speed=speed,
        explain=explain,
        dry_run=dry_run,
        training_args=training_args,
    )
    return run_training(request)


def _build_request(
    *,
    type: str,
    model: str,
    dataset: Any,
    output: str,
    epochs: float,
    batch_size: int | str,
    learning_rate: float | None,
    peft: bool | str,
    eval: bool,
    push_to_hub: bool,
    seed: int,
    speed: str,
    explain: bool,
    dry_run: bool,
    training_args: dict[str, Any],
) -> TrainRequest:
    if not type or not isinstance(type, str):
        raise ConfigError("type is required, e.g. type='text-classification'.")
    if not model or not isinstance(model, str):
        raise ConfigError("model is required, e.g. model='distilbert/distilbert-base-uncased'.")
    if dataset is None or dataset == "":
        raise ConfigError(
            "dataset is required. Pass a Hub id, local CSV/JSON path, "
            "datasets.Dataset, or a mapping dict."
        )
    if not output or not isinstance(output, str):
        raise ConfigError("output is required (directory to save the model).")
    try:
        epochs_value = float(epochs)
    except (TypeError, ValueError) as exc:
        raise ConfigError("epochs must be a positive number.") from exc
    if epochs_value <= 0:
        raise ConfigError("epochs must be a positive number.")
    if isinstance(batch_size, str) and batch_size != "auto":
        raise ConfigError("batch_size must be an int or 'auto'.")
    if speed not in {"auto", "stable", "max"}:
        raise ConfigError("speed must be 'auto', 'stable', or 'max'.")
    if peft not in {"auto", "lora", "qlora", "none", "full", True, False}:
        raise ConfigError("peft must be 'auto', 'lora', 'qlora', 'none', True, or False.")
    return TrainRequest(
        task_type=type,
        model=model,
        dataset=dataset,
        output=output,
        epochs=epochs_value,
        batch_size=batch_size,
        learning_rate=learning_rate,
        peft=peft,
        eval=eval,
        push_to_hub=push_to_hub,
        seed=seed,
        speed=speed,
        explain=explain or dry_run,
        dry_run=dry_run,
        training_args=training_args,
    )


class EasyTrainer:
    """Sklearn-style wrapper around the same `train()` router."""

    def __init__(self, **kwargs: Any) -> None:
        self.params = kwargs
        self.result: TrainResult | None = None

    def fit(self) -> EasyTrainer:
        self.result = train(**self.params)
        return self

    def evaluate(self) -> dict[str, float]:
        if self.result is None:
            raise ConfigError("Call fit() before evaluate().")
        if self.result.trainer is None:
            return dict(self.result.metrics)
        metrics = self.result.trainer.evaluate()
        numeric = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        self.result.metrics.update(numeric)
        return numeric

    def save(self, path: str | None = None) -> str:
        if self.result is None:
            raise ConfigError("Call fit() before save().")
        output = path or self.result.output_dir
        if self.result.trainer is not None:
            self.result.trainer.save_model(output)
        return output
