from __future__ import annotations

from typing import Any

from easytrain.core.compat import filter_training_arguments
from easytrain.core.speed import Hardware, SpeedPlan
from easytrain.errors import ConfigError
from easytrain.result import TrainRequest


def build_training_arguments(
    request: TrainRequest,
    speed: SpeedPlan,
    hardware: Hardware,
    *,
    has_eval: bool,
    learning_rate: float,
    metric_for_best_model: str,
) -> Any:
    from transformers import TrainingArguments

    eval_on = bool(request.eval and has_eval)
    defaults: dict[str, Any] = {
        "output_dir": request.output,
        "num_train_epochs": request.epochs,
        "per_device_train_batch_size": speed.batch_size,
        "per_device_eval_batch_size": speed.batch_size,
        "learning_rate": learning_rate,
        "seed": request.seed,
        "bf16": speed.precision == "bf16",
        "fp16": speed.precision == "fp16",
        "tf32": speed.tf32,
        "optim": speed.optim,
        "dataloader_pin_memory": speed.pin_memory,
        "dataloader_num_workers": speed.num_workers,
        "gradient_checkpointing": speed.gradient_checkpointing,
        "eval_strategy": "epoch" if eval_on else "no",
        "save_strategy": "epoch",
        "load_best_model_at_end": eval_on,
        "greater_is_better": True,
        "logging_strategy": "steps",
        "logging_steps": 10,
        "report_to": "none",
        "save_total_limit": 2,
        "warmup_ratio": 0.06,
        "lr_scheduler_type": "linear",
        "remove_unused_columns": True,
        "use_cpu": hardware.device == "cpu",
        "no_cuda": hardware.device == "cpu",
    }
    if eval_on:
        defaults["metric_for_best_model"] = metric_for_best_model
    if hardware.device == "cpu":
        defaults["bf16"] = False
        defaults["fp16"] = False

    merged = {**defaults, **request.training_args}
    merged["output_dir"] = request.output
    merged["num_train_epochs"] = request.epochs
    merged["seed"] = request.seed
    if request.learning_rate is not None:
        merged["learning_rate"] = request.learning_rate
    if isinstance(request.batch_size, int):
        merged["per_device_train_batch_size"] = request.batch_size
        merged["per_device_eval_batch_size"] = request.batch_size

    if merged.get("load_best_model_at_end") and merged.get("metric_for_best_model") in {None, ""}:
        merged["load_best_model_at_end"] = False
    eval_strategy = merged.get("eval_strategy", merged.get("evaluation_strategy"))
    if merged.get("load_best_model_at_end") and str(eval_strategy) == "no":
        merged["load_best_model_at_end"] = False
    if merged.get("load_best_model_at_end"):
        save_strategy = merged.get("save_strategy")
        if eval_strategy and save_strategy and eval_strategy != save_strategy:
            merged["save_strategy"] = eval_strategy

    filtered = filter_training_arguments(merged)
    filtered.pop("self", None)
    try:
        return TrainingArguments(**filtered)
    except (ValueError, TypeError) as exc:
        if filtered.get("optim") == "adamw_torch_fused":
            filtered["optim"] = "adamw_torch"
            try:
                return TrainingArguments(**filtered)
            except Exception:
                pass
        raise ConfigError(f"Invalid TrainingArguments: {exc}") from exc


def training_arguments_as_dict(args: Any) -> dict[str, Any]:
    keys = (
        "output_dir",
        "num_train_epochs",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "learning_rate",
        "seed",
        "bf16",
        "fp16",
        "tf32",
        "optim",
        "dataloader_pin_memory",
        "dataloader_num_workers",
        "gradient_checkpointing",
        "eval_strategy",
        "evaluation_strategy",
        "save_strategy",
        "load_best_model_at_end",
        "metric_for_best_model",
        "warmup_ratio",
        "lr_scheduler_type",
        "report_to",
    )
    out = {}
    for key in keys:
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                out[key] = value
    return out
