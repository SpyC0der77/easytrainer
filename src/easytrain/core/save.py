from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from easytrain.constants import __version__
from easytrain.core.explain import plan_to_config, trainer_snippet
from easytrain.result import TrainRequest, TrainingPlan


def write_educational_artifacts(output_dir: str | Path, plan: TrainingPlan) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    config_path = path / "train_config.json"
    snippet_path = path / "train_snippet.py"
    config_path.write_text(json.dumps(plan_to_config(plan), indent=2, default=str) + "\n", encoding="utf-8")
    snippet_path.write_text(trainer_snippet(plan), encoding="utf-8")


def write_model_card(
    output_dir: str | Path,
    request: TrainRequest,
    plan: TrainingPlan,
    metrics: dict[str, Any],
) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    card_path = path / "README.md"
    metric_lines = "\n".join(f"- {key}: {value}" for key, value in metrics.items()) or "- (not evaluated)"
    tags = [plan.task_type, "easytrain", "generated_from_easytrain"]
    yaml = "\n".join(
        [
            "---",
            "library_name: easytrain",
            f"base_model: {plan.model}",
            f"pipeline_tag: {plan.task_type}",
            "tags:",
            *[f"- {tag}" for tag in tags],
            "---",
            "",
        ]
    )
    body = f"""# {Path(request.output).name}

Fine-tuned with [EasyTrain](https://github.com/SpyC0der77/easytrainer) `{__version__}`.

EasyTrain is `transformers.pipeline` for training: it selected the Hugging Face
stack below. The concepts (dataset, loss, epochs, batches, learning rate,
backprop, fine-tuning, evaluation) are unchanged.

## Task

- **type:** `{plan.task_type}`
- **base model:** `{plan.model}`
- **model class:** `{plan.model_class}`
- **dataset:** `{plan.dataset_source}`
- **columns:** {", ".join(plan.columns)}
- **labels:** {plan.labels.num_labels} `{plan.labels.names}`

## Hyperparameters

- epochs: {plan.epochs}
- learning rate: {plan.learning_rate}
- batch size: {plan.speed.batch_size}
- precision: {plan.speed.precision}
- attention: {plan.speed.attn_implementation}
- optimizer: {plan.speed.optim}
- PEFT: {plan.peft}
- seed: {request.seed}

## Metrics

{metric_lines}

## Why it's fast

{plan.why_fast}

## Graduate out of EasyTrain

See `train_config.json` and `train_snippet.py` in this directory for the
equivalent `Trainer` script.
"""
    card_path.write_text(yaml + body, encoding="utf-8")
    return card_path


def save_trained(
    *,
    trainer: Any,
    tokenizer: Any,
    request: TrainRequest,
    plan: TrainingPlan,
    metrics: dict[str, Any],
) -> None:
    output = Path(request.output)
    output.mkdir(parents=True, exist_ok=True)
    model = trainer.model
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
        model.save_pretrained(output)
    else:
        trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    write_educational_artifacts(output, plan)
    write_model_card(output, request, plan, metrics)


def maybe_push_to_hub(request: TrainRequest, trainer: Any) -> None:
    if not request.push_to_hub:
        return
    trainer.push_to_hub()
