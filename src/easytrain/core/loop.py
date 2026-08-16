from __future__ import annotations

from pathlib import Path
from typing import Any

from easytrain.core.arguments import build_training_arguments, training_arguments_as_dict
from easytrain.core.compat import trainer_tokenizer_kwarg
from easytrain.core.data import DatasetBundle, ensure_eval_split, load_dataset_spec
from easytrain.core.explain import format_plan
from easytrain.core.save import maybe_push_to_hub, save_trained, write_educational_artifacts
from easytrain.core.speed import (
    apply_runtime_flags,
    detect_hardware,
    estimate_params,
    maybe_compile,
    plan_speed,
)
from easytrain.errors import ConfigError
from easytrain.result import TrainRequest, TrainResult, TrainingPlan
from easytrain.tasks import get_task


def _load_tokenizer(model: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"
    return tokenizer


def _load_config(model: str):
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(model)
    except Exception:
        return None


def _max_length(tokenizer: Any) -> int:
    raw = getattr(tokenizer, "model_max_length", 512) or 512
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 512
    if value > 10_000:
        return 512
    return max(8, min(value, 512))


def _load_model(task, model_name: str, labels, speed):
    cls = task.get_model_class()
    kwargs: dict[str, Any] = {
        "num_labels": labels.num_labels,
        "id2label": labels.id2label,
        "label2id": labels.label2id,
        "ignore_mismatched_sizes": True,
    }
    if task.type == "text-classification":
        kwargs["problem_type"] = "single_label_classification"
    if speed.torch_dtype is not None:
        kwargs["torch_dtype"] = speed.torch_dtype
    if speed.attn_implementation:
        kwargs["attn_implementation"] = speed.attn_implementation
    try:
        return cls.from_pretrained(model_name, **kwargs)
    except (TypeError, ValueError, OSError):
        kwargs.pop("attn_implementation", None)
        kwargs.pop("torch_dtype", None)
        return cls.from_pretrained(model_name, **kwargs)


def _apply_peft(model, task, peft_name: str):
    if peft_name in {"none", "", None}:
        return model
    from peft import LoraConfig, TaskType, get_peft_model

    task_type = getattr(TaskType, task.peft_task_type)
    config = LoraConfig(
        task_type=task_type,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
    )
    if peft_name == "qlora":
        raise ConfigError(
            "QLoRA is reserved for large LLMs. v1 encoder tasks use full fine-tune or LoRA. "
            "Pass peft='lora' or peft='auto'."
        )
    return get_peft_model(model, config)


def _map_splits(bundle: DatasetBundle, fn) -> DatasetBundle:
    return DatasetBundle(
        train=fn(bundle.train),
        validation=fn(bundle.validation) if bundle.validation is not None else None,
        test=fn(bundle.test) if bundle.test is not None else None,
        source=bundle.source,
        mapping=bundle.mapping,
    )


def _build_plan(
    request: TrainRequest,
    task,
    bundle: DatasetBundle,
    schema,
    labels,
    tokenizer,
    hardware,
    speed,
    learning_rate: float,
    training_args_dict: dict[str, Any],
) -> TrainingPlan:
    alignment = None
    if request.task_type == "token-classification" and tokenizer is not None:
        try:
            alignment = task.format_alignment_example(
                bundle.train[0],
                tokenizer,
                labels,
                _max_length(tokenizer),
            )
        except Exception as exc:
            alignment = f"(could not render alignment example: {exc})"

    eval_split = None
    if bundle.validation is not None:
        eval_split = "validation"
    elif request.eval:
        eval_split = "none (train too small to hold out)"

    peft_label = speed.peft
    if peft_label == "none":
        peft_label = "none (full fine-tune; encoder < 1B or peft=auto)"

    notes = list(schema.notes) + list(task.explain_notes(labels))
    if bundle.mapping:
        notes.append(f"Applied column mapping: {bundle.mapping}")

    tokenizer_class = type(tokenizer).__name__ if tokenizer is not None else None
    return TrainingPlan(
        task_type=request.task_type,
        model=request.model,
        model_class=task.model_class_name,
        dataset_source=bundle.source,
        columns=list(bundle.train.column_names),
        schema_mode=schema.mode,
        collator=task.collator_class_name,
        metrics=task.metrics_names,
        tokenizer_class=tokenizer_class,
        labels=labels,
        peft=peft_label,
        hardware=hardware,
        speed=speed,
        learning_rate=learning_rate,
        epochs=request.epochs,
        eval_enabled=bool(request.eval and bundle.validation is not None),
        eval_split=eval_split,
        why_fast=speed.why_fast,
        preprocess=task.preprocess_summary,
        notes=notes,
        alignment_example=alignment,
        estimated_vram_gb=speed.estimated_vram_gb,
        training_arguments=training_args_dict,
    )


def run_training(request: TrainRequest) -> TrainResult:
    """Shared Trainer core used by every v1 task plugin."""
    from transformers import Trainer, set_seed

    task = get_task(request.task_type)
    set_seed(request.seed)

    bundle = load_dataset_spec(request.dataset)
    schema = task.validate_schema(bundle.train)
    labels = task.infer_labels(bundle.train)
    bundle = ensure_eval_split(
        bundle,
        enabled=request.eval,
        seed=request.seed,
        stratify_column=getattr(task, "stratify_column", None),
    )

    hardware = detect_hardware()
    tokenizer = _load_tokenizer(request.model)
    config = _load_config(request.model)
    n_params = estimate_params(config)
    speed = plan_speed(
        hardware,
        speed=request.speed,
        batch_size=request.batch_size,
        peft=request.peft,
        n_params=n_params,
        hidden_size=getattr(config, "hidden_size", None) or getattr(config, "dim", None),
        num_layers=getattr(config, "num_hidden_layers", None) or getattr(config, "n_layers", None),
    )
    learning_rate = (
        request.learning_rate if request.learning_rate is not None else task.default_learning_rate
    )
    dummy_args = build_training_arguments(
        request,
        speed,
        hardware,
        has_eval=bundle.validation is not None,
        learning_rate=learning_rate,
        metric_for_best_model=task.metric_for_best_model,
    )
    plan = _build_plan(
        request,
        task,
        bundle,
        schema,
        labels,
        tokenizer,
        hardware,
        speed,
        learning_rate,
        training_arguments_as_dict(dummy_args),
    )

    if request.explain or request.dry_run:
        print(format_plan(plan))
        write_educational_artifacts(request.output, plan)

    if request.dry_run:
        return TrainResult(
            metrics={},
            output_dir=request.output,
            model_id=request.output,
            trainer=None,
            plan=plan,
        )

    apply_runtime_flags(speed, hardware)
    model = _load_model(task, request.model, labels, speed)
    model = _apply_peft(model, task, speed.peft)
    model, compile_note = maybe_compile(model, speed)
    if compile_note:
        print(compile_note)
        plan.notes.append(compile_note)

    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    max_length = _max_length(tokenizer)
    tokenized = _map_splits(
        bundle,
        lambda split: task.preprocess(split, tokenizer, labels, max_length),
    )

    args = dummy_args
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized.train,
        eval_dataset=tokenized.validation if plan.eval_enabled else None,
        data_collator=task.get_collator(tokenizer),
        compute_metrics=task.compute_metrics(labels) if plan.eval_enabled else None,
        **trainer_tokenizer_kwarg(tokenizer),
    )
    trainer.train()

    metrics: dict[str, float] = {}
    if plan.eval_enabled:
        raw = trainer.evaluate()
        metrics = {key: float(value) for key, value in raw.items() if isinstance(value, (int, float))}

    save_trained(
        trainer=trainer,
        tokenizer=tokenizer,
        request=request,
        plan=plan,
        metrics=metrics,
    )
    maybe_push_to_hub(request, trainer)

    model_id = Path(request.output).name
    return TrainResult(
        metrics=metrics,
        output_dir=request.output,
        model_id=model_id,
        trainer=trainer,
        plan=plan,
    )
