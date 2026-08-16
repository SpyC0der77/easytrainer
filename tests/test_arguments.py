from __future__ import annotations

import pytest

from easytrain.api import EasyTrainer
from easytrain.core.arguments import build_training_arguments, training_arguments_as_dict
from easytrain.core.speed import Hardware, plan_speed
from easytrain.errors import ConfigError
from easytrain.result import TrainRequest, TrainResult


def _args(tmp_path, **training_args):
    hardware = Hardware(device="cpu", name="CPU", capability=None, vram_bytes=None)
    speed = plan_speed(hardware)
    request = TrainRequest(
        task_type="text-classification",
        model="m",
        dataset="d",
        output=str(tmp_path / "out"),
        epochs=1,
        eval=True,
        training_args=training_args,
    )
    return build_training_arguments(
        request,
        speed,
        hardware,
        has_eval=True,
        learning_rate=2e-5,
        metric_for_best_model="eval_f1",
    )


def test_unknown_training_arg_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="Unsupported TrainingArguments"):
        _args(tmp_path, not_a_real_flag=1)


def test_evaluation_strategy_alias_overrides_default(tmp_path):
    args = _args(tmp_path, evaluation_strategy="no")
    strategy = getattr(args, "eval_strategy", None) or args.evaluation_strategy
    assert "no" in str(strategy).lower()


def test_loss_metric_sets_greater_is_better_false(tmp_path):
    args = _args(tmp_path, metric_for_best_model="eval_loss")
    assert args.greater_is_better is False


def test_explicit_greater_is_better_is_preserved(tmp_path):
    args = _args(tmp_path, metric_for_best_model="eval_loss", greater_is_better=True)
    assert args.greater_is_better is True


def test_training_arguments_as_dict_uses_plain_values(tmp_path):
    args = _args(tmp_path)
    plain = training_arguments_as_dict(args)
    strategy = plain.get("eval_strategy", plain.get("evaluation_strategy"))
    assert isinstance(strategy, str)


def test_easy_trainer_evaluate_skips_when_eval_disabled():
    boom = type("T", (), {"evaluate": lambda self: (_ for _ in ()).throw(AssertionError("evaluate"))})()
    wrapper = EasyTrainer(type="text-classification", model="m", dataset="d", output="o", epochs=1)
    wrapper.result = TrainResult(
        metrics={"kept": 1.0},
        output_dir="o",
        model_id="o",
        trainer=boom,
        plan=type("P", (), {"eval_enabled": False})(),
    )
    assert wrapper.evaluate() == {"kept": 1.0}
