from __future__ import annotations

from pathlib import Path

import pytest

from easytrain.api import train
from easytrain.errors import ConfigError, UnknownTaskError
from easytrain.tasks import get_task, registered_task_modules


def test_unknown_task_is_rejected():
    with pytest.raises(UnknownTaskError, match="Unknown task type"):
        get_task("not-a-task")


def test_planned_task_mentions_v1():
    with pytest.raises(UnknownTaskError, match="planned but not in v1"):
        get_task("causal-language-modeling")


def test_alias_hint():
    with pytest.raises(UnknownTaskError, match="token-classification"):
        get_task("ner")


def test_required_arg_validation():
    with pytest.raises(ConfigError, match="epochs"):
        train(type="text-classification", model="m", dataset="d", output="o", epochs=0)
    with pytest.raises(ConfigError, match="batch_size"):
        train(
            type="text-classification",
            model="m",
            dataset="d",
            output="o",
            epochs=1,
            batch_size="huge",
        )
    with pytest.raises(ConfigError, match="epochs"):
        train(type="text-classification", model="m", dataset="d", output="o", epochs=float("inf"))
    with pytest.raises(ConfigError, match="batch_size"):
        train(type="text-classification", model="m", dataset="d", output="o", epochs=1, batch_size=2.5)
    with pytest.raises(ConfigError, match="batch_size"):
        train(type="text-classification", model="m", dataset="d", output="o", epochs=1, batch_size=True)
    with pytest.raises(ConfigError, match="speed"):
        train(type="text-classification", model="m", dataset="d", output="o", epochs=1, speed=["max"])


def test_task_plugins_do_not_construct_trainer():
    root = Path(__file__).resolve().parents[1] / "src" / "easytrain" / "tasks"
    forbidden = ("Trainer(", "TrainingArguments", "Seq2SeqTrainer", "SFTTrainer")
    for path in root.rglob("*.py"):
        if path.name in {"__init__.py", "base.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} must not contain {token}; training lives in the core"


def test_both_tasks_are_registered_plugins():
    plugins = registered_task_modules()
    assert set(plugins) == {"text-classification", "token-classification"}
    text = plugins["text-classification"]
    token = plugins["token-classification"]
    assert text.model_class_name != token.model_class_name
    assert text.collator_class_name != token.collator_class_name
    for plugin in (text, token):
        assert callable(plugin.validate_schema)
        assert callable(plugin.infer_labels)
        assert callable(plugin.preprocess)
        assert callable(plugin.compute_metrics)
        assert callable(plugin.get_model_class)
        assert callable(plugin.get_collator)


def test_core_loop_constructs_shared_trainer(tiny_bert_dir, tmp_path, monkeypatch):
    captured: dict = {}

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            captured["data_collator"] = kwargs.get("data_collator")
            captured["compute_metrics"] = kwargs.get("compute_metrics")
            self.model = kwargs["model"]
            self.args = kwargs["args"]
            self.tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
            self.processing_class = kwargs.get("processing_class") or kwargs.get("tokenizer")

        def train(self):
            captured["trained"] = True

        def evaluate(self):
            return {"eval_f1": 1.0, "eval_accuracy": 1.0}

        def save_model(self, *args, **kwargs):
            pass

        def push_to_hub(self, *args, **kwargs):
            pass

    monkeypatch.setattr("transformers.Trainer", FakeTrainer)
    from datasets import Dataset, DatasetDict

    from easytrain.core.loop import run_training
    from easytrain.result import TrainRequest

    data = DatasetDict(
        {
            "train": Dataset.from_dict({"text": ["hello", "bad", "good", "sad"], "label": [1, 0, 1, 0]}),
            "validation": Dataset.from_dict({"text": ["hello", "bad"], "label": [1, 0]}),
        }
    )
    run_training(
        TrainRequest(
            task_type="text-classification",
            model=str(tiny_bert_dir),
            dataset=data,
            output=str(tmp_path / "core-loop"),
            epochs=1,
            eval=True,
            batch_size=2,
        )
    )
    assert captured.get("trained") is True
    assert captured["data_collator"] is not None
    assert captured["data_collator"].__class__.__name__ == "DataCollatorWithPadding"
    assert callable(captured["compute_metrics"])


def test_both_tasks_route_through_run_training(monkeypatch):
    seen: list[str] = []

    def fake_run(request):
        seen.append(request.task_type)
        return type(
            "R",
            (),
            {
                "metrics": {},
                "output_dir": request.output,
                "model_id": "x",
                "trainer": None,
                "plan": None,
            },
        )()

    monkeypatch.setattr("easytrain.api.run_training", fake_run)
    train(type="text-classification", model="m", dataset="d", output="o", epochs=1)
    train(type="token-classification", model="m", dataset="d", output="o", epochs=1)
    assert seen == ["text-classification", "token-classification"]
