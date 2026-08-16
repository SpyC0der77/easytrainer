from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict

from easytrain.core.data import ensure_eval_split, load_dataset_spec
from easytrain.errors import ConfigError, SchemaError
from easytrain.tasks import text_classification


def test_load_in_memory_dataset():
    ds = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})
    bundle = load_dataset_spec(ds)
    assert len(bundle.train) == 2
    assert bundle.validation is None


def test_load_dataset_dict_splits():
    data = DatasetDict(
        {
            "train": Dataset.from_dict({"text": ["a"], "label": [0]}),
            "validation": Dataset.from_dict({"text": ["b"], "label": [1]}),
            "test": Dataset.from_dict({"text": ["c"], "label": [0]}),
        }
    )
    bundle = load_dataset_spec(data)
    assert bundle.validation is not None
    assert bundle.test is not None


def test_column_mapping_dict(tmp_path):
    csv_path = tmp_path / "reviews.csv"
    csv_path.write_text("review,sentiment\nhello,0\nworld,1\n", encoding="utf-8")
    bundle = load_dataset_spec({"path": str(csv_path), "text": "review", "label": "sentiment"})
    assert "text" in bundle.train.column_names
    assert "label" in bundle.train.column_names
    schema = text_classification.validate_schema(bundle.train)
    assert schema.mode == "single"


def test_local_csv(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("text,label\nhi,0\nbye,1\n", encoding="utf-8")
    bundle = load_dataset_spec(str(csv_path))
    assert list(bundle.train["text"]) == ["hi", "bye"]


def test_local_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"text": "hi", "label": 0}\n{"text": "bye", "label": 1}\n', encoding="utf-8")
    bundle = load_dataset_spec(str(path))
    assert "text" in bundle.train.column_names
    assert len(bundle.train) == 2


def test_split_directory(tmp_path):
    (tmp_path / "train.csv").write_text("text,label\na,0\n", encoding="utf-8")
    (tmp_path / "validation.csv").write_text("text,label\nb,1\n", encoding="utf-8")
    bundle = load_dataset_spec(str(tmp_path))
    assert bundle.validation is not None
    assert bundle.train["text"][0] == "a"
    assert bundle.validation["text"][0] == "b"


def test_mapping_collision_raises(tmp_path):
    csv_path = tmp_path / "both.csv"
    csv_path.write_text("text,review,label\na,b,0\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="already has a 'text' column"):
        load_dataset_spec({"path": str(csv_path), "text": "review", "label": "label"})


def test_split_selects_local_directory_split(tmp_path):
    (tmp_path / "train.csv").write_text("text,label\ntrain-row,0\n", encoding="utf-8")
    (tmp_path / "validation.csv").write_text("text,label\nval-row,1\n", encoding="utf-8")
    bundle = load_dataset_spec({"path": str(tmp_path), "split": "validation"})
    assert list(bundle.train["text"]) == ["val-row"]
    assert bundle.validation is None


def test_unknown_mapping_key_is_rejected(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("text,label\na,0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown dataset mapping"):
        load_dataset_spec({"path": str(csv_path), "foo": "bar"})


def test_int_csv_labels_can_stratify():
    labels = [0, 1] * 10
    texts = [f"row-{i}" for i in range(20)]
    ds = Dataset.from_dict({"text": texts, "label": labels})
    from easytrain.core.data import DatasetBundle

    bundle = DatasetBundle(train=ds, validation=None, test=None, source="mem", mapping={})
    split = ensure_eval_split(bundle, enabled=True, seed=0, stratify_column="label")
    assert split.validation is not None
    assert set(split.validation["label"]) == {0, 1}
