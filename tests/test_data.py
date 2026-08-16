from __future__ import annotations

from datasets import Dataset, DatasetDict

from easytrain.core.data import load_dataset_spec
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
