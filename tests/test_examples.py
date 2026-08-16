from __future__ import annotations

import py_compile
from pathlib import Path

from easytrain.core.data import load_dataset_spec
from easytrain.tasks import text_classification, token_classification

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_example_scripts_compile():
    scripts = sorted(EXAMPLES.glob("*.py"))
    assert scripts
    for path in scripts:
        py_compile.compile(str(path), doraise=True)


def test_example_review_csv_matches_text_schema():
    bundle = load_dataset_spec(str(EXAMPLES / "data" / "reviews.csv"))
    info = text_classification.validate_schema(bundle.train)
    assert info.mode == "single"
    labels = text_classification.infer_labels(bundle.train)
    assert set(labels.names) == {"negative", "positive"}


def test_example_raw_reviews_mapping():
    bundle = load_dataset_spec(
        {
            "path": str(EXAMPLES / "data" / "reviews_raw.csv"),
            "text": "review",
            "label": "sentiment",
        }
    )
    info = text_classification.validate_schema(bundle.train)
    assert info.mode == "single"
    assert "text" in bundle.train.column_names
    assert "label" in bundle.train.column_names


def test_example_pairs_csv_matches_pair_schema():
    bundle = load_dataset_spec(str(EXAMPLES / "data" / "nli_pairs.csv"))
    info = text_classification.validate_schema(bundle.train)
    assert info.mode == "pair"


def test_example_splits_use_dev_as_validation():
    bundle = load_dataset_spec(str(EXAMPLES / "data" / "splits"))
    assert bundle.validation is not None
    assert len(bundle.train) == 12
    assert len(bundle.validation) == 4


def test_example_ner_jsonl_matches_token_schema():
    bundle = load_dataset_spec(str(EXAMPLES / "data" / "ner.jsonl"))
    info = token_classification.validate_schema(bundle.train)
    assert info.mode == "tokens"
    labels = token_classification.infer_labels(bundle.train)
    assert "O" in labels.names
    assert "B-PER" in labels.names


def test_example_raw_ner_mapping():
    bundle = load_dataset_spec(
        {
            "path": str(EXAMPLES / "data" / "ner_raw.jsonl"),
            "tokens": "words",
            "ner_tags": "tags",
        }
    )
    info = token_classification.validate_schema(bundle.train)
    assert info.mode == "tokens"
    labels = token_classification.infer_labels(bundle.train)
    assert "B-PER" in labels.names
    assert "I-PER" in labels.names
