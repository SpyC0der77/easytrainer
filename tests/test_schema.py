from __future__ import annotations

import pytest
from datasets import Dataset

from easytrain.errors import SchemaError
from easytrain.tasks import text_classification, token_classification


def test_text_classification_requires_text_and_label():
    ds = Dataset.from_dict({"review": ["a"], "sentiment": [0]})
    with pytest.raises(SchemaError, match="text, label"):
        text_classification.validate_schema(ds)


def test_text_classification_accepts_text_label():
    ds = Dataset.from_dict({"text": ["a"], "label": [0]})
    info = text_classification.validate_schema(ds)
    assert info.mode == "single"


def test_text_classification_accepts_sentence_pair():
    ds = Dataset.from_dict({"sentence1": ["a"], "sentence2": ["b"], "label": [0]})
    info = text_classification.validate_schema(ds)
    assert info.mode == "pair"


def test_text_classification_error_includes_mapping_example():
    ds = Dataset.from_dict({"review": ["a"], "sentiment": [0]})
    with pytest.raises(SchemaError, match="dataset=\\{\"path\""):
        text_classification.validate_schema(ds)


def test_token_classification_requires_tokens_and_ner_tags():
    ds = Dataset.from_dict({"words": [["a"]], "tags": [[0]]})
    with pytest.raises(SchemaError, match="tokens"):
        token_classification.validate_schema(ds)


def test_token_classification_accepts_tokens_ner_tags():
    ds = Dataset.from_dict({"tokens": [["a"]], "ner_tags": [[0]]})
    info = token_classification.validate_schema(ds)
    assert info.mode == "tokens"


def test_token_classification_error_includes_mapping_example():
    ds = Dataset.from_dict({"words": [["a"]]})
    with pytest.raises(SchemaError, match="ner.jsonl"):
        token_classification.validate_schema(ds)
