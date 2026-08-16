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
    with pytest.raises(SchemaError, match='dataset=\\{"path"'):
        text_classification.validate_schema(ds)


def test_token_classification_requires_tokens_and_ner_tags():
    ds = Dataset.from_dict({"words": [["a"]], "tags": [[0]]})
    with pytest.raises(SchemaError, match="tokens"):
        token_classification.validate_schema(ds)


def test_token_classification_accepts_tokens_ner_tags():
    ds = Dataset.from_dict({"tokens": [["a"]], "ner_tags": [[0]]})
    info = token_classification.validate_schema(ds)
    assert info.mode == "tokens"


def test_token_classification_rejects_length_mismatch_shorter():
    ds = Dataset.from_dict({"tokens": [["a", "b"]], "ner_tags": [["O"]]})
    with pytest.raises(SchemaError, match="Row 0"):
        token_classification.validate_schema(ds)


def test_token_classification_rejects_length_mismatch_longer():
    ds = Dataset.from_dict({"tokens": [["a"]], "ner_tags": [["O", "B-PER"]]})
    with pytest.raises(SchemaError, match="Row 0"):
        token_classification.validate_schema(ds)


def test_token_classification_rejects_unnamed_integer_tags():
    ds = Dataset.from_dict({"tokens": [["a", "b"]], "ner_tags": [[0, 1]]})
    token_classification.validate_schema(ds)
    with pytest.raises(SchemaError, match="named BIO"):
        token_classification.infer_labels(ds)


def test_preprocess_prefers_text_when_pair_columns_also_exist(tiny_bert_dir):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tiny_bert_dir))
    ds = Dataset.from_dict(
        {
            "text": ["hello"],
            "sentence1": ["bad"],
            "sentence2": ["world"],
            "label": [0],
        }
    )
    info = text_classification.validate_schema(ds)
    assert info.mode == "single"
    labels = text_classification.infer_labels(ds)
    out = text_classification.preprocess(ds, tokenizer, labels, 32)
    single = tokenizer("hello", truncation=True, max_length=32)
    pair = tokenizer("bad", "world", truncation=True, max_length=32)
    assert list(out[0]["input_ids"]) == list(single["input_ids"])
    assert list(out[0]["input_ids"]) != list(pair["input_ids"])
