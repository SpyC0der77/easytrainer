from __future__ import annotations

import pytest
from datasets import ClassLabel, Dataset, Features, Sequence, Value

from easytrain.core.labels import infer_label_info
from easytrain.errors import SchemaError
from easytrain.tasks.token_classification import align_labels_with_tokens


def test_align_specials_and_continuations_are_ignored():
    # words: 0="paris" split into two subwords, plus CLS/SEP
    labels = [1]
    word_ids = [None, 0, 0, None]
    aligned = align_labels_with_tokens(labels, word_ids)
    assert aligned == [-100, 1, -100, -100]


def test_align_first_subword_keeps_label():
    labels = [0, 1, 2]
    word_ids = [None, 0, 1, 2, 2, None]
    assert align_labels_with_tokens(labels, word_ids) == [-100, 0, 1, 2, -100, -100]


def test_infer_classlabel_feature():
    features = Features({"label": ClassLabel(names=["neg", "pos"]), "text": Value("string")})
    ds = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]}, features=features)
    info = infer_label_info(ds, "label", kind="class")
    assert info.num_labels == 2
    assert info.id2label[0] == "neg"
    assert info.label2id["pos"] == 1


def test_infer_string_labels_sorted():
    ds = Dataset.from_dict({"label": ["pos", "neg", "pos"]})
    info = infer_label_info(ds, "label", kind="class")
    assert info.names == ["neg", "pos"]


def test_infer_bio_order():
    features = Features(
        {
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=["O", "B-LOC", "I-LOC"])),
        }
    )
    ds = Dataset.from_dict(
        {"tokens": [["I", "Paris"]], "ner_tags": [[0, 1]]},
        features=features,
    )
    info = infer_label_info(ds, "ner_tags", kind="bio")
    assert info.names == ["O", "B-LOC", "I-LOC"]
    assert info.num_labels == 3


def test_infer_bio_from_strings():
    ds = Dataset.from_dict({"ner_tags": [["O", "B-PER", "I-PER"], ["B-LOC", "O"]]})
    info = infer_label_info(ds, "ner_tags", kind="bio")
    assert info.names == ["O", "B-LOC", "B-PER", "I-PER"]
    assert info.label2id["B-PER"] < info.label2id["I-PER"]


def test_mixed_int_and_string_labels_are_rejected():
    class FakeDataset:
        column_names = ["label"]
        features = {}

        def __getitem__(self, key):
            return [1, "1"]

    with pytest.raises(SchemaError, match="Mixed"):
        infer_label_info(FakeDataset(), "label", kind="class")


def test_unhashable_labels_fall_back_to_string_key():
    from easytrain.core.labels import _unique_values

    class FakeDataset:
        def __getitem__(self, key):
            return [{"a": 1}, {"a": 1}, {"b": 2}]

    assert _unique_values(FakeDataset(), "label") == [{"a": 1}, {"b": 2}]
