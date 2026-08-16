from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "true")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from tests.helpers import write_tiny_bert


@pytest.fixture(scope="session")
def tiny_bert_dir(tmp_path_factory):
    return write_tiny_bert(tmp_path_factory.mktemp("tiny-bert"))
