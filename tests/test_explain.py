from __future__ import annotations

from datasets import Dataset

from easytrain import train


def _text_ds():
    return Dataset.from_dict(
        {
            "text": ["hello world", "bad bad", "good cat", "sad mat", "hello cat", "bad world"],
            "label": [1, 0, 1, 0, 1, 0],
        }
    )


def test_dry_run_writes_plan_without_training(tiny_bert_dir, tmp_path, capsys):
    output = tmp_path / "dry"
    result = train(
        type="text-classification",
        model=str(tiny_bert_dir),
        dataset=_text_ds(),
        output=str(output),
        epochs=1,
        dry_run=True,
        batch_size=2,
    )
    captured = capsys.readouterr().out
    assert "EasyTrain plan" in captured
    assert "AutoModelForSequenceClassification" in captured
    assert "DataCollatorWithPadding" in captured
    assert "Why it's fast" in captured
    assert result.trainer is None
    assert (output / "train_config.json").exists()
    assert (output / "train_snippet.py").exists()
    assert not (output / "model.safetensors").exists()
    assert not (output / "pytorch_model.bin").exists()


def test_token_dry_run_documents_alignment(tiny_bert_dir, tmp_path, capsys):
    ds = Dataset.from_dict(
        {
            "tokens": [["john", "lives", "in", "paris"], ["hello", "world"]],
            "ner_tags": [["B-PER", "O", "O", "B-LOC"], ["O", "O"]],
        }
    )
    result = train(
        type="token-classification",
        model=str(tiny_bert_dir),
        dataset=ds,
        output=str(tmp_path / "ner-dry"),
        epochs=1,
        dry_run=True,
        eval=False,
    )
    captured = capsys.readouterr().out
    assert "word_ids" in captured
    assert "-100" in captured
    assert "seqeval" in captured.lower() or "entity-level" in captured
    assert result.plan.alignment_example is not None
    assert "AutoModelForTokenClassification" in captured


def test_easy_trainer_dry_run(tiny_bert_dir, tmp_path):
    from easytrain import EasyTrainer

    trainer = EasyTrainer(
        type="text-classification",
        model=str(tiny_bert_dir),
        dataset=_text_ds(),
        output=str(tmp_path / "sklearn"),
        epochs=1,
        dry_run=True,
        eval=False,
    ).fit()
    assert trainer.result is not None
    assert trainer.result.trainer is None
