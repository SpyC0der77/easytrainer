from __future__ import annotations

from datasets import Dataset, DatasetDict

from easytrain import train


def _dataset():
    texts = [
        "hello world",
        "bad bad",
        "good cat",
        "sad mat",
        "hello cat",
        "bad world",
        "good world",
        "sad cat",
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0]
    train_ds = Dataset.from_dict({"text": texts, "label": labels})
    val_ds = Dataset.from_dict({"text": ["hello", "bad"], "label": [1, 0]})
    return DatasetDict({"train": train_ds, "validation": val_ds})


def test_text_classification_happy_path(tiny_bert_dir, tmp_path):
    output = tmp_path / "clf"
    result = train(
        type="text-classification",
        model=str(tiny_bert_dir),
        dataset=_dataset(),
        output=str(output),
        epochs=1,
        batch_size=2,
        eval=True,
        max_steps=2,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        load_best_model_at_end=False,
    )
    assert result.output_dir == str(output)
    assert result.trainer is not None
    assert result.plan.task_type == "text-classification"
    assert result.plan.model_class == "AutoModelForSequenceClassification"
    assert result.plan.collator == "DataCollatorWithPadding"
    assert (output / "config.json").exists()
    assert (output / "tokenizer.json").exists() or (output / "vocab.txt").exists()
    assert (output / "train_config.json").exists()
    assert (output / "README.md").exists()

    from transformers import pipeline

    clf = pipeline("text-classification", model=str(output), tokenizer=str(output), device=-1)
    out = clf("hello world")
    assert isinstance(out, list)
    assert "label" in out[0]
