from __future__ import annotations

from datasets import Dataset, DatasetDict

from easytrain import train


def _dataset():
    train_ds = Dataset.from_dict(
        {
            "tokens": [
                ["john", "lives", "in", "paris"],
                ["hello", "world"],
                ["the", "cat", "sat"],
                ["love", "paris"],
                ["john", "love", "york"],
                ["good", "bad"],
                ["new", "york"],
                ["lives", "in", "york"],
            ],
            "ner_tags": [
                ["B-PER", "O", "O", "B-LOC"],
                ["O", "O"],
                ["O", "O", "O"],
                ["O", "B-LOC"],
                ["B-PER", "O", "B-LOC"],
                ["O", "O"],
                ["B-LOC", "I-LOC"],
                ["O", "O", "B-LOC"],
            ],
        }
    )
    val_ds = Dataset.from_dict(
        {
            "tokens": [["john", "in", "paris"], ["hello", "cat"]],
            "ner_tags": [["B-PER", "O", "B-LOC"], ["O", "O"]],
        }
    )
    return DatasetDict({"train": train_ds, "validation": val_ds})


def test_token_classification_happy_path(tiny_bert_dir, tmp_path):
    output = tmp_path / "ner"
    result = train(
        type="token-classification",
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
    assert result.plan.task_type == "token-classification"
    assert result.plan.model_class == "AutoModelForTokenClassification"
    assert result.plan.collator == "DataCollatorForTokenClassification"
    assert result.plan.alignment_example is not None
    assert (output / "config.json").exists()
    assert (output / "train_snippet.py").exists()

    from transformers import pipeline

    ner = pipeline("token-classification", model=str(output), tokenizer=str(output), device=-1)
    out = ner("john lives in paris")
    assert isinstance(out, list)
