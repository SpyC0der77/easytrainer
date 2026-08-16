"""Map non-canonical JSONL columns (words, tags) onto tokens + ner_tags."""

from pathlib import Path

from transformers import pipeline

from easytrain import train

ROOT = Path(__file__).resolve().parent
MODEL = "distilbert/distilbert-base-uncased"
OUTPUT = ROOT / "output" / "token-classification-mapping"


def main() -> None:
    result = train(
        type="token-classification",
        model=MODEL,
        dataset={
            "path": str(ROOT / "data" / "ner_raw.jsonl"),
            "tokens": "words",
            "ner_tags": "tags",
        },
        output=str(OUTPUT),
        epochs=1,
        batch_size=4,
    )
    print(result.metrics)

    ner = pipeline("token-classification", model=str(OUTPUT), tokenizer=str(OUTPUT), aggregation_strategy="simple")
    print(ner("Ada Lovelace lived in London."))


if __name__ == "__main__":
    main()
