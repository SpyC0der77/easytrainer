"""Named-entity recognition from a JSONL file (columns: tokens, ner_tags)."""

from pathlib import Path

from transformers import pipeline

from easytrain import train

ROOT = Path(__file__).resolve().parent
MODEL = "distilbert/distilbert-base-uncased"
OUTPUT = ROOT / "output" / "token-classification"


def main() -> None:
    result = train(
        type="token-classification",
        model=MODEL,
        dataset=str(ROOT / "data" / "ner.jsonl"),
        output=str(OUTPUT),
        epochs=1,
        batch_size=4,
    )
    print(result.metrics)

    ner = pipeline("token-classification", model=str(OUTPUT), tokenizer=str(OUTPUT), aggregation_strategy="simple")
    print(ner("Alice lives in Paris."))


if __name__ == "__main__":
    main()
