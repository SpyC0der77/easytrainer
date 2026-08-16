"""Classify movie reviews from a local CSV (columns: text, label)."""

from pathlib import Path

from transformers import pipeline

from easytrain import train

ROOT = Path(__file__).resolve().parent
MODEL = "distilbert/distilbert-base-uncased"
OUTPUT = ROOT / "output" / "text-classification"


def main() -> None:
    result = train(
        type="text-classification",
        model=MODEL,
        dataset=str(ROOT / "data" / "reviews.csv"),
        output=str(OUTPUT),
        epochs=1,
        batch_size=8,
    )
    print(result.metrics)

    clf = pipeline("text-classification", model=str(OUTPUT), tokenizer=str(OUTPUT))
    print(clf("I loved this movie."))
    print(clf("This was a waste of time."))


if __name__ == "__main__":
    main()
