"""Map non-canonical CSV columns (review, sentiment) onto text + label."""

from pathlib import Path

from transformers import pipeline

from easytrain import train

ROOT = Path(__file__).resolve().parent
MODEL = "distilbert/distilbert-base-uncased"
OUTPUT = ROOT / "output" / "column-mapping"


def main() -> None:
    result = train(
        type="text-classification",
        model=MODEL,
        dataset={
            "path": str(ROOT / "data" / "reviews_raw.csv"),
            "text": "review",
            "label": "sentiment",
        },
        output=str(OUTPUT),
        epochs=1,
        batch_size=8,
    )
    print(result.metrics)

    clf = pipeline("text-classification", model=str(OUTPUT), tokenizer=str(OUTPUT))
    print(clf("The espresso was perfect."))
    print(clf("Cold coffee and rude service."))


if __name__ == "__main__":
    main()
