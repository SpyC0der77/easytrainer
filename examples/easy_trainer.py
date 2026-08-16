"""Sklearn-style wrapper around the same train() router."""

from pathlib import Path

from transformers import pipeline

from easytrain import EasyTrainer

ROOT = Path(__file__).resolve().parent
MODEL = "distilbert/distilbert-base-uncased"
OUTPUT = ROOT / "output" / "easy-trainer"


def main() -> None:
    clf = EasyTrainer(
        type="text-classification",
        model=MODEL,
        dataset=str(ROOT / "data" / "reviews.csv"),
        output=str(OUTPUT),
        epochs=1,
        batch_size=8,
    ).fit()
    print(clf.evaluate())
    clf.save()

    pipe = pipeline("text-classification", model=str(OUTPUT), tokenizer=str(OUTPUT))
    print(pipe("A charming and surprisingly funny story."))


if __name__ == "__main__":
    main()
