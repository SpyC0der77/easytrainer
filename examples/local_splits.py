"""Load a local directory: train.csv plus an aliased validation split (dev.csv)."""

from pathlib import Path

from transformers import pipeline

from easytrain import train

ROOT = Path(__file__).resolve().parent
MODEL = "distilbert/distilbert-base-uncased"
OUTPUT = ROOT / "output" / "local-splits"


def main() -> None:
    result = train(
        type="text-classification",
        model=MODEL,
        dataset=str(ROOT / "data" / "splits"),
        output=str(OUTPUT),
        epochs=1,
        batch_size=8,
    )
    print(result.metrics)
    print("eval split:", result.plan.eval_split)

    clf = pipeline("text-classification", model=str(OUTPUT), tokenizer=str(OUTPUT))
    print(clf("Would buy this again."))


if __name__ == "__main__":
    main()
