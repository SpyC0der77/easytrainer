"""Sentence-pair classification from sentence1 / sentence2 / label columns."""

from pathlib import Path

from transformers import pipeline

from easytrain import train

ROOT = Path(__file__).resolve().parent
MODEL = "distilbert/distilbert-base-uncased"
OUTPUT = ROOT / "output" / "sentence-pairs"


def main() -> None:
    result = train(
        type="text-classification",
        model=MODEL,
        dataset=str(ROOT / "data" / "nli_pairs.csv"),
        output=str(OUTPUT),
        epochs=1,
        batch_size=8,
    )
    print(result.metrics)

    nli = pipeline("text-classification", model=str(OUTPUT), tokenizer=str(OUTPUT))
    print(nli({"text": "A man is playing guitar.", "text_pair": "Someone is making music."}))
    print(nli({"text": "A woman is jogging.", "text_pair": "A woman is sitting down."}))


if __name__ == "__main__":
    main()
