"""Load a Hub dataset and map its columns. This downloads GLUE SST-2.

The run is capped with max_steps so the example stays a demo. Remove max_steps
to train for a full epoch.
"""

from pathlib import Path

from easytrain import train

ROOT = Path(__file__).resolve().parent
MODEL = "distilbert/distilbert-base-uncased"
OUTPUT = ROOT / "output" / "hub-sst2"


def main() -> None:
    result = train(
        type="text-classification",
        model=MODEL,
        dataset={
            "path": "glue",
            "name": "sst2",
            "text": "sentence",
            "label": "label",
        },
        output=str(OUTPUT),
        epochs=1,
        batch_size=16,
        max_steps=20,
    )
    print(result.metrics)


if __name__ == "__main__":
    main()
