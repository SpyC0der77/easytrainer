"""Print the resolved plan and write train_config.json + train_snippet.py without training."""

from pathlib import Path

from easytrain import train

ROOT = Path(__file__).resolve().parent
MODEL = "distilbert/distilbert-base-uncased"
OUTPUT = ROOT / "output" / "dry-run"


def main() -> None:
    result = train(
        type="token-classification",
        model=MODEL,
        dataset=str(ROOT / "data" / "ner.jsonl"),
        output=str(OUTPUT),
        epochs=3,
        dry_run=True,
    )
    print("model_id:", result.model_id)
    print("wrote:", OUTPUT / "train_config.json")
    print("wrote:", OUTPUT / "train_snippet.py")
    print("trainer:", result.trainer)


if __name__ == "__main__":
    main()
