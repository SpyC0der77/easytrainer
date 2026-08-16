"""Public constants. Task names are strings, not enums."""

__version__ = "0.1.0"

V1_TASKS = (
    "text-classification",
    "token-classification",
)

PLANNED_TASKS = (
    "causal-language-modeling",
    "seq2seq",
    "image-classification",
    "image-to-text",
    "text-to-image",
)

ALL_TASKS = V1_TASKS + PLANNED_TASKS

# Hints for common aliases. These are never accepted as `type=`.
TYPE_HINTS = {
    "ner": "token-classification",
    "named-entity-recognition": "token-classification",
    "token-class": "token-classification",
    "classification": "text-classification",
    "sentiment": "text-classification",
    "text-class": "text-classification",
    "sequence-classification": "text-classification",
    "causal-lm": "causal-language-modeling",
    "clm": "causal-language-modeling",
    "llm": "causal-language-modeling",
    "sft": "causal-language-modeling",
    "summarization": "seq2seq",
    "translation": "seq2seq",
    "captioning": "image-to-text",
    "diffusion": "text-to-image",
}

CANONICAL_COLUMNS = (
    "text",
    "label",
    "sentence1",
    "sentence2",
    "tokens",
    "ner_tags",
    "input",
    "target",
    "image",
    "messages",
)
