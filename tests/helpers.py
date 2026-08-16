from __future__ import annotations

from pathlib import Path

VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "hello",
    "world",
    "foo",
    "bar",
    "paris",
    "love",
    "i",
    "a",
    "the",
    "cat",
    "sat",
    "on",
    "mat",
    "good",
    "bad",
    "positive",
    "negative",
    "john",
    "lives",
    "in",
    "new",
    "york",
    "happy",
    "sad",
]


def write_tiny_bert(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "vocab.txt").write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    from transformers import BertConfig, BertModel, BertTokenizerFast

    tokenizer = BertTokenizerFast(vocab_file=str(path / "vocab.txt"), do_lower_case=True)
    config = BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=0,
        unk_token_id=1,
        cls_token_id=2,
        sep_token_id=3,
        type_vocab_size=2,
    )
    model = BertModel(config)
    tokenizer.save_pretrained(path)
    config.save_pretrained(path)
    model.save_pretrained(path)
    return path
