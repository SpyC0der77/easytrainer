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
    "##zz",
]


def write_tiny_bert(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "vocab.txt").write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    import torch
    from tokenizers import Tokenizer
    from tokenizers.decoders import WordPiece as WordPieceDecoder
    from tokenizers.models import WordPiece
    from tokenizers.pre_tokenizers import BertPreTokenizer
    from tokenizers.processors import TemplateProcessing
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    vocab_map = {token: index for index, token in enumerate(VOCAB)}
    backend = Tokenizer(WordPiece(vocab_map, unk_token="[UNK]"))
    backend.pre_tokenizer = BertPreTokenizer()
    backend.decoder = WordPieceDecoder(prefix="##")
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab_map["[CLS]"]), ("[SEP]", vocab_map["[SEP]"])],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        do_lower_case=True,
        model_max_length=128,
    )
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
    torch.manual_seed(0)
    model = BertModel(config)
    tokenizer.save_pretrained(path)
    config.save_pretrained(path)
    model.save_pretrained(path)
    return path
