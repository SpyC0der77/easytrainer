# token-class

Assign a label to every token. This folder is a drop-in trainer for any token-classification dataset: swap the example in `preprocess.py` and `config.json`, then run.

## Example: emotion spans on GoEmotions

The bundled example finds emotion *spans* in Reddit comments from [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions), using [sdeakin/GoEmotions-Projected-BIO-Emotions](https://huggingface.co/datasets/sdeakin/GoEmotions-Projected-BIO-Emotions). Each row is a comment plus character/token spans labeled with an emotion (`joy`, `sadness`, `anger`, …).

`preprocess.py` turns those spans into per-token BIO tags:

| token | tag |
| --- | --- |
| I | `O` |
| am | `O` |
| so | `B-Joy` |
| happy | `I-Joy` |
| today | `O` |

- `B-{emotion}` starts a span, `I-{emotion}` continues it, `O` is outside any span.
- Emotion names are taken from each span's `subtype` (or `type`) and normalized to `Joy`, `Sadness`, etc. Spans missing an emotion label or token indices are skipped.
- Empty trailing tokens are stripped after BIO tags are assigned (span indices refer to the original token list); comments with no tokens are dropped.
- Labels are collected from the data (`O` first, then every `B-*` / `I-*` tag that appears).
- The split is 90/10 train/val (`test_size` and `seed` in `config.json`).

Training fine-tunes `distilbert-base-uncased` as a token classifier. Wordpiece tokens inherit the word's label; subword continuations and special tokens are ignored (`-100`) so they do not affect loss or metrics. Eval runs each epoch; training stops if `span_f1` does not improve for `early_stopping_patience` epochs, and the best checkpoint is saved.

`evaluate.py` reports:

- **token accuracy** — share of labeled word tokens predicted correctly
- **span precision / recall / F1** — exact match on `(start, end, emotion)` spans, which is the metric used to pick the best checkpoint (`span_f1`)

`infer.py` prints a few validation comments that contain at least one gold span, with gold vs predicted spans and a token-level diff (`!` on mismatches).

## Layout

```
.
├── config.json    # model, data URL, split, and training settings
├── preprocess.py  # load GoEmotions spans and build BIO labels
├── train.py       # train and save the best checkpoint
├── evaluate.py    # score the saved model (token accuracy, span P/R/F1)
├── infer.py       # run the saved model on validation examples
├── README.md
```

Edit `config.json`, then from this folder:

```
python train.py
python evaluate.py
python infer.py
```
