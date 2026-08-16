# EasyTrain examples

Small, runnable scripts for v1 (`text-classification` and `token-classification`).

## Setup

From the repo root:

```bash
pip install -e .
```

The first run downloads DistilBERT from the Hub (~250MB). Training examples use the tiny CSVs/JSONL under `data/`, so they finish in a couple of minutes on CPU after that download.

Outputs land in `examples/output/` (gitignored). After `train()`, `transformers.pipeline(type, output)` works on that folder.

## Scripts

| Script | What it shows |
|---|---|
| `text_classification.py` | Local CSV with canonical `text` + `label` columns |
| `column_mapping.py` | Mapping `review` / `sentiment` onto `text` + `label` |
| `sentence_pairs.py` | `sentence1` + `sentence2` + `label` |
| `token_classification.py` | JSONL with `tokens` + BIO `ner_tags` |
| `token_classification_mapping.py` | Mapping `words` / `tags` onto `tokens` + `ner_tags` |
| `local_splits.py` | Directory with `train.csv` and `dev.csv` (alias for validation) |
| `easy_trainer.py` | Sklearn-style `EasyTrainer`: `fit()`, then `evaluate()`, then `save()` |
| `dry_run.py` | Print the plan; write `train_config.json` + `train_snippet.py`; do not train |
| `hub_dataset.py` | Hub id `glue` / `sst2` with a column mapping (downloads data) |

```bash
python examples/dry_run.py
python examples/text_classification.py
python examples/token_classification.py
```

EasyTrain does not guess column names. If your file uses different headers, pass a mapping dict as in `column_mapping.py`.
