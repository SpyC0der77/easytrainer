# Review policy (Kilo Code Reviews)

This repo is a **quick-start training kit**, not production software. Prefer short, readable scripts over defensive error handling.

## Do not flag

- Missing `try/except`, `weights_only`, cloned tensors, logger monkey patches, or Hugging Face warning filters
- Expected Hugging Face noise: new classifier head, LayerNorm `gamma`/`beta` vs `weight`/`bias`, `warmup_ratio` deprecation, DataParallel gather warning
- Extra production hardening, version-compat wrappers, or key-alias shims on checkpoints
- Issues only in generated `*.ipynb` files (they copy `*.py`; review the Python sources)

## Do flag

- Bugs that break the happy-path train / eval / infer loop on the default `config.json`
- `train.py` doing preprocess or metrics, `preprocess.py` doing training, or eval/infer living in the wrong file

## Severity

Use a **lenient** bar. Nitpicks and “would be nicer in prod” comments should not be posted.

## Sub-agents

Use 0 sub-agents unless the PR is large and cross-cutting. Stay read-only; do not post comments from sub-agents.
