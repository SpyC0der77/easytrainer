from __future__ import annotations

import inspect
from typing import Any


def training_arguments_params() -> set[str]:
    from transformers import TrainingArguments

    return set(inspect.signature(TrainingArguments.__init__).parameters)


def filter_training_arguments(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep kwargs that the installed transformers.TrainingArguments accepts."""
    params = training_arguments_params()
    out = dict(kwargs)
    if "eval_strategy" in out and "eval_strategy" not in params and "evaluation_strategy" in params:
        out["evaluation_strategy"] = out.pop("eval_strategy")
    if "evaluation_strategy" in out and "evaluation_strategy" not in params and "eval_strategy" in params:
        out["eval_strategy"] = out.pop("evaluation_strategy")
    return {key: value for key, value in out.items() if key in params or key == "self"}


def trainer_tokenizer_kwarg(tokenizer: Any) -> dict[str, Any]:
    from transformers import Trainer

    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def supports_fused_adamw() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available()) and hasattr(torch.optim, "AdamW")
    except Exception:
        return False
