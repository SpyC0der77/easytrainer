from __future__ import annotations

import inspect
from typing import Any

from easytrain.errors import ConfigError

_STRATEGY_ALIASES = {"eval_strategy", "evaluation_strategy"}


def training_arguments_params() -> set[str]:
    from transformers import TrainingArguments

    return set(inspect.signature(TrainingArguments.__init__).parameters)


def filter_training_arguments(
    kwargs: dict[str, Any],
    *,
    user_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Keep kwargs that the installed transformers.TrainingArguments accepts.

    EasyTrain-generated keys that this transformers version does not know are
    dropped as compatibility shims. Caller-supplied **training_args that are
    not valid names (and not strategy aliases) raise ConfigError.
    """
    params = training_arguments_params()
    out = dict(kwargs)
    if "eval_strategy" in out and "eval_strategy" not in params and "evaluation_strategy" in params:
        out["evaluation_strategy"] = out.pop("eval_strategy")
    if "evaluation_strategy" in out and "evaluation_strategy" not in params and "eval_strategy" in params:
        out["eval_strategy"] = out.pop("evaluation_strategy")
    kept = {key: value for key, value in out.items() if key in params or key == "self"}
    if user_keys:
        dropped = {key for key in user_keys if key not in kept and key not in _STRATEGY_ALIASES}
        if dropped:
            names = ", ".join(sorted(dropped))
            raise ConfigError(
                f"Unsupported TrainingArguments option(s): {names}. "
                "These names are not accepted by the installed transformers version."
            )
    return kept


def trainer_tokenizer_kwarg(tokenizer: Any) -> dict[str, Any]:
    from transformers import Trainer

    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}
