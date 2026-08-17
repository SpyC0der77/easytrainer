"""Keep tqdm/transformers logs readable when the output pane is narrow or resized.

Shared by every task. Import this before `datasets` or `transformers` so
progress bars are configured first.

Kaggle's log viewer (and most notebook captures) do not treat ``\\r`` as
"overwrite this line", and they do not expose the pane width as ``COLUMNS``.
A full-width tqdm bar then wraps mid-update and stacks into a staircase.
This module disables those bars in captured/Kaggle output and uses a compact,
``dynamic_ncols`` bar in a real terminal so a resize still fits. It also drops
a few known-harmless messages (DataParallel scalar-gather, and in captured
environments the DistilBERT load-mismatch table and unauthenticated Hub
notice) without changing framework log levels. Trainer log lines pad numeric
fields with zeros so loss, grad norm, learning rate, and epoch share a stable
width.
"""

from __future__ import annotations

import logging
import numbers
import os
import sys
import warnings
from pathlib import Path

_EXPECTED_LOG_NEEDLES = (
    "unauthenticated requests to the HF Hub",
    "LOAD REPORT",
)

_GATHER_WARNING = "Was asked to gather along dimension 0"
_log_filter: _ExpectedLogFilter | None = None
_gather_showwarning_wrapped = False


class _ExpectedLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(needle in msg for needle in _EXPECTED_LOG_NEEDLES)


def captured_display() -> bool:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return True
    if Path("/kaggle/input").exists() or Path("/kaggle/working").exists():
        return True
    if os.environ.get("COLAB_RELEASE_TAG"):
        return True
    try:
        return not sys.stdout.isatty()
    except Exception:
        return True


def disable_tqdm() -> bool:
    return captured_display()


def _quiet_gather_warning() -> None:
    """Drop DataParallel's scalar-gather UserWarning even if filters are reset."""
    global _gather_showwarning_wrapped
    warnings.filterwarnings("ignore", message=r".*gather along dimension 0.*", category=UserWarning)
    if _gather_showwarning_wrapped:
        return
    original = warnings.showwarning

    def showwarning(message, category, filename, lineno, file=None, line=None):
        if _GATHER_WARNING in str(message) and issubclass(category, UserWarning):
            return
        return original(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = showwarning
    _gather_showwarning_wrapped = True


def configure() -> None:
    _quiet_gather_warning()
    if disable_tqdm():
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")
        _install_expected_log_filter()
        try:
            from datasets.utils.logging import disable_progress_bar as disable_datasets_bar

            disable_datasets_bar()
        except Exception:
            pass
        try:
            from transformers.utils.logging import disable_progress_bar as disable_hf_bar

            disable_hf_bar()
        except Exception:
            pass
        return

    os.environ.setdefault("TQDM_DYNAMIC_NCOLS", "True")
    _patch_tqdm_for_resize()


def _in_filter_namespace(name: str) -> bool:
    return (
        name == "transformers"
        or name.startswith("transformers.")
        or name == "huggingface_hub"
        or name.startswith("huggingface_hub.")
    )


def _attach_expected_log_filter(logger: object) -> None:
    if _log_filter is None or not isinstance(logger, logging.Logger):
        return
    if not _in_filter_namespace(logger.name):
        return
    if _log_filter not in logger.filters:
        logger.addFilter(_log_filter)


def _install_expected_log_filter() -> None:
    global _log_filter
    if _log_filter is None:
        _log_filter = _ExpectedLogFilter()
    manager = logging.Logger.manager
    for logger in list(manager.loggerDict.values()):
        _attach_expected_log_filter(logger)
    for name in ("transformers", "huggingface_hub"):
        _attach_expected_log_filter(logging.getLogger(name))
    if not getattr(manager, "_easytrainer_filtered_getLogger", False):
        original = manager.getLogger

        def getLogger(name):
            logger = original(name)
            _attach_expected_log_filter(logger)
            return logger

        manager.getLogger = getLogger
        manager._easytrainer_filtered_getLogger = True


def _patch_tqdm_for_resize() -> None:
    try:
        import tqdm.std as std
    except Exception:
        return

    original_init = std.tqdm.__init__

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("dynamic_ncols", True)
        kwargs.setdefault("mininterval", 1.0)
        kwargs.setdefault(
            "bar_format",
            "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        original_init(self, *args, **kwargs)

    std.tqdm.__init__ = __init__


def format_log_number(key: str, value: object) -> object:
    """Pad numeric Trainer fields with leading/trailing zeros so columns line up."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return value
    name = str(key).lower()
    if name == "epoch" or name.endswith("_epoch"):
        return f"{float(value):010.7f}"
    if "learning_rate" in name or name == "lr":
        return f"{float(value):0.4e}"
    if name in {"step", "global_step"} or name.endswith("_step") or name.endswith("_steps"):
        return f"{int(value):06d}"
    return f"{float(value):07.4f}"


def format_trainer_logs(logs: dict) -> dict:
    return {key: format_log_number(key, value) for key, value in logs.items() if key != "total_flos"}


def _pop_callback(trainer, callback_cls) -> None:
    try:
        trainer.pop_callback(callback_cls)
    except Exception:
        pass


def attach_aligned_logging(trainer) -> None:
    """Use zero-padded loss lines instead of PrinterCallback when tqdm is off.

    On a TTY, Trainer already logs through ProgressCallback; adding another
    printer would duplicate every ``on_log`` record.
    """
    from transformers.trainer_callback import PrinterCallback, TrainerCallback

    class AlignedLogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or not getattr(state, "is_local_process_zero", True):
                return
            print(format_trainer_logs(logs), flush=True)

    _pop_callback(trainer, PrinterCallback)
    if disable_tqdm():
        trainer.add_callback(AlignedLogCallback())


def print_eval_metrics(trainer) -> dict:
    """Evaluate and print formatted metrics even if ``evaluate()`` never logs.

    Standalone ``Trainer.evaluate()`` only started calling ``self.log`` in
    transformers ~4.44. Always print the returned dict so older installs still
    show scores, and drop default printers so newer installs do not print twice.
    """
    from transformers.trainer_callback import PrinterCallback, ProgressCallback

    _pop_callback(trainer, PrinterCallback)
    _pop_callback(trainer, ProgressCallback)
    metrics = trainer.evaluate()
    if metrics:
        print(format_trainer_logs(metrics), flush=True)
    return metrics


configure()
