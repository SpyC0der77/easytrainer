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
notice) without changing framework log levels.
"""

from __future__ import annotations

import logging
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


def configure() -> None:
    warnings.filterwarnings("ignore", message=_GATHER_WARNING, category=UserWarning)
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


def _install_expected_log_filter() -> None:
    global _log_filter
    if _log_filter is None:
        _log_filter = _ExpectedLogFilter()
    for name in ("transformers", "huggingface_hub"):
        logger = logging.getLogger(name)
        if _log_filter not in logger.filters:
            logger.addFilter(_log_filter)


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


configure()
