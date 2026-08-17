"""Keep tqdm/transformers logs readable when the output pane is narrow or resized.

Import this before `datasets` or `transformers` so progress bars are configured first.

Kaggle's log viewer does not treat ``\\r`` as overwrite, so a full-width tqdm bar
wraps into a staircase. This module turns those bars off in captured/Kaggle
output and uses a compact bar in a real terminal.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path


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
    warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0", category=UserWarning)
    if disable_tqdm():
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        return

    os.environ.setdefault("TQDM_DYNAMIC_NCOLS", "True")
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
