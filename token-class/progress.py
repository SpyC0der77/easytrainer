"""Keep tqdm/transformers logs readable when the output pane is narrow or resized.

Kaggle's log viewer (and most notebook captures) do not treat ``\\r`` as
"overwrite this line", and they do not expose the pane width as ``COLUMNS``.
A full-width tqdm bar then wraps mid-update and stacks into a staircase.
This module disables those bars in captured/Kaggle output and uses a compact,
``dynamic_ncols`` bar in a real terminal so a resize still fits.
"""

from __future__ import annotations

import os
import sys
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
    if disable_tqdm():
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
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
