"""Shared training core. Task plugins must not reimplement this loop."""

from easytrain.core.loop import run_training

__all__ = ["run_training"]
