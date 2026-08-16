"""Loud, educational errors. EasyTrain fails with a mapping example, not a guess."""


class EasyTrainError(Exception):
    """Base error for the EasyTrain public API."""


class UnknownTaskError(EasyTrainError):
    """`type=` is not a supported task."""


class SchemaError(EasyTrainError):
    """Dataset columns do not match the task convention."""


class ConfigError(EasyTrainError):
    """Invalid train() arguments or resolved training configuration."""
