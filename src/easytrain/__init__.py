"""EasyTrain — `transformers.pipeline` for training."""

from easytrain.api import EasyTrainer, train
from easytrain.constants import V1_TASKS, __version__
from easytrain.errors import ConfigError, EasyTrainError, SchemaError, UnknownTaskError
from easytrain.result import TrainResult

__all__ = [
    "train",
    "EasyTrainer",
    "TrainResult",
    "EasyTrainError",
    "SchemaError",
    "UnknownTaskError",
    "ConfigError",
    "V1_TASKS",
    "__version__",
]
