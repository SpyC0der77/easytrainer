from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from easytrain.constants import CANONICAL_COLUMNS
from easytrain.errors import ConfigError, SchemaError

SPLIT_TRAIN = ("train", "training")
SPLIT_VAL = ("validation", "val", "valid", "dev")
SPLIT_TEST = ("test", "testing")
LOAD_KEYS = {"path", "data", "name", "split", "data_files", "config", "builder"}


@dataclass
class DatasetBundle:
    train: Any
    validation: Any | None
    test: Any | None
    source: str
    mapping: dict[str, str]


def _is_hf_dataset(obj: Any) -> bool:
    try:
        from datasets import Dataset, DatasetDict, IterableDataset

        return isinstance(obj, (Dataset, DatasetDict, IterableDataset))
    except Exception:
        return False


def _as_dataset_dict(obj: Any):
    from datasets import Dataset, DatasetDict, IterableDataset

    if isinstance(obj, IterableDataset):
        raise ConfigError("IterableDataset is not supported in v1. Load a map-style datasets.Dataset.")
    if isinstance(obj, DatasetDict):
        return obj
    if isinstance(obj, Dataset):
        return DatasetDict({"train": obj})
    raise ConfigError(
        f"Unsupported dataset type: {type(obj).__name__}. "
        "Pass a Hub id, local CSV/JSON path, datasets.Dataset, or a mapping dict."
    )


def _pick_split(dataset_dict, names: tuple[str, ...]):
    for name in names:
        if name in dataset_dict:
            return dataset_dict[name]
    return None


def _extract_mapping(spec: Mapping[str, Any]) -> dict[str, str]:
    mapping = {}
    for key, value in spec.items():
        if key in CANONICAL_COLUMNS and isinstance(value, str):
            mapping[key] = value
    return mapping


def _rename_columns(dataset, mapping: dict[str, str]):
    if not mapping:
        return dataset
    rename = {}
    for canonical, source in mapping.items():
        if source == canonical:
            continue
        if source not in dataset.column_names:
            raise SchemaError(
                f"Mapped column {source!r} (for {canonical!r}) is not in the dataset. "
                f"Found columns: {dataset.column_names}"
            )
        if canonical in dataset.column_names and canonical != source:
            raise SchemaError(
                f"Cannot map {source!r} to {canonical!r}: the dataset already has a {canonical!r} column. "
                f"Rename or drop one of them so the mapping is unambiguous. Found columns: {dataset.column_names}"
            )
        rename[source] = canonical
    if not rename:
        return dataset
    return dataset.rename_columns(rename)


def _apply_mapping_to_dict(dataset_dict, mapping: dict[str, str]):
    from datasets import DatasetDict

    return DatasetDict({name: _rename_columns(split, mapping) for name, split in dataset_dict.items()})


def _load_local_file(path: Path):
    from datasets import load_dataset as hf_load_dataset

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return hf_load_dataset("csv", data_files=str(path))
    if suffix in {".json", ".jsonl"}:
        return hf_load_dataset("json", data_files=str(path))
    if suffix == ".parquet":
        return hf_load_dataset("parquet", data_files=str(path))
    raise ConfigError(f"Unsupported local file type {suffix!r} for {path}. Use CSV, JSON/JSONL, or Parquet.")


def _load_local_dir(path: Path):
    from datasets import DatasetDict
    from datasets import load_dataset as hf_load_dataset

    files: dict[str, str] = {}
    for file in sorted(path.iterdir()):
        stem = file.stem.lower()
        if file.suffix.lower() not in {".csv", ".json", ".jsonl", ".parquet"}:
            continue
        if stem in SPLIT_TRAIN:
            files["train"] = str(file)
        elif stem in SPLIT_VAL:
            files["validation"] = str(file)
        elif stem in SPLIT_TEST:
            files["test"] = str(file)
    if not files:
        singles = [file for file in path.iterdir() if file.suffix.lower() in {".csv", ".json", ".jsonl", ".parquet"}]
        if len(singles) == 1:
            return _load_local_file(singles[0])
        raise ConfigError(
            f"No train/validation/test CSV or JSON files found in {path}. "
            "Expected names like train.csv and validation.csv."
        )
    first = Path(next(iter(files.values())))
    builder = {".csv": "csv", ".json": "json", ".jsonl": "json", ".parquet": "parquet"}[first.suffix.lower()]
    loaded = hf_load_dataset(builder, data_files=files)
    if isinstance(loaded, DatasetDict):
        return loaded
    return loaded


def _load_hub(path: str, name: str | None = None, split: str | None = None):
    from datasets import load_dataset as hf_load_dataset

    hub_name = path
    config = name
    if config is None and ":" in path and not path.startswith(("http://", "https://")):
        hub_name, _, config = path.partition(":")
    kwargs: dict[str, Any] = {}
    if config:
        kwargs["name"] = config
    if split:
        kwargs["split"] = split
    return hf_load_dataset(hub_name, **kwargs)


def _canonical_split_name(split: str) -> str:
    key = split.lower()
    if key in SPLIT_TRAIN:
        return "train"
    if key in SPLIT_VAL:
        return "validation"
    if key in SPLIT_TEST:
        return "test"
    return split


def _restrict_to_split(loaded, split: str | None):
    if not split:
        return loaded
    from datasets import DatasetDict

    if isinstance(loaded, DatasetDict):
        for name in (split, _canonical_split_name(split)):
            if name in loaded:
                return loaded[name]
        raise SchemaError(f"split={split!r} was not found. Available splits: {list(loaded.keys())}")
    return loaded


def _load_from_path(path: str, name: str | None = None, split: str | None = None):
    local = Path(path).expanduser()
    if local.exists():
        if local.is_dir():
            loaded = _load_local_dir(local)
        else:
            loaded = _load_local_file(local)
        return _restrict_to_split(loaded, split)
    return _load_hub(path, name=name, split=split)


def load_dataset_spec(dataset: Any) -> DatasetBundle:
    mapping: dict[str, str] = {}
    source = "unknown"

    if isinstance(dataset, Mapping) and not _is_hf_dataset(dataset):
        unknown = [key for key in dataset if key not in LOAD_KEYS and key not in CANONICAL_COLUMNS]
        if unknown:
            supported = ", ".join(sorted(LOAD_KEYS | set(CANONICAL_COLUMNS)))
            raise ConfigError(f"Unknown dataset mapping key(s): {unknown}. Supported keys: {supported}.")
        mapping = _extract_mapping(dataset)
        if "data" in dataset:
            loaded = dataset["data"]
            source = "mapping.data"
        elif "path" in dataset:
            loaded = _load_from_path(
                str(dataset["path"]),
                name=dataset.get("name") or dataset.get("config"),
                split=dataset.get("split"),
            )
            source = str(dataset["path"])
        elif "data_files" in dataset:
            from datasets import load_dataset as hf_load_dataset

            loaded = hf_load_dataset(
                dataset.get("builder", "csv"),
                data_files=dataset["data_files"],
                split=dataset.get("split"),
            )
            source = str(dataset["data_files"])
        else:
            raise ConfigError(
                "dataset mapping dict needs a 'path' (Hub id or file) or a 'data' Dataset. "
                'Example: {"path": "reviews.csv", "text": "review", "label": "sentiment"}'
            )
    elif isinstance(dataset, (str, Path)):
        source = str(dataset)
        loaded = _load_from_path(str(dataset))
    elif _is_hf_dataset(dataset):
        loaded = dataset
        source = "datasets.Dataset"
    else:
        raise ConfigError(
            f"Unsupported dataset type: {type(dataset).__name__}. "
            "Pass a Hub id, local CSV/JSON path, datasets.Dataset, or a mapping dict like "
            '{"path": "data.csv", "text": "review", "label": "sentiment"}.'
        )

    dataset_dict = _as_dataset_dict(loaded)
    if mapping:
        dataset_dict = _apply_mapping_to_dict(dataset_dict, mapping)

    train = _pick_split(dataset_dict, SPLIT_TRAIN)
    if train is None:
        if len(dataset_dict) == 1:
            train = next(iter(dataset_dict.values()))
        else:
            raise SchemaError(f"No train split found. Available splits: {list(dataset_dict.keys())}")

    return DatasetBundle(
        train=train,
        validation=_pick_split(dataset_dict, SPLIT_VAL),
        test=_pick_split(dataset_dict, SPLIT_TEST),
        source=source,
        mapping=mapping,
    )


def ensure_eval_split(
    bundle: DatasetBundle,
    *,
    enabled: bool,
    seed: int,
    stratify_column: str | None,
) -> DatasetBundle:
    if not enabled:
        return bundle
    if bundle.validation is not None:
        return bundle
    n = len(bundle.train)
    if n < 4:
        return bundle
    test_size = max(1, min(max(1, n // 10), n // 2))
    kwargs: dict[str, Any] = {"test_size": test_size, "seed": seed}
    split = None
    if stratify_column and stratify_column in bundle.train.column_names:
        prepared = _prepare_stratify_column(bundle.train, stratify_column)
        if prepared is not None:
            dataset, strat_name = prepared
            try:
                split = dataset.train_test_split(stratify_by_column=strat_name, **kwargs)
                if strat_name == "_easytrain_stratify":
                    split = {
                        "train": split["train"].remove_columns([strat_name]),
                        "test": split["test"].remove_columns([strat_name]),
                    }
            except Exception:
                split = None
        if split is None:
            import warnings

            warnings.warn(
                f"Could not stratify on {stratify_column!r}; using a random validation split.",
                stacklevel=2,
            )
    if split is None:
        split = bundle.train.train_test_split(**kwargs)
    return replace(bundle, train=split["train"], validation=split["test"])


def _prepare_stratify_column(dataset, column: str) -> tuple[Any, str] | None:
    """Return `(dataset, column)` ready for HF stratify, without remapping labels."""
    from datasets import ClassLabel

    feature = dataset.features.get(column) if hasattr(dataset, "features") else None
    if feature is not None and getattr(feature, "names", None):
        return dataset, column
    try:
        ranks: list[int] = []
        unique: list[Any] = []
        index_of: dict[Any, int] = {}
        for raw in dataset[column]:
            value = raw
            if isinstance(value, bool):
                value = int(value)
            elif hasattr(value, "item") and type(value).__module__ == "numpy":
                value = value.item()
            if value not in index_of:
                index_of[value] = len(unique)
                unique.append(value)
            ranks.append(index_of[value])
        if len(unique) < 2:
            return None
        names = [str(item) for item in unique]
        mapped = dataset.add_column("_easytrain_stratify", ranks)
        mapped = mapped.cast_column("_easytrain_stratify", ClassLabel(names=names))
        return mapped, "_easytrain_stratify"
    except Exception:
        return None
