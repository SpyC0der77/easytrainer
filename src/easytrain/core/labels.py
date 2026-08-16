from __future__ import annotations

from typing import Any

from easytrain.errors import SchemaError
from easytrain.result import LabelInfo


def _class_label_names(feature: Any) -> list[str] | None:
    names = getattr(feature, "names", None)
    if names:
        return list(names)
    inner = getattr(feature, "feature", None)
    inner_names = getattr(inner, "names", None)
    if inner_names:
        return list(inner_names)
    return None


def _python_label(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if hasattr(value, "item") and type(value).__module__ == "numpy":
        return value.item()
    return value


def _as_items(row: Any) -> list[Any]:
    if isinstance(row, (str, bytes)):
        return [row]
    if isinstance(row, (list, tuple)):
        return list(row)
    if hasattr(row, "tolist") and not isinstance(row, dict):
        converted = row.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    return [row]


def _unique_values(dataset, column: str) -> list[Any]:
    values: list[Any] = []
    seen: set[Any] = set()
    for row in dataset[column]:
        items = _as_items(row)
        for item in items:
            item = _python_label(item)
            try:
                key = (type(item), item)
                if key in seen:
                    continue
                seen.add(key)
            except TypeError:
                key = (type(item).__name__, str(item))
                if key in seen:
                    continue
                seen.add(key)
            values.append(item)
    return values


def infer_label_info(dataset, column: str, *, kind: str) -> LabelInfo:
    if column not in dataset.column_names:
        raise SchemaError(f"Cannot infer labels: {column!r} is not a column.")

    feature = dataset.features.get(column) if hasattr(dataset, "features") else None
    names = _class_label_names(feature) if feature is not None else None
    if names:
        id2label = {i: name for i, name in enumerate(names)}
        label2id = {name: i for i, name in enumerate(names)}
        return LabelInfo(
            num_labels=len(names),
            id2label=id2label,
            label2id=label2id,
            label_column=column,
            names=names,
        )

    unique = _unique_values(dataset, column)
    if not unique:
        raise SchemaError(f"Column {column!r} has no labels to infer.")

    ints = [v for v in unique if isinstance(v, int) and not isinstance(v, bool)]
    strs = [v for v in unique if isinstance(v, str)]
    if strs and not ints:
        names = _sort_names(strs, kind=kind)
        id2label = {i: name for i, name in enumerate(names)}
        label2id = {name: i for i, name in enumerate(names)}
        return LabelInfo(
            num_labels=len(names),
            id2label=id2label,
            label2id=label2id,
            label_column=column,
            names=names,
        )
    if ints and not strs:
        max_id = max(ints)
        min_id = min(ints)
        if min_id < 0:
            raise SchemaError(f"Negative label ids in {column!r} are not supported.")
        names = [f"LABEL_{i}" for i in range(max_id + 1)]
        id2label = {i: name for i, name in enumerate(names)}
        label2id = {name: i for i, name in enumerate(names)}
        return LabelInfo(
            num_labels=len(names),
            id2label=id2label,
            label2id=label2id,
            label_column=column,
            names=names,
        )
    raise SchemaError(
        f"Mixed or unsupported label types in {column!r}: {unique[:8]!r}. Use a single integer or string label space."
    )


def _sort_names(names: list[str], *, kind: str) -> list[str]:
    if kind != "bio":
        return sorted(names)

    def key(name: str) -> tuple:
        if name == "O":
            return (0, "", 0)
        if name.startswith("B-"):
            return (1, name[2:], 0)
        if name.startswith("I-"):
            return (1, name[2:], 1)
        return (2, name, 0)

    return sorted(names, key=key)


def encode_label_value(value: Any, labels: LabelInfo) -> int:
    value = _python_label(value)
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0 or value >= labels.num_labels:
            raise SchemaError(f"Label id {value} is outside 0..{labels.num_labels - 1} ({labels.names}).")
        return value
    key = str(value)
    if key not in labels.label2id:
        raise SchemaError(f"Unknown label {value!r}. Known labels: {labels.names}")
    return labels.label2id[key]
