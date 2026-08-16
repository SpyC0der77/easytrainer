from __future__ import annotations

from easytrain.constants import ALL_TASKS, PLANNED_TASKS, TYPE_HINTS, V1_TASKS
from easytrain.errors import UnknownTaskError

from . import text_classification, token_classification

_PLUGINS = {
    "text-classification": text_classification,
    "token-classification": token_classification,
}


def get_task(task_type: str):
    plugin = _PLUGINS.get(task_type)
    if plugin is not None:
        return plugin

    hint = TYPE_HINTS.get(task_type)
    hint_line = f" Did you mean {hint!r}?" if hint else ""
    if task_type in PLANNED_TASKS:
        raise UnknownTaskError(
            f"{task_type!r} is planned but not in v1.{hint_line}\n\n"
            f"v1 supports: {', '.join(V1_TASKS)}\n"
            f"Coming later: {', '.join(PLANNED_TASKS)}"
        )
    raise UnknownTaskError(
        f"Unknown task type {task_type!r}.{hint_line}\n\n"
        f"v1 supports: {', '.join(V1_TASKS)}\n"
        f"Known later types: {', '.join(ALL_TASKS)}"
    )


def registered_task_modules():
    return dict(_PLUGINS)
