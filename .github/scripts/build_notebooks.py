"""Turn each task folder into a training notebook (no infer.py)."""

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP = {".git", ".github"}


def deps():
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]["dependencies"]


def cell(kind, source):
    if not source.endswith("\n"):
        source += "\n"
    lines = source.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def write_file_cell(name, content):
    return cell("code", f"%%writefile {name}\n{content.rstrip()}")


def task_dirs():
    for path in sorted(ROOT.iterdir()):
        if path.name in SKIP or not path.is_dir():
            continue
        if (path / "train.py").exists() and (path / "preprocess.py").exists():
            yield path


def build(folder, packages):
    readme = folder / "README.md"
    cells = []
    if readme.exists():
        cells.append(cell("markdown", readme.read_text(encoding="utf-8")))
    cells += [
        cell("markdown", "## Setup"),
        cell("code", "!pip install uv"),
        cell("code", "!uv pip install --system " + " ".join(packages)),
        cell(
            "code",
            "from IPython.display import HTML, display\n"
            "display(HTML(\"<style>pre,.output_text{white-space:pre-wrap!important;"
            "word-break:break-word!important}</style>\"))",
        ),
        cell("markdown", "## Config"),
        write_file_cell("config.json", (folder / "config.json").read_text(encoding="utf-8")),
        cell("markdown", "## Logs"),
        write_file_cell("progress.py", (ROOT / "progress.py").read_text(encoding="utf-8")),
        cell("markdown", "## Preprocess"),
        write_file_cell("preprocess.py", (folder / "preprocess.py").read_text(encoding="utf-8")),
        cell("markdown", "## Evaluate"),
        write_file_cell("evaluate.py", (folder / "evaluate.py").read_text(encoding="utf-8")),
        cell("markdown", "## Train"),
        write_file_cell("train.py", (folder / "train.py").read_text(encoding="utf-8")),
        cell("code", "!python train.py"),
        cell("code", "!python evaluate.py"),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def main():
    packages = deps()
    for folder in task_dirs():
        out = folder / f"{folder.name}.ipynb"
        out.write_text(json.dumps(build(folder, packages), indent=1) + "\n", encoding="utf-8")
        print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
