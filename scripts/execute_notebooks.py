"""Execute every curriculum notebook in an isolated temporary copy."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    notebooks = sorted((root / "curriculum").rglob("*.ipynb"))
    if not notebooks:
        raise SystemExit("No curriculum notebooks found")

    for source in notebooks:
        with tempfile.TemporaryDirectory() as temporary:
            lesson_copy = Path(temporary) / source.parent.name
            shutil.copytree(source.parent, lesson_copy)
            notebook_path = lesson_copy / source.name
            notebook = nbformat.read(notebook_path, as_version=4)
            NotebookClient(
                notebook,
                timeout=args.timeout,
                kernel_name="computer-vision-field-guide",
                resources={"metadata": {"path": str(lesson_copy)}},
            ).execute()
            print(f"executed: {source.relative_to(root)}")


if __name__ == "__main__":
    main()
