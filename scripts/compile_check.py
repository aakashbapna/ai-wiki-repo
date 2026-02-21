"""Compile-check all Python files in the repo."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import py_compile


EXCLUDED_DIRS: set[str] = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "data",
}


def iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        yield path


def compile_all(root: Path) -> int:
    failures: list[str] = []
    for path in iter_python_files(root):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"{path}: {exc.msg}")
    if failures:
        print("Compilation errors:")
        for item in failures:
            print(item)
        return 1
    print("Compilation check passed.")
    return 0


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    return compile_all(root)


if __name__ == "__main__":
    raise SystemExit(main())
