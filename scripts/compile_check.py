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


def compile_all(root: Path) -> list[str]:
    failures: list[str] = []
    for path in iter_python_files(root):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"{path}: {exc.msg}")
    return failures


def import_check(modules: list[str]) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    skipped: list[str] = []
    for name in modules:
        try:
            __import__(name)
        except ModuleNotFoundError as exc:
            skipped.append(f"{name}: missing dependency {exc.name}")
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    return failures, skipped


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    compile_failures = compile_all(root)
    import_failures, import_skipped = import_check(["app", "repo_analyzer"])
    failures = compile_failures + import_failures
    if failures:
        print("Compilation/import errors:")
        for item in failures:
            print(item)
        return 1
    if import_skipped:
        print("Import checks skipped due to missing dependencies:")
        for item in import_skipped:
            print(item)
    print("Compilation/import check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
