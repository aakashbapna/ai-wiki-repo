"""Scan a cloned repository directory and produce RepoFile instances."""

import os
import time
from pathlib import Path

from .models import RepoFile
from .utils import is_project_file, is_scan_excluded_file

EXCLUDED_FOLDERS: frozenset[str] = frozenset({
    ".git",
    "node_modules",
    "bower_components",
    "dist",
    ".venv",
    ".env",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "__pycache__",
})


def scan_repo_files(repo_hash: str, clone_path: Path | str) -> list[RepoFile]:
    """
    Walk a cloned repo directory and return an unsaved RepoFile for every file.

    Skips any directory whose name appears in EXCLUDED_FOLDERS.

    Args:
        repo_hash: Hash identifying the owning repo row.
        clone_path: Absolute path to the cloned repository root.

    Returns:
        List of unsaved RepoFile ORM objects ready to be added to a session.
    """
    root = Path(clone_path).resolve()
    if not root.is_dir():
        return []

    repo_files: list[RepoFile] = []
    now = int(time.time())

    for dirpath, _dirnames, filenames in os.walk(root):
        if any(folder in Path(dirpath).parts for folder in EXCLUDED_FOLDERS):
            continue
        for name in filenames:
            full_path = Path(dirpath) / name
            try:
                rel_path = full_path.relative_to(root)
            except ValueError:
                continue
            #remove file name from path
            rel_path_str = str(rel_path).replace(name, "").replace("\\", "/").strip("/")
            try:
                modified_at_epoch = int(full_path.stat().st_mtime)
            except OSError:
                modified_at_epoch = now
            repo_files.append(RepoFile(
                repo_hash=repo_hash,
                file_path=rel_path_str,
                file_name=name,
                created_at=now,
                modified_at=modified_at_epoch,
                metadata_json=None,
                is_scan_excluded=is_scan_excluded_file(name),
                is_project_file=is_project_file(name),
                project_name=None,
            ))

    return repo_files
