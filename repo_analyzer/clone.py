"""Clone GitHub (or other Git) repositories using dulwich."""

import warnings
from pathlib import Path

from dulwich import porcelain


def _repo_slug(repo_url: str) -> str:
    """Derive a filesystem-safe slug from a repo URL (e.g. owner/repo or repo)."""
    # Strip .git and trailing slashes, then take the last path part or owner/repo
    url = repo_url.rstrip("/").removesuffix(".git")
    parts = url.rstrip("/").split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}-{parts[-1]}"
    return parts[-1] if parts else "repo"


def clone_repo(
    repo_url: str,
    data_dir: Path | str | None = None,
    *,
    depth: int = 1,
    target_name: str | None = None,
) -> Path:
    """
    Clone a Git repository into data_dir/repos/<target_name or repo_slug>.

    Args:
        repo_url: Git clone URL (e.g. https://github.com/axios/axios.git).
        data_dir: Base directory for data; repos go in data_dir/repos/.
                  Defaults to <repo root>/data (caller's project root).
        depth: Clone depth; 1 for shallow clone.
        target_name: Folder name under repos/ (e.g. repo hash). If None, uses slug from URL.

    Returns:
        Path to the cloned repository directory.
    """
    if data_dir is None:
        # Default: repo root is parent of repo_analyzer package
        repo_root = Path(__file__).resolve().parent.parent
        data_dir = repo_root / "data"
    else:
        data_dir = Path(data_dir)

    repos_dir = data_dir / "repos"
    repos_dir.mkdir(parents=True, exist_ok=True)

    name = target_name if target_name else _repo_slug(repo_url)
    target = repos_dir / name

    if target.exists():
        warnings.warn(
            f"clone_repo: target directory already exists, skipping clone: {target}",
            stacklevel=2,
        )
        return target

    porcelain.clone(
        repo_url,
        str(target),
        depth=depth,
    )

    return target
