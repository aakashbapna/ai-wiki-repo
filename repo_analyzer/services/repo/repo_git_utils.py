"""Git operations for repositories using dulwich porcelain.

Provides clone, checkout, commit, and push operations for managing
repository state.
"""

import logging
import warnings
from pathlib import Path

from dulwich import porcelain

logger = logging.getLogger(__name__)

WIKI_BRANCH = "ai-repo-wiki"


def _parse_repo_url(repo_url: str) -> tuple[str, str | None]:
    """
    Parse repo URL and optional branch (url:branch).
    Branch is the segment after the last colon if it contains no slash (e.g. pre-prod).
    """
    parts = repo_url.rsplit(":", 1)
    if len(parts) == 2 and "/" not in parts[1]:
        return parts[0], parts[1]
    return repo_url, None


def _repo_slug(repo_url: str) -> str:
    """Derive a filesystem-safe slug from a repo URL (e.g. owner/repo or repo)."""
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
                  Optional branch can be appended with a colon, e.g.
                  https://github.fkinternal.com/Flipkart/sm-ui-server:pre-prod
        data_dir: Base directory for data; repos go in data_dir/repos/.
                  Defaults to <repo root>/data (caller's project root).
        depth: Clone depth; 1 for shallow clone.
        target_name: Folder name under repos/ (e.g. repo hash). If None, uses slug from URL.

    Returns:
        Path to the cloned repository directory.
    """
    url_only, branch = _parse_repo_url(repo_url)

    if data_dir is None:
        repo_root = Path(__file__).resolve().parents[3]
        data_dir = repo_root / "data"
    else:
        data_dir = Path(data_dir)

    repos_dir = data_dir / "repos"
    repos_dir.mkdir(parents=True, exist_ok=True)

    name = target_name if target_name else _repo_slug(url_only)
    target = repos_dir / name

    if target.exists():
        warnings.warn(
            f"clone_repo: target directory already exists, skipping clone: {target}",
            stacklevel=2,
        )
        return target

    clone_kwargs: dict[str, int | str | None] = {"depth": depth}
    if branch is not None:
        clone_kwargs["branch"] = branch

    porcelain.clone(
        url_only,
        str(target),
        **clone_kwargs,
    )

    return target


def checkout_or_create_branch(repo_path: Path | str, branch: str = WIKI_BRANCH) -> None:
    """
    Switch to the given branch, creating it from HEAD if it does not exist.

    Args:
        repo_path: Path to the repository.
        branch: Branch name to checkout or create.
    """
    repo_path = Path(repo_path)
    if not (repo_path / ".git").exists():
        raise ValueError(f"Not a git repository: {repo_path}")

    branches = porcelain.branch_list(repo_path)
    branch_exists = branch.encode("utf-8") in branches
    if branch_exists:
        porcelain.checkout(repo_path, target=branch)
    else:
        porcelain.branch_create(repo_path, branch)
        porcelain.checkout(repo_path, target=branch)


def stage_and_commit(
    repo_path: Path | str,
    paths: list[str],
    message: str,
    *,
    author: str | None = None,
) -> bytes:
    """
    Stage the given paths and create a commit.

    Args:
        repo_path: Path to the repository.
        paths: List of paths to stage (relative to repo root).
        message: Commit message.
        author: Optional "Name <email>" for author/committer.

    Returns:
        SHA of the new commit.
    """
    repo_path = Path(repo_path)
    porcelain.add(repo_path, paths=paths)
    return porcelain.commit(
        repo_path,
        message=message,
        author=author or "AI Wiki <ai-wiki@local>",
        committer=author or "AI Wiki <ai-wiki@local>",
    )


def push_branch(
    repo_path: Path | str,
    branch: str = WIKI_BRANCH,
    *,
    remote: str = "origin",
    force: bool = False,
) -> None:
    """
    Push the given branch to the remote.

    Args:
        repo_path: Path to the repository.
        branch: Branch name to push.
        remote: Remote name (default: origin).
        force: Force push overwriting remote refs.
    """
    repo_path = Path(repo_path)
    refspec = f"refs/heads/{branch}:refs/heads/{branch}"
    porcelain.push(
        repo_path,
        remote_location=remote,
        refspecs=[refspec],
        force=force,
    )
