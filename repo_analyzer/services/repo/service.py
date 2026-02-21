"""Service for repo-level operations."""

from pathlib import Path
from typing import Callable, TypedDict

from repo_analyzer.db import get_default_adapter
from repo_analyzer.db_managers import RepoManager
from repo_analyzer.models import Repo


class FetchRepoResult(TypedDict):
    repo_hash: str
    files: list[dict[str, object]]


class ClearDataResult(TypedDict):
    repos_deleted: int
    files_deleted: int


class RepoSummary(TypedDict):
    repo_hash: str
    owner: str
    repo_name: str
    clone_path: str
    url: str
    created_at: int


class RepoService:
    """Repo-related service operations."""

    @staticmethod
    def fetch_repo_with_files(
        *,
        repo_url: str,
        clone_repo_fn: Callable[[str, Path, str], Path],
        data_dir: Path,
    ) -> FetchRepoResult:
        owner, repo_name = Repo.parse_owner_repo(repo_url)
        repo_hash = Repo.compute_hash(owner, repo_name)
        path = clone_repo_fn(repo_url, data_dir=data_dir, target_name=repo_hash)

        adapter = get_default_adapter()
        with adapter.session() as session:
            repo_manager = RepoManager(session)
            existing = repo_manager.get_by_owner_repo(owner, repo_name)
            if existing:
                files = repo_manager.list_repo_files(existing.repo_hash, filter_scan_excluded=True)
                repo_hash = existing.repo_hash
            else:
                repo = repo_manager.add_repo_from_url(repo_url, path)
                repo_manager.populate_files_for_repo(repo.repo_hash, path)
                files = repo_manager.list_repo_files(repo.repo_hash, filter_scan_excluded=True)
                repo_hash = repo.repo_hash

            file_list = [
                {
                    "file_id": f.file_id,
                    "file_path": f.file_path,
                    "file_name": f.file_name,
                    "is_project_file": f.is_project_file,
                    "is_scan_excluded": f.is_scan_excluded,
                    "file_size": f.file_size,
                }
                for f in files
            ]

        return {
            "repo_hash": repo_hash,
            "files": file_list,
        }

    @staticmethod
    def clear_all_data() -> ClearDataResult:
        adapter = get_default_adapter()
        with adapter.session() as session:
            repo_manager = RepoManager(session)
            files_deleted, repos_deleted = repo_manager.clear_all()
        return {
            "repos_deleted": repos_deleted,
            "files_deleted": files_deleted,
        }

    @staticmethod
    def list_repos() -> list[RepoSummary]:
        adapter = get_default_adapter()
        with adapter.session() as session:
            repo_manager = RepoManager(session)
            repos = repo_manager.list_repos()
            return [
                {
                    "repo_hash": r.repo_hash,
                    "owner": r.owner,
                    "repo_name": r.repo_name,
                    "clone_path": r.clone_path,
                    "url": r.url,
                    "created_at": r.created_at,
                }
                for r in repos
            ]

    @staticmethod
    def get_repo_detail(repo_hash: str) -> RepoSummary:
        adapter = get_default_adapter()
        with adapter.session() as session:
            repo_manager = RepoManager(session)
            repo = repo_manager.get_by_hash(repo_hash)
            if repo is None:
                raise ValueError(f"Repo not found: {repo_hash}")
            return {
                "repo_hash": repo.repo_hash,
                "owner": repo.owner,
                "repo_name": repo.repo_name,
                "clone_path": repo.clone_path,
                "url": repo.url,
                "created_at": repo.created_at,
            }
