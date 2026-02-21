"""Manager for Repo model: CRUD using a DB session."""

import time
from pathlib import Path

from sqlalchemy.orm import Session

from ..models import Repo, RepoFile
from ..repo_scanner import scan_repo_files

class RepoManager:
    """Provides access to Repo model. Takes a DB session (handle) as input."""

    def __init__(self, session: Session):
        self._session = session

    def add_repo(
        self,
        *,
        owner: str,
        repo_name: str,
        clone_path: str | Path,
        url: str,
    ) -> Repo:
        """Insert a cloned repo. Uses repo_hash(owner, repo_name) as unique key."""
        h = Repo.compute_hash(owner, repo_name)
        clone_path_str = str(Path(clone_path).resolve())
        repo = Repo(
            repo_hash=h,
            owner=owner,
            repo_name=repo_name,
            clone_path=clone_path_str,
            url=url,
            created_at=int(time.time()),
        )
        self._session.add(repo)
        self._session.flush()  # so repo.id is set if needed
        return repo

    def add_repo_from_url(self, repo_url: str, clone_path: str | Path) -> Repo:
        """Parse owner/repo from URL and add or update. Idempotent for same owner/repo."""
        owner, repo_name = Repo.parse_owner_repo(repo_url)
        clone_path_str = str(Path(clone_path).resolve())
        existing = self.get_by_owner_repo(owner, repo_name)
        if existing:
            existing.clone_path = clone_path_str
            existing.url = repo_url.strip()
            existing.repo_hash = Repo.compute_hash(owner, repo_name)  # migrate to short hash
            return existing
        return self.add_repo(
            owner=owner,
            repo_name=repo_name,
            clone_path=clone_path,
            url=repo_url.strip(),
        )

    def get_by_hash(self, repo_hash: str) -> Repo | None:
        """Return Repo by repo_hash or None."""
        return self._session.query(Repo).filter(Repo.repo_hash == repo_hash).first()

    def get_by_owner_repo(self, owner: str, repo_name: str) -> Repo | None:
        """Return Repo by owner and repo_name or None (case-insensitive)."""
        o = owner.strip().lower()
        r = repo_name.strip().lower()
        return (
            self._session.query(Repo)
            .filter(Repo.owner.ilike(o), Repo.repo_name.ilike(r))
            .first()
        )

    def list_repos(self) -> list[Repo]:
        """Return all cloned repos, ordered by created_at descending."""
        return (
            self._session.query(Repo)
            .order_by(Repo.created_at.desc())
            .all()
        )

    def populate_files_for_repo(self, repo_hash: str, clone_path: Path | str) -> int:
        """
        Scan the cloned repo directory and create RepoFile rows for every file.
        Returns the number of files added.
        """
        repo_files = scan_repo_files(repo_hash, clone_path)
        for rf in repo_files:
            self._session.add(rf)
        return len(repo_files)

    def list_repo_files(
        self,
        repo_hash: str,
        *,
        filter_scan_excluded: bool = False,
        project_file: bool | None = None,
    ) -> list[RepoFile]:
        """
        Return RepoFile entities for the given repo_hash, ordered by file_path.

        Args:
            repo_hash: Hash identifying the repo.
            filter_scan_excluded: When True, exclude files where is_scan_excluded
                is True (i.e. return scannable files only).
            project_file: When True, return only project files. When False,
                return only non-project files. When None (default), no filter.
        """
        query = (
            self._session.query(RepoFile)
            .filter(RepoFile.repo_hash == repo_hash)
        )
        if filter_scan_excluded:
            query = query.filter(RepoFile.is_scan_excluded == False)  # noqa: E712
        if project_file is not None:
            query = query.filter(RepoFile.is_project_file == project_file)  # noqa: E712
        return query.order_by(RepoFile.file_path).all()

    def clear_all(self) -> tuple[int, int]:
        """Delete all RepoFile and Repo rows. Returns (files_deleted, repos_deleted)."""
        files_deleted = self._session.query(RepoFile).delete()
        repos_deleted = self._session.query(Repo).delete()
        return (files_deleted, repos_deleted)
