"""Manager for RepoSubsystem model: CRUD using a DB session."""

from sqlalchemy.orm import Session

from ..models import RepoSubsystem


class SubsystemManager:
    """Provides access to RepoSubsystem model. Takes a DB session as input."""

    def __init__(self, session: Session):
        self._session = session

    def add_subsystem(
        self,
        *,
        repo_hash: str,
        name: str,
        description: str,
        file_ids: list[int],
        keywords: list[str],
    ) -> RepoSubsystem:
        subsystem = RepoSubsystem(
            repo_hash=repo_hash,
            name=name,
            description=description,
        )
        subsystem.set_meta({
            "file_ids": file_ids,
            "keywords": keywords,
        })
        self._session.add(subsystem)
        self._session.flush()
        return subsystem

    def list_by_repo(self, repo_hash: str) -> list[RepoSubsystem]:
        return (
            self._session.query(RepoSubsystem)
            .filter(RepoSubsystem.repo_hash == repo_hash)
            .order_by(RepoSubsystem.subsystem_id)
            .all()
        )

    def delete_by_repo(self, repo_hash: str) -> int:
        return (
            self._session.query(RepoSubsystem)
            .filter(RepoSubsystem.repo_hash == repo_hash)
            .delete()
        )
