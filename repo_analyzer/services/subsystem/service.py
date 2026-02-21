"""Service for subsystem operations."""

from typing import TypedDict

from repo_analyzer.db import get_default_adapter
from repo_analyzer.db_managers import RepoManager, SubsystemManager
from repo_analyzer.services.subsystem.subsystem_builder import create_subsystems


class SubsystemResponse(TypedDict):
    subsystem_id: int
    name: str
    description: str
    meta: dict[str, object] | None
    created_at: int


class SubsystemService:
    """Subsystem-related service operations."""

    @staticmethod
    def list_subsystems(repo_hash: str) -> list[SubsystemResponse]:
        adapter = get_default_adapter()
        with adapter.session() as session:
            repo_manager = RepoManager(session)
            repo = repo_manager.get_by_hash(repo_hash)
            if repo is None:
                raise ValueError(f"Repo not found: {repo_hash}")
            subsystem_manager = SubsystemManager(session)
            subsystems = subsystem_manager.list_by_repo(repo_hash)
            return [
                {
                    "subsystem_id": s.subsystem_id,
                    "name": s.name,
                    "description": s.description,
                    "meta": s.get_meta(),
                    "created_at": s.created_at,
                }
                for s in subsystems
            ]

    @staticmethod
    def build_subsystems(repo_hash: str) -> dict[str, object]:
        return create_subsystems(repo_hash)
