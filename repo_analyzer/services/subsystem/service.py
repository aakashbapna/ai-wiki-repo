"""Service for subsystem operations."""

import time
from typing import TypedDict

from repo_analyzer.db import get_default_adapter
from repo_analyzer.db_managers import RepoManager, SubsystemManager
from repo_analyzer.models import IndexTask
from repo_analyzer.models.index_task import TaskStatus, TaskType, is_task_stale
from repo_analyzer.services.subsystem.subsystem_builder import create_subsystems
from constants import STALE_TASK_TIMEOUT_SECONDS


class SubsystemResponse(TypedDict):
    subsystem_id: int
    name: str
    description: str
    meta: dict[str, object] | None
    created_at: int


class SubsystemTaskStatus(TypedDict):
    repo_hash: str
    status: str
    total_files: int
    completed_files: int
    remaining_files: int
    task_id: int


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
        status = create_subsystems(repo_hash)
        return status

    @staticmethod
    def get_build_status(repo_hash: str) -> SubsystemTaskStatus | None:
        adapter = get_default_adapter()
        with adapter.session() as session:
            task = (
                session.query(IndexTask)
                .filter(
                    IndexTask.repo_hash == repo_hash,
                    IndexTask.task_type == TaskType.BUILD_SUBSYSTEM.value,
                )
                .order_by(IndexTask.created_at.desc())
                .first()
            )
            if task is None:
                return None
            now = int(time.time())
            if is_task_stale(task, timeout_seconds=STALE_TASK_TIMEOUT_SECONDS, now=now):
                task.status = TaskStatus.STALE.value
                task.updated_at = now
                task.last_error = "Marked stale due to missing heartbeat."
                session.commit()
            remaining = max(0, task.total_files - task.completed_files)
            return SubsystemTaskStatus(
                repo_hash=task.repo_hash,
                status=task.status,
                total_files=task.total_files,
                completed_files=task.completed_files,
                remaining_files=remaining,
                task_id=task.task_id,
            )
