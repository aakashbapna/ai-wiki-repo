"""Service for file and indexing operations."""

import time
from pathlib import Path
from typing import TypedDict

from repo_analyzer.db import get_default_adapter
from repo_analyzer.services.file.code_analyzer import FileForIndex, IndexTaskStatus, index_file
from repo_analyzer.services.wiki.service import WikiService
from repo_analyzer.db_managers import RepoManager
from repo_analyzer.models import IndexTask, Repo, RepoFile, RepoFileMetadata
from constants import STALE_TASK_TIMEOUT_SECONDS
from repo_analyzer.models.index_task import TaskProgress, TaskStatus, TaskType, is_task_stale


class ListFilesResult(TypedDict):
    repo_hash: str
    total: int
    files: list[dict[str, object]]


class ReindexResult(TypedDict):
    repo_hash: str
    file_id: int
    metadata: RepoFileMetadata


class StopResult(TypedDict):
    repo_hash: str
    stopped_tasks: int


class RepoFileContent(TypedDict):
    repo_hash: str
    file_id: int
    file_name: str
    file_path: str
    content: str
    file_size: int


class FileService:
    """File-related service operations."""

    @staticmethod
    def list_repo_files(repo_hash: str, *, project_only: bool) -> ListFilesResult:
        adapter = get_default_adapter()
        with adapter.session() as session:
            repo_manager = RepoManager(session)
            repo = repo_manager.get_by_hash(repo_hash)
            if repo is None:
                raise ValueError(f"Repo not found: {repo_hash}")

            files = repo_manager.list_repo_files(
                repo_hash,
                filter_scan_excluded=True,
                project_file=True if project_only else None,
            )
            file_list: list[dict[str, object]] = []
            for f in files:
                meta = f.get_metadata()
                if not meta or not bool(meta.get("entry_point")):
                    continue
                file_list.append({
                    "file_id": f.file_id,
                    "file_path": f.file_path,
                    "file_name": f.file_name,
                    "is_project_file": f.is_project_file,
                    "is_scan_excluded": f.is_scan_excluded,
                    "metadata": meta,
                    "last_index_at": f.last_index_at,
                    "file_size": f.file_size,
                })

        return {
            "repo_hash": repo_hash,
            "total": len(file_list),
            "files": file_list,
        }

    @staticmethod
    def reindex_single_file(repo_hash: str, file_id: int) -> ReindexResult:
        adapter = get_default_adapter()
        with adapter.session() as session:
            repo_manager = RepoManager(session)
            repo = repo_manager.get_by_hash(repo_hash)
            if repo is None:
                raise ValueError(f"Repo not found: {repo_hash}")

            repo_file = (
                session.query(RepoFile)
                .filter(RepoFile.repo_hash == repo_hash, RepoFile.file_id == file_id)
                .first()
            )
            if repo_file is None:
                raise ValueError(f"File not found: {file_id}")

            meta = _index_repo_file(repo, repo_file)
            WikiService.build_wiki(repo_hash)
            return {
                "repo_hash": repo_hash,
                "file_id": file_id,
                "metadata": meta,
            }

    @staticmethod
    def get_index_task_status(repo_hash: str) -> IndexTaskStatus | None:
        adapter = get_default_adapter()
        with adapter.session() as session:
            task = (
                session.query(IndexTask)
                .filter(
                    IndexTask.repo_hash == repo_hash,
                    IndexTask.task_type == TaskType.INDEX_FILE.value,
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
            return IndexTaskStatus(
                repo_hash=task.repo_hash,
                status=task.status,
                total_files=task.total_files,
                completed_files=task.completed_files,
                remaining_files=remaining,
                task_id=task.task_id,
                progress=task.get_progress(),
            )

    @staticmethod
    def stop_indexing(repo_hash: str) -> StopResult:
        adapter = get_default_adapter()
        with adapter.session() as session:
            repo_manager = RepoManager(session)
            repo = repo_manager.get_by_hash(repo_hash)
            if repo is None:
                raise ValueError(f"Repo not found: {repo_hash}")

            tasks = (
                session.query(IndexTask)
                .filter(
                    IndexTask.repo_hash == repo_hash,
                    IndexTask.status == TaskStatus.RUNNING.value,
                    IndexTask.task_type == TaskType.INDEX_FILE.value,
                )
                .all()
            )
            now = int(time.time())
            stopped = 0
            for task in tasks:
                task.status = TaskStatus.STOPPED.value
                task.updated_at = now
                stopped += 1
            return {
                "repo_hash": repo_hash,
                "stopped_tasks": stopped,
            }

    @staticmethod
    def get_repo_file_content(repo_hash: str, file_id: int) -> RepoFileContent:
        adapter = get_default_adapter()
        with adapter.session() as session:
            repo_manager = RepoManager(session)
            repo = repo_manager.get_by_hash(repo_hash)
            if repo is None:
                raise ValueError(f"Repo not found: {repo_hash}")
            repo_file = (
                session.query(RepoFile)
                .filter(RepoFile.repo_hash == repo_hash, RepoFile.file_id == file_id)
                .first()
            )
            if repo_file is None:
                raise ValueError(f"File not found: {file_id}")
            rel_path = Path(repo_file.full_rel_path())
            file_path = repo.clone_path_resolved / rel_path
            if not file_path.exists() or not file_path.is_file():
                raise ValueError(f"File content not found at path: {rel_path.as_posix()}")
            content = _read_file_text(file_path)
            return {
                "repo_hash": repo_hash,
                "file_id": repo_file.file_id,
                "file_name": repo_file.file_name,
                "file_path": repo_file.file_path,
                "content": content,
                "file_size": repo_file.file_size,
            }


def _index_repo_file(repo: Repo, repo_file: RepoFile) -> RepoFileMetadata:
    rel_path = Path(repo_file.full_rel_path())
    file_path = repo.clone_path_resolved / rel_path
    content = _read_file_text(file_path)
    file_size_bytes = 0
    if file_path.exists() and file_path.is_file():
        try:
            file_size_bytes = file_path.stat().st_size
        except OSError:
            file_size_bytes = 0
    summaries = index_file([
        FileForIndex(
            file_path=rel_path.as_posix(),
            file_name=repo_file.file_name,
            content=content,
            file_size_bytes=file_size_bytes,
        )
    ])
    summary = summaries[0]
    meta: RepoFileMetadata = {
        "responsibility": summary["responsibility"],
        "key_elements": summary["key_elements"],
        "dependent_files": summary["dependent_files"],
        "entry_point": summary["entry_point"],
    }
    repo_file.set_metadata(meta)
    repo_file.last_index_at = int(time.time())
    return meta


def _read_file_text(path: Path, *, max_bytes: int = 200_000) -> str:
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")
