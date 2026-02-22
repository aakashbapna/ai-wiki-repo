"""Service for wiki operations."""

import time
from typing import TypedDict

from constants import STALE_TASK_TIMEOUT_SECONDS
from repo_analyzer.db import get_default_adapter
from repo_analyzer.db_managers import RepoManager, WikiManager
from repo_analyzer.models import IndexTask, RepoFile
from repo_analyzer.models.index_task import TaskStatus, TaskType, is_task_stale
from repo_analyzer.services.wiki.wiki_builder import build_wiki


class WikiTaskStatus(TypedDict):
    repo_hash: str
    status: str
    total_files: int
    completed_files: int
    remaining_files: int
    task_id: int


class WikiService:
    """Wiki-related service operations."""

    @staticmethod
    def build_wiki(repo_hash: str) -> WikiTaskStatus:
        return build_wiki(repo_hash)

    @staticmethod
    def get_build_status(repo_hash: str) -> WikiTaskStatus | None:
        adapter = get_default_adapter()
        with adapter.session() as session:
            task = (
                session.query(IndexTask)
                .filter(
                    IndexTask.repo_hash == repo_hash,
                    IndexTask.task_type == TaskType.BUILD_WIKI.value,
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
            return WikiTaskStatus(
                repo_hash=task.repo_hash,
                status=task.status,
                total_files=task.total_files,
                completed_files=task.completed_files,
                remaining_files=remaining,
                task_id=task.task_id,
            )

    @staticmethod
    def list_sidebars(repo_hash: str) -> list[dict[str, object]]:
        adapter = get_default_adapter()
        with adapter.session() as session:
            repo_manager = RepoManager(session)
            repo = repo_manager.get_by_hash(repo_hash)
            if repo is None:
                raise ValueError(f"Repo not found: {repo_hash}")
            wiki_manager = WikiManager(session)
            nodes = wiki_manager.list_sidebars(repo_hash)
            return [
                {
                    "node_id": n.node_id,
                    "repo_hash": n.repo_hash,
                    "parent_node_id": n.parent_node_id,
                    "name": n.name,
                    "page_id": n.page_id,
                    "is_active": n.is_active,
                    "meta": n.get_meta(),
                    "created_at": n.created_at,
                    "updated_at": n.updated_at,
                }
                for n in nodes
            ]

    @staticmethod
    def list_pages(repo_hash: str) -> list[dict[str, object]]:
        adapter = get_default_adapter()
        with adapter.session() as session:
            repo_manager = RepoManager(session)
            repo = repo_manager.get_by_hash(repo_hash)
            if repo is None:
                raise ValueError(f"Repo not found: {repo_hash}")
            wiki_manager = WikiManager(session)
            pages = wiki_manager.list_pages(repo_hash)
            return [
                {
                    "page_id": p.page_id,
                    "repo_hash": p.repo_hash,
                    "title": p.title,
                    "meta": p.get_meta(),
                    "created_at": p.created_at,
                    "updated_at": p.updated_at,
                }
                for p in pages
            ]

    @staticmethod
    def list_page_contents(repo_hash: str, page_id: int) -> list[dict[str, object]]:
        adapter = get_default_adapter()
        with adapter.session() as session:
            repo_manager = RepoManager(session)
            repo = repo_manager.get_by_hash(repo_hash)
            if repo is None:
                raise ValueError(f"Repo not found: {repo_hash}")
            wiki_manager = WikiManager(session)
            contents = wiki_manager.list_page_contents(page_id)
            return [
                {
                    "content_id": c.content_id,
                    "page_id": c.page_id,
                    "content_type": c.content_type,
                    "content": c.content,
                    "meta": c.get_meta(),
                    "created_at": c.created_at,
                    "updated_at": c.updated_at,
                }
                for c in contents
            ]

    @staticmethod
    def get_page_with_contents(repo_hash: str, page_id: int) -> dict[str, object]:
        adapter = get_default_adapter()
        with adapter.session() as session:
            repo_manager = RepoManager(session)
            repo = repo_manager.get_by_hash(repo_hash)
            if repo is None:
                raise ValueError(f"Repo not found: {repo_hash}")
            wiki_manager = WikiManager(session)
            pages = wiki_manager.list_pages(repo_hash)
            page = next((p for p in pages if p.page_id == page_id), None)
            if page is None:
                raise ValueError(f"Page not found: {page_id}")
            contents = wiki_manager.list_page_contents(page_id)
            source_ids: set[int] = set()
            for content in contents:
                meta = content.get_meta() or {}
                ids = meta.get("source_file_ids") or []
                source_ids.update(int(file_id) for file_id in ids)
            source_files: dict[int, RepoFile] = {}
            if source_ids:
                files = (
                    session.query(RepoFile)
                    .filter(
                        RepoFile.repo_hash == repo_hash,
                        RepoFile.file_id.in_(sorted(source_ids)),
                    )
                    .all()
                )
                source_files = {file.file_id: file for file in files}
            return {
                "page": {
                    "page_id": page.page_id,
                    "repo_hash": page.repo_hash,
                    "title": page.title,
                    "meta": page.get_meta(),
                    "created_at": page.created_at,
                    "updated_at": page.updated_at,
                },
                "contents": [
                    {
                        "content_id": c.content_id,
                        "page_id": c.page_id,
                        "content_type": c.content_type,
                        "content": c.content,
                        "meta": c.get_meta(),
                        "title": (c.get_meta() or {}).get("title") or "",
                        "sources": [
                            {
                                "file_id": file_id,
                                "file_name": source_files[file_id].file_name,
                                "file_path": source_files[file_id].file_path,
                            }
                            for file_id in (c.get_meta() or {}).get("source_file_ids", [])
                            if file_id in source_files
                        ],
                        "created_at": c.created_at,
                        "updated_at": c.updated_at,
                    }
                    for c in contents
                ],
            }
