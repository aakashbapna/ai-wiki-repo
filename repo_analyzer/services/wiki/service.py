"""Service for wiki operations."""

from typing import TypedDict

from repo_analyzer.db import get_default_adapter
from repo_analyzer.db_managers import RepoManager, WikiManager
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
                        "created_at": c.created_at,
                        "updated_at": c.updated_at,
                    }
                    for c in contents
                ],
            }
