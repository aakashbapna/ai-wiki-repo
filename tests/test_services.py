"""Tests for FileService, SubsystemService, and WikiService."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from repo_analyzer.models.index_task import IndexTask, TaskType
from repo_analyzer.models.wiki_page import WikiPage
from repo_analyzer.models.wiki_page_content import WikiPageContent
from repo_analyzer.models.wiki_sidebar import WikiSidebar
from repo_analyzer.services.file.service import FileService
from repo_analyzer.services.subsystem.service import SubsystemService
from repo_analyzer.services.wiki.service import WikiService
from tests.conftest import (
    make_index_task,
    make_openai_response,
    make_repo,
    make_repo_file,
    make_subsystem,
    make_wiki_page,
)


# ── FileService.list_repo_files ─────────────────────────────────────────────

class TestFileServiceListRepoFiles:
    def test_raises_when_repo_not_found(self, session, mock_adapter) -> None:
        with pytest.raises(ValueError, match="Repo not found"):
            FileService.list_repo_files("no_hash", project_only=False)

    def test_returns_all_non_excluded_files(self, session, mock_adapter) -> None:
        repo = make_repo(session)
        make_repo_file(session, repo, file_name="app.py", is_scan_excluded=False)
        make_repo_file(session, repo, file_name="image.png", is_scan_excluded=True)
        session.commit()

        result = FileService.list_repo_files(repo.repo_hash, project_only=False)

        names = [f["file_name"] for f in result["files"]]
        assert "app.py" in names
        assert "image.png" not in names
        assert result["total"] == 1

    def test_project_only_returns_only_project_files(self, session, mock_adapter) -> None:
        repo = make_repo(session)
        pf = make_repo_file(session, repo, file_name="package.json", is_scan_excluded=False)
        pf.is_project_file = True
        rf = make_repo_file(session, repo, file_name="app.py", is_scan_excluded=False)
        rf.is_project_file = False
        session.flush()
        session.commit()

        result = FileService.list_repo_files(repo.repo_hash, project_only=True)

        names = [f["file_name"] for f in result["files"]]
        assert "package.json" in names
        assert "app.py" not in names

    def test_result_contains_expected_fields(self, session, mock_adapter) -> None:
        repo = make_repo(session)
        make_repo_file(session, repo, file_name="utils.py")
        session.commit()

        result = FileService.list_repo_files(repo.repo_hash, project_only=False)

        assert result["repo_hash"] == repo.repo_hash
        assert isinstance(result["total"], int)
        file_entry = result["files"][0]
        for key in ("file_id", "file_path", "file_name", "is_project_file", "is_scan_excluded", "last_index_at"):
            assert key in file_entry


# ── FileService.get_index_task_status ───────────────────────────────────────

class TestFileServiceGetIndexTaskStatus:
    def test_returns_none_when_no_task_exists(self, session, mock_adapter) -> None:
        repo = make_repo(session)
        session.commit()

        result = FileService.get_index_task_status(repo.repo_hash)
        assert result is None

    def test_returns_latest_task_status(self, session, mock_adapter) -> None:
        repo = make_repo(session)
        make_index_task(session, repo, status="completed", total_files=5, completed_files=5)
        # Give task2 a clearly later created_at so ORDER BY created_at DESC picks it first
        task2 = make_index_task(session, repo, status="running", total_files=3, completed_files=1)
        task2.created_at = int(time.time()) + 10
        session.commit()

        result = FileService.get_index_task_status(repo.repo_hash)

        assert result is not None
        assert result["status"] == "running"
        assert result["task_id"] == task2.task_id

    def test_status_includes_remaining_files(self, session, mock_adapter) -> None:
        repo = make_repo(session)
        make_index_task(session, repo, status="running", total_files=10, completed_files=3)
        session.commit()

        result = FileService.get_index_task_status(repo.repo_hash)

        assert result["remaining_files"] == 7


# ── FileService.stop_indexing ───────────────────────────────────────────────

class TestFileServiceStopIndexing:
    def test_raises_when_repo_not_found(self, session, mock_adapter) -> None:
        with pytest.raises(ValueError, match="Repo not found"):
            FileService.stop_indexing("no_hash")

    def test_stops_all_running_tasks(self, session, mock_adapter) -> None:
        repo = make_repo(session)
        t1 = make_index_task(session, repo, status="running")
        t2 = make_index_task(session, repo, status="running")
        make_index_task(session, repo, status="completed")
        session.commit()

        result = FileService.stop_indexing(repo.repo_hash)

        assert result["stopped_tasks"] == 2
        session.expire_all()
        assert session.get(IndexTask, t1.task_id).status == "stopped"
        assert session.get(IndexTask, t2.task_id).status == "stopped"

    def test_returns_zero_when_no_running_tasks(self, session, mock_adapter) -> None:
        repo = make_repo(session)
        make_index_task(session, repo, status="completed")
        session.commit()

        result = FileService.stop_indexing(repo.repo_hash)

        assert result["stopped_tasks"] == 0


# ── FileService.reindex_single_file ─────────────────────────────────────────

class TestFileServiceReindexSingleFile:
    def test_raises_when_repo_not_found(self, session, mock_adapter) -> None:
        with pytest.raises(ValueError, match="Repo not found"):
            FileService.reindex_single_file("no_hash", 1)

    def test_raises_when_file_not_found(self, session, mock_adapter) -> None:
        repo = make_repo(session)
        session.commit()

        with pytest.raises(ValueError, match="File not found"):
            FileService.reindex_single_file(repo.repo_hash, 9999)

    def test_returns_updated_metadata(
        self, session, mock_adapter, monkeypatch, tmp_path
    ) -> None:
        repo = make_repo(session, clone_path=str(tmp_path))
        (tmp_path / "app.py").write_text("x = 1")
        rf = make_repo_file(session, repo, file_path=".", file_name="app.py")
        session.commit()

        response_payload = json.dumps([{
            "file_path": "./app.py",
            "responsibility": "main app",
            "key_elements": ["x"],
            "dependent_files": [],
            "entry_point": True,
        }])
        mock_client = MagicMock()
        mock_client.responses.create.return_value = make_openai_response(response_payload)
        monkeypatch.setattr(
            "repo_analyzer.services.file.code_analyzer._get_openai_client",
            lambda: mock_client,
        )

        # Mock WikiService.build_wiki so it doesn't cascade
        monkeypatch.setattr(
            "repo_analyzer.services.file.service.WikiService.build_wiki",
            lambda _: {"status": "completed", "total_files": 0, "completed_files": 0,
                       "remaining_files": 0, "task_id": 1, "repo_hash": repo.repo_hash},
        )

        result = FileService.reindex_single_file(repo.repo_hash, rf.file_id)

        assert result["repo_hash"] == repo.repo_hash
        assert result["file_id"] == rf.file_id
        assert result["metadata"]["responsibility"] == "main app"


# ── SubsystemService.list_subsystems ────────────────────────────────────────

class TestSubsystemServiceListSubsystems:
    def test_raises_when_repo_not_found(self, session, mock_adapter) -> None:
        with pytest.raises(ValueError, match="Repo not found"):
            SubsystemService.list_subsystems("no_hash")

    def test_returns_empty_list_when_no_subsystems(self, session, mock_adapter) -> None:
        repo = make_repo(session)
        session.commit()

        result = SubsystemService.list_subsystems(repo.repo_hash)
        assert result == []

    def test_returns_subsystem_data(self, session, mock_adapter) -> None:
        repo = make_repo(session)
        make_subsystem(session, repo, name="Auth", description="Handles auth",
                       file_ids=[1, 2], keywords=["jwt"])
        session.commit()

        result = SubsystemService.list_subsystems(repo.repo_hash)

        assert len(result) == 1
        sub = result[0]
        assert sub["name"] == "Auth"
        assert sub["description"] == "Handles auth"
        assert sub["meta"]["file_ids"] == [1, 2]
        assert sub["meta"]["keywords"] == ["jwt"]


# ── WikiService.list_sidebars ────────────────────────────────────────────────

class TestWikiServiceListSidebars:
    def test_raises_when_repo_not_found(self, session, mock_adapter) -> None:
        with pytest.raises(ValueError, match="Repo not found"):
            WikiService.list_sidebars("no_hash")

    def test_returns_empty_list_when_no_sidebars(self, session, mock_adapter) -> None:
        repo = make_repo(session)
        session.commit()

        result = WikiService.list_sidebars(repo.repo_hash)
        assert result == []

    def test_returns_sidebar_data(self, session, mock_adapter) -> None:
        repo = make_repo(session)
        page = make_wiki_page(session, repo, title="Auth Page")
        now = int(time.time())
        sidebar = WikiSidebar(
            repo_hash=repo.repo_hash,
            parent_node_id=None,
            name="Auth",
            page_id=page.page_id,
            is_active=True,
            meta_json=json.dumps({"sub_system_ids": [1]}),
            created_at=now,
            updated_at=now,
        )
        session.add(sidebar)
        session.commit()

        result = WikiService.list_sidebars(repo.repo_hash)

        assert len(result) == 1
        assert result[0]["name"] == "Auth"
        assert result[0]["is_active"] is True
        assert result[0]["page_id"] == page.page_id


# ── WikiService.list_pages ───────────────────────────────────────────────────

class TestWikiServiceListPages:
    def test_raises_when_repo_not_found(self, session, mock_adapter) -> None:
        with pytest.raises(ValueError, match="Repo not found"):
            WikiService.list_pages("no_hash")

    def test_returns_all_pages(self, session, mock_adapter) -> None:
        repo = make_repo(session)
        make_wiki_page(session, repo, title="Page 1")
        make_wiki_page(session, repo, title="Page 2")
        session.commit()

        result = WikiService.list_pages(repo.repo_hash)

        assert len(result) == 2
        titles = {p["title"] for p in result}
        assert titles == {"Page 1", "Page 2"}


# ── WikiService.get_page_with_contents ──────────────────────────────────────

class TestWikiServiceGetPageWithContents:
    def test_raises_when_repo_not_found(self, session, mock_adapter) -> None:
        with pytest.raises(ValueError, match="Repo not found"):
            WikiService.get_page_with_contents("no_hash", 1)

    def test_raises_when_page_not_found(self, session, mock_adapter) -> None:
        repo = make_repo(session)
        session.commit()

        with pytest.raises(ValueError, match="Page not found"):
            WikiService.get_page_with_contents(repo.repo_hash, 9999)

    def test_returns_page_and_contents(self, session, mock_adapter) -> None:
        repo = make_repo(session)
        page = make_wiki_page(session, repo, title="Auth Docs")
        now = int(time.time())
        content = WikiPageContent(
            page_id=page.page_id,
            content_type="markdown",
            content="# Auth\nJWT details",
            meta_json=json.dumps({"source_file_ids": [1]}),
            created_at=now,
            updated_at=now,
        )
        session.add(content)
        session.commit()

        result = WikiService.get_page_with_contents(repo.repo_hash, page.page_id)

        assert result["page"]["title"] == "Auth Docs"
        assert len(result["contents"]) == 1
        assert "JWT" in result["contents"][0]["content"]

    def test_returns_empty_contents_when_none_exist(self, session, mock_adapter) -> None:
        repo = make_repo(session)
        page = make_wiki_page(session, repo, title="Empty Page")
        session.commit()

        result = WikiService.get_page_with_contents(repo.repo_hash, page.page_id)

        assert result["page"]["title"] == "Empty Page"
        assert result["contents"] == []


# ── WikiService.build_wiki ───────────────────────────────────────────────────

class TestWikiServiceBuildWiki:
    def test_delegates_to_build_wiki_function(
        self, session, mock_adapter, monkeypatch
    ) -> None:
        repo = make_repo(session)
        session.commit()

        expected = {
            "repo_hash": repo.repo_hash,
            "status": "completed",
            "total_files": 0,
            "completed_files": 0,
            "remaining_files": 0,
            "task_id": 42,
        }
        monkeypatch.setattr(
            "repo_analyzer.services.wiki.service.build_wiki",
            lambda repo_hash: expected,
        )

        result = WikiService.build_wiki(repo.repo_hash)
        assert result == expected
