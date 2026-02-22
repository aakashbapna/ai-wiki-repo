"""Tests for repo_analyzer.services.wiki.wiki_builder."""

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from repo_analyzer.models.index_task import IndexTask, TaskType
from repo_analyzer.models.wiki_page import WikiPage
from repo_analyzer.models.wiki_page_content import WikiPageContent
from repo_analyzer.models.wiki_sidebar import WikiSidebar
from repo_analyzer.services.wiki.wiki_builder import (
    _ensure_int_list,
    _normalize_page_spec,
    _parse_json_object,
    _read_file_text,
    build_wiki,
)
from tests.conftest import (
    make_index_task,
    make_repo,
    make_repo_file,
    make_subsystem,
    make_wiki_page,
)


# ── _parse_json_object ──────────────────────────────────────────────────────

class TestParseJsonObject:
    def test_parses_valid_dict(self) -> None:
        result = _parse_json_object('{"title": "Auth Overview"}')
        assert result == {"title": "Auth Overview"}

    def test_raises_on_list(self) -> None:
        with pytest.raises(ValueError, match="Expected JSON object"):
            _parse_json_object('[1, 2]')

    def test_raises_on_invalid_json(self) -> None:
        with pytest.raises(Exception):
            _parse_json_object("not json")


# ── _normalize_page_spec ────────────────────────────────────────────────────

class TestNormalizePageSpec:
    def test_full_valid_spec(self) -> None:
        raw = {
            "title": "Auth Module",
            "contents": [
                {
                    "content_type": "markdown",
                    "content": "# Auth\nHandles JWT",
                    "source_file_ids": [1, 2],
                }
            ],
        }
        spec = _normalize_page_spec(raw)
        assert spec["title"] == "Auth Module"
        assert len(spec["contents"]) == 1
        assert spec["contents"][0]["content_type"] == "markdown"
        assert spec["contents"][0]["source_file_ids"] == [1, 2]

    def test_defaults_title_when_missing(self) -> None:
        spec = _normalize_page_spec({})
        assert spec["title"] == "Subsystem Overview"

    def test_empty_contents_list(self) -> None:
        spec = _normalize_page_spec({"title": "T", "contents": []})
        assert spec["contents"] == []

    def test_skips_non_dict_content_nodes(self) -> None:
        raw = {"title": "T", "contents": [{"content_type": "markdown", "content": "x", "source_file_ids": []}, "bad"]}
        spec = _normalize_page_spec(raw)
        assert len(spec["contents"]) == 1

    def test_defaults_content_type_to_markdown(self) -> None:
        raw = {"title": "T", "contents": [{"content": "text", "source_file_ids": []}]}
        spec = _normalize_page_spec(raw)
        assert spec["contents"][0]["content_type"] == "markdown"


# ── _ensure_int_list ────────────────────────────────────────────────────────

class TestEnsureIntList:
    def test_list_of_ints(self) -> None:
        assert _ensure_int_list([1, 2, 3]) == [1, 2, 3]

    def test_converts_string_ints(self) -> None:
        assert _ensure_int_list(["1", "2"]) == [1, 2]

    def test_skips_unconvertible(self) -> None:
        assert _ensure_int_list(["1", "x"]) == [1]

    def test_non_list_returns_empty(self) -> None:
        assert _ensure_int_list(None) == []


# ── _read_file_text ─────────────────────────────────────────────────────────

class TestReadFileText:
    def test_reads_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "wiki.md"
        f.write_text("# Doc")
        assert _read_file_text(f) == "# Doc"

    def test_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        assert _read_file_text(tmp_path / "missing.md") == ""

    def test_truncates_to_max_bytes(self, tmp_path: Path) -> None:
        f = tmp_path / "big.txt"
        f.write_bytes(b"y" * 1000)
        result = _read_file_text(f, max_bytes=100)
        assert len(result) == 100


# ── build_wiki ──────────────────────────────────────────────────────────────

def _make_sync_thread(monkeypatch):
    """Patch threading.Thread so the target runs synchronously."""
    class SyncThread:
        def __init__(self, target=None, daemon=None, name=None, **kwargs):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    monkeypatch.setattr("repo_analyzer.services.wiki.wiki_builder.threading.Thread", SyncThread)


def _sidebar_tree_json(page_title: str, name: str) -> str:
    """Build a minimal valid sidebar tree JSON with Introduction + named node."""
    return json.dumps({
        "nodes": [
            {"name": "Introduction", "page_title": page_title, "children": []},
            {"name": name, "page_title": page_title, "children": []},
        ]
    })


class TestBuildWiki:
    def test_raises_on_empty_repo_hash(self, session, mock_adapter) -> None:
        with pytest.raises(ValueError, match="repo_hash is required"):
            build_wiki("  ")

    def test_raises_when_repo_not_found(self, session, mock_adapter) -> None:
        with pytest.raises(ValueError, match="Repo not found"):
            build_wiki("deadbeef")

    def test_returns_running_task_if_already_running(
        self, session, mock_adapter
    ) -> None:
        repo = make_repo(session)
        task = make_index_task(
            session, repo,
            task_type=TaskType.BUILD_WIKI.value,
            status="running",
            total_files=4,
        )
        session.commit()

        result = build_wiki(repo.repo_hash)

        assert result["status"] == "running"
        assert result["task_id"] == task.task_id

    def test_creates_completed_task_with_no_subsystems(
        self, session, mock_adapter, monkeypatch
    ) -> None:
        _make_sync_thread(monkeypatch)
        repo = make_repo(session)
        session.commit()

        build_wiki(repo.repo_hash)

        task = session.query(IndexTask).filter(
            IndexTask.repo_hash == repo.repo_hash,
            IndexTask.task_type == TaskType.BUILD_WIKI.value,
        ).first()
        assert task is not None
        assert task.status == "completed"
        assert task.total_files == 0

    def test_builds_wiki_page_per_subsystem(
        self, session, mock_adapter, monkeypatch, tmp_path
    ) -> None:
        _make_sync_thread(monkeypatch)
        repo = make_repo(session, clone_path=str(tmp_path))
        rf = make_repo_file(session, repo, file_path="src", file_name="auth.py")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "auth.py").write_text("# auth code")

        sub = make_subsystem(session, repo, name="Auth", file_ids=[rf.file_id])
        session.commit()

        response_payload = json.dumps({
            "title": "Auth Overview",
            "contents": [
                {
                    "content_type": "markdown",
                    "content": "# Auth\nJWT implementation",
                    "source_file_ids": [rf.file_id],
                }
            ],
        })
        sidebar_payload = _sidebar_tree_json("Auth Overview", "Auth")

        monkeypatch.setattr(
            "repo_analyzer.services.wiki.wiki_builder.stream_batch",
            lambda requests, **kwargs: [(0, response_payload)],
        )
        monkeypatch.setattr(
            "repo_analyzer.services.wiki.wiki_builder.run_batch",
            lambda requests, **kwargs: [sidebar_payload],
        )

        build_wiki(repo.repo_hash)

        task = session.query(IndexTask).filter(
            IndexTask.repo_hash == repo.repo_hash,
            IndexTask.task_type == TaskType.BUILD_WIKI.value,
        ).first()
        assert task is not None
        assert task.status == "completed"

        pages = session.query(WikiPage).filter(WikiPage.repo_hash == repo.repo_hash).all()
        assert len(pages) == 1
        assert pages[0].title == "Auth Overview"

        contents = session.query(WikiPageContent).filter(
            WikiPageContent.page_id == pages[0].page_id
        ).all()
        assert len(contents) == 1
        assert "JWT" in contents[0].content

    def test_creates_sidebar_nodes(
        self, session, mock_adapter, monkeypatch, tmp_path
    ) -> None:
        _make_sync_thread(monkeypatch)
        repo = make_repo(session, clone_path=str(tmp_path))
        rf = make_repo_file(session, repo, file_path=".", file_name="app.py")
        (tmp_path / "app.py").write_text("app = True")

        make_subsystem(session, repo, name="AppCore", file_ids=[rf.file_id])
        session.commit()

        response_payload = json.dumps({
            "title": "AppCore Overview",
            "contents": [{"content_type": "markdown", "content": "docs", "source_file_ids": [rf.file_id]}],
        })
        sidebar_payload = _sidebar_tree_json("AppCore Overview", "AppCore")

        monkeypatch.setattr(
            "repo_analyzer.services.wiki.wiki_builder.stream_batch",
            lambda requests, **kwargs: [(0, response_payload)],
        )
        monkeypatch.setattr(
            "repo_analyzer.services.wiki.wiki_builder.run_batch",
            lambda requests, **kwargs: [sidebar_payload],
        )

        build_wiki(repo.repo_hash)

        sidebars = session.query(WikiSidebar).filter(WikiSidebar.repo_hash == repo.repo_hash).all()
        assert len(sidebars) >= 1
        names = [s.name for s in sidebars]
        assert "AppCore" in names

    def test_no_sidebar_when_no_pages_generated(
        self, session, mock_adapter, monkeypatch, tmp_path
    ) -> None:
        """Empty subsystems produce no pages, so no sidebar is generated."""
        _make_sync_thread(monkeypatch)
        repo = make_repo(session, clone_path=str(tmp_path))
        make_subsystem(session, repo, name="Empty", file_ids=[])
        session.commit()

        build_wiki(repo.repo_hash)

        task = session.query(IndexTask).filter(
            IndexTask.repo_hash == repo.repo_hash,
            IndexTask.task_type == TaskType.BUILD_WIKI.value,
        ).first()
        assert task is not None
        assert task.status == "completed"

        sidebars = session.query(WikiSidebar).filter(WikiSidebar.repo_hash == repo.repo_hash).all()
        assert len(sidebars) == 0

    def test_deletes_existing_wiki_before_rebuild(
        self, session, mock_adapter, monkeypatch, tmp_path
    ) -> None:
        _make_sync_thread(monkeypatch)
        repo = make_repo(session, clone_path=str(tmp_path))
        # Old wiki page that should be wiped
        make_wiki_page(session, repo, title="OldPage")
        rf = make_repo_file(session, repo, file_path=".", file_name="new.py")
        (tmp_path / "new.py").write_text("x = 1")
        make_subsystem(session, repo, name="NewSub", file_ids=[rf.file_id])
        session.commit()

        response_payload = json.dumps({
            "title": "NewPage",
            "contents": [{"content_type": "markdown", "content": "new content", "source_file_ids": [rf.file_id]}],
        })
        sidebar_payload = _sidebar_tree_json("NewPage", "NewSub")

        monkeypatch.setattr(
            "repo_analyzer.services.wiki.wiki_builder.stream_batch",
            lambda requests, **kwargs: [(0, response_payload)],
        )
        monkeypatch.setattr(
            "repo_analyzer.services.wiki.wiki_builder.run_batch",
            lambda requests, **kwargs: [sidebar_payload],
        )

        build_wiki(repo.repo_hash)

        pages = session.query(WikiPage).filter(WikiPage.repo_hash == repo.repo_hash).all()
        titles = [p.title for p in pages]
        assert "OldPage" not in titles
        assert "NewPage" in titles

    def test_marks_task_failed_on_openai_error(
        self, session, mock_adapter, monkeypatch, tmp_path
    ) -> None:
        _make_sync_thread(monkeypatch)
        repo = make_repo(session, clone_path=str(tmp_path))
        rf = make_repo_file(session, repo, file_path=".", file_name="x.py")
        (tmp_path / "x.py").write_text("x = 1")
        make_subsystem(session, repo, name="Sub", file_ids=[rf.file_id])
        session.commit()

        error = RuntimeError("wiki LLM error")
        monkeypatch.setattr(
            "repo_analyzer.services.wiki.wiki_builder.stream_batch",
            lambda requests, **kwargs: (_ for _ in ()).throw(error),
        )

        # build_wiki doesn't raise — error is caught in background thread
        build_wiki(repo.repo_hash)

        task = session.query(IndexTask).filter(
            IndexTask.repo_hash == repo.repo_hash,
            IndexTask.task_type == TaskType.BUILD_WIKI.value,
        ).first()
        assert task is not None
        assert task.status == "failed"
        assert "wiki LLM error" in task.last_error

    def test_task_completed_files_matches_file_count(
        self, session, mock_adapter, monkeypatch, tmp_path
    ) -> None:
        _make_sync_thread(monkeypatch)
        repo = make_repo(session, clone_path=str(tmp_path))
        rf1 = make_repo_file(session, repo, file_path=".", file_name="a.py")
        rf2 = make_repo_file(session, repo, file_path=".", file_name="b.py")
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        make_subsystem(session, repo, name="Sub", file_ids=[rf1.file_id, rf2.file_id])
        session.commit()

        response_payload = json.dumps({
            "title": "Sub Overview",
            "contents": [{"content_type": "markdown", "content": "docs", "source_file_ids": [rf1.file_id]}],
        })
        sidebar_payload = _sidebar_tree_json("Sub Overview", "Sub")

        monkeypatch.setattr(
            "repo_analyzer.services.wiki.wiki_builder.stream_batch",
            lambda requests, **kwargs: [(0, response_payload)],
        )
        monkeypatch.setattr(
            "repo_analyzer.services.wiki.wiki_builder.run_batch",
            lambda requests, **kwargs: [sidebar_payload],
        )

        build_wiki(repo.repo_hash)

        task = session.query(IndexTask).filter(
            IndexTask.repo_hash == repo.repo_hash,
            IndexTask.task_type == TaskType.BUILD_WIKI.value,
        ).first()
        assert task is not None
        assert task.completed_files == 2  # 2 files in the subsystem
        assert task.status == "completed"
