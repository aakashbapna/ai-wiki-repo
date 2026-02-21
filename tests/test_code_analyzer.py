"""Tests for repo_analyzer.services.file.code_analyzer."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from repo_analyzer.models.index_task import IndexTask, TaskType
from repo_analyzer.models.repo_file import RepoFile
from repo_analyzer.services.file.code_analyzer import (
    FileForIndex,
    FileSummary,
    _apply_summaries,
    _batch,
    _build_file_payloads,
    _build_user_prompt,
    _ensure_string_list,
    _filter_files_to_index,
    _normalize_summary,
    _parse_json_list,
    _read_file_text,
    index_file,
    index_repo,
)
from tests.conftest import (
    make_index_task,
    make_openai_response,
    make_repo,
    make_repo_file,
)


# ── _parse_json_list ────────────────────────────────────────────────────────

class TestParseJsonList:
    def test_parses_json_array(self) -> None:
        result = _parse_json_list('[{"a": 1}, {"b": 2}]')
        assert result == [{"a": 1}, {"b": 2}]

    def test_wraps_single_dict_in_list(self) -> None:
        result = _parse_json_list('{"file_path": "foo.py"}')
        assert result == [{"file_path": "foo.py"}]

    def test_filters_non_dicts_from_array(self) -> None:
        result = _parse_json_list('[{"a": 1}, "string", 42]')
        assert result == [{"a": 1}]

    def test_raises_on_non_list_non_dict(self) -> None:
        with pytest.raises(ValueError, match="Expected JSON list"):
            _parse_json_list('"just a string"')

    def test_raises_on_invalid_json(self) -> None:
        with pytest.raises(Exception):
            _parse_json_list("not json at all")


# ── _normalize_summary ──────────────────────────────────────────────────────

class TestNormalizeSummary:
    def test_full_valid_item(self) -> None:
        item = {
            "file_path": "src/app.py",
            "responsibility": "Entry point",
            "key_elements": ["main", "app"],
            "dependent_files": ["utils.py"],
            "entry_point": True,
        }
        result = _normalize_summary(item, default_file_path="fallback.py")
        assert result["file_path"] == "src/app.py"
        assert result["responsibility"] == "Entry point"
        assert result["key_elements"] == ["main", "app"]
        assert result["dependent_files"] == ["utils.py"]
        assert result["entry_point"] is True

    def test_uses_default_file_path_when_missing(self) -> None:
        result = _normalize_summary({}, default_file_path="fallback.py")
        assert result["file_path"] == "fallback.py"

    def test_coerces_string_key_elements_to_list(self) -> None:
        item = {"key_elements": "single_element"}
        result = _normalize_summary(item, default_file_path="x.py")
        assert result["key_elements"] == ["single_element"]

    def test_defaults_entry_point_to_false(self) -> None:
        result = _normalize_summary({}, default_file_path="x.py")
        assert result["entry_point"] is False


# ── _ensure_string_list ─────────────────────────────────────────────────────

class TestEnsureStringList:
    def test_list_of_strings(self) -> None:
        assert _ensure_string_list(["a", "b"]) == ["a", "b"]

    def test_list_of_mixed_types_coerced(self) -> None:
        assert _ensure_string_list([1, 2.0]) == ["1", "2.0"]

    def test_single_string_wrapped(self) -> None:
        assert _ensure_string_list("hello") == ["hello"]

    def test_empty_string_returns_empty(self) -> None:
        assert _ensure_string_list("") == []

    def test_none_returns_empty(self) -> None:
        assert _ensure_string_list(None) == []

    def test_non_string_non_list_returns_empty(self) -> None:
        assert _ensure_string_list(42) == []


# ── _batch ──────────────────────────────────────────────────────────────────

class TestBatch:
    def test_splits_evenly(self) -> None:
        items = list(range(9))
        batches = list(_batch(items, 3))  # type: ignore[arg-type]
        assert len(batches) == 3
        assert batches[0] == [0, 1, 2]

    def test_last_batch_smaller(self) -> None:
        items = list(range(5))
        batches = list(_batch(items, 3))  # type: ignore[arg-type]
        assert batches[-1] == [3, 4]

    def test_batch_size_larger_than_list(self) -> None:
        items = [1, 2]
        batches = list(_batch(items, 10))  # type: ignore[arg-type]
        assert batches == [[1, 2]]

    def test_batch_size_zero_treated_as_one(self) -> None:
        items = [1, 2, 3]
        batches = list(_batch(items, 0))  # type: ignore[arg-type]
        assert len(batches) == 3


# ── _read_file_text ─────────────────────────────────────────────────────────

class TestReadFileText:
    def test_reads_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.py"
        f.write_text("print('hello')")
        assert _read_file_text(f) == "print('hello')"

    def test_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        assert _read_file_text(tmp_path / "nope.py") == ""

    def test_truncates_at_max_bytes(self, tmp_path: Path) -> None:
        f = tmp_path / "big.py"
        f.write_bytes(b"x" * 300)
        result = _read_file_text(f, max_bytes=100)
        assert len(result) == 100

    def test_replaces_invalid_utf8(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.py"
        f.write_bytes(b"\xff\xfe")
        result = _read_file_text(f)
        assert isinstance(result, str)


# ── _build_user_prompt ──────────────────────────────────────────────────────

class TestBuildUserPrompt:
    def test_contains_file_path_and_content(self) -> None:
        files = [FileForIndex(file_path="foo.py", file_name="foo.py", content="x = 1")]
        prompt = _build_user_prompt(files)
        assert "foo.py" in prompt
        assert "x = 1" in prompt

    def test_contains_multiple_files(self) -> None:
        files = [
            FileForIndex(file_path="a.py", file_name="a.py", content="a"),
            FileForIndex(file_path="b.py", file_name="b.py", content="b"),
        ]
        prompt = _build_user_prompt(files)
        assert "FILE 1 PATH" in prompt
        assert "FILE 2 PATH" in prompt


# ── _filter_files_to_index ──────────────────────────────────────────────────

class TestFilterFilesToIndex:
    def test_returns_unindexed_files_when_no_task_created_at(
        self, session, engine
    ) -> None:
        repo = make_repo(session)
        f1 = make_repo_file(session, repo, file_name="a.py", last_index_at=0)
        f2 = make_repo_file(session, repo, file_name="b.py", last_index_at=int(time.time()))
        result = _filter_files_to_index([f1, f2], task_created_at=None)
        assert f1 in result
        assert f2 not in result

    def test_returns_files_indexed_before_task_created_at(
        self, session, engine
    ) -> None:
        repo = make_repo(session)
        old_ts = int(time.time()) - 100
        f1 = make_repo_file(session, repo, file_name="old.py", last_index_at=old_ts)
        task_created_at = int(time.time())
        result = _filter_files_to_index([f1], task_created_at=task_created_at)
        assert f1 in result

    def test_excludes_recently_indexed_files(self, session, engine) -> None:
        repo = make_repo(session)
        now = int(time.time())
        f1 = make_repo_file(session, repo, file_name="new.py", last_index_at=now + 10)
        result = _filter_files_to_index([f1], task_created_at=now)
        assert f1 not in result


# ── _apply_summaries ────────────────────────────────────────────────────────

class TestApplySummaries:
    def test_sets_metadata_and_last_index_at(self, session, engine) -> None:
        repo = make_repo(session)
        rf = make_repo_file(session, repo, file_name="x.py")
        task = make_index_task(session, repo, total_files=1, completed_files=0)

        summaries: list[FileSummary] = [{
            "file_path": "src/x.py",
            "responsibility": "does stuff",
            "key_elements": ["X"],
            "dependent_files": [],
            "entry_point": False,
        }]
        rf.file_path = "src"
        rf.file_name = "x.py"
        session.flush()

        _apply_summaries(session, task, [rf], summaries)

        assert rf.last_index_at > 0
        assert rf.get_metadata() is not None
        assert rf.get_metadata()["responsibility"] == "does stuff"
        assert task.completed_files == 1

    def test_skips_file_with_no_matching_summary(self, session, engine) -> None:
        repo = make_repo(session)
        rf = make_repo_file(session, repo, file_name="z.py")
        task = make_index_task(session, repo, total_files=1, completed_files=0)

        _apply_summaries(session, task, [rf], [])  # empty summaries

        assert task.completed_files == 0
        assert rf.last_index_at == 0


# ── index_file ──────────────────────────────────────────────────────────────

class TestIndexFile:
    def test_calls_openai_and_returns_summaries(self, monkeypatch) -> None:
        response_json = json.dumps([{
            "file_path": "app.py",
            "responsibility": "main entry",
            "key_elements": ["main"],
            "dependent_files": [],
            "entry_point": True,
        }])
        mock_client = MagicMock()
        mock_client.responses.create.return_value = make_openai_response(response_json)
        monkeypatch.setattr(
            "repo_analyzer.services.file.code_analyzer._get_openai_client",
            lambda: mock_client,
        )

        files = [FileForIndex(file_path="app.py", file_name="app.py", content="x=1")]
        results = index_file(files)

        assert len(results) == 1
        assert results[0]["responsibility"] == "main entry"
        assert results[0]["entry_point"] is True

    def test_returns_empty_list_for_no_files(self) -> None:
        assert index_file([]) == []

    def test_raises_when_summary_count_mismatches(self, monkeypatch) -> None:
        response_json = json.dumps([
            {"file_path": "a.py", "responsibility": "r", "key_elements": [], "dependent_files": [], "entry_point": False},
            {"file_path": "b.py", "responsibility": "r", "key_elements": [], "dependent_files": [], "entry_point": False},
        ])
        mock_client = MagicMock()
        mock_client.responses.create.return_value = make_openai_response(response_json)
        monkeypatch.setattr(
            "repo_analyzer.services.file.code_analyzer._get_openai_client",
            lambda: mock_client,
        )
        files = [FileForIndex(file_path="a.py", file_name="a.py", content="x")]
        with pytest.raises(ValueError, match="Expected 1 summaries"):
            index_file(files)


# ── index_repo ──────────────────────────────────────────────────────────────

class TestIndexRepo:
    def test_raises_when_repo_not_found(self, session, mock_adapter) -> None:
        with pytest.raises(ValueError, match="Repo not found"):
            index_repo("nonexistent_hash")

    def test_raises_on_empty_repo_hash(self, session, mock_adapter) -> None:
        with pytest.raises(ValueError, match="repo_hash is required"):
            index_repo("   ")

    def test_returns_existing_running_task_status(self, session, mock_adapter) -> None:
        repo = make_repo(session)
        task = make_index_task(session, repo, status="running", total_files=5, completed_files=2)
        session.commit()

        result = index_repo(repo.repo_hash)

        assert result["status"] == "running"
        assert result["total_files"] == 5
        assert result["task_id"] == task.task_id

    def test_creates_completed_task_when_no_files_to_index(
        self, session, mock_adapter
    ) -> None:
        repo = make_repo(session)
        # File already indexed
        make_repo_file(session, repo, file_name="done.py", last_index_at=int(time.time()))
        session.commit()

        result = index_repo(repo.repo_hash)

        assert result["status"] == "completed"
        assert result["total_files"] == 0

    def test_indexes_files_and_returns_completed_status(
        self, session, mock_adapter, monkeypatch, tmp_path
    ) -> None:
        repo = make_repo(session, clone_path=str(tmp_path))
        # Create the actual file on disk
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("x = 1")

        rf = make_repo_file(session, repo, file_path="src", file_name="app.py", last_index_at=0)
        session.commit()

        response_json = json.dumps([{
            "file_path": "src/app.py",
            "responsibility": "app entry",
            "key_elements": ["x"],
            "dependent_files": [],
            "entry_point": True,
        }])
        mock_client = MagicMock()
        mock_client.responses.create.return_value = make_openai_response(response_json)
        monkeypatch.setattr(
            "repo_analyzer.services.file.code_analyzer._get_openai_client",
            lambda: mock_client,
        )

        result = index_repo(repo.repo_hash, batch_size=8)

        assert result["status"] == "completed"
        assert result["completed_files"] == 1
        assert result["remaining_files"] == 0

    def test_marks_task_failed_on_openai_error(
        self, session, mock_adapter, monkeypatch, tmp_path
    ) -> None:
        repo = make_repo(session, clone_path=str(tmp_path))
        (tmp_path / "err.py").write_text("boom")
        make_repo_file(session, repo, file_path=".", file_name="err.py", last_index_at=0)
        session.commit()

        mock_client = MagicMock()
        mock_client.responses.create.side_effect = RuntimeError("OpenAI down")
        monkeypatch.setattr(
            "repo_analyzer.services.file.code_analyzer._get_openai_client",
            lambda: mock_client,
        )

        with pytest.raises(RuntimeError, match="OpenAI down"):
            index_repo(repo.repo_hash)

        # Task should be marked failed in DB
        task = session.query(IndexTask).filter(
            IndexTask.repo_hash == repo.repo_hash,
            IndexTask.task_type == TaskType.INDEX_FILE.value,
        ).first()
        assert task is not None
        assert task.status == "failed"
        assert "OpenAI down" in task.last_error
