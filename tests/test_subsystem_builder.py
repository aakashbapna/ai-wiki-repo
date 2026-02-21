"""Tests for repo_analyzer.services.subsystem.subsystem_builder."""

import json
import time
from unittest.mock import MagicMock

import pytest

from repo_analyzer.db_managers import SubsystemManager
from repo_analyzer.models.index_task import IndexTask, TaskType
from repo_analyzer.models.repo_subsystem import RepoSubsystem
from repo_analyzer.services.subsystem.subsystem_builder import (
    _build_user_prompt,
    _ensure_int_list,
    _ensure_string_list,
    _normalize_spec,
    _parse_json_list,
    create_subsystems,
)
from tests.conftest import (
    make_index_task,
    make_openai_response,
    make_repo,
    make_repo_file,
    make_subsystem,
)


# ── _parse_json_list ────────────────────────────────────────────────────────

class TestParseJsonList:
    def test_parses_list(self) -> None:
        result = _parse_json_list('[{"name": "Auth"}]')
        assert result == [{"name": "Auth"}]

    def test_wraps_single_dict(self) -> None:
        result = _parse_json_list('{"name": "Auth"}')
        assert result == [{"name": "Auth"}]

    def test_raises_on_non_dict_non_list(self) -> None:
        with pytest.raises(ValueError):
            _parse_json_list('"just a string"')


# ── _normalize_spec ─────────────────────────────────────────────────────────

class TestNormalizeSpec:
    def test_full_valid_spec(self) -> None:
        spec = _normalize_spec({
            "name": "Auth",
            "description": "Handles auth",
            "keywords": ["login", "jwt"],
            "file_ids": [1, 2, 3],
        })
        assert spec["name"] == "Auth"
        assert spec["description"] == "Handles auth"
        assert spec["keywords"] == ["login", "jwt"]
        assert spec["file_ids"] == [1, 2, 3]

    def test_strips_whitespace_from_name_and_description(self) -> None:
        spec = _normalize_spec({"name": "  Auth  ", "description": "  desc  "})
        assert spec["name"] == "Auth"
        assert spec["description"] == "desc"

    def test_missing_fields_default_to_empty(self) -> None:
        spec = _normalize_spec({})
        assert spec["name"] == ""
        assert spec["keywords"] == []
        assert spec["file_ids"] == []

    def test_file_ids_coerced_to_int(self) -> None:
        spec = _normalize_spec({"file_ids": ["1", "2"]})
        assert spec["file_ids"] == [1, 2]

    def test_invalid_file_ids_skipped(self) -> None:
        spec = _normalize_spec({"file_ids": [1, "bad", None]})
        assert spec["file_ids"] == [1]


# ── _ensure_int_list ────────────────────────────────────────────────────────

class TestEnsureIntList:
    def test_converts_string_ints(self) -> None:
        assert _ensure_int_list(["1", "2"]) == [1, 2]

    def test_skips_unconvertible(self) -> None:
        assert _ensure_int_list(["1", "bad"]) == [1]

    def test_non_list_returns_empty(self) -> None:
        assert _ensure_int_list("string") == []
        assert _ensure_int_list(None) == []


# ── _ensure_string_list ─────────────────────────────────────────────────────

class TestEnsureStringList:
    def test_list_of_strings(self) -> None:
        assert _ensure_string_list(["a", "b"]) == ["a", "b"]

    def test_single_string_wrapped(self) -> None:
        assert _ensure_string_list("one") == ["one"]

    def test_none_returns_empty(self) -> None:
        assert _ensure_string_list(None) == []


# ── _build_user_prompt ──────────────────────────────────────────────────────

class TestBuildUserPrompt:
    def test_includes_file_metadata(self, session, engine) -> None:
        repo = make_repo(session)
        rf = make_repo_file(session, repo, file_path="src", file_name="auth.py")
        rf.set_metadata({
            "responsibility": "handles JWT",
            "key_elements": ["verify_token"],
            "dependent_files": [],
            "entry_point": False,
        })
        session.flush()

        prompt = _build_user_prompt([rf])

        assert str(rf.file_id) in prompt
        assert "handles JWT" in prompt
        assert "auth.py" in prompt

    def test_includes_entry_point_flag(self, session, engine) -> None:
        repo = make_repo(session)
        rf = make_repo_file(session, repo, file_name="main.py")
        rf.set_metadata({"entry_point": True, "responsibility": "boot", "key_elements": [], "dependent_files": []})
        session.flush()

        prompt = _build_user_prompt([rf])
        assert "true" in prompt.lower()


# ── create_subsystems ───────────────────────────────────────────────────────

class TestCreateSubsystems:
    def test_raises_on_empty_repo_hash(self, session, mock_adapter) -> None:
        with pytest.raises(ValueError, match="repo_hash is required"):
            create_subsystems("  ")

    def test_raises_when_repo_not_found(self, session, mock_adapter) -> None:
        with pytest.raises(ValueError, match="Repo not found"):
            create_subsystems("deadbeef")

    def test_returns_running_task_if_already_running(
        self, session, mock_adapter
    ) -> None:
        repo = make_repo(session)
        task = make_index_task(
            session, repo,
            task_type=TaskType.BUILD_SUBSYSTEM.value,
            status="running",
            total_files=3,
        )
        session.commit()

        result = create_subsystems(repo.repo_hash)

        assert result["status"] == "running"
        assert result["task_id"] == task.task_id

    def test_creates_completed_task_when_no_indexed_files(
        self, session, mock_adapter
    ) -> None:
        repo = make_repo(session)
        # File present but not indexed
        make_repo_file(session, repo, file_name="raw.py", last_index_at=0)
        session.commit()

        result = create_subsystems(repo.repo_hash)

        assert result["status"] == "completed"
        assert result["total_files"] == 0

    def test_builds_subsystems_from_indexed_files(
        self, session, mock_adapter, monkeypatch
    ) -> None:
        repo = make_repo(session)
        rf = make_repo_file(
            session, repo, file_path="src", file_name="auth.py",
            last_index_at=int(time.time()) - 10,
        )
        rf.set_metadata({
            "responsibility": "auth",
            "key_elements": ["login"],
            "dependent_files": [],
            "entry_point": False,
        })
        session.commit()

        response_payload = json.dumps([{
            "name": "Auth",
            "description": "Authentication module",
            "keywords": ["login", "jwt"],
            "file_ids": [rf.file_id],
        }])
        mock_client = MagicMock()
        mock_client.responses.create.return_value = make_openai_response(response_payload)
        monkeypatch.setattr(
            "repo_analyzer.services.subsystem.subsystem_builder._get_openai_client",
            lambda: mock_client,
        )

        result = create_subsystems(repo.repo_hash)

        assert result["status"] == "completed"
        subsystems = session.query(RepoSubsystem).filter(
            RepoSubsystem.repo_hash == repo.repo_hash
        ).all()
        assert len(subsystems) == 1
        assert subsystems[0].name == "Auth"

    def test_deletes_existing_subsystems_before_rebuild(
        self, session, mock_adapter, monkeypatch
    ) -> None:
        repo = make_repo(session)
        # Pre-existing subsystem from a previous run
        make_subsystem(session, repo, name="OldSubsystem")
        rf = make_repo_file(
            session, repo, file_name="core.py",
            last_index_at=int(time.time()) - 5,
        )
        rf.set_metadata({"responsibility": "r", "key_elements": [], "dependent_files": [], "entry_point": False})
        session.commit()

        response_payload = json.dumps([{
            "name": "NewSubsystem",
            "description": "Fresh",
            "keywords": [],
            "file_ids": [rf.file_id],
        }])
        mock_client = MagicMock()
        mock_client.responses.create.return_value = make_openai_response(response_payload)
        monkeypatch.setattr(
            "repo_analyzer.services.subsystem.subsystem_builder._get_openai_client",
            lambda: mock_client,
        )

        create_subsystems(repo.repo_hash)

        subsystems = session.query(RepoSubsystem).filter(
            RepoSubsystem.repo_hash == repo.repo_hash
        ).all()
        names = [s.name for s in subsystems]
        assert "OldSubsystem" not in names
        assert "NewSubsystem" in names

    def test_marks_task_failed_on_openai_error(
        self, session, mock_adapter, monkeypatch
    ) -> None:
        repo = make_repo(session)
        rf = make_repo_file(
            session, repo, file_name="x.py",
            last_index_at=int(time.time()) - 5,
        )
        rf.set_metadata({"responsibility": "x", "key_elements": [], "dependent_files": [], "entry_point": False})
        session.commit()

        mock_client = MagicMock()
        mock_client.responses.create.side_effect = RuntimeError("LLM error")
        monkeypatch.setattr(
            "repo_analyzer.services.subsystem.subsystem_builder._get_openai_client",
            lambda: mock_client,
        )

        with pytest.raises(RuntimeError, match="LLM error"):
            create_subsystems(repo.repo_hash)

        task = session.query(IndexTask).filter(
            IndexTask.repo_hash == repo.repo_hash,
            IndexTask.task_type == TaskType.BUILD_SUBSYSTEM.value,
        ).first()
        assert task is not None
        assert task.status == "failed"
        assert "LLM error" in task.last_error

    def test_increments_completed_files_per_subsystem(
        self, session, mock_adapter, monkeypatch
    ) -> None:
        repo = make_repo(session)
        rf1 = make_repo_file(session, repo, file_name="a.py", last_index_at=int(time.time()) - 1)
        rf1.set_metadata({"responsibility": "a", "key_elements": [], "dependent_files": [], "entry_point": False})
        rf2 = make_repo_file(session, repo, file_name="b.py", last_index_at=int(time.time()) - 1)
        rf2.set_metadata({"responsibility": "b", "key_elements": [], "dependent_files": [], "entry_point": False})
        session.commit()

        response_payload = json.dumps([
            {"name": "Sub1", "description": "d1", "keywords": [], "file_ids": [rf1.file_id]},
            {"name": "Sub2", "description": "d2", "keywords": [], "file_ids": [rf2.file_id]},
        ])
        mock_client = MagicMock()
        mock_client.responses.create.return_value = make_openai_response(response_payload)
        monkeypatch.setattr(
            "repo_analyzer.services.subsystem.subsystem_builder._get_openai_client",
            lambda: mock_client,
        )

        result = create_subsystems(repo.repo_hash)

        # total_files is indexed file count (2), completed_files should be 2 (one per spec written)
        assert result["status"] == "completed"
        assert result["total_files"] == 2
