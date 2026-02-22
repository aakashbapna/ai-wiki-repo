"""Tests for repo_analyzer.services.subsystem.subsystem_builder.

Covers pure helpers, Phase 1/2/3 functions, and end-to-end create_subsystems.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from repo_analyzer.db_managers import SubsystemManager
from repo_analyzer.models.index_task import IndexTask, TaskType
from repo_analyzer.models.repo_subsystem import RepoSubsystem
from repo_analyzer.services.subsystem.subsystem_builder import (
    FileMetadataSnapshot,
    FileBatch,
    SubsystemSpec,
    _build_batch_items_from_snapshots,
    _build_file_metadata_snapshots,
    _build_phase1_prompt,
    _build_phase2_prompt,
    _build_phase3_prompt,
    _ensure_int_list,
    _ensure_string_list,
    _merge_single_round,
    _normalize_spec,
    _parse_json_list,
    _parse_merge_response,
    _parse_phase1_response,
    _specs_are_stable,
    create_subsystems,
)
from tests.conftest import (
    make_index_task,
    make_repo,
    make_repo_file,
    make_subsystem,
)


def _make_sync_thread(monkeypatch):
    """Patch threading.Thread so the target runs synchronously."""
    class SyncThread:
        def __init__(self, target=None, daemon=None, name=None, **kwargs):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    monkeypatch.setattr("repo_analyzer.services.subsystem.subsystem_builder.threading.Thread", SyncThread)


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


# ── _build_batch_items_from_snapshots (Phase 1) ──────────────────────────────

class TestBuildBatchItems:
    def test_extracts_fields(self, session, engine) -> None:
        repo = make_repo(session)
        rf = make_repo_file(session, repo, file_path="src", file_name="app.py")
        rf.set_metadata({
            "responsibility": "main entry",
            "key_elements": [],
            "dependent_files": [],
            "entry_point": True,
        })
        session.flush()

        snapshots = _build_file_metadata_snapshots([rf])
        items = _build_batch_items_from_snapshots(list(snapshots.values()))

        assert len(items) == 1
        assert items[0]["file_id"] == rf.file_id
        assert items[0]["file_path"] == "src/app.py"
        assert items[0]["file_name"] == "app.py"
        assert items[0]["entry_point"] is True

    def test_handles_missing_metadata(self, session, engine) -> None:
        repo = make_repo(session)
        rf = make_repo_file(session, repo, file_name="bare.py")
        session.flush()

        snapshots = _build_file_metadata_snapshots([rf])
        items = _build_batch_items_from_snapshots(list(snapshots.values()))

        assert items[0]["entry_point"] is False


# ── _build_phase1_prompt ────────────────────────────────────────────────────

class TestBuildPhase1Prompt:
    def test_includes_file_ids_and_paths(self) -> None:
        items: list[dict[str, object]] = [
            {"file_id": 10, "file_path": "src/a.py", "file_name": "a.py",
             "is_project_file": False, "entry_point": False},
        ]
        prompt = _build_phase1_prompt(items)
        assert "FILE_ID: 10" in prompt
        assert "PATH: src/a.py" in prompt

    def test_includes_flags(self) -> None:
        items: list[dict[str, object]] = [
            {"file_id": 1, "file_path": "x.py", "file_name": "x.py",
             "is_project_file": True, "entry_point": True},
        ]
        prompt = _build_phase1_prompt(items)
        assert "IS_PROJECT_FILE: true" in prompt
        assert "ENTRY_POINT: true" in prompt


# ── _parse_phase1_response ──────────────────────────────────────────────────

class TestParsePhase1Response:
    def test_parses_valid_batches(self) -> None:
        text = json.dumps([
            {"batch_id": 1, "file_ids": [1, 2]},
            {"batch_id": 2, "file_ids": [3]},
        ])
        batches = _parse_phase1_response(text, valid_file_ids={1, 2, 3})
        assert len(batches) == 2
        assert batches[0]["file_ids"] == [1, 2]
        assert batches[1]["file_ids"] == [3]

    def test_creates_remainder_batch_for_missing_ids(self) -> None:
        text = json.dumps([{"batch_id": 1, "file_ids": [1]}])
        batches = _parse_phase1_response(text, valid_file_ids={1, 2, 3})
        assert len(batches) == 2
        assert set(batches[1]["file_ids"]) == {2, 3}

    def test_filters_invalid_file_ids(self) -> None:
        text = json.dumps([{"batch_id": 1, "file_ids": [1, 999]}])
        batches = _parse_phase1_response(text, valid_file_ids={1, 2})
        assert batches[0]["file_ids"] == [1]
        # 999 is not valid, 2 is missing → remainder batch
        assert 2 in batches[1]["file_ids"]

    def test_deduplicates_across_batches(self) -> None:
        text = json.dumps([
            {"batch_id": 1, "file_ids": [1, 2]},
            {"batch_id": 2, "file_ids": [2, 3]},
        ])
        batches = _parse_phase1_response(text, valid_file_ids={1, 2, 3})
        # 2 already in batch 1, so batch 2 should only have 3
        assert batches[0]["file_ids"] == [1, 2]
        assert batches[1]["file_ids"] == [3]

    def test_wraps_single_dict(self) -> None:
        text = json.dumps({"batch_id": 1, "file_ids": [1]})
        batches = _parse_phase1_response(text, valid_file_ids={1})
        assert len(batches) == 1

    def test_assigns_batch_ids_when_missing(self) -> None:
        text = json.dumps([{"file_ids": [1]}, {"file_ids": [2]}])
        batches = _parse_phase1_response(text, valid_file_ids={1, 2})
        assert batches[0]["batch_id"] == 1
        assert batches[1]["batch_id"] == 2

    def test_returns_empty_for_all_invalid_ids(self) -> None:
        text = json.dumps([{"batch_id": 1, "file_ids": [999]}])
        batches = _parse_phase1_response(text, valid_file_ids={1})
        # 999 is filtered out → batch is empty → not added; 1 is missing → remainder
        assert len(batches) == 1
        assert batches[0]["file_ids"] == [1]


# ── _build_file_metadata_snapshots (Phase 2) ────────────────────────────────

class TestBuildFileMetadataSnapshots:
    def test_builds_lookup_from_repo_files(self, session, engine) -> None:
        repo = make_repo(session)
        rf = make_repo_file(session, repo, file_path="src", file_name="auth.py")
        rf.set_metadata({
            "responsibility": "auth logic",
            "key_elements": ["login"],
            "dependent_files": ["db.py"],
            "entry_point": True,
        })
        session.flush()

        lookup = _build_file_metadata_snapshots([rf])

        snap = lookup[rf.file_id]
        assert snap.file_path == "src/auth.py"
        assert snap.responsibility == "auth logic"
        assert snap.key_elements == ("login",)
        assert snap.dependent_files == ("db.py",)
        assert snap.entry_point is True

    def test_handles_files_with_no_metadata(self, session, engine) -> None:
        repo = make_repo(session)
        rf = make_repo_file(session, repo, file_name="bare.py")
        session.flush()

        lookup = _build_file_metadata_snapshots([rf])

        snap = lookup[rf.file_id]
        assert snap.responsibility == ""
        assert snap.key_elements == ()
        assert snap.entry_point is False


# ── _build_phase2_prompt ────────────────────────────────────────────────────

class TestBuildPhase2Prompt:
    def test_includes_full_metadata(self) -> None:
        snap = FileMetadataSnapshot(
            file_id=42,
            file_path="src/auth.py",
            is_project_file=False,
            responsibility="handles JWT",
            key_elements=("login", "verify"),
            dependent_files=("db.py",),
            entry_point=True,
        )
        prompt = _build_phase2_prompt([snap])
        assert "FILE_ID: 42" in prompt
        assert "PATH: src/auth.py" in prompt
        assert "RESPONSIBILITY: handles JWT" in prompt
        assert '"login"' in prompt
        assert "ENTRY_POINT: true" in prompt
        assert '"db.py"' in prompt


# ── _build_phase3_prompt ────────────────────────────────────────────────────

class TestBuildPhase3Prompt:
    def test_includes_all_subsystem_specs(self) -> None:
        specs: list[SubsystemSpec] = [
            SubsystemSpec(name="Auth", description="d", keywords=["k"], file_ids=[1, 2]),
            SubsystemSpec(name="DB", description="d2", keywords=[], file_ids=[3]),
        ]
        prompt = _build_phase3_prompt(specs)
        assert "SUBSYSTEM 1:" in prompt
        assert "NAME: Auth" in prompt
        assert "SUBSYSTEM 2:" in prompt
        assert "NAME: DB" in prompt
        assert "FILE_IDS: [1, 2]" in prompt


# ── _parse_merge_response ──────────────────────────────────────────────────

class TestParseMergeResponse:
    def test_parses_valid_merge_response(self) -> None:
        text = json.dumps({
            "subsystems": [
                {"name": "Auth", "description": "d", "keywords": ["k"], "file_ids": [1, 2]},
            ],
            "continue_merging": False,
        })
        result = _parse_merge_response(text)
        assert len(result["subsystems"]) == 1
        assert result["subsystems"][0]["name"] == "Auth"
        assert result["continue_merging"] is False

    def test_defaults_continue_merging_to_false(self) -> None:
        text = json.dumps({
            "subsystems": [{"name": "X", "description": "", "keywords": [], "file_ids": [1]}],
        })
        result = _parse_merge_response(text)
        assert result["continue_merging"] is False

    def test_raises_on_missing_subsystems_key(self) -> None:
        text = json.dumps({"continue_merging": True})
        with pytest.raises(ValueError, match="Missing 'subsystems'"):
            _parse_merge_response(text)

    def test_raises_on_non_dict(self) -> None:
        text = json.dumps([1, 2, 3])
        with pytest.raises(ValueError, match="Expected JSON object"):
            _parse_merge_response(text)


# ── _specs_are_stable ───────────────────────────────────────────────────────

class TestSpecsAreStable:
    def test_identical_specs_are_stable(self) -> None:
        a: list[SubsystemSpec] = [SubsystemSpec(name="Auth", description="", keywords=[], file_ids=[])]
        b: list[SubsystemSpec] = [SubsystemSpec(name="Auth", description="", keywords=[], file_ids=[])]
        assert _specs_are_stable(a, b) is True

    def test_different_names_are_not_stable(self) -> None:
        a: list[SubsystemSpec] = [SubsystemSpec(name="Auth", description="", keywords=[], file_ids=[])]
        b: list[SubsystemSpec] = [SubsystemSpec(name="Security", description="", keywords=[], file_ids=[])]
        assert _specs_are_stable(a, b) is False

    def test_reordered_specs_are_stable(self) -> None:
        a: list[SubsystemSpec] = [
            SubsystemSpec(name="Auth", description="", keywords=[], file_ids=[]),
            SubsystemSpec(name="DB", description="", keywords=[], file_ids=[]),
        ]
        b: list[SubsystemSpec] = [
            SubsystemSpec(name="DB", description="", keywords=[], file_ids=[]),
            SubsystemSpec(name="Auth", description="", keywords=[], file_ids=[]),
        ]
        assert _specs_are_stable(a, b) is True

    def test_empty_lists_are_stable(self) -> None:
        assert _specs_are_stable([], []) is True

    def test_different_lengths_are_not_stable(self) -> None:
        a: list[SubsystemSpec] = [SubsystemSpec(name="X", description="", keywords=[], file_ids=[])]
        assert _specs_are_stable(a, []) is False


# ── _merge_single_round (Phase 3 worker) ────────────────────────────────────

class TestMergeSingleRound:
    def test_returns_merged_specs_and_flag(self, monkeypatch) -> None:
        specs: list[SubsystemSpec] = [
            SubsystemSpec(name="Auth", description="d1", keywords=["k1"], file_ids=[1]),
            SubsystemSpec(name="Users", description="d2", keywords=["k2"], file_ids=[2]),
        ]
        payload = json.dumps({
            "subsystems": [
                {"name": "Auth & Users", "description": "merged", "keywords": ["k1", "k2"], "file_ids": [1, 2]},
            ],
            "continue_merging": False,
        })
        monkeypatch.setattr(
            "repo_analyzer.services.subsystem.subsystem_builder.run_batch",
            lambda requests, **kwargs: [payload],
        )

        result = _merge_single_round(specs, model="gpt-5-mini")
        assert len(result["subsystems"]) == 1
        assert result["subsystems"][0]["name"] == "Auth & Users"
        assert result["continue_merging"] is False


# ── create_subsystems (integration) ─────────────────────────────────────────

class TestCreateSubsystems:
    """Integration tests for the full three-phase pipeline."""

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
        make_repo_file(session, repo, file_name="raw.py", last_index_at=0)
        session.commit()

        result = create_subsystems(repo.repo_hash)

        assert result["status"] == "completed"
        assert result["total_files"] == 0

    def test_builds_subsystems_from_indexed_files(
        self, session, mock_adapter, monkeypatch
    ) -> None:
        """Full pipeline: Phase 1 batches into 1 group → Phase 2 clusters → done."""
        _make_sync_thread(monkeypatch)
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

        # Phase 1 response: single batch containing the one file
        phase1_resp = json.dumps([{"batch_id": 1, "file_ids": [rf.file_id]}])
        # Phase 2 response: one subsystem
        phase2_resp = json.dumps([{
            "name": "Auth",
            "description": "Authentication module",
            "keywords": ["login", "jwt"],
            "file_ids": [rf.file_id],
        }])

        responses = iter([phase1_resp, phase2_resp])

        def fake_run_batch(requests, **kwargs):
            return [next(responses)]

        def fake_stream_batch(requests, **kwargs):
            text = next(responses)
            for i in range(len(requests)):
                yield (i, text)

        monkeypatch.setattr(
            "repo_analyzer.services.subsystem.subsystem_builder.run_batch",
            fake_run_batch,
        )
        monkeypatch.setattr(
            "repo_analyzer.services.subsystem.subsystem_builder.stream_batch",
            fake_stream_batch,
        )

        create_subsystems(repo.repo_hash)

        task = session.query(IndexTask).filter(
            IndexTask.repo_hash == repo.repo_hash,
            IndexTask.task_type == TaskType.BUILD_SUBSYSTEM.value,
        ).first()
        assert task is not None
        assert task.status == "completed"

        subsystems = session.query(RepoSubsystem).filter(
            RepoSubsystem.repo_hash == repo.repo_hash
        ).all()
        assert len(subsystems) == 1
        assert subsystems[0].name == "Auth"

    def test_deletes_existing_subsystems_before_rebuild(
        self, session, mock_adapter, monkeypatch
    ) -> None:
        _make_sync_thread(monkeypatch)
        repo = make_repo(session)
        make_subsystem(session, repo, name="OldSubsystem")
        rf = make_repo_file(
            session, repo, file_name="core.py",
            last_index_at=int(time.time()) - 5,
        )
        rf.set_metadata({"responsibility": "r", "key_elements": [], "dependent_files": [], "entry_point": False})
        session.commit()

        phase1_resp = json.dumps([{"batch_id": 1, "file_ids": [rf.file_id]}])
        phase2_resp = json.dumps([{
            "name": "NewSubsystem",
            "description": "Fresh",
            "keywords": [],
            "file_ids": [rf.file_id],
        }])

        responses = iter([phase1_resp, phase2_resp])

        monkeypatch.setattr(
            "repo_analyzer.services.subsystem.subsystem_builder.run_batch",
            lambda requests, **kwargs: [next(responses)],
        )
        monkeypatch.setattr(
            "repo_analyzer.services.subsystem.subsystem_builder.stream_batch",
            lambda requests, **kwargs: [(0, next(responses))],
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
        _make_sync_thread(monkeypatch)
        repo = make_repo(session)
        rf = make_repo_file(
            session, repo, file_name="x.py",
            last_index_at=int(time.time()) - 5,
        )
        rf.set_metadata({"responsibility": "x", "key_elements": [], "dependent_files": [], "entry_point": False})
        session.commit()

        monkeypatch.setattr(
            "repo_analyzer.services.subsystem.subsystem_builder.run_batch",
            lambda requests, **kwargs: [RuntimeError("LLM error")],
        )

        # create_subsystems doesn't raise — error is caught in background thread
        create_subsystems(repo.repo_hash)

        task = session.query(IndexTask).filter(
            IndexTask.repo_hash == repo.repo_hash,
            IndexTask.task_type == TaskType.BUILD_SUBSYSTEM.value,
        ).first()
        assert task is not None
        assert task.status == "failed"
        assert "LLM error" in task.last_error

    def test_progress_tracking_through_phases(
        self, session, mock_adapter, monkeypatch
    ) -> None:
        """completed_files should equal total indexed files after Phase 2."""
        _make_sync_thread(monkeypatch)
        repo = make_repo(session)
        rf1 = make_repo_file(session, repo, file_name="a.py", last_index_at=int(time.time()) - 1)
        rf1.set_metadata({"responsibility": "a", "key_elements": [], "dependent_files": [], "entry_point": False})
        rf2 = make_repo_file(session, repo, file_name="b.py", last_index_at=int(time.time()) - 1)
        rf2.set_metadata({"responsibility": "b", "key_elements": [], "dependent_files": [], "entry_point": False})
        session.commit()

        # Phase 1: two batches, one file each
        phase1_resp = json.dumps([
            {"batch_id": 1, "file_ids": [rf1.file_id]},
            {"batch_id": 2, "file_ids": [rf2.file_id]},
        ])
        # Phase 2: one subsystem per batch
        phase2_resp_a = json.dumps([{"name": "SubA", "description": "d", "keywords": [], "file_ids": [rf1.file_id]}])
        phase2_resp_b = json.dumps([{"name": "SubB", "description": "d", "keywords": [], "file_ids": [rf2.file_id]}])

        phase2_resps = iter([phase2_resp_a, phase2_resp_b])

        monkeypatch.setattr(
            "repo_analyzer.services.subsystem.subsystem_builder.run_batch",
            lambda requests, **kwargs: [phase1_resp],
        )
        monkeypatch.setattr(
            "repo_analyzer.services.subsystem.subsystem_builder.stream_batch",
            lambda requests, **kwargs: [(i, next(phase2_resps)) for i in range(len(requests))],
        )

        create_subsystems(repo.repo_hash)

        task = session.query(IndexTask).filter(
            IndexTask.repo_hash == repo.repo_hash,
            IndexTask.task_type == TaskType.BUILD_SUBSYSTEM.value,
        ).first()
        assert task is not None
        assert task.status == "completed"
        assert task.total_files == 2
        assert task.completed_files == 2

    def test_multiple_batches_produce_multiple_subsystems(
        self, session, mock_adapter, monkeypatch
    ) -> None:
        """Two batches each producing a subsystem → 2 subsystems in DB."""
        _make_sync_thread(monkeypatch)
        repo = make_repo(session)
        rf1 = make_repo_file(session, repo, file_path="auth", file_name="login.py",
                             last_index_at=int(time.time()) - 1)
        rf1.set_metadata({"responsibility": "auth", "key_elements": ["login"], "dependent_files": [], "entry_point": False})
        rf2 = make_repo_file(session, repo, file_path="db", file_name="models.py",
                             last_index_at=int(time.time()) - 1)
        rf2.set_metadata({"responsibility": "db models", "key_elements": ["orm"], "dependent_files": [], "entry_point": False})
        session.commit()

        phase1_resp = json.dumps([
            {"batch_id": 1, "file_ids": [rf1.file_id]},
            {"batch_id": 2, "file_ids": [rf2.file_id]},
        ])
        phase2_resp_1 = json.dumps([{"name": "Auth", "description": "Auth module", "keywords": ["login"], "file_ids": [rf1.file_id]}])
        phase2_resp_2 = json.dumps([{"name": "Database", "description": "DB layer", "keywords": ["orm"], "file_ids": [rf2.file_id]}])

        phase2_resps = iter([phase2_resp_1, phase2_resp_2])

        monkeypatch.setattr(
            "repo_analyzer.services.subsystem.subsystem_builder.run_batch",
            lambda requests, **kwargs: [phase1_resp],
        )
        monkeypatch.setattr(
            "repo_analyzer.services.subsystem.subsystem_builder.stream_batch",
            lambda requests, **kwargs: [(i, next(phase2_resps)) for i in range(len(requests))],
        )

        create_subsystems(repo.repo_hash)

        task = session.query(IndexTask).filter(
            IndexTask.repo_hash == repo.repo_hash,
            IndexTask.task_type == TaskType.BUILD_SUBSYSTEM.value,
        ).first()
        assert task is not None
        assert task.status == "completed"

        subsystems = session.query(RepoSubsystem).filter(
            RepoSubsystem.repo_hash == repo.repo_hash
        ).all()
        names = sorted(s.name for s in subsystems)
        assert names == ["Auth", "Database"]

    def test_phase3_merge_triggered_when_too_many_subsystems(
        self, session, mock_adapter, monkeypatch
    ) -> None:
        """When Phase 2 produces >10 subsystems, Phase 3 merges them."""
        _make_sync_thread(monkeypatch)
        repo = make_repo(session)
        # Create 12 files so we can produce >10 subsystems
        files = []
        for i in range(12):
            rf = make_repo_file(
                session, repo,
                file_path="src", file_name=f"f{i}.py",
                last_index_at=int(time.time()) - 1,
            )
            rf.set_metadata({"responsibility": f"task {i}", "key_elements": [], "dependent_files": [], "entry_point": False})
            files.append(rf)
        session.commit()

        file_ids = [f.file_id for f in files]

        # Phase 1: all files in one batch
        phase1_resp = json.dumps([{"batch_id": 1, "file_ids": file_ids}])

        # Phase 2: one subsystem per file (12 total) → triggers Phase 3
        phase2_specs = [
            {"name": f"Sub{i}", "description": f"d{i}", "keywords": [], "file_ids": [fid]}
            for i, fid in enumerate(file_ids)
        ]
        phase2_resp = json.dumps(phase2_specs)

        # Phase 3: merge down to 4 subsystems
        merged_specs = [
            {"name": f"Merged{i}", "description": f"merged {i}", "keywords": [],
             "file_ids": file_ids[i * 3:(i + 1) * 3] if i < 3 else file_ids[9:]}
            for i in range(4)
        ]
        phase3_resp = json.dumps({
            "subsystems": merged_specs,
            "continue_merging": False,
        })

        run_batch_resps = iter([phase1_resp, phase3_resp])

        monkeypatch.setattr(
            "repo_analyzer.services.subsystem.subsystem_builder.run_batch",
            lambda requests, **kwargs: [next(run_batch_resps)],
        )
        monkeypatch.setattr(
            "repo_analyzer.services.subsystem.subsystem_builder.stream_batch",
            lambda requests, **kwargs: [(0, phase2_resp)],
        )

        create_subsystems(repo.repo_hash)

        task = session.query(IndexTask).filter(
            IndexTask.repo_hash == repo.repo_hash,
            IndexTask.task_type == TaskType.BUILD_SUBSYSTEM.value,
        ).first()
        assert task is not None
        assert task.status == "completed"

        subsystems = session.query(RepoSubsystem).filter(
            RepoSubsystem.repo_hash == repo.repo_hash
        ).all()
        names = sorted(s.name for s in subsystems)
        assert names == ["Merged0", "Merged1", "Merged2", "Merged3"]

    def test_phase3_skipped_when_within_target(
        self, session, mock_adapter, monkeypatch
    ) -> None:
        """When Phase 2 produces ≤10 subsystems, Phase 3 is skipped."""
        _make_sync_thread(monkeypatch)
        repo = make_repo(session)
        rf = make_repo_file(
            session, repo, file_name="only.py",
            last_index_at=int(time.time()) - 1,
        )
        rf.set_metadata({"responsibility": "only", "key_elements": [], "dependent_files": [], "entry_point": False})
        session.commit()

        phase1_resp = json.dumps([{"batch_id": 1, "file_ids": [rf.file_id]}])
        phase2_resp = json.dumps([{"name": "Solo", "description": "d", "keywords": [], "file_ids": [rf.file_id]}])

        run_batch_call_count = 0

        def counting_run_batch(requests, **kwargs):
            nonlocal run_batch_call_count
            run_batch_call_count += 1
            return [phase1_resp]

        monkeypatch.setattr(
            "repo_analyzer.services.subsystem.subsystem_builder.run_batch",
            counting_run_batch,
        )
        monkeypatch.setattr(
            "repo_analyzer.services.subsystem.subsystem_builder.stream_batch",
            lambda requests, **kwargs: [(0, phase2_resp)],
        )

        create_subsystems(repo.repo_hash)

        task = session.query(IndexTask).filter(
            IndexTask.repo_hash == repo.repo_hash,
            IndexTask.task_type == TaskType.BUILD_SUBSYSTEM.value,
        ).first()
        assert task is not None
        assert task.status == "completed"
        # Only 1 run_batch call (Phase 1), no Phase 3
        assert run_batch_call_count == 1
