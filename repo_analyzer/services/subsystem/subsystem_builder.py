"""Build repository subsystems via three-phase hierarchical clustering.

Phase 1 — Batch:   lightweight LLM call groups files into ≤30 batches.
Phase 2 — Cluster: concurrent per-batch LLM calls produce subsystem specs.
Phase 3 — Merge:   iterative LLM rounds consolidate to ≤K final subsystems.
"""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TypedDict

from openai import OpenAI
from sqlalchemy.orm import Session

from constants import (
    BUILD_SUBSYSTEM_MAX_CONCURRENCY,
    STALE_TASK_TIMEOUT_SECONDS,
    SUBSYSTEM_MAX_FINAL_COUNT,
    SUBSYSTEM_MAX_INITIAL_BATCHES,
    SUBSYSTEM_MAX_MERGE_ROUNDS,
)
from repo_analyzer.db import get_default_adapter
from repo_analyzer.db_managers import RepoManager, SubsystemManager
from repo_analyzer.models import IndexTask, Repo, RepoFile
from repo_analyzer.models.index_task import TaskProgress, TaskStatus, TaskType, is_task_stale
from repo_analyzer.prompts import (
    SUBSYSTEM_BATCH_SYSTEM_PROMPT,
    SUBSYSTEM_CLUSTER_SYSTEM_PROMPT,
    SUBSYSTEM_MERGE_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public / shared types
# ---------------------------------------------------------------------------

class SubsystemStatus(TypedDict):
    repo_hash: str
    status: str
    total_files: int
    completed_files: int
    remaining_files: int
    task_id: int
    progress: TaskProgress


class SubsystemSpec(TypedDict):
    name: str
    description: str
    keywords: list[str]
    file_ids: list[int]


class FileBatch(TypedDict):
    batch_id: int
    file_ids: list[int]


class MergeResult(TypedDict):
    subsystems: list[SubsystemSpec]
    continue_merging: bool


@dataclass(frozen=True)
class FileMetadataSnapshot:
    """Thread-safe snapshot of a file's indexed metadata."""

    file_id: int
    file_path: str
    is_project_file: bool
    responsibility: str
    key_elements: tuple[str, ...]
    dependent_files: tuple[str, ...]
    entry_point: bool


# ===================================================================
# Entry point
# ===================================================================

def create_subsystems(repo_hash: str, *, model: str = "gpt-5-mini") -> SubsystemStatus:
    """Start rebuilding subsystems for a repo in a background thread.

    Returns immediately with initial task status. Poll the task status
    endpoint to track progress.
    """
    if not repo_hash.strip():
        raise ValueError("repo_hash is required")

    adapter = get_default_adapter()

    # ── Phase 0: read repo + guard against duplicate runs ────────────────────
    with adapter.session() as session:
        repo_manager = RepoManager(session)
        repo = repo_manager.get_by_hash(repo_hash)
        if repo is None:
            raise ValueError(f"Repo not found: {repo_hash}")

        existing = _get_running_task(session, repo_hash)
        if existing is not None:
            now = int(time.time())
            if is_task_stale(existing, timeout_seconds=STALE_TASK_TIMEOUT_SECONDS, now=now):
                logger.warning(
                    "Subsystem task stale repo_hash=%s task_id=%d updated_at=%d",
                    repo_hash,
                    existing.task_id,
                    existing.updated_at,
                )
                existing.status = TaskStatus.STALE.value
                existing.updated_at = now
                existing.last_error = "Marked stale due to missing heartbeat."
                # session auto-commits on __exit__
            else:
                return _task_status(existing)

        files = repo_manager.list_repo_files(repo_hash, filter_scan_excluded=True)
        indexed_files = [f for f in files if f.last_index_at > 0 and f.get_metadata() is not None]

        if not indexed_files:
            task = IndexTask(
                repo_hash=repo_hash,
                task_type=TaskType.BUILD_SUBSYSTEM.value,
                status=TaskStatus.COMPLETED.value,
                total_files=0,
                completed_files=0,
                created_at=int(time.time()),
                updated_at=int(time.time()),
            )
            session.add(task)
            session.flush()
            return _task_status(task)

        # Build snapshots from ORM objects before session closes.
        snapshots = _build_file_metadata_snapshots(indexed_files)
        total_files = len(indexed_files)
        repo_hash_val = repo.repo_hash

    # ── Create running task record ────────────────────────────────────────────
    with adapter.session() as session:
        task = IndexTask(
            repo_hash=repo_hash,
            task_type=TaskType.BUILD_SUBSYSTEM.value,
            status=TaskStatus.RUNNING.value,
            total_files=total_files,
            completed_files=0,
            created_at=int(time.time()),
            updated_at=int(time.time()),
        )
        task.set_progress(TaskProgress(
            phase="Starting",
            steps_done=0,
            steps_total=total_files,
        ))
        session.add(task)
        session.flush()
        task_id = task.task_id
        initial_status = _task_status(task)
        logger.info("Subsystem task created repo_hash=%s task_id=%d", repo_hash, task_id)

    # ── Launch background thread ──────────────────────────────────────────────
    def _background() -> None:
        build_failed = False
        build_error = ""
        try:
            _rebuild_subsystems(
                repo_hash=repo_hash_val,
                snapshots=snapshots,
                total_files=total_files,
                model=model,
                task_id=task_id,
            )
        except Exception as exc:
            build_failed = True
            build_error = str(exc)
            logger.exception("Subsystem build failed repo_hash=%s", repo_hash)

        with adapter.session() as fin_session:
            db_task = fin_session.query(IndexTask).filter(IndexTask.task_id == task_id).one()
            db_task.status = TaskStatus.FAILED.value if build_failed else TaskStatus.COMPLETED.value
            if build_failed:
                db_task.last_error = build_error
            else:
                db_task.set_progress(TaskProgress(
                    phase="Completed",
                    steps_done=db_task.completed_files,
                    steps_total=db_task.total_files,
                ))
            db_task.updated_at = int(time.time())
            # fin_session auto-commits on __exit__

        logger.info("Subsystem build finished repo_hash=%s failed=%s", repo_hash, build_failed)

    threading.Thread(
        target=_background,
        daemon=True,
        name=f"subsystem-build-{repo_hash[:8]}",
    ).start()

    return initial_status


# ===================================================================
# Orchestrator (three-phase hierarchical clustering)
# ===================================================================

def _rebuild_subsystems(
    *,
    repo_hash: str,
    snapshots: dict[int, "FileMetadataSnapshot"],
    total_files: int,
    model: str,
    task_id: int,
) -> None:
    """Three-phase hierarchical clustering.

    Phase 1: lightweight LLM call batches files into ≤30 groups.
    Phase 2: concurrent per-batch LLM calls produce subsystem specs.
    Phase 3: iterative merge rounds consolidate to ≤K subsystems.
    """
    adapter = get_default_adapter()

    # Clear existing subsystems in its own session.
    with adapter.session() as clear_session:
        SubsystemManager(clear_session).delete_by_repo(repo_hash)
        # auto-commits on __exit__

    all_file_ids = set(snapshots.keys())
    files_list = list(snapshots.values())

    # Phase 1: lightweight batching — update progress.
    _update_task_progress(task_id, TaskProgress(
        phase="Batching files",
        steps_done=0,
        steps_total=total_files,
    ))

    batches = _phase1_initial_batching_from_snapshots(files_list, model=model, valid_file_ids=all_file_ids)
    logger.info("Phase 1 complete: %d batches for repo_hash=%s", len(batches), repo_hash)

    # Phase 2: concurrent per-batch clustering.
    _update_task_progress(task_id, TaskProgress(
        phase="Clustering files",
        steps_done=0,
        steps_total=len(batches),
    ))

    all_specs = _phase2_cluster_batches(
        batches,
        snapshots,
        model=model,
        repo_hash=repo_hash,
        task_id=task_id,
        total_batches=len(batches),
    )
    logger.info("Phase 2 complete: %d subsystems for repo_hash=%s", len(all_specs), repo_hash)

    # Phase 3: iterative merge (only if needed).
    if len(all_specs) > SUBSYSTEM_MAX_FINAL_COUNT:
        _update_task_progress(task_id, TaskProgress(
            phase="Merging subsystems",
            steps_done=0,
            steps_total=SUBSYSTEM_MAX_MERGE_ROUNDS,
        ))
        final_specs = _phase3_merge_rounds(
            all_specs,
            model=model,
            repo_hash=repo_hash,
            task_id=task_id,
        )
        logger.info(
            "Phase 3 complete: %d final subsystems for repo_hash=%s",
            len(final_specs),
            repo_hash,
        )
    else:
        logger.info(
            "Phase 3 skipped: %d subsystems already within target",
            len(all_specs),
        )


def _update_task_progress(task_id: int, progress: TaskProgress) -> None:
    """Update only the task progress meta in a short-lived session."""
    adapter = get_default_adapter()
    try:
        with adapter.session() as session:
            db_task = session.query(IndexTask).filter(IndexTask.task_id == task_id).one()
            db_task.set_progress(progress)
            db_task.updated_at = int(time.time())
            # auto-commits on __exit__
    except Exception:
        logger.debug("Could not update task progress for task_id=%d", task_id, exc_info=True)


# ===================================================================
# Phase 1 — Initial batching (from snapshots, no ORM)
# ===================================================================

def _build_batch_items_from_snapshots(snapshots: list[FileMetadataSnapshot]) -> list[dict[str, object]]:
    """Extract lightweight metadata from snapshot objects for Phase 1."""
    items: list[dict[str, object]] = []
    for snap in snapshots:
        items.append({
            "file_id": snap.file_id,
            "file_path": snap.file_path,
            "file_name": snap.file_path.split("/")[-1] if snap.file_path else "",
            "is_project_file": snap.is_project_file,
            "entry_point": snap.entry_point,
        })
    return items


def _build_phase1_prompt(items: list[dict[str, object]]) -> str:
    """Build the user prompt for the Phase 1 batching LLM call."""
    lines: list[str] = [
        f"Analyze these {len(items)} repository files and group them into batches.",
        "Return only JSON.",
    ]
    for item in items:
        lines.append("")
        lines.append(f"FILE_ID: {item['file_id']}")
        lines.append(f"PATH: {item['file_path']}")
        lines.append(f"FILE_NAME: {item['file_name']}")
        lines.append(f"IS_PROJECT_FILE: {str(item['is_project_file']).lower()}")
        lines.append(f"ENTRY_POINT: {str(item['entry_point']).lower()}")
    return "\n".join(lines)


def _parse_phase1_response(
    text: str,
    *,
    valid_file_ids: set[int],
) -> list[FileBatch]:
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("Expected JSON list from Phase 1.")

    batches: list[FileBatch] = []
    seen_ids: set[int] = set()
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            continue
        file_ids = _ensure_int_list(item.get("file_ids"))
        unique_ids = [fid for fid in file_ids if fid in valid_file_ids and fid not in seen_ids]
        seen_ids.update(unique_ids)
        if unique_ids:
            batches.append(FileBatch(
                batch_id=item.get("batch_id", idx + 1) if isinstance(item.get("batch_id"), int) else idx + 1,
                file_ids=unique_ids,
            ))

    missing = valid_file_ids - seen_ids
    if missing:
        logger.warning("Phase 1 missed %d file_ids, adding remainder batch", len(missing))
        batches.append(FileBatch(
            batch_id=len(batches) + 1,
            file_ids=sorted(missing),
        ))

    return batches


def _phase1_initial_batching_from_snapshots(
    snapshots: list[FileMetadataSnapshot],
    *,
    model: str,
    valid_file_ids: set[int],
) -> list[FileBatch]:
    """Send lightweight file metadata to LLM, get back ≤30 batches."""
    items = _build_batch_items_from_snapshots(snapshots)
    prompt = _build_phase1_prompt(items)
    system_prompt = SUBSYSTEM_BATCH_SYSTEM_PROMPT.format(
        max_batches=SUBSYSTEM_MAX_INITIAL_BATCHES,
    )
    response_text = _openai_response(
        model=model,
        system_prompt=system_prompt,
        user_prompt=prompt,
    )
    batches = _parse_phase1_response(response_text, valid_file_ids=valid_file_ids)

    if not batches:
        logger.warning("Phase 1 returned 0 batches; falling back to single batch")
        batches = [FileBatch(batch_id=1, file_ids=sorted(valid_file_ids))]

    return batches


# ===================================================================
# Phase 2 — Concurrent per-batch clustering
# ===================================================================

def _build_file_metadata_snapshots(
    files: list[RepoFile],
) -> dict[int, FileMetadataSnapshot]:
    """Build a lookup of file_id → FileMetadataSnapshot from ORM objects."""
    lookup: dict[int, FileMetadataSnapshot] = {}
    for f in files:
        meta = f.get_metadata() or {}
        lookup[f.file_id] = FileMetadataSnapshot(
            file_id=f.file_id,
            file_path=f.full_rel_path(),
            is_project_file=bool(f.is_project_file),
            responsibility=str(meta.get("responsibility", "")),
            key_elements=tuple(str(k) for k in (meta.get("key_elements") or [])),
            dependent_files=tuple(str(d) for d in (meta.get("dependent_files") or [])),
            entry_point=bool(meta.get("entry_point", False)),
        )
    return lookup


def _build_phase2_prompt(snapshots: list[FileMetadataSnapshot]) -> str:
    """Build the user prompt for a Phase 2 cluster LLM call."""
    lines: list[str] = [
        f"Analyze these {len(snapshots)} files and define subsystems.",
        "Return only JSON.",
    ]
    for snap in snapshots:
        lines.append("")
        lines.append(f"FILE_ID: {snap.file_id}")
        lines.append(f"PATH: {snap.file_path}")
        lines.append(f"RESPONSIBILITY: {snap.responsibility}")
        lines.append(f"KEY_ELEMENTS: {json.dumps(list(snap.key_elements))}")
        lines.append(f"DEPENDENT_FILES: {json.dumps(list(snap.dependent_files))}")
        lines.append(f"ENTRY_POINT: {str(snap.entry_point).lower()}")
    return "\n".join(lines)


def _cluster_single_batch(
    snapshots: list[FileMetadataSnapshot],
    *,
    model: str,
) -> list[SubsystemSpec]:
    """Call LLM with full metadata for one batch, return SubsystemSpecs."""
    prompt = _build_phase2_prompt(snapshots)
    response_text = _openai_response(
        model=model,
        system_prompt=SUBSYSTEM_CLUSTER_SYSTEM_PROMPT,
        user_prompt=prompt,
    )
    return [_normalize_spec(s) for s in _parse_json_list(response_text)]


def _phase2_cluster_batches(
    batches: list[FileBatch],
    snapshot_lookup: dict[int, FileMetadataSnapshot],
    *,
    model: str,
    repo_hash: str,
    task_id: int,
    total_batches: int,
) -> list[SubsystemSpec]:
    """Run Phase 2 concurrently across batches.

    Writes each subsystem to DB as results arrive and updates task progress.
    """
    adapter = get_default_adapter()
    all_specs: list[SubsystemSpec] = []
    batches_done = 0

    def _process_batch(batch: FileBatch) -> list[SubsystemSpec]:
        snapshots = [
            snapshot_lookup[fid]
            for fid in batch["file_ids"]
            if fid in snapshot_lookup
        ]
        if not snapshots:
            return []
        return _cluster_single_batch(snapshots, model=model)

    with ThreadPoolExecutor(max_workers=BUILD_SUBSYSTEM_MAX_CONCURRENCY) as executor:
        future_to_batch = {
            executor.submit(_process_batch, batch): batch
            for batch in batches
        }
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            specs = future.result()  # re-raises on exception
            all_specs.extend(specs)
            batches_done += 1

            # Write specs to DB and update progress.
            with adapter.session() as write_session:
                sub_mgr = SubsystemManager(write_session)
                for spec in specs:
                    sub_mgr.add_subsystem(
                        repo_hash=repo_hash,
                        name=spec["name"],
                        description=spec["description"],
                        file_ids=spec["file_ids"],
                        keywords=spec["keywords"],
                    )
                db_task = write_session.query(IndexTask).filter(
                    IndexTask.task_id == task_id,
                ).one()
                db_task.completed_files += len(batch["file_ids"])
                db_task.updated_at = int(time.time())
                db_task.set_progress(TaskProgress(
                    phase=f"Clustering files — batch {batches_done}/{total_batches}",
                    steps_done=batches_done,
                    steps_total=total_batches,
                ))
                write_session.commit()

            logger.info(
                "Phase 2 batch done batch_id=%d specs=%d repo_hash=%s",
                batch["batch_id"],
                len(specs),
                repo_hash,
            )

    return all_specs


# ===================================================================
# Phase 3 — Iterative merge rounds
# ===================================================================

def _build_phase3_prompt(specs: list[SubsystemSpec]) -> str:
    """Build the user prompt for a Phase 3 merge LLM call."""
    lines: list[str] = [
        f"Review these {len(specs)} subsystems and merge related ones.",
        f"Target at most {SUBSYSTEM_MAX_FINAL_COUNT} final subsystems.",
        "Return only JSON.",
    ]
    for idx, spec in enumerate(specs, 1):
        lines.append("")
        lines.append(f"SUBSYSTEM {idx}:")
        lines.append(f"NAME: {spec['name']}")
        lines.append(f"DESCRIPTION: {spec['description']}")
        lines.append(f"KEYWORDS: {json.dumps(spec['keywords'])}")
        lines.append(f"FILE_IDS: {json.dumps(spec['file_ids'])}")
    return "\n".join(lines)


def _parse_merge_response(text: str) -> MergeResult:
    """Parse the Phase 3 LLM response into a MergeResult."""
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object from Phase 3 merge.")

    raw_subsystems = parsed.get("subsystems")
    if not isinstance(raw_subsystems, list):
        raise ValueError("Missing 'subsystems' key in merge response.")

    specs = [_normalize_spec(s) for s in raw_subsystems if isinstance(s, dict)]
    continue_merging = bool(parsed.get("continue_merging", False))

    return MergeResult(subsystems=specs, continue_merging=continue_merging)


def _merge_single_round(
    specs: list[SubsystemSpec],
    *,
    model: str,
) -> MergeResult:
    """Call LLM with current subsystem list, return merged list + flag."""
    prompt = _build_phase3_prompt(specs)
    system_prompt = SUBSYSTEM_MERGE_SYSTEM_PROMPT.format(
        max_final=SUBSYSTEM_MAX_FINAL_COUNT,
    )
    response_text = _openai_response(
        model=model,
        system_prompt=system_prompt,
        user_prompt=prompt,
    )
    return _parse_merge_response(response_text)


def _specs_are_stable(
    prev: list[SubsystemSpec],
    curr: list[SubsystemSpec],
) -> bool:
    prev_names = sorted(s["name"] for s in prev)
    curr_names = sorted(s["name"] for s in curr)
    return prev_names == curr_names


def _write_subsystem_specs(
    specs: list[SubsystemSpec],
    *,
    repo_hash: str,
    task_id: int,
) -> None:
    """Delete existing subsystems and write a fresh set in one session."""
    adapter = get_default_adapter()
    with adapter.session() as write_session:
        SubsystemManager(write_session).delete_by_repo(repo_hash)
        sub_mgr = SubsystemManager(write_session)
        for spec in specs:
            sub_mgr.add_subsystem(
                repo_hash=repo_hash,
                name=spec["name"],
                description=spec["description"],
                file_ids=spec["file_ids"],
                keywords=spec["keywords"],
            )
        db_task = write_session.query(IndexTask).filter(
            IndexTask.task_id == task_id,
        ).one()
        db_task.updated_at = int(time.time())
        write_session.commit()


def _phase3_merge_rounds(
    initial_specs: list[SubsystemSpec],
    *,
    model: str,
    repo_hash: str,
    task_id: int,
) -> list[SubsystemSpec]:
    """Run iterative merge rounds until convergence or limits reached."""
    current_specs = initial_specs

    for round_num in range(1, SUBSYSTEM_MAX_MERGE_ROUNDS + 1):
        if len(current_specs) <= SUBSYSTEM_MAX_FINAL_COUNT:
            logger.info(
                "Phase 3 round %d: already at %d subsystems, stopping",
                round_num,
                len(current_specs),
            )
            break

        _update_task_progress(task_id, TaskProgress(
            phase=f"Merging subsystems — round {round_num}",
            steps_done=round_num - 1,
            steps_total=SUBSYSTEM_MAX_MERGE_ROUNDS,
        ))

        try:
            result = _merge_single_round(current_specs, model=model)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Phase 3 round %d: invalid LLM response, stopping: %s", round_num, exc)
            break

        new_specs = result["subsystems"]
        if not new_specs:
            logger.warning("Phase 3 round %d: LLM returned empty subsystems, stopping", round_num)
            break

        if _specs_are_stable(current_specs, new_specs):
            logger.info("Phase 3 round %d: specs stabilized", round_num)
            current_specs = new_specs
            break

        current_specs = new_specs

        _write_subsystem_specs(
            current_specs,
            repo_hash=repo_hash,
            task_id=task_id,
        )

        logger.info(
            "Phase 3 round %d: merged to %d subsystems repo_hash=%s",
            round_num,
            len(current_specs),
            repo_hash,
        )

        if not result["continue_merging"]:
            logger.info("Phase 3 round %d: LLM says stop merging", round_num)
            break

        if len(current_specs) <= SUBSYSTEM_MAX_FINAL_COUNT:
            logger.info(
                "Phase 3 round %d: reached target count %d",
                round_num,
                len(current_specs),
            )
            break

    return current_specs


# ===================================================================
# Shared helpers
# ===================================================================

def _get_running_task(session: Session, repo_hash: str) -> IndexTask | None:
    return (
        session.query(IndexTask)
        .filter(
            IndexTask.repo_hash == repo_hash,
            IndexTask.status == TaskStatus.RUNNING.value,
            IndexTask.task_type == TaskType.BUILD_SUBSYSTEM.value,
        )
        .order_by(IndexTask.created_at.desc())
        .first()
    )


def _task_status(task: IndexTask) -> SubsystemStatus:
    remaining = max(0, task.total_files - task.completed_files)
    return SubsystemStatus(
        repo_hash=task.repo_hash,
        status=task.status,
        total_files=task.total_files,
        completed_files=task.completed_files,
        remaining_files=remaining,
        task_id=task.task_id,
        progress=task.get_progress(),
    )


def _build_user_prompt(files: list[RepoFile]) -> str:
    """Legacy prompt builder (kept for backward compatibility)."""
    lines: list[str] = []
    lines.append("Input files with metadata. Build subsystems from these files.")
    lines.append("Return only JSON.")
    for file in files:
        meta = file.get_metadata() or {}
        entry_point = bool(meta.get("entry_point") or False)
        key_elements = meta.get("key_elements") or []
        dependent_files = meta.get("dependent_files") or []
        lines.append("")
        lines.append(f"FILE_ID: {file.file_id}")
        lines.append(f"PATH: {file.full_rel_path()}")
        lines.append(f"RESPONSIBILITY: {meta.get('responsibility', '')}")
        lines.append(f"KEY_ELEMENTS: {json.dumps(key_elements)}")
        lines.append(f"DEPENDENT_FILES: {json.dumps(dependent_files)}")
        lines.append(f"ENTRY_POINT: {str(entry_point).lower()}")
    return "\n".join(lines)


def _openai_response(*, model: str, system_prompt: str, user_prompt: str) -> str:
    client = _get_openai_client()
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
    )
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    raise ValueError("Unexpected OpenAI response format.")


def _get_openai_client() -> OpenAI:
    return OpenAI()


def _parse_json_list(text: str) -> list[dict[str, object]]:
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        out: list[dict[str, object]] = []
        for item in parsed:
            if isinstance(item, dict):
                out.append(item)
        return out
    raise ValueError("Expected JSON list response.")


def _normalize_spec(item: dict[str, object]) -> SubsystemSpec:
    name = str(item.get("name") or "").strip()
    description = str(item.get("description") or "").strip()
    keywords_raw = item.get("keywords")
    file_ids_raw = item.get("file_ids")
    keywords = _ensure_string_list(keywords_raw)
    file_ids = _ensure_int_list(file_ids_raw)
    return SubsystemSpec(
        name=name,
        description=description,
        keywords=keywords,
        file_ids=file_ids,
    )


def _ensure_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _ensure_int_list(value: object) -> list[int]:
    if isinstance(value, list):
        out: list[int] = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out
    return []
