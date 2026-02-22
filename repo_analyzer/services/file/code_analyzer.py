"""Analyze repository files and store indexing metadata."""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TypedDict

from openai import OpenAI
from sqlalchemy.orm import Session

from constants import INDEX_FILE_MAX_CONCURRENCY, STALE_TASK_TIMEOUT_SECONDS
from repo_analyzer.db import get_default_adapter
from repo_analyzer.prompts import INDEX_FILE_SYSTEM_PROMPT
from repo_analyzer.db_managers import RepoManager
from repo_analyzer.models import IndexTask, Repo, RepoFile, RepoFileMetadata
from repo_analyzer.models.index_task import TaskProgress, TaskStatus, TaskType, is_task_stale

logger = logging.getLogger(__name__)


class IndexTaskStatus(TypedDict):
    repo_hash: str
    status: str
    total_files: int
    completed_files: int
    remaining_files: int
    task_id: int
    progress: TaskProgress


class FileSummary(TypedDict):
    file_path: str
    responsibility: str
    key_elements: list[str]
    dependent_files: list[str]
    entry_point: bool
    file_summary: str


@dataclass(frozen=True)
class FileForIndex:
    file_path: str
    file_name: str
    content: str
    file_size_bytes: int


def index_repo(repo_hash: str, *, batch_size: int = 3) -> IndexTaskStatus:
    """
    Start indexing a repository's files in a background thread.

    Returns immediately with initial task status. Ensures only one running task
    per repo. If one exists, returns its current status.
    """
    if not repo_hash.strip():
        raise ValueError("repo_hash is required")

    logger.info("Starting index_repo for repo_hash=%s batch_size=%d", repo_hash, batch_size)
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
                    "Index task stale repo_hash=%s task_id=%d updated_at=%d",
                    repo_hash,
                    existing.task_id,
                    existing.updated_at,
                )
                existing.status = TaskStatus.STALE.value
                existing.updated_at = now
                existing.last_error = "Marked stale due to missing heartbeat."
                # session auto-commits on __exit__
            else:
                logger.info(
                    "Index task already running repo_hash=%s completed=%d total=%d",
                    repo_hash,
                    existing.completed_files,
                    existing.total_files,
                )
                return _task_status(existing)

        files = repo_manager.list_repo_files(repo_hash, filter_scan_excluded=True)
        logger.info(
            "Scannable files loaded repo_hash=%s count=%d",
            repo_hash,
            len(files),
        )
        created_at = int(time.time())
        files_to_index = _filter_files_to_index(files, task_created_at=created_at)
        logger.info(
            "Initial files_to_index repo_hash=%s count=%d",
            repo_hash,
            len(files_to_index),
        )

        # Snapshot plain data (file IDs + paths) before session closes.
        repo_hash_val = repo.repo_hash
        repo_clone_path = repo.clone_path_resolved
        file_ids_to_index = [f.file_id for f in files_to_index]

    # ── No files to index: create completed task immediately ─────────────────
    if not file_ids_to_index:
        with adapter.session() as session:
            task = IndexTask(
                repo_hash=repo_hash,
                task_type=TaskType.INDEX_FILE.value,
                status=TaskStatus.COMPLETED.value,
                total_files=0,
                completed_files=0,
                created_at=created_at,
                updated_at=created_at,
            )
            session.add(task)
            session.flush()
            logger.info("No files to index for repo_hash=%s", repo_hash)
            return _task_status(task)

    # ── Create running task record ────────────────────────────────────────────
    with adapter.session() as session:
        task = IndexTask(
            repo_hash=repo_hash,
            task_type=TaskType.INDEX_FILE.value,
            status=TaskStatus.RUNNING.value,
            total_files=len(file_ids_to_index),
            completed_files=0,
            created_at=created_at,
            updated_at=created_at,
        )
        task.set_progress(TaskProgress(
            phase="Starting",
            steps_done=0,
            steps_total=len(file_ids_to_index),
        ))
        session.add(task)
        session.flush()
        task_id = task.task_id
        initial_status = _task_status(task)
        logger.info("Index task created repo_hash=%s task_id=%d", repo_hash, task_id)

    # ── Launch background thread ──────────────────────────────────────────────
    def _background() -> None:
        build_failed = False
        build_error = ""
        try:
            _run_indexing(
                repo_hash=repo_hash_val,
                repo_clone_path=repo_clone_path,
                task_id=task_id,
                file_ids=file_ids_to_index,
                batch_size=batch_size,
            )
        except Exception as exc:
            build_failed = True
            build_error = str(exc)
            logger.exception("Index task failed repo_hash=%s", repo_hash)

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

        logger.info("Index task finished repo_hash=%s failed=%s", repo_hash, build_failed)

    threading.Thread(
        target=_background,
        daemon=True,
        name=f"index-{repo_hash[:8]}",
    ).start()

    return initial_status


def index_file(files: list[FileForIndex], *, model: str = "gpt-5-mini") -> list[FileSummary]:
    """Send a batch of files to the model and return per-file summaries."""
    if not files:
        return []

    system_prompt = INDEX_FILE_SYSTEM_PROMPT
    logger.debug("Indexing %d files with model=%s", len(files), model)
    user_prompt = _build_user_prompt(files)
    response_text = _openai_response(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    parsed = _parse_json_list(response_text)
    if len(parsed) != len(files):
        logger.warning(
            "Index repo validation mismatch: expected %d summaries, got %d. "
            "Continuing with best-effort reconciliation.",
            len(files),
            len(parsed),
        )
    reconciled = _reconcile_summaries(parsed, files)
    return reconciled


def _get_running_task(session: Session, repo_hash: str) -> IndexTask | None:
    return (
        session.query(IndexTask)
        .filter(
            IndexTask.repo_hash == repo_hash,
            IndexTask.status == TaskStatus.RUNNING.value,
            IndexTask.task_type == TaskType.INDEX_FILE.value,
        )
        .order_by(IndexTask.created_at.desc())
        .first()
    )


def _task_status(task: IndexTask) -> IndexTaskStatus:
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


def _filter_files_to_index(files: Iterable[RepoFile], *, task_created_at: int | None) -> list[RepoFile]:
    to_index: list[RepoFile] = []
    total: int = 0
    never_indexed: int = 0
    stale_index: int = 0
    skipped_recent: int = 0
    for file in files:
        total += 1
        last_index_at = int(file.last_index_at or 0)
        if task_created_at is None:
            if last_index_at <= 0:
                to_index.append(file)
                never_indexed += 1
        else:
            if last_index_at <= 0 or last_index_at < task_created_at:
                to_index.append(file)
                if last_index_at <= 0:
                    never_indexed += 1
                else:
                    stale_index += 1
            else:
                skipped_recent += 1
    logger.debug(
        "Index filter summary total=%d to_index=%d never_indexed=%d stale_index=%d skipped_recent=%d "
        "task_created_at=%s",
        total,
        len(to_index),
        never_indexed,
        stale_index,
        skipped_recent,
        str(task_created_at),
    )
    return to_index


def _run_indexing(
    *,
    repo_hash: str,
    repo_clone_path: Path,
    task_id: int,
    file_ids: list[int],
    batch_size: int,
) -> None:
    """
    Submit all batches concurrently (up to INDEX_FILE_MAX_CONCURRENCY workers).
    As each batch's LLM call completes, write the results to the DB immediately
    using a dedicated session so progress is visible without waiting for the full run.
    """
    adapter = get_default_adapter()
    total_files = len(file_ids)

    # Reload RepoFile objects in the background thread's own session.
    with adapter.session() as read_session:
        files: list[RepoFile] = (
            read_session.query(RepoFile)
            .filter(RepoFile.file_id.in_(file_ids))
            .all()
        )
        # Detach by reading into plain list; session closes after block.

    batches = list(_batch(files, batch_size))
    logger.debug(
        "Indexing repo=%s path=%s total_batches=%d concurrency=%d",
        repo_hash, repo_clone_path, len(batches), INDEX_FILE_MAX_CONCURRENCY,
    )

    def _process_batch(
        batch: list[RepoFile],
    ) -> tuple[list[int], list[FileSummary]]:
        payloads = _build_file_payloads(repo_clone_path, batch)
        summaries = index_file(payloads)
        return ([f.file_id for f in batch], summaries)

    with ThreadPoolExecutor(max_workers=INDEX_FILE_MAX_CONCURRENCY) as executor:
        future_to_batch = {
            executor.submit(_process_batch, batch): batch
            for batch in batches
        }
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            logger.info(
                "Batch complete repo_hash=%s batch_size=%d",
                repo_hash, len(batch),
            )
            try:
                file_ids_done, summaries = future.result()
            except Exception as exc:
                logger.exception(
                    "Batch failed repo_hash=%s batch_size=%d",
                    repo_hash,
                    len(batch),
                )
                with adapter.session() as write_session:
                    db_task = write_session.query(IndexTask).filter(
                        IndexTask.task_id == task_id
                    ).one()
                    db_task.updated_at = int(time.time())
                    db_task.last_error = str(exc)
                    write_session.commit()
                continue

            # Write this batch's results in its own session so they land immediately.
            with adapter.session() as write_session:
                db_files: list[RepoFile] = (
                    write_session.query(RepoFile)
                    .filter(RepoFile.file_id.in_(file_ids_done))
                    .all()
                )
                db_task = write_session.query(IndexTask).filter(
                    IndexTask.task_id == task_id
                ).one()
                _apply_summaries(write_session, db_task, db_files, summaries, repo_root=repo_clone_path)
                db_task.updated_at = int(time.time())
                db_task.set_progress(TaskProgress(
                    phase="Indexing files",
                    steps_done=db_task.completed_files,
                    steps_total=total_files,
                ))
                write_session.commit()
                logger.info(
                    "Batch written repo_hash=%s completed=%d total=%d",
                    repo_hash, db_task.completed_files, db_task.total_files,
                )


def _build_file_payloads(repo_root: Path, files: list[RepoFile]) -> list[FileForIndex]:
    payloads: list[FileForIndex] = []
    for file in files:
        rel_path = Path(file.full_rel_path())
        full_path = repo_root / rel_path
        content = _read_file_text(full_path)
        file_size_bytes = 0
        if full_path.exists() and full_path.is_file():
            try:
                file_size_bytes = full_path.stat().st_size
            except OSError:
                file_size_bytes = 0
        payloads.append(FileForIndex(
            file_path=rel_path.as_posix(),
            file_name=file.file_name,
            content=content,
            file_size_bytes=file_size_bytes,
        ))
    return payloads


def _apply_summaries(
    session: Session,
    task: IndexTask,
    files: list[RepoFile],
    summaries: list[FileSummary],
    *,
    repo_root: Path,
) -> None:
    summary_by_path = {summary["file_path"]: summary for summary in summaries}
    now = int(time.time())
    for file in files:
        full_path = repo_root / Path(file.full_rel_path())
        if full_path.exists() and full_path.is_file():
            try:
                file.file_size = full_path.stat().st_size
            except OSError:
                pass
        summary = summary_by_path.get(file.full_rel_path())
        if summary is None:
            logger.debug("Missing summary for file_path=%s", file.file_path)
            continue
        file_summary_raw = str(summary.get("file_summary") or "")
        file_summary = file_summary_raw[:1000]
        meta: RepoFileMetadata = {
            "responsibility": summary["responsibility"],
            "key_elements": summary["key_elements"],
            "dependent_files": summary["dependent_files"],
            "entry_point": summary["entry_point"],
            "file_summary": file_summary,
        }
        file.set_metadata(meta)
        file.last_index_at = now
        task.completed_files += 1
        session.flush()


def _batch(items: list[RepoFile], batch_size: int) -> Iterable[list[RepoFile]]:
    size = max(1, batch_size)
    for idx in range(0, len(items), size):
        yield items[idx:idx + size]


def _read_file_text(path: Path, *, max_bytes: int = 200_000) -> str:
    if not path.exists():
        logger.debug("File missing at path=%s", path)
        return ""
    if not path.is_file():
        logger.debug("Path is not a file: %s", path)
        return ""
    data = path.read_bytes()
    if len(data) > max_bytes:
        logger.debug("Truncating file %s to %d bytes", path, max_bytes)
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")


def _build_user_prompt(files: list[FileForIndex]) -> str:
    lines: list[str] = []
    lines.append("You will be given multiple files. Return a JSON array with one object per file.")
    lines.append(
        "Each object must include: file_path, responsibility, key_elements, dependent_files, entry_point, file_summary."
    )
    lines.append(
        "Only include file_summary if file_size_bytes > 10240, otherwise return an empty string for file_summary."
    )
    lines.append("Return only JSON. No extra text.")
    for idx, file in enumerate(files, start=1):
        lines.append("")
        lines.append(f"FILE {idx} PATH: {file.file_path}")
        lines.append(f"FILE {idx} SIZE BYTES: {file.file_size_bytes}")
        lines.append("CONTENT:")
        lines.append(file.content)
        lines.append(f"END FILE {idx}")
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


def _reconcile_summaries(
    parsed: list[dict[str, object]],
    files: list[FileForIndex],
) -> list[FileSummary]:
    summaries_by_path: dict[str, dict[str, object]] = {}
    unmatched: list[dict[str, object]] = []
    for item in parsed:
        file_path_value = item.get("file_path")
        file_path = str(file_path_value) if file_path_value else ""
        if file_path and file_path not in summaries_by_path:
            summaries_by_path[file_path] = item
        else:
            unmatched.append(item)

    reconciled: list[FileSummary] = []
    unmatched_idx = 0
    for file in files:
        item = summaries_by_path.get(file.file_path)
        if item is None and unmatched_idx < len(unmatched):
            item = unmatched[unmatched_idx]
            unmatched_idx += 1
        if item is None:
            item = {}
        reconciled.append(
            _normalize_summary(item, default_file_path=file.file_path)
        )
    return reconciled


def _normalize_summary(item: dict[str, object], *, default_file_path: str) -> FileSummary:
    file_path = str(item.get("file_path") or default_file_path or "")
    responsibility = str(item.get("responsibility") or "")
    entry_point = bool(item.get("entry_point") or False)
    key_elements_raw = item.get("key_elements")
    dependent_raw = item.get("dependent_files")
    file_summary = str(item.get("file_summary") or "")

    key_elements = _ensure_string_list(key_elements_raw)
    dependent_files = _ensure_string_list(dependent_raw)

    return FileSummary(
        file_path=file_path,
        responsibility=responsibility,
        key_elements=key_elements,
        dependent_files=dependent_files,
        entry_point=entry_point,
        file_summary=file_summary,
    )


def _ensure_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []
