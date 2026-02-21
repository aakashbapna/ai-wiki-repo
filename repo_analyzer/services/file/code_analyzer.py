"""Analyze repository files and store indexing metadata."""

import json
import logging
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
from repo_analyzer.models.index_task import TaskStatus, TaskType, is_task_stale

logger = logging.getLogger(__name__)

class IndexTaskStatus(TypedDict):
    repo_hash: str
    status: str
    total_files: int
    completed_files: int
    remaining_files: int
    task_id: int


class FileSummary(TypedDict):
    file_path: str
    responsibility: str
    key_elements: list[str]
    dependent_files: list[str]
    entry_point: bool


@dataclass(frozen=True)
class FileForIndex:
    file_path: str
    file_name: str
    content: str


def index_repo(repo_hash: str, *, batch_size: int = 3) -> IndexTaskStatus:
    """
    Index a repository's files and store metadata in RepoFile.

    Ensures only one running task per repo. If one exists, returns its status.
    """
    if not repo_hash.strip():
        raise ValueError("repo_hash is required")

    logger.info("Starting index_repo for repo_hash=%s batch_size=%d", repo_hash, batch_size)
    adapter = get_default_adapter()
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
                session.commit()
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
        logger.info(
            "Files to index repo_hash=%s total=%d",
            repo_hash,
            len(files_to_index),
        )
        if not files_to_index:
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

        task = IndexTask(
            repo_hash=repo_hash,
            task_type=TaskType.INDEX_FILE.value,
            status=TaskStatus.RUNNING.value,
            total_files=len(files_to_index),
            completed_files=0,
            created_at=created_at,
            updated_at=created_at,
        )
        session.add(task)
        session.flush()
        logger.info("Index task created repo_hash=%s task_id=%d", repo_hash, task.task_id)

        try:
            files_to_index = _filter_files_to_index(files, task_created_at=task.created_at)
            logger.info(
                "Filtered files_to_index repo_hash=%s task_id=%d created_at=%d count=%d",
                repo_hash,
                task.task_id,
                task.created_at,
                len(files_to_index),
            )
            task.total_files = len(files_to_index)
            session.commit()
            _run_indexing(session, repo, task, files_to_index, batch_size=batch_size)
            # Reload task to pick up completed_files written by worker sessions.
            session.expire(task)
            task.status = TaskStatus.COMPLETED.value
            task.updated_at = int(time.time())
            session.commit()
            logger.info(
                "Index task completed repo_hash=%s completed=%d total=%d",
                repo_hash,
                task.completed_files,
                task.total_files,
            )
        except Exception as exc:
            task.status = TaskStatus.FAILED.value
            task.updated_at = int(time.time())
            task.last_error = str(exc)
            session.commit()
            logger.exception("Index task failed repo_hash=%s", repo_hash)
            raise

        return _task_status(task)


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
    session: Session,
    repo: Repo,
    task: IndexTask,
    files: list[RepoFile],
    *,
    batch_size: int,
) -> None:
    """
    Submit all batches concurrently (up to INDEX_FILE_MAX_CONCURRENCY workers).
    As each batch's LLM call completes, write the results to the DB immediately
    using a dedicated session so progress is visible without waiting for the full run.
    """
    repo_root = repo.clone_path_resolved
    task_id = task.task_id
    repo_hash = task.repo_hash
    adapter = get_default_adapter()
    batches = list(_batch(files, batch_size))
    logger.debug(
        "Indexing repo=%s path=%s total_batches=%d concurrency=%d",
        repo_hash, repo_root, len(batches), INDEX_FILE_MAX_CONCURRENCY,
    )

    # Each worker receives pre-built payloads (pure data, no ORM objects).
    # Returns (batch_file_ids, summaries) so we can match back to DB rows.
    def _process_batch(
        batch: list[RepoFile],
    ) -> tuple[list[int], list[FileSummary]]:
        payloads = _build_file_payloads(repo_root, batch)
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
                file_ids, summaries = future.result()  # re-raises on exception
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
                # Reload the files from DB by ID to get a fresh ORM-bound objects.
                db_files: list[RepoFile] = (
                    write_session.query(RepoFile)
                    .filter(RepoFile.file_id.in_(file_ids))
                    .all()
                )
                db_task = write_session.query(IndexTask).filter(
                    IndexTask.task_id == task_id
                ).one()
                _apply_summaries(write_session, db_task, db_files, summaries)
                db_task.updated_at = int(time.time())
                write_session.commit()
                logger.info(
                    "Batch written repo_hash=%s completed=%d total=%d",
                    repo_hash, db_task.completed_files, db_task.total_files,
                )


def _build_file_payloads(repo_root: Path, files: list[RepoFile]) -> list[FileForIndex]:
    payloads: list[FileForIndex] = []
    for file in files:
        rel_path = Path(file.full_rel_path())
        content = _read_file_text(repo_root / rel_path)
        payloads.append(FileForIndex(
            file_path=rel_path.as_posix(),
            file_name=file.file_name,
            content=content,
        ))
    return payloads


def _apply_summaries(
    session: Session,
    task: IndexTask,
    files: list[RepoFile],
    summaries: list[FileSummary],
) -> None:
    summary_by_path = {summary["file_path"]: summary for summary in summaries}
    now = int(time.time())
    for file in files:
        summary = summary_by_path.get(file.full_rel_path())
        if summary is None:
            logger.debug("Missing summary for file_path=%s", file.file_path)
            continue
        meta: RepoFileMetadata = {
            "responsibility": summary["responsibility"],
            "key_elements": summary["key_elements"],
            "dependent_files": summary["dependent_files"],
            "entry_point": summary["entry_point"],
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
    lines.append("Each object must include: file_path, responsibility, key_elements, dependent_files, entry_point.")
    lines.append("Return only JSON. No extra text.")
    for idx, file in enumerate(files, start=1):
        lines.append("")
        lines.append(f"FILE {idx} PATH: {file.file_path}")
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

    key_elements = _ensure_string_list(key_elements_raw)
    dependent_files = _ensure_string_list(dependent_raw)

    return FileSummary(
        file_path=file_path,
        responsibility=responsibility,
        key_elements=key_elements,
        dependent_files=dependent_files,
        entry_point=entry_point,
    )


def _ensure_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []
