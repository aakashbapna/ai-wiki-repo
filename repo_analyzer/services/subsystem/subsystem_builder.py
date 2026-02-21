"""Build repository subsystems from indexed file metadata."""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict

from openai import OpenAI
from sqlalchemy.orm import Session

from constants import BUILD_SUBSYSTEM_MAX_CONCURRENCY, STALE_TASK_TIMEOUT_SECONDS
from repo_analyzer.db import get_default_adapter
from repo_analyzer.prompts import SUBSYSTEM_BUILDER_SYSTEM_PROMPT
from repo_analyzer.db_managers import RepoManager, SubsystemManager
from repo_analyzer.models import IndexTask, Repo, RepoFile
from repo_analyzer.models.index_task import TaskStatus, TaskType, is_task_stale

logger = logging.getLogger(__name__)


class SubsystemStatus(TypedDict):
    repo_hash: str
    status: str
    total_files: int
    completed_files: int
    remaining_files: int
    task_id: int


class SubsystemSpec(TypedDict):
    name: str
    description: str
    keywords: list[str]
    file_ids: list[int]


def create_subsystems(repo_hash: str, *, model: str = "gpt-5-mini") -> SubsystemStatus:
    """Rebuild all subsystems for a repo using indexed file metadata."""
    if not repo_hash.strip():
        raise ValueError("repo_hash is required")

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
                    "Subsystem task stale repo_hash=%s task_id=%d updated_at=%d",
                    repo_hash,
                    existing.task_id,
                    existing.updated_at,
                )
                existing.status = TaskStatus.STALE.value
                existing.updated_at = now
                existing.last_error = "Marked stale due to missing heartbeat."
                session.commit()
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

        task = IndexTask(
            repo_hash=repo_hash,
            task_type=TaskType.BUILD_SUBSYSTEM.value,
            status=TaskStatus.RUNNING.value,
            total_files=len(indexed_files),
            completed_files=0,
            created_at=int(time.time()),
            updated_at=int(time.time()),
        )
        session.add(task)
        session.flush()

        try:
            _rebuild_subsystems(session, repo, indexed_files, model=model, task_id=task.task_id)
            # Reload task to pick up completed_files written by worker sessions.
            session.expire(task)
            task.status = TaskStatus.COMPLETED.value
            task.updated_at = int(time.time())
            session.commit()
        except Exception as exc:
            task.status = TaskStatus.FAILED.value
            task.updated_at = int(time.time())
            task.last_error = str(exc)
            session.commit()
            raise

        return _task_status(task)


def _rebuild_subsystems(
    session: Session,
    repo: Repo,
    files: list[RepoFile],
    *,
    model: str,
    task_id: int,
) -> None:
    """
    Delete existing subsystems, call the LLM, then write each spec to the DB
    in its own session as it comes off the parsed list so results are immediately
    visible. Uses BUILD_SUBSYSTEM_MAX_CONCURRENCY workers for the write phase.
    """
    adapter = get_default_adapter()
    repo_hash = repo.repo_hash

    # Clear existing subsystems and commit so worker threads can acquire the
    # write lock immediately instead of waiting for the outer session to close.
    SubsystemManager(session).delete_by_repo(repo_hash)
    session.commit()

    prompt = _build_user_prompt(files)
    response_text = _openai_response(
        model=model,
        system_prompt=SUBSYSTEM_BUILDER_SYSTEM_PROMPT,
        user_prompt=prompt,
    )
    specs = [_normalize_spec(s) for s in _parse_json_list(response_text)]
    logger.info("Parsed %d subsystem specs for repo_hash=%s", len(specs), repo_hash)

    def _write_spec(spec: SubsystemSpec) -> None:
        with adapter.session() as write_session:
            SubsystemManager(write_session).add_subsystem(
                repo_hash=repo_hash,
                name=spec["name"],
                description=spec["description"],
                file_ids=spec["file_ids"],
                keywords=spec["keywords"],
            )
            db_task = write_session.query(IndexTask).filter(
                IndexTask.task_id == task_id
            ).one()
            db_task.completed_files += 1
            db_task.updated_at = int(time.time())
            write_session.commit()
            logger.info(
                "Subsystem written name=%r repo_hash=%s", spec["name"], repo_hash
            )

    with ThreadPoolExecutor(max_workers=BUILD_SUBSYSTEM_MAX_CONCURRENCY) as executor:
        futures = {executor.submit(_write_spec, spec): spec for spec in specs}
        for future in as_completed(futures):
            future.result()  # re-raises any exception from the worker


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
    )


def _build_user_prompt(files: list[RepoFile]) -> str:
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
