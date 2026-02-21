"""Build repository subsystems from indexed file metadata."""

import json
import logging
import time
from typing import TypedDict

from openai import OpenAI
from sqlalchemy.orm import Session

from db import get_default_adapter
from prompts import SUBSYSTEM_BUILDER_SYSTEM_PROMPT
from repo_analyzer.db_managers import RepoManager, SubsystemManager
from repo_analyzer.models import IndexTask, Repo, RepoFile
from repo_analyzer.models.index_task import TaskType

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
            return _task_status(existing)

        files = repo_manager.list_repo_files(repo_hash, filter_scan_excluded=True)
        indexed_files = [f for f in files if f.last_index_at > 0 and f.get_metadata() is not None]
        if not indexed_files:
            task = IndexTask(
                repo_hash=repo_hash,
                task_type=TaskType.BUILD_SUBSYSTEM.value,
                status="completed",
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
            status="running",
            total_files=len(indexed_files),
            completed_files=0,
            created_at=int(time.time()),
            updated_at=int(time.time()),
        )
        session.add(task)
        session.flush()

        try:
            _rebuild_subsystems(session, repo, indexed_files, model=model)
            task.completed_files = task.total_files
            task.status = "completed"
            task.updated_at = int(time.time())
            session.commit()
        except Exception as exc:
            task.status = "failed"
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
) -> None:
    subsystem_manager = SubsystemManager(session)
    subsystem_manager.delete_by_repo(repo.repo_hash)

    prompt = _build_user_prompt(files)
    response_text = _openai_response(
        model=model,
        system_prompt=SUBSYSTEM_BUILDER_SYSTEM_PROMPT,
        user_prompt=prompt,
    )
    specs = _parse_json_list(response_text)
    for spec in specs:
        normalized = _normalize_spec(spec)
        subsystem_manager.add_subsystem(
            repo_hash=repo.repo_hash,
            name=normalized["name"],
            description=normalized["description"],
            file_ids=normalized["file_ids"],
            keywords=normalized["keywords"],
        )
    session.flush()


def _get_running_task(session: Session, repo_hash: str) -> IndexTask | None:
    return (
        session.query(IndexTask)
        .filter(
            IndexTask.repo_hash == repo_hash,
            IndexTask.status == "running",
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
