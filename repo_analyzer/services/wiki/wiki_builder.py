"""Build wiki pages and sidebars from subsystems."""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from openai import OpenAI
from sqlalchemy.orm import Session

from constants import BUILD_WIKI_MAX_CONCURRENCY, STALE_TASK_TIMEOUT_SECONDS
from repo_analyzer.db import get_default_adapter
from repo_analyzer.db_managers import RepoManager, WikiManager
from repo_analyzer.models import IndexTask, Repo, RepoFile, RepoSubsystem
from repo_analyzer.models.index_task import TaskStatus, TaskType, is_task_stale
from repo_analyzer.prompts import WIKI_BUILDER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class WikiTaskStatus(TypedDict):
    repo_hash: str
    status: str
    total_files: int
    completed_files: int
    remaining_files: int
    task_id: int


class WikiContentSpec(TypedDict):
    content_type: str
    content: str
    source_file_ids: list[int]


class WikiPageSpec(TypedDict):
    title: str
    contents: list[WikiContentSpec]


def build_wiki(repo_hash: str, *, model: str = "gpt-5-mini") -> WikiTaskStatus:
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
                    "Wiki task stale repo_hash=%s task_id=%d updated_at=%d",
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

        subsystems = _list_subsystems(session, repo_hash)
        total_files = sum(len((s.get_meta() or {}).get("file_ids", [])) for s in subsystems)
        task = IndexTask(
            repo_hash=repo_hash,
            task_type=TaskType.BUILD_WIKI.value,
            status=TaskStatus.RUNNING.value,
            total_files=total_files,
            completed_files=0,
            created_at=int(time.time()),
            updated_at=int(time.time()),
        )
        session.add(task)
        session.flush()

        try:
            _rebuild_wiki(session, repo, subsystems, task, model=model)
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


def _rebuild_wiki(
    session: Session,
    repo: Repo,
    subsystems: list[RepoSubsystem],
    task: IndexTask,
    *,
    model: str,
) -> None:
    """
    Delete all existing wiki data, then generate one page per subsystem
    concurrently (up to BUILD_WIKI_MAX_CONCURRENCY workers). Each worker
    writes its page, content nodes, and sidebar to the DB in its own session
    immediately on completion — no waiting for the full run to finish.
    """
    adapter = get_default_adapter()
    repo_hash = repo.repo_hash
    task_id = task.task_id

    # Delete existing wiki data and commit so worker threads can acquire the
    # write lock immediately instead of waiting for the outer session to close.
    WikiManager(session).delete_by_repo(repo_hash)
    session.commit()

    # Snapshot plain data from ORM objects so workers don't share session state.
    @dataclass(frozen=True)
    class SubsystemSnapshot:
        subsystem_id: int
        name: str
        description: str
        file_ids: list[int]
        meta_json: str | None

    snapshots: list[SubsystemSnapshot] = []
    for s in subsystems:
        meta = s.get_meta() or {}
        file_ids = [int(x) for x in (meta.get("file_ids") or [])]
        snapshots.append(SubsystemSnapshot(
            subsystem_id=s.subsystem_id,
            name=s.name,
            description=s.description,
            file_ids=file_ids,
            meta_json=s.meta_json,
        ))

    repo_full_name = repo.full_name
    repo_clone_path = repo.clone_path_resolved

    def _process_subsystem(snap: SubsystemSnapshot) -> None:
        """Generate and persist one wiki page inside its own DB session."""
        with adapter.session() as write_session:
            wiki_manager = WikiManager(write_session)

            if not snap.file_ids:
                wiki_manager.add_sidebar(
                    repo_hash=repo_hash,
                    parent_node_id=None,
                    name=snap.name,
                    page_id=None,
                    is_active=False,
                    sub_system_ids=[snap.subsystem_id],
                )
                write_session.commit()
                return

            db_files: list[RepoFile] = (
                write_session.query(RepoFile)
                .filter(RepoFile.repo_hash == repo_hash, RepoFile.file_id.in_(snap.file_ids))
                .all()
            )

            page_spec = _generate_page_from_snapshot(
                repo_full_name=repo_full_name,
                repo_clone_path=repo_clone_path,
                snap=snap,
                files=db_files,
                model=model,
            )

            page = wiki_manager.add_page(
                repo_hash=repo_hash,
                title=page_spec["title"],
                subsystem_ids=[snap.subsystem_id],
            )
            write_session.flush()

            content_nodes = page_spec.get("contents") or []
            is_active = bool(content_nodes)
            for node in content_nodes:
                wiki_manager.add_content(
                    page_id=page.page_id,
                    content_type=node["content_type"],
                    content=node["content"],
                    source_file_ids=node["source_file_ids"],
                )

            wiki_manager.add_sidebar(
                repo_hash=repo_hash,
                parent_node_id=None,
                name=snap.name,
                page_id=page.page_id,
                is_active=is_active,
                sub_system_ids=[snap.subsystem_id],
            )

            db_task = write_session.query(IndexTask).filter(
                IndexTask.task_id == task_id
            ).one()
            db_task.completed_files += len(snap.file_ids)
            db_task.updated_at = int(time.time())
            write_session.commit()
            logger.info(
                "Wiki page written subsystem=%r repo_hash=%s",
                snap.name, repo_hash,
            )

    with ThreadPoolExecutor(max_workers=BUILD_WIKI_MAX_CONCURRENCY) as executor:
        futures = {executor.submit(_process_subsystem, snap): snap for snap in snapshots}
        for future in as_completed(futures):
            future.result()  # re-raises any exception from the worker


def _generate_page_from_snapshot(
    *,
    repo_full_name: str,
    repo_clone_path: Path,
    snap: "SubsystemSnapshot",  # type: ignore[name-defined]
    files: list[RepoFile],
    model: str,
) -> WikiPageSpec:
    """Build a wiki page spec using plain snapshot data (safe for threaded use)."""
    lines: list[str] = []
    lines.append("Subsystem context:")
    lines.append(f"REPO: {repo_full_name}")
    lines.append(f"SUBSYSTEM_ID: {snap.subsystem_id}")
    lines.append(f"SUBSYSTEM_NAME: {snap.name}")
    lines.append(f"SUBSYSTEM_DESCRIPTION: {snap.description}")
    lines.append(f"SUBSYSTEM_FILE_IDS: {json.dumps(snap.file_ids)}")
    lines.append("")
    lines.append("FILES:")
    for file in files:
        file_path = repo_clone_path / Path(file.full_rel_path())
        content = _read_file_text(file_path)
        lines.append(f"FILE_ID: {file.file_id}")
        lines.append(f"FILE_PATH: {file.full_rel_path()}")
        lines.append("FILE_CONTENT:")
        lines.append(content)
        lines.append("END_FILE")
        lines.append("")
    user_prompt = "\n".join(lines)
    response_text = _openai_response(
        model=model,
        system_prompt=WIKI_BUILDER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    return _normalize_page_spec(_parse_json_object(response_text))


def _fetch_files_for_subsystem(
    session: Session,
    repo_hash: str,
    file_ids: list[int],
) -> list[RepoFile]:
    return (
        session.query(RepoFile)
        .filter(RepoFile.repo_hash == repo_hash, RepoFile.file_id.in_(file_ids))
        .all()
    )


def _create_sidebar_only(
    wiki_manager: WikiManager,
    repo_hash: str,
    subsystem: RepoSubsystem,
    *,
    is_active: bool,
    page_id: int | None = None,
) -> None:
    wiki_manager.add_sidebar(
        repo_hash=repo_hash,
        parent_node_id=None,
        name=subsystem.name,
        page_id=page_id,
        is_active=is_active,
        sub_system_ids=[subsystem.subsystem_id],
    )


def _generate_page(
    repo: Repo,
    subsystem: RepoSubsystem,
    files: list[RepoFile],
    *,
    model: str,
) -> WikiPageSpec:
    user_prompt = _build_user_prompt(repo, subsystem, files)
    response_text = _openai_response(
        model=model,
        system_prompt=WIKI_BUILDER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    parsed = _parse_json_object(response_text)
    return _normalize_page_spec(parsed)


def _build_user_prompt(repo: Repo, subsystem: RepoSubsystem, files: list[RepoFile]) -> str:
    lines: list[str] = []
    meta = subsystem.get_meta() or {}
    lines.append("Subsystem context:")
    lines.append(f"REPO: {repo.full_name}")
    lines.append(f"SUBSYSTEM_ID: {subsystem.subsystem_id}")
    lines.append(f"SUBSYSTEM_NAME: {subsystem.name}")
    lines.append(f"SUBSYSTEM_DESCRIPTION: {subsystem.description}")
    lines.append(f"SUBSYSTEM_FILE_IDS: {json.dumps(meta.get('file_ids') or [])}")
    lines.append("")
    lines.append("FILES:")
    for file in files:
        file_path = repo.clone_path_resolved / Path(file.full_rel_path())
        content = _read_file_text(file_path)
        lines.append(f"FILE_ID: {file.file_id}")
        lines.append(f"FILE_PATH: {file.full_rel_path()}")
        lines.append("FILE_CONTENT:")
        lines.append(content)
        lines.append("END_FILE")
        lines.append("")
    return "\n".join(lines)


def _read_file_text(path: Path, *, max_bytes: int = 500_000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")


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


def _parse_json_object(text: str) -> dict[str, object]:
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object response.")
    return parsed


def _normalize_page_spec(item: dict[str, object]) -> WikiPageSpec:
    title = str(item.get("title") or "").strip()
    contents_raw = item.get("contents")
    contents: list[WikiContentSpec] = []
    if isinstance(contents_raw, list):
        for node in contents_raw:
            if not isinstance(node, dict):
                continue
            content_type = str(node.get("content_type") or "markdown")
            content = str(node.get("content") or "")
            source_file_ids = _ensure_int_list(node.get("source_file_ids"))
            contents.append({
                "content_type": content_type,
                "content": content,
                "source_file_ids": source_file_ids,
            })
    return {
        "title": title or "Subsystem Overview",
        "contents": contents,
    }


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


def _list_subsystems(session: Session, repo_hash: str) -> list[RepoSubsystem]:
    return (
        session.query(RepoSubsystem)
        .filter(RepoSubsystem.repo_hash == repo_hash)
        .order_by(RepoSubsystem.subsystem_id)
        .all()
    )


def _get_running_task(session: Session, repo_hash: str) -> IndexTask | None:
    return (
        session.query(IndexTask)
        .filter(
            IndexTask.repo_hash == repo_hash,
            IndexTask.status == TaskStatus.RUNNING.value,
            IndexTask.task_type == TaskType.BUILD_WIKI.value,
        )
        .order_by(IndexTask.created_at.desc())
        .first()
    )


def _task_status(task: IndexTask) -> WikiTaskStatus:
    remaining = max(0, task.total_files - task.completed_files)
    return WikiTaskStatus(
        repo_hash=task.repo_hash,
        status=task.status,
        total_files=task.total_files,
        completed_files=task.completed_files,
        remaining_files=remaining,
        task_id=task.task_id,
    )
