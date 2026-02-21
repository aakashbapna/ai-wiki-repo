"""Build wiki pages and sidebars from subsystems."""

import json
import logging
import time
from pathlib import Path
from typing import TypedDict

from openai import OpenAI
from sqlalchemy.orm import Session

from repo_analyzer.db import get_default_adapter
from repo_analyzer.db_managers import RepoManager, WikiManager
from repo_analyzer.models import IndexTask, Repo, RepoFile, RepoSubsystem
from repo_analyzer.models.index_task import TaskType
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
            return _task_status(existing)

        subsystems = _list_subsystems(session, repo_hash)
        total_files = sum(len((s.get_meta() or {}).get("file_ids", [])) for s in subsystems)
        task = IndexTask(
            repo_hash=repo_hash,
            task_type=TaskType.BUILD_WIKI.value,
            status="running",
            total_files=total_files,
            completed_files=0,
            created_at=int(time.time()),
            updated_at=int(time.time()),
        )
        session.add(task)
        session.flush()

        try:
            _rebuild_wiki(session, repo, subsystems, task, model=model)
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


def _rebuild_wiki(
    session: Session,
    repo: Repo,
    subsystems: list[RepoSubsystem],
    task: IndexTask,
    *,
    model: str,
) -> None:
    wiki_manager = WikiManager(session)

    wiki_manager.delete_by_repo(repo.repo_hash)

    for subsystem in subsystems:
        meta = subsystem.get_meta() or {}
        file_ids = [int(x) for x in (meta.get("file_ids") or [])]
        if not file_ids:
            _create_sidebar_only(wiki_manager, repo.repo_hash, subsystem, is_active=False)
            continue

        files = _fetch_files_for_subsystem(session, repo.repo_hash, file_ids)
        page_spec = _generate_page(repo, subsystem, files, model=model)
        page = wiki_manager.add_page(
            repo_hash=repo.repo_hash,
            title=page_spec["title"],
            subsystem_ids=[subsystem.subsystem_id],
        )

        content_nodes = page_spec.get("contents") or []
        if not content_nodes:
            _create_sidebar_only(wiki_manager, repo.repo_hash, subsystem, is_active=False, page_id=page.page_id)
            continue

        for node in content_nodes:
            wiki_manager.add_content(
                page_id=page.page_id,
                content_type=node["content_type"],
                content=node["content"],
                source_file_ids=node["source_file_ids"],
            )

        _create_sidebar_only(wiki_manager, repo.repo_hash, subsystem, is_active=True, page_id=page.page_id)
        task.completed_files += len(file_ids)
        task.updated_at = int(time.time())
        session.flush()


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
            IndexTask.status == "running",
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
