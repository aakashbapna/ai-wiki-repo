"""Build wiki pages and sidebars from subsystems."""

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from sqlalchemy.orm import Session

from constants import (
    BUILD_WIKI_MAX_CONCURRENCY,
    LLM_MODEL,
    STALE_TASK_TIMEOUT_SECONDS,
    WIKI_SIDEBAR_MAX_CHILDREN,
    WIKI_SIDEBAR_MAX_TOP_NODES,
)
from repo_analyzer.db import get_default_adapter
from repo_analyzer.db_managers import RepoManager, WikiManager
from repo_analyzer.models import IndexTask, RepoFile, RepoSubsystem
from repo_analyzer.models.index_task import TaskProgress, TaskStatus, TaskType, is_task_stale
from repo_analyzer.prompts import WIKI_BUILDER_SYSTEM_PROMPT, WIKI_SIDEBAR_SYSTEM_PROMPT
from repo_analyzer.utils.async_openai import OpenAIRequest, run_batch, stream_batch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed structures
# ---------------------------------------------------------------------------

class WikiTaskStatus(TypedDict):
    repo_hash: str
    status: str
    total_files: int
    completed_files: int
    remaining_files: int
    task_id: int
    progress: TaskProgress


class WikiContentSpec(TypedDict):
    title: str
    content_type: str
    content: str
    source_file_ids: list[int]


class WikiPageSpec(TypedDict):
    title: str
    contents: list[WikiContentSpec]


class SidebarNodeSpec(TypedDict):
    name: str
    page_title: str | None
    subsystem_ids: list[int]
    children: list["SidebarNodeSpec"]


class SidebarTreeSpec(TypedDict):
    nodes: list[SidebarNodeSpec]


@dataclass(frozen=True)
class PageSummary:
    """Lightweight summary of a generated wiki page for sidebar LLM input."""
    page_id: int
    title: str
    subsystem_id: int
    subsystem_name: str
    subsystem_description: str
    keywords: list[str]


@dataclass(frozen=True)
class SubsystemSnapshot:
    """Plain-data snapshot of a RepoSubsystem; safe to share across threads."""
    subsystem_id: int
    name: str
    description: str
    file_ids: list[int]
    meta_json: str | None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_wiki(repo_hash: str, *, model: str = LLM_MODEL) -> WikiTaskStatus:
    """Start a wiki build and return the task status immediately.

    The heavy work (LLM calls, page generation, sidebar) runs in a daemon
    thread so the caller is never blocked waiting for completion.  Poll the
    task status endpoint to track progress.
    """
    if not repo_hash.strip():
        raise ValueError("repo_hash is required")

    adapter = get_default_adapter()

    # ── Phase 0: read repo + guard against duplicate runs ────────────────────
    # Short-lived session — closes before any worker threads start.
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
                    repo_hash, existing.task_id, existing.updated_at,
                )
                existing.status = TaskStatus.STALE.value
                existing.updated_at = now
                existing.last_error = "Marked stale due to missing heartbeat."
                # session auto-commits on __exit__
            else:
                # A healthy build is already running — return its current status.
                return _task_status(existing)

        subsystems = _list_subsystems(session, repo_hash)
        total_files = sum(len((s.get_meta() or {}).get("file_ids", [])) for s in subsystems)

        # Snapshot plain data before the session closes.
        repo_full_name = repo.full_name
        repo_clone_path = repo.clone_path_resolved

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

    # ── Phase 0b: delete old wiki data + create task record ──────────────────
    # Own session, commits and closes immediately — DB fully unlocked afterward.
    with adapter.session() as session:
        WikiManager(session).delete_by_repo(repo_hash)
        task = IndexTask(
            repo_hash=repo_hash,
            task_type=TaskType.BUILD_WIKI.value,
            status=TaskStatus.RUNNING.value,
            total_files=total_files,
            completed_files=0,
            created_at=int(time.time()),
            updated_at=int(time.time()),
        )
        task.set_progress(TaskProgress(
            phase="Starting",
            steps_done=0,
            steps_total=len(snapshots),
        ))
        session.add(task)
        session.flush()
        task_id = task.task_id
        initial_status = _task_status(task)
        # session auto-commits on __exit__

    # ── Phases 1-3: run entirely in a background daemon thread ───────────────
    # The caller returns `initial_status` immediately.  The thread does the
    # LLM work (one session per page write, all short-lived) then marks done.
    def _background() -> None:
        build_failed = False
        build_error = ""
        try:
            _rebuild_wiki(
                repo_hash=repo_hash,
                repo_full_name=repo_full_name,
                repo_clone_path=repo_clone_path,
                snapshots=snapshots,
                task_id=task_id,
                model=model,
            )
        except Exception as exc:
            build_failed = True
            build_error = str(exc)
            logger.exception("Wiki build failed repo_hash=%s: %s", repo_hash, exc)

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

        logger.info("Wiki build finished repo_hash=%s failed=%s", repo_hash, build_failed)

    threading.Thread(
        target=_background,
        daemon=True,
        name=f"wiki-build-{repo_hash[:8]}",
    ).start()

    return initial_status


# ---------------------------------------------------------------------------
# Phase 1 + Phase 2 orchestration
# ---------------------------------------------------------------------------

def _rebuild_wiki(
    *,
    repo_hash: str,
    repo_full_name: str,
    repo_clone_path: Path,
    snapshots: list[SubsystemSnapshot],
    task_id: int,
    model: str,
) -> None:
    """
    Phase 1: Generate one wiki page per subsystem concurrently via
             ``stream_batch``. DB reads + prompt building happen eagerly,
             LLM calls run async, DB writes happen as each result arrives.
    Phase 2: Single LLM call to organize pages into a sidebar tree.
    """
    adapter = get_default_adapter()

    # Filter out empty subsystems.
    active_snapshots = [s for s in snapshots if s.file_ids]

    # ── 1a. Build all user prompts eagerly (serial DB + filesystem reads) ────
    user_prompts: list[str] = []
    for snap in active_snapshots:
        with adapter.session() as read_session:
            db_files: list[RepoFile] = (
                read_session.query(RepoFile)
                .filter(RepoFile.repo_hash == repo_hash, RepoFile.file_id.in_(snap.file_ids))
                .all()
            )
        user_prompts.append(_build_page_user_prompt(
            repo_full_name=repo_full_name,
            repo_clone_path=repo_clone_path,
            snap=snap,
            files=db_files,
        ))

    # ── 1b. Fire all LLM requests via stream_batch ──────────────────────────
    openai_requests = [
        OpenAIRequest(
            system_prompt=WIKI_BUILDER_SYSTEM_PROMPT,
            user_prompt=prompt,
            model=model,
            api="chat",
            response_format={"type": "json_object"},
        )
        for prompt in user_prompts
    ]

    page_summaries: list[PageSummary] = []

    for idx, result in stream_batch(openai_requests, max_concurrency=BUILD_WIKI_MAX_CONCURRENCY):
        snap = active_snapshots[idx]

        if isinstance(result, Exception):
            logger.error("Wiki page LLM failed subsystem=%r: %s", snap.name, result)
            continue

        try:
            parsed = _parse_json_object(result)
            page_spec = _normalize_page_spec(parsed)
        except Exception as exc:
            logger.exception("Wiki page parse failed subsystem=%r: %s", snap.name, exc)
            continue

        # Write page + contents.
        with adapter.session() as write_session:
            wiki_manager = WikiManager(write_session)
            page = wiki_manager.add_page(
                repo_hash=repo_hash,
                title=page_spec["title"],
                subsystem_ids=[snap.subsystem_id],
            )
            write_session.flush()

            for node in (page_spec.get("contents") or []):
                wiki_manager.add_content(
                    page_id=page.page_id,
                    title=node["title"],
                    content_type=node["content_type"],
                    content=node["content"],
                    source_file_ids=node["source_file_ids"],
                )
            page_id = page.page_id

        # Update task progress.
        with adapter.session() as task_session:
            db_task = task_session.query(IndexTask).filter(
                IndexTask.task_id == task_id
            ).one()
            db_task.completed_files += len(snap.file_ids)
            db_task.updated_at = int(time.time())
            db_task.set_progress(TaskProgress(
                phase="Generating pages",
                steps_done=db_task.completed_files,
                steps_total=db_task.total_files,
            ))

        logger.info("Wiki page saved subsystem=%r page_id=%d repo_hash=%s",
                     snap.name, page_id, repo_hash)

        meta = json.loads(snap.meta_json) if snap.meta_json else {}
        keywords_raw = meta.get("keywords", [])
        keywords = keywords_raw if isinstance(keywords_raw, list) else []

        page_summaries.append(PageSummary(
            page_id=page_id,
            title=page_spec["title"],
            subsystem_id=snap.subsystem_id,
            subsystem_name=snap.name,
            subsystem_description=snap.description,
            keywords=[str(k) for k in keywords],
        ))

    # ── Phase 2: Generate sidebar tree via single LLM call ───────────────────
    if page_summaries:
        try:
            with adapter.session() as prog_session:
                db_task = prog_session.query(IndexTask).filter(
                    IndexTask.task_id == task_id
                ).one()
                db_task.set_progress(TaskProgress(
                    phase="Generating sidebar",
                    steps_done=db_task.completed_files,
                    steps_total=db_task.total_files,
                ))
                db_task.updated_at = int(time.time())
        except Exception:
            logger.debug("Could not update progress before sidebar generation", exc_info=True)

        _generate_and_save_sidebar_tree(
            repo_hash=repo_hash,
            repo_full_name=repo_full_name,
            page_summaries=page_summaries,
            model=model,
        )
    else:
        logger.info(
            "No pages generated; skipping sidebar generation for repo_hash=%s",
            repo_hash,
        )


# ---------------------------------------------------------------------------
# Phase 2: Sidebar tree generation
# ---------------------------------------------------------------------------

_INTRO_SIGNALS = frozenset([
    "overview", "introduction", "readme", "getting started", "about", "summary",
    "welcome", "guide", "quickstart", "start", "begin", "entry",
])


def _score_page(ps: PageSummary, signals: frozenset[str]) -> int:
    """Return a relevance score for a page against a set of signal words."""
    score = 0
    text = " ".join([
        ps.title.lower(),
        ps.subsystem_name.lower(),
        ps.subsystem_description.lower(),
        " ".join(k.lower() for k in ps.keywords),
    ])
    for signal in signals:
        if signal in text:
            score += 1
    return score


def _build_sidebar_user_prompt(
    repo_full_name: str,
    page_summaries: list[PageSummary],
) -> str:
    """Build the user prompt for sidebar tree generation.

    Identifies the best candidate for the mandatory Introduction node so the
    LLM can assign it without guessing.
    """
    intro_candidate: PageSummary | None = None
    if page_summaries:
        intro_candidate = max(page_summaries, key=lambda p: _score_page(p, _INTRO_SIGNALS))

    lines: list[str] = []
    lines.append(f"Organize these {len(page_summaries)} wiki pages into a sidebar tree.")
    lines.append(f"REPO: {repo_full_name}")
    lines.append(f"TOTAL_PAGES: {len(page_summaries)}")
    lines.append("")

    if intro_candidate:
        lines.append(f"INTRO_CANDIDATE: {intro_candidate.title}")
    lines.append("")

    lines.append("PAGES:")
    for ps in page_summaries:
        lines.append(f"  PAGE_ID: {ps.page_id}")
        lines.append(f"  TITLE: {ps.title}")
        lines.append(f"  SUBSYSTEM_ID: {ps.subsystem_id}")
        lines.append(f"  SUBSYSTEM_NAME: {ps.subsystem_name}")
        lines.append(f"  SUBSYSTEM_DESCRIPTION: {ps.subsystem_description}")
        lines.append(f"  KEYWORDS: {json.dumps(ps.keywords)}")
        lines.append("")
    lines.append("Return only JSON.")
    return "\n".join(lines)


def _normalize_sidebar_node(
    raw: dict[str, object],
    valid_titles: set[str],
    *,
    max_children: int,
) -> SidebarNodeSpec:
    """Normalize a single sidebar node from LLM output."""
    name = str(raw.get("name") or "").strip() or "Untitled"
    page_title_raw = raw.get("page_title")
    page_title: str | None = None
    if isinstance(page_title_raw, str) and page_title_raw.strip():
        candidate = page_title_raw.strip()
        if candidate in valid_titles:
            page_title = candidate

    subsystem_ids = _ensure_int_list(raw.get("subsystem_ids"))

    raw_children = raw.get("children")
    children: list[SidebarNodeSpec] = []
    if isinstance(raw_children, list):
        for child_raw in raw_children[:max_children]:
            if not isinstance(child_raw, dict):
                continue
            # Children are leaf nodes — no grandchildren allowed
            child = _normalize_sidebar_node(child_raw, valid_titles, max_children=0)
            children.append(child)

    return SidebarNodeSpec(
        name=name,
        page_title=page_title,
        subsystem_ids=subsystem_ids,
        children=children,
    )


def _parse_sidebar_tree(
    text: str,
    valid_titles: set[str],
    *,
    max_top_nodes: int,
    max_children: int,
    intro_candidate: PageSummary | None = None,
    title_to_subsystem_ids: dict[str, list[int]] | None = None,
) -> SidebarTreeSpec:
    """Parse LLM response into a validated SidebarTreeSpec.

    Truncates to max_top_nodes top-level and max_children per node.
    Validates page_title references against valid_titles.
    Ensures the mandatory 'Introduction' node is always first, assigning the
    best-fit page when the LLM omits it.
    """
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object for sidebar tree.")

    raw_nodes = parsed.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError("Missing 'nodes' key in sidebar tree response.")

    nodes: list[SidebarNodeSpec] = []
    for raw in raw_nodes[:max_top_nodes]:
        if not isinstance(raw, dict):
            continue
        node = _normalize_sidebar_node(raw, valid_titles, max_children=max_children)
        nodes.append(node)

    sub_ids_map = title_to_subsystem_ids or {}

    # Ensure the mandatory Introduction node is always first
    node_names = [n["name"].lower() for n in nodes]
    if "introduction" not in node_names:
        if intro_candidate and intro_candidate.title in valid_titles:
            intro_node: SidebarNodeSpec = SidebarNodeSpec(
                name="Introduction",
                page_title=intro_candidate.title,
                subsystem_ids=sub_ids_map.get(
                    intro_candidate.title, [intro_candidate.subsystem_id]
                ),
                children=[],
            )
        else:
            intro_node = SidebarNodeSpec(
                name="Introduction", page_title=None, subsystem_ids=[], children=[]
            )
        nodes.insert(0, intro_node)

    # Re-enforce max_top_nodes after mandatory insertion
    nodes = nodes[:max_top_nodes]

    return SidebarTreeSpec(nodes=nodes)


def _build_fallback_sidebar_tree(
    page_summaries: list[PageSummary],
) -> SidebarTreeSpec:
    """Build a flat (no nesting) sidebar tree as a fallback."""
    nodes: list[SidebarNodeSpec] = []
    for ps in page_summaries:
        nodes.append(SidebarNodeSpec(
            name=ps.subsystem_name,
            page_title=ps.title,
            subsystem_ids=[ps.subsystem_id],
            children=[],
        ))
    return SidebarTreeSpec(nodes=nodes)


def _save_sidebar_tree(
    *,
    repo_hash: str,
    tree: SidebarTreeSpec,
    title_to_page: dict[str, int],
    title_to_subsystem_ids: dict[str, list[int]],
) -> None:
    """Write the sidebar tree to the database.

    Inserts top-level nodes first, then children, relying on node_id
    insertion order for display ordering.
    """
    adapter = get_default_adapter()
    with adapter.session() as write_session:
        wiki_manager = WikiManager(write_session)

        for node in tree["nodes"]:
            # Resolve page_id from title
            page_id = title_to_page.get(node["page_title"]) if node["page_title"] else None
            sub_ids = node["subsystem_ids"]
            if not sub_ids and node["page_title"]:
                sub_ids = title_to_subsystem_ids.get(node["page_title"], [])

            is_active = page_id is not None or bool(node["children"])

            parent = wiki_manager.add_sidebar(
                repo_hash=repo_hash,
                parent_node_id=None,
                name=node["name"],
                page_id=page_id,
                is_active=is_active,
                sub_system_ids=sub_ids,
            )

            # Insert children with parent_node_id set
            for child in node["children"]:
                child_page_id = (
                    title_to_page.get(child["page_title"])
                    if child["page_title"] else None
                )
                child_sub_ids = child["subsystem_ids"]
                if not child_sub_ids and child["page_title"]:
                    child_sub_ids = title_to_subsystem_ids.get(child["page_title"], [])

                child_is_active = child_page_id is not None

                wiki_manager.add_sidebar(
                    repo_hash=repo_hash,
                    parent_node_id=parent.node_id,
                    name=child["name"],
                    page_id=child_page_id,
                    is_active=child_is_active,
                    sub_system_ids=child_sub_ids,
                )

        write_session.commit()


def _generate_and_save_sidebar_tree(
    *,
    repo_hash: str,
    repo_full_name: str,
    page_summaries: list[PageSummary],
    model: str,
) -> None:
    """Call LLM to generate sidebar tree, then persist to DB."""
    # Build title → page_id lookup
    title_to_page: dict[str, int] = {}
    title_to_subsystem_ids: dict[str, list[int]] = {}
    for ps in page_summaries:
        title_to_page[ps.title] = ps.page_id
        title_to_subsystem_ids.setdefault(ps.title, []).append(ps.subsystem_id)

    valid_titles = set(title_to_page.keys())

    # Identify best candidate for the mandatory Introduction node (same logic as user prompt)
    intro_candidate: PageSummary | None = None
    if page_summaries:
        intro_candidate = max(page_summaries, key=lambda p: _score_page(p, _INTRO_SIGNALS))

    # Build and send prompt
    user_prompt = _build_sidebar_user_prompt(repo_full_name, page_summaries)
    system_prompt = WIKI_SIDEBAR_SYSTEM_PROMPT.format(
        max_top_nodes=WIKI_SIDEBAR_MAX_TOP_NODES,
        max_children=WIKI_SIDEBAR_MAX_CHILDREN,
    )

    results = run_batch([OpenAIRequest(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        response_format={"type": "json_object"},
    )], max_concurrency=1)
    if isinstance(results[0], Exception):
        raise results[0]
    parsed = _parse_json_object(results[0])

    try:
        tree = _parse_sidebar_tree(
            json.dumps(parsed),
            valid_titles,
            max_top_nodes=WIKI_SIDEBAR_MAX_TOP_NODES,
            max_children=WIKI_SIDEBAR_MAX_CHILDREN,
            intro_candidate=intro_candidate,
            title_to_subsystem_ids=title_to_subsystem_ids,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Sidebar LLM returned invalid tree for repo_hash=%s: %s. "
            "Falling back to flat sidebar.",
            repo_hash, exc,
        )
        tree = _build_fallback_sidebar_tree(page_summaries)

    # Persist tree to DB
    _save_sidebar_tree(
        repo_hash=repo_hash,
        tree=tree,
        title_to_page=title_to_page,
        title_to_subsystem_ids=title_to_subsystem_ids,
    )

    logger.info(
        "Sidebar tree saved: %d top-level nodes for repo_hash=%s",
        len(tree["nodes"]),
        repo_hash,
    )


# ---------------------------------------------------------------------------
# Page generation helpers
# ---------------------------------------------------------------------------

def _build_page_user_prompt(
    *,
    repo_full_name: str,
    repo_clone_path: Path,
    snap: "SubsystemSnapshot",  # type: ignore[name-defined]
    files: list[RepoFile],
) -> str:
    """Build the user prompt for a wiki page — pure data, no LLM call."""
    lines: list[str] = []
    lines.append("Context:")
    lines.append(f"REPO: {repo_full_name}")
    lines.append(f"GROUP_NAME: {snap.name}")
    lines.append(f"GROUP_DESCRIPTION: {snap.description}")
    lines.append(f"FILE_IDS: {json.dumps(snap.file_ids)}")
    lines.append("")
    lines.append("Writing goals:")
    lines.append("- Explain expected behavior and workflows, not file descriptions.")
    lines.append("- Assume the reader wants to use the system and understand its architecture.")
    lines.append("- Use markdown headings (##/###/####), bullets, and short paragraphs.")
    lines.append("- Provide concrete steps or sequences when describing tasks.")
    lines.append("")
    lines.append("FILES:")
    for file in files:
        file_path = repo_clone_path / Path(file.full_rel_path())
        meta = file.get_metadata() or {}
        file_summary = str(meta.get("file_summary") or "").strip()
        if file_summary and (file.file_size or 0) > 10_240:
            content = ""
        else:
            content = _read_file_text(file_path)
        lines.append(f"FILE_ID: {file.file_id}")
        lines.append(f"FILE_PATH: {file.full_rel_path()}")
        lines.append(f"FILE_SIZE_BYTES: {file.file_size}")
        if file_summary and (file.file_size or 0) > 10_240:
            lines.append("FILE_SUMMARY:")
            lines.append(file_summary)
        else:
            lines.append("FILE_CONTENT:")
            lines.append(content)
        lines.append("END_FILE")
        lines.append("")
    return "\n".join(lines)



# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _read_file_text(path: Path, *, max_bytes: int = 120_000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")


def _parse_json_object(text: str) -> dict[str, object]:
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object response.")
        return parsed
    except json.JSONDecodeError:
        # Attempt to salvage common invalid escape issues by escaping stray backslashes.
        sanitized = _escape_invalid_backslashes(text)
        parsed = json.loads(sanitized)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object response.")
        return parsed


_INVALID_ESCAPE_RE = re.compile(r"\\(?![\"\\/bfnrtu])")


def _escape_invalid_backslashes(text: str) -> str:
    return _INVALID_ESCAPE_RE.sub(r"\\\\", text)


_CONTENT_NODE_MAX_CHARS: int = 2000


def _normalize_page_spec(item: dict[str, object]) -> WikiPageSpec:
    title = str(item.get("title") or "").strip()
    contents_raw = item.get("contents")
    contents: list[WikiContentSpec] = []
    if isinstance(contents_raw, list):
        for node in contents_raw:
            if not isinstance(node, dict):
                continue
            node_title = str(node.get("title") or "").strip()
            content_type = str(node.get("content_type") or "markdown")
            content = str(node.get("content") or "")
            # Hard cap: split oversized nodes rather than silently truncating.
            chunks = _split_content(content, max_chars=_CONTENT_NODE_MAX_CHARS)
            source_file_ids = _ensure_int_list(node.get("source_file_ids"))
            for idx, chunk in enumerate(chunks):
                chunk_title = node_title if idx == 0 else f"{node_title} (cont.)"
                contents.append({
                    "title": chunk_title,
                    "content_type": content_type,
                    "content": chunk,
                    "source_file_ids": source_file_ids,
                })
    return {
        "title": title or "Subsystem Overview",
        "contents": contents,
    }


def _split_content(text: str, *, max_chars: int) -> list[str]:
    """Split content into chunks of at most max_chars, breaking on paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        # Try to split at the last blank line within the limit
        window = remaining[:max_chars]
        split_pos = window.rfind("\n\n")
        if split_pos == -1:
            # No paragraph break — fall back to last newline
            split_pos = window.rfind("\n")
        if split_pos == -1:
            # No newline at all — hard cut
            split_pos = max_chars
        chunks.append(remaining[:split_pos].rstrip())
        remaining = remaining[split_pos:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


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
        progress=task.get_progress(),
    )
