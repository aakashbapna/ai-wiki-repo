"""Build wiki pages and sidebars from subsystems."""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from openai import OpenAI
from sqlalchemy.orm import Session

from constants import (
    BUILD_WIKI_MAX_CONCURRENCY,
    STALE_TASK_TIMEOUT_SECONDS,
    WIKI_SIDEBAR_MAX_CHILDREN,
    WIKI_SIDEBAR_MAX_TOP_NODES,
)
from repo_analyzer.db import get_default_adapter
from repo_analyzer.db_managers import RepoManager, WikiManager
from repo_analyzer.models import IndexTask, Repo, RepoFile, RepoSubsystem
from repo_analyzer.models.index_task import TaskStatus, TaskType, is_task_stale
from repo_analyzer.prompts import WIKI_BUILDER_SYSTEM_PROMPT, WIKI_SIDEBAR_SYSTEM_PROMPT

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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Phase 1 + Phase 2 orchestration
# ---------------------------------------------------------------------------

def _rebuild_wiki(
    session: Session,
    repo: Repo,
    subsystems: list[RepoSubsystem],
    task: IndexTask,
    *,
    model: str,
) -> None:
    """
    Phase 1: Generate one wiki page per subsystem concurrently.
    Phase 2: Single LLM call to organize pages into a sidebar tree.
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

    def _process_subsystem(snap: SubsystemSnapshot) -> PageSummary | None:
        """Generate and persist one wiki page inside its own DB session.

        Returns PageSummary for sidebar generation, or None for empty subsystems.
        """
        with adapter.session() as write_session:
            wiki_manager = WikiManager(write_session)

            if not snap.file_ids:
                # No files → no page → skip (sidebar handled in Phase 2)
                return None

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
            for node in content_nodes:
                wiki_manager.add_content(
                    page_id=page.page_id,
                    title=node["title"],
                    content_type=node["content_type"],
                    content=node["content"],
                    source_file_ids=node["source_file_ids"],
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

            # Extract keywords from snapshot meta
            meta = json.loads(snap.meta_json) if snap.meta_json else {}
            keywords_raw = meta.get("keywords", [])
            keywords = keywords_raw if isinstance(keywords_raw, list) else []

            return PageSummary(
                page_id=page.page_id,
                title=page_spec["title"],
                subsystem_id=snap.subsystem_id,
                subsystem_name=snap.name,
                subsystem_description=snap.description,
                keywords=[str(k) for k in keywords],
            )

    # Phase 1: Generate wiki pages concurrently
    page_summaries: list[PageSummary] = []
    with ThreadPoolExecutor(max_workers=BUILD_WIKI_MAX_CONCURRENCY) as executor:
        futures = {executor.submit(_process_subsystem, snap): snap for snap in snapshots}
        for future in as_completed(futures):
            snap = futures[future]
            try:
                result = future.result()
                if result is not None:
                    page_summaries.append(result)
            except Exception as exc:
                logger.exception(
                    "Wiki page generation failed for subsystem=%r repo_hash=%s: %s",
                    snap.name,
                    repo_hash,
                    exc,
                )

    # Phase 2: Generate sidebar tree via single LLM call
    if page_summaries:
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

_ARCH_SIGNALS = frozenset([
    "architecture", "design", "structure", "system", "diagram", "flow",
    "infrastructure", "layout", "topology", "pipeline", "framework", "core",
    "engine", "platform", "model", "schema", "data model",
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

    Identifies the best candidates for the mandatory Introduction and Architecture
    nodes so the LLM can assign them without guessing.
    """
    # Score each page for intro and architecture fit
    intro_candidate: PageSummary | None = None
    arch_candidate: PageSummary | None = None

    if page_summaries:
        intro_ranked = sorted(page_summaries, key=lambda p: _score_page(p, _INTRO_SIGNALS), reverse=True)
        arch_ranked = sorted(page_summaries, key=lambda p: _score_page(p, _ARCH_SIGNALS), reverse=True)
        intro_candidate = intro_ranked[0]
        # Architecture candidate should be different from intro candidate if possible
        arch_candidate = next(
            (p for p in arch_ranked if p.page_id != intro_candidate.page_id),
            arch_ranked[0],
        )

    lines: list[str] = []
    lines.append(f"Organize these {len(page_summaries)} wiki pages into a sidebar tree.")
    lines.append(f"REPO: {repo_full_name}")
    lines.append(f"TOTAL_PAGES: {len(page_summaries)}")
    lines.append("")

    if intro_candidate:
        lines.append(f"INTRO_CANDIDATE: {intro_candidate.title}")
    if arch_candidate:
        lines.append(f"ARCH_CANDIDATE: {arch_candidate.title}")
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
    arch_candidate: PageSummary | None = None,
    title_to_subsystem_ids: dict[str, list[int]] | None = None,
) -> SidebarTreeSpec:
    """Parse LLM response into a validated SidebarTreeSpec.

    Truncates to max_top_nodes top-level and max_children per node.
    Validates page_title references against valid_titles.
    Ensures mandatory 'Introduction' and 'Architecture' nodes exist,
    assigning best-fit pages when the LLM omits them.
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

    def _make_mandatory_node(name: str, candidate: PageSummary | None) -> SidebarNodeSpec:
        """Build a mandatory node, assigning the candidate page if available."""
        if candidate and candidate.title in valid_titles:
            return SidebarNodeSpec(
                name=name,
                page_title=candidate.title,
                subsystem_ids=sub_ids_map.get(candidate.title, [candidate.subsystem_id]),
                children=[],
            )
        return SidebarNodeSpec(name=name, page_title=None, subsystem_ids=[], children=[])

    # Ensure mandatory nodes exist (Introduction first, Architecture second)
    node_names = [n["name"].lower() for n in nodes]
    if "introduction" not in node_names:
        nodes.insert(0, _make_mandatory_node("Introduction", intro_candidate))
    if "architecture" not in node_names:
        intro_idx = next(
            (i for i, n in enumerate(nodes) if n["name"].lower() == "introduction"), 0
        )
        nodes.insert(intro_idx + 1, _make_mandatory_node("Architecture", arch_candidate))

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

    # Identify best candidates for mandatory nodes (same logic as user prompt)
    intro_candidate: PageSummary | None = None
    arch_candidate: PageSummary | None = None
    if page_summaries:
        intro_ranked = sorted(page_summaries, key=lambda p: _score_page(p, _INTRO_SIGNALS), reverse=True)
        arch_ranked = sorted(page_summaries, key=lambda p: _score_page(p, _ARCH_SIGNALS), reverse=True)
        intro_candidate = intro_ranked[0]
        arch_candidate = next(
            (p for p in arch_ranked if p.page_id != intro_candidate.page_id),
            arch_ranked[0],
        )

    # Build and send prompt
    user_prompt = _build_sidebar_user_prompt(repo_full_name, page_summaries)
    system_prompt = WIKI_SIDEBAR_SYSTEM_PROMPT.format(
        max_top_nodes=WIKI_SIDEBAR_MAX_TOP_NODES,
        max_children=WIKI_SIDEBAR_MAX_CHILDREN,
    )

    parsed = _openai_json_object_response(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    try:
        tree = _parse_sidebar_tree(
            json.dumps(parsed),
            valid_titles,
            max_top_nodes=WIKI_SIDEBAR_MAX_TOP_NODES,
            max_children=WIKI_SIDEBAR_MAX_CHILDREN,
            intro_candidate=intro_candidate,
            arch_candidate=arch_candidate,
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
        content = ""
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
    user_prompt = "\n".join(lines)
    parsed = _openai_json_object_response(
        model=model,
        system_prompt=WIKI_BUILDER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    return _normalize_page_spec(parsed)


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
    parsed = _openai_json_object_response(
        model=model,
        system_prompt=WIKI_BUILDER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    return _normalize_page_spec(parsed)


def _build_user_prompt(repo: Repo, subsystem: RepoSubsystem, files: list[RepoFile]) -> str:
    lines: list[str] = []
    meta = subsystem.get_meta() or {}
    lines.append("Context:")
    lines.append(f"REPO: {repo.full_name}")
    lines.append(f"GROUP_NAME: {subsystem.name}")
    lines.append(f"GROUP_DESCRIPTION: {subsystem.description}")
    lines.append(f"FILE_IDS: {json.dumps(meta.get('file_ids') or [])}")
    lines.append("")
    lines.append("Writing goals:")
    lines.append("- Explain expected behavior and workflows, not file descriptions.")
    lines.append("- Assume the reader wants to use the system and understand its architecture.")
    lines.append("- Use markdown headings (##/###/####), bullets, and short paragraphs.")
    lines.append("- Provide concrete steps or sequences when describing tasks.")
    lines.append("")
    lines.append("FILES:")
    for file in files:
        file_path = repo.clone_path_resolved / Path(file.full_rel_path())
        meta = file.get_metadata() or {}
        file_summary = str(meta.get("file_summary") or "").strip()
        content = ""
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


def _openai_response(*, model: str, system_prompt: str, user_prompt: str) -> str:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    raise ValueError("Unexpected OpenAI response format.")


def _openai_json_object_response(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, object]:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise ValueError("Unexpected OpenAI response format.")
    return _parse_json_object(content)


def _get_openai_client() -> OpenAI:
    return OpenAI()


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
            source_file_ids = _ensure_int_list(node.get("source_file_ids"))
            contents.append({
                "title": node_title,
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
