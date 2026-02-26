"""Export wiki sidebar and pages as markdown files for GitHub-renderable docs."""

import re
import shutil
from pathlib import Path
from typing import TypedDict

from repo_analyzer.db import get_default_adapter
from repo_analyzer.db_managers import RepoManager, WikiManager
from repo_analyzer.services.repo.repo_git_utils import (
    WIKI_BRANCH,
    checkout_or_create_branch,
    push_branch,
    stage_and_commit,
)


class WikiExportResult(TypedDict):
    repo_hash: str
    branch: str
    wiki_path: str
    files_written: int
    commit_sha: str
    pr_url: str


def _build_pr_url(repo_url: str, branch: str) -> str:
    """Build GitHub/GitLab PR creation URL from repo URL and branch.

    Supports GitHub.com and GitHub Enterprise
    """
    url = repo_url.rstrip("/").removesuffix(".git").rstrip("/")
    if url.startswith("git@"):
        url = url.replace(":", "/", 1).replace("git@", "https://")
    if url.startswith("https://github.") or url.startswith("http://github."):
        return f"{url}/pull/new/{branch}"
    if url.startswith("https://gitlab.com/") or url.startswith("http://gitlab.com/"):
        return f"{url}/-/merge_requests/new?merge_request[source_branch]={branch}"
    if "github." in url and "gitlab" not in url:
        base = url.split("github.", 1)[-1].lstrip("/")
        return f"https://github.{base}/pull/new/{branch}"
    if "gitlab.com" in url:
        base = url.split("gitlab.com")[-1].strip("/")
        return f"https://gitlab.com{base}/-/merge_requests/new?merge_request[source_branch]={branch}"
    return ""


def _title_to_slug(title: str) -> str:
    """Convert page title to a filesystem-safe slug for markdown filename."""
    slug = re.sub(r"[^\w\s-]", "", title)
    slug = re.sub(r"[-\s]+", "-", slug).strip("-").lower()
    return slug or "untitled"


def _build_page_markdown(
    title: str,
    contents: list[dict[str, object]],
) -> str:
    """Build full markdown content for a wiki page."""
    lines: list[str] = [f"# {title}", ""]
    for block in contents:
        block_title = (block.get("meta") or {}).get("title") or ""
        content = block.get("content") or ""
        content_type = block.get("content_type") or "markdown"
        if block_title:
            lines.append(f"## {block_title}")
            lines.append("")
        if content_type == "markdown":
            lines.append(content.strip())
        else:
            lines.append("```")
            lines.append(content.strip())
            lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _build_sidebar_markdown(
    nodes: list[dict[str, object]],
    page_id_to_path: dict[int, str],
) -> str:
    """Build _Sidebar.md with nested navigation links."""
    lines: list[str] = ["# Navigation", ""]

    def emit_node(node: dict[str, object], indent: int = 0) -> None:
        name = node.get("name") or "Untitled"
        page_id = node.get("page_id")
        prefix = "  " * indent
        if page_id is not None and page_id in page_id_to_path:
            path = page_id_to_path[page_id]
            lines.append(f"{prefix}- [{name}]({path})")
        else:
            lines.append(f"{prefix}- **{name}**")
        children = node.get("children") or []
        for child in children:
            emit_node(child, indent + 1)

    for node in nodes:
        emit_node(node)

    return "\n".join(lines).rstrip() + "\n"


def _build_sidebar_tree(
    flat_nodes: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Convert flat sidebar list (with parent_node_id) to nested tree."""
    by_id: dict[int | None, list[dict[str, object]]] = {}
    node_map: dict[int, dict[str, object]] = {}
    for n in flat_nodes:
        node_id = n.get("node_id")
        parent_id = n.get("parent_node_id")
        node_copy = {**n, "children": []}
        if node_id is not None:
            node_map[node_id] = node_copy
        if parent_id not in by_id:
            by_id[parent_id] = []
        by_id[parent_id].append(node_copy)

    for parent_id, children in by_id.items():
        if parent_id is not None and parent_id in node_map:
            parent = node_map[parent_id]
            parent["children"] = children

    return by_id.get(None, [])


def export_wiki_to_repo(
    repo_hash: str,
    *,
    push: bool = True,
) -> WikiExportResult:
    """
    Export wiki sidebar and pages to a wiki/ folder in the repo, on branch ai-repo-wiki.

    Creates wiki/README.md, wiki/_Sidebar.md, and wiki/<slug>.md for each page.
    Checks out ai-repo-wiki, writes files, commits, and optionally pushes.

    Args:
        repo_hash: Repository hash.
        push: Whether to push the branch to remote after commit.

    Returns:
        WikiExportResult with paths and counts.
    """
    adapter = get_default_adapter()
    with adapter.session() as session:
        repo_manager = RepoManager(session)
        repo = repo_manager.get_by_hash(repo_hash)
        if repo is None:
            raise ValueError(f"Repo not found: {repo_hash}")

        wiki_manager = WikiManager(session)
        sidebars = wiki_manager.list_sidebars(repo_hash)
        pages = wiki_manager.list_pages(repo_hash)

    if not pages:
        raise ValueError(f"No wiki pages found for repo: {repo_hash}. Build the wiki first.")

    repo_path = repo.clone_path_resolved
    wiki_dir = repo_path / "wiki"
    if wiki_dir.exists():
        shutil.rmtree(wiki_dir)
    wiki_dir.mkdir(parents=True)

    checkout_or_create_branch(repo_path, branch=WIKI_BRANCH)

    page_id_to_path: dict[int, str] = {}
    files_written = 0

    sidebar_nodes = [
        {
            "node_id": n.node_id,
            "parent_node_id": n.parent_node_id,
            "name": n.name,
            "page_id": n.page_id,
            "children": [],
        }
        for n in sidebars
    ]
    tree = _build_sidebar_tree(sidebar_nodes)

    def assign_paths(
        nodes: list[dict[str, object]],
        parent_path: str = "",
    ) -> None:
        for node in nodes:
            slug = _title_to_slug(node.get("name") or "untitled")
            if parent_path:
                rel_path = f"{parent_path}/{slug}.md"
                dir_path = wiki_dir / parent_path
                dir_path.mkdir(parents=True, exist_ok=True)
            else:
                rel_path = f"{slug}.md"
            page_id = node.get("page_id")
            if page_id is not None:
                page_id_to_path[page_id] = rel_path
            children = node.get("children") or []
            child_parent = f"{parent_path}/{slug}" if parent_path else slug
            assign_paths(children, child_parent)

    assign_paths(tree)

    for page in pages:
        if page.page_id not in page_id_to_path:
            slug = _title_to_slug(page.title)
            page_id_to_path[page.page_id] = f"{slug}.md"
        rel_path = page_id_to_path[page.page_id]
        with adapter.session() as session:
            wiki_manager = WikiManager(session)
            contents = wiki_manager.list_page_contents(page.page_id)
        content_dicts = [
            {
                "content_type": c.content_type,
                "content": c.content,
                "meta": c.get_meta(),
            }
            for c in contents
        ]
        md = _build_page_markdown(page.title, content_dicts)
        out_path = wiki_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        files_written += 1

    sidebar_md = _build_sidebar_markdown(tree, page_id_to_path)
    (wiki_dir / "_Sidebar.md").write_text(sidebar_md, encoding="utf-8")
    files_written += 1

    readme_lines = [
        f"# {repo.full_name} — AI-Generated Wiki",
        "",
        "This wiki was automatically generated from the repository structure.",
        "",
        "## Navigation",
        "",
        "See [_Sidebar.md](_Sidebar.md) for the full navigation tree.",
        "",
    ]
    for page in pages:
        path = page_id_to_path.get(page.page_id, f"{_title_to_slug(page.title)}.md")
        readme_lines.append(f"- [{page.title}]({path})")
    readme_lines.append("")
    (wiki_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    files_written += 1

    paths_to_stage = [
        str(f.relative_to(repo_path))
        for f in wiki_dir.rglob("*")
        if f.is_file()
    ]
    commit_sha = stage_and_commit(
        repo_path,
        paths=paths_to_stage,
        message="Add AI-generated wiki (sidebar and pages)",
    )
    commit_sha_str = commit_sha.decode("utf-8") if isinstance(commit_sha, bytes) else str(commit_sha)

    if push:
        push_branch(repo_path, branch=WIKI_BRANCH)

    pr_url = _build_pr_url(repo.url, WIKI_BRANCH)

    return WikiExportResult(
        repo_hash=repo_hash,
        branch=WIKI_BRANCH,
        wiki_path=str(wiki_dir),
        files_written=files_written,
        commit_sha=commit_sha_str,
        pr_url=pr_url,
    )
