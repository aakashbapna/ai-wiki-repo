"""Shared pytest fixtures for all indexer task tests."""

import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

from repo_analyzer.models.base import Base
from repo_analyzer.models.repo import Repo
from repo_analyzer.models.repo_file import RepoFile
from repo_analyzer.models.index_task import IndexTask, TaskType
from repo_analyzer.models.repo_subsystem import RepoSubsystem
from repo_analyzer.models.wiki_page import WikiPage
from repo_analyzer.models.wiki_page_content import WikiPageContent
from repo_analyzer.models.wiki_sidebar import WikiSidebar


# ---------------------------------------------------------------------------
# In-memory SQLite engine + session factory
#
# We use a named shared-cache in-memory database so that multiple connections
# (and therefore multiple SQLAlchemy sessions, including those opened by
# ThreadPoolExecutor workers) all see the same data.  The URI format is:
#   file:<unique-name>?mode=memory&cache=shared
# Each test gets a unique name so tests are fully isolated from each other.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def engine():
    """Fresh shared-cache in-memory SQLite engine per test.

    Using a named URI with cache=shared lets multiple connections (including
    those opened inside ThreadPoolExecutor workers) share the same in-memory
    database, which is critical for testing concurrent DB writes.
    """
    db_name = f"test_{uuid.uuid4().hex}"
    url = f"file:{db_name}?mode=memory&cache=shared"
    eng = create_engine(
        f"sqlite:///{url}",
        connect_args={"check_same_thread": False, "uri": True},
    )
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture(scope="function")
def _session_factory(engine):
    """Shared session factory backed by the test engine."""
    return sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


@pytest.fixture(scope="function")
def session(_session_factory) -> Session:
    """Main test session; closed (not rolled back) after each test so that
    data committed by worker threads remains visible to assertions."""
    s = _session_factory()
    yield s
    s.close()


# ---------------------------------------------------------------------------
# DB adapter mock – thread-safe, backed by the shared in-memory engine
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_adapter(_session_factory, monkeypatch):
    """
    Patch get_default_adapter() everywhere to return an adapter whose
    .session() context-manager creates a **new** session per call from the
    shared in-memory session factory.  This lets ThreadPoolExecutor workers
    each open their own session while still reading/writing the same DB.
    """
    adapter = MagicMock()

    @contextmanager
    def thread_safe_session():
        s = _session_factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    adapter.session.side_effect = thread_safe_session

    # Patch in every module that calls get_default_adapter()
    for module_path in [
        "repo_analyzer.services.file.code_analyzer",
        "repo_analyzer.services.subsystem.subsystem_builder",
        "repo_analyzer.services.wiki.wiki_builder",
        "repo_analyzer.services.file.service",
        "repo_analyzer.services.subsystem.service",
        "repo_analyzer.services.wiki.service",
    ]:
        monkeypatch.setattr(f"{module_path}.get_default_adapter", lambda: adapter)

    return adapter


# ---------------------------------------------------------------------------
# Model factory helpers
# ---------------------------------------------------------------------------

def make_repo(
    session: Session,
    *,
    owner: str = "testowner",
    repo_name: str = "testrepo",
    clone_path: str = "/tmp/testrepo",
    url: str = "https://github.com/testowner/testrepo",
) -> Repo:
    repo = Repo(
        repo_hash=Repo.compute_hash(owner, repo_name),
        owner=owner,
        repo_name=repo_name,
        clone_path=clone_path,
        url=url,
        created_at=int(time.time()),
    )
    session.add(repo)
    session.flush()
    return repo


def make_repo_file(
    session: Session,
    repo: Repo,
    *,
    file_path: str = "src",
    file_name: str = "main.py",
    is_scan_excluded: bool = False,
    last_index_at: int = 0,
    metadata_json: str | None = None,
) -> RepoFile:
    rf = RepoFile(
        repo_hash=repo.repo_hash,
        file_path=file_path,
        file_name=file_name,
        created_at=int(time.time()),
        modified_at=int(time.time()),
        file_size=100,
        last_index_at=last_index_at,
        metadata_json=metadata_json,
        is_scan_excluded=is_scan_excluded,
        is_project_file=False,
        project_name=None,
    )
    session.add(rf)
    session.flush()
    return rf


def make_index_task(
    session: Session,
    repo: Repo,
    *,
    task_type: str = TaskType.INDEX_FILE.value,
    status: str = "running",
    total_files: int = 10,
    completed_files: int = 0,
) -> IndexTask:
    task = IndexTask(
        repo_hash=repo.repo_hash,
        task_type=task_type,
        status=status,
        total_files=total_files,
        completed_files=completed_files,
        created_at=int(time.time()),
        updated_at=int(time.time()),
    )
    session.add(task)
    session.flush()
    return task


def make_subsystem(
    session: Session,
    repo: Repo,
    *,
    name: str = "Core",
    description: str = "Core subsystem",
    file_ids: list[int] | None = None,
    keywords: list[str] | None = None,
) -> RepoSubsystem:
    import json
    meta = {"file_ids": file_ids or [], "keywords": keywords or []}
    sub = RepoSubsystem(
        repo_hash=repo.repo_hash,
        name=name,
        description=description,
        meta_json=json.dumps(meta),
        created_at=int(time.time()),
    )
    session.add(sub)
    session.flush()
    return sub


def make_wiki_page(
    session: Session,
    repo: Repo,
    *,
    title: str = "Test Page",
    subsystem_ids: list[int] | None = None,
) -> WikiPage:
    import json
    meta = {"subsystem_ids": subsystem_ids or []}
    now = int(time.time())
    page = WikiPage(
        repo_hash=repo.repo_hash,
        title=title,
        meta_json=json.dumps(meta),
        created_at=now,
        updated_at=now,
    )
    session.add(page)
    session.flush()
    return page


# ---------------------------------------------------------------------------
# OpenAI mock helpers (re-usable across test files)
# ---------------------------------------------------------------------------

def make_openai_response(text: str) -> MagicMock:
    """Return a mock object that looks like an OpenAI Responses API response."""
    mock_resp = MagicMock()
    mock_resp.output_text = text
    return mock_resp


def make_multi_response_client(*responses: str) -> MagicMock:
    """Return a mock OpenAI client that returns different text for successive calls.

    Useful when a single test triggers multiple LLM calls (e.g. Phase 1 + Phase 2
    in the hierarchical clustering pipeline).
    """
    mock_client = MagicMock()
    mock_responses = [make_openai_response(r) for r in responses]
    mock_client.responses.create.side_effect = mock_responses
    return mock_client
