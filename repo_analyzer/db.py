"""
Database adapter: abstract interface + SQLite implementation.

Allows swapping the backing store (e.g. PostgreSQL) by providing a different
adapter implementation while keeping the same interface.
"""

import os
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from constants import DATA_DIR
from .models.base import Base

# Columns to add to existing tables if missing: (table, column, sql_type, default)
_MIGRATIONS: list[tuple[str, str, str, str]] = [
    ("repo_files", "is_scan_excluded", "BOOLEAN", "0"),
    ("repo_files", "is_project_file", "BOOLEAN", "0"),
    ("repo_files", "project_name", "VARCHAR(255)", "NULL"),
    ("repo_files", "last_index_at", "INTEGER", "0"),
    ("repo_files", "file_size", "INTEGER", "0"),
    ("index_tasks", "task_type", "VARCHAR(32)", "'index_file'"),
]


class DBAdapter(ABC):
    """Abstract database adapter. Implement this to swap backends (e.g. PostgreSQL)."""

    @abstractmethod
    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Yield a session; commits on exit, rolls back on exception."""
        ...

    @abstractmethod
    def create_tables(self) -> None:
        """Create all tables defined in models."""
        ...

    @abstractmethod
    def migrate_tables(self) -> None:
        """Add any missing columns to existing tables (forward-only migrations)."""
        ...


class SQLiteAdapter(DBAdapter):
    """SQLite implementation of the DB adapter."""

    def __init__(self, url: str = "sqlite:///data/repos.db", *, echo: bool = False):
        self._url = url
        self._engine = create_engine(url, echo=echo, connect_args={"check_same_thread": False})
        self._session_factory = sessionmaker(
            self._engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_tables(self) -> None:
        Base.metadata.create_all(bind=self._engine)

    def migrate_tables(self) -> None:
        """Add missing columns declared in _MIGRATIONS to existing SQLite tables."""
        # Strip the leading "sqlite:///" to get the raw file path
        db_path = self._url.replace("sqlite:///", "", 1)
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            for table, column, sql_type, default in _MIGRATIONS:
                cur.execute(f"PRAGMA table_info({table})")
                existing = {row[1] for row in cur.fetchall()}
                if column not in existing:
                    cur.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {sql_type} NOT NULL DEFAULT {default}"
                    )
            conn.commit()
        finally:
            conn.close()


def get_default_adapter() -> DBAdapter:
    """Build the default adapter from environment/config."""
    url = os.environ.get("DATABASE_URL")
    if url:
        # Use DATABASE_URL (e.g. sqlite:///path or postgresql://... for future backends)
        if url.startswith("sqlite"):
            return SQLiteAdapter(url)
        # Add other backends here (e.g. PostgreSQL) by returning a different adapter
        raise ValueError("Only sqlite:// URLs are supported. Set DATABASE_URL to a sqlite path.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DATA_DIR / "repos.db"
    return SQLiteAdapter(f"sqlite:///{db_path}")
