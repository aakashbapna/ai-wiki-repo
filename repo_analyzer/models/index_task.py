"""IndexTask model: tracks repository indexing progress."""

import time

from enum import Enum

from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class TaskType(str, Enum):
    INDEX_FILE = "index_file"
    BUILD_SUBSYSTEM = "build_subsystem"
    BUILD_WIKI = "build_wiki"


class IndexTask(Base):
    """Tracks indexing tasks for a repo to support resume and status reporting."""

    __tablename__ = "index_tasks"

    task_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_hash: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    task_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=TaskType.INDEX_FILE.value,
        index=True,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    total_files: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_files: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[int] = mapped_column(
        Integer, nullable=False, default=lambda: int(time.time())
    )
    updated_at: Mapped[int] = mapped_column(
        Integer, nullable=False, default=lambda: int(time.time())
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<IndexTask {self.task_id} repo={self.repo_hash} "
            f"status={self.status} completed={self.completed_files}/{self.total_files}>"
        )
