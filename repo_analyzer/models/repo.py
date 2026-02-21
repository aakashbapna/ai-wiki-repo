"""Repo model for cloned repositories."""

import hashlib
import time
from pathlib import Path

from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base








class Repo(Base):
    """Cloned repository record."""

    __tablename__ = "repos"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    repo_hash: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    owner: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    repo_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    clone_path: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[int] = mapped_column(
        Integer, nullable=False, default=lambda: int(time.time())
    )  # UTC epoch

    @staticmethod
    def parse_owner_repo(repo_url: str) -> tuple[str, str]:
        """Parse owner and repo name from a Git URL. Returns (owner, repo_name)."""
        url = repo_url.rstrip("/").removesuffix(".git").rstrip("/")
        parts = [p for p in url.split("/") if p]
        if len(parts) >= 2:
            return parts[-2], parts[-1]
        if len(parts) == 1:
            return "", parts[0]

    @staticmethod
    def compute_hash(owner: str, repo_name: str) -> str:
        """Unique hash for a repo from 'owner/repo' (case-normalized). Uses MD5 (32 hex chars)."""
        key = f"{owner.strip().lower()}/{repo_name.strip().lower()}"
        return hashlib.md5(key.encode()).hexdigest()

    def __repr__(self) -> str:
        return f"<Repo {self.owner}/{self.repo_name} hash={self.repo_hash}>"

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo_name}" if self.owner else self.repo_name

    @property
    def clone_path_resolved(self) -> Path:
        return Path(self.clone_path)
