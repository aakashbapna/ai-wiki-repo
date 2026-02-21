"""RepoSubsystem model: logical subsystems within a repository."""

import json
import time
from typing import TypedDict, Union

from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base


class RepoSubsystemMeta(TypedDict, total=False):
    file_ids: list[int]
    keywords: list[str]


def repo_subsystem_meta_from_dict(
    data: dict[str, Union[str, int, list[int], list[str]]]
) -> RepoSubsystemMeta:
    return RepoSubsystemMeta(
        file_ids=[int(x) for x in (data.get("file_ids") or [])],
        keywords=[str(x) for x in (data.get("keywords") or [])],
    )


def repo_subsystem_meta_to_json(meta: RepoSubsystemMeta | None) -> str | None:
    if meta is None:
        return None
    return json.dumps({
        "file_ids": meta.get("file_ids", []),
        "keywords": meta.get("keywords", []),
    })


def repo_subsystem_meta_from_json(s: str | None) -> RepoSubsystemMeta | None:
    if not s or not s.strip():
        return None
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            return repo_subsystem_meta_from_dict(data)
    except (json.JSONDecodeError, TypeError):
        return None
    return None


class RepoSubsystem(Base):
    """Subsystem row tied to a repository."""

    __tablename__ = "repo_subsystems"

    subsystem_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_hash: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    meta_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=lambda: int(time.time()))

    def __repr__(self) -> str:
        return f"<RepoSubsystem {self.subsystem_id} {self.repo_hash} {self.name!r}>"

    def get_meta(self) -> RepoSubsystemMeta | None:
        return repo_subsystem_meta_from_json(self.meta_json)

    def set_meta(self, meta: RepoSubsystemMeta | None) -> None:
        self.meta_json = repo_subsystem_meta_to_json(meta)
