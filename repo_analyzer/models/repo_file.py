"""RepoFile model: one row per file in a cloned repo."""

import json
from typing import TypedDict, Union

from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base


class RepoFileMetadata(TypedDict, total=False):
    """
    Typed structure for metadata_json. Stored as JSON string in DB.
    All timestamps are UTC epoch (int).
    """

    dependent_files: list[str]
    responsibility: str
    entry_point: bool
    key_elements: list[str] | str
    last_index_at: int


def repo_file_metadata_from_dict(data: dict[str, Union[str, int, bool, list[str]]]) -> RepoFileMetadata:
    """Build metadata from dict (e.g. from JSON)."""
    return RepoFileMetadata(
        dependent_files=data.get("dependent_files") or [],
        responsibility=data.get("responsibility") or "",
        entry_point=bool(data.get("entry_point", False)),
        key_elements=data.get("key_elements") or [],
        last_index_at=int(data.get("last_index_at") or 0),
    )


def repo_file_metadata_to_json(meta: RepoFileMetadata | None) -> str | None:
    """Serialize metadata to JSON string for DB."""
    if meta is None:
        return None
    return json.dumps({
        "dependent_files": meta.get("dependent_files", []),
        "responsibility": meta.get("responsibility", ""),
        "entry_point": meta.get("entry_point", False),
        "key_elements": meta.get("key_elements", []),
        "last_index_at": meta.get("last_index_at", 0),
    })


def repo_file_metadata_from_json(s: str | None) -> RepoFileMetadata | None:
    """Parse metadata from JSON string from DB."""
    if not s or not s.strip():
        return None
    try:
        data = json.loads(s)
        return repo_file_metadata_from_dict(data)
    except (json.JSONDecodeError, TypeError):
        return None


class RepoFile(Base):
    """
    A file within a cloned repository.

    Timestamps (created_at, modified_at) are UTC epoch (int).
    metadata_json stores optional JSON matching RepoFileMetadataDict.
    """

    __tablename__ = "repo_files"

    file_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_hash: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)  # path relative to repo root
    file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)  # UTC epoch
    modified_at: Mapped[int] = mapped_column(Integer, nullable=False)  # UTC epoch
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON string
    is_scan_excluded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_project_file: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    project_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    def __repr__(self) -> str:
        return f"<RepoFile {self.file_id} {self.repo_hash} {self.file_path!r}>"

    def get_metadata(self) -> RepoFileMetadata | None:
        """Parse and return typed metadata from metadata_json."""
        return repo_file_metadata_from_json(self.metadata_json)

    def set_metadata(self, meta: RepoFileMetadata | None) -> None:
        """Set metadata_json from typed metadata."""
        self.metadata_json = repo_file_metadata_to_json(meta)
