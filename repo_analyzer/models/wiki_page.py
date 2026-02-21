"""WikiPage model."""

import json
import time
from typing import TypedDict

from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class WikiPageMeta(TypedDict, total=False):
    subsystem_ids: list[int]


def wiki_page_meta_to_json(meta: WikiPageMeta | None) -> str | None:
    if meta is None:
        return None
    return json.dumps({
        "subsystem_ids": meta.get("subsystem_ids", []),
    })


def wiki_page_meta_from_json(s: str | None) -> WikiPageMeta | None:
    if not s or not s.strip():
        return None
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            return WikiPageMeta(
                subsystem_ids=[int(x) for x in (data.get("subsystem_ids") or [])],
            )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


class WikiPage(Base):
    """Wiki page for a subsystem or topic."""

    __tablename__ = "wiki_pages"

    page_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_hash: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    meta_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=lambda: int(time.time()))
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False, default=lambda: int(time.time()))

    def __repr__(self) -> str:
        return f"<WikiPage {self.page_id} {self.repo_hash} {self.title!r}>"

    def get_meta(self) -> WikiPageMeta | None:
        return wiki_page_meta_from_json(self.meta_json)

    def set_meta(self, meta: WikiPageMeta | None) -> None:
        self.meta_json = wiki_page_meta_to_json(meta)
