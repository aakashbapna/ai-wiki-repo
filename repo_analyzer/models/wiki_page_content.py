"""WikiPageContent model."""

import json
import time
from typing import Literal, TypedDict

from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


ContentType = Literal["markdown", "string"]


class WikiPageContentMeta(TypedDict, total=False):
    source_file_ids: list[int]


def wiki_page_content_meta_to_json(meta: WikiPageContentMeta | None) -> str | None:
    if meta is None:
        return None
    return json.dumps({
        "source_file_ids": meta.get("source_file_ids", []),
    })


def wiki_page_content_meta_from_json(s: str | None) -> WikiPageContentMeta | None:
    if not s or not s.strip():
        return None
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            return WikiPageContentMeta(
                source_file_ids=[int(x) for x in (data.get("source_file_ids") or [])],
            )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


class WikiPageContent(Base):
    """Content nodes for a wiki page."""

    __tablename__ = "wiki_page_contents"

    content_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    page_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    content_type: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    meta_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=lambda: int(time.time()))
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False, default=lambda: int(time.time()))

    def __repr__(self) -> str:
        return f"<WikiPageContent {self.content_id} page={self.page_id}>"

    def get_meta(self) -> WikiPageContentMeta | None:
        return wiki_page_content_meta_from_json(self.meta_json)

    def set_meta(self, meta: WikiPageContentMeta | None) -> None:
        self.meta_json = wiki_page_content_meta_to_json(meta)
