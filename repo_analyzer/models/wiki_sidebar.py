"""WikiSidebar model: navigation nodes for wiki pages."""

import json
import time
from typing import TypedDict

from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class WikiSidebarMeta(TypedDict, total=False):
    sub_system_ids: list[int]


def wiki_sidebar_meta_to_json(meta: WikiSidebarMeta | None) -> str | None:
    if meta is None:
        return None
    return json.dumps({
        "sub_system_ids": meta.get("sub_system_ids", []),
    })


def wiki_sidebar_meta_from_json(s: str | None) -> WikiSidebarMeta | None:
    if not s or not s.strip():
        return None
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            return WikiSidebarMeta(
                sub_system_ids=[int(x) for x in (data.get("sub_system_ids") or [])],
            )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


class WikiSidebar(Base):
    """Sidebar node for wiki navigation."""

    __tablename__ = "wiki_sidebars"

    node_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_hash: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    parent_node_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    page_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    meta_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=lambda: int(time.time()))
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False, default=lambda: int(time.time()))

    def __repr__(self) -> str:
        return f"<WikiSidebar {self.node_id} {self.repo_hash} {self.name!r}>"

    def get_meta(self) -> WikiSidebarMeta | None:
        return wiki_sidebar_meta_from_json(self.meta_json)

    def set_meta(self, meta: WikiSidebarMeta | None) -> None:
        self.meta_json = wiki_sidebar_meta_to_json(meta)
