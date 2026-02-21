"""Manager for Wiki models: CRUD using a DB session."""

from sqlalchemy.orm import Session

from repo_analyzer.models import WikiPage, WikiPageContent, WikiSidebar


class WikiManager:
    """Provides access to wiki models. Takes a DB session as input."""

    def __init__(self, session: Session):
        self._session = session

    def delete_by_repo(self, repo_hash: str) -> None:
        page_ids = [
            row[0]
            for row in self._session.query(WikiPage.page_id)
            .filter(WikiPage.repo_hash == repo_hash)
            .all()
        ]
        if page_ids:
            self._session.query(WikiPageContent).filter(
                WikiPageContent.page_id.in_(page_ids)
            ).delete(synchronize_session=False)
        self._session.query(WikiPage).filter(WikiPage.repo_hash == repo_hash).delete()
        self._session.query(WikiSidebar).filter(WikiSidebar.repo_hash == repo_hash).delete()

    def add_sidebar(
        self,
        *,
        repo_hash: str,
        parent_node_id: int | None,
        name: str,
        page_id: int | None,
        is_active: bool,
        sub_system_ids: list[int],
    ) -> WikiSidebar:
        sidebar = WikiSidebar(
            repo_hash=repo_hash,
            parent_node_id=parent_node_id,
            name=name,
            page_id=page_id,
            is_active=is_active,
        )
        sidebar.set_meta({"sub_system_ids": sub_system_ids})
        self._session.add(sidebar)
        self._session.flush()
        return sidebar

    def add_page(self, *, repo_hash: str, title: str, subsystem_ids: list[int]) -> WikiPage:
        page = WikiPage(
            repo_hash=repo_hash,
            title=title,
        )
        page.set_meta({"subsystem_ids": subsystem_ids})
        self._session.add(page)
        self._session.flush()
        return page

    def add_content(
        self,
        *,
        page_id: int,
        content_type: str,
        content: str,
        source_file_ids: list[int],
    ) -> WikiPageContent:
        node = WikiPageContent(
            page_id=page_id,
            content_type=content_type,
            content=content,
        )
        node.set_meta({"source_file_ids": source_file_ids})
        self._session.add(node)
        self._session.flush()
        return node

    def list_sidebars(self, repo_hash: str) -> list[WikiSidebar]:
        return (
            self._session.query(WikiSidebar)
            .filter(WikiSidebar.repo_hash == repo_hash)
            .order_by(WikiSidebar.node_id)
            .all()
        )

    def list_pages(self, repo_hash: str) -> list[WikiPage]:
        return (
            self._session.query(WikiPage)
            .filter(WikiPage.repo_hash == repo_hash)
            .order_by(WikiPage.page_id)
            .all()
        )

    def list_page_contents(self, page_id: int) -> list[WikiPageContent]:
        return (
            self._session.query(WikiPageContent)
            .filter(WikiPageContent.page_id == page_id)
            .order_by(WikiPageContent.content_id)
            .all()
        )
