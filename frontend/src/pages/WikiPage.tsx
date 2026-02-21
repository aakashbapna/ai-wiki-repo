import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  fetchRepoDetail,
  fetchWikiPage,
  fetchWikiSidebars,
  RepoSummary,
  WikiPageWithContents,
  WikiSidebarNode,
} from "../api";

function pickDefaultPage(sidebars: WikiSidebarNode[]): number | null {
  const withPage = sidebars.find((node) => node.page_id !== null && node.is_active);
  return withPage?.page_id ?? null;
}

export default function WikiPage(): JSX.Element {
  const { repoHash, pageId } = useParams();
  const navigate = useNavigate();
  const [sidebars, setSidebars] = useState<WikiSidebarNode[]>([]);
  const [pageData, setPageData] = useState<WikiPageWithContents | null>(null);
  const [repoDetail, setRepoDetail] = useState<RepoSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const selectedPageId = useMemo(() => {
    if (pageId) {
      const parsed = Number(pageId);
      return Number.isNaN(parsed) ? null : parsed;
    }
    return null;
  }, [pageId]);

  useEffect(() => {
    const load = async (): Promise<void> => {
      if (!repoHash) {
        return;
      }
      try {
        setLoading(true);
        const [sidebarsData, repoInfo] = await Promise.all([
          fetchWikiSidebars(repoHash),
          fetchRepoDetail(repoHash),
        ]);
        console.debug("Wiki sidebars fetched:", sidebarsData);
        setRepoDetail(repoInfo);
        setSidebars(sidebarsData);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to load wiki.";
        setError(message);
      } finally {
        setLoading(false);
      }
    };
    void load();
  }, [repoHash]);

  useEffect(() => {
    if (!repoHash) {
      return;
    }
    const effectivePageId = selectedPageId ?? pickDefaultPage(sidebars);
    console.debug("Selected page id:", selectedPageId, "Effective page id:", effectivePageId);
    if (!effectivePageId) {
      setPageData(null);
      return;
    }
    if (!selectedPageId) {
      navigate(`/wiki/${repoHash}/page/${effectivePageId}`, { replace: true });
      return;
    }
    const loadPage = async (): Promise<void> => {
      try {
        const data = await fetchWikiPage(repoHash, effectivePageId);
        setPageData(data);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to load page.";
        setError(message);
      }
    };
    void loadPage();
  }, [repoHash, selectedPageId, sidebars, navigate]);

  if (!repoHash) {
    return <p className="text-ink/60">Missing repo hash.</p>;
  }

  return (
    <div className="grid gap-8 lg:grid-cols-[280px_1fr]">
      <aside className="rounded-2xl border border-ink/10 bg-white p-5 shadow-panel">
        <h2 className="font-display text-lg font-semibold">
          {repoDetail ? `${repoDetail.owner ? `${repoDetail.owner}/` : ""}${repoDetail.repo_name}` : "Repository"}
        </h2>
        <p className="text-xs text-ink/60">
          {repoDetail ? repoDetail.url : "Repository wiki"}
        </p>
        <div className="mt-4 space-y-2">
          {sidebars.map((node) => (
            <Link
              key={node.node_id}
              to={node.page_id ? `/wiki/${repoHash}/page/${node.page_id}` : "#"}
              className={`block rounded-lg px-3 py-2 text-sm ${
                node.page_id === (pageData?.page.page_id ?? selectedPageId)
                  ? "bg-ink text-white"
                  : "bg-mist text-ink"
              } ${!node.is_active ? "opacity-50" : ""}`}
            >
              {node.name}
            </Link>
          ))}
          {sidebars.length === 0 && !loading && (
            <p className="text-xs text-ink/60">No sidebar items yet.</p>
          )}
        </div>
      </aside>

      <section className="rounded-2xl border border-ink/10 bg-white p-8 shadow-panel">
        {loading && <p className="text-ink/60">Loading wiki…</p>}
        {error && <p className="text-red-600">{error}</p>}
        {!loading && !error && pageData && (
          <div>
            <h1 className="font-display text-3xl font-semibold">{pageData.page.title}</h1>
            <div className="mt-2 text-xs text-ink/60">
              Updated {new Date(pageData.page.updated_at * 1000).toLocaleString()}
            </div>
            <div className="mt-6 space-y-6">
              {pageData.contents.map((node) => (
                <article key={node.content_id} className="prose max-w-none">
                  <div className="rounded-lg border border-ink/10 bg-cloud p-4">
                    {node.content_type === "markdown" ? (
                      <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
                        {node.content}
                      </ReactMarkdown>
                    ) : (
                      <div className="whitespace-pre-wrap text-sm text-ink">
                        {node.content}
                      </div>
                    )}
                  </div>
                  <div className="mt-2 text-xs text-ink/50">
                    Sources: {(node.meta?.source_file_ids ?? []).join(", ") || "n/a"}
                  </div>
                </article>
              ))}
              {pageData.contents.length === 0 && (
                <p className="text-ink/60">No content yet for this page.</p>
              )}
            </div>
          </div>
        )}
        {!loading && !error && !pageData && (
          <p className="text-ink/60">No wiki pages available.</p>
        )}
      </section>
    </div>
  );
}
