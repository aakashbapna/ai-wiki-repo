import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  fetchRepoFileContent,
  fetchRepoDetail,
  fetchWikiPage,
  fetchWikiSidebars,
  RepoSummary,
  SourceFileSummary,
  WikiPageWithContents,
  WikiSidebarNode,
} from "../api";

function pickDefaultPage(sidebars: WikiSidebarNode[]): number | null {
  const withPage = sidebars.find((node) => node.page_id !== null && node.is_active);
  return withPage?.page_id ?? null;
}

type SidebarTreeNode = WikiSidebarNode & { children: SidebarTreeNode[]; depth: number };

function buildSidebarTree(nodes: WikiSidebarNode[]): SidebarTreeNode[] {
  const nodeMap: Map<number, SidebarTreeNode> = new Map();
  const roots: SidebarTreeNode[] = [];
  nodes.forEach((node) => {
    nodeMap.set(node.node_id, { ...node, children: [], depth: 0 });
  });
  nodeMap.forEach((node) => {
    if (node.parent_node_id && nodeMap.has(node.parent_node_id)) {
      const parent = nodeMap.get(node.parent_node_id);
      if (parent) {
        node.depth = parent.depth + 1;
        parent.children.push(node);
      }
    } else {
      roots.push(node);
    }
  });
  return roots;
}

function flattenSidebarTree(nodes: SidebarTreeNode[]): SidebarTreeNode[] {
  const output: SidebarTreeNode[] = [];
  const visit = (node: SidebarTreeNode): void => {
    output.push(node);
    node.children.forEach(visit);
  };
  nodes.forEach(visit);
  return output;
}

function guessLanguage(fileName: string): string {
  const extension = fileName.split(".").pop()?.toLowerCase() ?? "";
  const map: Record<string, string> = {
    ts: "typescript",
    tsx: "tsx",
    js: "javascript",
    jsx: "jsx",
    py: "python",
    go: "go",
    rs: "rust",
    java: "java",
    rb: "ruby",
    php: "php",
    html: "html",
    css: "css",
    json: "json",
    yml: "yaml",
    yaml: "yaml",
    md: "markdown",
    sh: "bash",
  };
  return map[extension] ?? "";
}

function formatCodeFence(content: string, language: string): string {
  const safeContent = content.replace(/```/g, "``\\`");
  return `\`\`\`${language}\n${safeContent}\n\`\`\``;
}

function normalizeMarkdownHeadings(content: string): string {
  return content
    .split("\n")
    .map((line) => {
      const match = line.match(/^(#{2,4})(\S)/);
      if (match) {
        return `${match[1]} ${match[2]}${line.slice(match[0].length)}`;
      }
      return line;
    })
    .join("\n");
}

export default function WikiPage(): JSX.Element {
  const { repoHash, pageId } = useParams();
  const navigate = useNavigate();
  const [sidebars, setSidebars] = useState<WikiSidebarNode[]>([]);
  const [pageData, setPageData] = useState<WikiPageWithContents | null>(null);
  const [repoDetail, setRepoDetail] = useState<RepoSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedSource, setSelectedSource] = useState<{
    file_id: number;
    file_name: string;
    file_path: string;
    content: string | null;
    file_size: number | null;
  } | null>(null);
  const [sourceError, setSourceError] = useState<string | null>(null);
  const [sourceLoading, setSourceLoading] = useState<boolean>(false);

  const selectedPageId = useMemo(() => {
    if (pageId) {
      const parsed = Number(pageId);
      return Number.isNaN(parsed) ? null : parsed;
    }
    return null;
  }, [pageId]);

  const sidebarList = useMemo(() => {
    const tree = buildSidebarTree(sidebars);
    return flattenSidebarTree(tree);
  }, [sidebars]);

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

  const openSourceModal = async (source: SourceFileSummary): Promise<void> => {
    if (!repoHash) {
      return;
    }
    setSourceError(null);
    setSourceLoading(true);
    setSelectedSource({
      file_id: source.file_id,
      file_name: source.file_name,
      file_path: source.file_path,
      content: null,
      file_size: null,
    });
    try {
      const data = await fetchRepoFileContent(repoHash, source.file_id);
      setSelectedSource({
        file_id: data.file_id,
        file_name: data.file_name,
        file_path: data.file_path,
        content: data.content,
        file_size: data.file_size,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load file.";
      setSourceError(message);
    } finally {
      setSourceLoading(false);
    }
  };

  const closeSourceModal = (): void => {
    setSelectedSource(null);
    setSourceError(null);
    setSourceLoading(false);
  };

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
          {sidebarList.map((node) => {
            const isActive = node.page_id === (pageData?.page.page_id ?? selectedPageId);
            const baseClass = `block rounded-lg px-3 py-2 text-sm ${
              isActive ? "bg-ink text-white" : "text-ink"
            } ${!node.is_active ? "opacity-50" : ""}`;
            const style = {
              marginLeft: `${node.depth * 12}px`,
              fontSize: node.depth > 0 ? "0.75rem" : undefined,
            };
            if (!node.page_id) {
              return (
                <div
                  key={node.node_id}
                  className={`${baseClass} cursor-default bg-transparent`}
                  style={style}
                >
                  {node.name}
                </div>
              );
            }
            return (
              <Link
                key={node.node_id}
                to={`/wiki/${repoHash}/page/${node.page_id}`}
                className={`${baseClass} hover:underline`}
                style={style}
              >
                {node.name}
              </Link>
            );
          })}
          {sidebarList.length === 0 && !loading && (
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
                  {node.title && (
                    <h2 className="mb-2 text-lg font-semibold text-ink">{node.title}</h2>
                  )}
                  <div className="rounded-lg border border-ink/10 bg-cloud p-4">
                    {node.content_type === "markdown" ? (
                      <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
                        {normalizeMarkdownHeadings(node.content)}
                      </ReactMarkdown>
                    ) : (
                      <div className="whitespace-pre-wrap text-sm text-ink">
                        {node.content}
                      </div>
                    )}
                  </div>
                  <div className="mt-2 text-xs text-ink/50">
                    Sources:{" "}
                    {node.sources.length > 0 ? (
                      <div className="mt-1 flex flex-wrap gap-2 text-ink/70">
                        {node.sources.map((source) => (
                          <button
                            key={source.file_id}
                            type="button"
                            onClick={() => openSourceModal(source)}
                            className="rounded-full border border-ink/15 bg-white px-3 py-1 text-xs text-ink hover:bg-mist"
                          >
                            {source.file_name}
                          </button>
                        ))}
                      </div>
                    ) : (
                      "n/a"
                    )}
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

      {selectedSource && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
          onClick={closeSourceModal}
        >
          <div
            className="w-full max-w-5xl rounded-2xl border border-ink/10 bg-white shadow-panel"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-ink/10 px-6 py-4">
              <div>
                <div className="text-sm font-semibold text-ink">
                  {selectedSource.file_name}
                </div>
                <div className="text-xs text-ink/60">
                  {selectedSource.file_path}
                  {selectedSource.file_size !== null && (
                    <span> • {(selectedSource.file_size / 1024).toFixed(1)} KB</span>
                  )}
                </div>
              </div>
              <button
                type="button"
                onClick={closeSourceModal}
                className="rounded-lg border border-ink/10 bg-cloud px-3 py-1 text-xs font-semibold text-ink hover:bg-mist"
              >
                Close
              </button>
            </div>
            <div className="max-h-[70vh] overflow-auto px-6 py-4">
              {sourceLoading && <p className="text-sm text-ink/60">Loading file…</p>}
              {sourceError && <p className="text-sm text-red-600">{sourceError}</p>}
              {!sourceLoading && !sourceError && (
                <div className="prose max-w-none">
                  <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
                    {formatCodeFence(
                      selectedSource.content ?? "",
                      guessLanguage(selectedSource.file_name),
                    )}
                  </ReactMarkdown>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
