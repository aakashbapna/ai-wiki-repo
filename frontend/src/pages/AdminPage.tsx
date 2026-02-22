import { useEffect, useState } from "react";
import {
  fetchIndexStatus,
  fetchWikiSidebars,
  fetchWikiStatus,
  clearAllData,
  fetchSubsystems,
  fetchRepos,
  IndexStatus,
  RepoSummary,
  SubsystemSummary,
  triggerIndex,
  triggerSubsystemBuild,
  triggerWikiBuild,
  WikiSidebarNode,
} from "../api";

export default function AdminPage(): JSX.Element {
  const [repos, setRepos] = useState<RepoSummary[]>([]);
  const [selectedRepo, setSelectedRepo] = useState<string>("");
  const [status, setStatus] = useState<IndexStatus | null>(null);
  const [subsystems, setSubsystems] = useState<SubsystemSummary[]>([]);
  const [subsystemsLoading, setSubsystemsLoading] = useState<boolean>(false);
  const [wikiStatus, setWikiStatus] = useState<IndexStatus | null>(null);
  const [wikiSidebars, setWikiSidebars] = useState<WikiSidebarNode[]>([]);
  const [wikiLoading, setWikiLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"index" | "subsystems" | "wiki">(
    "index",
  );

  useEffect(() => {
    const loadRepos = async (): Promise<void> => {
      try {
        setLoading(true);
        const data = await fetchRepos();
        setRepos(data.repos);
        if (data.repos.length > 0) {
          setSelectedRepo(data.repos[0].repo_hash);
        }
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to load repos.";
        setError(message);
      } finally {
        setLoading(false);
      }
    };
    void loadRepos();
  }, []);

  const refreshStatus = async (repoHash: string): Promise<void> => {
    try {
      const data = await fetchIndexStatus(repoHash);
      setStatus(data);
    } catch (err) {
      setStatus(null);
    }
  };

  const refreshSubsystems = async (repoHash: string): Promise<void> => {
    try {
      setSubsystemsLoading(true);
      const data = await fetchSubsystems(repoHash);
      setSubsystems(data.subsystems);
    } catch (err) {
      setSubsystems([]);
    } finally {
      setSubsystemsLoading(false);
    }
  };

  const refreshWikiStatus = async (repoHash: string): Promise<void> => {
    try {
      const data = await fetchWikiStatus(repoHash);
      setWikiStatus(data);
    } catch (err) {
      setWikiStatus(null);
    }
  };

  const refreshWikiSidebars = async (repoHash: string): Promise<void> => {
    try {
      setWikiLoading(true);
      const data = await fetchWikiSidebars(repoHash);
      setWikiSidebars(data);
    } catch (err) {
      setWikiSidebars([]);
    } finally {
      setWikiLoading(false);
    }
  };

  const handleIndex = async (): Promise<void> => {
    if (!selectedRepo) {
      return;
    }
    setActionMessage(null);
    try {
      const data = await triggerIndex(selectedRepo);
      setStatus(data);
      setActionMessage("Indexing started.");
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to start indexing.";
      setError(message);
    }
  };

  const handleSubsystems = async (): Promise<void> => {
    if (!selectedRepo) {
      return;
    }
    setActionMessage(null);
    try {
      const data = await triggerSubsystemBuild(selectedRepo);
      setStatus(data);
      setActionMessage("Subsystem build started.");
      const pollStatus = async (): Promise<IndexStatus> => {
        while (true) {
          const current = await fetchIndexStatus(selectedRepo);
          if (["completed", "failed", "stale", "stopped"].includes(current.status)) {
            return current;
          }
          await new Promise((resolve) => setTimeout(resolve, 2000));
        }
      };
      const finalStatus = await pollStatus();
      if (finalStatus.status === "completed") {
        await refreshSubsystems(selectedRepo);
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to build subsystems.";
      setError(message);
    }
  };

  const handleWiki = async (): Promise<void> => {
    if (!selectedRepo) {
      return;
    }
    setActionMessage(null);
    try {
      const data = await triggerWikiBuild(selectedRepo);
      setWikiStatus(data);
      setActionMessage("Wiki build started.");
      const pollStatus = async (): Promise<IndexStatus> => {
        while (true) {
          const current = await fetchWikiStatus(selectedRepo);
          if (["completed", "failed", "stale", "stopped"].includes(current.status)) {
            return current;
          }
          await new Promise((resolve) => setTimeout(resolve, 2000));
        }
      };
      const finalStatus = await pollStatus();
      if (finalStatus.status === "completed") {
        await refreshWikiSidebars(selectedRepo);
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to build wiki.";
      setError(message);
    }
  };

  const handleClearAll = async (): Promise<void> => {
    if (!confirm("This will delete all repos and files. Continue?")) {
      return;
    }
    setActionMessage(null);
    try {
      await clearAllData();
      setRepos([]);
      setSelectedRepo("");
      setStatus(null);
      setActionMessage("All data cleared.");
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to clear data.";
      setError(message);
    }
  };

  useEffect(() => {
    if (!selectedRepo) {
      return;
    }
    void refreshStatus(selectedRepo);
  }, [selectedRepo]);

  useEffect(() => {
    if (!selectedRepo || activeTab !== "subsystems") {
      return;
    }
    void refreshSubsystems(selectedRepo);
  }, [selectedRepo, activeTab]);

  useEffect(() => {
    if (!selectedRepo || activeTab !== "wiki") {
      return;
    }
    void refreshWikiStatus(selectedRepo);
    void refreshWikiSidebars(selectedRepo);
  }, [selectedRepo, activeTab]);

  return (
    <section>
      <div className="mb-6">
        <h1 className="font-display text-3xl font-semibold">Admin Console</h1>
        <p className="text-sm text-ink/60">
          Trigger indexing and wiki refresh jobs.
        </p>
      </div>

      {loading && <p className="text-ink/60">Loading repos…</p>}
      {error && <p className="text-red-600">{error}</p>}

      <div className="rounded-2xl border border-ink/10 bg-white p-6 shadow-panel">
        <label className="text-sm font-medium">Select repository</label>
        <select
          className="mt-2 w-full rounded-lg border border-ink/10 bg-cloud p-3 text-sm"
          value={selectedRepo}
          onChange={(event) => setSelectedRepo(event.target.value)}
        >
          {repos.map((repo) => (
            <option key={repo.repo_hash} value={repo.repo_hash}>
              {repo.owner ? `${repo.owner}/` : ""}
              {repo.repo_name}
            </option>
          ))}
        </select>

        <div className="mt-6 grid gap-2 md:grid-cols-3">
          {[
            { key: "index", label: "Index" },
            { key: "subsystems", label: "Subsystems" },
            { key: "wiki", label: "Wiki" },
          ].map((tab) => (
            <button
              key={tab.key}
              type="button"
              onClick={() =>
                setActiveTab(tab.key as "index" | "subsystems" | "wiki")
              }
              className={
                activeTab === tab.key
                  ? "rounded-xl bg-ink px-4 py-3 text-sm font-semibold text-white"
                  : "rounded-xl border border-ink/20 bg-white px-4 py-3 text-sm font-semibold"
              }
            >
              {tab.label}
            </button>
          ))}
        </div>

        {activeTab === "index" && (
          <div className="mt-6 grid gap-4">
            <div className="rounded-xl border border-ink/10 bg-cloud p-4 text-sm">
              <div className="flex items-center justify-between">
                <span className="font-semibold">Index Status</span>
                <button
                  type="button"
                  onClick={() => selectedRepo && refreshStatus(selectedRepo)}
                  className="text-xs text-accentDark"
                >
                  Refresh
                </button>
              </div>
              {status ? (
                <div className="mt-3 grid gap-1 text-xs">
                  <div>Status: {status.status}</div>
                  <div>
                    Completed: {status.completed_files} / {status.total_files}
                  </div>
                  <div>Remaining: {status.remaining_files}</div>
                </div>
              ) : (
                <p className="mt-3 text-xs text-ink/60">
                  No status available.
                </p>
              )}
              {actionMessage && (
                <p className="mt-2 text-xs text-accentDark">{actionMessage}</p>
              )}
            </div>
            <button
              type="button"
              onClick={handleIndex}
              className="rounded-xl bg-ink px-4 py-3 text-sm font-semibold text-white hover:bg-ink/90"
            >
              Build Index
            </button>
          </div>
        )}

        {activeTab === "subsystems" && (
          <div className="mt-6 grid gap-4">
            <div className="rounded-xl border border-ink/10 bg-cloud p-4 text-sm">
              <div className="flex items-center justify-between">
                <span className="font-semibold">Subsystem Generation</span>
                <button
                  type="button"
                  onClick={() => selectedRepo && refreshSubsystems(selectedRepo)}
                  className="text-xs text-accentDark"
                >
                  Refresh
                </button>
              </div>
              <p className="mt-2 text-xs text-ink/60">
                Build subsystem groupings from indexed files.
              </p>
              {actionMessage && (
                <p className="mt-2 text-xs text-accentDark">{actionMessage}</p>
              )}
              <div className="mt-4">
                {subsystemsLoading && (
                  <p className="text-xs text-ink/60">Loading subsystems…</p>
                )}
                {!subsystemsLoading && subsystems.length === 0 && (
                  <p className="text-xs text-ink/60">No subsystems available.</p>
                )}
                {!subsystemsLoading && subsystems.length > 0 && (
                  <div className="space-y-2 text-xs text-ink/70">
                    {subsystems.map((subsystem) => (
                      <div
                        key={subsystem.subsystem_id}
                        className="rounded-lg border border-ink/10 bg-white px-3 py-2"
                      >
                        <div className="flex items-center justify-between text-ink">
                          <span className="font-semibold">
                          {subsystem.name}
                          </span>
                          <span className="text-[10px] text-ink/50">
                            #{subsystem.subsystem_id}
                          </span>
                        </div>
                        <p className="mt-1 text-ink/60">
                          {subsystem.description || "No description available."}
                        </p>
                        <div className="mt-2 text-[10px] text-ink/50">
                          Created {new Date(subsystem.created_at * 1000).toLocaleString()}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
            <button
              type="button"
              onClick={handleSubsystems}
              className="rounded-xl bg-ink px-4 py-3 text-sm font-semibold text-white hover:bg-ink/90"
            >
              Build Subsystems
            </button>
          </div>
        )}

        {activeTab === "wiki" && (
          <div className="mt-6 grid gap-4">
            <div className="rounded-xl border border-ink/10 bg-cloud p-4 text-sm">
              <div className="flex items-center justify-between">
                <span className="font-semibold">Wiki Generation</span>
                <button
                  type="button"
                  onClick={() => selectedRepo && refreshWikiSidebars(selectedRepo)}
                  className="text-xs text-accentDark"
                >
                  Refresh
                </button>
              </div>
              <p className="mt-2 text-xs text-ink/60">
                Build wiki pages and sidebars from subsystems.
              </p>
              {wikiStatus ? (
                <div className="mt-3 grid gap-1 text-xs">
                  <div>Status: {wikiStatus.status}</div>
                  <div>
                    Completed: {wikiStatus.completed_files} / {wikiStatus.total_files}
                  </div>
                  <div>Remaining: {wikiStatus.remaining_files}</div>
                </div>
              ) : (
                <p className="mt-3 text-xs text-ink/60">No status available.</p>
              )}
              {actionMessage && (
                <p className="mt-2 text-xs text-accentDark">{actionMessage}</p>
              )}
              <div className="mt-4">
                {wikiLoading && (
                  <p className="text-xs text-ink/60">Loading sidebar…</p>
                )}
                {!wikiLoading && wikiSidebars.length === 0 && (
                  <p className="text-xs text-ink/60">No sidebar nodes yet.</p>
                )}
                {!wikiLoading && wikiSidebars.length > 0 && (
                  <div className="space-y-2 text-xs text-ink/70">
                    {wikiSidebars.map((node) => (
                      <div
                        key={node.node_id}
                        className="rounded-lg border border-ink/10 bg-white px-3 py-2"
                        style={{
                          marginLeft: node.parent_node_id ? "12px" : undefined,
                        }}
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-semibold">{node.name}</span>
                          <span className="text-[10px] text-ink/50">
                            #{node.node_id}
                          </span>
                        </div>
                        <div className="mt-1 text-[10px] text-ink/50">
                          {node.page_id ? `Page ${node.page_id}` : "No page linked"}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
            <button
              type="button"
              onClick={handleWiki}
              className="rounded-xl bg-ink px-4 py-3 text-sm font-semibold text-white hover:bg-ink/90"
            >
              Build Wiki
            </button>
          </div>
        )}
        <div className="mt-4">
          <button
            type="button"
            onClick={handleClearAll}
            className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-semibold text-red-700"
          >
            Clear All Data
          </button>
        </div>
      </div>
    </section>
  );
}
