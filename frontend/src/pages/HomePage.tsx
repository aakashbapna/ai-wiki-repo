import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  fetchIndexStatus,
  fetchRepo,
  fetchRepos,
  fetchSubsystemStatus,
  fetchWikiStatus,
  IndexStatus,
  RepoSummary,
  triggerIndex,
  triggerSubsystemBuild,
  triggerWikiBuild,
} from "../api";

type IngestStatusMap = {
  index: IndexStatus | null;
  subsystem: IndexStatus | null;
  wiki: IndexStatus | null;
};

const TERMINAL_STATUSES = new Set(["completed", "failed", "stale", "stopped"]);
const POLL_INTERVAL_MS = 2000;

function isTerminalStatus(status: string): boolean {
  return TERMINAL_STATUSES.has(status);
}

function formatProgress(status: IndexStatus | null): string {
  if (!status) {
    return "Not started";
  }
  return `${status.status} • ${status.completed_files}/${status.total_files}`;
}

export default function HomePage(): JSX.Element {
  const [repos, setRepos] = useState<RepoSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [repoUrl, setRepoUrl] = useState<string>("");
  const [ingestError, setIngestError] = useState<string | null>(null);
  const [ingestStatus, setIngestStatus] = useState<IngestStatusMap>({
    index: null,
    subsystem: null,
    wiki: null,
  });
  const [ingesting, setIngesting] = useState<boolean>(false);
  const [ingestMessage, setIngestMessage] = useState<string | null>(null);
  const cancelRef = useRef<boolean>(false);

  const loadRepos = async (): Promise<void> => {
    try {
      setLoading(true);
      const data = await fetchRepos();
      setRepos(data.repos);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load repos.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadRepos();
  }, []);

  useEffect(() => {
    return () => {
      cancelRef.current = true;
    };
  }, []);

  const resetIngestState = (): void => {
    setIngestError(null);
    setIngestMessage(null);
    setIngestStatus({ index: null, subsystem: null, wiki: null });
  };

  const pollStatus = async (
    fetchStatus: () => Promise<IndexStatus>,
    updateStatus: (status: IndexStatus) => void,
  ): Promise<IndexStatus> => {
    while (!cancelRef.current) {
      const status = await fetchStatus();
      updateStatus(status);
      if (isTerminalStatus(status.status)) {
        return status;
      }
      await new Promise((resolve) => {
        setTimeout(resolve, POLL_INTERVAL_MS);
      });
    }
    throw new Error("Polling cancelled.");
  };

  const handleIngest = async (): Promise<void> => {
    if (!repoUrl.trim() || ingesting) {
      return;
    }
    resetIngestState();
    setIngesting(true);
    try {
      const repo = await fetchRepo(repoUrl.trim());
      const repoHash = repo.repo_hash;

      const indexStart = await triggerIndex(repoHash);
      setIngestStatus((prev) => ({ ...prev, index: indexStart }));
      const indexFinal = await pollStatus(
        () => fetchIndexStatus(repoHash),
        (status) => setIngestStatus((prev) => ({ ...prev, index: status })),
      );
      if (indexFinal.status !== "completed") {
        throw new Error(`Indexing ended with status: ${indexFinal.status}`);
      }

      const subsystemStart = await triggerSubsystemBuild(repoHash);
      setIngestStatus((prev) => ({ ...prev, subsystem: subsystemStart }));
      const subsystemFinal = await pollStatus(
        () => fetchSubsystemStatus(repoHash),
        (status) => setIngestStatus((prev) => ({ ...prev, subsystem: status })),
      );
      if (subsystemFinal.status !== "completed") {
        throw new Error(`Subsystem build ended with status: ${subsystemFinal.status}`);
      }

      const wikiStart = await triggerWikiBuild(repoHash);
      setIngestStatus((prev) => ({ ...prev, wiki: wikiStart }));
      const wikiFinal = await pollStatus(
        () => fetchWikiStatus(repoHash),
        (status) => setIngestStatus((prev) => ({ ...prev, wiki: status })),
      );
      if (wikiFinal.status !== "completed") {
        throw new Error(`Wiki build ended with status: ${wikiFinal.status}`);
      }

      await loadRepos();
      setIngestMessage("Ingestion complete. Repo list updated.");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to ingest repository.";
      setIngestError(message);
    } finally {
      setIngesting(false);
    }
  };

  const refreshIngestStatus = async (): Promise<void> => {
    if (!ingestStatus.index && !ingestStatus.subsystem && !ingestStatus.wiki) {
      return;
    }
    if (!repoUrl.trim()) {
      return;
    }
    setIngestError(null);
    try {
      const repo = await fetchRepo(repoUrl.trim());
      const repoHash = repo.repo_hash;
      try {
        const index = await fetchIndexStatus(repoHash);
        setIngestStatus((prev) => ({ ...prev, index }));
      } catch {
        setIngestStatus((prev) => ({ ...prev, index: prev.index }));
      }
      try {
        const subsystem = await fetchSubsystemStatus(repoHash);
        setIngestStatus((prev) => ({ ...prev, subsystem }));
      } catch {
        setIngestStatus((prev) => ({ ...prev, subsystem: prev.subsystem }));
      }
      try {
        const wiki = await fetchWikiStatus(repoHash);
        setIngestStatus((prev) => ({ ...prev, wiki }));
      } catch {
        setIngestStatus((prev) => ({ ...prev, wiki: prev.wiki }));
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to refresh status.";
      setIngestError(message);
    }
  };

  return (
    <section>
      <div className="mb-8 flex flex-col gap-6">
        <div>
          <h1 className="font-display text-3xl font-semibold">Repositories</h1>
          <p className="text-sm text-ink/70">
            Browse repos and jump straight into the wiki.
          </p>
        </div>
        <div className="rounded-2xl border border-ink/10 bg-white p-6 shadow-panel">
          <div className="flex flex-col gap-4 md:flex-row md:items-end">
            <label className="flex-1">
              <span className="text-sm font-medium">Repository URL</span>
              <input
                type="text"
                value={repoUrl}
                onChange={(event) => setRepoUrl(event.target.value)}
                placeholder="https://github.com/owner/repo.git"
                className="mt-2 w-full rounded-xl border border-ink/10 bg-cloud px-4 py-3 text-sm"
                disabled={ingesting}
              />
            </label>
            <button
              type="button"
              onClick={handleIngest}
              disabled={ingesting || !repoUrl.trim()}
              className="rounded-xl bg-ink px-5 py-3 text-sm font-semibold text-white hover:bg-ink/90 disabled:cursor-not-allowed disabled:bg-ink/40"
            >
              {ingesting ? "Ingesting…" : "Ingest Repo"}
            </button>
          </div>

          <div className="mt-5 grid gap-2 text-sm text-ink/70">
            <div>Indexing: {formatProgress(ingestStatus.index)}</div>
            <div>Subsystems: {formatProgress(ingestStatus.subsystem)}</div>
            <div>Wiki: {formatProgress(ingestStatus.wiki)}</div>
          </div>
          <div className="mt-3">
            <button
              type="button"
              onClick={refreshIngestStatus}
              className="text-xs font-semibold text-accentDark hover:underline"
            >
              Refresh Status
            </button>
          </div>
          {ingestMessage && (
            <p className="mt-3 text-sm text-accentDark">{ingestMessage}</p>
          )}
          {ingestError && (
            <p className="mt-3 text-sm text-red-600">{ingestError}</p>
          )}
        </div>
      </div>

      {loading && <p className="text-ink/60">Loading repositories…</p>}
      {error && <p className="text-red-600">{error}</p>}

      {!loading && !error && repos.length === 0 && (
        <p className="text-ink/60">No repos found. Add one from the Admin page.</p>
      )}

      <div className="grid gap-6 md:grid-cols-2">
        {repos.map((repo) => (
          <div
            key={repo.repo_hash}
            className="rounded-2xl border border-ink/10 bg-white p-6 shadow-panel"
          >
            <div className="flex items-start justify-between">
              <div>
                <h2 className="font-display text-xl font-semibold">
                  {repo.owner ? `${repo.owner}/` : ""}{repo.repo_name}
                </h2>
                <p className="text-xs text-ink/60">{repo.repo_hash}</p>
              </div>
              <span className="rounded-full bg-mist px-3 py-1 text-xs text-ink/60">
                {new Date(repo.created_at * 1000).toLocaleDateString()}
              </span>
            </div>
            <p className="mt-3 text-sm text-ink/70">{repo.url}</p>
            <div className="mt-6 flex items-center gap-3">
              <Link
                to={`/wiki/${repo.repo_hash}`}
                className="rounded-full bg-ink px-4 py-2 text-sm font-semibold text-white hover:bg-ink/90"
              >
                Open Wiki
              </Link>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
