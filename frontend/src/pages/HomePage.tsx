import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchRepos, RepoSummary } from "../api";

export default function HomePage(): JSX.Element {
  const [repos, setRepos] = useState<RepoSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const load = async (): Promise<void> => {
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
    void load();
  }, []);

  return (
    <section>
      <div className="mb-8 flex items-end justify-between">
        <div>
          <h1 className="font-display text-3xl font-semibold">Repositories</h1>
          <p className="text-sm text-ink/70">
            Browse repos and jump straight into the wiki.
          </p>
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
