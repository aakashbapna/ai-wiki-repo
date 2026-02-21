export type RepoSummary = {
  repo_hash: string;
  owner: string;
  repo_name: string;
  clone_path: string;
  url: string;
  created_at: number;
};

export type RepoListResponse = {
  total: number;
  repos: RepoSummary[];
};

export type WikiSidebarNode = {
  node_id: number;
  repo_hash: string;
  parent_node_id: number | null;
  name: string;
  page_id: number | null;
  is_active: boolean;
  meta: { sub_system_ids?: number[] } | null;
  created_at: number;
  updated_at: number;
};

export type WikiPage = {
  page_id: number;
  repo_hash: string;
  title: string;
  meta: { subsystem_ids?: number[] } | null;
  created_at: number;
  updated_at: number;
};

export type WikiPageContent = {
  content_id: number;
  page_id: number;
  content_type: string;
  content: string;
  meta: { source_file_ids?: number[] } | null;
  created_at: number;
  updated_at: number;
};

export type WikiPageWithContents = {
  page: WikiPage;
  contents: WikiPageContent[];
};

export type IndexStatus = {
  repo_hash: string;
  status: string;
  total_files: number;
  completed_files: number;
  remaining_files: number;
  task_id: number;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers ?? {}),
    },
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Request failed ${response.status}: ${errorText}`);
  }
  return (await response.json()) as T;
}

export async function fetchRepos(): Promise<RepoListResponse> {
  return requestJson<RepoListResponse>("/repos");
}

export async function fetchWikiSidebars(repoHash: string): Promise<WikiSidebarNode[]> {
  const data = await requestJson<{ sidebars: WikiSidebarNode[] }>(`/repos/${repoHash}/wiki/sidebars`);
  return data.sidebars;
}

export async function fetchWikiPages(repoHash: string): Promise<WikiPage[]> {
  const data = await requestJson<{ pages: WikiPage[] }>(`/repos/${repoHash}/wiki/pages`);
  return data.pages;
}

export async function fetchWikiPage(repoHash: string, pageId: number): Promise<WikiPageWithContents> {
  return requestJson<WikiPageWithContents>(`/repos/${repoHash}/wiki/pages/${pageId}`);
}

export async function fetchRepoDetail(repoHash: string): Promise<RepoSummary> {
  return requestJson<RepoSummary>(`/repos/${repoHash}`);
}

export async function triggerIndex(repoHash: string): Promise<IndexStatus> {
  return requestJson<IndexStatus>(`/repos/${repoHash}/index`, { method: "POST" });
}

export async function fetchIndexStatus(repoHash: string): Promise<IndexStatus> {
  return requestJson<IndexStatus>(`/repos/${repoHash}/index`);
}

export async function triggerSubsystemBuild(repoHash: string): Promise<IndexStatus> {
  return requestJson<IndexStatus>(`/repos/${repoHash}/subsystems/build`, { method: "POST" });
}

export async function triggerWikiBuild(repoHash: string): Promise<IndexStatus> {
  return requestJson<IndexStatus>(`/repos/${repoHash}/wiki/build`, { method: "POST" });
}

export async function clearAllData(): Promise<{ repos_deleted: number; files_deleted: number }> {
  return requestJson<{ repos_deleted: number; files_deleted: number }>(`/data`, { method: "DELETE" });
}
