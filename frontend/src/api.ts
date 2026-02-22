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
  title: string;
  sources: SourceFileSummary[];
  created_at: number;
  updated_at: number;
};

export type WikiPageWithContents = {
  page: WikiPage;
  contents: WikiPageContent[];
};

export type SourceFileSummary = {
  file_id: number;
  file_name: string;
  file_path: string;
};

export type SubsystemSummary = {
  subsystem_id: number;
  name: string;
  description: string;
  meta: Record<string, unknown> | null;
  created_at: number;
};

export type SubsystemListResponse = {
  repo_hash: string;
  total: number;
  subsystems: SubsystemSummary[];
};

export type IndexStatus = {
  repo_hash: string;
  status: string;
  total_files: number;
  completed_files: number;
  remaining_files: number;
  task_id: number;
};

export type RepoFileContent = {
  repo_hash: string;
  file_id: number;
  file_name: string;
  file_path: string;
  content: string;
  file_size: number;
};

export type FetchRepoResponse = {
  repo_hash: string;
  files: {
    file_id: number;
    file_path: string;
    file_name: string;
    is_project_file: boolean;
    is_scan_excluded: boolean;
    file_size: number;
  }[];
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

export async function fetchRepo(url: string): Promise<FetchRepoResponse> {
  return requestJson<FetchRepoResponse>("/fetch-repo", {
    method: "POST",
    body: JSON.stringify({ url }),
  });
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

export async function fetchRepoFileContent(repoHash: string, fileId: number): Promise<RepoFileContent> {
  return requestJson<RepoFileContent>(`/repos/${repoHash}/files/${fileId}/content`);
}

export async function fetchSubsystems(repoHash: string): Promise<SubsystemListResponse> {
  return requestJson<SubsystemListResponse>(`/repos/${repoHash}/subsystems`);
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

export async function fetchSubsystemStatus(repoHash: string): Promise<IndexStatus> {
  return requestJson<IndexStatus>(`/repos/${repoHash}/subsystems/build`);
}

export async function triggerWikiBuild(repoHash: string): Promise<IndexStatus> {
  return requestJson<IndexStatus>(`/repos/${repoHash}/wiki/build`, { method: "POST" });
}

export async function fetchWikiStatus(repoHash: string): Promise<IndexStatus> {
  return requestJson<IndexStatus>(`/repos/${repoHash}/wiki/build`);
}

export async function clearAllData(): Promise<{ repos_deleted: number; files_deleted: number }> {
  return requestJson<{ repos_deleted: number; files_deleted: number }>(`/data`, { method: "DELETE" });
}
