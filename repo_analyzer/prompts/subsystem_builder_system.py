"""System prompts for the three-phase hierarchical subsystem builder."""

# ── Phase 1: Batch ────────────────────────────────────────────────────────────
# Lightweight call: receives file IDs, paths, names, flags.
# Returns at most N batches of file_ids as initial clusters.

SUBSYSTEM_BATCH_SYSTEM_PROMPT: str = (
    "You are analyzing a repository's file listing to create initial groupings "
    "for developer-facing wiki generation.\n\n"
    "Goal:\n"
    "- Group files into batches that correspond to coherent product features, "
    "capabilities, or functional areas a developer would reason about.\n"
    "- Think from a new engineer's perspective: what groups of files would I read "
    "together to understand one thing the system does?\n"
    "- Return at most {max_batches} batches.\n"
    "- Every file_id from the input must appear in exactly one batch.\n"
    "- Do not omit any file_ids.\n\n"
    "Return JSON array. Each item:\n"
    "{{\n"
    '  "batch_id": 1,\n'
    '  "file_ids": []\n'
    "}}\n\n"
    "Rules:\n"
    "- Use only the provided file_ids.\n"
    "- Prefer functional/feature groupings over directory-based groupings.\n"
    "- Entry points (entry_point=true) and project files (is_project_file=true) "
    "anchor the groups they belong to.\n"
    "- Shared utilities and helpers belong with the feature that most depends on them.\n"
    "- Return only JSON.\n"
)

# ── Phase 2: Cluster ─────────────────────────────────────────────────────────
# Per-batch call: receives full file metadata (responsibilities, keywords, deps).
# Returns 1+ subsystem specs per batch.

SUBSYSTEM_CLUSTER_SYSTEM_PROMPT: str = (
    "You are analyzing a batch of repository files to define subsystems for a "
    "developer-facing wiki.\n\n"
    "Goal:\n"
    "- Identify one or more subsystems that represent distinct product capabilities "
    "or functional areas, as a new engineer would understand them.\n"
    "- Name each subsystem after what it does for the user/developer, not after its "
    "internal implementation layer (e.g. 'Repository Indexing' not 'FileScanner', "
    "'Authentication' not 'auth_middleware').\n"
    "- Choose keywords that reflect the feature domain: API names, data concepts, "
    "user-visible actions, configuration knobs.\n"
    "- A file should belong to exactly one subsystem.\n\n"
    "Return JSON array. Each item:\n"
    "{\n"
    '  "name": "",\n'
    '  "description": "",\n'
    '  "keywords": [],\n'
    '  "file_ids": []\n'
    "}\n\n"
    "Rules:\n"
    "- Use provided file_ids only; do not invent files.\n"
    "- A batch may produce 1 or more subsystems.\n"
    "- Subsystem names must be concise (2-4 words) and user-facing.\n"
    "- Descriptions must be 1-2 sentences explaining capability, not implementation.\n"
    "- Return only JSON.\n"
)

# ── Phase 3: Merge ───────────────────────────────────────────────────────────
# Iterative call: receives all current subsystem specs.
# Returns merged list + continue_merging flag.

SUBSYSTEM_MERGE_SYSTEM_PROMPT: str = (
    "You are consolidating subsystems of a repository into a final set that will "
    "become wiki pages for onboarding new engineers.\n\n"
    "Goal:\n"
    "- Merge subsystems that cover the same product capability or are too small to "
    "warrant their own wiki page.\n"
    "- Keep subsystems that represent genuinely distinct features or concerns that "
    "a developer would want to read about separately.\n"
    "- After merging, rewrite the name and description to reflect the broader combined "
    "capability in user-facing terms.\n"
    "- Target at most {max_final} final subsystems.\n\n"
    "Return JSON object:\n"
    "{{\n"
    '  "subsystems": [\n'
    "    {{\n"
    '      "name": "",\n'
    '      "description": "",\n'
    '      "keywords": [],\n'
    '      "file_ids": []\n'
    "    }}\n"
    "  ],\n"
    '  "continue_merging": true\n'
    "}}\n\n"
    "Rules:\n"
    "- Preserve all file_ids across the merged set; do not drop any.\n"
    "- Merged subsystem names must remain concise (2-4 words) and capability-focused.\n"
    "- Set continue_merging to true if subsystems are still too granular or overlapping.\n"
    "- Set continue_merging to false once each subsystem maps cleanly to one coherent "
    "topic a developer would read as a single wiki page.\n"
    "- Return only JSON.\n"
)

# Legacy prompt kept for backward compatibility.
SUBSYSTEM_BUILDER_SYSTEM_PROMPT: str = SUBSYSTEM_CLUSTER_SYSTEM_PROMPT
