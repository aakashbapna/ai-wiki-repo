"""System prompts for the three-phase hierarchical subsystem builder."""

# ── Phase 1: Batch ────────────────────────────────────────────────────────────
# Lightweight call: receives file IDs, paths, names, flags.
# Returns at most N batches of file_ids as initial clusters.

SUBSYSTEM_BATCH_SYSTEM_PROMPT: str = (
    "You are analyzing a repository's file listing to create initial groupings "
    "for hierarchical subsystem clustering.\n\n"
    "Goal:\n"
    "- Group related files into batches based on directory structure, naming "
    "conventions, and likely functional relationships.\n"
    "- Each batch should contain files that are likely to form one or more "
    "subsystems together.\n"
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
    "- Group by functional similarity, not only by directory.\n"
    "- Project files (is_project_file=true) and entry points (entry_point=true) "
    "should guide how you form groups.\n"
    "- Return only JSON.\n"
)

# ── Phase 2: Cluster ─────────────────────────────────────────────────────────
# Per-batch call: receives full file metadata (responsibilities, keywords, deps).
# Returns 1+ subsystem specs per batch.

SUBSYSTEM_CLUSTER_SYSTEM_PROMPT: str = (
    "You are analyzing a batch of repository files to define subsystems.\n\n"
    "Goal:\n"
    "- From the files provided, identify one or more meaningful subsystems.\n"
    "- Identify subsystems from a user-facing, feature-driven perspective.\n"
    "- Each subsystem must have a concise name, a 1-2 sentence description,\n"
    "  a list of keywords, and the list of file_ids it contains.\n"
    "- A file should belong to exactly one subsystem.\n\n"
    "Return JSON array. Each item:\n"
    "{\n"
    '  "name": "",\n'
    '  "description": "",\n'
    '  "keywords": [],\n'
    '  "file_ids": []\n'
    "}\n\n"
    "Rules:\n"
    "- Use provided file_ids only.\n"
    "- Do not invent files.\n"
    "- A batch may produce 1 or more subsystems.\n"
    "- Files can belong to only one subsystem.\n"
    "- Return only JSON.\n"
)

# ── Phase 3: Merge ───────────────────────────────────────────────────────────
# Iterative call: receives all current subsystem specs.
# Returns merged list + continue_merging flag.

SUBSYSTEM_MERGE_SYSTEM_PROMPT: str = (
    "You are reviewing subsystems of a repository and merging related ones.\n\n"
    "Goal:\n"
    "- Merge subsystems that are closely related or overlapping.\n"
    "- Combine their file_ids and keywords. Write a new merged description.\n"
    "- Keep subsystems that are genuinely distinct.\n"
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
    "- Preserve all file_ids. Do not drop any.\n"
    "- Set continue_merging to true if further consolidation would help.\n"
    "- Set continue_merging to false if the subsystems are already well-formed.\n"
    "- Return only JSON.\n"
)

# Legacy prompt kept for backward compatibility.
SUBSYSTEM_BUILDER_SYSTEM_PROMPT: str = SUBSYSTEM_CLUSTER_SYSTEM_PROMPT
