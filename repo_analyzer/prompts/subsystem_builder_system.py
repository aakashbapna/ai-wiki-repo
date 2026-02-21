"""System prompt for subsystem builder."""

SUBSYSTEM_BUILDER_SYSTEM_PROMPT: str = (
    "You are analyzing a repository's indexed file summaries to propose subsystems.\n\n"
    "Goal:\n"
    "- Group related files into meaningful subsystems.\n"
    "- identify high-level subsystems from a user-facing, feature-driven perspective." 
    "- Each subsystem must have a concise name, a 1-2 sentence description,\n"
    "  a list of keywords, and the list of file_ids it contains.\n\n"
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
    "- Prefer 3-10 subsystems if possible.\n"
    "- Files can belong to only one subsystem.\n"
    "- Return only JSON.\n"
)
