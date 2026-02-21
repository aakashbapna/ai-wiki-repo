"""System prompt for indexing files."""

INDEX_FILE_SYSTEM_PROMPT: str = (
    "You are analyzing a code file.\n\n"
    "Summarize:\n"
    "- What user-facing responsibility it serves (max 500 chars), if its markdown file summarize the content\n"
    "- key_elements: Key functions/classes/tests/commands/etc. this file exports. note any side effects run \n"
    "- Important external dependencies file paths\n"
    "- Whether it appears to be an entry point\n\n"
    "Return JSON:\n\n"
    "{\n"
    '  "responsibility": "",\n'
    '  "key_elements": [],\n'
    '  "dependent_files": [],\n'
    '  "entry_point": true/false\n'
    "}\n"
)
