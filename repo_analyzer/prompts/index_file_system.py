"""System prompt for indexing files."""

INDEX_FILE_SYSTEM_PROMPT: str = (
    "You are analyzing a code file.\n\n"
    "Summarize:\n"
    "- What user-facing responsibility it serves (max 500 chars), if its markdown file summarize the content\n"
    "- key_elements: keywords that describe the file's content, this is a list of keywords\n"
    "- dependent_files: Important external dependencies file paths, this is a list of file paths (strictly) that this file depends on\n"
    "- entry_point: whether this file appears to be an entry point for the subsystem, this is a boolean value\n"
    "- file_summary: Only if file_size_bytes > 10240, return a concise list (max 1000 chars) of imports, exports, "
    "function signatures, class definitions, and test specs (it()/describe() calls). Do not include implementation.\n\n"
    "Return JSON:\n\n"
    "{\n"
    '  "responsibility": "",\n'
    '  "key_elements": [],\n'
    '  "dependent_files": [],\n'
    '  "entry_point": true/false,\n'
    '  "file_summary": ""\n'
    "}\n"
)
