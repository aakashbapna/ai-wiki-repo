"""App constants, overridable via environment variables."""

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

# Data directory; default "data" under repo root, overridable via DATA_DIR env
DATA_DIR = Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") else _REPO_ROOT / "data"

# Max file size (bytes) to include in scanning/indexing; overridable via MAX_SCAN_FILE_BYTES env
MAX_SCAN_FILE_BYTES = int(os.environ.get("MAX_SCAN_FILE_BYTES", str(500 * 1024)))

# ── Task concurrency ──────────────────────────────────────────────────────────
# Maximum number of concurrent LLM calls per task type.
# Each unit is one thread in the ThreadPoolExecutor for that task.

# INDEX_FILE: concurrent batch calls to the LLM (each batch = N files)
INDEX_FILE_MAX_CONCURRENCY: int = int(os.environ.get("INDEX_FILE_MAX_CONCURRENCY", "5"))

# BUILD_SUBSYSTEM: concurrent LLM calls (currently one call builds all subsystems,
# so this governs future per-chunk parallelism; kept at 1 to match current design)
BUILD_SUBSYSTEM_MAX_CONCURRENCY: int = int(os.environ.get("BUILD_SUBSYSTEM_MAX_CONCURRENCY", "1"))

# BUILD_WIKI: concurrent page-generation LLM calls, one per subsystem
BUILD_WIKI_MAX_CONCURRENCY: int = int(os.environ.get("BUILD_WIKI_MAX_CONCURRENCY", "5"))

# Task staleness timeout (seconds) for running tasks without recent updates.
STALE_TASK_TIMEOUT_SECONDS: int = int(os.environ.get("STALE_TASK_TIMEOUT_SECONDS", "60"))
