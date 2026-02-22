"""App constants, overridable via environment variables."""

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

# Data directory; default "data" under repo root, overridable via DATA_DIR env
DATA_DIR = Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") else _REPO_ROOT / "data"

# Max file size (bytes) to include in scanning/indexing; overridable via MAX_SCAN_FILE_BYTES env
MAX_SCAN_FILE_BYTES = 500 * 1024

# ── Task concurrency ──────────────────────────────────────────────────────────
# Maximum number of concurrent LLM calls per task type.
# Each unit is one thread in the ThreadPoolExecutor for that task.

# INDEX_FILE: concurrent batch calls to the LLM (each batch = N files)
INDEX_FILE_MAX_CONCURRENCY: int = 5

# BUILD_SUBSYSTEM: concurrent LLM calls for per-batch clustering in Phase 2
BUILD_SUBSYSTEM_MAX_CONCURRENCY: int = 5

# BUILD_WIKI: concurrent page-generation LLM calls, one per subsystem
BUILD_WIKI_MAX_CONCURRENCY: int = 5

# Task staleness timeout (seconds) for running tasks without recent updates.
STALE_TASK_TIMEOUT_SECONDS: int = 30

# ── Hierarchical subsystem clustering ─────────────────────────────────────────
# Phase 1: max number of initial file batches the LLM can produce
SUBSYSTEM_MAX_INITIAL_BATCHES: int = 30

# Phase 3: max merge rounds before stopping
SUBSYSTEM_MAX_MERGE_ROUNDS: int = 5

# Phase 3: target upper bound for final subsystem count
SUBSYSTEM_MAX_FINAL_COUNT: int = 10

# ── Wiki sidebar generation ──────────────────────────────────────────────────
# Max number of top-level sidebar nodes the LLM can produce
WIKI_SIDEBAR_MAX_TOP_NODES: int = 20

# Max number of children per sidebar node
WIKI_SIDEBAR_MAX_CHILDREN: int = 20
