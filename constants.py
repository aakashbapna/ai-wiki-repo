"""App constants, overridable via environment variables."""

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

# Data directory; default "data" under repo root, overridable via DATA_DIR env
DATA_DIR = Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") else _REPO_ROOT / "data"

# Max file size (bytes) to include in scanning/indexing; overridable via MAX_SCAN_FILE_BYTES env
MAX_SCAN_FILE_BYTES = int(os.environ.get("MAX_SCAN_FILE_BYTES", str(500 * 1024)))
