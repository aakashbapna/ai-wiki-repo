"""Config for hierarchical subsystem clustering."""

import os

SUBSYSTEM_CLUSTER_MAX_BATCHES: int = int(os.environ.get("SUBSYSTEM_CLUSTER_MAX_BATCHES", "30"))
SUBSYSTEM_CLUSTER_MAX_ROUNDS: int = int(os.environ.get("SUBSYSTEM_CLUSTER_MAX_ROUNDS", "5"))
SUBSYSTEM_CLUSTER_MAX_CONCURRENCY: int = int(os.environ.get("SUBSYSTEM_CLUSTER_MAX_CONCURRENCY", "4"))
SUBSYSTEM_CLUSTER_STABILITY_THRESHOLD: float = float(
    os.environ.get("SUBSYSTEM_CLUSTER_STABILITY_THRESHOLD", "0.9")
)
