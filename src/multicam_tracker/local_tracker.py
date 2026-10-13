"""Select the per-camera tracker without coupling the pipeline to one backend."""

from __future__ import annotations

from .bytetrack import ByteTracker
from .config import PipelineConfig
from .hybridsort import HybridSortTracker
from .ocsort import OCSortTracker


LOCAL_TRACKERS = ("hybridsort", "ocsort", "bytetrack")


def create_local_tracker(cfg: PipelineConfig):
    """Construct the configured local tracker with the shared pipeline API."""
    name = cfg.local_tracker.strip().lower()
    if name == "hybridsort":
        return HybridSortTracker(cfg)
    if name == "ocsort":
        return OCSortTracker(cfg)
    if name == "bytetrack":
        return ByteTracker(cfg)
    choices = ", ".join(LOCAL_TRACKERS)
    raise ValueError(f"Unknown local tracker {cfg.local_tracker!r}; choose one of: {choices}")
