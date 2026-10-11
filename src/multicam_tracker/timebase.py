"""Timestamp contracts shared by ingest, tracking and fusion.

``capture_ts_s`` is event time: when the image was captured.  It is the only
time used for motion and cross-camera association. ``arrival_ts_s`` is system
time: when the process received the frame, useful for latency metrics only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class TimestampQuality(str, Enum):
    """How trustworthy a packet's capture timestamp is."""

    SOURCE_PTS = "source_pts"
    FRAME_INDEX = "frame_index"
    ARRIVAL_FALLBACK = "arrival_fallback"


@dataclass(frozen=True)
class ClockModel:
    """Affine mapping from one camera clock to common event time."""

    offset_s: float = 0.0
    scale: float = 1.0

    def normalize(self, camera_ts_s: float) -> float:
        return float(camera_ts_s) * self.scale + self.offset_s


@dataclass(frozen=True)
class FramePacket:
    camera: str
    image: np.ndarray
    capture_ts_s: float
    arrival_ts_s: float
    sequence: int
    source_frame_index: int
    timestamp_quality: TimestampQuality = TimestampQuality.FRAME_INDEX


@dataclass(frozen=True)
class FrameBatch:
    """A dynamic micro-batch; cameras are not required to be lockstep."""

    packets: dict[str, FramePacket]
    emitted_at_s: float
    dropped_frames: dict[str, int] = field(default_factory=dict)
