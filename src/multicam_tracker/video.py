"""Synchronized multi-camera video reading.

The dataset's cameras are frame-synchronized: frame N of every video was
captured at the same instant. That lets us "step time" by reading one frame
from each video per tick — no timestamp alignment needed. (With live RTSP
cameras you would align on timestamps instead; DeepStream does this with
its GStreamer batching element, nvstreammux.)

Decoding is the hidden cost of multi-camera pipelines: 4 x 1080p30 streams
is ~250 Mpixels/s before any AI runs. We use `grab()` to skip frames
cheaply (decode-only, no color conversion) so a stride of 2-3 keeps the
demo real-time on a laptop.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class SyncedVideoReader:
    """Iterate over synchronized frames from N camera videos.

    Yields ``(frame_index, {camera: BGR image})`` tuples, advancing all
    videos in lockstep and skipping ``stride - 1`` frames between ticks.
    """

    def __init__(self, scene_dir: Path, cameras: list[str], stride: int = 1):
        self.cameras = cameras
        self.stride = max(1, stride)
        self.captures: dict[str, cv2.VideoCapture] = {}
        for cam in cameras:
            path = scene_dir / cam / "video.mp4"
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise FileNotFoundError(f"Cannot open {path} — run `download-data` first")
            self.captures[cam] = capture

        first = self.captures[cameras[0]]
        self.fps = first.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(first.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        frame_index = 0
        while True:
            frames: dict[str, np.ndarray] = {}
            for cam, capture in self.captures.items():
                ok, frame = capture.read()
                if not ok:
                    return  # any stream ending ends the demo
                frames[cam] = frame
            yield frame_index, frames

            # Cheap skip: grab() decodes but skips retrieval/conversion.
            for _ in range(self.stride - 1):
                for capture in self.captures.values():
                    if not capture.grab():
                        return
            frame_index += self.stride

    def close(self) -> None:
        for capture in self.captures.values():
            capture.release()
