import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from multicam_tracker.config import PipelineConfig
from multicam_tracker.demo import run_pipeline
from multicam_tracker.timebase import FrameBatch, FramePacket


class _Calibration:
    width = 100
    height = 100

    def image_to_world(self, image_xy):
        return np.asarray(image_xy, dtype=float).reshape((-1, 2))


class _Detector:
    def detect_batch(self, frames):
        return [np.empty((0, 5), dtype=float) for _ in frames]


class _Tracker:
    def __init__(self, centers):
        self.centers = iter(centers)
        self.last_events = []
        self.tracks = []
        self.metrics = {
            "tracks_created": 1,
            "reactivations": 0,
            "tracks_removed_tentative": 0,
            "tracks_removed_timeout": 0,
            "high_confidence_matches": 0,
            "low_confidence_matches": 0,
            "ocm_matches": 0,
            "ocr_matches": 0,
            "oru_reupdates": 0,
            "byte_extension_matches": 0,
            "peak_retained_tracks": 1,
        }

    def update(self, _detections, timestamp_s=None):
        center = next(self.centers)
        if center is None:
            self.tracks = []
            return []
        track = SimpleNamespace(
            track_id=1,
            xyxy=np.array([center - 0.5, -1.0, center + 0.5, 0.0]),
            conf=0.9,
        )
        self.tracks = [track]
        return [track]


def _batch(sequence, timestamp):
    packet = FramePacket(
        camera="a",
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        capture_ts_s=timestamp,
        arrival_ts_s=timestamp,
        sequence=sequence,
        source_frame_index=sequence,
    )
    return FrameBatch({"a": packet}, emitted_at_s=timestamp)


class TrajectoryRuntimeTests(unittest.TestCase):
    def test_runtime_rejects_outlier_without_contaminating_history(self) -> None:
        cfg = PipelineConfig(
            cameras=["a"],
            fusion_min_hits=1,
            fusion_trajectory_smoothing_enabled=True,
            fusion_trajectory_window_s=1.0,
            fusion_trajectory_outlier_gate_m=0.25,
            fusion_max_jump_m=2.0,
            fusion_jump_window_s=1.0,
        )
        tracker = _Tracker([0.0, 0.1, 10.0, 0.3, None])
        batches = [
            _batch(0, 0.0),
            _batch(1, 0.1),
            _batch(2, 0.2),
            _batch(3, 0.3),
            _batch(4, 2.0),
        ]

        with (
            patch(
                "multicam_tracker.demo.load_calibrations",
                return_value={"a": _Calibration()},
            ),
            patch(
                "multicam_tracker.demo.create_local_tracker",
                return_value=tracker,
            ),
        ):
            ticks = list(run_pipeline(cfg, batch_iter=batches, detector=_Detector()))

        self.assertEqual(ticks[2]["observations"], [])
        self.assertEqual(len(ticks[3]["observations"]), 1)
        np.testing.assert_allclose(
            ticks[3]["observations"][0].world_xy,
            [0.3, 0.0],
            atol=1e-8,
        )
        self.assertEqual(ticks[3]["metrics"]["trajectory_smoothing_rejected"], 1)
        self.assertEqual(ticks[-1]["metrics"]["trajectory_smoothing_histories"], 0)
        self.assertEqual(ticks[-1]["metrics"]["trajectory_smoothing_samples"], 0)


if __name__ == "__main__":
    unittest.main()
