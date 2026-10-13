import unittest

import numpy as np

from multicam_tracker.bytetrack import ByteTracker
from multicam_tracker.config import PipelineConfig
from multicam_tracker.local_tracker import create_local_tracker
from multicam_tracker.ocsort import OCSortTracker


def _det(x: float, confidence: float = 0.95) -> np.ndarray:
    return np.array([[x, 0.0, x + 20.0, 40.0, confidence]], dtype=float)


class OCSortTests(unittest.TestCase):
    def test_ocsort_is_default_and_bytetrack_remains_available_for_ablation(self):
        self.assertIsInstance(create_local_tracker(PipelineConfig()), OCSortTracker)
        self.assertIsInstance(
            create_local_tracker(PipelineConfig(local_tracker="bytetrack")),
            ByteTracker,
        )
        with self.assertRaisesRegex(ValueError, "Unknown local tracker"):
            create_local_tracker(PipelineConfig(local_tracker="invalid"))

    def test_recovery_uses_last_observation_and_reupdates_after_occlusion(self):
        cfg = PipelineConfig(
            track_min_hits=1,
            track_min_iou=0.30,
            track_max_age_s=1.5,
            ocsort_use_byte=False,
            ocsort_oru_enabled=True,
        )
        tracker = OCSortTracker(cfg)
        original_id = tracker.update(_det(0.0), timestamp_s=0.0)[0].track_id
        tracker.update(_det(10.0), timestamp_s=0.1)
        for tick in range(2, 9):
            tracker.update(np.empty((0, 5)), timestamp_s=tick / 10.0)

        # The Kalman state keeps moving right, while the observation returns
        # near the last measured box. OCR must recover it and ORU must replay
        # the missed interval rather than allocate a fragment ID.
        visible = tracker.update(_det(10.5), timestamp_s=0.9)

        self.assertEqual([track.track_id for track in visible], [original_id])
        np.testing.assert_allclose(visible[0].xyxy, _det(10.5)[0, :4])
        self.assertEqual(tracker._next_id, 2)
        self.assertGreaterEqual(tracker.metrics["ocr_matches"], 1)
        self.assertGreaterEqual(tracker.metrics["oru_reupdates"], 1)

    def test_crossing_tracks_keep_observed_motion_direction(self):
        cfg = PipelineConfig(
            track_min_hits=1,
            track_min_iou=0.05,
            ocsort_delta_t=2,
            ocsort_inertia=0.30,
            ocsort_use_byte=False,
        )
        tracker = OCSortTracker(cfg)

        def person(center_x: float) -> list[float]:
            return [center_x - 25.0, 40.0, center_x + 25.0, 140.0, 0.95]

        for index in range(13):
            right = 90.0 + 10.0 * index
            left = 210.0 - 10.0 * index
            if index == 6:
                detections = np.asarray([person(right)])
            else:
                detections = np.asarray([person(right), person(left)])
                if index >= 5:
                    detections = detections[::-1]
            tracker.update(detections, timestamp_s=index / 10.0)

        by_id = {track.track_id: track for track in tracker.tracks}
        center_1 = float(np.mean(by_id[1].xyxy[[0, 2]]))
        center_2 = float(np.mean(by_id[2].xyxy[[0, 2]]))
        self.assertGreater(center_1, center_2)
        self.assertGreater(by_id[1].velocity[0], 0.0)
        self.assertLess(by_id[2].velocity[0], 0.0)

    def test_low_confidence_detection_can_update_but_cannot_birth(self):
        cfg = PipelineConfig(track_min_hits=1, ocsort_use_byte=True)
        tracker = OCSortTracker(cfg)
        original_id = tracker.update(_det(0.0), timestamp_s=0.0)[0].track_id

        visible = tracker.update(_det(0.5, confidence=0.30), timestamp_s=0.1)
        self.assertEqual([track.track_id for track in visible], [original_id])
        self.assertEqual(tracker.metrics["byte_extension_matches"], 1)

        empty = OCSortTracker(cfg)
        self.assertEqual(empty.update(_det(0.0, confidence=0.30), timestamp_s=0.0), [])
        self.assertEqual(empty.tracks, [])

    def test_oru_is_release_gated_but_can_be_enabled_explicitly(self):
        self.assertFalse(PipelineConfig().ocsort_oru_enabled)

    def test_timestamp_lifecycle_and_monotonicity(self):
        tracker = OCSortTracker(PipelineConfig(track_min_hits=1, track_max_age_s=0.5))
        tracker.update(_det(0.0), timestamp_s=2.0)
        with self.assertRaisesRegex(ValueError, "monotonic"):
            tracker.update(_det(0.0), timestamp_s=1.9)

        tracker.update(np.empty((0, 5)), timestamp_s=2.6)
        self.assertEqual(tracker.tracks, [])


if __name__ == "__main__":
    unittest.main()
