import unittest
import numpy as np

from multicam_tracker.config import PipelineConfig
from multicam_tracker.hybridsort import HybridSortTracker


class CrowdCascadeTests(unittest.TestCase):
    def tracker(self, **overrides):
        values = dict(hybridsort_cascade_enabled=True, track_min_hits=1,
                      track_detection_temporal_confirmation_frames=1,
                      hybridsort_reappearance_enabled=True,
                      track_max_age_s=.2, hybridsort_reappearance_max_gap_s=2.)
        values.update(overrides)
        return HybridSortTracker(PipelineConfig(**values))

    def test_stale_track_cannot_steal_detection_when_descriptor_missing(self):
        tracker = self.tracker()
        frame = np.full((200, 200, 3), 100, dtype=np.uint8)
        box = np.array([[10, 10, 40, 120, .9]])
        old = tracker.update(box, timestamp_s=0., frame=frame)[0].track_id
        tracker.update(np.empty((0, 5)), timestamp_s=.1, frame=frame)
        # Still retained for appearance recovery; no descriptor available now.
        found = tracker.update(box, timestamp_s=.5, frame=None)
        self.assertNotEqual(found[0].track_id, old)
        self.assertIn(old, [track.track_id for track in tracker.identity_tracks()])

    def test_visible_owner_has_priority_over_lost_neighbor(self):
        tracker = self.tracker(track_max_age_s=2.)
        boxes = np.array([[10, 10, 40, 120, .9], [45, 10, 75, 120, .9]])
        initial = tracker.update(boxes, timestamp_s=0.)
        first = initial[0].track_id
        tracker.update(boxes[:1], timestamp_s=.1)
        output = tracker.update(boxes[:1], timestamp_s=.2)
        self.assertEqual(output[0].track_id, first)
        self.assertEqual(len(output), 1)

    def test_merged_detection_does_not_create_third_id(self):
        tracker = self.tracker(hybridsort_ambiguity_suppression_enabled=True,
                              hybridsort_ambiguity_min_iou=.20,
                              hybridsort_ambiguity_min_area_ratio=1.1)
        boxes = np.array([[10, 10, 40, 120, .9], [40, 10, 70, 120, .9]])
        tracker.update(boxes, timestamp_s=0.)
        tracker.update(np.array([[10, 10, 70, 120, .9]]), timestamp_s=.1)
        self.assertEqual(tracker.metrics["tracks_created"], 2)


if __name__ == "__main__":
    unittest.main()
