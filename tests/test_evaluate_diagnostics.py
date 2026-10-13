import unittest

import numpy as np

from multicam_tracker.evaluate import _classify_switch_events


class EvaluateDiagnosticsTests(unittest.TestCase):
    def test_switches_keep_local_and_crossing_context(self) -> None:
        switches = [
            (3, 10, 11, 7, 8),
            (4, 20, 21, 9, 10),
        ]
        source_maps = {
            10: {7: [("camera_a", 2)]},
            11: {8: [("camera_a", 2)]},
            20: {9: [("camera_b", 4)]},
            21: {10: [("camera_b", 5)]},
        }
        predictions = [
            (10, {7: np.zeros(2)}),
            (11, {7: np.zeros(2), 8: np.zeros(2)}),
            (20, {9: np.zeros(2)}),
            (21, {10: np.zeros(2)}),
        ]
        ground_truth = {
            11: {3: np.array([0.0, 0.0]), 99: np.array([0.5, 0.0])},
            21: {4: np.array([3.0, 0.0]), 99: np.array([9.0, 0.0])},
        }

        records = _classify_switch_events(
            switches, source_maps, predictions, ground_truth, frame_offset=0
        )

        self.assertEqual(records[0]["classification"], "global_changed_local_stable")
        self.assertTrue(records[0]["near_crossing"])
        self.assertTrue(records[0]["previous_gid_present_at_switch"])
        self.assertEqual(records[1]["classification"], "local_fragmentation")
        self.assertEqual(records[1]["same_camera_fragment"], ["camera_b"])
        self.assertFalse(records[1]["near_crossing"])


if __name__ == "__main__":
    unittest.main()
