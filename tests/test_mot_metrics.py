import unittest

import numpy as np

from multicam_tracker.mot_metrics import evaluate_mot


class MotMetricTests(unittest.TestCase):
    @staticmethod
    def _point(x: float, y: float = 0.0) -> np.ndarray:
        return np.array([x, y], dtype=float)

    def test_perfect_tracks_score_one(self) -> None:
        predictions = [
            (frame, {10: self._point(frame), 20: self._point(frame, 2.0)})
            for frame in range(3)
        ]
        ground_truth = {
            frame: {1: self._point(frame), 2: self._point(frame, 2.0)}
            for frame in range(3)
        }
        metrics = evaluate_mot(predictions, ground_truth)

        for name in ("IDF1", "IDP", "IDR", "MOTA", "MOTP", "HOTA", "DetA", "AssA", "LocA"):
            self.assertAlmostEqual(metrics[name], 1.0, msg=name)
        self.assertEqual(metrics["IDSW"], 0)
        self.assertEqual(metrics["Frag"], 0)
        self.assertAlmostEqual(metrics["MOTP_distance_m"], 0.0)

    def test_missing_detection_reduces_identity_and_hota(self) -> None:
        predictions = [(0, {10: self._point(0.0)}), (1, {})]
        ground_truth = {
            0: {1: self._point(0.0)},
            1: {1: self._point(1.0)},
        }
        metrics = evaluate_mot(predictions, ground_truth)

        self.assertEqual(metrics["IDTP"], 1)
        self.assertEqual(metrics["IDFP"], 0)
        self.assertEqual(metrics["IDFN"], 1)
        self.assertAlmostEqual(metrics["IDP"], 1.0)
        self.assertAlmostEqual(metrics["IDR"], 0.5)
        self.assertAlmostEqual(metrics["IDF1"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["MOTA"], 0.5)
        self.assertAlmostEqual(metrics["HOTA"], 0.5)

    def test_identity_switch_penalizes_all_identity_metrics(self) -> None:
        predictions = [
            (frame, {(10 if frame < 2 else 11): self._point(frame)})
            for frame in range(4)
        ]
        ground_truth = {
            frame: {1: self._point(frame)} for frame in range(4)
        }
        metrics = evaluate_mot(predictions, ground_truth)

        self.assertEqual(metrics["IDSW"], 1)
        self.assertEqual(metrics["IDTP"], 2)
        self.assertEqual(metrics["IDFP"], 2)
        self.assertEqual(metrics["IDFN"], 2)
        self.assertAlmostEqual(metrics["IDP"], 0.5)
        self.assertAlmostEqual(metrics["IDR"], 0.5)
        self.assertAlmostEqual(metrics["IDF1"], 0.5)
        self.assertAlmostEqual(metrics["MOTA"], 0.75)
        self.assertAlmostEqual(metrics["HOTA"], np.sqrt(0.5))
        self.assertEqual(metrics["switch_events"][0][3:], (10, 11))

    def test_reacquisition_after_a_missed_visible_frame_is_fragmentation(self) -> None:
        predictions = [
            (0, {10: self._point(0.0)}),
            (1, {}),
            (2, {10: self._point(2.0)}),
        ]
        ground_truth = {
            frame: {1: self._point(float(frame))} for frame in range(3)
        }

        metrics = evaluate_mot(predictions, ground_truth)

        self.assertEqual(metrics["Frag"], 1)
        self.assertEqual(metrics["IDSW"], 0)

    def test_point_outside_world_gate_is_fp_and_fn(self) -> None:
        predictions = [(0, {10: self._point(1.01)})]
        ground_truth = {0: {1: self._point(0.0)}}
        metrics = evaluate_mot(predictions, ground_truth, match_gate_m=1.0)

        self.assertEqual(metrics["TP"], 0)
        self.assertEqual(metrics["FP"], 1)
        self.assertEqual(metrics["FN"], 1)
        self.assertEqual(metrics["IDTP"], 0)
        self.assertAlmostEqual(metrics["MOTA"], -1.0)
        self.assertAlmostEqual(metrics["HOTA"], 0.0)

    def test_match_gate_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            evaluate_mot([], {}, match_gate_m=0.0)


if __name__ == "__main__":
    unittest.main()
