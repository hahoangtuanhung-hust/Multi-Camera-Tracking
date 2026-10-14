import unittest

import numpy as np

from multicam_tracker.mot_metrics import (
    build_identity_error_taxonomy,
    evaluate_mot,
    evaluate_mot_slices,
    identity_trajectory_diagnostics,
)


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

    def test_slices_keep_cumulative_and_independent_metrics_separate(self) -> None:
        predictions = [
            (frame, {(10 if frame < 2 else 11): self._point(frame)})
            for frame in range(4)
        ]
        ground_truth = {
            frame: {1: self._point(frame)} for frame in range(4)
        }

        report = evaluate_mot_slices(
            predictions,
            ground_truth,
            cumulative_frames=[2, 4],
            window_frames=2,
        )

        self.assertEqual([item["ticks"] for item in report["cumulative_metrics"]], [2, 4])
        self.assertEqual(len(report["metric_windows"]), 2)
        self.assertAlmostEqual(report["metric_windows"][0]["IDF1"], 1.0)
        self.assertAlmostEqual(report["metric_windows"][1]["IDF1"], 1.0)
        self.assertAlmostEqual(report["cumulative_metrics"][-1]["IDF1"], 0.5)

    def test_identity_diagnostics_exposes_gid_fragmentation_and_long_gap(self) -> None:
        predictions = [
            (0, {10: self._point(0.0), 20: self._point(0.4)}),
            (1, {11: self._point(1.0), 20: self._point(1.4)}),
            (4, {12: self._point(4.0), 20: self._point(4.4)}),
        ]
        ground_truth = {
            frame: {1: self._point(float(frame)), 2: self._point(float(frame), 0.4)}
            for frame in (0, 1, 4)
        }

        report = identity_trajectory_diagnostics(
            predictions,
            ground_truth,
            long_gap_frames=2,
            crossing_distance_m=0.5,
        )
        person = next(item for item in report["per_gt"] if item["gt_id"] == 1)

        self.assertEqual(person["gid_count"], 3)
        self.assertEqual(person["switch_count"], 2)
        self.assertEqual(person["longest_missing_gap"], 2)
        self.assertEqual(report["predicted_gids_per_gt"]["max"], 3.0)
        self.assertEqual(report["long_gap_reconnect_attempts"], 2)
        self.assertEqual(report["long_gap_reconnect_successes"], 1)
        self.assertGreater(report["crossing_switch_rate"], 0.0)

    def test_identity_error_taxonomy_preserves_evidence_quality(self) -> None:
        taxonomy = build_identity_error_taxonomy(
            {"FN": 12},
            {
                "long_gap_reconnect_attempts": 4,
                "long_gap_reconnect_successes": 1,
            },
            {
                "local_tracks_removed_tentative": 5,
                "fusion_motion_jump_rejected": 7,
                "trajectory_matches_rejected": 8,
                "new_gid_trajectory_rejected": 2,
                "new_gid_entry_deferred": 9,
            },
            switch_diagnostics=[{
                "near_crossing": True,
                "classification": "local_fragmentation",
                "gap_frames": 3,
            }, {
                "near_crossing": False,
                "classification": "global_changed_local_stable",
                "gap_frames": 1,
            }],
            track_exists_not_output=11,
        )

        self.assertEqual(taxonomy["detector_miss"]["count"], 12)
        self.assertEqual(
            taxonomy["detector_miss"]["evidence"], "upper_bound_proxy"
        )
        self.assertEqual(taxonomy["local_switch_at_crossing"]["count"], 1)
        self.assertEqual(taxonomy["dormant_reconnect_failed"]["count"], 3)
        self.assertEqual(
            taxonomy["trajectory_verification_rejection"]["count"], 10
        )
        self.assertEqual(taxonomy["track_exists_not_output"]["count"], 11)


if __name__ == "__main__":
    unittest.main()
