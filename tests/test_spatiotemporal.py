import unittest

import numpy as np

from multicam_tracker.spatiotemporal import (
    ProjectionError,
    SlidingWindowTrajectorySmoother,
    SpatialConsistencyMatcher,
    SpatioTemporalAssociator,
    TopDownProjector,
    WorldTrackletObservation,
    build_spatial_cost_matrix,
    estimate_ground_plane_location,
    get_ground_plane_location,
    reassignment_confidence,
    spatial_consistency_distance,
)


class GroundPlaneLocationTests(unittest.TestCase):
    def test_confident_ankles_use_their_midpoint(self) -> None:
        keypoints = np.zeros((17, 3), dtype=float)
        keypoints[15] = [20.0, 110.0, 0.91]
        keypoints[16] = [40.0, 114.0, 0.87]

        result = estimate_ground_plane_location(
            [10.0, 20.0, 40.0, 100.0], keypoints, tau_pose=0.5
        )

        np.testing.assert_allclose(result.xy, [30.0, 112.0])
        self.assertEqual(result.source, "ankles")

    def test_missing_or_low_confidence_ankle_falls_back_to_bbox(self) -> None:
        keypoints = np.zeros((17, 3), dtype=float)
        keypoints[15] = [20.0, 110.0, 0.90]
        keypoints[16] = [40.0, 114.0, 0.50]

        xywh = get_ground_plane_location(
            [10.0, 20.0, 40.0, 100.0], keypoints, tau_pose=0.5
        )
        xyxy = get_ground_plane_location(
            [10.0, 20.0, 50.0, 120.0], None, bbox_format="xyxy"
        )

        np.testing.assert_allclose(xywh, [30.0, 120.0])
        np.testing.assert_allclose(xyxy, [30.0, 120.0])

    def test_invalid_bbox_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "x2>x1"):
            get_ground_plane_location(
                [10.0, 20.0, 5.0, 30.0], None, bbox_format="xyxy"
            )


class TopDownProjectionTests(unittest.TestCase):
    def test_projector_performs_homogeneous_divide(self) -> None:
        projector = TopDownProjector({
            "a": np.array([
                [2.0, 0.0, 10.0],
                [0.0, 3.0, -5.0],
                [0.1, 0.0, 1.0],
            ])
        })

        result = projector.project("a", [2.0, 4.0])

        np.testing.assert_allclose(result, [14.0 / 1.2, 7.0 / 1.2])

    def test_projector_supports_batches_and_rejects_horizon(self) -> None:
        identity = TopDownProjector({"a": np.eye(3)})
        points = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(identity.project("a", points), points)

        horizon = TopDownProjector({
            "a": np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, -1.0],
            ])
        })
        with self.assertRaises(ProjectionError):
            horizon.project("a", [1.0, 3.0])


class SlidingWindowSmoothingTests(unittest.TestCase):
    def test_weighted_window_rejects_sudden_outlier(self) -> None:
        smoother = SlidingWindowTrajectorySmoother(
            window_s=1.0,
            recency_half_life_s=0.2,
            outlier_gate_m=0.25,
            max_speed_mps=2.0,
        )
        smoother.update(("a", 1), 0.0, [0.0, 0.0])
        normal = smoother.update(("a", 1), 0.1, [0.1, 0.0])
        outlier = smoother.update(("a", 1), 0.2, [10.0, 0.0])

        self.assertTrue(normal.accepted)
        self.assertFalse(outlier.accepted)
        self.assertEqual(outlier.sample_count, 2)
        self.assertLess(outlier.smoothed_xy[0], 0.1)
        self.assertEqual(smoother.metrics["outliers_rejected"], 1)

    def test_window_prunes_old_samples(self) -> None:
        smoother = SlidingWindowTrajectorySmoother(
            window_s=0.5,
            outlier_gate_m=1.0,
            max_speed_mps=3.2,
        )
        smoother.update(("a", 1), 0.0, [0.0, 0.0])
        result = smoother.update(("a", 1), 1.0, [1.0, 0.0])

        self.assertEqual(result.sample_count, 1)
        np.testing.assert_allclose(result.smoothed_xy, [1.0, 0.0])
        self.assertEqual(smoother.metrics["samples_pruned"], 1)


class SpatialConsistencyTests(unittest.TestCase):
    @staticmethod
    def _observation(
        camera: str,
        local_id: int,
        timestamp_s: float,
        xy: tuple[float, float],
        velocity: tuple[float, float] | None = None,
    ) -> WorldTrackletObservation:
        return WorldTrackletObservation(
            camera_id=camera,
            local_id=local_id,
            timestamp_s=timestamp_s,
            world_xy=np.asarray(xy, dtype=float),
            velocity_xy=(
                None if velocity is None else np.asarray(velocity, dtype=float)
            ),
        )

    def test_euclidean_cost_matrix(self) -> None:
        matrix = build_spatial_cost_matrix(
            [[0.0, 0.0], [3.0, 4.0]],
            [[0.0, 4.0], [3.0, 0.0]],
        )

        np.testing.assert_allclose(matrix, [[4.0, 3.0], [3.0, 4.0]])

    def test_hungarian_assignment_is_one_to_one_and_order_independent(self) -> None:
        matcher = SpatialConsistencyMatcher(
            max_distance_m=1.0,
            max_time_delta_s=0.2,
        )
        left = [
            self._observation("a", 1, 1.0, (0.0, 0.0)),
            self._observation("a", 2, 1.0, (10.0, 0.0)),
        ]
        right = [
            self._observation("b", 20, 1.05, (10.1, 0.0)),
            self._observation("b", 10, 1.05, (0.1, 0.0)),
        ]

        result = matcher.match(left, right)
        pairs = {(match.left_key, match.right_key) for match in result.matches}

        self.assertEqual(
            pairs,
            {
                (("a", 1), ("b", 10)),
                (("a", 2), ("b", 20)),
            },
        )
        self.assertEqual(result.unmatched_left, ())
        self.assertEqual(result.unmatched_right, ())

    def test_short_time_offset_is_compensated_by_velocity(self) -> None:
        matcher = SpatialConsistencyMatcher(
            max_distance_m=0.1,
            max_time_delta_s=1.0,
            temporal_weight_m_per_s=0.0,
        )
        left = [self._observation("a", 1, 0.0, (0.0, 0.0), (1.0, 0.0))]
        right = [self._observation("b", 7, 1.0, (1.0, 0.0))]

        result = matcher.match(left, right)

        self.assertEqual(len(result.matches), 1)
        self.assertAlmostEqual(result.matches[0].spatial_distance_m, 0.0)

    def test_distance_and_time_gates_leave_tracks_unmatched(self) -> None:
        matcher = SpatialConsistencyMatcher(
            max_distance_m=1.0,
            max_time_delta_s=0.1,
        )
        left = [self._observation("a", 1, 0.0, (0.0, 0.0))]
        right = [self._observation("b", 7, 0.2, (0.1, 0.0))]

        result = matcher.match(left, right)

        self.assertEqual(result.matches, ())
        self.assertEqual(result.unmatched_left, (0,))
        self.assertEqual(result.unmatched_right, (0,))

    def test_spatial_consistency_excludes_peer_outlier(self) -> None:
        distance = spatial_consistency_distance(
            [0.0, 0.0],
            [[0.0, 0.0], [0.1, 0.0], [100.0, 100.0]],
        )

        self.assertLess(distance, 0.01)
        self.assertAlmostEqual(reassignment_confidence(4.0, 1.0), 0.75)


class AssociatorFacadeTests(unittest.TestCase):
    def test_detection_to_cross_camera_match_without_reid(self) -> None:
        associator = SpatioTemporalAssociator(
            TopDownProjector({"a": np.eye(3), "b": np.eye(3)}),
            smoother=SlidingWindowTrajectorySmoother(
                window_s=1.0,
                outlier_gate_m=1.0,
                max_speed_mps=3.2,
            ),
            matcher=SpatialConsistencyMatcher(
                max_distance_m=0.5,
                max_time_delta_s=0.1,
            ),
            tau_pose=0.5,
        )
        keypoints = np.zeros((17, 3), dtype=float)
        keypoints[15] = [1.0, 2.0, 0.9]
        keypoints[16] = [3.0, 2.0, 0.9]
        left = associator.observe(
            camera_id="a",
            local_id=1,
            timestamp_s=0.0,
            detection=[0.0, 0.0, 4.0, 2.0],
            pose_keypoints=keypoints,
            bbox_format="xywh",
        )
        right = associator.observe(
            camera_id="b",
            local_id=7,
            timestamp_s=0.0,
            detection=[0.1, 0.0, 4.0, 2.0],
            pose_keypoints=None,
            bbox_format="xywh",
        )

        result = associator.associate([left], [right])

        self.assertEqual(left.ground_point_source, "ankles")
        self.assertEqual(right.ground_point_source, "bbox")
        self.assertEqual(len(result.matches), 1)


if __name__ == "__main__":
    unittest.main()
