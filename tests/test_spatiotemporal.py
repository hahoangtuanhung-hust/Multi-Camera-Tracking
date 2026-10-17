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
    fit_weighted_trajectory,
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
    def test_robust_fit_downweights_an_accepted_history_outlier(self) -> None:
        fit = fit_weighted_trajectory(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [8.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
            [1.0] * 6,
            reference_ts_s=5.0,
            recency_half_life_s=10.0,
            min_acceleration_duration_s=10.0,
        )

        self.assertAlmostEqual(fit.velocity_xy[0], 1.0, delta=0.20)
        self.assertFalse(fit.acceleration_reliable)

    def test_acceleration_requires_temporal_support(self) -> None:
        fit = fit_weighted_trajectory(
            [0.0, 0.1, 0.2, 0.3],
            [[0.0, 0.0], [0.01, 0.0], [0.04, 0.0], [0.09, 0.0]],
            [1.0] * 4,
            reference_ts_s=0.3,
            recency_half_life_s=0.75,
            min_acceleration_duration_s=1.0,
        )

        self.assertFalse(fit.acceleration_reliable)
        np.testing.assert_array_equal(fit.acceleration_xy, np.zeros(2))

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
        np.testing.assert_allclose(outlier.smoothed_xy, [0.2, 0.0])
        self.assertEqual(smoother.metrics["outliers_rejected"], 1)

        recovered = smoother.update(("a", 1), 0.3, [0.3, 0.0])
        self.assertTrue(recovered.accepted)
        self.assertEqual(recovered.sample_count, 3)

    def test_clipped_motion_cannot_make_an_impossible_segment_valid(self) -> None:
        smoother = SlidingWindowTrajectorySmoother(
            window_s=3.0,
            recency_half_life_s=1.0,
            outlier_gate_m=100.0,
            max_speed_mps=1.0,
            max_acceleration_mps2=4.0,
            max_jump_m=100.0,
        )
        smoother.update(("a", 1), 0.0, [0.0, 0.0])
        impossible = smoother.update(("a", 1), 1.0, [10.0, 0.0])

        self.assertFalse(impossible.accepted)
        self.assertEqual(impossible.sample_count, 1)
        self.assertEqual(smoother.metrics["speed_limit_rejected"], 1)
        np.testing.assert_allclose(impossible.smoothed_xy, [0.0, 0.0])

    def test_weighted_regression_estimates_velocity_and_acceleration(self) -> None:
        linear = SlidingWindowTrajectorySmoother(
            window_s=4.0,
            recency_half_life_s=1.0,
            outlier_gate_m=20.0,
            max_jump_m=100.0,
        )
        linear.update(("a", 1), 0.0, [0.0, 0.0], confidence=1.0)
        linear.update(("a", 1), 1.0, [10.0, 0.0], confidence=0.001)
        estimate = linear.update(("a", 1), 2.0, [2.0, 0.0], confidence=1.0)
        self.assertAlmostEqual(estimate.velocity_xy[0], 1.0, delta=0.02)

        accelerated = SlidingWindowTrajectorySmoother(
            window_s=4.0,
            outlier_gate_m=5.0,
            max_jump_m=100.0,
            max_speed_mps=10.0,
            max_acceleration_mps2=5.0,
        )
        result = None
        for timestamp in (0.0, 0.5, 1.0, 1.5):
            result = accelerated.update(
                ("a", 2), timestamp, [timestamp * timestamp, 0.0]
            )
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.smoothed_xy, [2.25, 0.0], atol=1e-8)
        np.testing.assert_allclose(result.velocity_xy, [3.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(result.acceleration_xy, [2.0, 0.0], atol=1e-8)

    def test_out_of_order_sample_does_not_rewind_or_mutate_history(self) -> None:
        smoother = SlidingWindowTrajectorySmoother(window_s=3.5)
        smoother.update(("a", 1), 1.0, [1.0, 0.0])
        smoother.update(("a", 1), 2.0, [2.0, 0.0])

        late = smoother.update(("a", 1), 1.5, [100.0, 0.0])
        recovered = smoother.update(("a", 1), 3.0, [3.0, 0.0])

        self.assertFalse(late.accepted)
        self.assertTrue(recovered.accepted)
        self.assertEqual(recovered.sample_count, 3)
        np.testing.assert_allclose(recovered.velocity_xy, [1.0, 0.0])
        self.assertEqual(smoother.metrics["out_of_order_rejected"], 1)

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

    def test_global_prune_bounds_inactive_tracklet_state(self) -> None:
        smoother = SlidingWindowTrajectorySmoother(window_s=1.0)
        smoother.update(("a", 1), 0.0, [0.0, 0.0])
        smoother.update(("b", 2), 0.5, [1.0, 0.0])

        smoother.prune(2.0)

        self.assertEqual(smoother.histories, {})
        self.assertEqual(smoother.metrics["histories_pruned"], 2)


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
