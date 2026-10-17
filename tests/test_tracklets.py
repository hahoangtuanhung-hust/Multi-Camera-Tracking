import unittest

import numpy as np

from multicam_tracker.config import PipelineConfig
from multicam_tracker.tracklets import (
    CameraClockOffsetEstimator,
    ClockEstimate,
    TrackletMatcher,
    TrackletStore,
)


def _history(
    camera: str,
    local_id: int,
    event_times: np.ndarray,
    timestamp_offset: float = 0.0,
    xy_offset: tuple[float, float] = (0.0, 0.0),
    reverse: bool = False,
    drop_every: int = 0,
    jitter: float = 0.0,
):
    cfg = PipelineConfig(tracklet_history_s=20.0)
    store = TrackletStore(cfg)
    rng = np.random.default_rng(7)
    for index, event_time in enumerate(event_times):
        if drop_every and index % drop_every == 0:
            continue
        motion_time = -event_time if reverse else event_time
        xy = np.array([0.7 * motion_time + xy_offset[0], 0.2 * motion_time + xy_offset[1]])
        if jitter:
            xy += rng.normal(0.0, jitter, 2)
        store.append(camera, local_id, event_time + timestamp_offset, xy, 0.95)
    return store.get((camera, local_id))


class TrackletTests(unittest.TestCase):
    def test_cached_arrays_refresh_after_append_replace_and_expiry(self):
        store = TrackletStore(PipelineConfig(tracklet_history_s=3.0))
        store.append("a", 1, 0., np.zeros(2), .9)
        history = store.get(("a", 1))
        first = history.arrays()
        self.assertIs(first, history.arrays())
        store.append("a", 1, 1., np.ones(2), .9)
        self.assertEqual(len(history.arrays()[0]), 2)
        store.append("a", 1, 1., np.array([2., 3.]), .8)
        np.testing.assert_array_equal(history.arrays()[1][-1], [2., 3.])
        self.assertEqual(store.metrics["points_stored"], 2)
        store.advance(3.5)
        np.testing.assert_array_equal(history.arrays()[0], [1.])
        self.assertEqual(store.metrics["points_stored"], 1)
        store.advance(5.)
        self.assertEqual(store.metrics["points_stored"], 0)

    def setUp(self) -> None:
        self.cfg = PipelineConfig(tracklet_history_s=20.0)
        self.matcher = TrackletMatcher(self.cfg)
        self.times = np.arange(0.0, 3.01, 0.1)

    def test_finds_positive_camera_timestamp_offset(self) -> None:
        left = _history("a", 1, self.times)
        right = _history("b", 9, self.times, timestamp_offset=2.0)
        result = self.matcher.best_offset(left, right)
        self.assertTrue(result.accepted)
        self.assertAlmostEqual(result.delta_s, -2.0, delta=0.051)
        self.assertLess(result.position_rmse_m, 0.03)

    def test_finds_negative_camera_timestamp_offset(self) -> None:
        left = _history("a", 1, self.times, timestamp_offset=1.5)
        right = _history("b", 9, self.times)
        result = self.matcher.best_offset(left, right)
        self.assertTrue(result.accepted)
        self.assertAlmostEqual(result.delta_s, 1.5, delta=0.051)

    def test_jitter_and_dropped_points_still_match(self) -> None:
        left = _history("a", 1, self.times)
        right = _history(
            "b", 9, self.times, timestamp_offset=1.2, drop_every=4, jitter=0.005
        )
        result = self.matcher.best_offset(left, right)
        self.assertTrue(result.accepted)
        self.assertAlmostEqual(result.delta_s, -1.2, delta=0.11)

    def test_spatially_different_trajectory_is_rejected(self) -> None:
        left = _history("a", 1, self.times)
        # Lateral separation cannot be explained by a clock shift.
        right = _history("b", 9, self.times, xy_offset=(0.0, 2.0))
        result = self.matcher.best_offset(left, right)
        self.assertFalse(result.accepted)

    def test_opposite_direction_is_rejected(self) -> None:
        left = _history("a", 1, self.times)
        right = _history("b", 9, self.times, reverse=True)
        result = self.matcher.compare(left, right, 0.0)
        self.assertFalse(result.accepted)
        self.assertLess(result.direction_cosine, 0.0)

    def test_short_tracklet_cannot_match(self) -> None:
        short_times = np.arange(0.0, 0.51, 0.1)
        left = _history("a", 1, short_times)
        right = _history("b", 9, short_times)
        result = self.matcher.best_offset(left, right)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "insufficient_overlap")

    def test_assignment_margin_rejects_ambiguous_twins(self) -> None:
        left = _history("a", 1, self.times)
        right_one = _history("b", 8, self.times)
        right_two = _history("b", 9, self.times)
        self.assertEqual(self.matcher.assign([left], [right_one, right_two]), [])

    def test_history_is_bounded_and_too_late_data_is_dropped(self) -> None:
        cfg = PipelineConfig(tracklet_history_s=5.0, tracklet_late_drop_s=5.0)
        store = TrackletStore(cfg)
        for timestamp in range(11):
            store.append("a", 1, timestamp, np.array([timestamp, 0.0]), 0.9)
        history = store.get(("a", 1))
        self.assertGreaterEqual(history.start_s, 5.0)
        self.assertFalse(store.append("a", 1, 4.0, np.array([4.0, 0.0]), 0.9))
        self.assertEqual(store.metrics["late_dropped"], 1)

    def test_clock_estimator_rejects_outlier_and_models_drift(self) -> None:
        estimator = CameraClockOffsetEstimator(self.cfg, reference_camera="a")
        for timestamp in range(20):
            correction = -2.0 + 0.001 * timestamp
            estimator.observe("b", timestamp, correction, 0.95)
        before = estimator.estimate("b")
        estimator.observe("b", 21.0, 1.5, 0.99)
        after = estimator.estimate("b")
        self.assertAlmostEqual(after.offset_s, before.offset_s, delta=0.05)
        self.assertGreater(after.scale, 1.0)
        self.assertGreater(after.confidence, 0.8)

    def test_clock_learning_can_be_disabled_or_limited_to_offset(self) -> None:
        fixed_cfg = PipelineConfig(clock_learning_enabled=False)
        fixed = CameraClockOffsetEstimator(fixed_cfg, reference_camera="a")
        for timestamp in (0.0, 5.0, 10.0):
            fixed.observe("b", timestamp, 0.5 + 0.001 * timestamp, 0.9)
        self.assertEqual(fixed.estimate("b"), ClockEstimate())

        offset_cfg = PipelineConfig(
            offset_max_drift_ppm=0.0,
            offset_min_drift_span_s=1.0,
        )
        offset_only = CameraClockOffsetEstimator(
            offset_cfg, reference_camera="a"
        )
        for timestamp in (0.0, 5.0, 10.0):
            offset_only.observe(
                "b", timestamp, 0.5 + 0.001 * timestamp, 0.9
            )
        estimate = offset_only.estimate("b")
        self.assertEqual(estimate.scale, 1.0)
        self.assertGreater(estimate.samples, 0)


if __name__ == "__main__":
    unittest.main()
