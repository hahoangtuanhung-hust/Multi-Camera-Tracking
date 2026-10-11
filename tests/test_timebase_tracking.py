import numpy as np
import unittest

from multicam_tracker.bytetrack import ByteTracker, ConstantVelocityKalman
from multicam_tracker.config import PipelineConfig
from multicam_tracker.timebase import ClockModel


def _det(x: float = 0.0) -> np.ndarray:
    return np.array([[x, 0.0, x + 10.0, 20.0, 0.9]], dtype=float)


class TimebaseTrackingTests(unittest.TestCase):
    def test_cached_kalman_matrices_match_constant_acceleration_equations(self):
        rng = np.random.default_rng(42)
        kf = ConstantVelocityKalman(np.array([1., 2.]), .05, .1)
        state, covariance = kf.x.copy(), kf.P.copy()
        for dt in (0., .033, .1, .5, 2., .033):
            transition = np.eye(4)
            transition[:2, 2:] = np.eye(2) * dt
            process = .05 * np.block([[np.eye(2) * dt**4 / 4, np.eye(2) * dt**3 / 2],
                                     [np.eye(2) * dt**3 / 2, np.eye(2) * dt**2]])
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
            kf.predict(dt)
            np.testing.assert_allclose(kf.x, state, atol=1e-12)
            np.testing.assert_allclose(kf.P, covariance, atol=1e-12)
            measurement = rng.normal(size=2)
            gain = covariance @ kf.H.T @ np.linalg.inv(kf.H @ covariance @ kf.H.T + kf.R)
            state += gain @ (measurement - kf.H @ state)
            covariance = (np.eye(4) - gain @ kf.H) @ covariance
            kf.update(measurement)

    def test_clock_model_applies_scale_then_offset(self) -> None:
        clock = ClockModel(offset_s=-1.5, scale=1.001)
        self.assertAlmostEqual(clock.normalize(10.0), 8.51)

    def test_kalman_prediction_uses_variable_dt(self) -> None:
        kf = ConstantVelocityKalman(np.array([0.0]), 0.01, 0.01)
        kf.x[1] = 2.0
        self.assertAlmostEqual(kf.predict(dt=0.25)[0], 0.5)
        self.assertAlmostEqual(kf.predict(dt=1.5)[0], 3.5)

    def test_tracker_timeout_is_seconds_with_irregular_frames(self) -> None:
        cfg = PipelineConfig(track_min_hits=2, track_max_age_s=1.0)
        tracker = ByteTracker(cfg)
        tracker.update(_det(), timestamp_s=10.0)
        visible = tracker.update(_det(), timestamp_s=10.1)
        self.assertEqual(len(visible), 1)

        tracker.update(np.empty((0, 5)), timestamp_s=10.8)
        self.assertEqual(len(tracker.tracks), 1)
        tracker.update(np.empty((0, 5)), timestamp_s=11.2)
        self.assertEqual(tracker.tracks, [])

    def test_tracker_rejects_non_monotonic_capture_time(self) -> None:
        tracker = ByteTracker(PipelineConfig())
        tracker.update(_det(), timestamp_s=2.0)
        with self.assertRaisesRegex(ValueError, "monotonic"):
            tracker.update(_det(), timestamp_s=1.9)

    def test_legacy_tick_api_is_still_supported(self) -> None:
        cfg = PipelineConfig(track_min_hits=2, track_max_age=1)
        tracker = ByteTracker(cfg)
        tracker.update(_det())
        self.assertEqual(len(tracker.update(_det())), 1)
        tracker.update(np.empty((0, 5)))
        tracker.update(np.empty((0, 5)))
        self.assertEqual(tracker.tracks, [])
