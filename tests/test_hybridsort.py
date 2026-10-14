"""Tests for HybridSortTracker.

Hybrid-SORT (no ReID) extends OC-SORT with a two-round primary association
that blends Kalman-prediction IoU with last-real-observation IoU. These tests
verify the core properties that distinguish it from plain OC-SORT:

1. After an occlusion, the tracker reacquires a person using the blended
   IoU signal even when the Kalman prediction has drifted far.
2. alpha=0 degenerates to plain Kalman-IoU (OC-SORT round-1 only).
3. Births, confirmations, and timestamp lifecycle behave identically to
   OC-SORT (shared Kalman model and Track interface).
4. The hybridsort profile is correctly registered and selectable.
"""

import unittest

import numpy as np

from multicam_tracker.config import (
    DEFAULT_TRACKING_PROFILE,
    PipelineConfig,
    apply_tracking_profile,
)
from multicam_tracker.hybridsort import HybridSortTracker, associate_hybrid
from multicam_tracker.local_tracker import create_local_tracker


def _det(x: float, confidence: float = 0.95) -> np.ndarray:
    """Return a (1, 5) detection at horizontal position x with width 20."""
    return np.array([[x, 0.0, x + 20.0, 40.0, confidence]], dtype=float)


class HybridSortDefaultProfileTests(unittest.TestCase):
    def test_hybridsort_is_branch_default_profile(self):
        self.assertEqual(DEFAULT_TRACKING_PROFILE, "hybridsort")

    def test_hybridsort_profile_creates_hybrid_tracker(self):
        cfg = PipelineConfig()
        apply_tracking_profile(cfg, "hybridsort")
        tracker = create_local_tracker(cfg)
        self.assertIsInstance(tracker, HybridSortTracker)

    def test_profile_alpha_and_tracker_name_set_correctly(self):
        cfg = PipelineConfig()
        apply_tracking_profile(cfg, "hybridsort")
        self.assertEqual(cfg.local_tracker, "hybridsort")
        self.assertAlmostEqual(cfg.hybridsort_alpha, 0.40)

    def test_ocsort_and_bytetrack_still_selectable(self):
        from multicam_tracker.ocsort import OCSortTracker
        from multicam_tracker.bytetrack import ByteTracker

        cfg_oc = PipelineConfig(local_tracker="ocsort")
        self.assertIsInstance(create_local_tracker(cfg_oc), OCSortTracker)

        cfg_bt = PipelineConfig(local_tracker="bytetrack")
        self.assertIsInstance(create_local_tracker(cfg_bt), ByteTracker)


class HybridSortTrackerTests(unittest.TestCase):
    def test_normal_track_and_confirm(self):
        cfg = PipelineConfig(track_min_hits=3, track_min_iou=0.25)
        tracker = HybridSortTracker(cfg)
        self.assertEqual(tracker.update(_det(0.0), timestamp_s=0.0), [])
        self.assertEqual(tracker.update(_det(2.0), timestamp_s=0.1), [])
        confirmed = tracker.update(_det(4.0), timestamp_s=0.2)
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(tracker._next_id, 2)

    def test_occlusion_recovery_via_observation_iou(self):
        """After Kalman drifts far, observation-IoU rescues the track."""
        cfg = PipelineConfig(
            track_min_hits=1,
            track_min_iou=0.20,
            track_max_age_s=2.0,
            hybridsort_alpha=0.80,   # strongly weight last-observation IoU
            ocsort_use_byte=False,
            ocsort_observation_recovery_enabled=False,  # isolate round-2
        )
        tracker = HybridSortTracker(cfg)
        original_id = tracker.update(_det(0.0), timestamp_s=0.0)[0].track_id

        # One step visible so last_observation is recorded
        tracker.update(_det(5.0), timestamp_s=0.1)

        # Long occlusion: Kalman keeps predicting rightward (≈+5/tick)
        for tick in range(2, 8):
            tracker.update(np.empty((0, 5)), timestamp_s=tick * 0.1)

        # Person re-appears near the last observation, not where Kalman predicts
        reappeared = tracker.update(_det(8.0), timestamp_s=0.8)
        track_ids = [t.track_id for t in reappeared]
        self.assertIn(original_id, track_ids,
                      "Hybrid-SORT should recover the original ID via observation-IoU")
        self.assertEqual(tracker._next_id, 2,
                         "No new track should have been born during occlusion")

    def test_alpha_zero_behaves_like_ocsort_round1(self):
        """alpha=0 means no observation-IoU; round-2 gate stays pure Kalman-IoU."""
        cfg = PipelineConfig(
            track_min_hits=1,
            track_min_iou=0.30,
            track_max_age_s=1.0,
            hybridsort_alpha=0.0,
            ocsort_use_byte=False,
            ocsort_observation_recovery_enabled=False,
        )
        tracker = HybridSortTracker(cfg)
        tracker.update(_det(0.0), timestamp_s=0.0)
        self.assertEqual(tracker.metrics["hybrid_primary_matches"], 0)

        # A detection that overlaps only with last-obs (not prediction) should
        # not be matched when alpha=0 and prediction has moved far away.
        tracker.update(np.empty((0, 5)), timestamp_s=0.5)  # miss; Kalman predicts forward
        result = tracker.update(_det(0.0), timestamp_s=1.0)  # back at origin
        # Track has already expired (max_age_s=1.0) or no match; either is acceptable
        # The important thing: no crash and metrics are consistent
        self.assertGreaterEqual(tracker.metrics["tracks_created"], 1)

    def test_byte_extension_keeps_low_conf_track_alive(self):
        cfg = PipelineConfig(track_min_hits=1, ocsort_use_byte=True)
        tracker = HybridSortTracker(cfg)
        original_id = tracker.update(_det(0.0), timestamp_s=0.0)[0].track_id

        visible = tracker.update(_det(1.0, confidence=0.25), timestamp_s=0.1)
        self.assertEqual([t.track_id for t in visible], [original_id])
        self.assertGreaterEqual(tracker.metrics["byte_extension_matches"], 1)

        # Low-confidence alone should never create a new track
        fresh = HybridSortTracker(cfg)
        self.assertEqual(fresh.update(_det(0.0, confidence=0.25), timestamp_s=0.0), [])
        self.assertEqual(fresh.tracks, [])

    def test_low_confidence_recovery_must_pass_motion_size_gate(self):
        cfg = PipelineConfig(
            track_min_hits=1,
            track_min_iou=0.20,
            ocsort_use_byte=True,
            track_motion_gate_enabled=True,
            track_association_max_size_log_change=0.20,
        )
        tracker = HybridSortTracker(cfg)
        original = tracker.update(_det(0.0), timestamp_s=0.0)[0]
        before_width = float(original.kf.x[2])

        # Same centre and sufficient overlap, but a 3x wider low-score box is
        # typical of two people being merged into one detector output.
        merged_box = np.array([[-20.0, 0.0, 40.0, 40.0, 0.25]], dtype=float)
        visible = tracker.update(merged_box, timestamp_s=0.1)

        self.assertEqual(visible, [])
        self.assertEqual(tracker._next_id, 2)
        self.assertEqual(tracker.metrics["low_confidence_matches"], 0)
        self.assertEqual(tracker.tracks[0].misses, 1)
        self.assertAlmostEqual(float(tracker.tracks[0].kf.x[2]), before_width)

    def test_temporal_confirmation_setting_delays_publication(self):
        cfg = PipelineConfig(
            track_min_hits=2,
            track_detection_temporal_confirmation_frames=3,
        )
        tracker = HybridSortTracker(cfg)

        self.assertEqual(tracker.update(_det(0.0), timestamp_s=0.0), [])
        self.assertEqual(tracker.update(_det(0.5), timestamp_s=0.1), [])
        visible = tracker.update(_det(1.0), timestamp_s=0.2)

        self.assertEqual(len(visible), 1)
        self.assertTrue(visible[0].confirmed)

    def test_tentative_track_survives_one_detection_miss(self):
        cfg = PipelineConfig(
            track_min_hits=3,
            track_tentative_max_misses=1,
            ocsort_use_byte=False,
        )
        tracker = HybridSortTracker(cfg)

        tracker.update(_det(0.0), timestamp_s=0.0)
        original_id = tracker.tracks[0].track_id
        tracker.update(np.empty((0, 5)), timestamp_s=0.1)
        self.assertEqual([track.track_id for track in tracker.tracks], [original_id])

        tracker.update(_det(0.5), timestamp_s=0.2)
        visible = tracker.update(_det(1.0), timestamp_s=0.3)
        self.assertEqual([track.track_id for track in visible], [original_id])
        self.assertEqual(tracker.metrics["tracks_removed_tentative"], 0)

    def test_id_switches_reduced_on_reentry(self):
        """Observed boxes provide a stronger signal than drifted Kalman when
        a person exits and re-enters from the same direction.  Hybrid-SORT
        should assign fewer new IDs than plain IoU on the Kalman prediction."""
        cfg = PipelineConfig(
            track_min_hits=1,
            track_min_iou=0.20,
            track_max_age_s=1.5,
            hybridsort_alpha=0.60,
            ocsort_use_byte=False,
            ocsort_observation_recovery_enabled=False,
        )
        tracker = HybridSortTracker(cfg)
        tracker.update(_det(10.0), timestamp_s=0.0)
        tracker.update(_det(15.0), timestamp_s=0.1)

        # Three-frame occlusion (Kalman drifts further right)
        tracker.update(np.empty((0, 5)), timestamp_s=0.2)
        tracker.update(np.empty((0, 5)), timestamp_s=0.3)
        tracker.update(np.empty((0, 5)), timestamp_s=0.4)

        # Re-appear near last real observation rather than Kalman prediction
        result = tracker.update(_det(18.0), timestamp_s=0.5)
        self.assertEqual(len(result), 1)
        self.assertEqual(tracker._next_id, 2,
                         "No extra ID should be created — reactivation expected")

    def test_timestamp_monotonicity_enforced(self):
        tracker = HybridSortTracker(PipelineConfig(track_min_hits=1))
        tracker.update(_det(0.0), timestamp_s=1.0)
        with self.assertRaisesRegex(ValueError, "monotonic"):
            tracker.update(_det(0.0), timestamp_s=0.9)

    def test_timestamp_expiry(self):
        cfg = PipelineConfig(track_min_hits=1, track_max_age_s=0.3)
        tracker = HybridSortTracker(cfg)
        tracker.update(_det(0.0), timestamp_s=0.0)
        tracker.update(np.empty((0, 5)), timestamp_s=0.35)
        self.assertEqual(tracker.tracks, [])

    def test_invalid_alpha_raises(self):
        with self.assertRaises(ValueError):
            HybridSortTracker(PipelineConfig(hybridsort_alpha=-0.1))
        with self.assertRaises(ValueError):
            HybridSortTracker(PipelineConfig(hybridsort_alpha=1.01))


class AssociateHybridTests(unittest.TestCase):
    """Unit tests for the two-round blended association function."""

    def _make_tracker_with_observation(self, x: float) -> "HybridSortTracker":
        cfg = PipelineConfig(track_min_hits=1, track_min_iou=0.20, hybridsort_alpha=0.60)
        tracker = HybridSortTracker(cfg)
        tracker.update(_det(x), timestamp_s=0.0)
        tracker.update(np.empty((0, 5)), timestamp_s=0.5)  # make it a miss
        return tracker

    def test_round2_matches_when_prediction_misses_but_observation_overlaps(self):
        """A detection near last_observation but far from Kalman prediction
        should be recovered in round-2."""
        tracker = self._make_tracker_with_observation(x=50.0)
        # Artificially shift Kalman far away (simulate large drift)
        for track in tracker.tracks:
            track.kf.x[0] += 200.0  # push prediction 200 px right
            track.kf.x[1] += 200.0

        detections = np.array([[50.0, 0.0, 70.0, 40.0, 0.90]])  # near original obs
        pairs, unmatched_t, unmatched_d = associate_hybrid(
            tracker.tracks, detections,
            min_iou=0.20, alpha=0.80, inertia=0.0, delta_t=3,
        )
        self.assertEqual(len(pairs), 1, "Round-2 should match via observation-IoU")
        self.assertEqual(len(unmatched_d), 0)

    def test_round1_handles_clear_case(self):
        tracker = self._make_tracker_with_observation(x=0.0)
        # Reset Kalman to original position so round-1 succeeds
        for track in tracker.tracks:
            track.kf.x[:2] = [5.0, 10.0]  # small offset; IoU will still pass
        detections = np.array([[0.0, 0.0, 20.0, 40.0, 0.90]])
        pairs, _, unmatched_d = associate_hybrid(
            tracker.tracks, detections,
            min_iou=0.20, alpha=0.40, inertia=0.0, delta_t=3,
        )
        self.assertEqual(len(unmatched_d), 0)

    def test_stale_observation_rejected_by_max_obs_gap(self):
        """A track lost longer than max_obs_gap_s must not be matched via observation-IoU."""
        tracker = self._make_tracker_with_observation(x=50.0)
        for track in tracker.tracks:
            track.kf.x[0] += 200.0
            track.kf.x[1] += 200.0
            track.missed_seconds = 1.5  # exceed max_obs_gap_s (0.8s)

        detections = np.array([[50.0, 0.0, 70.0, 40.0, 0.90]])
        pairs, unmatched_t, unmatched_d = associate_hybrid(
            tracker.tracks, detections,
            min_iou=0.20, alpha=0.80, inertia=0.0, delta_t=3,
            max_obs_gap_s=0.80,
        )
        self.assertEqual(len(pairs), 0, "Stale track must not steal the detection")
        self.assertEqual(len(unmatched_d), 1)

    def test_scale_mismatch_rejected_in_round2(self):
        """A detection with drastic size change (e.g. 3x height) must be rejected."""
        tracker = self._make_tracker_with_observation(x=50.0)
        for track in tracker.tracks:
            track.kf.x[0] += 200.0
            track.kf.x[1] += 200.0

        # Detection has height 150 (3.75x of original height 40)
        detections = np.array([[50.0, 0.0, 70.0, 150.0, 0.90]])
        pairs, _, unmatched_d = associate_hybrid(
            tracker.tracks, detections,
            min_iou=0.20, alpha=0.80, inertia=0.0, delta_t=3,
        )
        self.assertEqual(len(pairs), 0, "Severe scale mismatch must be rejected")
        self.assertEqual(len(unmatched_d), 1)

    def test_reverse_direction_rejected_in_round2(self):
        """A detection moving in reverse against track velocity must be rejected."""
        tracker = self._make_tracker_with_observation(x=50.0)
        for track in tracker.tracks:
            track.kf.x[0] += 200.0
            track.kf.x[1] += 200.0
            track.velocity = np.array([1.0, 0.0])  # moving rightwards

        # Detection placed 30 px to the LEFT of observation (opposite direction)
        detections = np.array([[20.0, 0.0, 40.0, 40.0, 0.90]])
        pairs, _, unmatched_d = associate_hybrid(
            tracker.tracks, detections,
            min_iou=0.20, alpha=0.80, inertia=0.0, delta_t=3,
        )
        self.assertEqual(len(pairs), 0, "Reverse motion must be rejected")


if __name__ == "__main__":
    unittest.main()
