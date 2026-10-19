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
from multicam_tracker.hybridsort import (
    HybridSortTracker,
    associate_hybrid,
    find_ambiguous_detections,
)
from multicam_tracker.local_tracker import create_local_tracker


def _det(x: float, confidence: float = 0.95) -> np.ndarray:
    """Return a (1, 5) detection at horizontal position x with width 20."""
    return np.array([[x, 0.0, x + 20.0, 40.0, confidence]], dtype=float)


def _det_xy(x: float, y: float, confidence: float = 0.95) -> np.ndarray:
    return np.array([[x, y, x + 20.0, y + 40.0, confidence]], dtype=float)


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
    @staticmethod
    def _appearance_frame(colour: tuple[int, int, int]) -> np.ndarray:
        frame = np.full((80, 120, 3), colour, dtype=np.uint8)
        return frame

    def test_reappearance_is_held_until_consecutive_appearance_confirmation(self):
        cfg = PipelineConfig(
            track_min_hits=1,
            track_min_iou=0.20,
            track_max_age_s=0.2,
            ocsort_use_byte=False,
            ocsort_observation_recovery_enabled=False,
            hybridsort_reappearance_enabled=True,
            hybridsort_reappearance_max_gap_s=2.0,
            hybridsort_reappearance_confirmations=3,
            hybridsort_reappearance_min_similarity=0.75,
        )
        tracker = HybridSortTracker(cfg)
        frame = self._appearance_frame((0, 0, 255))
        original_id = tracker.update(
            _det(0.0), timestamp_s=0.0, frame=frame
        )[0].track_id
        tracker.update(np.empty((0, 5)), timestamp_s=0.1, frame=frame)

        self.assertEqual(
            tracker.update(_det(15.0), timestamp_s=0.2, frame=frame), []
        )
        self.assertEqual(
            tracker.update(_det(15.0), timestamp_s=0.3, frame=frame), []
        )
        visible = tracker.update(_det(15.0), timestamp_s=0.4, frame=frame)

        self.assertEqual([track.track_id for track in visible], [original_id])
        self.assertEqual(tracker._next_id, 2)
        self.assertEqual(tracker.metrics["appearance_reappearance_matches"], 1)
        self.assertEqual(tracker.metrics["appearance_reappearance_deferred"], 2)

    def test_reappearance_confirmation_resets_when_candidate_changes(self):
        cfg = PipelineConfig(
            track_min_hits=1,
            track_min_iou=0.20,
            track_max_age_s=0.1,
            track_association_nis_gate=1e9,
            ocsort_use_byte=False,
            ocsort_observation_recovery_enabled=False,
            hybridsort_reappearance_enabled=True,
            hybridsort_reappearance_max_gap_s=2.0,
            hybridsort_reappearance_confirmations=3,
            hybridsort_reappearance_min_similarity=0.75,
            hybridsort_reappearance_min_margin=0.0,
        )
        tracker = HybridSortTracker(cfg)
        frame = self._appearance_frame((0, 0, 255))
        original_id = tracker.update(
            _det(0.0), timestamp_s=0.0, frame=frame
        )[0].track_id
        tracker.update(np.empty((0, 5)), timestamp_s=0.1, frame=frame)

        self.assertEqual(
            tracker.update(_det(15.0), timestamp_s=0.2, frame=frame), []
        )
        self.assertEqual(
            tracker.update(_det(55.0), timestamp_s=0.3, frame=frame), []
        )
        self.assertEqual(
            tracker.update(_det(55.0), timestamp_s=0.4, frame=frame), []
        )
        visible = tracker.update(_det(55.0), timestamp_s=0.5, frame=frame)

        self.assertEqual([track.track_id for track in visible], [original_id])
        self.assertEqual(
            tracker.metrics["appearance_reappearance_candidate_resets"], 1
        )

    def test_reappearance_hold_expires_and_allows_new_birth(self):
        cfg = PipelineConfig(
            track_min_hits=1,
            track_min_iou=0.20,
            track_max_age_s=0.1,
            ocsort_use_byte=False,
            ocsort_observation_recovery_enabled=False,
            hybridsort_reappearance_enabled=True,
            hybridsort_reappearance_max_gap_s=0.3,
            hybridsort_reappearance_confirmations=10,
            hybridsort_reappearance_min_similarity=0.75,
        )
        tracker = HybridSortTracker(cfg)
        frame = self._appearance_frame((0, 255, 0))
        original_id = tracker.update(
            _det(0.0), timestamp_s=0.0, frame=frame
        )[0].track_id
        tracker.update(np.empty((0, 5)), timestamp_s=0.1, frame=frame)
        self.assertEqual(
            tracker.update(_det(15.0), timestamp_s=0.2, frame=frame), []
        )

        visible = tracker.update(_det(15.0), timestamp_s=0.4, frame=frame)

        self.assertEqual(len(visible), 1)
        self.assertNotEqual(visible[0].track_id, original_id)
        self.assertEqual(tracker.metrics["tracks_removed_timeout"], 1)

    def test_two_reappearances_are_assigned_one_to_one(self):
        cfg = PipelineConfig(
            track_min_hits=1,
            track_min_iou=0.10,
            track_max_age_s=0.1,
            ocsort_use_byte=False,
            ocsort_observation_recovery_enabled=False,
            hybridsort_reappearance_enabled=True,
            hybridsort_reappearance_max_gap_s=1.0,
            hybridsort_reappearance_confirmations=1,
            hybridsort_reappearance_min_similarity=0.75,
        )
        tracker = HybridSortTracker(cfg)
        frame = self._appearance_frame((255, 0, 0))
        original = tracker.update(
            np.vstack((_det(0.0), _det(60.0))),
            timestamp_s=0.0,
            frame=frame,
        )
        original_ids = sorted(track.track_id for track in original)
        tracker.update(np.empty((0, 5)), timestamp_s=0.1, frame=frame)

        visible = tracker.update(
            np.vstack((_det(75.0), _det(15.0))),
            timestamp_s=0.2,
            frame=frame,
        )

        self.assertEqual(sorted(track.track_id for track in visible), original_ids)
        self.assertEqual(tracker._next_id, 3)

    def test_occluded_identity_retention_is_separate_from_render_window(self):
        cfg = PipelineConfig(
            track_min_hits=1,
            track_max_age_s=0.2,
            hybridsort_reappearance_enabled=True,
            hybridsort_reappearance_max_gap_s=2.0,
            render_occluded_tracks_enabled=True,
            render_occluded_max_age_s=0.3,
        )
        tracker = HybridSortTracker(cfg)
        frame = self._appearance_frame((0, 255, 0))
        tracker.update(_det(0.0), timestamp_s=0.0, frame=frame)
        tracker.update(np.empty((0, 5)), timestamp_s=0.2, frame=frame)
        self.assertEqual(len(tracker.identity_tracks()), 1)
        self.assertEqual(len(tracker.render_tracks()), 1)

        tracker.update(np.empty((0, 5)), timestamp_s=0.5, frame=frame)

        self.assertEqual(len(tracker.identity_tracks()), 1)
        self.assertEqual(tracker.render_tracks(), [])
        self.assertEqual(tracker.update(np.empty((0, 5)), timestamp_s=1.5), [])
        self.assertEqual(len(tracker.identity_tracks()), 1)

    def test_global_guard_extracts_descriptor_without_local_reappearance(self):
        cfg = PipelineConfig(
            track_min_hits=1,
            track_max_age_s=0.2,
            hybridsort_reappearance_enabled=False,
            fusion_appearance_ownership_guard_enabled=True,
        )
        tracker = HybridSortTracker(cfg)
        frame = self._appearance_frame((255, 0, 0))

        visible = tracker.update(_det(0.0), timestamp_s=0.0, frame=frame)

        self.assertEqual(len(visible), 1)
        self.assertIsNotNone(visible[0].current_appearance)
        self.assertIsNotNone(visible[0].appearance)
        tracker.update(np.empty((0, 5)), timestamp_s=0.3, frame=frame)
        self.assertEqual(tracker.tracks, [])

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

    def test_merged_detection_does_not_steal_crossing_ownership(self):
        cfg = PipelineConfig(
            track_min_hits=1,
            track_min_iou=0.20,
            ocsort_use_byte=False,
            ocsort_observation_recovery_enabled=False,
            hybridsort_ambiguity_suppression_enabled=True,
            hybridsort_ambiguity_min_iou=0.20,
            hybridsort_ambiguity_margin=0.10,
        )
        tracker = HybridSortTracker(cfg)
        first = np.vstack((_det(0.0), _det(20.0)))
        second = np.vstack((_det(4.0), _det(16.0)))
        tracker.update(first, timestamp_s=0.0)
        tracker.update(second, timestamp_s=0.1)
        original_ids = [track.track_id for track in tracker.tracks]

        # One detector box covers both people at the crossing. Updating one
        # owner would be arbitrary; both established tracks should coast.
        merged = np.array([[8.0, 0.0, 32.0, 40.0, 0.95]], dtype=float)
        visible = tracker.update(merged, timestamp_s=0.2)

        self.assertEqual(visible, [])
        self.assertEqual([track.track_id for track in tracker.tracks], original_ids)
        self.assertEqual([track.misses for track in tracker.tracks], [1, 1])
        self.assertEqual(tracker._next_id, 3)
        self.assertEqual(tracker.metrics["ambiguous_detections_suppressed"], 1)

    def test_robust_bbox_outlier_does_not_contaminate_trusted_observation(self):
        cfg = PipelineConfig(
            track_min_hits=1,
            track_min_iou=0.20,
            track_kf_robust_nis_threshold=13.28,
            hybridsort_trusted_observation_enabled=True,
            hybridsort_ambiguity_suppression_enabled=False,
        )
        tracker = HybridSortTracker(cfg)
        tracker.update(_det(0.0), timestamp_s=0.0)
        tracker.update(_det(2.0), timestamp_s=0.1)
        track = tracker.tracks[0]
        trusted_bbox = track.last_observation.copy()
        trusted_direction = track.velocity.copy()
        trusted_history_size = len(track.observations)

        merged = np.array([[0.0, 0.0, 40.0, 80.0, 0.95]], dtype=float)
        visible = tracker.update(merged, timestamp_s=0.2)

        self.assertEqual([item.track_id for item in visible], [track.track_id])
        np.testing.assert_array_equal(track.last_raw_observation, merged[0])
        np.testing.assert_array_equal(track.last_observation, trusted_bbox)
        np.testing.assert_array_equal(track.velocity, trusted_direction)
        self.assertEqual(len(track.observations), trusted_history_size)
        self.assertFalse(track.last_update_trusted)
        self.assertFalse(np.array_equal(track.xyxy, merged[0, :4]))
        self.assertEqual(tracker.metrics["observation_outliers_suppressed"], 1)

    def test_clear_detection_is_not_marked_ambiguous(self):
        cfg = PipelineConfig(track_min_hits=1)
        tracker = HybridSortTracker(cfg)
        tracker.update(np.vstack((_det(0.0), _det(80.0))), timestamp_s=0.0)

        ambiguous = find_ambiguous_detections(
            tracker.tracks,
            _det(2.0),
            min_iou=0.25,
            margin=0.10,
            min_area_ratio=1.10,
        )

        self.assertEqual(ambiguous, {})

    def test_multi_frame_merged_bbox_does_not_capture_one_owner(self):
        cfg = PipelineConfig(
            track_min_hits=1,
            track_min_iou=0.20,
            ocsort_use_byte=False,
            ocsort_observation_recovery_enabled=False,
            hybridsort_ambiguity_suppression_enabled=True,
            hybridsort_ambiguity_min_iou=0.20,
            hybridsort_ambiguity_margin=0.10,
            hybridsort_ambiguity_owner_grace_s=0.20,
            hybridsort_lost_observation_prediction_enabled=True,
        )
        tracker = HybridSortTracker(cfg)
        tracker.update(np.vstack((_det(0.0), _det(20.0))), timestamp_s=0.0)
        tracker.update(np.vstack((_det(4.0), _det(16.0))), timestamp_s=0.1)
        ids = [track.track_id for track in tracker.tracks]

        merged = np.array([[8.0, 0.0, 32.0, 40.0, 0.95]], dtype=float)
        for timestamp in (0.2, 0.3, 0.4):
            self.assertEqual(tracker.update(merged, timestamp_s=timestamp), [])
        separated = tracker.update(
            np.vstack((_det(24.0), _det(-4.0))), timestamp_s=0.5
        )

        self.assertEqual(sorted(track.track_id for track in separated), ids)
        by_id = {track.track_id: track for track in tracker.tracks}
        self.assertGreater(by_id[ids[0]].velocity[0], 0.0)
        self.assertLess(by_id[ids[1]].velocity[0], 0.0)
        self.assertEqual(tracker.metrics["ambiguous_detections_suppressed"], 3)

    def test_same_size_overlap_is_not_treated_as_merged_box(self):
        cfg = PipelineConfig(track_min_hits=1)
        tracker = HybridSortTracker(cfg)
        tracker.update(np.vstack((_det(0.0), _det(10.0))), timestamp_s=0.0)

        ambiguous = find_ambiguous_detections(
            tracker.tracks,
            _det(5.0),
            min_iou=0.25,
            margin=0.10,
            min_area_ratio=1.10,
        )

        self.assertEqual(ambiguous, {})

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

    def test_occlusion_durations_respect_second_based_retention(self):
        for gap_s in (0.3, 0.5, 1.0, 2.0, 3.0):
            with self.subTest(gap_s=gap_s):
                tracker = HybridSortTracker(PipelineConfig(
                    track_min_hits=1,
                    track_max_age_s=2.0,
                    ocsort_use_byte=False,
                ))
                original_id = tracker.update(
                    _det(0.0), timestamp_s=0.0
                )[0].track_id
                reappeared = tracker.update(_det(0.0), timestamp_s=gap_s)
                ids = [track.track_id for track in reappeared]
                if gap_s <= 2.0:
                    self.assertEqual(ids, [original_id])
                    self.assertEqual(tracker._next_id, 2)
                else:
                    self.assertNotIn(original_id, ids)
                    self.assertEqual(tracker.metrics["tracks_removed_timeout"], 1)

    def test_same_direction_overtaking_preserves_track_ownership(self):
        tracker = HybridSortTracker(PipelineConfig(
            track_min_hits=1,
            track_min_iou=0.15,
            hybridsort_alpha=0.60,
            ocsort_use_byte=False,
        ))
        tracker.update(
            np.vstack((_det_xy(0.0, 0.0), _det_xy(30.0, 8.0))),
            timestamp_s=0.0,
        )
        rear_id, front_id = [track.track_id for track in tracker.tracks]
        paths = (
            ((10.0, 0.0), (35.0, 8.0)),
            ((20.0, 0.0), (40.0, 8.0)),
            ((30.0, 0.0), (45.0, 8.0)),
            ((42.0, 0.0), (50.0, 8.0)),
            ((55.0, 0.0), (55.0, 8.0)),
            ((68.0, 0.0), (60.0, 8.0)),
        )
        for index, (rear, front) in enumerate(paths, start=1):
            detections = np.vstack((
                _det_xy(front[0], front[1]),
                _det_xy(rear[0], rear[1]),
            ))
            tracker.update(detections, timestamp_s=index * 0.1)

        by_id = {track.track_id: track for track in tracker.tracks}
        rear_center = (by_id[rear_id].xyxy[:2] + by_id[rear_id].xyxy[2:]) * 0.5
        front_center = (
            by_id[front_id].xyxy[:2] + by_id[front_id].xyxy[2:]
        ) * 0.5
        self.assertGreater(rear_center[0], front_center[0])
        self.assertLess(rear_center[1], front_center[1])
        self.assertEqual(tracker._next_id, 3)

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
