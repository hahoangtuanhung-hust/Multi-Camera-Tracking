import unittest
import numpy as np
from multicam_tracker.config import PipelineConfig, build_effective_config
from multicam_tracker.hybridsort import HybridSortTracker


class CausalAssociationTests(unittest.TestCase):
    def tracker(self, **kwargs):
        cfg = dict(hybridsort_causal_association_enabled=True,
                   hybridsort_reappearance_enabled=True, track_min_hits=1,
                   track_detection_temporal_confirmation_frames=1,
                   track_max_age_s=.1, hybridsort_reappearance_max_gap_s=2.)
        cfg.update(kwargs)
        return HybridSortTracker(PipelineConfig(**cfg))

    def test_repeat_timestamp_cannot_confirm(self):
        t = self.tracker()
        box = np.array([10., 10., 30., 80., .95])
        t._create_track(box, 0., appearance=np.array([1., 0.]))
        track = t.tracks[0]
        self.assertEqual(t._record_reappearance_vote(track, track.appearance, .2, box[:4]), 1)
        self.assertEqual(t._record_reappearance_vote(track, track.appearance, .2, box[:4]), 1)
        self.assertEqual(t._record_reappearance_vote(track, track.appearance, .3, box[:4]), 2)

    def test_retained_track_without_descriptor_cannot_enter_normal_matching(self):
        t = self.tracker()
        box = np.array([[10., 10., 30., 80., .95]])
        t._create_track(box[0], 0., appearance=np.array([1., 0.]))
        t.update(np.empty((0, 5)), timestamp_s=.1)
        visible = t.update(box, timestamp_s=.5, frame=None)
        self.assertNotEqual(visible[0].track_id, 1)
        self.assertIn(1, [x.track_id for x in t.identity_tracks()])

    def test_hold_has_absolute_deadline_even_with_same_appearance(self):
        t = self.tracker(hybridsort_reappearance_confirmations=100,
                         track_association_nis_gate=1e9)
        box = np.array([[10., 10., 30., 80., .95]])
        t._create_track(box[0], 0., appearance=np.array([1., 0.]))
        t.tracks[0].misses = 1
        _, held, _ = t._associate_reappearances([0], [0], box, [np.array([1., 0.])], .2)
        self.assertEqual(held, {0})
        _, held, remaining = t._associate_reappearances([0], [0], box, [np.array([1., 0.])], .6)
        self.assertFalse(held)
        self.assertEqual(remaining, [0])

    def test_motion_veto_precedes_appearance(self):
        t = self.tracker(track_association_nis_gate=1.)
        t._create_track(np.array([10., 10., 30., 80., .95]), 0., appearance=np.array([1., 0.]))
        t.tracks[0].misses = 1
        pairs, held, _ = t._associate_reappearances([0], [0],
            np.array([[10000., 10000., 10020., 10070., .95]]), [np.array([1., 0.])], .2)
        self.assertFalse(pairs)
        self.assertFalse(held)

    def test_candidate_budget_does_not_drop_unexamined_detections(self):
        t = self.tracker(hybridsort_recovery_max_candidates=1)
        t._create_track(np.array([10., 10., 30., 80., .95]), 0., appearance=np.array([1., 0.]))
        t.tracks[0].misses = 1
        _, _, remaining = t._associate_reappearances([0], [0, 1],
            np.array([[10., 10., 30., 80., .95], [60., 10., 80., 80., .95]]), [None, None], .2)
        self.assertEqual(remaining, [0, 1])

    def test_profiles_keep_commit_delay_and_failed_features_off(self):
        ref = build_effective_config(profile="hybridsort-late-binding-balanced")
        cfg = build_effective_config(profile="hybridsort-causal-association-v1")
        self.assertEqual(cfg.fusion_identity_commit_delay_s, ref.fusion_identity_commit_delay_s)
        self.assertFalse(cfg.hybridsort_cascade_enabled)
        self.assertFalse(cfg.fusion_terminal_binding_enabled)
        self.assertFalse(cfg.fusion_stationary_reconnect_enabled)
        self.assertFalse(cfg.hybridsort_reappearance_enabled)
        self.assertFalse(cfg.fusion_reconnect_appearance_enabled)
        self.assertFalse(cfg.fusion_appearance_ownership_guard_enabled)

    def test_motion_only_profile_never_extracts_appearance(self):
        from unittest.mock import patch
        tracker = HybridSortTracker(build_effective_config(profile="hybridsort-causal-association-v1"))
        with patch.object(tracker.appearance_backend, "extract", side_effect=AssertionError("appearance forbidden")):
            for i in range(8):
                tracker.update(np.array([[10.+i, 10., 30.+i, 80., .95]]), timestamp_s=i*.1)

    def test_rescue_requires_history_and_exclusive_candidate(self):
        from collections import deque
        t = self.tracker()
        box = np.array([10., 10., 30., 80., .95])
        t._create_track(box, 0.)
        track = t.tracks[0]
        track.misses = 1
        track.observation_velocity_px_s = np.zeros(2)
        self.assertEqual(t._causal_motion_rescue([0], [0], box[None, :], .1), [])
        track.observations = deque([(i, box.copy()) for i in range(3)])
        self.assertEqual(t._causal_motion_rescue([0], [0], box[None, :], .1), [(0, 0)])
        self.assertEqual(t._causal_motion_rescue([0], [0, 1], np.stack([box, box]), .1), [])
        self.assertEqual(t._causal_motion_rescue([0], [0], box[None, :], 3.), [])

    def test_rescue_never_receives_already_matched_owner(self):
        from unittest.mock import patch
        cfg = build_effective_config(profile="hybridsort-causal-association-v1")
        cfg.track_min_hits = cfg.track_detection_temporal_confirmation_frames = 1
        t = HybridSortTracker(cfg)
        box = np.array([[10., 10., 30., 80., .95]])
        t.update(box, timestamp_s=0.)
        with patch.object(t, "_causal_motion_rescue", wraps=t._causal_motion_rescue) as rescue:
            t.update(box, timestamp_s=.1)
            self.assertEqual(rescue.call_args.args[0], [])
            self.assertEqual(rescue.call_args.args[1], [])


if __name__ == "__main__":
    unittest.main()
