import unittest
from unittest.mock import patch

import numpy as np

from multicam_tracker.config import PipelineConfig, apply_tracking_profile, DEFAULT_TRACKING_PROFILE
from multicam_tracker.fusion import GlobalTrackManager, CameraObservation
from multicam_tracker.identity_commit import IdentityCommitBuffer
from multicam_tracker.identity_lifecycle import IdentityLifecycle


def obs(local, ts, x=0.0, camera="a", appearance=None):
    return CameraObservation(camera, local, np.array([x, 0.]), np.zeros(4), .95,
                             capture_ts_s=ts, appearance=appearance)


def manager(**overrides):
    values = dict(cameras=["a", "b"], fusion_identity_lifecycle_enabled=True,
                  fusion_terminal_binding_enabled=True, fusion_terminal_batch_enabled=True,
                  fusion_deferred_gid_allocation_enabled=True, fusion_identity_commit_delay_s=3.,
                  fusion_late_identity_max_defer_s=2., fusion_terminal_grace_s=1.,
                  fusion_provisional_min_age_s=.3, fusion_min_hits=2,
                  fusion_trajectory_min_points=3, fusion_trajectory_min_duration_s=.2,
                  fusion_new_gid_entry_required=False, fusion_instant_merge_enabled=False,
                  fusion_duplicate_suppression_enabled=False, fusion_batch_reconnect_enabled=False,
                  fusion_local_reconnect_gate_m=0., fusion_terminal_confirmations=3,
                  tracklet_match_interval_s=100.)
    values.update(overrides)
    return GlobalTrackManager(PipelineConfig(**values))


class TerminalBindingTests(unittest.TestCase):
    def feed(self, m, local, times, x=0.):
        for ts in times:
            m.step([obs(local, ts, x)])

    def owner(self, m):
        self.feed(m, 1, [0., .1, .2, .3])
        gid = m.affinity[("a", 1)]
        self.assertGreater(gid, 0)
        return gid

    def test_profile_is_explicit_and_legacy_default_unchanged(self):
        cfg = apply_tracking_profile(PipelineConfig(), "hybridsort-crowd-recovery-v2")
        self.assertTrue(cfg.fusion_identity_lifecycle_enabled)
        self.assertFalse(cfg.fusion_terminal_binding_enabled)
        self.assertFalse(cfg.hybridsort_cascade_enabled)
        experimental = apply_tracking_profile(PipelineConfig(), "hybridsort-crowd-terminal-batch-experimental")
        self.assertTrue(experimental.fusion_terminal_batch_enabled)
        self.assertFalse(cfg.fusion_stationary_reconnect_enabled)
        self.assertEqual(cfg.conf_threshold, .25)
        self.assertEqual(DEFAULT_TRACKING_PROFILE, "hybridsort")

    def test_terminal_buffer_horizon_must_cover_decision(self):
        with self.assertRaises(ValueError):
            manager(fusion_identity_commit_delay_s=2.)
        with self.assertRaises(ValueError):
            manager(fusion_late_identity_max_defer_s=float("nan"))

    def test_long_stationary_gap_binds_retained_dormant_owner(self):
        m = manager(fusion_identity_retention_s=1., fusion_dormant_retention_s=12.)
        old = self.owner(m)
        m.step([], timestamp_s=5.)
        self.assertIn(old, m.dormant_tracks)
        self.feed(m, 2, [5.1, 5.2, 5.3, 5.4, 5.5, 5.6], x=.01)
        self.assertEqual(m.affinity[("a", 2)], old)
        self.assertEqual(m.metrics["public_gid_allocations"], 1)

    def test_valid_interior_birth_and_unique_outcome(self):
        m = manager()
        gid = self.owner(m)
        m.finalize_identities()
        report = m.lifecycle.report()
        self.assertTrue(report["conservation_ok"])
        self.assertEqual(report["unique_counts"]["PROMOTED_NEW_GID"], 1)
        self.assertEqual(report["records"][0]["public_gid"], gid)
        m.finalize_identities()
        self.assertEqual(len(m.lifecycle.completed), 1)

    def test_short_false_track_rejected_without_public_allocation(self):
        m = manager()
        m.step([obs(1, 0.)])
        m.step([], timestamp_s=3.1)
        self.assertEqual(m.metrics["public_gid_allocations"], 0)
        self.assertEqual(m.lifecycle.counts["REJECTED_FALSE_TRACK"], 1)
        self.assertFalse(m.lifecycle.pending)

    def test_disappeared_valid_tracklet_gets_terminal_promotion_and_backfill(self):
        m = manager(fusion_provisional_min_age_s=1.)
        buffer = IdentityCommitBuffer(3.)
        for i, ts in enumerate([0., .1, .2, .3]):
            m.step([obs(1, ts)])
            buffer.push_tick(dict(frame_index=i, capture_ts_s=ts,
                identity_hypotheses=m.identity_hypotheses_snapshot(),
                identity_aliases=dict(m.aliases), identity_decisions=m.lifecycle.drain(),
                identity_terminal_contract=True, identity_finalizer=m.finalize_identities))
        frames = buffer.flush()
        self.assertTrue(any(frame.predictions for frame in frames))
        self.assertEqual(m.lifecycle.counts["PROMOTED_NEW_GID"], 1)
        self.assertEqual(buffer.metrics["identity_commit_unresolved_hypotheses_dropped"], 0)

    def test_stationary_fragment_binds_same_gid_after_distinct_observations(self):
        m = manager()
        old = self.owner(m)
        self.feed(m, 2, [.5, .6, .7, .8, .9], x=.01)
        self.assertEqual(m.affinity[("a", 2)], old)
        self.assertEqual(m.metrics["public_gid_allocations"], 1)
        self.assertEqual(m.lifecycle.counts["BOUND_EXISTING_GID"], 1)

    def test_timeout_does_not_promote_when_owner_still_plausible(self):
        m = manager(fusion_terminal_confirmations=1000)
        self.owner(m)
        self.feed(m, 2, [.5, .6, .7, 1., 2., 3.5], x=.01)
        self.assertEqual(m.metrics["public_gid_allocations"], 1)
        self.assertEqual(m.lifecycle.counts["REJECTED_FALSE_TRACK"], 1)
        self.assertEqual(m.lifecycle.completed[-1]["decision_reason"], "ambiguous_owner_at_deadline")

    def test_same_observation_and_flush_cannot_add_confirmation(self):
        m = manager()
        self.owner(m)
        self.feed(m, 2, [.5, .6, .7], x=.01)
        votes = dict(m.terminal_binding.votes)
        m.terminal_binding.step()
        self.assertEqual(m.terminal_binding.votes, votes)
        m.finalize_identities()
        self.assertEqual(m.lifecycle.counts["BOUND_EXISTING_GID"], 0)

    def test_competing_equal_candidates_do_not_steal_owner(self):
        m = manager()
        old = self.owner(m)
        for ts in [.5, .6, .7, .8, .9, 1.]:
            m.step([obs(2, ts, -.01), obs(3, ts, .01)])
        self.assertNotEqual(m.affinity[("a", 2)], old)
        self.assertNotEqual(m.affinity[("a", 3)], old)
        self.assertEqual(m.metrics["public_gid_allocations"], 1)

    def test_simultaneous_same_camera_people_never_merge(self):
        m = manager()
        old = self.owner(m)
        for ts in [.5, .6, .7, .8, .9, 1.]:
            m.step([obs(1, ts), obs(2, ts, .01)])
        self.assertNotEqual(m.affinity[("a", 2)], old)

    def test_appearance_conflict_vetoes_same_camera_only(self):
        m = manager()
        old = self.owner(m)
        m.tracks[old].appearance_by_camera["a"] = np.array([1., 0.])
        for ts in [.5, .6, .7, .8, .9]:
            m.step([obs(2, ts, .01, appearance=np.array([0., 1.]))])
        self.assertNotEqual(m.affinity[("a", 2)], old)

    def test_private_merge_has_one_terminal_outcome_per_original(self):
        ledger = IdentityLifecycle(2)
        ledger.birth(-1, 0., ("a", 1))
        ledger.birth(-2, .1, ("b", 2))
        ledger.merge(-2, -1, .2, "private_merge")
        ledger.finish(-1, "PROMOTED_NEW_GID", 1., "verified", 7)
        ledger.finish(-1, "PROMOTED_NEW_GID", 1., "verified", 7)
        self.assertEqual(ledger.counts["PROMOTED_NEW_GID"], 2)
        self.assertTrue(ledger.report()["conservation_ok"])

    def test_missing_decision_raises_instead_of_hiding_drop(self):
        buffer = IdentityCommitBuffer(0.)
        with self.assertRaises(RuntimeError):
            buffer.push_tick(dict(frame_index=0, capture_ts_s=0.,
                                  identity_terminal_contract=True,
                                  identity_hypotheses={-1: np.zeros(2)}))

    def test_explicit_rejection_is_separate_from_unresolved(self):
        buffer = IdentityCommitBuffer(0.)
        frames = buffer.push_tick(dict(frame_index=0, capture_ts_s=0.,
            identity_terminal_contract=True, identity_hypotheses={-1: np.zeros(2)},
            identity_decisions={-1: {"outcome": "REJECTED_FALSE_TRACK", "reason": "short"}}))
        self.assertFalse(frames[0].predictions)
        self.assertEqual(buffer.metrics["identity_commit_intentionally_rejected_hypotheses"], 1)
        self.assertEqual(buffer.metrics["identity_commit_unresolved_hypotheses_dropped"], 0)


if __name__ == "__main__":
    unittest.main()
