import unittest
from unittest.mock import patch

import numpy as np

from multicam_tracker.config import PipelineConfig
from multicam_tracker.fusion import (
    CameraObservation,
    ConstantVelocityKalman,
    GlobalTrack,
    GlobalTrackManager,
)


def _obs(
    camera: str,
    local_id: int,
    timestamp: float,
    xy: tuple[float, float],
    velocity: tuple[float, float] | None = None,
    near_boundary: bool = False,
):
    return CameraObservation(
        camera=camera,
        local_id=local_id,
        world_xy=np.asarray(xy, dtype=float),
        bbox_xyxy=np.zeros(4, dtype=float),
        conf=0.95,
        capture_ts_s=timestamp,
        world_velocity=(
            None if velocity is None else np.asarray(velocity, dtype=float)
        ),
        near_boundary=near_boundary,
    )


class EventTimeFusionTests(unittest.TestCase):
    def _config(self, **overrides) -> PipelineConfig:
        values = dict(
            cameras=["a", "b", "c"],
            fusion_min_hits=1,
            tracklet_history_s=6.0,
            tracklet_match_interval_s=0.1,
        )
        values.update(overrides)
        return PipelineConfig(**values)

    def test_provisional_ids_converge_to_published_canonical_id(self) -> None:
        manager = GlobalTrackManager(self._config())
        first_ids = None
        visible = []
        for event_time in np.arange(0.0, 2.01, 0.1):
            xy = (0.6 * event_time, 0.15 * event_time)
            visible = manager.step([
                _obs("a", 1, event_time, xy),
                _obs("b", 7, event_time + 2.0, xy),
            ])
            if first_ids is None:
                first_ids = (manager.affinity[("a", 1)], manager.affinity[("b", 7)])
        self.assertNotEqual(first_ids[0], first_ids[1])
        # Camera B is the only identity still fresh enough to be published
        # before delayed trajectory matching converges. Preserve that public
        # ID instead of visibly renaming it to the numerically smaller GID.
        self.assertEqual(manager.affinity[("a", 1)], first_ids[1])
        self.assertEqual(manager.affinity[("b", 7)], first_ids[1])
        self.assertEqual(manager.resolve_gid(first_ids[0]), first_ids[1])
        self.assertEqual(len({track.gid for track in visible}), 1)
        self.assertGreaterEqual(manager.metrics["global_id_merges"], 1)
        self.assertGreater(manager.clock_estimator.estimate("b").confidence, 0.55)

    def test_two_people_match_one_to_one_without_false_merge(self) -> None:
        manager = GlobalTrackManager(self._config())
        for event_time in np.arange(0.0, 2.01, 0.1):
            manager.step([
                _obs("a", 1, event_time, (event_time, 0.0)),
                _obs("a", 2, event_time, (event_time, 3.0)),
                _obs("b", 11, event_time + 1.5, (event_time, 0.0)),
                _obs("b", 12, event_time + 1.5, (event_time, 3.0)),
            ])
        first = manager.affinity[("a", 1)]
        second = manager.affinity[("a", 2)]
        self.assertEqual(first, manager.affinity[("b", 11)])
        self.assertEqual(second, manager.affinity[("b", 12)])
        self.assertNotEqual(first, second)

    def test_inconsistent_people_do_not_train_camera_clock(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a", "b"],
            tracklet_min_duration_s=0.5,
            clock_max_pair_delta_spread_s=0.2,
        ))
        # The two paths imply different clock shifts. This is characteristic
        # of spatial calibration bias or a wrong identity pairing, not one
        # physical camera clock. One early single-track bootstrap sample is
        # allowed, but inconsistent multi-person cycles must add no samples.
        for event_time in np.arange(0.0, 2.01, 0.1):
            manager.step([
                _obs("a", 1, event_time, (event_time, 0.0)),
                _obs("a", 2, event_time, (event_time, 3.0)),
                _obs("b", 11, event_time + 1.0, (event_time, 0.0)),
                _obs("b", 12, event_time + 2.0, (event_time, 3.0)),
            ])

        self.assertGreater(
            manager.metrics["clock_updates_rejected_delta_spread"], 0
        )
        self.assertLessEqual(manager.clock_estimator.estimate("b").samples, 1)

    def test_consistent_people_still_learn_two_second_clock_offset(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a", "b"],
            tracklet_min_duration_s=0.5,
            clock_max_pair_delta_spread_s=0.4,
        ))
        for event_time in np.arange(0.0, 2.01, 0.1):
            manager.step([
                _obs("a", 1, event_time, (event_time, 0.0)),
                _obs("a", 2, event_time, (event_time, 3.0)),
                _obs("b", 11, event_time + 2.0, (event_time, 0.0)),
                _obs("b", 12, event_time + 2.0, (event_time, 3.0)),
            ])

        estimate = manager.clock_estimator.estimate("b")
        self.assertGreater(estimate.confidence, 0.55)
        self.assertAlmostEqual(estimate.offset_s, -2.0, delta=0.25)
        self.assertEqual(
            manager.affinity[("a", 1)], manager.affinity[("b", 11)]
        )
        self.assertEqual(
            manager.affinity[("a", 2)], manager.affinity[("b", 12)]
        )

    def test_overlapping_same_camera_tracks_cannot_merge(self) -> None:
        manager = GlobalTrackManager(self._config())
        for timestamp in (0.0, 0.5, 1.0):
            manager.step([
                _obs("a", 1, timestamp, (timestamp, 0.0)),
                _obs("a", 2, timestamp, (timestamp, 0.1)),
            ])
        first = manager.affinity[("a", 1)]
        second = manager.affinity[("a", 2)]
        self.assertFalse(manager.merge_gids(first, second))
        self.assertNotEqual(manager.resolve_gid(first), manager.resolve_gid(second))

    def test_merge_is_idempotent_and_aliases_path_compress(self) -> None:
        manager = GlobalTrackManager(self._config())
        manager.step([
            _obs("a", 1, 0.0, (0.0, 0.0)),
            _obs("b", 1, 0.0, (5.0, 0.0)),
            _obs("c", 1, 0.0, (10.0, 0.0)),
        ])
        gids = [manager.affinity[(camera, 1)] for camera in ("a", "b", "c")]
        self.assertTrue(manager.merge_gids(gids[1], gids[2]))
        self.assertTrue(manager.merge_gids(gids[0], gids[1]))
        self.assertTrue(manager.merge_gids(gids[0], gids[2]))
        self.assertEqual(manager.resolve_gid(gids[2]), gids[0])
        self.assertEqual(manager.aliases[gids[2]], gids[0])

    def test_merge_preserves_the_id_already_published_to_consumers(self) -> None:
        manager = GlobalTrackManager(self._config())
        manager.step([
            _obs("a", 1, 0.0, (0.0, 0.0)),
            _obs("b", 1, 0.0, (5.0, 0.0)),
        ])
        first = manager.affinity[("a", 1)]
        second = manager.affinity[("b", 1)]
        manager.tracks[first].emitted = False
        manager.tracks[second].emitted = True

        self.assertTrue(manager.merge_gids(first, second))
        self.assertEqual(manager.resolve_gid(first), second)
        self.assertEqual(manager.affinity[("a", 1)], second)
        self.assertEqual(manager.metrics["published_id_preserved_on_merge"], 1)

    def test_late_packet_is_kept_for_matching_but_does_not_rewind_kf(self) -> None:
        manager = GlobalTrackManager(self._config())
        for timestamp in (0.0, 1.0, 2.0):
            manager.step([_obs("a", 1, timestamp, (timestamp, 0.0))])
        gid = manager.affinity[("a", 1)]
        state_time = manager.tracks[gid].state_timestamp_s
        manager.step([_obs("a", 1, 1.5, (1.5, 0.0))], timestamp_s=2.0)
        self.assertEqual(manager.tracks[gid].state_timestamp_s, state_time)
        self.assertEqual(manager.metrics["out_of_sequence_kf_updates_skipped"], 1)
        self.assertEqual(manager.tracklets.metrics["out_of_order_accepted"], 1)

    def test_packet_older_than_late_window_is_dropped(self) -> None:
        manager = GlobalTrackManager(self._config(tracklet_late_drop_s=3.0))
        manager.step([_obs("a", 1, 10.0, (0.0, 0.0))])
        manager.step([_obs("a", 1, 6.0, (0.0, 0.0))], timestamp_s=10.0)
        self.assertEqual(manager.metrics["late_observations_dropped"], 1)

    def test_stale_track_is_retained_for_identity_but_not_published(self) -> None:
        manager = GlobalTrackManager(self._config(fusion_output_grace_s=0.1))
        observation = _obs("a", 1, 1.0, (0.0, 0.0))
        observation.arrival_ts_s = 10.0
        self.assertEqual(len(manager.step(
            [observation], timestamp_s=1.0, arrival_timestamp_s=10.0
        )), 1)
        visible = manager.step([], timestamp_s=1.2, arrival_timestamp_s=10.2)
        self.assertEqual(visible, [])
        self.assertIn(manager.affinity[("a", 1)], manager.tracks)

    def test_confident_clock_enables_instant_geometry_for_short_tracks(self) -> None:
        manager = GlobalTrackManager(self._config())
        for timestamp in range(5):
            manager.clock_estimator.observe("b", timestamp + 2.0, -2.0, 0.95)
        manager.step([
            _obs("a", 1, 10.0, (1.0, 2.0)),
            _obs("b", 7, 12.0, (1.1, 2.0)),
        ])
        self.assertEqual(manager.affinity[("a", 1)], manager.affinity[("b", 7)])
        self.assertEqual(manager.metrics["instant_geometry_merges"], 1)

    def test_crossing_defers_opposite_motion_then_merges_after_separation(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a", "b"],
            fusion_crossing_guard_m=0.0,
            fusion_instant_opposite_direction_cosine=-0.25,
            fusion_instant_margin_m=0.1,
            fusion_published_merge_confirmations=2,
            fusion_instant_velocity_gate_mps=10.0,
            tracklet_min_duration_s=10.0,
        ))
        # Projection noise puts each camera-B observation closer to the wrong
        # person at the crossing. Their velocity directions remain distinct.
        for timestamp, shift in ((0.0, 0.0), (0.1, 0.1)):
            manager.step([
                _obs("a", 1, timestamp, (-0.6 + shift, 0.0), (1.0, 0.0)),
                _obs("a", 2, timestamp, (0.6 - shift, 0.0), (-1.0, 0.0)),
                _obs("b", 11, timestamp, (0.1 + shift, 0.0), (1.0, 0.0)),
                _obs("b", 12, timestamp, (-0.1 - shift, 0.0), (-1.0, 0.0)),
            ])

        self.assertNotEqual(manager.affinity[("a", 1)], manager.affinity[("b", 11)])
        self.assertGreater(manager.metrics["instant_opposite_direction_vetoes"], 0)

        manager.step([
            _obs("a", 1, 0.2, (0.2, 0.0), (1.0, 0.0)),
            _obs("a", 2, 0.2, (-0.2, 0.0), (-1.0, 0.0)),
            _obs("b", 11, 0.2, (0.25, 0.0), (1.0, 0.0)),
            _obs("b", 12, 0.2, (-0.25, 0.0), (-1.0, 0.0)),
        ])
        manager.step([
            _obs("a", 1, 0.3, (0.3, 0.0), (1.0, 0.0)),
            _obs("a", 2, 0.3, (-0.3, 0.0), (-1.0, 0.0)),
            _obs("b", 11, 0.3, (0.35, 0.0), (1.0, 0.0)),
            _obs("b", 12, 0.3, (-0.35, 0.0), (-1.0, 0.0)),
        ])
        self.assertEqual(manager.affinity[("a", 1)], manager.affinity[("b", 11)])
        self.assertEqual(manager.affinity[("a", 2)], manager.affinity[("b", 12)])
        self.assertNotEqual(manager.affinity[("a", 1)], manager.affinity[("a", 2)])

    def test_crossing_ignores_one_frame_velocity_reversal(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a", "b"],
            fusion_crossing_guard_m=0.0,
            fusion_instant_opposite_direction_cosine=-0.25,
            fusion_instant_margin_m=0.1,
            fusion_instant_merge_enabled=False,
            tracklet_min_duration_s=10.0,
        ))
        for timestamp, shift in ((0.0, 0.0), (0.2, 0.2)):
            manager.step([
                _obs("a", 1, timestamp, (-0.6 + shift, 0.0), (1.0, 0.0)),
                _obs("a", 2, timestamp, (0.6 - shift, 0.0), (-1.0, 0.0)),
                _obs("b", 11, timestamp, (-0.6 + shift, 0.0), (1.0, 0.0)),
                _obs("b", 12, timestamp, (0.6 - shift, 0.0), (-1.0, 0.0)),
            ])
        manager.cfg.fusion_instant_merge_enabled = True
        # Camera-B position noise favours the wrong spatial pairing and its
        # supplied one-frame velocities are reversed. The 0.4 s path history
        # still has the correct direction.
        manager.step([
            _obs("a", 1, 0.4, (-0.2, 0.0), (-1.0, 0.0)),
            _obs("a", 2, 0.4, (0.2, 0.0), (1.0, 0.0)),
            _obs("b", 11, 0.4, (0.1, 0.0), (-1.0, 0.0)),
            _obs("b", 12, 0.4, (-0.1, 0.0), (1.0, 0.0)),
        ])
        self.assertNotEqual(manager.affinity[("a", 1)], manager.affinity[("b", 11)])
        manager.step([
            _obs("a", 1, 0.6, (0.2, 0.0), (1.0, 0.0)),
            _obs("a", 2, 0.6, (-0.2, 0.0), (-1.0, 0.0)),
            _obs("b", 11, 0.6, (0.25, 0.0), (1.0, 0.0)),
            _obs("b", 12, 0.6, (-0.25, 0.0), (-1.0, 0.0)),
        ])
        manager.step([
            _obs("a", 1, 0.8, (0.4, 0.0), (1.0, 0.0)),
            _obs("a", 2, 0.8, (-0.4, 0.0), (-1.0, 0.0)),
            _obs("b", 11, 0.8, (0.45, 0.0), (1.0, 0.0)),
            _obs("b", 12, 0.8, (-0.45, 0.0), (-1.0, 0.0)),
        ])

        self.assertEqual(manager.affinity[("a", 1)], manager.affinity[("b", 11)])
        self.assertEqual(manager.affinity[("a", 2)], manager.affinity[("b", 12)])
        self.assertGreater(manager.metrics["stable_motion_estimates_used"], 0)

    def test_ambiguous_merge_can_require_repeated_confirmation(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a", "b"],
            fusion_crossing_guard_m=0.75,
            fusion_instant_confirmations=2,
            tracklet_min_duration_s=10.0,
        ))
        manager.step([
            _obs("a", 1, 0.0, (-0.2, 0.0), (0.0, 0.0)),
            _obs("a", 2, 0.0, (0.2, 0.0), (0.0, 0.0)),
            _obs("b", 11, 0.0, (0.15, 0.0), (0.0, 0.0)),
            _obs("b", 12, 0.0, (-0.15, 0.0), (0.0, 0.0)),
        ])

        self.assertNotEqual(manager.affinity[("a", 1)], manager.affinity[("b", 11)])
        self.assertNotEqual(manager.affinity[("a", 2)], manager.affinity[("b", 12)])
        self.assertEqual(manager.metrics["instant_geometry_merges"], 0)
        self.assertGreater(
            manager.metrics["instant_merges_deferred_for_confirmation"], 0
        )

    def test_fragmented_local_id_reconnects_to_retained_same_camera_track(self) -> None:
        manager = GlobalTrackManager(self._config())
        for timestamp in (0.0, 0.2, 0.4):
            manager.step([_obs("a", 1, timestamp, (timestamp, 0.0))])
        original_gid = manager.affinity[("a", 1)]
        manager.step([_obs("a", 2, 0.6, (0.6, 0.0))])
        self.assertEqual(manager.affinity[("a", 2)], original_gid)
        self.assertEqual(manager.metrics["local_fragments_reconnected"], 1)
        self.assertEqual(
            manager.metrics["provisional_ids_reconnected_before_publish"], 1
        )

    def test_published_owner_requires_consecutive_fragment_evidence(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a"],
            fusion_provisional_buffer_enabled=True,
            fusion_provisional_min_age_s=0.5,
            fusion_provisional_buffer_bootstrap_s=0.1,
            fusion_gid_ownership_hysteresis_enabled=True,
            fusion_local_reconnect_confirmations=3,
            fusion_local_reconnect_confirmation_window_s=0.35,
            fusion_instant_merge_enabled=False,
            tracklet_min_duration_s=10.0,
        ))
        visible = manager.step([
            _obs("a", 1, 0.0, (0.0, 0.0), (1.0, 0.0))
        ])
        original_gid = manager.affinity[("a", 1)]
        self.assertEqual([track.gid for track in visible], [original_gid])

        # A post-bootstrap local fragment stays internal while ownership is
        # still only a one/two-frame hypothesis.
        visible = manager.step([
            _obs("a", 2, 0.2, (0.2, 0.0), (1.0, 0.0))
        ])
        self.assertEqual([track.gid for track in visible], [original_gid])
        provisional_gid = manager.affinity[("a", 2)]
        self.assertNotEqual(provisional_gid, original_gid)
        self.assertNotIn(("a", 2), manager.public_affinity_snapshot())
        self.assertEqual(manager.step([
            _obs("a", 2, 0.3, (0.3, 0.0), (1.0, 0.0))
        ]), [])

        visible = manager.step([
            _obs("a", 2, 0.4, (0.4, 0.0), (1.0, 0.0))
        ])
        self.assertEqual(manager.affinity[("a", 2)], original_gid)
        self.assertEqual(
            manager.public_affinity_snapshot()[("a", 2)], original_gid
        )
        self.assertEqual([track.gid for track in visible], [original_gid])
        self.assertGreaterEqual(
            manager.metrics["local_reconnects_deferred_for_confirmation"], 2
        )

    def test_fragment_challenger_must_be_consecutive(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a"],
            fusion_min_hits=1,
            fusion_provisional_buffer_enabled=True,
            fusion_provisional_min_age_s=0.5,
            fusion_provisional_buffer_bootstrap_s=0.1,
            fusion_gid_ownership_hysteresis_enabled=True,
            fusion_local_reconnect_confirmations=2,
            fusion_local_reconnect_confirmation_window_s=0.35,
            fusion_instant_merge_enabled=False,
            tracklet_min_duration_s=10.0,
        ))
        manager.step([_obs("a", 1, 0.0, (0.0, 0.0))])
        original_gid = manager.affinity[("a", 1)]

        manager.step([_obs("a", 2, 0.2, (0.2, 0.0))])
        self.assertNotEqual(manager.affinity[("a", 2)], original_gid)
        # No compatible owner on the next update clears the pending vote.
        manager.step([_obs("a", 2, 0.3, (5.0, 0.0))])
        self.assertNotIn(("a", 2), manager._fragment_reconnect_votes)
        # Returning close to the old owner starts at one vote again.
        manager.step([_obs("a", 2, 0.4, (0.4, 0.0))])
        self.assertNotEqual(manager.affinity[("a", 2)], original_gid)

    def test_crossing_fragments_reconnect_by_motion_not_nearest_position(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a"],
            fusion_local_reconnect_velocity_weight_s=0.25,
        ))
        manager.step([
            _obs("a", 1, 0.0, (-0.6, 0.0), (1.0, 0.0)),
            _obs("a", 2, 0.0, (0.6, 0.0), (-1.0, 0.0)),
        ])
        manager.step([
            _obs("a", 1, 0.2, (-0.4, 0.0), (1.0, 0.0)),
            _obs("a", 2, 0.2, (0.4, 0.0), (-1.0, 0.0)),
        ])
        first_gid = manager.affinity[("a", 1)]
        second_gid = manager.affinity[("a", 2)]

        # New local IDs appear around the crossing. Position alone favours
        # the wrong retained GID; direction continuity favours the right one.
        manager.step([
            _obs("a", 3, 0.4, (0.1, 0.0), (1.0, 0.0)),
            _obs("a", 4, 0.4, (-0.1, 0.0), (-1.0, 0.0)),
        ])
        manager.step([
            _obs("a", 3, 0.6, (0.3, 0.0), (1.0, 0.0)),
            _obs("a", 4, 0.6, (-0.3, 0.0), (-1.0, 0.0)),
        ])

        self.assertEqual(manager.affinity[("a", 3)], first_gid)
        self.assertEqual(manager.affinity[("a", 4)], second_gid)
        self.assertGreater(manager.metrics["local_reconnect_motion_penalties"], 0)

    def test_long_fragment_requires_boundary_and_motion_evidence(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a"],
            fusion_local_reconnect_confirmations=2,
            fusion_identity_retention_s=8.0,
            fusion_local_reconnect_window_s=1.0,
            fusion_local_reconnect_long_window_s=8.0,
        ))
        manager.step([
            _obs("a", 1, 0.0, (0.0, 0.0), (1.0, 0.0), near_boundary=True)
        ])
        manager.step([
            _obs("a", 1, 0.2, (0.2, 0.0), (1.0, 0.0), near_boundary=True)
        ])
        original_gid = manager.affinity[("a", 1)]
        manager.step([
            _obs("a", 2, 4.0, (4.0, 0.0), (1.0, 0.0), near_boundary=True)
        ])
        manager.step([
            _obs("a", 2, 4.2, (4.2, 0.0), (1.0, 0.0), near_boundary=True)
        ])

        self.assertEqual(manager.affinity[("a", 2)], original_gid)
        self.assertEqual(manager.metrics["local_fragments_reconnected"], 1)

    def test_long_fragment_inside_scene_gets_a_new_gid(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a"],
            fusion_identity_retention_s=8.0,
            fusion_local_reconnect_window_s=1.0,
            fusion_local_reconnect_long_window_s=8.0,
        ))
        manager.step([_obs("a", 1, 0.0, (0.0, 0.0), (1.0, 0.0))])
        manager.step([_obs("a", 1, 0.2, (0.2, 0.0), (1.0, 0.0))])
        original_gid = manager.affinity[("a", 1)]
        manager.step([_obs("a", 2, 4.0, (4.0, 0.0), (1.0, 0.0))])
        manager.step([_obs("a", 2, 4.2, (4.2, 0.0), (1.0, 0.0))])

        self.assertNotEqual(manager.affinity[("a", 2)], original_gid)
        self.assertGreater(manager.metrics["long_reconnect_boundary_rejected"], 0)

    def test_duplicate_globals_merge_only_after_consecutive_evidence(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a", "b"],
            fusion_instant_merge_enabled=False,
            fusion_duplicate_merge_m=0.30,
            fusion_duplicate_confirmations=3,
            tracklet_min_duration_s=10.0,
        ))
        for timestamp in (0.0, 0.1):
            manager.step([
                _obs("a", 1, timestamp, (timestamp, 0.0), (1.0, 0.0)),
                _obs("b", 7, timestamp, (timestamp + 0.05, 0.0), (1.0, 0.0)),
            ])
            self.assertNotEqual(
                manager.affinity[("a", 1)], manager.affinity[("b", 7)]
            )
        manager.step([
            _obs("a", 1, 0.2, (0.2, 0.0), (1.0, 0.0)),
            _obs("b", 7, 0.2, (0.25, 0.0), (1.0, 0.0)),
        ])

        self.assertEqual(manager.affinity[("a", 1)], manager.affinity[("b", 7)])
        self.assertEqual(manager.metrics["duplicate_global_tracks_merged"], 1)

    def test_competing_gid_vote_resets_ownership_hysteresis(self) -> None:
        votes: dict[tuple[int, int], tuple[int, float]] = {}
        self.assertEqual(
            GlobalTrackManager._record_pair_vote(votes, (1, 2), 0.0, 0.5),
            1,
        )
        self.assertEqual(
            GlobalTrackManager._record_pair_vote(votes, (1, 3), 0.1, 0.5),
            1,
        )
        self.assertEqual(
            GlobalTrackManager._record_pair_vote(votes, (1, 2), 0.2, 0.5),
            1,
        )
        self.assertNotIn((1, 3), votes)

    def test_simultaneous_same_camera_tracks_do_not_reconnect(self) -> None:
        manager = GlobalTrackManager(self._config())
        manager.step([_obs("a", 1, 0.0, (0.0, 0.0))])
        manager.step([
            _obs("a", 1, 0.2, (0.2, 0.0)),
            _obs("a", 2, 0.2, (0.25, 0.0)),
        ])
        self.assertNotEqual(manager.affinity[("a", 1)], manager.affinity[("a", 2)])

    def test_batch_fragment_assignment_is_independent_of_observation_order(self) -> None:
        def run(new_observations):
            manager = GlobalTrackManager(self._config(
                cameras=["a"],
                fusion_batch_reconnect_enabled=True,
                fusion_local_reconnect_velocity_weight_s=0.25,
            ))
            manager.step([
                _obs("a", 1, 0.0, (-0.6, 0.0), (1.0, 0.0)),
                _obs("a", 2, 0.0, (0.6, 0.0), (-1.0, 0.0)),
            ])
            manager.step([
                _obs("a", 1, 0.2, (-0.4, 0.0), (1.0, 0.0)),
                _obs("a", 2, 0.2, (0.4, 0.0), (-1.0, 0.0)),
            ])
            expected = (
                manager.affinity[("a", 1)],
                manager.affinity[("a", 2)],
            )
            manager.step(new_observations)
            manager.step([
                _obs("a", 3, 0.6, (0.3, 0.0), (1.0, 0.0)),
                _obs("a", 4, 0.6, (-0.3, 0.0), (-1.0, 0.0)),
            ])
            return expected, (
                manager.affinity[("a", 3)],
                manager.affinity[("a", 4)],
            )

        forward = [
            _obs("a", 3, 0.4, (0.1, 0.0), (1.0, 0.0)),
            _obs("a", 4, 0.4, (-0.1, 0.0), (-1.0, 0.0)),
        ]
        expected_forward, assigned_forward = run(forward)
        expected_reverse, assigned_reverse = run(list(reversed(forward)))
        self.assertEqual(assigned_forward, expected_forward)
        self.assertEqual(assigned_reverse, expected_reverse)

    def test_provisional_age_buffer_hides_short_false_track(self) -> None:
        cfg = self._config(
            cameras=["a"],
            fusion_min_hits=1,
            fusion_provisional_buffer_enabled=True,
            fusion_provisional_min_age_s=0.2,
            fusion_instant_merge_enabled=False,
        )
        manager = GlobalTrackManager(cfg)
        self.assertEqual(manager.step([_obs("a", 1, 0.0, (0.0, 0.0))]), [])
        self.assertEqual(manager.step([_obs("a", 1, 0.1, (0.1, 0.0))]), [])
        visible = manager.step([_obs("a", 1, 0.2, (0.2, 0.0))])
        self.assertEqual(len(visible), 1)
        self.assertEqual(manager.metrics["provisional_ids_published"], 1)

        false_track = GlobalTrackManager(cfg)
        self.assertEqual(
            false_track.step([_obs("a", 9, 0.0, (0.0, 0.0))]), []
        )
        self.assertEqual(
            false_track.step([], timestamp_s=0.4, arrival_timestamp_s=0.4), []
        )
        self.assertEqual(false_track.metrics["provisional_ids_published"], 0)

    def test_deferred_allocation_does_not_expose_gid_before_decision(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a"],
            fusion_min_hits=1,
            fusion_provisional_buffer_enabled=True,
            fusion_provisional_min_age_s=0.75,
            fusion_deferred_gid_allocation_enabled=True,
            fusion_instant_merge_enabled=False,
        ))
        self.assertEqual(manager.step([_obs("a", 1, 0.0, (0.0, 0.0))]), [])
        provisional = manager.affinity[("a", 1)]
        self.assertLess(provisional, 0)
        self.assertEqual(manager.public_affinity_snapshot(), {})
        self.assertEqual(manager.step([_obs("a", 1, 0.5, (0.5, 0.0))]), [])

        visible = manager.step([_obs("a", 1, 0.75, (0.75, 0.0))])
        self.assertEqual([track.gid for track in visible], [1])
        self.assertEqual(manager.resolve_gid(provisional), 1)
        self.assertEqual(manager.public_affinity_snapshot()[("a", 1)], 1)
        self.assertEqual(manager.metrics["public_gid_allocations"], 1)

    def test_fragments_reconnect_inside_buffer_before_one_public_gid_is_allocated(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a"],
            fusion_min_hits=1,
            fusion_provisional_buffer_enabled=True,
            fusion_provisional_min_age_s=0.75,
            fusion_deferred_gid_allocation_enabled=True,
            fusion_instant_merge_enabled=False,
        ))
        manager.step([_obs("a", 1, 0.0, (0.0, 0.0), (1.0, 0.0))])
        first_provisional = manager.affinity[("a", 1)]
        manager.step([_obs("a", 2, 0.2, (0.2, 0.0), (1.0, 0.0))])
        self.assertEqual(manager.affinity[("a", 2)], first_provisional)
        manager.step([_obs("a", 2, 0.8, (0.8, 0.0), (1.0, 0.0))])

        self.assertEqual(manager.metrics["provisional_ids_created"], 2)
        self.assertEqual(manager.metrics["public_gid_allocations"], 1)
        self.assertEqual(manager.affinity[("a", 1)], 1)
        self.assertEqual(manager.affinity[("a", 2)], 1)

    def test_new_public_gid_requires_entry_after_bootstrap(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a"],
            fusion_min_hits=1,
            fusion_provisional_buffer_enabled=True,
            fusion_provisional_min_age_s=0.5,
            fusion_deferred_gid_allocation_enabled=True,
            fusion_new_gid_entry_required=True,
            fusion_new_gid_bootstrap_s=0.2,
            fusion_local_reconnect_gate_m=0.1,
            fusion_instant_merge_enabled=False,
        ))
        manager.step([], timestamp_s=0.0, arrival_timestamp_s=0.0)
        manager.step([_obs("a", 1, 0.5, (0.0, 0.0))])
        manager.step([_obs("a", 1, 1.1, (0.0, 0.0))])
        self.assertLess(manager.affinity[("a", 1)], 0)
        self.assertEqual(manager.public_affinity_snapshot(), {})

        manager.step([_obs("a", 2, 1.2, (10.0, 0.0), near_boundary=True)])
        visible = manager.step([
            _obs("a", 2, 1.8, (10.0, 0.0), near_boundary=True)
        ])
        self.assertEqual(len(visible), 1)
        self.assertGreater(visible[0].gid, 0)
        self.assertGreater(manager.metrics["new_gid_entry_deferred"], 0)

    def test_fixed_population_capacity_keeps_excess_identity_pending(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a"],
            fusion_min_hits=1,
            fusion_provisional_buffer_enabled=True,
            fusion_provisional_min_age_s=0.5,
            fusion_deferred_gid_allocation_enabled=True,
            fusion_fixed_population_size=2,
            fusion_new_gid_bootstrap_s=2.0,
            fusion_instant_merge_enabled=False,
        ))
        observations = [
            _obs("a", 1, 0.0, (0.0, 0.0)),
            _obs("a", 2, 0.0, (5.0, 0.0)),
            _obs("a", 3, 0.0, (10.0, 0.0)),
        ]
        manager.step(observations)
        visible = manager.step([
            _obs("a", 1, 0.5, (0.0, 0.0)),
            _obs("a", 2, 0.5, (5.0, 0.0)),
            _obs("a", 3, 0.5, (10.0, 0.0)),
        ])
        self.assertEqual(len(visible), 2)
        self.assertEqual(manager.metrics["public_identity_slots"], 2)
        self.assertEqual(manager.metrics["pending_provisional_tracks"], 1)
        self.assertGreater(manager.metrics["new_gid_capacity_deferred"], 0)

    def test_confirmed_exit_frees_capacity_for_exactly_one_new_gid(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a"],
            tracklet_history_s=0.1,
            tracklet_late_drop_s=0.1,
            fusion_max_age_s=0.2,
            fusion_identity_retention_s=0.2,
            fusion_min_hits=1,
            fusion_provisional_buffer_enabled=True,
            fusion_provisional_min_age_s=0.1,
            fusion_deferred_gid_allocation_enabled=True,
            fusion_fixed_population_size=1,
            fusion_new_gid_bootstrap_s=2.0,
            fusion_retire_requires_exit_boundary=True,
            fusion_forced_retire_s=0.0,
            fusion_local_reconnect_gate_m=0.1,
            fusion_instant_merge_enabled=False,
        ))
        manager.step([_obs("a", 1, 0.0, (0.0, 0.0), near_boundary=True)])
        manager.step([_obs("a", 1, 0.1, (0.1, 0.0), near_boundary=True)])
        self.assertEqual(manager.metrics["public_gid_allocations"], 1)

        manager.step([_obs("a", 2, 0.2, (10.0, 0.0), near_boundary=True)])
        self.assertLess(manager.affinity[("a", 2)], 0)
        manager.step([_obs("a", 2, 0.4, (10.1, 0.0), near_boundary=True)])

        self.assertEqual(manager.metrics["confirmed_identity_exits"], 1)
        self.assertEqual(manager.metrics["public_gid_allocations"], 2)
        self.assertEqual(manager.affinity[("a", 2)], 2)
        self.assertEqual(manager.metrics["public_identity_slots"], 1)

    def test_public_identity_retires_only_after_boundary_exit_evidence(self) -> None:
        cfg = self._config(
            cameras=["a"],
            tracklet_history_s=0.1,
            tracklet_late_drop_s=0.1,
            fusion_max_age_s=0.2,
            fusion_identity_retention_s=0.2,
            fusion_min_hits=1,
            fusion_provisional_buffer_enabled=True,
            fusion_provisional_min_age_s=0.1,
            fusion_deferred_gid_allocation_enabled=True,
            fusion_new_gid_bootstrap_s=1.0,
            fusion_retire_requires_exit_boundary=True,
            fusion_forced_retire_s=0.0,
            fusion_instant_merge_enabled=False,
        )
        manager = GlobalTrackManager(cfg)
        manager.step([_obs("a", 1, 0.0, (0.0, 0.0))])
        manager.step([_obs("a", 1, 0.1, (0.1, 0.0))])
        gid = manager.affinity[("a", 1)]
        manager.step([], timestamp_s=0.5, arrival_timestamp_s=0.5)
        self.assertIn(gid, manager.tracks)
        self.assertEqual(manager.metrics["retirement_deferred_no_exit"], 1)

        exiting = GlobalTrackManager(cfg)
        exiting.step([
            _obs("a", 1, 0.0, (0.0, 0.0), near_boundary=True)
        ])
        exiting.step([
            _obs("a", 1, 0.1, (0.1, 0.0), near_boundary=True)
        ])
        exiting_gid = exiting.affinity[("a", 1)]
        exiting.step([], timestamp_s=0.5, arrival_timestamp_s=0.5)
        self.assertNotIn(exiting_gid, exiting.tracks)
        self.assertEqual(exiting.metrics["confirmed_identity_exits"], 1)

    def test_clear_published_pair_uses_confirmation_when_motion_is_noisy(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a", "b"],
            fusion_min_hits=1,
            fusion_adaptive_hysteresis_enabled=True,
            fusion_published_motion_gate_enabled=True,
            fusion_published_merge_confirmations=1,
            fusion_motion_min_span_s=10.0,
            tracklet_min_duration_s=10.0,
            fusion_duplicate_merge_m=0.0,
            # This test deliberately injects a 3.95 m / 0.1 s noisy sample to
            # isolate published-pair hysteresis, not physical feasibility.
            fusion_max_jump_m=10.0,
            fusion_max_speed_mps=50.0,
        ))
        manager.step([
            _obs("a", 1, 0.0, (0.0, 0.0), (1.0, 0.0)),
            _obs("b", 7, 0.0, (5.0, 0.0), (-1.0, 0.0)),
        ])
        manager.step([
            _obs("a", 1, 0.1, (1.0, 0.0), (1.0, 0.0)),
            _obs("b", 7, 0.1, (1.05, 0.0), (-1.0, 0.0)),
        ])
        self.assertNotEqual(
            manager.affinity[("a", 1)], manager.affinity[("b", 7)]
        )
        manager.step([
            _obs("a", 1, 0.2, (2.0, 0.0), (1.0, 0.0)),
            _obs("b", 7, 0.2, (2.05, 0.0), (-1.0, 0.0)),
        ])
        self.assertEqual(
            manager.affinity[("a", 1)], manager.affinity[("b", 7)]
        )
        self.assertGreaterEqual(manager.metrics["published_merge_motion_softened"], 2)
        self.assertEqual(manager.metrics["published_merge_motion_rejected"], 0)

    def test_duplicate_output_is_suppressed_before_irreversible_merge(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a", "b"],
            fusion_min_hits=1,
            fusion_instant_merge_enabled=False,
            fusion_duplicate_merge_m=0.30,
            fusion_duplicate_confirmations=3,
            fusion_duplicate_suppression_enabled=True,
            fusion_duplicate_suppression_min_confirmations=2,
            tracklet_min_duration_s=10.0,
        ))
        first = manager.step([
            _obs("a", 1, 0.0, (0.0, 0.0), (1.0, 0.0)),
            _obs("b", 7, 0.0, (0.05, 0.0), (1.0, 0.0)),
        ])
        second = manager.step([
            _obs("a", 1, 0.1, (0.1, 0.0), (1.0, 0.0)),
            _obs("b", 7, 0.1, (0.15, 0.0), (1.0, 0.0)),
        ])
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 1)
        self.assertEqual(manager.metrics["duplicate_outputs_suppressed"], 1)

    def test_physical_jump_gate_rejects_teleport_without_state_update(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a"],
            fusion_instant_merge_enabled=False,
            fusion_max_speed_mps=1.0,
            fusion_max_acceleration_mps2=0.0,
            fusion_motion_uncertainty_sigma=0.0,
            fusion_motion_measurement_tolerance_m=0.0,
            fusion_jump_window_s=1.0,
            fusion_max_jump_m=2.0,
        ))
        manager.step([_obs("a", 1, 0.0, (0.0, 0.0))])
        gid = manager.affinity[("a", 1)]
        before_x = manager.tracks[gid].kf.x.copy()
        before_p = manager.tracks[gid].kf.P.copy()
        before_points = manager.tracklets.metrics["points_stored"]

        manager.step([_obs("a", 1, 0.5, (5.0, 0.0))])

        track = manager.tracks[gid]
        np.testing.assert_allclose(track.kf.x, before_x)
        np.testing.assert_allclose(track.kf.P, before_p)
        self.assertEqual(track.state_timestamp_s, 0.0)
        self.assertEqual(track.last_seen_ts_s, 0.0)
        self.assertEqual(manager.affinity[("a", 1)], gid)
        self.assertEqual(manager.tracklets.metrics["points_stored"], before_points)
        self.assertEqual(manager.metrics["fusion_motion_jump_rejected"], 1)

    def test_short_window_absolute_jump_cap_overrides_large_speed_budget(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a"],
            fusion_instant_merge_enabled=False,
            fusion_max_speed_mps=100.0,
            fusion_max_acceleration_mps2=0.0,
            fusion_motion_uncertainty_sigma=0.0,
            fusion_motion_measurement_tolerance_m=0.0,
            fusion_jump_window_s=1.0,
            fusion_max_jump_m=2.0,
        ))
        manager.step([_obs("a", 1, 0.0, (0.0, 0.0))])
        gid = manager.affinity[("a", 1)]

        manager.step([_obs("a", 1, 0.5, (2.01, 0.0))])

        self.assertEqual(manager.tracks[gid].state_timestamp_s, 0.0)
        self.assertEqual(manager.metrics["fusion_motion_jump_rejected"], 1)

    def test_physical_gate_uses_irregular_capture_time_delta(self) -> None:
        def build() -> GlobalTrackManager:
            return GlobalTrackManager(self._config(
                cameras=["a"],
                fusion_instant_merge_enabled=False,
                fusion_max_speed_mps=1.0,
                fusion_max_acceleration_mps2=0.0,
                fusion_motion_uncertainty_sigma=0.0,
                fusion_motion_measurement_tolerance_m=0.0,
                fusion_jump_window_s=2.0,
                fusion_max_jump_m=2.0,
            ))

        short_gap = build()
        short_gap.step([_obs("a", 1, 0.0, (0.0, 0.0))])
        short_gap.step([_obs("a", 1, 0.2, (0.8, 0.0))])
        self.assertEqual(short_gap.metrics["fusion_motion_jump_rejected"], 1)

        long_gap = build()
        long_gap.step([_obs("a", 1, 0.0, (0.0, 0.0))])
        long_gap.step([_obs("a", 1, 1.0, (0.8, 0.0))])
        gid = long_gap.affinity[("a", 1)]
        self.assertEqual(long_gap.metrics["fusion_motion_gate_accepted"], 1)
        self.assertEqual(long_gap.tracks[gid].state_timestamp_s, 1.0)

    def test_trajectory_plausibility_gates_public_gid_promotion(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a"],
            fusion_min_hits=1,
            fusion_provisional_buffer_enabled=True,
            fusion_provisional_min_age_s=0.40,
            fusion_deferred_gid_allocation_enabled=True,
            fusion_new_gid_bootstrap_s=0.10,
            fusion_trajectory_verification_enabled=True,
            fusion_trajectory_min_points=4,
            fusion_trajectory_min_duration_s=0.30,
            fusion_trajectory_max_jump_m=1.50,
            fusion_instant_merge_enabled=False,
        ))
        # Bootstrap tick
        manager.step([], timestamp_s=0.0, arrival_timestamp_s=0.0)

        # 1. Teleporting track (jump 5m) should fail trajectory verification
        manager.step([_obs("a", 1, 0.2, (0.0, 0.0))])
        manager.step([_obs("a", 1, 0.3, (5.0, 0.0))])
        manager.step([_obs("a", 1, 0.4, (10.0, 0.0))])
        manager.step([_obs("a", 1, 0.65, (15.0, 0.0))])
        self.assertLess(manager.affinity[("a", 1)], 0, "Teleporting track must not get public GID")

        # 2. Smooth, plausible walking trajectory should be accepted
        manager.step([_obs("a", 2, 0.70, (0.0, 0.0))])
        manager.step([_obs("a", 2, 0.80, (0.1, 0.0))])
        manager.step([_obs("a", 2, 0.95, (0.2, 0.0))])
        visible = manager.step([_obs("a", 2, 1.15, (0.35, 0.0))])
        self.assertGreater(manager.affinity[("a", 2)], 0, "Plausible walking trajectory must be granted public GID")
        self.assertEqual(len(visible), 1)

    def test_closed_world_interior_birth_rejected_after_bootstrap(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a"],
            fusion_min_hits=1,
            fusion_provisional_buffer_enabled=True,
            fusion_provisional_min_age_s=0.30,
            fusion_deferred_gid_allocation_enabled=True,
            fusion_new_gid_bootstrap_s=0.20,
            fusion_world_entry_zones=[(-10.0, -5.0, 0.0, 5.0)],
            fusion_instant_merge_enabled=False,
        ))
        # Tick in bootstrap
        manager.step([], timestamp_s=0.0, arrival_timestamp_s=0.0)

        # An otherwise plausible interior trajectory still cannot allocate a
        # public identity after bootstrap.
        manager.step([_obs("a", 1, 0.5, (0.0, 0.0))])
        manager.step([_obs("a", 1, 0.8, (0.1, 0.0))])
        manager.step([_obs("a", 1, 1.1, (0.2, 0.0))])
        manager.step([_obs("a", 1, 1.4, (0.3, 0.0))])
        self.assertLess(manager.affinity[("a", 1)], 0, "Interior birth after bootstrap must remain provisional")
        self.assertGreater(manager.metrics["new_gid_entry_deferred"], 0)

        # A real entry starts inside the door footprint, crosses into the room
        # with positive velocity, and supplies four-plus plausible samples.
        manager.step([_obs("a", 2, 1.5, (-7.0, 2.0), (1.8, 0.0))])
        manager.step([_obs("a", 2, 1.8, (-6.5, 2.0), (1.8, 0.0))])
        manager.step([_obs("a", 2, 2.1, (-5.9, 2.0), (1.8, 0.0))])
        manager.step([_obs("a", 2, 2.4, (-5.3, 2.0), (1.8, 0.0))])
        visible = manager.step([_obs("a", 2, 2.7, (-4.8, 2.0), (1.8, 0.0))])
        self.assertGreater(manager.affinity[("a", 2)], 0, "Entry zone birth must be granted public GID")
        self.assertEqual(len(visible), 1)

    def test_dormant_identity_reconnected_via_kinematic_matching(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a"],
            fusion_min_hits=1,
            tracklet_history_s=0.30,
            fusion_max_age_s=0.30,
            fusion_identity_retention_s=0.30,
            fusion_dormant_retention_s=5.0,
            fusion_kinematic_match_gate_m=2.5,
            fusion_provisional_buffer_enabled=True,
            fusion_provisional_min_age_s=0.10,
            fusion_deferred_gid_allocation_enabled=True,
            fusion_new_gid_bootstrap_s=1.0,
            fusion_instant_merge_enabled=False,
        ))
        # Person 1 active at t=0.0 to t=0.2 moving in +X direction at 1.0 m/s
        manager.step([_obs("a", 1, 0.0, (0.0, 0.0), (1.0, 0.0))])
        visible = manager.step([_obs("a", 1, 0.2, (0.2, 0.0), (1.0, 0.0))])
        self.assertEqual(len(visible), 1)
        original_gid = visible[0].gid
        self.assertGreater(original_gid, 0)
        allocations_before_reconnect = manager.metrics["public_gid_allocations"]

        # Person 1 disappears for two seconds and moves to the dormant pool.
        manager.step([], timestamp_s=0.8, arrival_timestamp_s=0.8)
        self.assertIn(original_gid, manager.dormant_tracks)
        self.assertTrue(manager.tracks[original_gid].dormant)

        # It reappears at the two-second forward prediction.
        manager.step([_obs("a", 2, 2.2, (2.2, 0.0), (1.0, 0.0))])
        manager.step([_obs("a", 2, 2.35, (2.35, 0.0), (1.0, 0.0))])

        # Kinematic matching should awaken and reconnect to original_gid
        self.assertEqual(manager.resolve_gid(manager.affinity[("a", 2)]), original_gid)
        self.assertEqual(manager.metrics["dormant_reconnects_accepted"], 1)
        self.assertNotIn(original_gid, manager.dormant_tracks)
        self.assertEqual(
            manager.metrics["public_gid_allocations"],
            allocations_before_reconnect,
        )
        self.assertAlmostEqual(manager.tracks[original_gid].state_timestamp_s, 2.35)

    def test_dormant_candidate_with_opposite_direction_is_rejected(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a"],
            fusion_min_hits=1,
            tracklet_history_s=0.30,
            fusion_max_age_s=0.30,
            fusion_identity_retention_s=0.30,
            fusion_dormant_retention_s=5.0,
            fusion_kinematic_match_gate_m=2.5,
            fusion_provisional_buffer_enabled=True,
            fusion_provisional_min_age_s=0.10,
            fusion_deferred_gid_allocation_enabled=True,
            fusion_new_gid_entry_required=True,
            fusion_new_gid_bootstrap_s=0.10,
            fusion_trajectory_min_direction_cosine=0.0,
            fusion_instant_merge_enabled=False,
        ))
        manager.step([_obs("a", 1, 0.0, (0.0, 0.0), (1.0, 0.0))])
        manager.step([_obs("a", 1, 0.2, (0.2, 0.0), (1.0, 0.0))])
        original_gid = manager.affinity[("a", 1)]
        manager.step([], timestamp_s=0.8, arrival_timestamp_s=0.8)

        manager.step([_obs("a", 2, 2.2, (2.2, 0.0), (-1.0, 0.0))])
        manager.step([_obs("a", 2, 2.35, (2.05, 0.0), (-1.0, 0.0))])

        self.assertIn(original_gid, manager.dormant_tracks)
        self.assertNotEqual(
            manager.resolve_gid(manager.affinity[("a", 2)]), original_gid
        )
        self.assertLess(manager.affinity[("a", 2)], 0)
        self.assertGreater(manager.metrics["trajectory_direction_rejected"], 0)
        self.assertEqual(manager.metrics["dormant_reconnects_accepted"], 0)

    def test_entry_gate_uses_birth_point_not_later_position(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a"],
            fusion_min_hits=1,
            fusion_provisional_buffer_enabled=True,
            fusion_provisional_min_age_s=0.3,
            fusion_deferred_gid_allocation_enabled=True,
            fusion_new_gid_bootstrap_s=0.2,
            fusion_world_entry_zones=[(-10.0, -5.0, 0.0, 5.0)],
            fusion_instant_merge_enabled=False,
        ))
        manager.step([], timestamp_s=0.0, arrival_timestamp_s=0.0)
        # Born just inside the room, then walks backwards into the door zone.
        # The current point is in the zone, but this is not a new entry.
        manager.step([_obs("a", 1, 0.5, (-4.0, 2.0), (-1.0, 0.0))])
        manager.step([_obs("a", 1, 0.8, (-4.4, 2.0), (-1.0, 0.0))])
        manager.step([_obs("a", 1, 1.1, (-4.8, 2.0), (-1.0, 0.0))])
        manager.step([_obs("a", 1, 1.4, (-5.2, 2.0), (-1.0, 0.0))])

        self.assertLess(manager.affinity[("a", 1)], 0)
        self.assertEqual(manager.public_affinity_snapshot(), {})

    def test_world_exit_zone_releases_gid_only_with_outward_motion(self) -> None:
        cfg = self._config(
            cameras=["a"],
            tracklet_history_s=0.1,
            tracklet_late_drop_s=0.1,
            fusion_max_age_s=0.2,
            fusion_identity_retention_s=0.2,
            fusion_dormant_retention_s=5.0,
            fusion_min_hits=1,
            fusion_provisional_buffer_enabled=True,
            fusion_provisional_min_age_s=0.1,
            fusion_deferred_gid_allocation_enabled=True,
            fusion_new_gid_bootstrap_s=1.0,
            fusion_world_exit_zones=[(4.5, 5.5, -1.0, 1.0)],
            fusion_instant_merge_enabled=False,
        )
        manager = GlobalTrackManager(cfg)
        manager.step([_obs("a", 1, 0.0, (4.0, 0.0), (1.0, 0.0))])
        manager.step([_obs("a", 1, 0.1, (4.8, 0.0), (1.0, 0.0))])
        gid = manager.affinity[("a", 1)]

        manager.step([], timestamp_s=0.5, arrival_timestamp_s=0.5)

        self.assertNotIn(gid, manager.tracks)
        self.assertNotIn(gid, manager.dormant_tracks)
        self.assertEqual(manager.metrics["confirmed_identity_exits"], 1)

        wrong_direction = GlobalTrackManager(cfg)
        wrong_direction.step([
            _obs("a", 1, 0.0, (4.0, 0.0), (-1.0, 0.0))
        ])
        wrong_direction.step([
            _obs("a", 1, 0.1, (4.8, 0.0), (-1.0, 0.0))
        ])
        retained_gid = wrong_direction.affinity[("a", 1)]
        wrong_direction.step([], timestamp_s=0.5, arrival_timestamp_s=0.5)
        self.assertIn(retained_gid, wrong_direction.dormant_tracks)
        self.assertEqual(wrong_direction.metrics["confirmed_identity_exits"], 0)

    def test_anti_pingpong_blocks_fast_alternating_id_switch(self) -> None:
        manager = GlobalTrackManager(self._config(
            cameras=["a"],
            fusion_min_hits=1,
            fusion_anti_pingpong_enabled=True,
            fusion_anti_pingpong_frames=4,
            fusion_anti_pingpong_distance_m=1.5,
        ))
        # Establish two published tracks in camera a
        t1 = manager.step([_obs("a", 1, 0.0, (0.0, 0.0))])
        gid1 = manager.affinity[("a", 1)]
        manager.tracks[gid1].emitted = True

        # Manually create a second track gid2 close by
        gid2 = manager._next_gid
        manager._next_gid += 1
        manager.tracks[gid2] = GlobalTrack(
            gid=gid2,
            kf=ConstantVelocityKalman(
                np.array([0.5, 0.0]),
                process_noise=1.0,
                measurement_noise=0.5,
            ),
            emitted=True,
            confirmed=True,
            created_at_s=-1.0,
            state_timestamp_s=0.0,
            last_seen_ts_s=0.0,
        )

        # A proposal must remain stable for all four consecutive frames. The
        # first three are blocked, eliminating one-frame ownership flips.
        with patch.object(
            manager, "_batch_local_reconnect", return_value={("a", 1): (gid2, False)}
        ):
            for timestamp in (0.1, 0.2, 0.3):
                manager.step([_obs("a", 1, timestamp, (0.0, 0.0))])
                self.assertEqual(
                    manager.resolve_gid(manager.affinity[("a", 1)]), gid1
                )
            manager.step([_obs("a", 1, 0.4, (0.0, 0.0))])

        self.assertEqual(manager.metrics["anti_pingpong_switches_blocked"], 3)
        self.assertEqual(manager.resolve_gid(manager.affinity[("a", 1)]), gid2)


if __name__ == "__main__":
    unittest.main()
