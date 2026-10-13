import unittest

import numpy as np

from multicam_tracker.config import PipelineConfig
from multicam_tracker.fusion import CameraObservation, GlobalTrackManager


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


if __name__ == "__main__":
    unittest.main()
