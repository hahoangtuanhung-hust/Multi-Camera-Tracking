import unittest

import numpy as np

from multicam_tracker.config import PipelineConfig
from multicam_tracker.fusion import CameraObservation, GlobalTrackManager


def _obs(camera: str, local_id: int, timestamp: float, xy: tuple[float, float]):
    return CameraObservation(
        camera=camera,
        local_id=local_id,
        world_xy=np.asarray(xy, dtype=float),
        bbox_xyxy=np.zeros(4, dtype=float),
        conf=0.95,
        capture_ts_s=timestamp,
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

    def test_provisional_ids_converge_to_oldest_canonical_id(self) -> None:
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
        self.assertEqual(manager.affinity[("a", 1)], first_ids[0])
        self.assertEqual(manager.affinity[("b", 7)], first_ids[0])
        self.assertEqual(manager.resolve_gid(first_ids[1]), first_ids[0])
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

    def test_fragmented_local_id_reconnects_to_retained_same_camera_track(self) -> None:
        manager = GlobalTrackManager(self._config())
        for timestamp in (0.0, 0.2, 0.4):
            manager.step([_obs("a", 1, timestamp, (timestamp, 0.0))])
        original_gid = manager.affinity[("a", 1)]
        manager.step([_obs("a", 2, 0.6, (0.6, 0.0))])
        self.assertEqual(manager.affinity[("a", 2)], original_gid)
        self.assertEqual(manager.metrics["local_fragments_reconnected"], 1)

    def test_simultaneous_same_camera_tracks_do_not_reconnect(self) -> None:
        manager = GlobalTrackManager(self._config())
        manager.step([_obs("a", 1, 0.0, (0.0, 0.0))])
        manager.step([
            _obs("a", 1, 0.2, (0.2, 0.0)),
            _obs("a", 2, 0.2, (0.25, 0.0)),
        ])
        self.assertNotEqual(manager.affinity[("a", 1)], manager.affinity[("a", 2)])


if __name__ == "__main__":
    unittest.main()
