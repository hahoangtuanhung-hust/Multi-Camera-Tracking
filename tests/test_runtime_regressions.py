"""Regression cases for delayed cameras, bounded state and tracker lifecycle."""
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from multicam_tracker.bytetrack import ByteTracker, match_by_iou
from multicam_tracker.config import PipelineConfig, apply_tracking_profile
from multicam_tracker.demo import run_pipeline
from multicam_tracker.fusion import CameraObservation, GlobalTrackManager
from multicam_tracker.timebase import FrameBatch, FramePacket
from multicam_tracker.tracklets import TrackletStore
from multicam_tracker.video import AsyncVideoReader, SyncedVideoReader


def observation(camera, local_id, timestamp, x=0.0):
    return CameraObservation(camera, local_id, np.array([x, 0.0]), np.zeros(4),
                             0.9, timestamp, arrival_ts_s=timestamp)


def detection():
    return np.array([[0., 0., 10., 20., .9]])


class LifecycleRegressionTests(unittest.TestCase):
    def test_baseline_profile_restores_legacy_clock_delta_policy(self) -> None:
        cfg = PipelineConfig()
        self.assertEqual(cfg.clock_max_pair_delta_spread_s, 0.40)

        apply_tracking_profile(cfg, "identity-v1-baseline")
        self.assertTrue(np.isinf(cfg.clock_max_pair_delta_spread_s))

        apply_tracking_profile(cfg, "identity-v1-drift-guard")
        self.assertEqual(cfg.clock_max_pair_delta_spread_s, 0.40)

    def test_ownership_profile_and_rollback_are_explicit(self) -> None:
        cfg = PipelineConfig()
        apply_tracking_profile(cfg, "identity-v1-ownership")
        self.assertTrue(cfg.fusion_gid_ownership_hysteresis_enabled)
        self.assertEqual(cfg.fusion_provisional_min_age_s, 0.5)
        self.assertEqual(cfg.fusion_provisional_buffer_bootstrap_s, 1.0)
        self.assertEqual(cfg.fusion_local_reconnect_confirmations, 3)
        self.assertEqual(cfg.fusion_published_merge_confirmations, 5)

        apply_tracking_profile(cfg, "identity-v1-drift-guard")
        self.assertFalse(cfg.fusion_gid_ownership_hysteresis_enabled)
        self.assertFalse(cfg.fusion_provisional_buffer_enabled)
        self.assertEqual(cfg.fusion_provisional_buffer_bootstrap_s, 0.0)
        self.assertEqual(cfg.fusion_local_reconnect_confirmations, 1)
        self.assertEqual(cfg.fusion_published_merge_confirmations, 1)

    def test_hybridsort_ablation_profiles_do_not_mutate_baseline(self) -> None:
        baseline = PipelineConfig()
        apply_tracking_profile(baseline, "hybridsort")
        self.assertFalse(baseline.fusion_trajectory_smoothing_enabled)
        self.assertEqual(baseline.track_tentative_max_misses, 0)

        trajectory = PipelineConfig()
        apply_tracking_profile(trajectory, "hybridsort-trajectory")
        self.assertTrue(trajectory.fusion_trajectory_smoothing_enabled)
        self.assertEqual(trajectory.fusion_trajectory_window_s, 3.5)
        self.assertEqual(trajectory.tracklet_history_s, 7.0)

        combined = PipelineConfig()
        apply_tracking_profile(combined, "hybridsort-stable")
        self.assertEqual(combined.track_tentative_max_misses, 1)
        self.assertTrue(combined.ocsort_observation_recovery_enabled)
        self.assertEqual(combined.fusion_identity_retention_s, 5.0)
        self.assertEqual(combined.fusion_max_jump_m, 2.0)
        self.assertEqual(combined.fusion_max_speed_mps, 3.2)
        self.assertEqual(combined.fusion_max_acceleration_mps2, 4.0)

    def test_zero_trail_length_disables_history(self):
        manager = GlobalTrackManager(PipelineConfig(cameras=["a"], fusion_min_hits=1, trail_length=0))
        for sequence in range(40):
            manager.step([observation("a", 1, sequence / 30.)])
        self.assertEqual(next(iter(manager.tracks.values())).trail, [])

    def test_matching_respects_interval_while_metadata_updates_every_frame(self):
        manager = GlobalTrackManager(PipelineConfig(cameras=["a"], tracklet_match_interval_s=.25))
        with patch.object(manager, "_match_tracklets") as match:
            for sequence in range(90):
                timestamp = sequence / 30.
                manager.step([observation("a", 1, timestamp)], timestamp_s=timestamp)
        self.assertGreaterEqual(match.call_count, 10)
        self.assertLessEqual(match.call_count, 12)
        self.assertEqual(manager.tracklets.metrics["points_stored"], 90)

    def test_no_frame_arrays_in_tracklet_metadata(self):
        manager = GlobalTrackManager(PipelineConfig(cameras=["a"]))
        manager.step([observation("a", 1, 0.0)])
        point = manager.tracklets.get(("a", 1)).points[0]
        self.assertEqual(point.xy.shape, (2,))
        self.assertEqual(point.velocity.shape, (2,))
        self.assertFalse(hasattr(point, "image"))
        self.assertFalse(hasattr(point, "bbox_xyxy"))

    def test_one_hit_config_confirms_on_birth(self):
        tracker = ByteTracker(PipelineConfig(track_min_hits=1))
        self.assertEqual(len(tracker.update(detection(), timestamp_s=0.0)), 1)

    def test_detection_after_timeout_cannot_resurrect_local_track(self):
        tracker = ByteTracker(PipelineConfig(track_min_hits=2, track_max_age_s=1.0))
        tracker.update(detection(), timestamp_s=0.0)
        original = tracker.update(detection(), timestamp_s=0.1)[0].track_id
        tracker.update(detection(), timestamp_s=3.0)
        self.assertNotIn(original, [track.track_id for track in tracker.tracks])

    def test_iou_preserves_strong_identity_over_two_ambiguous_matches(self):
        # Prefer the strong .90 match to forcing two ambiguous .30 matches.
        with patch("multicam_tracker.bytetrack.iou_matrix", return_value=np.array([[.24, .30], [.30, .90]])):
            pairs, _, _ = match_by_iou(np.zeros((2, 4)), np.zeros((2, 4)), .25)
        self.assertEqual(set(pairs), {(1, 1)})

    def test_metadata_expires_during_empty_scene(self):
        manager = GlobalTrackManager(PipelineConfig(cameras=["a"], fusion_min_hits=1))
        manager.step([observation("a", 1, 0.0)])
        manager.step([], timestamp_s=20.0, arrival_timestamp_s=20.0)
        self.assertEqual(manager.tracklets.histories, {})
        self.assertEqual(manager.tracklets.metrics["points_stored"], 0)

    def test_expired_unknown_observations_do_not_reach_geometry(self):
        manager = GlobalTrackManager(PipelineConfig(cameras=["a", "b"], fusion_min_hits=1))
        manager.step([observation("a", 1, 10.0)])
        manager.step([observation("a", 99, 0.0), observation("b", 99, 0.0)], timestamp_s=10.0)
        self.assertNotIn(("a", 99), manager.affinity)
        self.assertEqual(manager.metrics["late_observations_dropped"], 2)

    def test_same_camera_conflict_includes_first_observation(self):
        manager = GlobalTrackManager(PipelineConfig(cameras=["a"], fusion_min_hits=1))
        manager.step([observation("a", 1, 0.0), observation("a", 2, 0.0)])
        self.assertFalse(manager.merge_gids(manager.affinity[("a", 1)], manager.affinity[("a", 2)]))

    def test_late_geometry_merge_does_not_rewind_live_position(self):
        manager = GlobalTrackManager(PipelineConfig(cameras=["a", "b"], fusion_min_hits=1))
        manager.step([observation("a", 1, 0.0), observation("b", 1, 0.0, 20.0)])
        manager.step([observation("a", 1, 2.0, 2.0)])
        gid = manager.affinity[("a", 1)]
        state = manager.tracks[gid].kf.x.copy()
        manager.step([observation("a", 1, 1.0, 1.0), observation("b", 1, 1.0, 1.1)], timestamp_s=2.0)
        np.testing.assert_array_equal(manager.tracks[gid].kf.x, state)

    def test_metadata_and_aliases_remain_bounded_with_fragmentation(self):
        cfg = PipelineConfig(cameras=["a", "b"], fusion_min_hits=1,
                             fusion_local_reconnect_gate_m=0.0)
        manager = GlobalTrackManager(cfg)
        for timestamp in range(80):
            manager.step([observation("a", timestamp, timestamp, timestamp * 10.),
                          observation("b", timestamp, timestamp, timestamp * 10.)])
        retained_seconds = max(cfg.tracklet_history_s, cfg.tracklet_late_drop_s)
        retained_steps = int(np.ceil(retained_seconds)) + 1
        self.assertLessEqual(len(manager.aliases), retained_steps)
        self.assertLessEqual(len(manager.affinity), 2 * retained_steps)

    def test_supplied_velocity_is_not_overwritten_on_next_append(self):
        store = TrackletStore(PipelineConfig())
        store.append("a", 1, 0., np.zeros(2), .9, np.array([3., 4.]))
        store.append("a", 1, .1, np.ones(2), .9, np.array([5., 6.]))
        np.testing.assert_array_equal(store.get(("a", 1)).points[0].velocity, [3., 4.])


class FakeCapture:
    """Tiny deterministic camera, including an optional blocked first read."""
    def __init__(self, source, *args, **kwargs):
        self.source = str(source)
        self.index = 0
        self.released = False
        self.count = 3 if "short" in self.source else 12

    def isOpened(self):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: 30., cv2.CAP_PROP_FRAME_COUNT: self.count}.get(prop, 0.)

    def read(self):
        if "slow" in self.source:
            time.sleep(.15)
        if self.index >= self.count:
            return False, None
        self.index += 1
        return True, np.full((8, 8, 3), self.index, np.uint8)

    def grab(self):
        self.index += 1
        return self.index <= self.count

    def release(self):
        self.released = True


class IngestRegressionTests(unittest.TestCase):
    def test_reader_exception_finishes_camera_and_preserves_healthy_frames(self):
        def open_camera(source, *args, **kwargs):
            if "bad" in source:
                raise OSError("capture disconnected")
            return FakeCapture(source)

        with patch("multicam_tracker.video.open_video", open_camera):
            reader = AsyncVideoReader({"a": "a", "b": "bad"})
            try:
                batches = list(reader.iter_batches())
                self.assertEqual(sum("a" in batch.packets for batch in batches), 12)
                self.assertIn("b", reader.errors)
            finally:
                reader.close()
            self.assertFalse(any(thread.is_alive() for thread in reader._threads))

    @patch("multicam_tracker.video.cv2.VideoCapture", FakeCapture)
    def test_sync_eof_does_not_discard_healthy_camera_frames(self):
        reader = SyncedVideoReader(Path("unused"), ["long", "short"])
        try:
            packets = [packet for batch in reader.iter_batches() for packet in batch.packets.values()]
        finally:
            reader.close()
        self.assertEqual([p.sequence for p in packets if p.camera == "long"], list(range(12)))

    @patch("multicam_tracker.video.cv2.VideoCapture", FakeCapture)
    def test_async_slow_consumer_keeps_every_frame_with_bounded_queue(self):
        reader = AsyncVideoReader({"a": "a", "b": "b"}, queue_size=2)
        try:
            time.sleep(.5)
            self.assertTrue(all(queue.qsize() <= 2 for queue in reader.queues.values()))
            packets = [packet for batch in reader.iter_batches() for packet in batch.packets.values()]
        finally:
            reader.close()
        for camera in ("a", "b"):
            self.assertEqual([p.sequence for p in packets if p.camera == camera], list(range(12)))
        self.assertEqual(sum(reader.dropped_frames.values()), 0)

    @patch("multicam_tracker.video.cv2.VideoCapture", FakeCapture)
    def test_async_slow_camera_does_not_block_healthy_camera(self):
        reader = AsyncVideoReader({"a": "a", "b": "slow"}, queue_size=2, microbatch_ms=5)
        try:
            iterator = reader.iter_batches()
            first = next(iterator)
            self.assertIn("a", first.packets)
            self.assertNotIn("b", first.packets)
            self.assertLessEqual(reader.queues["a"].qsize(), 2)
        finally:
            reader.close()


if __name__ == "__main__":
    unittest.main()
