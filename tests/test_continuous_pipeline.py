import unittest
from unittest.mock import patch

import numpy as np

from multicam_tracker.config import PipelineConfig
from multicam_tracker.demo import run_pipeline
from multicam_tracker.timebase import FrameBatch, FramePacket


class _CountingDetector:
    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []

    def detect_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        self.calls += 1
        self.batch_sizes.append(len(images))
        return [np.empty((0, 5), dtype=float) for _ in images]


def _batches(count: int) -> list[FrameBatch]:
    batches = []
    for sequence in range(count):
        timestamp = sequence / 30.0
        packets = {
            camera: FramePacket(
                camera=camera,
                image=np.zeros((8, 8, 3), dtype=np.uint8),
                capture_ts_s=timestamp,
                arrival_ts_s=timestamp,
                sequence=sequence,
                source_frame_index=sequence,
            )
            for camera in ("a", "b")
        }
        batches.append(FrameBatch(packets=packets, emitted_at_s=timestamp))
    return batches


class ContinuousPipelineTests(unittest.TestCase):
    @patch("multicam_tracker.demo.ByteTracker.update", return_value=[])
    @patch("multicam_tracker.demo.load_calibrations", return_value={})
    def test_four_camera_detector_once_and_tracking_every_frame(self, _load, update):
        detector = _CountingDetector()
        batches = []
        from dataclasses import replace
        for batch in _batches(75):
            packets = dict(batch.packets)
            packets.update({name: replace(batch.packets["a"], camera=name) for name in ("c", "d")})
            batches.append(FrameBatch(packets, batch.emitted_at_s))
        list(run_pipeline(PipelineConfig(cameras=["a", "b", "c", "d"]),
                          batch_iter=batches, detector=detector))
        self.assertEqual(detector.calls, 75)
        self.assertEqual(detector.batch_sizes, [4] * 75)
        self.assertEqual(update.call_count, 300)

    @patch("multicam_tracker.demo.load_calibrations", return_value={"a": object(), "b": object()})
    def test_dynamic_batches_and_delayed_camera_are_processed_once(self, _load):
        batches = _batches(75)
        stream = []
        for batch in batches:
            stream.append(FrameBatch({"a": batch.packets["a"]}, batch.emitted_at_s))
            if batch.packets["b"].sequence >= 3:
                delayed = batches[batch.packets["b"].sequence - 3].packets["b"]
                stream.append(FrameBatch({"b": delayed}, batch.emitted_at_s))
        detector = _CountingDetector()
        ticks = list(run_pipeline(PipelineConfig(cameras=["a", "b"]),
                                  max_frames=50, batch_iter=stream, detector=detector))
        self.assertEqual(detector.calls, 100)
        self.assertEqual(sum(detector.batch_sizes), 100)
        for camera in ("a", "b"):
            self.assertEqual([t["packets"][camera].sequence for t in ticks if camera in t["packets"]], list(range(50)))

    @patch("multicam_tracker.demo.load_calibrations", return_value={"a": object(), "b": object()})
    def test_reordered_or_duplicate_packet_does_not_reach_tracker(self, _load):
        batches = _batches(3)
        detector = _CountingDetector()
        ticks = list(run_pipeline(PipelineConfig(cameras=["a", "b"]),
                                  batch_iter=[batches[0], batches[2], batches[1], batches[2]],
                                  detector=detector))
        self.assertEqual(detector.calls, 2)
        self.assertEqual(ticks[-1]["frame_index"], 2)

    @patch("multicam_tracker.demo.load_calibrations", return_value={"a": object(), "b": object()})
    def test_detector_cardinality_mismatch_fails_clearly(self, _load):
        detector = _CountingDetector()
        detector.detect_batch = lambda images: []
        with self.assertRaisesRegex(ValueError, "one detection array"):
            list(run_pipeline(PipelineConfig(cameras=["a", "b"]),
                              batch_iter=_batches(1), detector=detector))

    @patch("multicam_tracker.demo.load_calibrations", return_value={"a": object(), "b": object()})
    def test_default_processes_every_frame_without_a_50_frame_gate(self, _load) -> None:
        cfg = PipelineConfig(cameras=["a", "b"])
        detector = _CountingDetector()

        ticks = list(run_pipeline(cfg, batch_iter=_batches(75), detector=detector))

        self.assertEqual(cfg.stride, 1)
        self.assertEqual(detector.calls, 75)
        self.assertEqual(detector.batch_sizes, [2] * 75)
        self.assertEqual([tick["frame_index"] for tick in ticks], list(range(75)))
        self.assertEqual([tick["tick_index"] for tick in ticks], list(range(1, 76)))
        self.assertEqual(ticks[-1]["metrics"]["detector_batches"], 75)

    @patch("multicam_tracker.demo.load_calibrations", return_value={"a": object(), "b": object()})
    def test_frame_limit_is_only_applied_when_explicitly_requested(self, _load) -> None:
        cfg = PipelineConfig(cameras=["a", "b"])
        detector = _CountingDetector()

        ticks = list(
            run_pipeline(
                cfg,
                max_frames=50,
                batch_iter=_batches(75),
                detector=detector,
            )
        )

        self.assertEqual(len(ticks), 50)
        self.assertEqual(detector.calls, 50)
        self.assertEqual(ticks[-1]["frame_index"], 49)


if __name__ == "__main__":
    unittest.main()
