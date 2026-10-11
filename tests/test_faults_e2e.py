import unittest
from dataclasses import replace

import numpy as np

from multicam_tracker.config import PipelineConfig
from multicam_tracker.faults import FaultConfig, FaultInjectedBatchStream
from multicam_tracker.fusion import CameraObservation, GlobalTrackManager
from multicam_tracker.timebase import FrameBatch, FramePacket


def _packet(camera: str, timestamp: float, sequence: int) -> FramePacket:
    return FramePacket(
        camera=camera,
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        capture_ts_s=timestamp,
        arrival_ts_s=timestamp,
        sequence=sequence,
        source_frame_index=sequence,
    )


def _source(duration_s: float = 4.0):
    for sequence, timestamp in enumerate(np.arange(0.0, duration_s + 0.001, 0.1)):
        packets = {
            camera: _packet(camera, float(timestamp), sequence) for camera in ("a", "b")
        }
        yield FrameBatch(packets, emitted_at_s=float(timestamp))


class FaultAndEndToEndTests(unittest.TestCase):
    def test_one_to_three_second_delay_keeps_canonical_id(self):
        for delay in (1., 2., 3.):
            with self.subTest(delay=delay):
                manager = GlobalTrackManager(PipelineConfig(cameras=["a", "b"], fusion_min_hits=1))
                canonical = None
                for batch in FaultInjectedBatchStream(_source(9.), FaultConfig(camera="b", delay_s=delay)):
                    observations = [CameraObservation(
                        p.camera, 1, np.array([.5 * p.source_frame_index / 10., 0.]),
                        np.zeros(4), .95, p.capture_ts_s, np.array([.5, 0.]), p.arrival_ts_s,
                    ) for p in batch.packets.values()]
                    manager.step(observations, max(p.capture_ts_s for p in batch.packets.values()), batch.emitted_at_s)
                    current = manager.affinity.get(("a", 1))
                    if canonical is None:
                        canonical = current
                    elif current is not None:
                        self.assertEqual(current, canonical)
                self.assertEqual(manager.affinity[("a", 1)], manager.affinity[("b", 1)])
                self.assertEqual(manager.tracklets.metrics["late_dropped"], 0)

    def test_real_monotonic_arrivals_share_simulation_clock_with_delayed_camera(self):
        source = [FrameBatch({camera: replace(packet, arrival_ts_s=500_000.0 + packet.capture_ts_s)
                              for camera, packet in batch.packets.items()}, 500_000.0 + batch.emitted_at_s)
                  for batch in _source(.2)]
        packets = [packet for batch in FaultInjectedBatchStream(source, FaultConfig(camera="b", delay_s=2.))
                   for packet in batch.packets.values()]
        for packet in packets:
            expected_delay = 2.0 if packet.camera == "b" else 0.0
            self.assertAlmostEqual(packet.arrival_ts_s - packet.capture_ts_s, expected_delay)

    def test_network_delay_changes_arrival_not_capture_time(self) -> None:
        stream = FaultInjectedBatchStream(
            _source(0.2), FaultConfig(camera="b", delay_s=2.0)
        )
        delayed = [
            packet
            for batch in stream
            for packet in batch.packets.values()
            if packet.camera == "b"
        ]
        self.assertEqual([packet.capture_ts_s for packet in delayed], [0.0, 0.1, 0.2])
        self.assertEqual([packet.arrival_ts_s for packet in delayed], [2.0, 2.1, 2.2])

    def test_full_delay_jitter_drop_and_clock_offset_recovers_one_gid(self) -> None:
        cfg = PipelineConfig(
            cameras=["a", "b"],
            fusion_min_hits=1,
            tracklet_history_s=5.0,
            tracklet_late_drop_s=5.0,
            tracklet_match_interval_s=0.1,
        )
        manager = GlobalTrackManager(cfg)
        stream = FaultInjectedBatchStream(
            _source(5.0),
            FaultConfig(
                camera="b",
                delay_s=2.0,
                jitter_s=0.03,
                drop_rate=0.08,
                clock_offset_s=1.7,
                clock_drift_ppm=300.0,
                seed=11,
            ),
        )
        for batch in stream:
            observations = []
            for packet in batch.packets.values():
                # Ground truth motion is derived from source index, independent
                # of the intentionally faulty sensor clock.
                event_time = packet.source_frame_index * 0.1
                observations.append(CameraObservation(
                    camera=packet.camera,
                    local_id=1,
                    world_xy=np.array([0.55 * event_time, 0.2 * event_time]),
                    bbox_xyxy=np.zeros(4),
                    conf=0.96,
                    capture_ts_s=packet.capture_ts_s,
                    arrival_ts_s=packet.arrival_ts_s,
                ))
            manager.step(observations, max(packet.capture_ts_s for packet in batch.packets.values()))

        self.assertIn(("a", 1), manager.affinity)
        self.assertIn(("b", 1), manager.affinity)
        self.assertEqual(manager.affinity[("a", 1)], manager.affinity[("b", 1)])
        self.assertGreater(manager.metrics["global_id_merges"], 0)
        self.assertGreater(stream.metrics["dropped"], 0)
        self.assertLessEqual(manager.tracklets.approximate_bytes, 2 * 60 * 6 * 8)

    def test_total_drop_of_one_camera_does_not_block_healthy_camera(self) -> None:
        stream = FaultInjectedBatchStream(
            _source(1.0), FaultConfig(camera="b", drop_rate=1.0)
        )
        batches = list(stream)
        self.assertEqual(sum("a" in batch.packets for batch in batches), 11)
        self.assertFalse(any("b" in batch.packets for batch in batches))
        self.assertEqual(stream.metrics["dropped"], 11)


if __name__ == "__main__":
    unittest.main()
