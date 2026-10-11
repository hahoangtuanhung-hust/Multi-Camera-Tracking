"""Deterministic capture-clock and network-delay simulation for demos/tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
from collections.abc import Iterable, Iterator

import numpy as np

from .timebase import FrameBatch, FramePacket


@dataclass(frozen=True)
class FaultConfig:
    camera: str
    delay_s: float = 0.0
    jitter_s: float = 0.0
    drop_rate: float = 0.0
    clock_offset_s: float = 0.0
    clock_drift_ppm: float = 0.0
    seed: int = 7


class FaultInjectedBatchStream:
    """Simulate arrival on the capture timeline, independently of CPU runtime.

    The injector necessarily holds delayed images; production ingest queues
    and the metadata-only tracklet store are separate from this test helper.
    """

    def __init__(self, source: Iterable[FrameBatch], fault: FaultConfig):
        self.source = source
        self.fault = fault
        self.metrics = {"delayed": 0, "dropped": 0, "released": 0}
        self._rng = np.random.default_rng(fault.seed)
        self._order = 0
        self._clock_origin_s: float | None = None

    def __iter__(self) -> Iterator[FrameBatch]:
        heap: list[tuple[float, int, FramePacket]] = []
        watermark = float("-inf")
        for batch in self.source:
            if batch.packets:
                watermark = max(watermark, max(p.capture_ts_s for p in batch.packets.values()))
            immediate: dict[str, FramePacket] = {}
            for packet in batch.packets.values():
                if packet.camera != self.fault.camera:
                    # Never mix real monotonic arrival timestamps with the
                    # simulated event-time arrivals of delayed packets.
                    immediate[packet.camera] = replace(packet, arrival_ts_s=packet.capture_ts_s)
                    continue
                if self._rng.random() < self.fault.drop_rate:
                    self.metrics["dropped"] += 1
                    continue
                if self._clock_origin_s is None:
                    self._clock_origin_s = packet.capture_ts_s
                elapsed = packet.capture_ts_s - self._clock_origin_s
                skewed_capture = (
                    packet.capture_ts_s
                    + self.fault.clock_offset_s
                    + elapsed * self.fault.clock_drift_ppm * 1e-6
                )
                jitter = self._rng.normal(0.0, self.fault.jitter_s) if self.fault.jitter_s else 0.0
                release_s = packet.capture_ts_s + max(0.0, self.fault.delay_s + jitter)
                delayed = replace(
                    packet,
                    capture_ts_s=skewed_capture,
                    arrival_ts_s=release_s,
                )
                heapq.heappush(heap, (release_s, self._order, delayed))
                self._order += 1
                self.metrics["delayed"] += 1
            if immediate:
                yield FrameBatch(immediate, watermark, dict(batch.dropped_frames))
            while heap and heap[0][0] <= watermark:
                release_s, _order, packet = heapq.heappop(heap)
                self.metrics["released"] += 1
                yield FrameBatch({packet.camera: packet}, release_s, dict(batch.dropped_frames))
        while heap:
            release_s, _order, packet = heapq.heappop(heap)
            self.metrics["released"] += 1
            yield FrameBatch({packet.camera: packet}, release_s)
