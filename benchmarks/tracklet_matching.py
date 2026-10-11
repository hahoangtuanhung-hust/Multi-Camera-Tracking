"""Small CPU benchmark for the metadata-only cross-camera matcher."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from multicam_tracker.config import PipelineConfig
from multicam_tracker.tracklets import TrackletMatcher, TrackletStore


def main() -> None:
    cfg = PipelineConfig(
        cameras=[f"camera_{index}" for index in range(4)],
        tracklet_history_s=6.0,
    )
    stores = {camera: TrackletStore(cfg) for camera in cfg.cameras}
    offsets = [0.0, 0.8, -1.2, 2.0]
    people = 5
    for camera, offset in zip(cfg.cameras, offsets):
        for local_id in range(people):
            for event_time in np.arange(0.0, 5.0, 0.1):
                xy = np.array([0.7 * event_time + local_id * 0.15, local_id * 1.5])
                stores[camera].append(camera, local_id, event_time + offset, xy, 0.95)

    matcher = TrackletMatcher(cfg)
    left = stores[cfg.cameras[0]].eligible()
    started = perf_counter()
    matches = 0
    for camera in cfg.cameras[1:]:
        matches += len(matcher.assign(left, stores[camera].eligible()))
    cold_ms = (perf_counter() - started) * 1000.0
    started = perf_counter()
    warm_matches = 0
    for camera, offset in zip(cfg.cameras[1:], offsets[1:]):
        warm_matches += len(matcher.assign(
            left,
            stores[camera].eligible(),
            center_s=-offset,
            radius_s=cfg.offset_refine_window_s,
        ))
    warm_ms = (perf_counter() - started) * 1000.0
    points = sum(len(history.points) for store in stores.values() for history in store.histories.values())
    memory_kib = sum(store.approximate_bytes for store in stores.values()) / 1024.0
    print(
        f"4 cameras x {people} people, {points} points: "
        f"{matches} cold matches in {cold_ms:.1f} ms; "
        f"{warm_matches} steady-state matches in {warm_ms:.1f} ms; "
        f"metadata ~ {memory_kib:.1f} KiB"
    )


if __name__ == "__main__":
    main()
