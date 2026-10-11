"""Long metadata-only run with ID churn and no retained images/predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil

from multicam_tracker.config import PipelineConfig
from multicam_tracker.fusion import CameraObservation, GlobalTrackManager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=3600)
    parser.add_argument("--out", type=Path, default=Path("data/cpu-audit/retention.json"))
    args = parser.parse_args()
    cfg = PipelineConfig(cameras=["a", "b", "c", "d"], fusion_min_hits=1)
    manager = GlobalTrackManager(cfg)
    process = psutil.Process()
    samples = []
    started = perf_counter()
    for tick in range(args.ticks):
        timestamp = tick / 30.
        # Fresh local IDs each second simulate fragmentation while old
        # aliases, memberships and histories must stay bounded.
        observations = [CameraObservation(
            camera, tick // 30, np.array([.3 * timestamp, 0.]), np.zeros(4), .95,
            timestamp, np.array([.3, 0.]), timestamp,
        ) for camera in cfg.cameras]
        manager.step(observations, timestamp, timestamp)
        if tick % 300 == 299:
            sample = {"tick": tick + 1, "rss_mib": process.memory_info().rss / 2**20,
                      "histories": len(manager.tracklets.histories),
                      "points": manager.tracklets.metrics["points_stored"],
                      "aliases": len(manager.aliases), "affinities": len(manager.affinity),
                      "members": sum(map(len, manager.members.values()))}
            samples.append(sample)
            print(json.dumps(sample), flush=True)
            assert sample["points"] <= 4 * 152
            assert sample["histories"] <= 4 * 7
            assert sample["aliases"] <= 4 * 7
            assert sample["affinities"] <= 4 * 7
            assert sample["members"] <= 4 * 7
    summary = {"ticks": args.ticks, "event_seconds": args.ticks / 30.,
               "elapsed_s": perf_counter() - started, "samples": samples}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
