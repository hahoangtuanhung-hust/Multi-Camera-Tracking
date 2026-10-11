"""Reproducible stride-one CPU benchmark, with no renderer or frame cache.

uv run python benchmarks/cpu_pipeline.py --frames 300 --out data/cpu-audit/run.json
Add --profile for cProfile (timings then include profiler overhead).
"""
from __future__ import annotations

import argparse
import cProfile
import json
import pickle
from pathlib import Path
import pstats
from time import perf_counter

import cv2
import numpy as np
import psutil
import torch

from multicam_tracker.config import PipelineConfig
from multicam_tracker.demo import run_pipeline
from multicam_tracker.detector import PersonDetector
from multicam_tracker.evaluate import ground_truth_by_frame
from evaluate_thresholds import _score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cameras", nargs="+")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--capture-detections", type=Path,
                        help="Save compact detection/packet metadata for offline regression replay")
    args = parser.parse_args()
    cfg = PipelineConfig(device="cpu", stride=1, async_ingest=False)
    if args.cameras:
        cfg.cameras = args.cameras
    if args.threads is not None and hasattr(cfg, "cpu_threads"):
        cfg.cpu_threads = args.threads
    args.out.parent.mkdir(parents=True, exist_ok=True)
    detector = PersonDetector(cfg)
    process = psutil.Process()
    rss_samples = []
    predictions = []
    batch_sizes = []
    detection_s = 0.0
    last_detections = []
    replay = []
    original_detect = detector.detect_batch

    def detect(frames):
        nonlocal detection_s, last_detections
        started = perf_counter()
        result = original_detect(frames)
        detection_s += perf_counter() - started
        batch_sizes.append(len(frames))
        last_detections = result
        return result

    detector.detect_batch = detect
    profile = cProfile.Profile() if args.profile else None
    if profile:
        profile.enable()
    started = perf_counter()
    warm_started = None
    warm_ticks = 0
    last_metrics = {}
    for tick in run_pipeline(cfg, max_frames=args.frames, detector=detector):
        if args.capture_detections:
            replay.append({
                "packets": {camera: {name: getattr(packet, name) for name in
                            ("camera", "capture_ts_s", "arrival_ts_s", "sequence", "source_frame_index")}
                            for camera, packet in tick["packets"].items()},
                "detections": last_detections,
            })
        predictions.append((tick["frame_index"], {
            track.gid: track.world_xy.copy() for track in tick["global_tracks"]
        }))
        last_metrics = tick["metrics"]
        if tick["tick_index"] == 10:
            warm_started = perf_counter()
        elif tick["tick_index"] > 10:
            warm_ticks += 1
        rss = process.memory_info().rss / 2**20
        rss_samples.append(rss)
        if tick["tick_index"] % 25 == 0:
            print(f"ticks={tick['tick_index']} people={len(tick['global_tracks'])} "
                  f"elapsed={perf_counter()-started:.1f}s RSS={rss:.1f}MiB "
                  f"merges={last_metrics.get('global_id_merges', 0)} "
                  f"late_drop={last_metrics.get('tracklet_late_dropped', 0)}", flush=True)
    elapsed = perf_counter() - started
    if args.capture_detections:
        args.capture_detections.write_bytes(pickle.dumps(replay))
    warm_elapsed = perf_counter() - warm_started if warm_started else elapsed
    if profile:
        profile.disable()
        profile.dump_stats(str(args.out.with_suffix(".prof")))
        with args.out.with_suffix(".profile.txt").open("w", encoding="utf-8") as handle:
            pstats.Stats(profile, stream=handle).strip_dirs().sort_stats("cumulative").print_stats(55)
    gt = ground_truth_by_frame(cfg.scene_dir, cfg.cameras, cfg.camera_start_frame_offsets)
    summary = {
        "cameras": cfg.cameras, "device": cfg.device, "stride": cfg.stride,
        "ticks": len(predictions), "camera_frames": sum(batch_sizes),
        "elapsed_s": elapsed, "ticks_per_s": len(predictions) / elapsed,
        "warm_ticks_per_s": warm_ticks / warm_elapsed,
        "detection_s": detection_s, "profiled": args.profile,
        "rss_peak_mib": max(rss_samples, default=0), "rss_samples_mib": rss_samples,
        "people_last": len(predictions[-1][1]) if predictions else 0,
        "score": _score(predictions, gt), "metrics": last_metrics,
        "threads": {"torch": torch.get_num_threads(), "opencv": cv2.getNumThreads()},
        "errors": [],
    }
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rss_samples_mib"}, indent=2))


if __name__ == "__main__":
    main()
