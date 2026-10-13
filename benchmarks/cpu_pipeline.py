"""Reproducible stride-one CPU benchmark, with no renderer or frame cache.

uv run python benchmarks/cpu_pipeline.py --frames 300 --out data/cpu-audit/run.json
Add --profile for cProfile (timings then include profiler overhead).
"""
from __future__ import annotations

import argparse
import cProfile
from datetime import datetime, timezone
import hashlib
import json
import pickle
from pathlib import Path
import pstats
from time import perf_counter

import cv2
import numpy as np
import psutil
import torch

from multicam_tracker.config import (
    PipelineConfig,
    TRACKING_PROFILE_OVERRIDES,
    DEFAULT_TRACKING_PROFILE,
    apply_tracking_profile,
)
from multicam_tracker.demo import run_pipeline
from multicam_tracker.detector import PersonDetector
from multicam_tracker.evaluate import ground_truth_by_frame
from multicam_tracker.local_tracker import LOCAL_TRACKERS
from evaluate_thresholds import _score


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_metadata(cfg: PipelineConfig, ticks: int) -> dict[str, object]:
    """Fingerprint every input that can change cached detector output."""
    model_path = Path(cfg.model_name).resolve()
    scene = cfg.scene_dir.resolve()
    videos = {
        camera: {
            "path": str((scene / camera / "video.mp4").resolve()),
            "bytes": (scene / camera / "video.mp4").stat().st_size,
            "mtime_ns": (scene / camera / "video.mp4").stat().st_mtime_ns,
        }
        for camera in cfg.cameras
    }
    return {
        "schema": "multicam-detection-cache-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ticks": ticks,
        "detector": {
            "model": str(model_path),
            "model_sha256": _sha256(model_path),
            "imgsz": cfg.imgsz,
            "conf_threshold": cfg.conf_threshold,
            "person_class_id": cfg.person_class_id,
            "device": cfg.device,
        },
        "input": {
            "scene": str(scene),
            "cameras": list(cfg.cameras),
            "stride": cfg.stride,
            "camera_start_frame_offsets": dict(cfg.camera_start_frame_offsets),
            "videos": videos,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--tracking-profile",
        choices=sorted(TRACKING_PROFILE_OVERRIDES),
        default=DEFAULT_TRACKING_PROFILE,
        help=f"Identity policy to benchmark (default: {DEFAULT_TRACKING_PROFILE})",
    )
    parser.add_argument("--cameras", nargs="+")
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--local-tracker",
        choices=LOCAL_TRACKERS,
        default="ocsort",
        help="Per-camera tracker (default: ocsort)",
    )
    parser.add_argument("--capture-detections", type=Path,
                        help="Save compact detection/packet metadata for offline regression replay")
    args = parser.parse_args()
    cfg = PipelineConfig(device="cpu", stride=1, async_ingest=False)
    apply_tracking_profile(cfg, args.tracking_profile)
    cfg.local_tracker = args.local_tracker
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
        args.capture_detections.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": _cache_metadata(cfg, len(replay)),
            "rows": replay,
        }
        args.capture_detections.write_bytes(
            pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        )
        args.capture_detections.with_suffix(".meta.json").write_text(
            json.dumps(payload["metadata"], indent=2), encoding="utf-8"
        )
    warm_elapsed = perf_counter() - warm_started if warm_started else elapsed
    if profile:
        profile.disable()
        profile.dump_stats(str(args.out.with_suffix(".prof")))
        with args.out.with_suffix(".profile.txt").open("w", encoding="utf-8") as handle:
            pstats.Stats(profile, stream=handle).strip_dirs().sort_stats("cumulative").print_stats(55)
    gt = ground_truth_by_frame(cfg.scene_dir, cfg.cameras, cfg.camera_start_frame_offsets)
    summary = {
        "cameras": cfg.cameras, "device": cfg.device, "stride": cfg.stride,
        "tracking_profile": args.tracking_profile,
        "local_tracker": cfg.local_tracker,
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
