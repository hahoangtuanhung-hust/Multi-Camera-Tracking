"""Evaluate fused global tracks against the dataset's ground truth.

Runs the exact same pipeline as the live demo (no visualization), then
matches predicted world positions to ground-truth world positions and reports:

- CLEAR MOT: MOTA, MOTP and ID switches
- Identity: IDF1, ID precision (IDP) and ID recall (IDR)
- HOTA: HOTA, DetA, AssA and LocA over 19 localization thresholds
- world-plane precision, recall, mean distance and position RMSE

Run: ``uv run evaluate --max-frames 1200``
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import PipelineConfig
from .demo import run_pipeline
from .mot_metrics import evaluate_mot

MATCH_GATE_M = 1.0


def ground_truth_by_frame(
    scene_dir: Path,
    cameras: list[str],
    camera_frame_offsets: dict[str, int] | None = None,
    max_frame: int | None = None,
) -> dict[int, dict[int, np.ndarray]]:
    """frame -> {person_id: world_xy}, deduplicated across cameras.

    Only people visible in at least one of the *evaluated* cameras count —
    the ground truth covers all 10 cameras, but the pipeline can only see
    the ones that were downloaded.
    """
    camera_ids = {int(cam.split("_")[1]) for cam in cameras}
    offsets_by_id = {
        int(camera.split("_")[1]): int(offset)
        for camera, offset in (camera_frame_offsets or {}).items()
    }
    accumulators: dict[int, dict[int, tuple[np.ndarray, int]]] = defaultdict(dict)
    # np.loadtxt converts every value in the ~100 MB file even when an
    # evaluation needs only a few cameras and its first N frames. Streaming
    # keeps startup bounded and skips the six unused bbox fields immediately.
    with (scene_dir / "ground_truth.txt").open("r", encoding="utf-8") as source:
        for line in source:
            values = line.split()
            if len(values) < 9:
                continue
            camera_id = int(values[0])
            if camera_id not in camera_ids:
                continue
            logical_frame = int(values[2]) - offsets_by_id.get(camera_id, 0)
            if logical_frame < 0 or (max_frame is not None and logical_frame >= max_frame):
                continue
            person_id = int(values[1])
            xy = np.array([float(values[7]), float(values[8])])
            total, count = accumulators[logical_frame].get(
                person_id, (np.zeros(2, dtype=float), 0)
            )
            accumulators[logical_frame][person_id] = (total + xy, count + 1)
    return {
        frame: {
            person_id: total / count
            for person_id, (total, count) in people.items()
        }
        for frame, people in accumulators.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--max-frames", type=int, default=1200)
    parser.add_argument("--device", default=None, choices=["mps", "cpu", "cuda"])
    parser.add_argument("--cameras", nargs="+", default=None, metavar="CAM")
    parser.add_argument(
        "--match-gate-m",
        type=float,
        default=MATCH_GATE_M,
        help="Maximum world-plane match distance in meters (default: 1.0)",
    )
    parser.add_argument(
        "--track-kf-time-process-noise",
        type=float,
        default=None,
        help="Override timestamped local-track Kalman acceleration variance",
    )
    parser.add_argument(
        "--track-kf-robust-nis-threshold",
        type=float,
        default=None,
        help="Override robust Kalman NIS threshold (0 disables it)",
    )
    parser.add_argument(
        "--switch-details",
        action="store_true",
        help="Print the frame/person/GID transition for every measured ID switch",
    )
    args = parser.parse_args()

    cfg = PipelineConfig(data_root=args.data_root, async_ingest=False)
    available = sorted(p.name for p in cfg.scene_dir.glob("camera_*") if p.is_dir())
    if not available:
        raise SystemExit("No cameras found — run `uv run download-data` first.")
    cfg.cameras = args.cameras or available
    if args.device:
        cfg.device = args.device
    if args.track_kf_time_process_noise is not None:
        cfg.track_kf_time_process_noise = args.track_kf_time_process_noise
    if args.track_kf_robust_nis_threshold is not None:
        cfg.track_kf_robust_nis_threshold = args.track_kf_robust_nis_threshold

    gt_frames = ground_truth_by_frame(
        cfg.scene_dir,
        cfg.cameras,
        cfg.camera_start_frame_offsets,
        max_frame=args.max_frames + 3,
    )
    print(f"Evaluating {len(cfg.cameras)} cameras over {args.max_frames} frames (stride={cfg.stride})...", flush=True)

    predictions: list[tuple[int, dict[int, np.ndarray]]] = []
    last_metrics: dict = {}
    started = time.perf_counter()
    for tick in run_pipeline(cfg, max_frames=args.max_frames):
        predictions.append((
            tick["frame_index"],
            {t.gid: t.world_xy.copy() for t in tick["global_tracks"]},
        ))
        last_metrics = tick["metrics"]
        if len(predictions) % 50 == 0:
            print(f"Evaluated {len(predictions)} ticks in {time.perf_counter() - started:.1f}s", flush=True)
    elapsed = time.perf_counter() - started

    # The GT frame column is 1-based-ish; find the constant offset to our
    # 0-based video frame index that minimizes error (0, 1, or 2).
    best_offset, best_rmse = 0, np.inf
    for offset in (0, 1, 2):
        errors = _match_errors(
            predictions[:100], gt_frames, offset, args.match_gate_m
        )
        if errors and np.sqrt(np.mean(np.square(errors))) < best_rmse:
            best_rmse = np.sqrt(np.mean(np.square(errors)))
            best_offset = offset

    metrics = evaluate_mot(
        predictions,
        gt_frames,
        frame_offset=best_offset,
        match_gate_m=args.match_gate_m,
    )
    switch_events = metrics["switch_events"]

    print(f"\nFrame offset used     : {best_offset}")
    print(f"World match gate      : {args.match_gate_m:.2f} m")
    print(f"TP / FP / FN          : {metrics['TP']} / {metrics['FP']} / {metrics['FN']}")
    print(f"Precision / Recall    : {metrics['precision']:.3f} / {metrics['recall']:.3f}")
    print(f"Position RMSE         : {metrics['position_RMSE_m']:.3f} m")
    print(f"Global ID switches    : {metrics['IDSW']}")
    print("\nIdentity metrics")
    print(f"IDTP / IDFP / IDFN    : {metrics['IDTP']} / {metrics['IDFP']} / {metrics['IDFN']}")
    print(f"IDP / IDR / IDF1      : {metrics['IDP']:.3f} / {metrics['IDR']:.3f} / {metrics['IDF1']:.3f}")
    print("\nCLEAR MOT metrics")
    print(f"MOTA                  : {metrics['MOTA']:.3f}")
    print(f"MOTP similarity       : {metrics['MOTP']:.3f} (higher is better)")
    print(f"MOTP distance         : {metrics['MOTP_distance_m']:.3f} m (lower is better)")
    print("\nHOTA metrics (mean over alpha=0.05:0.05:0.95)")
    print(f"HOTA                  : {metrics['HOTA']:.3f}")
    print(f"DetA / AssA / LocA    : {metrics['DetA']:.3f} / {metrics['AssA']:.3f} / {metrics['LocA']:.3f}")
    print(f"AssPr / AssRe         : {metrics['AssPr']:.3f} / {metrics['AssRe']:.3f}")
    print(f"CPU pipeline          : {len(predictions) / max(elapsed, 1e-6):.3f} ticks/s ({elapsed:.2f}s)")
    if args.switch_details:
        print("\nID switch events (person: previous_frame/GID -> frame/GID):")
        for person_id, previous_frame, frame, previous_gid, gid in switch_events:
            print(
                f"  {person_id}: {previous_frame}/{previous_gid} -> {frame}/{gid}"
            )
        print("\nFinal fusion counters:")
        for name in (
            "global_id_merges",
            "instant_geometry_merges",
            "trajectory_matches_accepted",
            "local_fragments_reconnected",
            "late_reassociations",
            "published_id_preserved_on_merge",
            "same_camera_merge_conflicts",
        ):
            print(f"  {name}: {last_metrics.get(name, 0)}")


def _match_errors(
    predictions, gt_frames, offset: int, match_gate_m: float = MATCH_GATE_M
) -> list[float]:
    errors: list[float] = []
    for frame_index, preds in predictions:
        gt = gt_frames.get(frame_index + offset, {})
        if not gt or not preds:
            continue
        gt_xy = np.array(list(gt.values()))
        pred_xy = np.array(list(preds.values()))
        dist = np.linalg.norm(gt_xy[:, None] - pred_xy[None, :], axis=2)
        for r, c in zip(*linear_sum_assignment(dist)):
            if dist[r, c] < match_gate_m:
                errors.append(float(dist[r, c]))
    return errors


if __name__ == "__main__":
    main()
