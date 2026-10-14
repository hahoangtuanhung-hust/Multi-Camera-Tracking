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
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import (
    DEFAULT_TRACKING_PROFILE,
    PipelineConfig,
    TRACKING_PROFILE_OVERRIDES,
    apply_tracking_profile,
)
from .demo import run_pipeline
from .local_tracker import LOCAL_TRACKERS
from .mot_metrics import (
    evaluate_mot,
    evaluate_mot_slices,
    identity_trajectory_diagnostics,
)
from .reporting import build_run_metadata, configure_determinism

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
    parser.add_argument(
        "--local-tracker",
        choices=LOCAL_TRACKERS,
        default=None,
        help="Per-camera tracker (default: ocsort)",
    )
    parser.add_argument("--cameras", nargs="+", default=None, metavar="CAM")
    parser.add_argument(
        "--profile",
        choices=sorted(TRACKING_PROFILE_OVERRIDES),
        default=DEFAULT_TRACKING_PROFILE,
        help=f"Named identity ablation profile (default: {DEFAULT_TRACKING_PROFILE})",
    )
    parser.add_argument(
        "--match-gate-m",
        type=float,
        default=MATCH_GATE_M,
        help="Maximum world-plane match distance in meters (default: 1.0)",
    )
    parser.add_argument(
        "--expected-people",
        type=int,
        default=None,
        help="Override fixed public-GID capacity (0 disables the cap)",
    )
    parser.add_argument(
        "--metric-window-frames",
        type=int,
        default=1000,
        help="Also report IDF1/MOTA/HOTA per N evaluated ticks (0 disables)",
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
    parser.add_argument(
        "--identity-events",
        type=Path,
        default=None,
        help="Stream bounded identity lifecycle diagnostics as JSONL",
    )
    parser.add_argument(
        "--identity-events-max",
        type=int,
        default=250_000,
        help="Maximum fusion diagnostic events written (0 is unlimited)",
    )
    parser.add_argument(
        "--cumulative-frames",
        type=int,
        nargs="*",
        default=[500, 1000, 1500, 2000, 4000],
        help="Report from-start metrics at these evaluated tick counts",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optionally save the complete reproducible metrics report as JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used by Python, NumPy, OpenCV and PyTorch",
    )
    args = parser.parse_args()
    if args.metric_window_frames < 0:
        parser.error("--metric-window-frames must be >= 0")
    if args.expected_people is not None and args.expected_people < 0:
        parser.error("--expected-people must be >= 0")
    if args.identity_events_max < 0:
        parser.error("--identity-events-max must be >= 0")
    if any(value <= 0 for value in args.cumulative_frames):
        parser.error("--cumulative-frames values must be > 0")
    if args.seed < 0:
        parser.error("--seed must be >= 0")

    configure_determinism(args.seed)

    cfg = PipelineConfig(data_root=args.data_root, async_ingest=False)
    apply_tracking_profile(cfg, args.profile)
    cfg.identity_diagnostics_path = args.identity_events
    cfg.identity_diagnostics_max_events = args.identity_events_max
    if args.expected_people is not None:
        cfg.fusion_fixed_population_size = args.expected_people
    available = sorted(p.name for p in cfg.scene_dir.glob("camera_*") if p.is_dir())
    if not available:
        raise SystemExit("No cameras found — run `uv run download-data` first.")
    cfg.cameras = args.cameras or available
    if args.device:
        cfg.device = args.device
    if args.local_tracker:
        cfg.local_tracker = args.local_tracker
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
    print(
        f"Evaluating {len(cfg.cameras)} cameras over {args.max_frames} frames "
        f"(stride={cfg.stride}, tracker={cfg.local_tracker})...",
        flush=True,
    )

    predictions: list[tuple[int, dict[int, np.ndarray]]] = []
    source_maps: dict[int, dict[int, list[tuple[str, int]]]] = {}
    last_metrics: dict = {}
    started = time.perf_counter()
    for tick in run_pipeline(cfg, max_frames=args.max_frames):
        predictions.append((
            tick["frame_index"],
            {t.gid: t.world_xy.copy() for t in tick["global_tracks"]},
        ))
        sources: defaultdict[int, list[tuple[str, int]]] = defaultdict(list)
        for observation in tick["observations"]:
            key = observation.camera, int(observation.local_id)
            gid = tick["local_to_gid"].get(key)
            if gid is not None:
                sources[int(gid)].append(key)
        source_maps[tick["frame_index"]] = dict(sources)
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
    metric_slices = evaluate_mot_slices(
        predictions,
        gt_frames,
        frame_offset=best_offset,
        match_gate_m=args.match_gate_m,
        cumulative_frames=args.cumulative_frames,
        window_frames=args.metric_window_frames,
    )
    identity_diagnostics = identity_trajectory_diagnostics(
        predictions,
        gt_frames,
        frame_offset=best_offset,
        match_gate_m=args.match_gate_m,
    )
    switch_events = metrics["switch_events"]
    switch_diagnostics = _classify_switch_events(
        switch_events,
        source_maps,
        predictions,
        gt_frames,
        best_offset,
    )
    if args.identity_events is not None:
        with args.identity_events.open("a", encoding="utf-8") as destination:
            for record in switch_diagnostics:
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics.update({
        "frame_offset": best_offset,
        "match_gate_m": args.match_gate_m,
        "max_frames": args.max_frames,
        "evaluated_ticks": len(predictions),
        "elapsed_s": elapsed,
        "ticks_per_s": len(predictions) / max(elapsed, 1e-9),
        "cameras": cfg.cameras,
        "tracking_profile": args.profile,
        "expected_people": cfg.fusion_fixed_population_size,
        **metric_slices,
        "identity_diagnostics": identity_diagnostics,
        "switch_diagnostics": switch_diagnostics,
        "pipeline_metrics": last_metrics,
        "run_metadata": build_run_metadata(cfg, seed=args.seed),
    })

    print(f"\nFrame offset used     : {best_offset}")
    print(f"World match gate      : {args.match_gate_m:.2f} m")
    print(f"TP / FP / FN          : {metrics['TP']} / {metrics['FP']} / {metrics['FN']}")
    print(f"Precision / Recall    : {metrics['precision']:.3f} / {metrics['recall']:.3f}")
    print(f"Position RMSE         : {metrics['position_RMSE_m']:.3f} m")
    print(f"Global ID switches    : {metrics['IDSW']}")
    print(f"Track fragmentations  : {metrics['Frag']}")
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
    print(f"Tracking profile      : {args.profile}")
    print(
        "Public identity slots : "
        f"{cfg.fusion_fixed_population_size or 'unbounded'}"
    )
    print(
        "GID state             : "
        f"public={last_metrics.get('public_identity_slots', 0)}, "
        f"pending={last_metrics.get('pending_provisional_tracks', 0)}, "
        f"allocated={last_metrics.get('public_gid_allocations', 0)}, "
        f"capacity_deferred={last_metrics.get('new_gid_capacity_deferred', 0)}, "
        f"entry_deferred={last_metrics.get('new_gid_entry_deferred', 0)}"
    )
    print(
        "Local lifecycle       : "
        f"created={last_metrics.get('local_tracks_created', 0)}, "
        f"reactivated={last_metrics.get('local_track_reactivations', 0)}, "
        f"tentative_removed={last_metrics.get('local_tracks_removed_tentative', 0)}, "
        f"timeout_removed={last_metrics.get('local_tracks_removed_timeout', 0)}, "
        f"retained={last_metrics.get('retained_local_tracks', 0)}, "
        f"peak={last_metrics.get('peak_retained_local_tracks', 0)}"
    )
    gaps = [event[2] - event[1] for event in switch_events]
    print(
        "Switch gaps           : "
        f"1={sum(gap == 1 for gap in gaps)}, "
        f"2-30={sum(2 <= gap <= 30 for gap in gaps)}, "
        f"31-90={sum(31 <= gap <= 90 for gap in gaps)}, "
        f">90={sum(gap > 90 for gap in gaps)}"
    )
    if metric_slices["metric_windows"]:
        print(
            f"\nWindow metrics ({args.metric_window_frames} evaluated ticks each)"
        )
        print("frames              IDF1   MOTA   HOTA   IDSW  Frag")
        for window in metric_slices["metric_windows"]:
            print(
                f"{window['start_frame']:6d}-{window['end_frame']:6d}  "
                f"{window['IDF1']:.3f}  {window['MOTA']:.3f}  "
                f"{window['HOTA']:.3f}  {window['IDSW']:5d}  "
                f"{window['Frag']:4d}"
            )
    if metric_slices["cumulative_metrics"]:
        print("\nCumulative metrics (from frame 0)")
        print("ticks    IDF1   IDP    IDR    HOTA   AssA   MOTA   IDSW  Frag")
        for cumulative in metric_slices["cumulative_metrics"]:
            print(
                f"{cumulative['ticks']:5d}  {cumulative['IDF1']:.3f}  "
                f"{cumulative['IDP']:.3f}  {cumulative['IDR']:.3f}  "
                f"{cumulative['HOTA']:.3f}  {cumulative['AssA']:.3f}  "
                f"{cumulative['MOTA']:.3f}  {cumulative['IDSW']:5d}  "
                f"{cumulative['Frag']:4d}"
            )
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
            "trajectory_matches_rejected",
            "trajectory_direction_rejected",
            "trajectory_speed_rejected",
            "fusion_motion_gate_accepted",
            "fusion_motion_jump_rejected",
            "local_fragments_reconnected",
            "dormant_reconnects_accepted",
            "anti_pingpong_switches_blocked",
            "new_gid_trajectory_rejected",
            "late_reassociations",
            "published_id_preserved_on_merge",
            "same_camera_merge_conflicts",
        ):
            print(f"  {name}: {last_metrics.get(name, 0)}")
    if switch_diagnostics:
        categories: defaultdict[str, int] = defaultdict(int)
        for record in switch_diagnostics:
            categories[record["classification"]] += 1
        print("\nSwitch diagnostic classes:")
        for name, count in sorted(categories.items()):
            print(f"  {name}: {count}")
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        print(f"\nSaved metrics to {args.metrics_out.resolve()}")


def _classify_switch_events(
    switch_events,
    source_maps: dict[int, dict[int, list[tuple[str, int]]]],
    predictions: list[tuple[int, dict[int, np.ndarray]]],
    gt_frames: dict[int, dict[int, np.ndarray]],
    frame_offset: int,
) -> list[dict]:
    """Attach bounded local/global context to evaluator switch events."""
    predictions_by_frame = dict(predictions)
    records: list[dict] = []
    for person_id, previous_frame, frame, previous_gid, gid in switch_events:
        previous_sources = set(
            source_maps.get(previous_frame, {}).get(previous_gid, [])
        )
        current_sources = set(source_maps.get(frame, {}).get(gid, []))
        shared_local = sorted(previous_sources & current_sources)
        same_camera_fragment = sorted({
            camera
            for camera, local_id in previous_sources
            for current_camera, current_local_id in current_sources
            if camera == current_camera and local_id != current_local_id
        })
        gap = frame - previous_frame
        if shared_local:
            classification = "global_changed_local_stable"
        elif same_camera_fragment:
            classification = "local_fragmentation"
        elif gap == 1:
            classification = "consecutive_cross_camera_or_global"
        elif gap <= 30:
            classification = "short_gap"
        else:
            classification = "long_gap"

        frame_gt = gt_frames.get(frame + frame_offset, {})
        point = frame_gt.get(person_id)
        neighbor_distance = None
        if point is not None:
            distances = [
                float(np.linalg.norm(point - other))
                for other_id, other in frame_gt.items()
                if other_id != person_id
            ]
            if distances:
                neighbor_distance = min(distances)
        previous_predictions = predictions_by_frame.get(previous_frame, {})
        current_predictions = predictions_by_frame.get(frame, {})
        records.append({
            "event": "evaluator_switch",
            "person_id": person_id,
            "previous_frame": previous_frame,
            "frame": frame,
            "gap_frames": gap,
            "previous_gid": previous_gid,
            "gid": gid,
            "classification": classification,
            "previous_sources": sorted(previous_sources),
            "current_sources": sorted(current_sources),
            "shared_local_sources": shared_local,
            "same_camera_fragment": same_camera_fragment,
            "nearest_gt_person_m": neighbor_distance,
            "near_crossing": (
                neighbor_distance is not None and neighbor_distance <= 1.5
            ),
            "previous_gid_present_at_switch": previous_gid in current_predictions,
            "new_gid_present_previous_frame": gid in previous_predictions,
        })
    return records


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
