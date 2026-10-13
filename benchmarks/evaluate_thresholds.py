"""Replay one detector pass through several trajectory-gate configurations."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from multicam_tracker.config import PipelineConfig
from multicam_tracker.demo import run_pipeline
from multicam_tracker.evaluate import MATCH_GATE_M, ground_truth_by_frame
from multicam_tracker.fusion import GlobalTrackManager
from multicam_tracker.mot_metrics import evaluate_mot


def _score(predictions, gt_frames):
    best_offset, best_rmse = 0, float("inf")
    for offset in (0, 1, 2):
        errors = []
        for frame_index, predictions_at_frame in predictions[:100]:
            gt = gt_frames.get(frame_index + offset, {})
            if not gt or not predictions_at_frame:
                continue
            gt_xy = np.array(list(gt.values()))
            pred_xy = np.array(list(predictions_at_frame.values()))
            distance = np.linalg.norm(gt_xy[:, None] - pred_xy[None, :], axis=2)
            for row, column in zip(*linear_sum_assignment(distance)):
                if distance[row, column] < MATCH_GATE_M:
                    errors.append(float(distance[row, column]))
        if errors and np.sqrt(np.mean(np.square(errors))) < best_rmse:
            best_offset = offset
            best_rmse = float(np.sqrt(np.mean(np.square(errors))))

    metrics = evaluate_mot(
        predictions,
        gt_frames,
        frame_offset=best_offset,
        match_gate_m=MATCH_GATE_M,
    )
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "rmse": metrics["position_RMSE_m"],
        "switches": metrics["IDSW"],
        "idf1": metrics["IDF1"],
        "mota": metrics["MOTA"],
        "hota": metrics["HOTA"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--cameras", nargs="+", default=None, metavar="CAM")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        help="Optional subset of fusion profiles to replay",
    )
    args = parser.parse_args()

    detector_cfg = PipelineConfig(
        data_root=args.data_root,
        device=args.device,
        timestamp_fusion_enabled=False,
        async_ingest=False,
        # Keep local detections/track IDs identical while replaying only the
        # global-fusion feature flags below.
        track_motion_association_enabled=False,
    )
    available = sorted(
        path.name for path in detector_cfg.scene_dir.glob("camera_*") if path.is_dir()
    )
    detector_cfg.cameras = args.cameras or available
    candidates = {
        "pre-identity-v2": dict(
            fusion_provisional_buffer_enabled=False,
            fusion_batch_reconnect_enabled=False,
            fusion_adaptive_hysteresis_enabled=False,
            fusion_duplicate_suppression_enabled=False,
        ),
        "batch-only": dict(
            fusion_provisional_buffer_enabled=True,
            fusion_provisional_min_age_s=0.0,
            fusion_batch_reconnect_enabled=True,
            fusion_adaptive_hysteresis_enabled=False,
            fusion_duplicate_suppression_enabled=False,
        ),
        "adaptive-only": dict(
            fusion_provisional_buffer_enabled=False,
            fusion_batch_reconnect_enabled=False,
            fusion_adaptive_hysteresis_enabled=True,
            fusion_duplicate_suppression_enabled=False,
        ),
        "duplicate-only": dict(
            fusion_provisional_buffer_enabled=False,
            fusion_batch_reconnect_enabled=False,
            fusion_adaptive_hysteresis_enabled=False,
            fusion_duplicate_suppression_enabled=True,
        ),
        "fusion-identity-safe": dict(
            fusion_provisional_buffer_enabled=True,
            fusion_provisional_min_age_s=0.0,
            fusion_batch_reconnect_enabled=True,
            fusion_adaptive_hysteresis_enabled=True,
            fusion_duplicate_suppression_enabled=True,
        ),
        "stable-default": dict(),
        "sequential-reconnect": dict(
            fusion_batch_reconnect_enabled=False,
        ),
        "hard-published-motion": dict(
            fusion_adaptive_hysteresis_enabled=False,
        ),
        "no-duplicate-suppression": dict(
            fusion_duplicate_suppression_enabled=False,
        ),
        "provisional-020": dict(
            fusion_provisional_min_age_s=0.20,
        ),
        "no-hysteresis": dict(
            fusion_instant_confirmations=1,
            fusion_published_merge_confirmations=1,
            fusion_local_reconnect_confirmations=1,
        ),
        "crossing-only": dict(
            fusion_instant_confirmations=2,
            fusion_published_merge_confirmations=1,
            fusion_local_reconnect_confirmations=1,
        ),
        "published-2": dict(
            fusion_instant_confirmations=1,
            fusion_published_merge_confirmations=2,
            fusion_local_reconnect_confirmations=1,
        ),
        "fragment-2": dict(
            fusion_instant_confirmations=1,
            fusion_published_merge_confirmations=1,
            fusion_local_reconnect_confirmations=2,
        ),
        "duplicate-045": dict(
            fusion_instant_confirmations=1,
            fusion_published_merge_confirmations=1,
            fusion_local_reconnect_confirmations=2,
            fusion_duplicate_merge_m=0.45,
            fusion_duplicate_confirmations=3,
        ),
        "minhits-4": dict(
            fusion_instant_confirmations=1,
            fusion_published_merge_confirmations=1,
            fusion_local_reconnect_confirmations=2,
            fusion_min_hits=4,
        ),
        "minhits-5": dict(
            fusion_instant_confirmations=1,
            fusion_published_merge_confirmations=1,
            fusion_local_reconnect_confirmations=2,
            fusion_min_hits=5,
        ),
        "hysteresis-2": dict(
            fusion_instant_confirmations=2,
            fusion_published_merge_confirmations=2,
            fusion_local_reconnect_confirmations=2,
        ),
        "no-duplicate": dict(
            fusion_duplicate_merge_m=0.0,
        ),
        "short-retention": dict(
            fusion_identity_retention_s=5.0,
            fusion_local_reconnect_long_window_s=3.0,
        ),
        "position": dict(
            fusion_crossing_guard_m=0.0,
            fusion_instant_velocity_weight_s=0.0,
            fusion_instant_opposite_direction_cosine=-1.1,
            fusion_local_reconnect_velocity_weight_s=0.0,
        ),
        "veto-only": dict(
            fusion_instant_velocity_weight_s=0.0,
            fusion_local_reconnect_velocity_weight_s=0.25,
        ),
        "motion-010": dict(
            fusion_instant_velocity_weight_s=0.10,
            fusion_local_reconnect_velocity_weight_s=0.25,
        ),
        "motion-015": dict(
            fusion_instant_velocity_weight_s=0.15,
            fusion_local_reconnect_velocity_weight_s=0.25,
        ),
        "confirm-2": dict(
            fusion_instant_velocity_weight_s=0.10,
            fusion_local_reconnect_velocity_weight_s=0.25,
            fusion_instant_confirmations=2,
        ),
        "confirm-3": dict(
            fusion_instant_velocity_weight_s=0.10,
            fusion_local_reconnect_velocity_weight_s=0.25,
            fusion_instant_confirmations=3,
        ),
        "no-reconnect-motion": dict(
            fusion_instant_velocity_weight_s=0.10,
            fusion_local_reconnect_velocity_weight_s=0.0,
        ),
    }
    if args.profiles:
        unknown = sorted(set(args.profiles) - set(candidates))
        if unknown:
            raise SystemExit(f"Unknown profiles: {', '.join(unknown)}")
        candidates = {name: candidates[name] for name in args.profiles}
    managers = {
        name: GlobalTrackManager(replace(
            detector_cfg,
            timestamp_fusion_enabled=True,
            **overrides,
        ))
        for name, overrides in candidates.items()
    }
    predictions = {name: [] for name in candidates}
    predictions["legacy"] = []
    for tick in run_pipeline(detector_cfg, max_frames=args.max_frames):
        predictions["legacy"].append((
            tick["frame_index"],
            {track.gid: track.world_xy.copy() for track in tick["global_tracks"]},
        ))
        event_ts = tick["capture_ts_s"]
        arrival_ts = max(packet.arrival_ts_s for packet in tick["packets"].values())
        for name, manager in managers.items():
            tracks = manager.step(tick["observations"], event_ts, arrival_ts)
            predictions[name].append((
                tick["frame_index"],
                {track.gid: track.world_xy.copy() for track in tracks},
            ))

    gt_frames = ground_truth_by_frame(
        detector_cfg.scene_dir,
        detector_cfg.cameras,
        detector_cfg.camera_start_frame_offsets,
        max_frame=args.max_frames + 3,
    )
    legacy_score = _score(predictions["legacy"], gt_frames)
    print(
        f"{'legacy':16s} P={legacy_score['precision']:.3f} "
        f"R={legacy_score['recall']:.3f} RMSE={legacy_score['rmse']:.3f} "
        f"IDSW={legacy_score['switches']:4d} IDF1={legacy_score['idf1']:.3f} "
        f"MOTA={legacy_score['mota']:.3f} HOTA={legacy_score['hota']:.3f} "
        "merges=n/a"
    )
    for name, manager in managers.items():
        score = _score(predictions[name], gt_frames)
        print(
            f"{name:16s} P={score['precision']:.3f} R={score['recall']:.3f} "
            f"RMSE={score['rmse']:.3f} IDSW={score['switches']:4d} "
            f"IDF1={score['idf1']:.3f} MOTA={score['mota']:.3f} "
            f"HOTA={score['hota']:.3f} "
            f"merges={manager.metrics['global_id_merges']} "
            f"preserved={manager.metrics['published_id_preserved_on_merge']}"
        )


if __name__ == "__main__":
    main()
