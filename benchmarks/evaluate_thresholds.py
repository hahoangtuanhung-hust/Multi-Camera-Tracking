"""Replay one detector pass through several trajectory-gate configurations."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from multicam_tracker.config import PipelineConfig
from multicam_tracker.demo import run_pipeline
from multicam_tracker.evaluate import MATCH_GATE_M, ground_truth_by_frame
from multicam_tracker.fusion import GlobalTrackManager


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

    tp = fp = fn = 0
    squared_errors: list[float] = []
    history: defaultdict[int, list[int]] = defaultdict(list)
    for frame_index, predictions_at_frame in predictions:
        gt = gt_frames.get(frame_index + best_offset, {})
        gt_ids = list(gt)
        pred_ids = list(predictions_at_frame)
        gt_xy = np.array([gt[key] for key in gt_ids]).reshape(-1, 2)
        pred_xy = np.array([predictions_at_frame[key] for key in pred_ids]).reshape(-1, 2)
        matched_gt, matched_pred = set(), set()
        if gt_ids and pred_ids:
            distance = np.linalg.norm(gt_xy[:, None] - pred_xy[None, :], axis=2)
            for row, column in zip(*linear_sum_assignment(distance)):
                if distance[row, column] < MATCH_GATE_M:
                    matched_gt.add(gt_ids[row])
                    matched_pred.add(pred_ids[column])
                    squared_errors.append(float(distance[row, column] ** 2))
                    history[gt_ids[row]].append(pred_ids[column])
        tp += len(matched_gt)
        fp += len(pred_ids) - len(matched_pred)
        fn += len(gt_ids) - len(matched_gt)
    switches = sum(
        sum(first != second for first, second in zip(gids, gids[1:]))
        for gids in history.values()
    )
    return {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "rmse": float(np.sqrt(np.mean(squared_errors))) if squared_errors else float("nan"),
        "switches": switches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    detector_cfg = PipelineConfig(
        data_root=args.data_root,
        device=args.device,
        timestamp_fusion_enabled=False,
        async_ingest=False,
    )
    detector_cfg.cameras = sorted(
        path.name for path in detector_cfg.scene_dir.glob("camera_*") if path.is_dir()
    )
    candidates = {
        "strict": (0.65, 0.60, 0.75, 0.20),
        "balanced": (0.85, 0.90, 0.65, 0.20),
        "robust": (1.00, 1.20, 0.50, 0.20),
        "wide": (1.20, 1.50, 0.30, 0.25),
    }
    managers = {
        name: GlobalTrackManager(replace(
            detector_cfg,
            timestamp_fusion_enabled=True,
            tracklet_position_gate_m=position,
            tracklet_velocity_gate_mps=velocity,
            tracklet_direction_cosine=direction,
            tracklet_assignment_margin=margin,
        ))
        for name, (position, velocity, direction, margin) in candidates.items()
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
    )
    legacy_score = _score(predictions["legacy"], gt_frames)
    print(
        f"{'legacy':8s} P={legacy_score['precision']:.3f} "
        f"R={legacy_score['recall']:.3f} RMSE={legacy_score['rmse']:.3f} "
        f"IDSW={legacy_score['switches']:4d} merges=n/a"
    )
    for name, manager in managers.items():
        score = _score(predictions[name], gt_frames)
        print(
            f"{name:8s} P={score['precision']:.3f} R={score['recall']:.3f} "
            f"RMSE={score['rmse']:.3f} IDSW={score['switches']:4d} "
            f"merges={manager.metrics['global_id_merges']}"
        )


if __name__ == "__main__":
    main()
