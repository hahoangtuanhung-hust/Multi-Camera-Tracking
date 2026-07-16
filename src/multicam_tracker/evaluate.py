"""Evaluate fused global tracks against the dataset's ground truth.

Runs the exact same pipeline as the live demo (no visualization), then
per frame Hungarian-matches predicted world positions to ground-truth
world positions within a 1 m gate and reports:

- **precision / recall** — how many predicted tracks are real people, and
  how many real people we track (a simplified MODA-style measure)
- **position RMSE** — world-plane localization error over matched pairs
- **ID switches** — how often the global ID assigned to a ground-truth
  person changes; the headline metric for cross-camera fusion quality

Run: ``uv run evaluate --max-frames 1200``
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .calibration import load_ground_truth
from .config import PipelineConfig
from .demo import run_pipeline

MATCH_GATE_M = 1.0


def ground_truth_by_frame(
    scene_dir: Path, cameras: list[str]
) -> dict[int, dict[int, np.ndarray]]:
    """frame -> {person_id: world_xy}, deduplicated across cameras.

    Only people visible in at least one of the *evaluated* cameras count —
    the ground truth covers all 10 cameras, but the pipeline can only see
    the ones that were downloaded.
    """
    camera_ids = {float(cam.split("_")[1]) for cam in cameras}
    gt = load_ground_truth(scene_dir)
    by_frame: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for camera_id, person_id, frame, *_bbox, wx, wy in gt:
        if camera_id in camera_ids:
            by_frame[int(frame)][int(person_id)] = np.array([wx, wy])
    return by_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--max-frames", type=int, default=1200)
    parser.add_argument("--device", default=None, choices=["mps", "cpu", "cuda"])
    args = parser.parse_args()

    cfg = PipelineConfig(data_root=args.data_root)
    available = sorted(p.name for p in cfg.scene_dir.glob("camera_*") if p.is_dir())
    if not available:
        raise SystemExit("No cameras found — run `uv run download-data` first.")
    cfg.cameras = available
    if args.device:
        cfg.device = args.device

    gt_frames = ground_truth_by_frame(cfg.scene_dir, cfg.cameras)
    print(f"Evaluating {len(available)} cameras over {args.max_frames} frames...")

    predictions: list[tuple[int, dict[int, np.ndarray]]] = []
    for tick in run_pipeline(cfg, max_frames=args.max_frames):
        predictions.append((
            tick["frame_index"],
            {t.gid: t.world_xy.copy() for t in tick["global_tracks"]},
        ))

    # The GT frame column is 1-based-ish; find the constant offset to our
    # 0-based video frame index that minimizes error (0, 1, or 2).
    best_offset, best_rmse = 0, np.inf
    for offset in (0, 1, 2):
        errors = _match_errors(predictions[:100], gt_frames, offset)
        if errors and np.sqrt(np.mean(np.square(errors))) < best_rmse:
            best_rmse = np.sqrt(np.mean(np.square(errors)))
            best_offset = offset

    tp, fp, fn = 0, 0, 0
    squared_errors: list[float] = []
    gt_id_history: dict[int, list[int]] = defaultdict(list)

    for frame_index, preds in predictions:
        gt = gt_frames.get(frame_index + best_offset, {})
        gt_ids, gt_xy = list(gt), np.array([gt[i] for i in gt]).reshape(-1, 2)
        pred_ids, pred_xy = list(preds), np.array([preds[i] for i in preds]).reshape(-1, 2)

        matched_gt, matched_pred = set(), set()
        if len(gt_ids) and len(pred_ids):
            dist = np.linalg.norm(gt_xy[:, None] - pred_xy[None, :], axis=2)
            for r, c in zip(*linear_sum_assignment(dist)):
                if dist[r, c] < MATCH_GATE_M:
                    matched_gt.add(gt_ids[r])
                    matched_pred.add(pred_ids[c])
                    squared_errors.append(float(dist[r, c] ** 2))
                    gt_id_history[gt_ids[r]].append(pred_ids[c])
        tp += len(matched_gt)
        fn += len(gt_ids) - len(matched_gt)
        fp += len(pred_ids) - len(matched_pred)

    id_switches = sum(
        sum(1 for a, b in zip(gids, gids[1:]) if a != b)
        for gids in gt_id_history.values()
    )

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    rmse = float(np.sqrt(np.mean(squared_errors))) if squared_errors else float("nan")

    print(f"\nFrame offset used     : {best_offset}")
    print(f"Matches (TP)          : {tp}")
    print(f"Precision             : {precision:.3f}")
    print(f"Recall                : {recall:.3f}")
    print(f"Position RMSE         : {rmse:.2f} m")
    print(f"Global ID switches    : {id_switches} "
          f"(over {len(gt_id_history)} people, {len(predictions)} ticks)")


def _match_errors(predictions, gt_frames, offset: int) -> list[float]:
    errors: list[float] = []
    for frame_index, preds in predictions:
        gt = gt_frames.get(frame_index + offset, {})
        if not gt or not preds:
            continue
        gt_xy = np.array(list(gt.values()))
        pred_xy = np.array(list(preds.values()))
        dist = np.linalg.norm(gt_xy[:, None] - pred_xy[None, :], axis=2)
        for r, c in zip(*linear_sum_assignment(dist)):
            if dist[r, c] < MATCH_GATE_M:
                errors.append(float(dist[r, c]))
    return errors


if __name__ == "__main__":
    main()
