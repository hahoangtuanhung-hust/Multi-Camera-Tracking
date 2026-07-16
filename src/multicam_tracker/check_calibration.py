"""Sanity-check the dataset calibration against the ground truth.

Two checks per camera, using ground-truth annotations only (no ML involved):

1. **Numeric round trip** — take each annotated bbox foot point, map it to
   the world floor with the inverse homography, and compare against the
   annotated (worldX, worldY). Reports the median/95th percentile error in
   meters. Expect well under half a meter.
2. **Visual overlay** (--save-images) — draw the annotated world positions
   re-projected into the image on top of an extracted video frame. The dots
   should land at people's feet.

Run after `download-data`:  ``uv run check-calibration``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .calibration import foot_point, load_calibrations, load_ground_truth
from .config import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--frame", type=int, default=100, help="GT frame to check")
    parser.add_argument(
        "--save-images", action="store_true",
        help="Write overlay images to ./calibration_check/ for visual inspection",
    )
    args = parser.parse_args()

    cfg = PipelineConfig(data_root=args.data_root)
    available = sorted(p.name for p in cfg.scene_dir.glob("camera_*") if p.is_dir())
    calibrations = load_calibrations(cfg.scene_dir, available)
    gt = load_ground_truth(cfg.scene_dir)
    print(f"Ground truth: {len(gt):,} rows, "
          f"{len(np.unique(gt[:, 1]))} people, "
          f"frames {int(gt[:, 2].min())}..{int(gt[:, 2].max())}")

    out_dir = Path("calibration_check")
    if args.save_images:
        out_dir.mkdir(exist_ok=True)

    all_ok = True
    for cam in available:
        calib = calibrations[cam]
        cam_id = int(cam.split("_")[1])
        rows = gt[(gt[:, 0] == cam_id) & (gt[:, 2] == args.frame)]
        if len(rows) == 0:
            print(f"{cam}: no GT rows at frame {args.frame}, skipping")
            continue

        feet_px = foot_point(rows[:, 3:7])
        world_est = calib.image_to_world(feet_px)
        world_gt = rows[:, 7:9]
        err = np.linalg.norm(world_est - world_gt, axis=1)
        median, p95 = np.median(err), np.percentile(err, 95)
        status = "OK " if median < 0.5 else "FAIL"
        if median >= 0.5:
            all_ok = False
        print(f"{cam}: {len(rows):2d} people | foot->world error "
              f"median {median:.2f} m, p95 {p95:.2f} m  [{status}]")

        if args.save_images:
            video = cv2.VideoCapture(str(cfg.scene_dir / cam / "video.mp4"))
            video.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
            ok, frame = video.read()
            video.release()
            if not ok:
                print(f"{cam}: could not read video frame {args.frame}")
                continue
            reprojected = calib.world_to_image(world_gt)
            for (u, v), (fx, fy) in zip(reprojected, feet_px):
                cv2.circle(frame, (int(u), int(v)), 12, (0, 255, 0), 3)   # GT world -> image
                cv2.drawMarker(frame, (int(fx), int(fy)), (0, 0, 255),   # bbox foot point
                               cv2.MARKER_CROSS, 20, 3)
            path = out_dir / f"{cam}_frame{args.frame}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"      overlay saved to {path} (green circles = GT, red crosses = bbox feet)")

    print("\nCalibration check", "PASSED" if all_ok else "FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
