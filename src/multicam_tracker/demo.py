"""Live demo: multi-camera 3D tracking in a Rerun viewer.

Ties the whole pipeline together — one readable loop per tick:

    read synced frames -> detect (batched) -> track per camera
    -> project feet to the floor -> fuse across cameras -> visualize

Run ``uv run demo`` after ``uv run download-data``.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from pathlib import Path

from .bytetrack import ByteTracker
from .calibration import load_calibrations
from .config import PipelineConfig
from .detector import PersonDetector
from .fusion import CameraObservation, GlobalTrackManager, foot_point_xyxy
from .video import SyncedVideoReader


def run_pipeline(cfg: PipelineConfig, max_frames: int | None = None) -> Iterator[dict]:
    """Generator yielding one dict per tick with every intermediate result.

    Shared by the live demo and the offline evaluator, so both always run
    the exact same code path.
    """
    calibrations = load_calibrations(cfg.scene_dir, cfg.cameras)
    reader = SyncedVideoReader(cfg.scene_dir, cfg.cameras, stride=cfg.stride)
    detector = PersonDetector(cfg)
    trackers = {cam: ByteTracker(cfg) for cam in cfg.cameras}
    fusion = GlobalTrackManager(cfg)

    try:
        for frame_index, frames in reader:
            if max_frames is not None and frame_index >= max_frames:
                break

            detections = detector.detect_batch([frames[cam] for cam in cfg.cameras])
            camera_tracks = {
                cam: trackers[cam].update(dets)
                for cam, dets in zip(cfg.cameras, detections)
            }

            observations = [
                CameraObservation(
                    camera=cam,
                    local_id=track.track_id,
                    world_xy=calibrations[cam].image_to_world(
                        foot_point_xyxy(track.xyxy)
                    )[0],
                    bbox_xyxy=track.xyxy,
                    conf=track.conf,
                )
                for cam, tracks in camera_tracks.items()
                for track in tracks
            ]
            global_tracks = fusion.step(observations)

            yield {
                "frame_index": frame_index,
                "frames": frames,
                "camera_tracks": camera_tracks,
                "global_tracks": global_tracks,
                "local_to_gid": dict(fusion.affinity),
                "calibrations": calibrations,
            }
    finally:
        reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--cameras", nargs="+", default=None, metavar="CAM",
                        help="Camera folders to use (default: all downloaded)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Process every Nth frame (default from config: 2)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after this source-frame index")
    parser.add_argument("--device", default=None, choices=["mps", "cpu", "cuda"],
                        help="Inference device (default: mps)")
    parser.add_argument("--save", default=None, metavar="FILE.rrd",
                        help="Record to a .rrd file instead of opening the viewer")
    args = parser.parse_args()

    cfg = PipelineConfig(data_root=args.data_root)
    if args.cameras:
        cfg.cameras = args.cameras
    else:
        available = sorted(p.name for p in cfg.scene_dir.glob("camera_*") if p.is_dir())
        if not available:
            raise SystemExit("No cameras found — run `uv run download-data` first.")
        cfg.cameras = available
    if args.stride:
        cfg.stride = args.stride
    if args.device:
        cfg.device = args.device

    # Imported here, not at module top: evaluate.py and render_preview.py
    # import run_pipeline from this module and should not pay for (or
    # depend on) the visualization stack they never use.
    from .viz import RerunLogger

    logger = None
    fps_smoothed: float | None = None
    t_prev = time.perf_counter()
    print(f"Cameras: {', '.join(cfg.cameras)} | device={cfg.device} stride={cfg.stride}")

    for tick in run_pipeline(cfg, max_frames=args.max_frames):
        if logger is None:  # created lazily so calibration errors surface first
            logger = RerunLogger(cfg, tick["calibrations"], save_path=args.save)

        now = time.perf_counter()
        instant = 1.0 / max(now - t_prev, 1e-6)
        fps_smoothed = instant if fps_smoothed is None else 0.9 * fps_smoothed + 0.1 * instant
        t_prev = now

        logger.log_tick(
            tick["frame_index"], tick["frames"], tick["camera_tracks"],
            tick["local_to_gid"], tick["global_tracks"], fps=fps_smoothed,
        )
        if tick["frame_index"] % 300 == 0:
            print(f"frame {tick['frame_index']:5d} | "
                  f"{len(tick['global_tracks'])} people tracked | "
                  f"{fps_smoothed:.1f} ticks/s")

    if args.save:
        print(f"Recording saved to {args.save}")


if __name__ == "__main__":
    main()
