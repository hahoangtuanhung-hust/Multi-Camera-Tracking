"""Run the demo with repeatable delay/jitter/drop/clock faults on one camera."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from .config import PipelineConfig
from .demo import run_pipeline
from .faults import FaultConfig, FaultInjectedBatchStream
from .timebase import ClockModel
from .video import SyncedVideoReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--delay-camera", required=True)
    parser.add_argument("--delay-s", type=float, default=2.0)
    parser.add_argument("--jitter-s", type=float, default=0.0)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--clock-offset-s", type=float, default=0.0)
    parser.add_argument("--clock-drift-ppm", type=float, default=0.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--save", default=None, metavar="FILE.rrd")
    parser.add_argument("--single-person-mode", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig(data_root=args.data_root, stride=args.stride, device=args.device)
    cfg.single_person_mode = args.single_person_mode
    available = sorted(path.name for path in cfg.scene_dir.glob("camera_*") if path.is_dir())
    if available:
        cfg.cameras = available
    if args.delay_camera not in cfg.cameras:
        raise SystemExit(f"Unknown camera {args.delay_camera!r}; choices: {', '.join(cfg.cameras)}")
    clocks = {
        camera: ClockModel(
            cfg.camera_clock_offsets_s.get(camera, 0.0),
            cfg.camera_clock_scales.get(camera, 1.0),
        )
        for camera in cfg.cameras
    }
    reader = SyncedVideoReader(
        cfg.scene_dir,
        cfg.cameras,
        stride=cfg.stride,
        frame_offsets=cfg.camera_start_frame_offsets,
        clock_models=clocks,
    )
    injected = FaultInjectedBatchStream(
        reader.iter_batches(),
        FaultConfig(
            camera=args.delay_camera,
            delay_s=args.delay_s,
            jitter_s=args.jitter_s,
            drop_rate=args.drop_rate,
            clock_offset_s=args.clock_offset_s,
            clock_drift_ppm=args.clock_drift_ppm,
        ),
    )
    from .viz import RerunLogger

    logger = None
    previous_s = time.perf_counter()
    fps_smoothed: float | None = None
    processed_batches = 0
    last_tick = None
    try:
        for tick in run_pipeline(cfg, max_frames=args.max_frames, batch_iter=injected):
            processed_batches += 1
            last_tick = tick
            if logger is None:
                logger = RerunLogger(cfg, tick["calibrations"], save_path=args.save)
            now_s = time.perf_counter()
            instant_fps = 1.0 / max(now_s - previous_s, 1e-6)
            fps_smoothed = (
                instant_fps if fps_smoothed is None
                else 0.9 * fps_smoothed + 0.1 * instant_fps
            )
            previous_s = now_s
            logger.log_tick(
                tick["frame_index"],
                tick["frames"],
                tick["camera_tracks"],
                tick["local_to_gid"],
                tick["global_tracks"],
                fps=fps_smoothed,
            )
            if tick["frame_index"] % 300 == 0:
                print(
                    f"frame={tick['frame_index']} people={len(tick['global_tracks'])} "
                    f"fault={injected.metrics} fusion={tick['metrics']}"
                )
    finally:
        reader.close()
    if last_tick is not None:
        print(
            f"completed batches={processed_batches} fault={injected.metrics} "
            f"fusion={last_tick['metrics']}"
        )


if __name__ == "__main__":
    main()
