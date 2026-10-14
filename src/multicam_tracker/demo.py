"""Live demo: multi-camera 3D tracking in a Rerun viewer.

Ties the whole pipeline together — one readable loop per tick:

    read timestamped frames -> detect (dynamic batch) -> track per camera
    -> project feet to the floor -> fuse across cameras -> visualize

Run ``uv run demo`` after ``uv run download-data``.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from time import monotonic

import numpy as np

from .calibration import load_calibrations
from .config import (
    DEFAULT_TRACKING_PROFILE,
    PipelineConfig,
    TRACKING_PROFILE_OVERRIDES,
    apply_tracking_profile,
)
from .detector import PersonDetector
from .fusion import CameraObservation, GlobalTrackManager, foot_point_xyxy
from .local_tracker import LOCAL_TRACKERS, create_local_tracker
from .spatiotemporal import SlidingWindowTrajectorySmoother
from .timebase import ClockModel, FrameBatch
from .video import AsyncVideoReader, SyncedVideoReader


def _bbox_near_boundary(
    bbox_xyxy: np.ndarray,
    width: int,
    height: int,
    margin_ratio: float,
) -> bool:
    """Whether a detection touches a configured image entry/exit band."""
    if margin_ratio <= 0:
        return False
    x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=float)
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio
    return bool(
        x1 <= margin_x
        or y1 <= margin_y
        or x2 >= width - margin_x
        or y2 >= height - margin_y
    )


def run_pipeline(
    cfg: PipelineConfig,
    max_frames: int | None = None,
    batch_iter: Iterable[FrameBatch] | None = None,
    detector: PersonDetector | None = None,
) -> Iterator[dict]:
    """Generator yielding one dict per tick with every intermediate result.

    Shared by the live demo and the offline evaluator, so both always run
    the exact same code path.
    """
    if cfg.stride < 1 or not cfg.cameras:
        raise ValueError("At least one camera and stride >= 1 are required")
    # Initialize before starting capture threads; model load failure must not
    # leak live decoders or leave bounded queues waiting forever.
    calibrations = load_calibrations(cfg.scene_dir, cfg.cameras)
    detector = detector or PersonDetector(cfg)
    clock_models = {
        camera: ClockModel(
            offset_s=cfg.camera_clock_offsets_s.get(camera, 0.0),
            scale=cfg.camera_clock_scales.get(camera, 1.0),
        )
        for camera in cfg.cameras
    }
    reader = None
    if batch_iter is None:
        if cfg.async_ingest:
            sources = {
                camera: cfg.scene_dir / camera / "video.mp4" for camera in cfg.cameras
            }
            reader = AsyncVideoReader(
                sources,
                stride=cfg.stride,
                queue_size=cfg.ingest_queue_size,
                microbatch_ms=cfg.microbatch_ms,
                clock_models=clock_models,
                frame_offsets=cfg.camera_start_frame_offsets,
                decoder_threads=cfg.decoder_threads,
            )
        else:
            reader = SyncedVideoReader(
                cfg.scene_dir,
                cfg.cameras,
                stride=cfg.stride,
                frame_offsets=cfg.camera_start_frame_offsets,
                clock_models=clock_models,
                decoder_threads=cfg.decoder_threads,
            )
        batch_iter = reader.iter_batches()
    trackers = {cam: create_local_tracker(cfg) for cam in cfg.cameras}
    fusion = GlobalTrackManager(cfg)
    trajectory_smoother = (
        SlidingWindowTrajectorySmoother(
            window_s=cfg.fusion_trajectory_window_s,
            recency_half_life_s=cfg.fusion_trajectory_recency_half_life_s,
            outlier_gate_m=cfg.fusion_trajectory_outlier_gate_m,
            max_speed_mps=cfg.fusion_max_speed_mps,
            max_acceleration_mps2=cfg.fusion_max_acceleration_mps2,
            jump_window_s=cfg.fusion_jump_window_s,
            max_jump_m=cfg.fusion_max_jump_m,
        )
        if cfg.fusion_trajectory_smoothing_enabled
        else None
    )
    world_motion: dict[tuple[str, int], tuple[float, np.ndarray, np.ndarray]] = {}
    last_capture_ts: dict[str, float] = {}
    out_of_order_packets_dropped: dict[str, int] = {camera: 0 for camera in cfg.cameras}
    processed_ticks = 0
    detector_frames = 0
    detector_detections = 0
    detector_high_confidence_detections = 0
    detector_low_confidence_candidates = 0
    detector_empty_frames = 0
    limited_cameras: set[str] = set()
    event_watermark = float("-inf")

    try:
        for batch in batch_iter:
            if not batch.packets:
                continue
            # Preserve the public logical frame index even when individual
            # sources have configured start-frame trims.
            cameras: list[str] = []
            for camera in cfg.cameras:
                if camera not in batch.packets:
                    continue
                if max_frames is not None and batch.packets[camera].sequence * cfg.stride >= max_frames:
                    limited_cameras.add(camera)
                    continue
                capture_ts = batch.packets[camera].capture_ts_s
                if not np.isfinite(capture_ts):
                    raise ValueError(f"Non-finite capture timestamp for {camera}")
                previous_ts = last_capture_ts.get(camera)
                if previous_ts is not None and capture_ts <= previous_ts + 1e-9:
                    out_of_order_packets_dropped[camera] += 1
                    continue
                last_capture_ts[camera] = capture_ts
                cameras.append(camera)
            if not cameras:
                if limited_cameras.issuperset(cfg.cameras):
                    break
                continue
            frame_index = max(batch.packets[camera].sequence * cfg.stride for camera in cameras)
            frames = {camera: batch.packets[camera].image for camera in cameras}
            # Detection and local tracking run for every emitted frame batch.
            # Tracklet history is metadata-only and never gates this hot path.
            detections = detector.detect_batch([frames[camera] for camera in cameras])
            if len(detections) != len(cameras):
                raise ValueError("Detector must return exactly one detection array per camera")
            detector_frames += len(detections)
            for camera_detections in detections:
                camera_detections = np.asarray(camera_detections, dtype=float).reshape((-1, 5))
                count = len(camera_detections)
                detector_detections += count
                detector_empty_frames += int(count == 0)
                if count:
                    high_count = int(np.count_nonzero(
                        camera_detections[:, 4] >= cfg.track_high_conf
                    ))
                    detector_high_confidence_detections += high_count
                    detector_low_confidence_candidates += count - high_count
            processed_ticks += 1
            camera_tracks = {
                camera: trackers[camera].update(
                    dets, timestamp_s=batch.packets[camera].capture_ts_s
                )
                for camera, dets in zip(cameras, detections)
            }
            processing_ts_s = monotonic()
            for camera in cameras:
                packet = batch.packets[camera]
                for event in trackers[camera].last_events:
                    event = dict(event)
                    event_name = event.pop("event")
                    local_id = int(event["local_track_id"])
                    mapped_gid = fusion.affinity.get((camera, local_id))
                    if mapped_gid is not None:
                        mapped_gid = fusion.resolve_gid(mapped_gid)
                    # Compose one mapping before expansion so a future local
                    # diagnostic field cannot cause a duplicate-key call
                    # failure. Pipeline-owned timing/context wins.
                    event.update({
                        "event_ts_s": packet.capture_ts_s,
                        "arrival_ts_s": packet.arrival_ts_s,
                        "processing_ts_s": processing_ts_s,
                        "camera_id": camera,
                        "frame_id": packet.sequence * cfg.stride,
                        "source_frame_index": packet.source_frame_index,
                        "detector_ran": True,
                        "global_id": mapped_gid,
                    })
                    fusion.record_diagnostic_event(event_name, **event)

            observations: list[CameraObservation] = []
            for camera, tracks in camera_tracks.items():
                packet = batch.packets[camera]
                for track in tracks:
                    raw_world_xy = calibrations[camera].image_to_world(
                        foot_point_xyxy(track.xyxy)
                    )[0]
                    if not np.all(np.isfinite(raw_world_xy)):
                        continue
                    key = camera, track.track_id
                    if trajectory_smoother is not None:
                        smoothed = trajectory_smoother.update(
                            key,
                            packet.capture_ts_s,
                            raw_world_xy,
                            confidence=float(track.conf),
                        )
                        if not smoothed.accepted:
                            fusion.record_diagnostic_event(
                                "trajectory_observation_rejected",
                                event_ts_s=packet.capture_ts_s,
                                arrival_ts_s=packet.arrival_ts_s,
                                camera_id=camera,
                                local_track_id=int(track.track_id),
                                raw_xy=smoothed.raw_xy,
                                predicted_xy=smoothed.predicted_xy,
                                residual_m=smoothed.residual_m,
                                allowed_residual_m=smoothed.allowed_residual_m,
                            )
                            continue
                        world_xy = smoothed.smoothed_xy
                        velocity = smoothed.velocity_xy
                    else:
                        world_xy = raw_world_xy
                        velocity = None
                        previous = world_motion.get(key)
                        if previous is not None:
                            previous_ts, previous_xy, previous_velocity = previous
                            dt = packet.capture_ts_s - previous_ts
                            if dt > 1e-6:
                                raw_velocity = (world_xy - previous_xy) / dt
                                velocity = (
                                    0.5 * raw_velocity
                                    + 0.5 * previous_velocity
                                )
                        if velocity is None:
                            velocity = world_xy * 0.0
                        world_motion[key] = (
                            packet.capture_ts_s,
                            world_xy.copy(),
                            velocity.copy(),
                        )
                    observations.append(CameraObservation(
                        camera=camera,
                        local_id=track.track_id,
                        world_xy=world_xy,
                        bbox_xyxy=track.xyxy,
                        conf=track.conf,
                        capture_ts_s=packet.capture_ts_s,
                        world_velocity=velocity,
                        arrival_ts_s=packet.arrival_ts_s,
                        near_boundary=_bbox_near_boundary(
                            track.xyxy,
                            calibrations[camera].width,
                            calibrations[camera].height,
                            cfg.fusion_boundary_margin_ratio,
                        ),
                    ))
            event_watermark = max(event_watermark, *(batch.packets[camera].capture_ts_s for camera in cameras))
            if trajectory_smoother is not None:
                trajectory_smoother.prune(event_watermark)
            cutoff = event_watermark - cfg.tracklet_history_s
            for key, (timestamp, _, _) in list(world_motion.items()):
                if timestamp < cutoff:
                    del world_motion[key]
            arrival_watermark = max(batch.packets[camera].arrival_ts_s for camera in cameras)
            global_tracks = fusion.step(
                observations,
                timestamp_s=event_watermark,
                arrival_timestamp_s=arrival_watermark,
            )

            yield {
                "tick_index": processed_ticks,
                "frame_index": frame_index,
                "capture_ts_s": event_watermark,
                "packets": {camera: batch.packets[camera] for camera in cameras},
                "frames": frames,
                "camera_tracks": camera_tracks,
                "observations": observations,
                "global_tracks": global_tracks,
                # Pending negative IDs are internal decision-buffer state and
                # must never leak into Rerun, MP4 labels, or API consumers.
                "local_to_gid": fusion.public_affinity_snapshot(),
                "gid_aliases": dict(fusion.aliases),
                "calibrations": calibrations,
                "metrics": {
                    **fusion.metrics_snapshot(),
                    "local_tracks_created": sum(
                        tracker.metrics["tracks_created"]
                        for tracker in trackers.values()
                    ),
                    "local_track_reactivations": sum(
                        tracker.metrics["reactivations"]
                        for tracker in trackers.values()
                    ),
                    "local_tracks_removed_tentative": sum(
                        tracker.metrics["tracks_removed_tentative"]
                        for tracker in trackers.values()
                    ),
                    "local_tracks_removed_timeout": sum(
                        tracker.metrics["tracks_removed_timeout"]
                        for tracker in trackers.values()
                    ),
                    "local_high_confidence_matches": sum(
                        tracker.metrics["high_confidence_matches"]
                        for tracker in trackers.values()
                    ),
                    "local_low_confidence_matches": sum(
                        tracker.metrics["low_confidence_matches"]
                        for tracker in trackers.values()
                    ),
                    "local_ocm_matches": sum(
                        tracker.metrics["ocm_matches"]
                        for tracker in trackers.values()
                    ),
                    "local_ocr_matches": sum(
                        tracker.metrics["ocr_matches"]
                        for tracker in trackers.values()
                    ),
                    "local_oru_reupdates": sum(
                        tracker.metrics["oru_reupdates"]
                        for tracker in trackers.values()
                    ),
                    "local_byte_extension_matches": sum(
                        tracker.metrics["byte_extension_matches"]
                        for tracker in trackers.values()
                    ),
                    "local_ambiguous_detections_suppressed": sum(
                        tracker.metrics.get("ambiguous_detections_suppressed", 0)
                        for tracker in trackers.values()
                    ),
                    "retained_local_tracks": sum(
                        len(tracker.tracks) for tracker in trackers.values()
                    ),
                    "peak_retained_local_tracks": sum(
                        tracker.metrics["peak_retained_tracks"]
                        for tracker in trackers.values()
                    ),
                    "detector_batches": processed_ticks,
                    "detector_frames": detector_frames,
                    "detector_detections": detector_detections,
                    "detector_high_confidence_detections": (
                        detector_high_confidence_detections
                    ),
                    "detector_low_confidence_candidates": (
                        detector_low_confidence_candidates
                    ),
                    "detector_empty_frames": detector_empty_frames,
                    "ingest_dropped_frames": dict(batch.dropped_frames),
                    "out_of_order_packets_dropped": dict(out_of_order_packets_dropped),
                    "world_motion_entries": len(world_motion),
                    "trajectory_smoothing_enabled": trajectory_smoother is not None,
                    "trajectory_smoothing_accepted": (
                        trajectory_smoother.metrics["accepted"]
                        if trajectory_smoother is not None else 0
                    ),
                    "trajectory_smoothing_rejected": (
                        trajectory_smoother.metrics["outliers_rejected"]
                        if trajectory_smoother is not None else 0
                    ),
                    "trajectory_smoothing_out_of_order_rejected": (
                        trajectory_smoother.metrics["out_of_order_rejected"]
                        if trajectory_smoother is not None else 0
                    ),
                    "trajectory_smoothing_histories": (
                        len(trajectory_smoother.histories)
                        if trajectory_smoother is not None else 0
                    ),
                    "trajectory_smoothing_samples": (
                        sum(
                            len(history)
                            for history in trajectory_smoother.histories.values()
                        )
                        if trajectory_smoother is not None else 0
                    ),
                    "ingest_errors": dict(getattr(reader, "errors", {})),
                },
            }
    finally:
        fusion.close()
        if reader is not None:
            reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--cameras", nargs="+", default=None, metavar="CAM",
                        help="Camera folders to use (default: all downloaded)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Process every Nth frame (default: 1, every frame)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional diagnostic limit; default runs until the stream ends")
    parser.add_argument("--log-every", type=int, default=30, metavar="TICKS",
                        help="Print status every N processed ticks (default: 30)")
    parser.add_argument("--device", default=None, choices=["mps", "cpu", "cuda"],
                        help="Inference device (default: cpu)")
    parser.add_argument(
        "--local-tracker",
        choices=LOCAL_TRACKERS,
        default=None,
        help="Per-camera tracker (default: ocsort; bytetrack is kept for A/B)",
    )
    parser.add_argument("--save", default=None, metavar="FILE.rrd",
                        help="Record to a .rrd file instead of opening the viewer")
    parser.add_argument("--async-ingest", action=argparse.BooleanOptionalAction, default=True,
                        help="Independent lossless bounded queues (default on; disable for lockstep replay)")
    parser.add_argument("--cpu-threads", type=int, default=None,
                        help="PyTorch CPU threads (default: up to 4)")
    parser.add_argument("--single-person-mode", action="store_true",
                        help="Optimize matching for at most one real person")
    parser.add_argument(
        "--profile",
        choices=sorted(TRACKING_PROFILE_OVERRIDES),
        default=DEFAULT_TRACKING_PROFILE,
        help=f"Named identity policy (default: {DEFAULT_TRACKING_PROFILE})",
    )
    parser.add_argument(
        "--expected-people",
        type=int,
        default=None,
        help="Override fixed public-GID capacity (0 disables the cap)",
    )
    args = parser.parse_args()
    if args.expected_people is not None and args.expected_people < 0:
        parser.error("--expected-people must be >= 0")

    cfg = PipelineConfig(data_root=args.data_root)
    apply_tracking_profile(cfg, args.profile)
    if args.cameras:
        cfg.cameras = args.cameras
    else:
        available = sorted(p.name for p in cfg.scene_dir.glob("camera_*") if p.is_dir())
        if not available:
            raise SystemExit("No cameras found — run `uv run download-data` first.")
        cfg.cameras = available
    if args.stride is not None:
        cfg.stride = args.stride
    if args.device:
        cfg.device = args.device
    if args.local_tracker:
        cfg.local_tracker = args.local_tracker
    cfg.async_ingest = args.async_ingest
    if args.cpu_threads is not None:
        cfg.cpu_threads = args.cpu_threads
    cfg.single_person_mode = args.single_person_mode
    if args.expected_people is not None:
        cfg.fusion_fixed_population_size = args.expected_people

    # Imported here, not at module top: evaluate.py and render_preview.py
    # import run_pipeline from this module and should not pay for (or
    # depend on) the visualization stack they never use.
    from .viz import RerunLogger

    logger = None
    fps_smoothed: float | None = None
    t_prev = time.perf_counter()
    started = t_prev
    processed_ticks = camera_frames = 0
    metrics = {}
    print(
        f"Cameras: {', '.join(cfg.cameras)} | device={cfg.device} "
        f"stride={cfg.stride} tracker={cfg.local_tracker} profile={args.profile} "
        f"identity_slots={cfg.fusion_fixed_population_size or 'unbounded'}"
    )

    for tick in run_pipeline(cfg, max_frames=args.max_frames):
        processed_ticks += 1
        camera_frames += len(tick["frames"])
        metrics = tick["metrics"]
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
        if tick["tick_index"] == 1 or (
            args.log_every > 0 and tick["tick_index"] % args.log_every == 0
        ):
            metrics = tick["metrics"]
            print(f"tick {tick['tick_index']:5d} frame {tick['frame_index']:5d} | "
                  f"{len(tick['global_tracks'])} people tracked | "
                  f"{fps_smoothed:.1f} ticks/s | "
                  f"merges={metrics.get('global_id_merges', 0)} "
                  f"late_drop={metrics.get('tracklet_late_dropped', 0)}")

    elapsed = time.perf_counter() - started
    print(f"Completed {processed_ticks} ticks / {camera_frames} camera frames in {elapsed:.2f}s "
          f"({processed_ticks / max(elapsed, 1e-6):.2f} ticks/s); "
          f"ingest_drop={metrics.get('ingest_dropped_frames', {})}; "
          f"errors={metrics.get('ingest_errors', {})}")

    if args.save:
        print(f"Recording saved to {args.save}")


if __name__ == "__main__":
    main()
