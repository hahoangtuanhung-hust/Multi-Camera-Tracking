"""Ablate CPU changes on identical saved detections (trusted local pickle only)."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter
import tracemalloc
from collections import defaultdict
from unittest.mock import patch

import numpy as np

from multicam_tracker import bytetrack, fusion, tracklets
from multicam_tracker.config import TRACKING_PROFILE_OVERRIDES, build_effective_config
from multicam_tracker.demo import run_pipeline
from multicam_tracker.detection_cache import (
    SUPPORTED_CACHE_SCHEMAS,
    configure_from_cache,
    filter_cached_detection_batch,
    validate_detection_cache,
)
from multicam_tracker.evaluate import (
    _classify_switch_events,
    ground_truth_by_frame,
)
from multicam_tracker.identity_commit import IdentityCommitBuffer, append_committed_frames
from multicam_tracker.mot_metrics import (
    build_identity_error_taxonomy,
    evaluate_mot,
    evaluate_mot_slices,
    identity_trajectory_diagnostics,
)
from multicam_tracker.local_tracker import LOCAL_TRACKERS
from multicam_tracker.reporting import (
    build_run_metadata,
    canonical_replay_tick_bytes,
    configure_determinism,
    committed_stream_hashes,
)
from multicam_tracker.stage_diagnostics import (
    PipelineStageDiagnostics,
    load_bbox_ground_truth,
)
from multicam_tracker.timebase import FrameBatch, FramePacket
from multicam_tracker.video import IndexedVideoFrameReader
from evaluate_thresholds import _score


def _timed_ticks(iterator, latencies_s: list[float]):
    """Yield pipeline ticks while measuring generator compute time only."""
    iterator = iter(iterator)
    while True:
        started = perf_counter()
        try:
            tick = next(iterator)
        except StopIteration:
            return
        latencies_s.append(perf_counter() - started)
        yield tick


def _deterministic_score(result: dict[str, object]) -> dict[str, object]:
    """Exclude runtime-only values before exact replay comparison."""
    ignored = {
        "elapsed_s",
        "ticks_per_s",
        "tracking_tick_latency_ms",
        "peak_traced_memory_mib",
    }
    return {key: value for key, value in result.items() if key not in ignored}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-source",
        type=Path,
        help="Source tree for historical ablations (not needed with --only-current)",
    )
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only-current", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Replay only the first N cached ticks after validating the full cache",
    )
    parser.add_argument("--frame-offset", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--allow-legacy-cache",
        action="store_true",
        help="Allow an unverifiable pre-v2 trusted local pickle",
    )
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Replay the current configuration twice and fail on any metric drift",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(TRACKING_PROFILE_OVERRIDES),
        default="identity-v1-baseline",
    )
    parser.add_argument("--identity-events", type=Path)
    parser.add_argument(
        "--skip-stage-diagnostics",
        action="store_true",
        help="Skip bbox-stage decomposition for fast candidate screening",
    )
    parser.add_argument(
        "--skip-memory-profile",
        action="store_true",
        help="Disable tracemalloc for fast candidate screening",
    )
    parser.add_argument(
        "--fast-metrics",
        action="store_true",
        help=(
            "Compute aggregate MOT metrics only; skip trajectory diagnostics, "
            "error taxonomy, and repeated window/cumulative evaluations"
        ),
    )
    parser.add_argument(
        "--video-frames",
        action="store_true",
        help=(
            "Decode the exact cached source frames so appearance-based "
            "association can be measured"
        ),
    )
    parser.add_argument(
        "--allow-placeholder-frames",
        action="store_true",
        help=(
            "Allow an appearance-enabled profile to run with 1x1 placeholder "
            "images for structural smoke tests only"
        ),
    )
    parser.add_argument(
        "--local-tracker",
        choices=LOCAL_TRACKERS,
        default=None,
        help="Explicit tracker override (default: selected profile)",
    )
    parser.add_argument("--ocsort-inertia", type=float, default=None)
    parser.add_argument("--ocsort-delta-t", type=int, default=None)
    parser.add_argument("--ocsort-oru-min-misses", type=int, default=None)
    parser.add_argument("--hybridsort-alpha", type=float, default=None)
    parser.add_argument("--hybridsort-min-blended-iou", type=float, default=None)
    parser.add_argument("--track-min-iou", type=float, default=None)
    parser.add_argument("--tracklet-match-interval-s", type=float, default=None)
    parser.add_argument("--tracklet-resample-hz", type=float, default=None)
    parser.add_argument(
        "--ocsort-use-byte",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--ocsort-observation-recovery",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--ocsort-oru",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be >= 0")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be > 0")
    if args.fast_metrics and args.identity_events:
        parser.error("--fast-metrics cannot be combined with --identity-events")
    if not args.only_current and args.baseline_source is None:
        parser.error("--baseline-source is required unless --only-current is set")
    configure_determinism(args.seed)
    base_demo = base_tracking = base_tracklets = base_fusion = None
    if not args.only_current:
        package = args.baseline_source / "multicam_tracker"
        spec = importlib.util.spec_from_file_location(
            "baseline_tracker",
            package / "__init__.py",
            submodule_search_locations=[str(package)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load baseline source from {package}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        base_demo = importlib.import_module("baseline_tracker.demo")
        base_tracking = importlib.import_module("baseline_tracker.bytetrack")
        base_tracklets = importlib.import_module("baseline_tracker.tracklets")
        base_fusion = importlib.import_module("baseline_tracker.fusion")
    payload = pickle.loads(args.detections.read_bytes())
    metadata = None
    if (
        isinstance(payload, dict)
        and payload.get("metadata", {}).get("schema") in SUPPORTED_CACHE_SCHEMAS
    ):
        metadata = payload["metadata"]
        rows = payload["rows"]
        print(json.dumps(metadata, indent=2), flush=True)
    else:
        if not args.allow_legacy_cache:
            raise ValueError(
                "Unverifiable legacy detection cache; pass --allow-legacy-cache "
                "only for a trusted local pickle"
            )
        rows = payload
    cfg = build_effective_config(
        data_root=args.data_root,
        model_name=args.model,
        async_ingest=False,
        fusion_output_time_policy="event",
        profile=args.profile,
        local_tracker=args.local_tracker,
    )
    if args.ocsort_inertia is not None:
        cfg.ocsort_inertia = args.ocsort_inertia
    if args.ocsort_delta_t is not None:
        cfg.ocsort_delta_t = args.ocsort_delta_t
    if args.ocsort_oru_min_misses is not None:
        cfg.ocsort_oru_min_misses = args.ocsort_oru_min_misses
    if args.ocsort_use_byte is not None:
        cfg.ocsort_use_byte = args.ocsort_use_byte
    if args.ocsort_observation_recovery is not None:
        cfg.ocsort_observation_recovery_enabled = args.ocsort_observation_recovery
    if args.ocsort_oru is not None:
        cfg.ocsort_oru_enabled = args.ocsort_oru
    if args.hybridsort_alpha is not None:
        if not np.isfinite(args.hybridsort_alpha):
            parser.error("--hybridsort-alpha must be finite")
        cfg.hybridsort_alpha = args.hybridsort_alpha
    for argument, attribute in (
        (args.hybridsort_min_blended_iou, "hybridsort_min_blended_iou"),
        (args.track_min_iou, "track_min_iou"),
        (args.tracklet_match_interval_s, "tracklet_match_interval_s"),
        (args.tracklet_resample_hz, "tracklet_resample_hz"),
    ):
        if argument is not None:
            if not np.isfinite(argument) or argument <= 0:
                parser.error(f"--{attribute.replace('_', '-')} must be finite and > 0")
            setattr(cfg, attribute, argument)
    if not 0.0 <= cfg.hybridsort_alpha <= 1.0:
        parser.error("--hybridsort-alpha must be in [0, 1]")
    for attribute in ("hybridsort_min_blended_iou", "track_min_iou"):
        if getattr(cfg, attribute) > 1.0:
            parser.error(f"--{attribute.replace('_', '-')} must be <= 1")
    if metadata is not None:
        configure_from_cache(cfg, metadata)
        validate_detection_cache(metadata, rows, cfg)
    appearance_enabled = bool(
        cfg.hybridsort_reappearance_enabled
        or cfg.fusion_reconnect_appearance_enabled
        or cfg.fusion_appearance_ownership_guard_enabled
    )
    if args.video_frames and metadata is None:
        parser.error("--video-frames requires a versioned, validated cache")
    if (
        appearance_enabled
        and not args.video_frames
        and not args.allow_placeholder_frames
    ):
        parser.error(
            "This profile enables appearance association. Pass --video-frames "
            "for a valid measurement, or --allow-placeholder-frames only for "
            "a structural smoke test."
        )
    if args.max_frames is not None:
        rows = rows[:args.max_frames]
    gt = ground_truth_by_frame(cfg.scene_dir, cfg.cameras, cfg.camera_start_frame_offsets)
    bbox_gt = (
        None
        if args.skip_stage_diagnostics
        else load_bbox_ground_truth(
            cfg.scene_dir,
            cfg.cameras,
            cfg.camera_start_frame_offsets,
            max_frame=len(rows) * cfg.stride + 3,
        )
    )
    scores = {}
    confidence_source_floor = (
        float(metadata["detector"]["conf_threshold"])
        if metadata is not None else float(cfg.conf_threshold)
    )

    class Detector:
        index = 0
        last_raw_detections = []

        def detect_batch(self, frames):
            result = rows[self.index]["detections"]
            self.index += 1
            self.last_raw_detections = [
                np.asarray(item, dtype=float).reshape((-1, 5)).copy()
                for item in result
            ]
            return filter_cached_detection_batch(result, cfg.conf_threshold)

    def batches():
        placeholder = np.zeros((1, 1, 3), np.uint8)
        frame_reader = (
            IndexedVideoFrameReader(
                cfg.scene_dir, cfg.cameras, cfg.decoder_threads
            )
            if args.video_frames else None
        )
        try:
            for row in rows:
                packets = {}
                for camera, values in row["packets"].items():
                    image = (
                        frame_reader.read(camera, values["source_frame_index"])
                        if frame_reader is not None else placeholder
                    )
                    packets[camera] = FramePacket(image=image, **values)
                yield FrameBatch(
                    packets,
                    max(packet.arrival_ts_s for packet in packets.values()),
                )
        finally:
            if frame_reader is not None:
                frame_reader.close()

    def score(name, pipeline=run_pipeline):
        run_metadata = build_run_metadata(cfg, seed=args.seed)
        print(f"Starting {name}: {args.profile}, {len(rows)} ticks", flush=True)
        configure_determinism(args.seed)
        if not args.skip_memory_profile:
            tracemalloc.start()
        started = perf_counter()
        predictions = []
        source_maps = {}
        no_public_affinity = 0
        provisional_affinity = 0
        public_affinity_not_output = 0
        prediction_stream = hashlib.sha256()
        assignment_stream = hashlib.sha256()
        tick_latencies_s: list[float] = []
        last_pipeline_metrics = {}
        identity_committer = IdentityCommitBuffer(cfg.fusion_identity_commit_delay_s)
        previous_diagnostics_path = cfg.identity_diagnostics_path
        stage_diagnostics = (
            PipelineStageDiagnostics(
                bbox_gt,
                stride=cfg.stride,
                pipeline_detection_threshold=cfg.conf_threshold,
                confidence_source_floor=confidence_source_floor,
            )
            if name.startswith("current") and bbox_gt is not None
            else None
        )
        replay_detector = Detector()
        cfg.identity_diagnostics_path = (
            args.identity_events if name == "current" else None
        )
        try:
            for tick in _timed_ticks(
                pipeline(cfg, batch_iter=batches(), detector=replay_detector),
                tick_latencies_s,
            ):
                prediction_bytes, assignment_bytes = canonical_replay_tick_bytes(tick)
                for digest, payload in (
                    (prediction_stream, prediction_bytes),
                    (assignment_stream, assignment_bytes),
                ):
                    digest.update(len(payload).to_bytes(8, "big"))
                    digest.update(payload)
                if stage_diagnostics is not None:
                    raw_detections = {
                        camera: detections
                        for camera, detections in zip(
                            tick["packets"], replay_detector.last_raw_detections
                        )
                    }
                    stage_diagnostics.update(
                        tick,
                        confidence_detections_by_camera=raw_detections,
                    )
                append_committed_frames(
                    identity_committer.push_tick(tick), predictions, source_maps
                )
                emitted_gids = {int(track.gid) for track in tick["global_tracks"]}
                for observation in tick["observations"]:
                    key = observation.camera, int(observation.local_id)
                    gid = tick["local_to_gid"].get(key)
                    if gid is None:
                        no_public_affinity += 1
                        internal_gid = tick["local_to_internal_gid"].get(key)
                        if internal_gid is not None and int(internal_gid) < 0:
                            provisional_affinity += 1
                    else:
                        if int(gid) not in emitted_gids:
                            public_affinity_not_output += 1
                last_pipeline_metrics = tick["metrics"]
                if len(tick_latencies_s) % 100 == 0:
                    print(f"{name}: {len(tick_latencies_s)}/{len(rows)} ticks", flush=True)
        finally:
            cfg.identity_diagnostics_path = previous_diagnostics_path
        append_committed_frames(identity_committer.flush(), predictions, source_maps)
        last_pipeline_metrics.update(identity_committer.final_pipeline_metrics)
        last_pipeline_metrics.update(identity_committer.metrics)
        elapsed_s = perf_counter() - started
        if args.skip_memory_profile:
            peak_bytes = None
        else:
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        frame_offset = args.frame_offset
        scores[name] = _score(predictions, gt, frame_offset=frame_offset)
        stage_report = (
            stage_diagnostics.finalize()
            if stage_diagnostics is not None else None
        )
        diagnostics = []
        detailed_metrics = {}
        if not args.fast_metrics:
            full = evaluate_mot(predictions, gt, frame_offset=frame_offset)
            identity_diagnostics = identity_trajectory_diagnostics(
                predictions,
                gt,
                frame_offset=frame_offset,
            )
            diagnostics = _classify_switch_events(
                full["switch_events"], source_maps, predictions, gt, frame_offset
            )
            detailed_metrics = {
                **evaluate_mot_slices(
                    predictions,
                    gt,
                    frame_offset=frame_offset,
                    cumulative_frames=(500, 1000, 1500, 2000, 4000),
                    window_frames=500,
                ),
                "identity_diagnostics": identity_diagnostics,
                "switch_diagnostics": diagnostics,
                "identity_error_taxonomy": build_identity_error_taxonomy(
                    full,
                    identity_diagnostics,
                    last_pipeline_metrics,
                    switch_diagnostics=diagnostics,
                    no_public_affinity=no_public_affinity,
                    provisional_affinity=provisional_affinity,
                    public_affinity_not_output=public_affinity_not_output,
                    stage_diagnostics=stage_report,
                ),
            }
        scores[name].update({
            **committed_stream_hashes(predictions, source_maps),
            "identity_lifecycle": identity_committer.lifecycle_report,
            "replay_frame_source": (
                "source_video" if args.video_frames else "placeholder_1x1"
            ),
            "frame_offset": frame_offset,
            **detailed_metrics,
            "stage_diagnostics": stage_report,
            "ticks": len(predictions),
            "prediction_stream_sha256": prediction_stream.hexdigest(),
            "assignment_stream_sha256": assignment_stream.hexdigest(),
            "elapsed_s": elapsed_s,
            "ticks_per_s": len(predictions) / max(elapsed_s, 1e-9),
            "tracking_tick_latency_ms": {
                "p50": float(np.percentile(tick_latencies_s, 50) * 1000.0),
                "p95": float(np.percentile(tick_latencies_s, 95) * 1000.0),
                "max": float(max(tick_latencies_s) * 1000.0),
            },
            "peak_traced_memory_mib": (
                None if peak_bytes is None else peak_bytes / 2**20
            ),
            "pipeline_metrics": last_pipeline_metrics,
        })
        for metric in (
            "local_tracks_created",
            "local_track_reactivations",
            "local_tracks_removed_tentative",
            "local_tracks_removed_timeout",
            "local_high_confidence_matches",
            "local_low_confidence_matches",
            "local_ocm_matches",
            "local_ocr_matches",
            "local_oru_reupdates",
            "local_byte_extension_matches",
            "local_observation_outliers_suppressed",
            "local_appearance_reappearance_matches",
            "local_appearance_reappearance_deferred",
            "local_appearance_reappearance_ambiguous",
            "local_appearance_reappearance_candidate_resets",
        ):
            scores[name][metric] = last_pipeline_metrics.get(metric, 0)
        if args.identity_events and name.startswith("current"):
            if name == "current":
                with args.identity_events.open("a", encoding="utf-8") as destination:
                    for record in diagnostics:
                        destination.write(json.dumps(record) + "\n")
            classes = defaultdict(int)
            for record in diagnostics:
                classes[record["classification"]] += 1
            scores[name]["switch_classes"] = dict(classes)
        end_metadata = build_run_metadata(cfg, seed=args.seed)
        if run_metadata["repository"]["source_fingerprint"] != end_metadata["repository"]["source_fingerprint"]:
            raise RuntimeError("Source changed during replay; rerun with a frozen source tree")
        scores["run_metadata"] = run_metadata
        print(name, {key: scores[name][key] for key in (
            "idf1", "hota", "precision", "recall", "switches", "fragmentations",
            "prediction_stream_sha256", "assignment_stream_sha256",
        )}, flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(scores, indent=2), encoding="utf-8")
        return scores[name]

    if args.only_current:
        current = score("current")
        if args.verify_determinism:
            repeated = score("current_repeat")
            if _deterministic_score(current) != _deterministic_score(repeated):
                raise RuntimeError("Deterministic replay metric mismatch")
            scores["deterministic_replay_verified"] = True
        scores["run_metadata"] = build_run_metadata(cfg, seed=args.seed)
        scores["detection_cache_metadata"] = metadata
        args.out.write_text(json.dumps(scores, indent=2), encoding="utf-8")
        return

    score("baseline", base_demo.run_pipeline)
    score("current")
    with patch.object(bytetrack, "match_by_iou", base_tracking.match_by_iou):
        score("old_iou")
    with patch(
        "multicam_tracker.demo.create_local_tracker",
        side_effect=lambda current_cfg: base_tracking.ByteTracker(current_cfg),
    ):
        score("old_local_lifecycle")
    with patch.object(fusion.GlobalTrackManager, "_has_same_camera_conflict", base_fusion.GlobalTrackManager._has_same_camera_conflict):
        score("old_same_camera_conflict")

    class OldVelocityStore(base_tracklets.TrackletStore):
        def advance(self, timestamp_s):
            self.max_seen_timestamp_s = max(self.max_seen_timestamp_s, timestamp_s)
            self._prune()
            self.metrics["points_stored"] = sum(len(item.points) for item in self.histories.values())
            for history in self.histories.values():
                history.invalidate()

        def append(self, *args, **kwargs):
            result = super().append(*args, **kwargs)
            for history in self.histories.values():
                history.invalidate()
            return result

    for name in ("arrays", "bounds", "invalidate"):
        setattr(base_tracklets.TrackletHistory, name, getattr(tracklets.TrackletHistory, name))
    base_tracklets.TrackletHistory._arrays = None
    base_tracklets.TrackletHistory._bounds = None
    with patch.object(fusion, "TrackletStore", OldVelocityStore):
        score("old_velocity")
    if args.verify_determinism:
        current = scores["current"]
        repeated = score("current_repeat")
        if _deterministic_score(current) != _deterministic_score(repeated):
            raise RuntimeError("Deterministic replay metric mismatch")
        scores["deterministic_replay_verified"] = True
    scores["run_metadata"] = build_run_metadata(cfg, seed=args.seed)
    scores["detection_cache_metadata"] = metadata
    args.out.write_text(json.dumps(scores, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
