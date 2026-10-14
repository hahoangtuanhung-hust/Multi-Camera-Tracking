"""Ablate CPU changes on identical saved detections (trusted local pickle only)."""
from __future__ import annotations

import argparse
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
from multicam_tracker.config import PipelineConfig, TRACKING_PROFILE_OVERRIDES, apply_tracking_profile
from multicam_tracker.demo import run_pipeline
from multicam_tracker.detection_cache import (
    SUPPORTED_CACHE_SCHEMAS,
    configure_from_cache,
    filter_cached_detection_batch,
    validate_detection_cache,
)
from multicam_tracker.evaluate import (
    _classify_switch_events,
    _match_errors,
    ground_truth_by_frame,
)
from multicam_tracker.mot_metrics import (
    build_identity_error_taxonomy,
    evaluate_mot,
    evaluate_mot_slices,
    identity_trajectory_diagnostics,
)
from multicam_tracker.local_tracker import LOCAL_TRACKERS
from multicam_tracker.reporting import build_run_metadata, configure_determinism
from multicam_tracker.timebase import FrameBatch, FramePacket
from evaluate_thresholds import _score


def _deterministic_score(result: dict[str, object]) -> dict[str, object]:
    """Exclude runtime-only values before exact replay comparison."""
    ignored = {"elapsed_s", "ticks_per_s", "peak_traced_memory_mib"}
    return {key: value for key, value in result.items() if key not in ignored}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only-current", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--seed", type=int, default=0)
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
        "--local-tracker",
        choices=LOCAL_TRACKERS,
        default="ocsort",
        help="Per-camera tracker used by the current pipeline replay",
    )
    parser.add_argument("--ocsort-inertia", type=float, default=None)
    parser.add_argument("--ocsort-delta-t", type=int, default=None)
    parser.add_argument("--ocsort-oru-min-misses", type=int, default=None)
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
    configure_determinism(args.seed)
    package = args.baseline_source / "multicam_tracker"
    spec = importlib.util.spec_from_file_location("baseline_tracker", package / "__init__.py",
                                                submodule_search_locations=[str(package)])
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
    cfg = PipelineConfig(
        data_root=args.data_root,
        model_name=args.model,
        async_ingest=False,
    )
    apply_tracking_profile(cfg, args.profile)
    cfg.local_tracker = args.local_tracker
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
    if metadata is not None:
        configure_from_cache(cfg, metadata)
        validate_detection_cache(metadata, rows, cfg)
    gt = ground_truth_by_frame(cfg.scene_dir, cfg.cameras, cfg.camera_start_frame_offsets)
    scores = {}

    class Detector:
        index = 0

        def detect_batch(self, frames):
            result = rows[self.index]["detections"]
            self.index += 1
            return filter_cached_detection_batch(result, cfg.conf_threshold)

    def batches():
        image = np.zeros((1, 1, 3), np.uint8)
        for row in rows:
            packets = {camera: FramePacket(image=image, **values) for camera, values in row["packets"].items()}
            yield FrameBatch(packets, max(p.arrival_ts_s for p in packets.values()))

    def score(name, pipeline=run_pipeline):
        configure_determinism(args.seed)
        tracemalloc.start()
        started = perf_counter()
        predictions = []
        source_maps = {}
        track_exists_not_output = 0
        last_pipeline_metrics = {}
        previous_diagnostics_path = cfg.identity_diagnostics_path
        cfg.identity_diagnostics_path = (
            args.identity_events if name == "current" else None
        )
        try:
            for tick in pipeline(cfg, batch_iter=batches(), detector=Detector()):
                predictions.append((
                    tick["frame_index"],
                    {track.gid: track.world_xy.copy() for track in tick["global_tracks"]},
                ))
                sources = defaultdict(list)
                for observation in tick["observations"]:
                    key = observation.camera, int(observation.local_id)
                    gid = tick["local_to_gid"].get(key)
                    if gid is None:
                        track_exists_not_output += 1
                    else:
                        sources[int(gid)].append(key)
                source_maps[tick["frame_index"]] = dict(sources)
                last_pipeline_metrics = tick["metrics"]
        finally:
            cfg.identity_diagnostics_path = previous_diagnostics_path
        elapsed_s = perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        scores[name] = _score(predictions, gt)
        best_offset = min(
            (0, 1, 2),
            key=lambda offset: np.sqrt(np.mean(np.square(
                _match_errors(predictions[:100], gt, offset)
            ))) if _match_errors(predictions[:100], gt, offset) else float("inf"),
        )
        full = evaluate_mot(predictions, gt, frame_offset=best_offset)
        identity_diagnostics = identity_trajectory_diagnostics(
            predictions,
            gt,
            frame_offset=best_offset,
        )
        diagnostics = _classify_switch_events(
            full["switch_events"], source_maps, predictions, gt, best_offset
        )
        scores[name].update({
            "frame_offset": best_offset,
            **evaluate_mot_slices(
                predictions,
                gt,
                frame_offset=best_offset,
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
                track_exists_not_output=track_exists_not_output,
            ),
            "ticks": len(predictions),
            "elapsed_s": elapsed_s,
            "ticks_per_s": len(predictions) / max(elapsed_s, 1e-9),
            "peak_traced_memory_mib": peak_bytes / 2**20,
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
        print(name, scores[name], flush=True)
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
