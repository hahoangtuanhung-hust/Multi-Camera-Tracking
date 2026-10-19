"""Automated ReID-free candidate benchmark using the official run_pipeline."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, List

import numpy as np

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from multicam_tracker.config import (
    DEFAULT_TRACKING_PROFILE,
    TRACKING_PROFILE_OVERRIDES,
    PipelineConfig,
    build_effective_config,
)
from multicam_tracker.demo import run_pipeline
from multicam_tracker.detection_cache import (
    configure_from_cache,
    filter_cached_detection_batch,
    validate_detection_cache,
)
from multicam_tracker.evaluate import ground_truth_by_frame
from multicam_tracker.identity_commit import IdentityCommitBuffer, append_committed_frames
from multicam_tracker.mot_metrics import evaluate_mot
from multicam_tracker.reporting import configure_determinism
from multicam_tracker.timebase import FrameBatch, FramePacket


def evaluate_candidate(
    cfg: PipelineConfig,
    rows: List[Dict[str, Any]],
    gt_frames: Dict[int, Dict[int, np.ndarray]],
    frame_offset: int = 0,
    match_gate_m: float = 1.0,
) -> Dict[str, Any]:
    class ReplayDetector:
        def __init__(self):
            self.index = 0

        def detect_batch(self, frames):
            result = rows[self.index]["detections"]
            self.index += 1
            return filter_cached_detection_batch(result, cfg.conf_threshold)

    placeholder = np.zeros((1, 1, 3), dtype=np.uint8)

    def batches():
        for row in rows:
            packets = {}
            for camera, values in row["packets"].items():
                packets[camera] = FramePacket(image=placeholder, **values)
            yield FrameBatch(
                packets,
                max(packet.arrival_ts_s for packet in packets.values()),
            )

    identity_committer = IdentityCommitBuffer(cfg.fusion_identity_commit_delay_s)
    predictions = []
    source_maps = {}

    start_time = time.perf_counter()
    detector = ReplayDetector()

    for tick in run_pipeline(cfg, batch_iter=batches(), detector=detector):
        append_committed_frames(
            identity_committer.push_tick(tick), predictions, source_maps
        )

    append_committed_frames(identity_committer.flush(), predictions, source_maps)
    elapsed_s = time.perf_counter() - start_time
    ticks_per_s = len(rows) / max(elapsed_s, 1e-6)

    metrics = evaluate_mot(
        predictions,
        gt_frames,
        frame_offset=frame_offset,
        match_gate_m=match_gate_m,
    )

    return {
        "IDF1": metrics["IDF1"],
        "IDP": metrics["IDP"],
        "IDR": metrics["IDR"],
        "MOTA": metrics["MOTA"],
        "HOTA": metrics["HOTA"],
        "AssA": metrics.get("AssA", 0.0),
        "DetA": metrics.get("DetA", 0.0),
        "IDSW": metrics["IDSW"],
        "Frag": metrics["Frag"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "RMSE": metrics["position_RMSE_m"],
        "elapsed_s": elapsed_s,
        "ticks_per_s": ticks_per_s,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("data/cpu-audit/detections-low-floor-300.pkl"))
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    configure_determinism(args.seed)

    print(f"Loading cache: {args.cache}", flush=True)
    payload = pickle.loads(args.cache.read_bytes())
    if isinstance(payload, dict) and "metadata" in payload:
        metadata = payload["metadata"]
        rows = payload["rows"]
    else:
        metadata = None
        rows = payload

    if args.max_frames:
        rows = rows[: args.max_frames]

    print(f"Loaded {len(rows)} ticks for evaluation.", flush=True)

    base_cfg = build_effective_config(
        data_root=Path("data"),
        async_ingest=False,
        fusion_output_time_policy="event",
        profile="identity-v1-baseline",
    )
    if metadata:
        configure_from_cache(base_cfg, metadata)

    print("Loading ground truth...", flush=True)
    gt = ground_truth_by_frame(
        base_cfg.scene_dir,
        base_cfg.cameras,
        base_cfg.camera_start_frame_offsets,
        max_frame=(len(rows) * base_cfg.stride) + 10,
    )
    print("Ground truth loaded.", flush=True)

    # Strictly enforce NO ReID
    no_reid_flags = {
        "hybridsort_reappearance_enabled": False,
        "fusion_reconnect_appearance_enabled": False,
        "fusion_appearance_ownership_guard_enabled": False,
    }

    candidates = {
        # 1. Baseline ByteTrack
        "01_bytetrack_baseline": {
            "profile": "identity-v1-baseline",
            **no_reid_flags,
        },
        # 2. Baseline ByteTrack + Drift Guard
        "02_bytetrack_drift_guard": {
            "profile": "identity-v1-drift-guard",
            **no_reid_flags,
        },
        # 3. Baseline ByteTrack + Duplicate Suppression (Safe)
        "03_bytetrack_safe": {
            "profile": "identity-v1-safe",
            **no_reid_flags,
        },
        # 4. OC-SORT pure
        "04_ocsort_pure": {
            "profile": "identity-v1-baseline",
            "local_tracker": "ocsort",
            "ocsort_observation_recovery_enabled": True,
            **no_reid_flags,
        },
        # 5. HybridSORT standard
        "05_hybridsort_default": {
            "profile": "hybridsort",
            **no_reid_flags,
        },
        # 6. HybridSORT + Tentative 1
        "06_hybridsort_tentative": {
            "profile": "hybridsort-tentative",
            **no_reid_flags,
        },
        # 7. HybridSORT + Crossing Guard (Ambiguity Suppression)
        "07_hybridsort_crossing_guard": {
            "profile": "hybridsort-crossing-guard",
            **no_reid_flags,
        },
        # 8. HybridSORT + Continuity (Tentative + Crossing Guard)
        "08_hybridsort_continuity": {
            "profile": "hybridsort-continuity",
            **no_reid_flags,
        },
        # 9. HybridSORT + Trajectory Smoothing
        "09_hybridsort_trajectory": {
            "profile": "hybridsort-trajectory",
            **no_reid_flags,
        },
        # 10. HybridSORT Late Binding Balanced (No ReID)
        "10_hybridsort_late_binding": {
            "profile": "hybridsort-late-binding-balanced",
            **no_reid_flags,
        },
        # 11. Senior CV Custom: HybridSORT + Alpha 0.50 + Crossing Guard + Extended Dormant
        "11_senior_cv_alpha50_crossing": {
            "profile": "hybridsort",
            "hybridsort_alpha": 0.50,
            "hybridsort_ambiguity_suppression_enabled": True,
            "hybridsort_ambiguity_margin": 0.04,
            "hybridsort_ambiguity_owner_grace_s": 0.20,
            "fusion_identity_retention_s": 5.0,
            "fusion_dormant_retention_s": 10.0,
            "fusion_local_reconnect_long_window_s": 5.0,
            **no_reid_flags,
        },
        # 12. Senior CV Custom: Late Binding + Crossing Guard + Trajectory Motion
        "12_senior_cv_latebind_crossing": {
            "profile": "hybridsort-late-binding-balanced",
            "hybridsort_alpha": 0.50,
            "hybridsort_ambiguity_suppression_enabled": True,
            "hybridsort_ambiguity_margin": 0.04,
            "hybridsort_ambiguity_owner_grace_s": 0.20,
            "fusion_identity_retention_s": 5.0,
            "fusion_dormant_retention_s": 10.0,
            "fusion_duplicate_suppression_enabled": True,
            "fusion_duplicate_merge_m": 0.35,
            "track_tentative_max_misses": 1,
            **no_reid_flags,
        },
        # 13. Senior CV Custom: Full Geometric Stack (Late Binding + Ambiguity + Trajectory Smoothing + Reconnect Confirmations)
        "13_senior_cv_full_geometric": {
            "profile": "hybridsort-late-binding-balanced",
            "hybridsort_alpha": 0.50,
            "hybridsort_ambiguity_suppression_enabled": True,
            "hybridsort_ambiguity_margin": 0.04,
            "hybridsort_ambiguity_owner_grace_s": 0.20,
            "fusion_identity_retention_s": 5.0,
            "fusion_dormant_retention_s": 10.0,
            "fusion_duplicate_suppression_enabled": True,
            "fusion_duplicate_merge_m": 0.35,
            "fusion_trajectory_smoothing_enabled": True,
            "fusion_robust_trajectory_motion_enabled": True,
            "fusion_local_reconnect_confirmations": 2,
            "track_tentative_max_misses": 1,
            **no_reid_flags,
        },
    }

    results = {}
    print("\nRunning systematic evaluation over candidates...\n", flush=True)
    header = f"{'Candidate':<32} | {'IDF1':<7} | {'MOTA':<7} | {'HOTA':<7} | {'IDSW':<5} | {'Frag':<5} | {'Ticks/s':<7}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for name, params in candidates.items():
        profile = params.get("profile", DEFAULT_TRACKING_PROFILE)
        cfg = build_effective_config(
            data_root=Path("data"),
            async_ingest=False,
            fusion_output_time_policy="event",
            profile=profile,
        )
        if metadata:
            configure_from_cache(cfg, metadata)

        # Apply candidate parameters
        for k, v in params.items():
            if k != "profile":
                setattr(cfg, k, v)

        # Evaluate candidate
        t0 = time.perf_counter()
        res = evaluate_candidate(cfg, rows, gt)
        duration = time.perf_counter() - t0
        results[name] = res

        print(
            f"{name:<32} | {res['IDF1']:.4f}  | {res['MOTA']:.4f}  | {res['HOTA']:.4f}  | "
            f"{res['IDSW']:<5} | {res['Frag']:<5} | {res['ticks_per_s']:.1f}",
            flush=True,
        )

    out_path = Path("scratch/benchmark_results.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved detailed benchmark results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
