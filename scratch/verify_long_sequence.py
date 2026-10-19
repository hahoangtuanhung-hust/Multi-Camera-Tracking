"""Long sequence verification (1000 ticks) comparing top candidates."""
import argparse
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from multicam_tracker.config import (
    DEFAULT_TRACKING_PROFILE,
    build_effective_config,
)
from multicam_tracker.demo import run_pipeline
from multicam_tracker.detection_cache import (
    configure_from_cache,
    filter_cached_detection_batch,
)
from multicam_tracker.evaluate import ground_truth_by_frame
from multicam_tracker.identity_commit import IdentityCommitBuffer, append_committed_frames
from multicam_tracker.mot_metrics import evaluate_mot
from multicam_tracker.reporting import configure_determinism
from multicam_tracker.timebase import FrameBatch, FramePacket


def evaluate_single(cfg, rows, gt_frames):
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

    metrics = evaluate_mot(predictions, gt_frames, frame_offset=0, match_gate_m=1.0)
    return {
        "IDF1": metrics["IDF1"],
        "IDP": metrics["IDP"],
        "IDR": metrics["IDR"],
        "MOTA": metrics["MOTA"],
        "HOTA": metrics["HOTA"],
        "IDSW": metrics["IDSW"],
        "Frag": metrics["Frag"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "elapsed_s": elapsed_s,
        "ticks_per_s": ticks_per_s,
    }


def main():
    cache_path = Path("data/cpu-audit/detections-low-floor-4000.pkl")
    max_frames = 1000
    print(f"Loading {max_frames} ticks from {cache_path}...", flush=True)

    payload = pickle.loads(cache_path.read_bytes())
    metadata = payload["metadata"]
    rows = payload["rows"][:max_frames]

    base_cfg = build_effective_config(profile="identity-v1-baseline")
    configure_from_cache(base_cfg, metadata)

    gt = ground_truth_by_frame(
        base_cfg.scene_dir,
        base_cfg.cameras,
        base_cfg.camera_start_frame_offsets,
        max_frame=(len(rows) * base_cfg.stride) + 10,
    )
    print("Ground truth loaded. Starting long-sequence comparison...\n", flush=True)

    no_reid_flags = {
        "hybridsort_reappearance_enabled": False,
        "fusion_reconnect_appearance_enabled": False,
        "fusion_appearance_ownership_guard_enabled": False,
    }

    models = {
        "Baseline (identity-v1-baseline)": {
            "profile": "identity-v1-baseline",
            **no_reid_flags,
        },
        "HybridSORT (hybridsort)": {
            "profile": "hybridsort",
            **no_reid_flags,
        },
        "Best Champion (hybridsort-late-binding-balanced)": {
            "profile": "hybridsort-late-binding-balanced",
            **no_reid_flags,
        },
    }

    print(f"{'Configuration':<50} | {'IDF1':<7} | {'MOTA':<7} | {'HOTA':<7} | {'IDSW':<5} | {'Frag':<5} | {'Ticks/s':<7}", flush=True)
    print("-" * 105, flush=True)

    for name, params in models.items():
        cfg = build_effective_config(profile=params["profile"])
        configure_from_cache(cfg, metadata)
        for k, v in params.items():
            if k != "profile":
                setattr(cfg, k, v)
        res = evaluate_single(cfg, rows, gt)
        print(
            f"{name:<50} | {res['IDF1']:.4f}  | {res['MOTA']:.4f}  | {res['HOTA']:.4f}  | "
            f"{res['IDSW']:<5} | {res['Frag']:<5} | {res['ticks_per_s']:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
