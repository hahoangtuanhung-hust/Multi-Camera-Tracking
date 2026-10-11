"""Ablate CPU changes on identical saved detections (trusted local pickle only)."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
from unittest.mock import patch

import numpy as np

from multicam_tracker import bytetrack, fusion, tracklets
from multicam_tracker.config import PipelineConfig
from multicam_tracker.demo import run_pipeline
from multicam_tracker.evaluate import ground_truth_by_frame
from multicam_tracker.timebase import FrameBatch, FramePacket
from evaluate_thresholds import _score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
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
    rows = pickle.loads(args.detections.read_bytes())
    cfg = PipelineConfig()
    gt = ground_truth_by_frame(cfg.scene_dir, cfg.cameras, cfg.camera_start_frame_offsets)
    scores = {}

    class Detector:
        index = 0

        def detect_batch(self, frames):
            result = rows[self.index]["detections"]
            self.index += 1
            return result

    def batches():
        image = np.zeros((1, 1, 3), np.uint8)
        for row in rows:
            packets = {camera: FramePacket(image=image, **values) for camera, values in row["packets"].items()}
            yield FrameBatch(packets, max(p.arrival_ts_s for p in packets.values()))

    def score(name, pipeline=run_pipeline):
        predictions = [(tick["frame_index"], {track.gid: track.world_xy.copy() for track in tick["global_tracks"]})
                       for tick in pipeline(cfg, batch_iter=batches(), detector=Detector())]
        scores[name] = _score(predictions, gt)
        print(name, scores[name], flush=True)
        args.out.write_text(json.dumps(scores, indent=2), encoding="utf-8")

    score("baseline", base_demo.run_pipeline)
    score("current")
    with patch.object(bytetrack, "match_by_iou", base_tracking.match_by_iou):
        score("old_iou")
    with patch.object(bytetrack.ByteTracker, "update", base_tracking.ByteTracker.update):
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


if __name__ == "__main__":
    main()
