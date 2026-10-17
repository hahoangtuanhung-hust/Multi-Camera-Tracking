import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from multicam_tracker.fusion import CameraObservation
from multicam_tracker.stage_diagnostics import (
    BboxGroundTruth,
    PipelineStageDiagnostics,
    load_bbox_ground_truth,
)


class StageDiagnosticsTests(unittest.TestCase):
    def test_stage_report_separates_detector_and_publication_losses(self) -> None:
        gt = {
            0: {
                "camera_0001": {
                    1: BboxGroundTruth(
                        np.array([0.0, 0.0, 10.0, 20.0]), np.array([0.0, 0.0])
                    ),
                    2: BboxGroundTruth(
                        np.array([100.0, 0.0, 110.0, 20.0]), np.array([5.0, 0.0])
                    ),
                }
            }
        }
        diagnostics = PipelineStageDiagnostics(gt, stride=1, segment_frames=500)
        local_track = SimpleNamespace(
            track_id=7,
            xyxy=np.array([0.0, 0.0, 10.0, 20.0]),
        )
        observation = CameraObservation(
            camera="camera_0001",
            local_id=7,
            world_xy=np.array([0.1, 0.0]),
            bbox_xyxy=local_track.xyxy.copy(),
            conf=0.9,
            capture_ts_s=0.0,
            arrival_ts_s=0.0,
        )
        tick = {
            "packets": {"camera_0001": SimpleNamespace(sequence=0)},
            "detections": {
                "camera_0001": np.array([[0.0, 0.0, 10.0, 20.0, 0.9]])
            },
            "camera_tracks": {"camera_0001": [local_track]},
            "observations": [observation],
            "accepted_fusion_keys": {("camera_0001", 7)},
            "local_to_gid": {("camera_0001", 7): 5},
            "global_tracks": [SimpleNamespace(gid=5)],
        }

        diagnostics.update(tick)
        report = diagnostics.finalize()

        self.assertEqual(report["overall"]["gt_instances"], 2)
        self.assertEqual(report["overall"]["detector"]["TP"], 1)
        self.assertEqual(report["overall"]["detector"]["FN"], 1)
        self.assertEqual(
            report["overall"]["stage_loss_gt_instances"]["complete"], 1
        )
        self.assertEqual(
            report["overall"]["stage_loss_gt_instances"]["detector"], 1
        )
        self.assertAlmostEqual(report["overall"]["projection_rmse_m"], 0.1)
        self.assertEqual(
            report["local_tracking_by_camera"]["camera_0001"]["IDSW"], 0
        )
        self.assertEqual(
            report["failure_events"][0]["first_failed_stage"], "detector"
        )

    def test_bbox_gt_loader_applies_camera_trim_once(self) -> None:
        with TemporaryDirectory() as root:
            scene = Path(root)
            (scene / "ground_truth.txt").write_text(
                "1 42 30 10 20 30 40 1.5 2.5\n"
                "7 42 60 11 21 31 41 1.6 2.6\n",
                encoding="utf-8",
            )
            loaded = load_bbox_ground_truth(
                scene,
                ["camera_0001", "camera_0007"],
                {"camera_0001": 30, "camera_0007": 60},
            )

        self.assertEqual(sorted(loaded), [0])
        self.assertEqual(
            loaded[0]["camera_0001"][42].bbox_xyxy.tolist(),
            [10.0, 20.0, 40.0, 60.0],
        )
        self.assertEqual(
            loaded[0]["camera_0007"][42].world_xy.tolist(), [1.6, 2.6]
        )

    def test_confidence_sweep_uses_raw_floor_without_changing_pipeline_stage(self) -> None:
        gt = {
            0: {
                "camera_0001": {
                    1: BboxGroundTruth(
                        np.array([0.0, 0.0, 10.0, 20.0]), np.zeros(2)
                    ),
                    2: BboxGroundTruth(
                        np.array([100.0, 0.0, 110.0, 20.0]), np.ones(2)
                    ),
                }
            }
        }
        diagnostics = PipelineStageDiagnostics(
            gt,
            pipeline_detection_threshold=0.25,
            confidence_source_floor=0.15,
        )
        tick = {
            "packets": {"camera_0001": SimpleNamespace(sequence=0)},
            "detections": {
                "camera_0001": np.array([[0.0, 0.0, 10.0, 20.0, 0.9]])
            },
            "camera_tracks": {"camera_0001": []},
            "observations": [],
            "accepted_fusion_keys": set(),
            "local_to_gid": {},
            "global_tracks": [],
        }
        raw = {
            "camera_0001": np.array([
                [0.0, 0.0, 10.0, 20.0, 0.9],
                [100.0, 0.0, 110.0, 20.0, 0.18],
            ])
        }

        diagnostics.update(tick, confidence_detections_by_camera=raw)
        report = diagnostics.finalize()

        self.assertEqual(report["overall"]["detector"]["TP"], 1)
        self.assertEqual(report["detection_by_confidence_threshold"]["0.15"]["TP"], 2)
        self.assertEqual(report["detection_by_confidence_threshold"]["0.20"]["TP"], 1)
        self.assertEqual(
            report["detection_by_confidence_threshold_and_camera"]
            ["camera_0001"]["0.15"]["TP"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
