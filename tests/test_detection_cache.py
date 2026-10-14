import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from multicam_tracker.config import PipelineConfig
from multicam_tracker.detection_cache import (
    DetectionCacheError,
    build_detection_cache_metadata,
    configure_from_cache,
    validate_detection_cache,
)


class DetectionCacheTests(unittest.TestCase):
    def _fixture(self, root: str):
        data_root = Path(root) / "data"
        camera_dir = data_root / "custom" / "camera_0001"
        camera_dir.mkdir(parents=True)
        (camera_dir / "video.mp4").write_bytes(b"representative-video")
        model = Path(root) / "model.pt"
        model.write_bytes(b"model")
        cfg = PipelineConfig(
            data_root=data_root,
            cameras=["camera_0001"],
            model_name=str(model),
            async_ingest=False,
        )
        rows = [{
            "packets": {
                "camera_0001": {
                    "camera": "camera_0001",
                    "capture_ts_s": 1.0,
                    "arrival_ts_s": 1.01,
                    "sequence": 0,
                    "source_frame_index": 0,
                }
            },
            "detections": [np.array(
                [[1.0, 2.0, 3.0, 5.0, 0.75]], dtype=np.float32
            )],
        }]
        return cfg, rows, camera_dir

    def test_round_trip_validation_and_cached_source_configuration(self) -> None:
        with TemporaryDirectory() as root:
            cfg, rows, _ = self._fixture(root)
            metadata = build_detection_cache_metadata(cfg, len(rows))
            replay_cfg = PipelineConfig(
                data_root=cfg.data_root,
                cameras=["wrong"],
                model_name=cfg.model_name,
                async_ingest=False,
            )

            configure_from_cache(replay_cfg, metadata)
            validate_detection_cache(metadata, rows, replay_cfg)

            self.assertEqual(replay_cfg.cameras, ["camera_0001"])
            self.assertIn("sample_sha256", metadata["input"]["videos"]["camera_0001"])

    def test_rejects_detector_or_video_mismatch(self) -> None:
        with TemporaryDirectory() as root:
            cfg, rows, camera_dir = self._fixture(root)
            metadata = build_detection_cache_metadata(cfg, len(rows))

            cfg.conf_threshold += 0.01
            with self.assertRaisesRegex(DetectionCacheError, "confidence"):
                validate_detection_cache(metadata, rows, cfg)
            cfg.conf_threshold -= 0.01

            (camera_dir / "video.mp4").write_bytes(b"different-video-data")
            self.assertEqual(
                (camera_dir / "video.mp4").stat().st_size,
                metadata["input"]["videos"]["camera_0001"]["bytes"],
            )
            with self.assertRaisesRegex(DetectionCacheError, "fingerprint"):
                validate_detection_cache(metadata, rows, cfg)

    def test_rejects_malformed_detection_rows(self) -> None:
        with TemporaryDirectory() as root:
            cfg, rows, _ = self._fixture(root)
            metadata = build_detection_cache_metadata(cfg, len(rows))
            rows[0]["detections"][0] = np.array([[3, 2, 1, 5, 2.0]])

            with self.assertRaisesRegex(DetectionCacheError, "invalid boxes"):
                validate_detection_cache(metadata, rows, cfg)


if __name__ == "__main__":
    unittest.main()
