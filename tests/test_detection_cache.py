import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from multicam_tracker.config import PipelineConfig
from multicam_tracker.detection_cache import (
    DetectionCacheError,
    build_detection_cache_metadata,
    configure_from_cache,
    filter_cached_detection_batch,
    validate_detection_cache,
)
from multicam_tracker.video import IndexedVideoFrameReader


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def grab(self) -> bool:
        if self.index >= len(self.frames):
            return False
        self.index += 1
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index].copy()
        self.index += 1
        return True, frame

    def release(self) -> None:
        self.released = True
from multicam_tracker.video import IndexedVideoFrameReader


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def grab(self) -> bool:
        if self.index >= len(self.frames):
            return False
        self.index += 1
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index].copy()
        self.index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


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

            # A low-floor cache can reproduce stricter floors by filtering.
            cfg.conf_threshold += 0.01
            validate_detection_cache(metadata, rows, cfg)
            cfg.conf_threshold -= 0.02
            with self.assertRaisesRegex(DetectionCacheError, "below.*floor"):
                validate_detection_cache(metadata, rows, cfg)
            cfg.conf_threshold += 0.01

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

    def test_cached_detection_batch_applies_stricter_floor(self) -> None:
        batch = [np.array([
            [0.0, 0.0, 10.0, 20.0, 0.15],
            [1.0, 0.0, 11.0, 20.0, 0.20],
            [2.0, 0.0, 12.0, 20.0, 0.80],
        ], dtype=np.float32)]

        filtered = filter_cached_detection_batch(batch, 0.20)

        self.assertEqual(filtered[0].shape, (2, 5))
        np.testing.assert_allclose(filtered[0][:, 4], [0.20, 0.80])
        self.assertFalse(np.shares_memory(filtered[0], batch[0]))

    def test_indexed_video_reader_decodes_exact_cached_source_frame(self) -> None:
        frames = [
            np.full((2, 3, 3), value, dtype=np.uint8)
            for value in range(5)
        ]
        capture = _FakeCapture(frames)
        with patch("multicam_tracker.video.open_video", return_value=capture):
            with IndexedVideoFrameReader(
                Path("unused"), ["camera_0001"]
            ) as reader:
                np.testing.assert_array_equal(
                    reader.read("camera_0001", 2), frames[2]
                )
                np.testing.assert_array_equal(
                    reader.read("camera_0001", 4), frames[4]
                )
        self.assertTrue(capture.released)

    def test_indexed_video_reader_rejects_frame_rewind(self) -> None:
        capture = _FakeCapture([
            np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)
        ])
        with patch("multicam_tracker.video.open_video", return_value=capture):
            with IndexedVideoFrameReader(
                Path("unused"), ["camera_0001"]
            ) as reader:
                reader.read("camera_0001", 1)
                with self.assertRaisesRegex(ValueError, "Non-monotonic"):
                    reader.read("camera_0001", 0)

if __name__ == "__main__":
    unittest.main()
