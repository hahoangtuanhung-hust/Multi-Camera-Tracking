import unittest
from unittest.mock import patch

import numpy as np

from multicam_tracker.config import PipelineConfig
from multicam_tracker.detector import PersonDetector


class _FakeYolo:
    def __init__(self, _model_name: str) -> None:
        self.calls = []
        self.callbacks = []

    def add_callback(self, event, callback) -> None:
        self.callbacks.append((event, callback))

    def __call__(self, frames, **kwargs):
        self.calls.append((frames, kwargs))
        return []


class DetectorPrecisionTests(unittest.TestCase):
    @patch("multicam_tracker.detector.YOLO", _FakeYolo)
    def test_cuda_uses_fp16_when_enabled(self) -> None:
        detector = PersonDetector(PipelineConfig(device="cuda", detector_fp16=True))

        detector.detect_batch([np.zeros((8, 8, 3), dtype=np.uint8)])

        kwargs = detector.model.calls[-1][1]
        self.assertEqual(kwargs["quantize"], 16)
        self.assertFalse(kwargs["compile"])

    @patch("multicam_tracker.detector.YOLO", _FakeYolo)
    def test_cuda_fp32_and_compile_are_explicit(self) -> None:
        detector = PersonDetector(PipelineConfig(
            device="cuda:0",
            detector_fp16=False,
            detector_compile=True,
        ))

        detector.detect_batch([np.zeros((8, 8, 3), dtype=np.uint8)])

        kwargs = detector.model.calls[-1][1]
        self.assertIsNone(kwargs["quantize"])
        self.assertTrue(kwargs["compile"])

    @patch("multicam_tracker.detector.YOLO", _FakeYolo)
    def test_cpu_ignores_cuda_precision_and_compile_flags(self) -> None:
        detector = PersonDetector(PipelineConfig(
            device="cpu",
            detector_fp16=True,
            detector_compile=True,
        ))

        detector.detect_batch([np.zeros((8, 8, 3), dtype=np.uint8)])

        kwargs = detector.model.calls[-1][1]
        self.assertIsNone(kwargs["quantize"])
        self.assertFalse(kwargs["compile"])


if __name__ == "__main__":
    unittest.main()
