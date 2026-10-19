import unittest
from unittest.mock import patch

import numpy as np

from multicam_tracker.appearance import PersonAppearanceBackend, gallery_similarity
from multicam_tracker.config import PipelineConfig


class AppearanceBackendTests(unittest.TestCase):
    def test_missing_trained_weights_fails_without_silent_fallback(self):
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            PersonAppearanceBackend(PipelineConfig(appearance_backend="torchscript"))

    def test_non_event_tick_does_not_encode(self):
        backend = PersonAppearanceBackend(PipelineConfig(appearance_event_triggered=True))
        self.assertEqual(backend.extract(np.ones((128, 128, 3), dtype=np.uint8),
                                        np.array([[10, 10, 50, 100, .9]]), event=False), [None])

    def test_small_truncated_and_overlapping_crops_are_rejected(self):
        backend = PersonAppearanceBackend(PipelineConfig(appearance_quality_filter_enabled=True))
        boxes = np.array([[0, 0, 10, 20, .9], [-1, 0, 40, 100, .9],
                          [50, 10, 90, 110, .9], [50, 10, 90, 110, .9]])
        self.assertEqual(backend.extract(np.ones((128, 128, 3), dtype=np.uint8), boxes), [None]*4)
        self.assertEqual(backend.quality_rejections, 4)

    def test_batch_mapping_normalization_and_no_grad(self):
        import torch

        class Encoder(torch.nn.Module):
            def forward(self, value):
                self.input_shape = tuple(value.shape)
                self.grad_enabled = torch.is_grad_enabled()
                return value.mean(dim=(2, 3))

        encoder = Encoder()
        cfg = PipelineConfig(device="cpu", appearance_backend="torchscript",
                             appearance_model_path="test-contract-only.pt", appearance_model_sha256="test")
        with patch("multicam_tracker.appearance._load_scripted_reid", return_value=encoder):
            backend = PersonAppearanceBackend(cfg)
        boxes = np.array([[10, 10, 40, 100, .9], [45, 10, 55, 20, .9], [70, 10, 100, 100, .9]])
        result = backend.extract(np.full((128, 128, 3), 120, dtype=np.uint8), boxes)
        self.assertEqual(encoder.input_shape, (2, 3, 256, 128))
        self.assertFalse(encoder.grad_enabled)
        self.assertIsNone(result[1])
        self.assertAlmostEqual(float(np.linalg.norm(result[0])), 1., places=6)
        self.assertEqual(backend.inference_crops, 2)
        self.assertAlmostEqual(gallery_similarity([result[0]], result[2]), 1., places=5)


if __name__ == "__main__":
    unittest.main()
