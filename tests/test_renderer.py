from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from multicam_tracker.render_preview import composite, render_video


class FakeBev:
    def __init__(self, *args):
        pass

    def render(self, tracks):
        return np.zeros((270, 540, 3), np.uint8)


class RendererTests(unittest.TestCase):
    def test_occluded_prediction_uses_render_stream_not_observed_stream(self):
        track = SimpleNamespace(
            track_id=3,
            misses=2,
            xyxy=np.array([8.0, 8.0, 24.0, 32.0]),
        )
        tick = {
            "frames": {"a": np.zeros((48, 64, 3), np.uint8)},
            "camera_tracks": {"a": []},
            "camera_render_tracks": {"a": [track]},
            "global_tracks": [],
            "global_render_tracks": [],
            "local_to_gid": {("a", 3): 7},
        }

        frame = composite(tick, FakeBev(), ["a"])

        # The predicted box top edge is around y=45 after 480x270 resize.
        self.assertGreater(np.count_nonzero(frame[40:51, 55:186]), 0)

    def test_missing_camera_renders_blank_cell(self):
        tick = {"frames": {"a": np.zeros((48, 64, 3), np.uint8)},
                "camera_tracks": {}, "global_tracks": [], "local_to_gid": {}}
        frame = composite(tick, FakeBev(), ["a", "b"])
        self.assertEqual(frame.shape, (270, 1500, 3))

    @patch("multicam_tracker.render_preview.BevRenderer", FakeBev)
    def test_mp4_uses_source_fps_and_gif_is_streamed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = root / "custom" / "camera_0001"
            camera.mkdir(parents=True)
            video = cv2.VideoWriter(str(camera / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24., (64, 48))
            self.assertTrue(video.isOpened())
            for index in range(48):
                video.write(np.full((48, 64, 3), index, np.uint8))
            video.release()

            def pipeline(cfg, max_frames):
                self.assertEqual(max_frames, 48)
                for index in range(max_frames):
                    yield {"frame_index": index, "calibrations": {},
                           "frames": {"camera_0001": np.full((48, 64, 3), index, np.uint8)},
                           "camera_tracks": {}, "global_tracks": [], "local_to_gid": {}}

            out, gif = root / "out.mp4", root / "out.gif"
            with patch("multicam_tracker.render_preview.run_pipeline", pipeline):
                render_video(data_root=root, out=out, seconds=2, gif=gif, gif_fps=8)
            capture = cv2.VideoCapture(str(out))
            try:
                self.assertTrue(capture.isOpened())
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 24.)
                self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_COUNT), 48)
                ok, frame = capture.read()
                self.assertTrue(ok)
                self.assertEqual(frame.shape, (270, 1500, 3))
            finally:
                capture.release()
            with Image.open(gif) as image:
                self.assertEqual(image.n_frames, 16)
                for index in range(image.n_frames):
                    image.seek(index)
                    image.load()


if __name__ == "__main__":
    unittest.main()
