import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from multicam_tracker.config import PipelineConfig
from multicam_tracker.reporting import build_run_metadata, validate_json_metadata


class ReportingTests(unittest.TestCase):
    def test_metadata_serializes_paths_and_nonfinite_config(self) -> None:
        with TemporaryDirectory() as root:
            data = Path(root) / "data"
            camera = data / "custom" / "camera_0001"
            camera.mkdir(parents=True)
            (camera / "video.mp4").write_bytes(b"video")
            (camera / "calibration.json").write_text("{}", encoding="utf-8")
            (data / "custom" / "ground_truth.txt").write_text("", encoding="utf-8")
            model = Path(root) / "model.pt"
            model.write_bytes(b"weights")
            cfg = PipelineConfig(
                data_root=data,
                cameras=["camera_0001"],
                model_name=str(model),
            )
            cfg.clock_max_pair_delta_spread_s = float("inf")

            with (
                patch("subprocess.run", side_effect=subprocess.SubprocessError),
                patch.dict(
                    "os.environ",
                    {
                        "SOURCE_GIT_SHA": "remote-sha",
                        "SOURCE_GIT_BRANCH": "hybrid",
                        "SOURCE_GIT_DIRTY": "0",
                    },
                ),
            ):
                metadata = build_run_metadata(cfg, repository_root=Path(root))

            validate_json_metadata(metadata)
            self.assertEqual(metadata["config"]["data_root"], str(data))
            self.assertEqual(
                metadata["config"]["clock_max_pair_delta_spread_s"], "Infinity"
            )
            self.assertEqual(
                metadata["inputs"]["model"]["sha256"],
                "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c",
            )
            self.assertEqual(metadata["repository"]["commit_sha"], "remote-sha")
            self.assertFalse(metadata["repository"]["dirty"])
            json.dumps(metadata, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
