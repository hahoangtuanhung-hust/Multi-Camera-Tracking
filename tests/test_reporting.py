import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from multicam_tracker.config import PipelineConfig
from multicam_tracker.reporting import (
    build_run_metadata,
    canonical_replay_tick_bytes,
    validate_json_metadata,
)


class ReportingTests(unittest.TestCase):
    def test_replay_tick_serialization_is_order_independent_and_exact(self) -> None:
        first = {
            "frame_index": 7,
            "global_tracks": [
                SimpleNamespace(gid=2, world_xy=np.array([2.0, 1.0])),
                SimpleNamespace(gid=1, world_xy=np.array([1.0, 1.0])),
            ],
            "local_to_gid": {("b", 2): 2, ("a", 1): 1},
            "local_to_internal_gid": {("b", 2): 2, ("a", 1): 1},
            "accepted_fusion_keys": {("b", 2), ("a", 1)},
            "gid_aliases": {4: 2, 3: 1},
        }
        reordered = {
            **first,
            "global_tracks": list(reversed(first["global_tracks"])),
            "local_to_gid": {("a", 1): 1, ("b", 2): 2},
            "gid_aliases": {3: 1, 4: 2},
        }
        self.assertEqual(
            canonical_replay_tick_bytes(first),
            canonical_replay_tick_bytes(reordered),
        )

        changed = dict(first)
        changed["global_tracks"] = [
            SimpleNamespace(gid=1, world_xy=np.array([1.0, 1.0])),
            SimpleNamespace(gid=2, world_xy=np.array([2.0, 1.001])),
        ]
        self.assertNotEqual(
            canonical_replay_tick_bytes(first)[0],
            canonical_replay_tick_bytes(changed)[0],
        )

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

            def unavailable_git(*args, **kwargs):
                raise FileNotFoundError

            with (
                patch.dict(
                    "os.environ",
                    {
                        "SOURCE_GIT_SHA": "remote-sha",
                        "SOURCE_GIT_BRANCH": "hybrid",
                        "SOURCE_GIT_DIRTY": "0",
                        "SOURCE_TREE_SHA256": "remote-tree-sha",
                    },
                ),
            ):
                metadata = build_run_metadata(
                    cfg,
                    repository_root=Path(root),
                    git_runner=unavailable_git,
                )

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
            self.assertEqual(
                metadata["repository"]["source_fingerprint"]["sha256"],
                "remote-tree-sha",
            )
            json.dumps(metadata, allow_nan=False)

    def test_dirty_source_fingerprint_changes_with_worktree_bytes(self) -> None:
        with TemporaryDirectory() as root:
            root_path = Path(root)
            source = root_path / "src" / "multicam_tracker" / "module.py"
            source.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n", encoding="utf-8")

            def fake_git(command, *args, **kwargs):
                values = {
                    ("rev-parse", "HEAD"): "abc123\n",
                    ("branch", "--show-current"): "hybrid\n",
                    ("status", "--porcelain"): " M src/multicam_tracker/module.py\n",
                    (
                        "ls-files",
                        "--cached",
                        "--others",
                        "--exclude-standard",
                    ): "src/multicam_tracker/module.py\n",
                }
                return type("Result", (), {"stdout": values[tuple(command[1:])]})()

            cfg = PipelineConfig(data_root=root_path / "data", cameras=[])
            before = build_run_metadata(
                cfg, repository_root=root_path, git_runner=fake_git
            )["repository"]["source_fingerprint"]
            source.write_text("VALUE = 2\n", encoding="utf-8")
            after = build_run_metadata(
                cfg, repository_root=root_path, git_runner=fake_git
            )["repository"]["source_fingerprint"]

            self.assertNotEqual(before["sha256"], after["sha256"])
            self.assertEqual(after["file_count"], 1)


if __name__ == "__main__":
    unittest.main()
