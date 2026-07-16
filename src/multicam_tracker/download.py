"""Download the demo dataset from Hugging Face.

Grabs a 4-camera subset (~1.1 GB) of NVIDIA's PhysicalAI-SmartSpaces
MTMC_Tracking_2024 scene_001 — synchronized 1080p warehouse cameras with
per-camera calibration and world-coordinate ground truth. The dataset is
CC-BY-4.0 and downloads anonymously (no account needed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import DEFAULT_CAMERAS, HF_DATASET, SCENE_PATH

ALL_CAMERAS = [f"camera_{i:04d}" for i in range(1, 11)]


def download(data_root: Path, cameras: list[str]) -> Path:
    patterns = [f"{SCENE_PATH}/{cam}/*" for cam in cameras]
    patterns += [
        f"{SCENE_PATH}/calibration_2025_format.json",
        f"{SCENE_PATH}/ground_truth.txt",
    ]
    snapshot_download(
        repo_id=HF_DATASET,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=data_root,
    )
    return data_root / SCENE_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Directory to download into (default: ./data)",
    )
    parser.add_argument(
        "--cameras", nargs="+", default=DEFAULT_CAMERAS, metavar="CAM",
        help=f"Camera folders to fetch (default: {' '.join(DEFAULT_CAMERAS)})",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Fetch all 10 cameras (~2.6 GB) instead of the default subset",
    )
    args = parser.parse_args()

    cameras = ALL_CAMERAS if args.full else args.cameras
    scene = download(args.data_root, cameras)
    print(f"\nDataset ready at {scene}")
    print(f"Cameras: {', '.join(cameras)}")


if __name__ == "__main__":
    main()
