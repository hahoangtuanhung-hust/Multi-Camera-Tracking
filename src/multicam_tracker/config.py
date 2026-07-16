"""Central configuration for the pipeline.

Every tunable lives here so readers can see, in one place, which knobs the
system has and what they trade off. Defaults are chosen for the bundled
NVIDIA PhysicalAI-SmartSpaces warehouse scene (4 cameras, people only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# The Hugging Face dataset the demo runs on (downloaded by `download-data`).
HF_DATASET = "nvidia/PhysicalAI-SmartSpaces"
SCENE_PATH = "MTMC_Tracking_2024/train/scene_001"

# Default 4-camera subset: spread around the warehouse for good overlap
# while keeping the download ~1 GB. Use `download-data --full` for all 10.
DEFAULT_CAMERAS = ["camera_0001", "camera_0003", "camera_0006", "camera_0009"]


@dataclass
class PipelineConfig:
    # --- data ---
    data_root: Path = Path("data")
    cameras: list[str] = field(default_factory=lambda: list(DEFAULT_CAMERAS))

    # --- video ---
    # Process every `stride`-th frame. The videos are 30 fps; stride=2 means
    # the pipeline sees 15 fps, which is plenty for tracking people and keeps
    # the demo fluid on a laptop.
    stride: int = 2

    # --- detection ---
    model_name: str = "yolo26n.pt"  # NMS-free end-to-end YOLO, nano size
    device: str = "mps"             # "mps" on Apple Silicon, "cpu" fallback
    imgsz: int = 640
    conf_threshold: float = 0.25    # low floor feeds ByteTrack's rescue stage
    person_class_id: int = 0        # COCO class 0 == person

    # --- single-camera tracking (bytetrack.py) ---
    # At stride=2 on 30 fps video the pipeline runs at 15 ticks/s.
    track_high_conf: float = 0.5    # first-pass association + track-birth threshold
    track_min_iou: float = 0.25    # gate: reject matches with IoU below this
    track_max_age: int = 30         # ticks a track survives unmatched (~2 s occlusion)
    track_min_hits: int = 3         # detections before a track is confirmed
    track_kf_process_noise: float = 2.0       # box motion uncertainty (px² units)
    track_kf_measurement_noise: float = 10.0  # detector box jitter (px² units)

    # --- cross-camera fusion (fusion.py) ---
    fusion_gate_m: float = 2.0      # max world-plane distance (meters) to match
    fusion_merge_m: float = 1.0     # cluster radius for births from orphans
    fusion_max_age: int = 45        # ticks a global track survives unseen (~3 s)
    fusion_min_hits: int = 3        # observations before a global track shows
    fusion_kf_process_noise: float = 0.05      # walking-speed changes (m² units)
    fusion_kf_measurement_noise: float = 0.05  # projected foot-point jitter (m² units)

    # --- visualization ---
    trail_length: int = 60          # BEV trail length in ticks (~4 s at stride 2)
    jpeg_quality: int = 70          # camera feed compression for the viewer

    @property
    def scene_dir(self) -> Path:
        return self.data_root / SCENE_PATH
