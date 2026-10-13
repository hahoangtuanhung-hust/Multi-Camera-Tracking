"""Central configuration for the pipeline.

Every tunable lives here so readers can see, in one place, which knobs the
system has and what they trade off. Defaults are chosen for the bundled
NVIDIA PhysicalAI-SmartSpaces warehouse scene (4 cameras, people only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os

# The Hugging Face dataset the demo runs on (downloaded by `download-data`).
HF_DATASET = "nvidia/PhysicalAI-SmartSpaces"
# SCENE_PATH = "MTMC_Tracking_2024/train/scene_001"
SCENE_PATH = "custom"
# Default 4-camera subset: spread around the warehouse for good overlap
# while keeping the download ~1 GB. Use `download-data --full` for all 10.
DEFAULT_CAMERAS = ["camera_0001", "camera_0007", "camera_0008", "camera_0010"]


@dataclass
class PipelineConfig:
    # --- data ---
    data_root: Path = Path("data")
    cameras: list[str] = field(default_factory=lambda: list(DEFAULT_CAMERAS))

    # --- video ---
    # Process every `stride`-th frame. Keep the runtime continuous by default:
    # stride=1 sends every decoded frame through detection and tracking.
    # Higher values are an explicit performance trade-off selected by the user.
    stride: int = 1
    # Dataset-specific trimming belongs in configuration, never in the
    # decoder. Values are source-frame counts (before ``stride`` is applied).
    camera_start_frame_offsets: dict[str, int] = field(default_factory=lambda: {
        "camera_0001": 30,
        "camera_0007": 60,
    })
    # Convert a camera clock into the common event-time domain with
    # ``t_common = t_camera * scale + offset``. Unknown cameras use 1/0.
    camera_clock_offsets_s: dict[str, float] = field(default_factory=dict)
    camera_clock_scales: dict[str, float] = field(default_factory=dict)
    async_ingest: bool = True      # offline evaluator/renderer opt into lockstep
    ingest_queue_size: int = 2
    microbatch_ms: float = 20.0
    decoder_threads: int = 1       # per camera; avoid N cameras x N CPU threads

    # --- detection ---
    model_name: str = "yolo26n.pt"  # NMS-free end-to-end YOLO, nano size
    device: str = "cpu"             # portable default; opt into "mps"/"cuda"
    imgsz: int = 640
    conf_threshold: float = 0.25    # low floor feeds ByteTrack's rescue stage
    person_class_id: int = 0        # COCO class 0 == person
    cpu_threads: int = field(default_factory=lambda: min(4, os.cpu_count() or 1))
    opencv_threads: int = 1

    # --- single-camera tracking (bytetrack.py) ---
    # At the default stride=1 on 30 fps video the pipeline receives 30 ticks/s.
    track_high_conf: float = 0.5    # first-pass association + track-birth threshold
    track_min_iou: float = 0.25    # gate: reject matches with IoU below this
    track_max_age: int = 60         # legacy tick fallback (~2 s at 30 fps)
    track_max_age_s: float = 2.0    # used when capture timestamps are supplied
    track_min_hits: int = 3         # detections before a track is confirmed
    track_kf_process_noise: float = 2.0       # box motion uncertainty (px² units)
    track_kf_time_process_noise: float = 2.0  # separately tunable timestamped px/s variance
    track_kf_measurement_noise: float = 10.0  # detector box jitter (px² units)
    # Optional experimental controls. Real-scene evaluation keeps them neutral
    # by default: aggressive adaptation/outlier damping reduced IoU continuity.
    track_kf_adaptive_noise: bool = False
    track_kf_reference_height_px: float = 100.0
    track_kf_noise_scale_min: float = 0.25
    track_kf_noise_scale_max: float = 4.0
    track_kf_confidence_noise_gain: float = 0.0
    track_kf_robust_nis_threshold: float = 0.0  # 0 disables robust down-weighting
    # CPU-cheap motion-aware association.  IoU remains the dominant cue for
    # visible tracks; Mahalanobis distance lets a recently lost track recover
    # after a short occlusion without accepting an arbitrary far-away box.
    # Kept behind the ``identity-full`` experimental profile: long-sequence
    # evaluation showed that low-IoU reactivation fragments this scene more
    # often than it recovers a true occlusion.
    track_motion_association_enabled: bool = False
    track_association_nis_gate: float = 13.28  # 99% chi-square gate, 4 DoF
    track_association_iou_weight: float = 0.60
    track_association_nis_weight: float = 0.25
    track_association_size_weight: float = 0.10
    track_association_direction_weight: float = 0.05
    track_association_max_size_log_change: float = 1.10
    track_motion_reactivation_s: float = 1.0
    track_motion_min_speed: float = 0.10

    # --- cross-camera fusion (fusion.py) ---
    fusion_gate_m: float = 2.0      # max world-plane distance (meters) to match
    fusion_merge_m: float = 1.0     # cluster radius for births from orphans
    fusion_max_age: int = 90        # legacy tick fallback (~3 s at 30 fps)
    fusion_max_age_s: float = 3.0   # normal short-gap association horizon
    # A hidden identity may remain available longer, but can only be revived
    # after the short horizon by strict motion + image-boundary evidence.
    fusion_identity_retention_s: float = 3.0
    fusion_output_grace_s: float = 0.25  # retain IDs longer than we publish stale positions
    # A new global identity is already hidden until ``fusion_min_hits``.  This
    # optional additional event-time hold is useful for ablation without
    # forcing latency/recall loss in the default real-time profile.
    fusion_provisional_buffer_enabled: bool = False
    fusion_provisional_min_age_s: float = 0.0
    fusion_batch_reconnect_enabled: bool = False
    # Experimental. Repeated spatial confirmation after a motion-gate failure
    # increased short-horizon ID switches in the real-scene ablation.
    fusion_adaptive_hysteresis_enabled: bool = False
    fusion_published_motion_gate_enabled: bool = False
    fusion_duplicate_suppression_enabled: bool = False
    fusion_instant_merge_enabled: bool = True
    fusion_event_time_gate_s: float = 0.15
    fusion_instant_margin_m: float = 0.20
    # Optional crossing controls. The bundled scene evaluates best with
    # position assignment; enable motion/confirmation only after camera-specific
    # tuning with ``evaluate`` or ``benchmarks/evaluate_thresholds.py``.
    fusion_crossing_guard_m: float = 0.75
    fusion_instant_min_speed_mps: float = 0.15
    fusion_instant_velocity_gate_mps: float = 1.50
    fusion_instant_direction_cosine: float = 0.20
    # Default to geometry assignment plus a strong opposite-direction veto.
    # A positive weight opts into soft motion re-ranking (useful for tuning,
    # but less stable with noisy local IDs in the bundled real scene).
    fusion_instant_velocity_weight_s: float = 0.0
    fusion_instant_opposite_direction_cosine: float = -1.1  # disabled by default
    fusion_motion_window_s: float = 0.60
    fusion_motion_min_span_s: float = 0.20
    # Do not delay clear cross-camera merges: ablation showed that doing so
    # creates duplicate outputs. Consecutive ownership confirmation is applied
    # to local-ID fragment reassignment below, where it reduces ID switches.
    fusion_instant_confirmations: int = 1
    fusion_published_merge_confirmations: int = 1
    fusion_instant_confirmation_window_s: float = 0.35
    # Consolidate duplicate global tracks only after repeated close, compatible
    # motion from disjoint cameras. This is deliberately much tighter than the
    # normal cross-camera birth/merge radius.
    fusion_duplicate_merge_m: float = 0.0
    fusion_duplicate_confirmations: int = 5
    fusion_duplicate_confirmation_window_s: float = 0.35
    fusion_duplicate_velocity_gate_mps: float = 0.75
    fusion_duplicate_suppression_min_confirmations: int = 2
    fusion_local_reconnect_gate_m: float = 1.0
    fusion_local_reconnect_window_s: float = 3.0
    fusion_local_reconnect_long_window_s: float = 3.0
    fusion_local_reconnect_nis_gate: float = 9.21  # 99% chi-square gate, 2 DoF
    fusion_local_reconnect_covariance_enabled: bool = False
    fusion_local_reconnect_confirmations: int = 1
    fusion_local_reconnect_confirmation_window_s: float = 0.35
    fusion_local_reconnect_velocity_weight_s: float = 0.0
    fusion_local_reconnect_velocity_gate_mps: float = 1.25
    fusion_local_reconnect_direction_cosine: float = 0.0
    # Long-gap reconnect is allowed only when the old fragment ended and the
    # new fragment began near an image entry/exit boundary.
    fusion_boundary_margin_ratio: float = 0.06
    fusion_min_hits: int = 3        # observations before a global track shows
    fusion_kf_process_noise: float = 0.05      # walking-speed changes (m² units)
    fusion_kf_measurement_noise: float = 0.05  # projected foot-point jitter (m² units)

    # --- asynchronous cross-camera tracklets ---
    timestamp_fusion_enabled: bool = True
    tracklet_history_s: float = 5.0
    tracklet_late_drop_s: float = 5.0
    tracklet_min_duration_s: float = 1.0
    tracklet_min_overlap_s: float = 0.8
    tracklet_resample_hz: float = 10.0
    tracklet_match_interval_s: float = 0.25
    max_camera_offset_s: float = 3.0
    offset_scan_step_s: float = 0.1
    offset_refine_window_s: float = 0.2
    offset_ewma_alpha: float = 0.15
    offset_min_confidence: float = 0.55
    offset_max_drift_ppm: float = 5_000.0
    offset_min_drift_span_s: float = 5.0

    # Tuned on the custom 4-camera scene after accounting for its 0.2–0.5 m
    # foot-point calibration noise. Assignment margin remains conservative.
    tracklet_position_gate_m: float = 0.85
    tracklet_velocity_gate_mps: float = 0.90
    tracklet_direction_cosine: float = 0.65
    tracklet_assignment_margin: float = 0.20
    tracklet_position_weight: float = 0.60
    tracklet_velocity_weight: float = 0.20
    tracklet_direction_weight: float = 0.15
    tracklet_confidence_weight: float = 0.05
    single_person_mode: bool = False

    # --- visualization ---
    trail_length: int = 120         # BEV trail length in ticks (~4 s at 30 fps)
    jpeg_quality: int = 70          # camera feed compression for the viewer

    @property
    def scene_dir(self) -> Path:
        return self.data_root / SCENE_PATH


_IDENTITY_V2_BASE: dict[str, object] = {
    "track_motion_association_enabled": True,
    "fusion_identity_retention_s": 8.0,
    "fusion_provisional_buffer_enabled": True,
    "fusion_provisional_min_age_s": 0.0,
    "fusion_batch_reconnect_enabled": True,
    "fusion_adaptive_hysteresis_enabled": True,
    "fusion_published_motion_gate_enabled": True,
    "fusion_duplicate_suppression_enabled": True,
    "fusion_duplicate_merge_m": 0.30,
    "fusion_local_reconnect_long_window_s": 8.0,
    "fusion_local_reconnect_window_s": 1.0,
    "fusion_local_reconnect_covariance_enabled": True,
    "fusion_local_reconnect_confirmations": 2,
    "fusion_local_reconnect_velocity_weight_s": 0.15,
}


TRACKING_PROFILE_OVERRIDES: dict[str, dict[str, object]] = {
    # Exact decision-policy compatibility profile for the best known 9k-frame
    # baseline. It keeps capture-time fusion/tracklets but does not introduce
    # the later published-pair motion gate or reconnect covariance gate.
    "identity-v1-baseline": {
        "track_motion_association_enabled": False,
        "fusion_identity_retention_s": 3.0,
        "fusion_provisional_buffer_enabled": False,
        "fusion_batch_reconnect_enabled": False,
        "fusion_adaptive_hysteresis_enabled": False,
        "fusion_published_motion_gate_enabled": False,
        "fusion_duplicate_suppression_enabled": False,
        "fusion_duplicate_merge_m": 0.0,
        "fusion_local_reconnect_long_window_s": 3.0,
        "fusion_local_reconnect_window_s": 3.0,
        "fusion_local_reconnect_covariance_enabled": False,
        "fusion_local_reconnect_confirmations": 1,
        "fusion_local_reconnect_velocity_weight_s": 0.0,
    },
    # Experimental candidate: v1's permissive clear-pair geometry policy plus
    # the new reversible duplicate-output suppression. Irreversible duplicate
    # merge still requires five confirmations. It improved most metrics but
    # missed the strict 9k-frame release gates, so it is not the default.
    "identity-v1-safe": {
        "track_motion_association_enabled": False,
        "fusion_identity_retention_s": 3.0,
        "fusion_provisional_buffer_enabled": False,
        "fusion_batch_reconnect_enabled": False,
        "fusion_adaptive_hysteresis_enabled": False,
        "fusion_published_motion_gate_enabled": False,
        "fusion_duplicate_suppression_enabled": True,
        "fusion_duplicate_merge_m": 0.30,
        "fusion_local_reconnect_long_window_s": 3.0,
        "fusion_local_reconnect_window_s": 3.0,
        "fusion_local_reconnect_covariance_enabled": False,
        "fusion_local_reconnect_confirmations": 1,
        "fusion_local_reconnect_velocity_weight_s": 0.0,
    },
    # Reproduces the decision structure immediately before the identity-v2
    # work, while keeping all code paths available for a fair ablation.
    "pre-identity-v2": {
        **_IDENTITY_V2_BASE,
        "track_motion_association_enabled": False,
        "fusion_provisional_buffer_enabled": False,
        "fusion_batch_reconnect_enabled": False,
        "fusion_adaptive_hysteresis_enabled": False,
        "fusion_published_motion_gate_enabled": True,
        "fusion_duplicate_suppression_enabled": False,
    },
    "motion-only": {
        **_IDENTITY_V2_BASE,
        "fusion_provisional_buffer_enabled": False,
        "fusion_batch_reconnect_enabled": False,
        "fusion_adaptive_hysteresis_enabled": False,
        "fusion_duplicate_suppression_enabled": False,
    },
    "motion-batch": {
        **_IDENTITY_V2_BASE,
        "fusion_adaptive_hysteresis_enabled": False,
        "fusion_duplicate_suppression_enabled": False,
    },
    "motion-batch-adaptive": {
        **_IDENTITY_V2_BASE,
        "fusion_duplicate_suppression_enabled": False,
    },
    # Conservative v2 ablation: retain the validated local ByteTrack association, use
    # order-independent fragment assignment and hide repeatedly confirmed
    # duplicates. Adaptive motion softening stays experimental because its
    # A/B ablation increased ID switches.
    "fusion-identity-safe": {
        **_IDENTITY_V2_BASE,
        "track_motion_association_enabled": False,
        "fusion_adaptive_hysteresis_enabled": False,
    },
    "identity-full": {
        **_IDENTITY_V2_BASE,
    },
    "identity-full-buffer-020": {
        **_IDENTITY_V2_BASE,
        "fusion_provisional_min_age_s": 0.20,
    },
}


def apply_tracking_profile(cfg: PipelineConfig, profile: str) -> PipelineConfig:
    """Apply a named, explicit ablation profile in-place."""
    if profile not in TRACKING_PROFILE_OVERRIDES:
        choices = ", ".join(sorted(TRACKING_PROFILE_OVERRIDES))
        raise ValueError(f"Unknown tracking profile {profile!r}; choose one of: {choices}")
    for name, value in TRACKING_PROFILE_OVERRIDES[profile].items():
        setattr(cfg, name, value)
    return cfg
