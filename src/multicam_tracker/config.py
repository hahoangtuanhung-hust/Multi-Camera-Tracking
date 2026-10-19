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
    conf_threshold: float = 0.25    # low floor can feed OC-SORT's BYTE recovery stage
    person_class_id: int = 0        # COCO class 0 == person
    # FP16 is CUDA-only and opt-in until a target-GPU A/B clears both the
    # throughput and identity-quality gates. CPU/MPS always stay on FP32.
    detector_fp16: bool = False
    # torch.compile has a material warm-up cost and remains opt-in until a
    # deployment-length benchmark amortizes it.
    detector_compile: bool = False
    cpu_threads: int = field(default_factory=lambda: min(4, os.cpu_count() or 1))
    opencv_threads: int = 1

    # --- single-camera tracking ---
    # OC-SORT is the default local tracker.  The old simplified ByteTrack
    # implementation remains selectable for reproducible A/B evaluation.
    local_tracker: str = "ocsort"
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
    # Low-score detections may recover an existing track, but never create a
    # new one. Their association is additionally gated by the box-space
    # Kalman covariance because metres/second are not defined before the
    # camera homography projects the foot point onto the world plane.
    track_detection_temporal_confirmation_frames: int = 2
    # Tentative tracks normally disappear on their first miss. Candidate
    # profiles may retain them briefly so one dropped detection does not force
    # a new local ID at an occlusion/crossing.
    track_tentative_max_misses: int = 0
    track_detection_low_conf_recovery_enabled: bool = True
    track_motion_gate_enabled: bool = True
    track_max_speed_mps: float = 3.20
    # OC-SORT uses detector observations, rather than only Kalman predictions,
    # to preserve motion direction through crossings and repair the state after
    # an occlusion. ``delta_t`` is the observation look-back in camera ticks.
    ocsort_delta_t: int = 3
    ocsort_inertia: float = 0.10
    ocsort_use_byte: bool = True
    # OCR is a cheap second association against the last real observation.
    ocsort_observation_recovery_enabled: bool = True
    # ORU is retained for explicit ablation, but this scene's 300-tick A/B
    # showed persistent wrong ownership despite fewer raw ID switches. Keep it
    # off until a long-sequence IDF1/HOTA evaluation proves otherwise.
    ocsort_oru_enabled: bool = False
    # A one-frame merged detection is common at crossings. Rebuilding the
    # filter from it can lock in the wrong direction, so ORU starts only after
    # a genuine multi-frame observation gap.
    ocsort_oru_min_misses: int = 2
    # Hybrid-SORT: weight for last-observation IoU in the two-round blended
    # association. 0.0 degenerates to OC-SORT; 1.0 ignores Kalman predictions.
    # Paper default is ~0.35–0.45 for appearance-free settings.
    hybridsort_alpha: float = 0.40
    # Maximum time elapsed since last real observation to be eligible for
    # observation-IoU recovery. Prevents stale tracks from stealing new identities.
    hybridsort_max_obs_gap_s: float = 0.80
    # Minimum score required to accept a blended association in round 2.
    hybridsort_min_blended_iou: float = 0.25
    # At a crossing, one merged person box can overlap two established tracks
    # almost equally. Updating either Kalman state from that box is effectively
    # a coin flip and often swaps ownership after the people separate. Optional
    # ambiguity suppression lets both tracks coast on motion for that tick and
    # prevents the merged box from spawning a third local ID.
    hybridsort_ambiguity_suppression_enabled: bool = False
    hybridsort_ambiguity_min_iou: float = 0.25
    hybridsort_ambiguity_margin: float = 0.08
    hybridsort_ambiguity_min_area_ratio: float = 1.10
    hybridsort_ambiguity_owner_grace_s: float = 0.0
    hybridsort_lost_observation_prediction_enabled: bool = False
    # Experimental A1: keep a matched raw bbox for diagnostics, but do not
    # let a high-NIS merged/outlier box replace trusted direction/history.
    hybridsort_trusted_observation_enabled: bool = False
    hybridsort_trusted_observation_nis_multiplier: float = 1.0
    # Event-triggered appearance association for a confirmed local track that
    # reappears after an observation gap. The descriptor is a compact HSV
    # histogram; no image crops or face features are retained.
    hybridsort_reappearance_enabled: bool = False
    hybridsort_cascade_enabled: bool = False
    hybridsort_causal_association_enabled: bool = False
    hybridsort_causal_motion_rescue_enabled: bool = False
    hybridsort_recovery_max_candidates: int = 64
    hybridsort_recovery_max_hold_s: float = 0.35
    appearance_backend: str = "hsv"
    appearance_model_path: str | None = None
    appearance_model_sha256: str | None = None
    appearance_event_triggered: bool = False
    appearance_quality_filter_enabled: bool = False
    appearance_gallery_size: int = 8
    appearance_min_crop_height: int = 64
    appearance_max_crop_overlap: float = 0.55
    hybridsort_reappearance_max_gap_s: float = 6.0
    hybridsort_reappearance_min_similarity: float = 0.78
    hybridsort_reappearance_min_margin: float = 0.08
    hybridsort_reappearance_appearance_weight: float = 0.70
    hybridsort_reappearance_confirmations: int = 3
    hybridsort_reappearance_confirmation_window_s: float = 0.35
    hybridsort_appearance_update_alpha: float = 0.15
    hybridsort_appearance_update_min_similarity: float = 0.55

    # Identity retention and drawing are separate contracts. Confirmed lost
    # tracks may remain available for reassociation without being emitted as
    # observed detections. Renderers may optionally show a short dashed
    # prediction while evaluators continue to consume observed tracks only.
    render_occluded_tracks_enabled: bool = False
    render_occluded_max_age_s: float = 2.0

    # --- cross-camera fusion (fusion.py) ---
    fusion_gate_m: float = 2.0      # max world-plane distance (meters) to match
    fusion_merge_m: float = 1.0     # cluster radius for births from orphans
    fusion_max_age: int = 90        # legacy tick fallback (~3 s at 30 fps)
    fusion_max_age_s: float = 3.0   # normal short-gap association horizon
    # A hidden identity may remain available longer, but can only be revived
    # after the short horizon by strict motion + image-boundary evidence.
    fusion_identity_retention_s: float = 5.0
    fusion_output_grace_s: float = 0.25  # retain IDs longer than we publish stale positions
    # Live serving uses arrival time for freshness/latency. Deterministic
    # offline evaluation and replay explicitly use event time instead so CPU
    # and GPU processing speed cannot change which identities are emitted.
    fusion_output_time_policy: str = "arrival"  # "arrival" (live) or "event" (offline)
    # A new global identity is already hidden until ``fusion_min_hits``.  This
    # optional additional event-time hold is useful for ablation without
    # forcing latency/recall loss in the default real-time profile.
    fusion_provisional_buffer_enabled: bool = False
    fusion_provisional_min_age_s: float = 0.0
    # Do not delay identities already present when a run starts. After this
    # bootstrap interval, the normal decision buffer applies to every birth.
    fusion_provisional_buffer_bootstrap_s: float = 0.0
    # When enabled, a new fragment receives an internal negative identifier.
    # A public positive GID is allocated only after the decision buffer and
    # entry/capacity checks pass.  This prevents short false fragments from
    # consuming externally visible identities.
    fusion_deferred_gid_allocation_enabled: bool = False
    # Late identity binding keeps a fresh fragment private while an existing
    # public owner remains kinematically reachable. Offline consumers may also
    # delay committing lightweight identity snapshots so a later alias can be
    # applied retroactively without buffering image frames.
    fusion_late_identity_binding_enabled: bool = False
    fusion_identity_commit_delay_s: float = 0.0
    # Opt-in lifecycle accounting and bounded terminal decisions. Legacy
    # profiles keep their output contract for controlled comparisons.
    fusion_identity_lifecycle_enabled: bool = False
    fusion_terminal_binding_enabled: bool = False
    fusion_terminal_batch_enabled: bool = False
    fusion_terminal_grace_s: float = 1.0
    fusion_terminal_confirmation_window_s: float = 0.35
    fusion_terminal_confirmations: int = 3
    fusion_terminal_max_cost: float = 1.0
    fusion_terminal_margin: float = 0.20
    fusion_terminal_nis_gate: float = 9.21
    fusion_lifecycle_report_limit: int = 10000
    fusion_late_identity_max_defer_s: float = 0.0
    # 0 keeps the population unconstrained. A positive value caps concurrently
    # retained public identities; pending fragments remain hidden until a slot
    # is released by a confirmed exit.
    fusion_fixed_population_size: int = 0
    fusion_new_gid_entry_required: bool = False
    # During startup, people may already be inside the scene rather than at an
    # image boundary, so seed the initial population without entry evidence.
    fusion_new_gid_bootstrap_s: float = 2.5
    # Optional escape hatch for a stable trajectory first detected inside the
    # scene (for example after a long detector miss). Entry evidence remains
    # preferred; zero keeps the strict closed-world policy.
    fusion_interior_birth_fallback_s: float = 0.0
    # After the interior fallback is ready, a merely reachable dormant GID may
    # defer publication only this much longer. Zero preserves strict/unbounded
    # waiting for baseline profiles.
    fusion_reachable_dormant_max_defer_s: float = 0.0
    # Public identities that disappear inside the scene stay dormant for
    # reconnect. Boundary evidence releases them normally; the forced timeout
    # is a bounded-memory safety valve for imperfect boundary calibration.
    fusion_retire_requires_exit_boundary: bool = False
    fusion_forced_retire_s: float = 30.0
    # Verification of trajectory plausibility before promoting a provisional track to a public GID
    fusion_trajectory_verification_enabled: bool = False
    fusion_trajectory_min_points: int = 4
    fusion_trajectory_min_duration_s: float = 0.30
    fusion_trajectory_max_jump_m: float = 1.60
    fusion_trajectory_max_speed_mps: float = 3.20
    # Hard event-time feasibility gate. Within the short jump window, the
    # absolute cap dominates even when Kalman covariance grows rapidly.
    fusion_jump_window_s: float = 1.0
    fusion_max_jump_m: float = 2.0
    fusion_max_speed_mps: float = 3.2
    fusion_max_acceleration_mps2: float = 4.0
    fusion_motion_uncertainty_sigma: float = 3.0
    fusion_motion_measurement_tolerance_m: float = 0.25
    # Multi-point trajectory cost used for dormant/interior reassociation.
    fusion_trajectory_window_s: float = 2.0
    fusion_trajectory_smoothing_enabled: bool = False
    fusion_robust_trajectory_motion_enabled: bool = False
    fusion_trajectory_recency_half_life_s: float = 0.75
    fusion_trajectory_outlier_gate_m: float = 0.75
    fusion_trajectory_min_acceleration_duration_s: float = 1.0
    fusion_trajectory_position_weight: float = 1.0
    fusion_trajectory_velocity_weight: float = 0.5
    fusion_trajectory_direction_weight: float = 1.0
    fusion_trajectory_acceleration_weight: float = 0.25
    fusion_trajectory_min_direction_cosine: float = 0.0
    # World coordinate entry/exit zones on the 2D ground plane (Z=0, in meters)
    # Each zone can be [xmin, xmax, ymin, ymax] or a polygon [(x1,y1), ...]
    fusion_world_entry_zones: list[
        tuple[float, float, float, float] | list[tuple[float, float]]
    ] = field(default_factory=list)
    fusion_world_exit_zones: list[
        tuple[float, float, float, float] | list[tuple[float, float]]
    ] = field(default_factory=list)
    # Dormant Identity Pool: Maximum retention for occluded identities inside the scene
    fusion_dormant_retention_s: float = 8.0
    # Kinematic matching gate and weights for reconnecting dormant tracks
    fusion_kinematic_match_gate_m: float = 2.5
    fusion_kinematic_direction_weight: float = 1.0
    fusion_kinematic_speed_weight: float = 0.5
    # Anti-PingPong Hysteresis: lock rapid 1-frame alternations between nearby tracks
    fusion_anti_pingpong_enabled: bool = True
    fusion_anti_pingpong_frames: int = 5
    fusion_anti_pingpong_distance_m: float = 1.2
    fusion_batch_reconnect_enabled: bool = False
    # Experimental. Repeated spatial confirmation after a motion-gate failure
    # increased short-horizon ID switches in the real-scene ablation.
    fusion_adaptive_hysteresis_enabled: bool = False
    fusion_published_motion_gate_enabled: bool = False
    fusion_duplicate_suppression_enabled: bool = False
    fusion_instant_merge_enabled: bool = True
    # Experimental A3: predict both observations to one clock-normalized
    # timestamp before the instant geometry distance is evaluated.
    fusion_common_timestamp_instant_merge_enabled: bool = False
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
    # Same-camera appearance is used only as a reconnect veto/tie-breaker and
    # to protect a public GID from a local-ID takeover. Cross-camera colour
    # descriptors are intentionally never compared.
    fusion_reconnect_appearance_enabled: bool = False
    fusion_reconnect_appearance_min_similarity: float = 0.72
    fusion_reconnect_appearance_weight_m: float = 0.50
    fusion_appearance_update_alpha: float = 0.15
    fusion_appearance_ownership_guard_enabled: bool = False
    fusion_appearance_conflict_min_similarity: float = 0.50
    fusion_appearance_conflict_confirmations: int = 2
    fusion_appearance_conflict_window_s: float = 0.35
    fusion_stationary_reconnect_enabled: bool = False
    fusion_stationary_speed_mps: float = 0.20
    fusion_stationary_reconnect_max_gap_s: float = 6.0
    # When enabled, a new local fragment cannot take ownership of an already
    # published GID from a single plausible observation. It must keep selecting
    # the same canonical candidate for ``fusion_local_reconnect_confirmations``
    # consecutive updates inside the confirmation window.
    fusion_gid_ownership_hysteresis_enabled: bool = False
    # Long-gap reconnect is allowed only when the old fragment ended and the
    # new fragment began near an image entry/exit boundary.
    fusion_boundary_margin_ratio: float = 0.06
    fusion_min_hits: int = 3        # observations before a global track shows
    fusion_kf_process_noise: float = 0.05      # walking-speed changes (m² units)
    fusion_kf_measurement_noise: float = 0.05  # projected foot-point jitter (m² units)

    # --- asynchronous cross-camera tracklets ---
    timestamp_fusion_enabled: bool = True
    tracklet_history_s: float = 7.0
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
    clock_learning_enabled: bool = True
    # Maximum spread between independently matched trajectory shifts in one
    # camera-pair cycle. The selected default prevents spatial calibration
    # bias from masquerading as a clock offset; the baseline profile restores
    # the legacy unlimited value for exact A/B and rollback.
    clock_max_pair_delta_spread_s: float = 0.40
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

    # --- optional bounded identity diagnostics ---
    # JSONL is streamed directly to disk; no unbounded event list is retained.
    identity_diagnostics_path: Path | None = None
    identity_diagnostics_max_events: int = 250_000
    identity_diagnostics_flush_every: int = 128

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
    "fusion_provisional_buffer_bootstrap_s": 0.0,
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
    "fusion_gid_ownership_hysteresis_enabled": False,
    "clock_max_pair_delta_spread_s": float("inf"),
}


TRACKING_PROFILE_OVERRIDES: dict[str, dict[str, object]] = {
    # Exact decision-policy compatibility profile for the best known 9k-frame
    # baseline. It keeps capture-time fusion/tracklets but does not introduce
    # the later published-pair motion gate or reconnect covariance gate.
    "identity-v1-baseline": {
        "track_motion_association_enabled": False,
        "fusion_identity_retention_s": 3.0,
        "fusion_provisional_buffer_enabled": False,
        "fusion_provisional_buffer_bootstrap_s": 0.0,
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
        "fusion_gid_ownership_hysteresis_enabled": False,
        "fusion_instant_confirmations": 1,
        "fusion_published_merge_confirmations": 1,
        "clock_max_pair_delta_spread_s": float("inf"),
    },
    # Experimental candidate: v1's permissive clear-pair geometry policy plus
    # the new reversible duplicate-output suppression. Irreversible duplicate
    # merge still requires five confirmations. It improved most metrics but
    # missed the strict 9k-frame release gates, so it is not the default.
    "identity-v1-safe": {
        "track_motion_association_enabled": False,
        "fusion_identity_retention_s": 3.0,
        "fusion_provisional_buffer_enabled": False,
        "fusion_provisional_buffer_bootstrap_s": 0.0,
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
        "fusion_gid_ownership_hysteresis_enabled": False,
        "fusion_instant_confirmations": 1,
        "fusion_published_merge_confirmations": 1,
        "clock_max_pair_delta_spread_s": float("inf"),
    },
    # ID-drift candidate: preserve the validated v1 identity policy while
    # rejecting camera-clock updates when independently matched people imply
    # mutually inconsistent time shifts. This retains 1-3 s offset learning
    # for a single/consistent path without feeding crossing ambiguity back
    # into subsequent event-time association.
    "identity-v1-drift-guard": {
        "track_motion_association_enabled": False,
        "fusion_identity_retention_s": 3.0,
        "fusion_provisional_buffer_enabled": False,
        "fusion_provisional_buffer_bootstrap_s": 0.0,
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
        "fusion_gid_ownership_hysteresis_enabled": False,
        "fusion_instant_confirmations": 1,
        "fusion_published_merge_confirmations": 1,
        "clock_max_pair_delta_spread_s": 0.40,
    },
    # Keep v1's proven geometry policy and clock guard, but protect externally
    # visible identities. Startup tracks publish normally; later births remain
    # hidden for 0.5 s while a local fragment must select the same published
    # owner three times. Published/published crossing merges require five
    # consecutive observations before the irreversible canonical merge.
    "identity-v1-ownership": {
        "track_motion_association_enabled": False,
        "fusion_identity_retention_s": 3.0,
        "fusion_provisional_buffer_enabled": True,
        "fusion_provisional_min_age_s": 0.50,
        "fusion_provisional_buffer_bootstrap_s": 1.0,
        "fusion_batch_reconnect_enabled": False,
        "fusion_adaptive_hysteresis_enabled": False,
        "fusion_published_motion_gate_enabled": False,
        "fusion_duplicate_suppression_enabled": False,
        "fusion_duplicate_merge_m": 0.0,
        "fusion_local_reconnect_long_window_s": 3.0,
        "fusion_local_reconnect_window_s": 3.0,
        "fusion_local_reconnect_covariance_enabled": False,
        "fusion_local_reconnect_confirmations": 3,
        "fusion_local_reconnect_confirmation_window_s": 0.35,
        "fusion_local_reconnect_velocity_weight_s": 0.0,
        "fusion_gid_ownership_hysteresis_enabled": True,
        "fusion_instant_confirmations": 3,
        "fusion_published_merge_confirmations": 5,
        "clock_max_pair_delta_spread_s": 0.40,
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
    # Experimental fixed-population policy.  New fragments are kept under an
    # internal ID for 0.75 s, allowing motion/tracklet reassociation to absorb
    # local fragmentation before a public GID exists. After startup, a public
    # GID needs boundary entry evidence and an available population slot. A
    # public identity disappearing away from a boundary remains dormant for
    # reconnect instead of immediately freeing a slot.
    "fixed-population-buffered": {
        "track_motion_association_enabled": False,
        "fusion_identity_retention_s": 8.0,
        "fusion_provisional_buffer_enabled": True,
        "fusion_provisional_min_age_s": 0.75,
        "fusion_provisional_buffer_bootstrap_s": 0.0,
        "fusion_deferred_gid_allocation_enabled": True,
        "fusion_fixed_population_size": 25,
        "fusion_new_gid_entry_required": True,
        "fusion_new_gid_bootstrap_s": 2.0,
        "fusion_retire_requires_exit_boundary": True,
        # Strict mode: do not release a public slot merely because time passed.
        # The 25-slot cap keeps this state bounded without a timeout eviction.
        "fusion_forced_retire_s": 0.0,
        "fusion_batch_reconnect_enabled": True,
        "fusion_adaptive_hysteresis_enabled": False,
        "fusion_published_motion_gate_enabled": False,
        "fusion_duplicate_suppression_enabled": True,
        "fusion_duplicate_merge_m": 0.30,
        "fusion_local_reconnect_long_window_s": 8.0,
        "fusion_local_reconnect_window_s": 1.0,
        "fusion_local_reconnect_covariance_enabled": True,
        "fusion_local_reconnect_confirmations": 2,
        "fusion_local_reconnect_velocity_weight_s": 0.15,
        "fusion_gid_ownership_hysteresis_enabled": False,
        "clock_max_pair_delta_spread_s": float("inf"),
    },
    # Hybrid-SORT (no ReID): two-round primary association blending
    # Kalman-IoU with last-observation-IoU. Uses the same fusion identity
    # policy as identity-v1-drift-guard so the comparison is controlled.
    # alpha=0.40: observation IoU accounts for 40% of the blended score;
    # keeping it below 0.5 means Kalman motion continuity still dominates
    # for clearly visible tracks while strongly helping after occlusion.
    "hybridsort": {
        "local_tracker": "hybridsort",
        "track_detection_temporal_confirmation_frames": 2,
        "track_detection_low_conf_recovery_enabled": True,
        "track_motion_gate_enabled": True,
        "track_max_speed_mps": 3.20,
        "track_kf_robust_nis_threshold": 13.28,
        "hybridsort_alpha": 0.40,
        "hybridsort_max_obs_gap_s": 0.80,
        "hybridsort_min_blended_iou": 0.25,
        "ocsort_observation_recovery_enabled": False,
        "track_motion_association_enabled": False,
        "fusion_identity_retention_s": 3.0,
        "fusion_provisional_buffer_enabled": True,
        "fusion_provisional_min_age_s": 0.50,
        "fusion_provisional_buffer_bootstrap_s": 1.5,
        "fusion_deferred_gid_allocation_enabled": True,
        "fusion_new_gid_entry_required": True,
        "fusion_new_gid_bootstrap_s": 2.5,
        "fusion_retire_requires_exit_boundary": True,
        "fusion_trajectory_verification_enabled": True,
        "fusion_trajectory_min_points": 4,
        "fusion_trajectory_min_duration_s": 0.30,
        "fusion_trajectory_max_jump_m": 1.60,
        "fusion_trajectory_max_speed_mps": 3.20,
        "fusion_jump_window_s": 1.0,
        "fusion_max_jump_m": 2.0,
        "fusion_max_speed_mps": 3.20,
        "fusion_max_acceleration_mps2": 4.0,
        "fusion_motion_uncertainty_sigma": 3.0,
        "fusion_motion_measurement_tolerance_m": 0.25,
        "fusion_trajectory_window_s": 3.0,
        "fusion_trajectory_position_weight": 1.0,
        "fusion_trajectory_velocity_weight": 0.5,
        "fusion_trajectory_direction_weight": 1.0,
        "fusion_trajectory_acceleration_weight": 0.25,
        "fusion_trajectory_min_direction_cosine": 0.0,
        "fusion_dormant_retention_s": 8.0,
        "fusion_kinematic_match_gate_m": 2.5,
        "fusion_kinematic_direction_weight": 1.0,
        "fusion_kinematic_speed_weight": 0.5,
        "fusion_anti_pingpong_enabled": True,
        "fusion_anti_pingpong_frames": 5,
        "fusion_anti_pingpong_distance_m": 1.2,
        "fusion_batch_reconnect_enabled": True,
        "fusion_adaptive_hysteresis_enabled": False,
        "fusion_published_motion_gate_enabled": False,
        "fusion_duplicate_suppression_enabled": True,
        "fusion_duplicate_merge_m": 0.30,
        "fusion_local_reconnect_long_window_s": 3.0,
        "fusion_local_reconnect_window_s": 3.0,
        "fusion_local_reconnect_covariance_enabled": False,
        "fusion_local_reconnect_confirmations": 2,
        "fusion_local_reconnect_confirmation_window_s": 0.35,
        "fusion_local_reconnect_velocity_weight_s": 0.0,
        "fusion_gid_ownership_hysteresis_enabled": True,
        "fusion_instant_confirmations": 2,
        "fusion_published_merge_confirmations": 3,
        "clock_max_pair_delta_spread_s": 0.40,
    },
}

# Explicit sequential A/B candidates. The validated ``hybridsort`` profile
# remains untouched until a same-run 2k/4k replay clears the release gates.
TRACKING_PROFILE_OVERRIDES["hybridsort-local-v2"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort"],
    "track_tentative_max_misses": 1,
    "ocsort_observation_recovery_enabled": True,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-detect-recall"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort"],
    # Keep the high-confidence birth threshold unchanged. Detections in
    # [0.15, 0.50) are recovery evidence for existing tracks only.
    "conf_threshold": 0.15,
    # 768 is a measured-cost candidate for small/far people: 1.44x the input
    # pixels of 640, materially cheaper than jumping straight to 960 (2.25x).
    "imgsz": 768,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-low-floor"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort"],
    "conf_threshold": 0.15,
    "imgsz": 640,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-mid-floor"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort"],
    "conf_threshold": 0.20,
    "imgsz": 640,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-tentative"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort"],
    "track_tentative_max_misses": 1,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-crossing-guard"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort"],
    "hybridsort_ambiguity_suppression_enabled": True,
    "hybridsort_ambiguity_min_iou": 0.25,
    "hybridsort_ambiguity_margin": 0.08,
    "hybridsort_ambiguity_min_area_ratio": 1.10,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-continuity"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-crossing-guard"],
    # At 30 fps this retains an unconfirmed trajectory for at most ~33 ms;
    # it reduces duplicate births without keeping false tracks visible.
    "track_tentative_max_misses": 1,
    "hybridsort_ambiguity_suppression_enabled": True,
    "hybridsort_ambiguity_min_iou": 0.25,
    "hybridsort_ambiguity_margin": 0.04,
    "hybridsort_ambiguity_min_area_ratio": 1.10,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-balanced"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-tentative"],
}
TRACKING_PROFILE_OVERRIDES["hybridsort-balanced-low-floor"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-balanced"],
    "conf_threshold": 0.15,
    "imgsz": 640,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-balanced-mid-floor"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-balanced"],
    "conf_threshold": 0.20,
    "imgsz": 640,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-a1-robust-local"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-balanced"],
    "hybridsort_trusted_observation_enabled": True,
    "hybridsort_trusted_observation_nis_multiplier": 4.0,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-a2-trajectory"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-balanced"],
    "fusion_trajectory_smoothing_enabled": True,
    "fusion_robust_trajectory_motion_enabled": True,
    "fusion_trajectory_window_s": 3.5,
    "fusion_trajectory_recency_half_life_s": 0.75,
    "tracklet_history_s": 7.0,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-a2-motion-only"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-balanced"],
    # Isolate the association-side benefit from changing emitted positions.
    "fusion_robust_trajectory_motion_enabled": True,
    "fusion_trajectory_window_s": 3.5,
    "fusion_trajectory_recency_half_life_s": 0.75,
    "tracklet_history_s": 7.0,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-a2-smoothing-only"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-balanced"],
    # Control arm: smooth emitted positions without using robust motion for
    # association.  This distinguishes localization/fragmentation effects.
    "fusion_trajectory_smoothing_enabled": True,
    "fusion_trajectory_window_s": 3.5,
    "fusion_trajectory_recency_half_life_s": 0.75,
    "tracklet_history_s": 7.0,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-a3-common-time"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-balanced"],
    "fusion_common_timestamp_instant_merge_enabled": True,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-a3-clock-fixed"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-balanced"],
    "clock_learning_enabled": False,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-a3-clock-offset-only"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-balanced"],
    "offset_max_drift_ppm": 0.0,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-a4-crossing"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-balanced"],
    "hybridsort_ambiguity_suppression_enabled": True,
    "hybridsort_ambiguity_min_iou": 0.25,
    "hybridsort_ambiguity_margin": 0.04,
    "hybridsort_ambiguity_min_area_ratio": 1.10,
    # Six frames at the dataset's nominal 30 FPS.
    "hybridsort_ambiguity_owner_grace_s": 0.20,
    "hybridsort_lost_observation_prediction_enabled": True,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-a5-lifecycle"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-balanced"],
    "fusion_interior_birth_fallback_s": 1.5,
    "fusion_reachable_dormant_max_defer_s": 1.5,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-late-binding"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-balanced"],
    # Give active/dormant ownership resolution time to converge before a
    # fragment receives an externally visible ID. The commit delay matches
    # the bounded maximum defer so offline output can be canonicalized without
    # losing the provisional portion of a valid trajectory.
    "fusion_identity_retention_s": 5.0,
    "fusion_provisional_min_age_s": 1.0,
    "fusion_late_identity_binding_enabled": True,
    "fusion_identity_commit_delay_s": 2.0,
    "fusion_late_identity_max_defer_s": 2.0,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-late-binding-balanced"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-late-binding"],
    # Preserve recall for valid interior births. A fragment that remains
    # reachable from an existing public owner is still held privately for the
    # bounded late-binding horizon before it can receive a new public ID.
    "fusion_new_gid_entry_required": False,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-crowd-local-reappearance"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-late-binding-balanced"],
    "hybridsort_reappearance_enabled": True,
    "hybridsort_reappearance_max_gap_s": 6.0,
    "hybridsort_reappearance_confirmations": 3,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-crowd-global-guard"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-late-binding-balanced"],
    "fusion_appearance_ownership_guard_enabled": True,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-crowd-stationary-reconnect"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-crowd-global-guard"],
    "fusion_reconnect_appearance_enabled": True,
    "fusion_stationary_reconnect_enabled": True,
    "fusion_local_reconnect_long_window_s": 6.0,
    "fusion_local_reconnect_confirmations": 3,
    "fusion_dormant_retention_s": 10.0,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-crowd-recovery"] = {
    # Compose only the two crowd policies that are neutral on the real-frame
    # screening replay. Stationary reconnect remains an explicit experimental
    # profile because it regressed IDF1/IDSW/fragmentation on that replay.
    **TRACKING_PROFILE_OVERRIDES["hybridsort-crowd-global-guard"],
    **{
        key: value
        for key, value in TRACKING_PROFILE_OVERRIDES[
            "hybridsort-crowd-local-reappearance"
        ].items()
        if key.startswith("hybridsort_reappearance_")
    },
    "render_occluded_tracks_enabled": True,
    "render_occluded_max_age_s": 2.0,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-crowd-recovery-mid-floor"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-crowd-recovery"],
    # Low-score boxes may recover an existing track but remain below the
    # high-confidence threshold required to create a new local identity.
    "conf_threshold": 0.20,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-crowd-lifecycle-audit"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-crowd-recovery"],
    "fusion_identity_lifecycle_enabled": True,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-crowd-terminal"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-crowd-lifecycle-audit"],
    "fusion_terminal_binding_enabled": True,
    # >= decision horizon; ordinary reference remains at two seconds.
    "fusion_identity_commit_delay_s": 3.0,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-crowd-terminal-batch-experimental"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-crowd-terminal"],
    "fusion_terminal_batch_enabled": True,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-crowd-recovery-v2"] = {
    # Controlled 300-tick screening rejected terminal binding and local
    # cascade. Roll back behavior; retain observability only, not a release.
    **TRACKING_PROFILE_OVERRIDES["hybridsort-late-binding-balanced"],
    "fusion_identity_lifecycle_enabled": True,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-crowd-local-cascade"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-crowd-recovery"],
    "hybridsort_cascade_enabled": True,
    "hybridsort_ambiguity_suppression_enabled": True,
    "hybridsort_ambiguity_margin": 0.04,
    "hybridsort_ambiguity_owner_grace_s": 0.20,
    "hybridsort_trusted_observation_enabled": True,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-crowd-recovery-low-floor"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-crowd-recovery"],
    "conf_threshold": 0.15,
}
# Explicit speed/quality ablation. The 2026-10-19 four-camera CPU audit found
# +26% throughput but -19.8% IDF1, +56% IDSW, and +154% fragmentation over
# 300 ticks. Keep it opt-in; it is not a release-quality default.
#   uv run demo --profile hybridsort-realtime --stride 2 --cpu-threads 4
TRACKING_PROFILE_OVERRIDES["hybridsort-realtime"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort"],
    "imgsz": 480,
    "conf_threshold": 0.20,
}
# IDF1-boost profile: crowd-recovery base + crossing ambiguity suppression +
# higher observation-IoU weight + extended identity/dormant retention windows.
# Best for scenes with frequent occlusions and crossings. No neural ReID model
# needed — uses HSV histogram gallery from appearance.py.
#   uv run demo --profile hybridsort-idf1-boost --stride 3
TRACKING_PROFILE_OVERRIDES["hybridsort-idf1-boost"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-crowd-recovery"],
    # Tăng weight last-observation IoU để phục hồi tốt hơn sau occlusion ngắn
    "hybridsort_alpha": 0.50,
    # Crossing guard: không update Kalman từ merged box khi 2 người giao nhau
    "hybridsort_ambiguity_suppression_enabled": True,
    "hybridsort_ambiguity_min_iou": 0.25,
    "hybridsort_ambiguity_margin": 0.04,
    "hybridsort_ambiguity_min_area_ratio": 1.10,
    "hybridsort_ambiguity_owner_grace_s": 0.20,
    # Giữ GID lâu hơn trước khi expire, giảm ID switch sau occlusion
    "fusion_identity_retention_s": 5.0,
    # Dormant track giữ lâu hơn cho warehouse (người khuất sau kệ)
    "fusion_dormant_retention_s": 10.0,
    # Extend reconnect window để vớt lại fragment sau gap dài
    "fusion_local_reconnect_long_window_s": 5.0,
    "fusion_local_reconnect_confirmations": 3,
    # Grace period: không retire ngay khi track mất signal
    "fusion_output_grace_s": 0.5,
    # Mid-floor: dùng low-conf detections để recover existing tracks
    "conf_threshold": 0.20,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-trajectory"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort"],
    "fusion_trajectory_smoothing_enabled": True,
    "fusion_robust_trajectory_motion_enabled": True,
    "fusion_trajectory_window_s": 3.5,
    "fusion_trajectory_recency_half_life_s": 0.75,
    "tracklet_history_s": 7.0,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-stable"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-local-v2"],
    "fusion_trajectory_smoothing_enabled": True,
    "fusion_robust_trajectory_motion_enabled": True,
    "fusion_trajectory_window_s": 3.5,
    "fusion_trajectory_recency_half_life_s": 0.75,
    "tracklet_history_s": 7.0,
    "fusion_identity_retention_s": 5.0,
    "fusion_output_grace_s": 0.75,
    "fusion_dormant_retention_s": 8.0,
    "fusion_interior_birth_fallback_s": 1.5,
    "fusion_local_reconnect_long_window_s": 5.0,
    "fusion_local_reconnect_confirmations": 5,
    "fusion_local_reconnect_confirmation_window_s": 0.50,
    "fusion_instant_confirmations": 5,
    "fusion_published_merge_confirmations": 7,
}

# This branch uses Hybrid-SORT as the default local tracker.
# Set to "identity-v1-drift-guard" or "identity-v1-baseline" for A/B.
DEFAULT_TRACKING_PROFILE = "hybridsort"

TRACKING_PROFILE_OVERRIDES["hybridsort-causal-association-v1"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-late-binding-balanced"],
    "hybridsort_reappearance_enabled": False,
    "hybridsort_causal_association_enabled": True,
    "hybridsort_lost_observation_prediction_enabled": False,
    "hybridsort_causal_motion_rescue_enabled": True,
    "fusion_reconnect_appearance_enabled": False,
    "fusion_appearance_ownership_guard_enabled": False,
}
TRACKING_PROFILE_OVERRIDES["hybridsort-causal-extrapolation-rejected"] = {
    **TRACKING_PROFILE_OVERRIDES["hybridsort-causal-association-v1"],
    "hybridsort_causal_motion_rescue_enabled": False,
    "hybridsort_lost_observation_prediction_enabled": True,
}


def apply_tracking_profile(cfg: PipelineConfig, profile: str) -> PipelineConfig:
    """Apply a named, explicit ablation profile in-place."""
    if profile not in TRACKING_PROFILE_OVERRIDES:
        choices = ", ".join(sorted(TRACKING_PROFILE_OVERRIDES))
        raise ValueError(f"Unknown tracking profile {profile!r}; choose one of: {choices}")
    for name, value in TRACKING_PROFILE_OVERRIDES[profile].items():
        setattr(cfg, name, value)
    return cfg


def build_effective_config(
    *,
    profile: str = DEFAULT_TRACKING_PROFILE,
    local_tracker: str | None = None,
    **pipeline_kwargs: object,
) -> PipelineConfig:
    """Build one effective config with explicit overrides applied last.

    ``None`` means that the named profile owns the local-tracker choice. This
    distinction prevents CLI defaults from silently replacing Hybrid-SORT with
    OC-SORT after a profile has already been applied.
    """
    cfg = PipelineConfig(**pipeline_kwargs)
    apply_tracking_profile(cfg, profile)
    if local_tracker is not None:
        cfg.local_tracker = local_tracker
    return cfg
