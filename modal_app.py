"""Render MP4s or evaluate MOT metrics on a Modal T4 GPU.

First run (uploads only ``data/custom`` plus ``yolo26n.pt``, then renders):

    modal run modal_app.py --upload --seconds 20 --stride 1 --out result.mp4

Later runs reuse the persistent Volume:

    modal run modal_app.py --seconds 60 --stride 1 --out result.mp4

Evaluate a longer sequence and save the returned metrics locally:

    modal run modal_app.py --evaluate --max-frames 4000 \
        --tracking-profile hybridsort-crowd-recovery \
        --cameras camera_0001,camera_0007,camera_0008,camera_0010 \
        --metrics-out mot-metrics.json

The local entrypoint waits for the remote job, then downloads the completed
MP4. Detection runs on CUDA; decoding, OC-SORT, fusion, and composition run
on the T4 container's CPU.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import uuid

import modal

from multicam_tracker.config import DEFAULT_TRACKING_PROFILE


APP_NAME = "multicam-tracker-t4"
VOLUME_NAME = "multicam-tracker-data"
VOLUME_MOUNT = "/mnt/multicam"
REMOTE_DATA_ROOT = f"{VOLUME_MOUNT}/data"
REMOTE_MODEL = f"{VOLUME_MOUNT}/models/yolo26n.pt"
REMOTE_OUTPUT_DIR = f"{VOLUME_MOUNT}/outputs"

app = modal.App(APP_NAME)
storage = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _local_git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def _local_source_fingerprint() -> str:
    """Hash the exact Python/config source mounted into the Modal worker."""
    root = Path(__file__).resolve().parent
    paths = [root / "modal_app.py", root / "pyproject.toml", root / "uv.lock"]
    paths.extend(sorted((root / "src" / "multicam_tracker").glob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


_SOURCE_GIT_SHA = _local_git_value("rev-parse", "HEAD")
_SOURCE_GIT_BRANCH = _local_git_value("branch", "--show-current")
_SOURCE_GIT_DIRTY = "1" if _local_git_value("status", "--porcelain") else "0"
_SOURCE_TREE_SHA256 = _local_source_fingerprint()

# uv_sync installs the exact project lockfile. Source is mounted separately so
# code-only changes do not rebuild the large PyTorch/Ultralytics image layer.
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .uv_sync(".", frozen=True)
    .env({
        "SOURCE_GIT_SHA": _SOURCE_GIT_SHA,
        "SOURCE_GIT_BRANCH": _SOURCE_GIT_BRANCH,
        "SOURCE_GIT_DIRTY": _SOURCE_GIT_DIRTY,
        "SOURCE_TREE_SHA256": _SOURCE_TREE_SHA256,
    })
    # Modal local mounts must be the final image operation. Keeping this last
    # also avoids rebuilding dependency layers for source-only changes.
    .add_local_dir("src/multicam_tracker", "/root/multicam_tracker")
)


@app.function(
    image=gpu_image,
    gpu="T4",
    cpu=4.0,
    memory=16_384,
    timeout=60 * 60,
    volumes={VOLUME_MOUNT: storage},
)
def render_on_t4(
    *,
    output_name: str,
    seconds: float = 20.0,
    start: int = 0,
    stride: int = 1,
    cameras: list[str] | None = None,
    tracking_profile: str = DEFAULT_TRACKING_PROFILE,
    expected_people: int | None = None,
    detector_fp16: bool = False,
    detector_compile: bool = False,
) -> dict[str, object]:
    """Run detection/tracking on T4 and persist a validated MP4."""
    import time

    import cv2
    import torch

    from multicam_tracker.render_preview import render_video

    if not torch.cuda.is_available():
        raise RuntimeError("Modal assigned a GPU, but PyTorch CUDA is unavailable")
    if seconds <= 0:
        raise ValueError("seconds must be > 0")
    if start < 0:
        raise ValueError("start must be >= 0")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if expected_people is not None and expected_people < 0:
        raise ValueError("expected_people must be >= 0")

    safe_name = Path(output_name).name
    if safe_name != output_name or Path(safe_name).suffix.lower() != ".mp4":
        raise ValueError("output_name must be a plain .mp4 filename")

    data_root = Path(REMOTE_DATA_ROOT)
    model_path = Path(REMOTE_MODEL)
    scene_dir = data_root / "custom"
    if not scene_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {scene_dir}. Run once with `modal run modal_app.py --upload`."
        )
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing {model_path}. Run once with `modal run modal_app.py --upload`."
        )

    output_dir = Path(REMOTE_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / safe_name

    torch.backends.cudnn.benchmark = True
    started = time.perf_counter()
    render_video(
        data_root=data_root,
        out=output_path,
        start=start,
        seconds=seconds,
        stride=stride,
        device="cuda",
        model_name=str(model_path),
        cameras=cameras,
        tracking_profile=tracking_profile,
        expected_people=expected_people,
        detector_fp16=detector_fp16,
        detector_compile=detector_compile,
    )
    elapsed_s = time.perf_counter() - started

    # Fail the cloud job instead of downloading a corrupt/empty container.
    capture = cv2.VideoCapture(str(output_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frame_count <= 0 or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Rendered MP4 failed validation: {output_path}")

    storage.commit()
    return {
        "volume_path": f"outputs/{safe_name}",
        "bytes": output_path.stat().st_size,
        "frames": frame_count,
        "fps": output_fps,
        "elapsed_s": elapsed_s,
        "gpu": torch.cuda.get_device_name(0),
    }


@app.function(
    image=gpu_image,
    gpu="T4",
    cpu=4.0,
    memory=16_384,
    timeout=6 * 60 * 60,
    volumes={VOLUME_MOUNT: storage},
)
def evaluate_on_t4(
    *,
    output_name: str,
    max_frames: int = 1200,
    match_gate_m: float = 1.0,
    frame_offset: int = 0,
    metric_window_frames: int = 500,
    cumulative_frames: list[int] | None = None,
    cameras: list[str] | None = None,
    tracking_profile: str = DEFAULT_TRACKING_PROFILE,
    expected_people: int | None = None,
    seed: int = 0,
    detector_fp16: bool = False,
    detector_compile: bool = False,
) -> dict[str, object]:
    """Run the full tracker on CUDA and return world-plane MOT metrics."""
    import json
    import resource
    import time

    import numpy as np
    import torch

    from multicam_tracker.config import build_effective_config
    from multicam_tracker.demo import run_pipeline
    from multicam_tracker.identity_commit import (
        IdentityCommitBuffer,
        append_committed_frames,
    )
    from multicam_tracker.evaluate import (
        _classify_switch_events,
        ground_truth_by_frame,
    )
    from multicam_tracker.mot_metrics import (
        build_identity_error_taxonomy,
        evaluate_mot,
        evaluate_mot_slices,
        identity_trajectory_diagnostics,
    )
    from multicam_tracker.reporting import build_run_metadata, configure_determinism, committed_stream_hashes
    from multicam_tracker.stage_diagnostics import (
        PipelineStageDiagnostics,
        load_bbox_ground_truth,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Modal assigned a GPU, but PyTorch CUDA is unavailable")
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")
    if not np.isfinite(match_gate_m) or match_gate_m <= 0:
        raise ValueError("match_gate_m must be a finite positive number")
    if frame_offset not in (0, 1, 2):
        raise ValueError("frame_offset must be 0, 1, or 2")
    if metric_window_frames <= 0:
        raise ValueError("metric_window_frames must be > 0")
    if expected_people is not None and expected_people < 0:
        raise ValueError("expected_people must be >= 0")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    cumulative_frames = cumulative_frames or [500, 1000, 1500, 2000, 4000]
    if any(value <= 0 for value in cumulative_frames):
        raise ValueError("cumulative_frames values must be > 0")

    configure_determinism(seed)

    safe_name = Path(output_name).name
    if safe_name != output_name or Path(safe_name).suffix.lower() != ".json":
        raise ValueError("output_name must be a plain .json filename")

    data_root = Path(REMOTE_DATA_ROOT)
    model_path = Path(REMOTE_MODEL)
    scene_dir = data_root / "custom"
    if not scene_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {scene_dir}. Run once with `modal run modal_app.py --upload`."
        )
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing {model_path}. Run once with `modal run modal_app.py --upload`."
        )

    cfg = build_effective_config(
        data_root=data_root,
        device="cuda",
        async_ingest=False,
        fusion_output_time_policy="event",
        profile=tracking_profile,
    )
    if expected_people is not None:
        cfg.fusion_fixed_population_size = expected_people
    cfg.detector_fp16 = detector_fp16
    cfg.detector_compile = detector_compile
    cfg.model_name = str(model_path)
    available = sorted(
        path.name for path in cfg.scene_dir.glob("camera_*") if path.is_dir()
    )
    cfg.cameras = list(cameras) if cameras else available
    missing = sorted(set(cfg.cameras) - set(available))
    if missing:
        raise FileNotFoundError(f"Camera folders not found: {', '.join(missing)}")
    if not cfg.cameras:
        raise FileNotFoundError(f"No camera folders found under {cfg.scene_dir}")

    gt_frames = ground_truth_by_frame(
        cfg.scene_dir,
        cfg.cameras,
        cfg.camera_start_frame_offsets,
        max_frame=max_frames + 3,
    )
    bbox_gt_frames = load_bbox_ground_truth(
        cfg.scene_dir,
        cfg.cameras,
        cfg.camera_start_frame_offsets,
        max_frame=max_frames + 3,
    )
    stage_diagnostics = PipelineStageDiagnostics(
        bbox_gt_frames,
        stride=cfg.stride,
        segment_frames=metric_window_frames,
        pipeline_detection_threshold=cfg.conf_threshold,
        confidence_source_floor=cfg.conf_threshold,
    )
    predictions: list[tuple[int, dict[int, np.ndarray]]] = []
    source_maps: dict[int, dict[int, list[tuple[str, int]]]] = {}
    no_public_affinity = 0
    provisional_affinity = 0
    public_affinity_not_output = 0
    last_pipeline_metrics: dict[str, object] = {}
    identity_committer = IdentityCommitBuffer(cfg.fusion_identity_commit_delay_s)
    pipeline_tick_latencies_s: list[float] = []
    torch.cuda.reset_peak_memory_stats()

    def timed_ticks(iterator):
        iterator = iter(iterator)
        while True:
            tick_started = time.perf_counter()
            try:
                tick = next(iterator)
            except StopIteration:
                return
            pipeline_tick_latencies_s.append(time.perf_counter() - tick_started)
            yield tick

    started = time.perf_counter()
    for tick in timed_ticks(run_pipeline(cfg, max_frames=max_frames)):
        stage_diagnostics.update(tick)
        append_committed_frames(
            identity_committer.push_tick(tick), predictions, source_maps
        )
        emitted_gids = {int(track.gid) for track in tick["global_tracks"]}
        for observation in tick["observations"]:
            key = observation.camera, int(observation.local_id)
            gid = tick["local_to_gid"].get(key)
            if gid is None:
                no_public_affinity += 1
                internal_gid = tick["local_to_internal_gid"].get(key)
                if internal_gid is not None and int(internal_gid) < 0:
                    provisional_affinity += 1
            else:
                if int(gid) not in emitted_gids:
                    public_affinity_not_output += 1
        last_pipeline_metrics = tick["metrics"]
        if int(tick["tick_index"]) % 250 == 0:
            print(
                f"Evaluated {len(predictions)} ticks in "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
    append_committed_frames(identity_committer.flush(), predictions, source_maps)
    last_pipeline_metrics.update(identity_committer.final_pipeline_metrics)
    last_pipeline_metrics.update(identity_committer.metrics)
    elapsed_s = time.perf_counter() - started
    if not predictions:
        raise RuntimeError("The tracking pipeline produced no evaluation ticks")

    metrics = evaluate_mot(
        predictions,
        gt_frames,
        frame_offset=frame_offset,
        match_gate_m=match_gate_m,
    )
    metric_slices = evaluate_mot_slices(
        predictions,
        gt_frames,
        frame_offset=frame_offset,
        match_gate_m=match_gate_m,
        cumulative_frames=cumulative_frames,
        window_frames=metric_window_frames,
    )
    identity_diagnostics = identity_trajectory_diagnostics(
        predictions,
        gt_frames,
        frame_offset=frame_offset,
        match_gate_m=match_gate_m,
    )
    switch_diagnostics = _classify_switch_events(
        metrics["switch_events"],
        source_maps,
        predictions,
        gt_frames,
        frame_offset,
    )
    stage_report = stage_diagnostics.finalize()
    error_taxonomy = build_identity_error_taxonomy(
        metrics,
        identity_diagnostics,
        last_pipeline_metrics,
        switch_diagnostics=switch_diagnostics,
        no_public_affinity=no_public_affinity,
        provisional_affinity=provisional_affinity,
        public_affinity_not_output=public_affinity_not_output,
        stage_diagnostics=stage_report,
    )
    fusion_counter_names = (
        "global_id_merges",
        "provisional_ids_created",
        "provisional_ids_published",
        "provisional_ids_reconnected_before_publish",
        "provisional_publication_deferred",
        "unique_published_gids",
        "instant_geometry_merges",
        "instant_merges_deferred_for_confirmation",
        "instant_geometry_ambiguous",
        "instant_opposite_direction_vetoes",
        "instant_motion_gate_failed",
        "published_merge_motion_rejected",
        "published_merge_motion_softened",
        "local_fragments_reconnected",
        "local_reconnects_deferred_for_confirmation",
        "local_reconnect_covariance_gate_failed",
        "local_reconnect_ambiguous",
        "local_reconnect_motion_gate_failed",
        "local_reconnect_appearance_rejected",
        "batch_reconnect_assignments",
        "long_reconnect_boundary_rejected",
        "long_reconnect_motion_rejected",
        "duplicate_global_tracks_merged",
        "duplicate_merges_deferred",
        "duplicate_outputs_suppressed",
        "same_camera_merge_conflicts",
        "published_id_preserved_on_merge",
        "peak_retained_global_tracks",
        "pending_provisional_tracks",
        "peak_pending_provisional_tracks",
        "public_identity_slots",
        "peak_public_identity_slots",
        "public_gid_allocations",
        "new_gid_entry_deferred",
        "interior_birth_fallbacks",
        "interior_birth_reachable_dormant_deferred",
        "late_identity_binding_deferred",
        "late_identity_binding_timeout_releases",
        "new_gid_capacity_deferred",
        "new_gid_trajectory_rejected",
        "retirement_deferred_no_exit",
        "dormant_identity_count",
        "dormant_reconnects_accepted",
        "confirmed_identity_exits",
        "forced_identity_retirements",
        "anti_pingpong_switches_blocked",
        "global_appearance_conflicts",
        "global_appearance_conflicts_deferred",
        "global_ids_protected_from_local_takeover",
        "appearance_takeover_immediate_reconnects_blocked",
        "retained_occluded_global_tracks",
        "retained_appearance_prototypes",
        "appearance_prototype_bytes",
        "tracklet_segments_reset",
        "tracklet_points_reset",
        "fusion_motion_gate_accepted",
        "fusion_motion_jump_rejected",
        "trajectory_matches_accepted",
        "trajectory_matches_rejected",
        "trajectory_direction_rejected",
        "trajectory_speed_rejected",
        "local_tracks_created",
        "local_track_reactivations",
        "local_appearance_reappearance_matches",
        "local_appearance_reappearance_deferred",
        "local_appearance_reappearance_ambiguous",
        "local_appearance_reappearance_candidate_resets",
        "retained_occluded_local_tracks",
        "peak_retained_local_tracks",
        "trajectory_smoothing_accepted",
        "trajectory_smoothing_rejected",
        "trajectory_smoothing_out_of_order_rejected",
        "identity_commit_frames_buffered",
        "identity_commit_frames_committed",
        "identity_commit_unresolved_hypotheses_dropped",
        "identity_commit_duplicate_hypotheses_collapsed",
        "identity_commit_intentionally_rejected_hypotheses",
        "identity_commit_backfilled_hypotheses",
        "terminal_bindings_accepted",
        "terminal_binding_defer_attempts",
    )
    switch_gaps = [event[2] - event[1] for event in metrics["switch_events"]]
    metrics.update({
        "frame_offset": frame_offset,
        "match_gate_m": match_gate_m,
        "max_frames": max_frames,
        "evaluated_ticks": len(predictions),
        "elapsed_s": elapsed_s,
        "ticks_per_s": len(predictions) / max(elapsed_s, 1e-9),
        "pipeline_tick_latency_ms": {
            "p50": float(np.percentile(pipeline_tick_latencies_s, 50) * 1000.0),
            "p95": float(np.percentile(pipeline_tick_latencies_s, 95) * 1000.0),
            "max": float(max(pipeline_tick_latencies_s) * 1000.0),
        },
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0,
        "gpu": torch.cuda.get_device_name(0),
        "cameras": cfg.cameras,
        "tracking_profile": tracking_profile,
        "expected_people": cfg.fusion_fixed_population_size,
        "switch_gap_buckets": {
            "1": sum(gap == 1 for gap in switch_gaps),
            "2_30": sum(2 <= gap <= 30 for gap in switch_gaps),
            "31_90": sum(31 <= gap <= 90 for gap in switch_gaps),
            "over_90": sum(gap > 90 for gap in switch_gaps),
        },
        **metric_slices,
        "identity_diagnostics": identity_diagnostics,
        "identity_lifecycle": identity_committer.lifecycle_report,
        "switch_diagnostics": switch_diagnostics,
        "identity_error_taxonomy": error_taxonomy,
        "stage_diagnostics": stage_report,
        "fusion_counters": {
            name: last_pipeline_metrics.get(name, 0)
            for name in fusion_counter_names
        },
        "clock_estimates": last_pipeline_metrics.get("clock_estimates", {}),
        "bounded_state": {
            name: last_pipeline_metrics.get(name, 0)
            for name in (
                "tracklet_points", "tracklet_bytes", "retained_global_tracks",
                "retained_aliases", "retained_affinities",
                "peak_retained_global_tracks", "retained_local_tracks",
                "peak_retained_local_tracks",
                "pending_provisional_tracks", "peak_pending_provisional_tracks",
                "public_identity_slots", "peak_public_identity_slots",
                "trajectory_smoothing_histories",
                "trajectory_smoothing_samples",
            )
        },
        "run_metadata": build_run_metadata(cfg, seed=seed),
        **committed_stream_hashes(predictions, source_maps),
    })

    output_dir = Path(REMOTE_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / safe_name
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    storage.commit()
    metrics["volume_path"] = f"outputs/{safe_name}"
    return metrics


def _upload_assets(local_data_root: Path, local_model: Path) -> None:
    local_scene_dir = local_data_root / "custom"
    if not local_scene_dir.is_dir():
        raise FileNotFoundError(f"Expected scene directory at {local_scene_dir}")
    if not local_model.is_file():
        raise FileNotFoundError(f"Model weights not found: {local_model}")

    print(f"Uploading only {local_scene_dir} and {local_model} to Volume {VOLUME_NAME!r}...")
    with storage.batch_upload(force=True) as batch:
        # The runtime expects PipelineConfig(data_root=/mnt/multicam/data), so
        # upload only the selected scene into its exact `custom/` subfolder.
        batch.put_directory(str(local_scene_dir), "/data/custom")
        batch.put_file(str(local_model), "/models/yolo26n.pt")
    print("Upload complete. Later runs can omit --upload.")


@app.local_entrypoint()
def main(
    out: str = "modal-result.mp4",
    seconds: float = 20.0,
    start: int = 0,
    stride: int = 1,
    cameras: str = "",
    upload: bool = False,
    data_root: str = "data",
    model: str = "yolo26n.pt",
    evaluate: bool = False,
    max_frames: int = 1200,
    match_gate_m: float = 1.0,
    frame_offset: int = 0,
    metrics_out: str = "modal-mot-metrics.json",
    metric_window_frames: int = 500,
    cumulative_frames: str = "500,1000,1500,2000,4000",
    tracking_profile: str = DEFAULT_TRACKING_PROFILE,
    expected_people: int = -1,
    seed: int = 0,
    detector_fp16: bool = False,
    detector_compile: bool = False,
) -> None:
    """Optionally upload inputs, then render MP4 or evaluate MOT metrics."""
    if upload:
        _upload_assets(Path(data_root).expanduser().resolve(), Path(model).expanduser().resolve())

    if expected_people < -1:
        raise ValueError("--expected-people must be >= 0, or omitted")
    expected_people_override = None if expected_people == -1 else expected_people
    selected_cameras = [value.strip() for value in cameras.split(",") if value.strip()]
    selected_cumulative_frames = [
        int(value.strip())
        for value in cumulative_frames.split(",")
        if value.strip()
    ]
    if evaluate:
        import json

        local_metrics = Path(metrics_out).expanduser().resolve()
        if local_metrics.suffix.lower() != ".json":
            raise ValueError("--metrics-out must end with .json")
        remote_name = f"{local_metrics.stem}-{uuid.uuid4().hex[:10]}.json"
        print(
            f"Starting Modal T4 evaluation: max_frames={max_frames}, "
            f"match_gate_m={match_gate_m}, profile={tracking_profile}..."
        )
        result = evaluate_on_t4.remote(
            output_name=remote_name,
            max_frames=max_frames,
            match_gate_m=match_gate_m,
            frame_offset=frame_offset,
            metric_window_frames=metric_window_frames,
            cumulative_frames=selected_cumulative_frames,
            cameras=selected_cameras or None,
            tracking_profile=tracking_profile,
            expected_people=expected_people_override,
            seed=seed,
            detector_fp16=detector_fp16,
            detector_compile=detector_compile,
        )
        local_metrics.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            key: value for key, value in result.items() if key != "volume_path"
        }
        local_metrics.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(serializable, indent=2, ensure_ascii=False))
        print(
            f"Saved metrics to {local_metrics}\n"
            f"Persistent Modal copy: {result['volume_path']}"
        )
        return

    local_out = Path(out).expanduser().resolve()
    if local_out.suffix.lower() != ".mp4":
        raise ValueError("--out must end with .mp4")
    remote_name = f"{local_out.stem}-{uuid.uuid4().hex[:10]}.mp4"
    print(
        f"Starting Modal T4 render: seconds={seconds}, stride={stride}, "
        f"profile={tracking_profile}..."
    )
    result = render_on_t4.remote(
        output_name=remote_name,
        seconds=seconds,
        start=start,
        stride=stride,
        cameras=selected_cameras or None,
        tracking_profile=tracking_profile,
        expected_people=expected_people_override,
        detector_fp16=detector_fp16,
        detector_compile=detector_compile,
    )

    local_out.parent.mkdir(parents=True, exist_ok=True)
    with local_out.open("wb") as destination:
        for chunk in storage.read_file(str(result["volume_path"])):
            destination.write(chunk)

    size_mb = int(result["bytes"]) / 1_000_000
    print(
        f"Downloaded {local_out} ({size_mb:.1f} MB, {result['frames']} frames, "
        f"{result['fps']:.1f} fps)\n"
        f"GPU: {result['gpu']} | remote elapsed: {result['elapsed_s']:.1f} s"
    )
