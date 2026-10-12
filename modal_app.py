"""Render MP4s or evaluate MOT metrics on a Modal T4 GPU.

First run (uploads only ``data/custom`` plus ``yolo26n.pt``, then renders):

    modal run modal_app.py --upload --seconds 20 --stride 1 --out result.mp4

Later runs reuse the persistent Volume:

    modal run modal_app.py --seconds 60 --stride 1 --out result.mp4

Evaluate a longer sequence and save the returned metrics locally:

    modal run modal_app.py --evaluate --max-frames 1800 \
        --cameras camera_0001,camera_0007,camera_0008,camera_0010 \
        --metrics-out mot-metrics.json

The local entrypoint waits for the remote job, then downloads the completed
MP4. Detection runs on CUDA; decoding, ByteTrack, fusion, and composition run
on the T4 container's CPU.
"""

from __future__ import annotations

from pathlib import Path
import uuid

import modal


APP_NAME = "multicam-tracker-t4"
VOLUME_NAME = "multicam-tracker-data"
VOLUME_MOUNT = "/mnt/multicam"
REMOTE_DATA_ROOT = f"{VOLUME_MOUNT}/data"
REMOTE_MODEL = f"{VOLUME_MOUNT}/models/yolo26n.pt"
REMOTE_OUTPUT_DIR = f"{VOLUME_MOUNT}/outputs"

app = modal.App(APP_NAME)
storage = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# uv_sync installs the exact project lockfile. Source is mounted separately so
# code-only changes do not rebuild the large PyTorch/Ultralytics image layer.
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .uv_sync(".", frozen=True)
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
    cameras: list[str] | None = None,
) -> dict[str, object]:
    """Run the full tracker on CUDA and return world-plane MOT metrics."""
    import json
    import time

    import numpy as np
    import torch

    from multicam_tracker.config import PipelineConfig
    from multicam_tracker.demo import run_pipeline
    from multicam_tracker.evaluate import ground_truth_by_frame, _match_errors
    from multicam_tracker.mot_metrics import evaluate_mot

    if not torch.cuda.is_available():
        raise RuntimeError("Modal assigned a GPU, but PyTorch CUDA is unavailable")
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")
    if not np.isfinite(match_gate_m) or match_gate_m <= 0:
        raise ValueError("match_gate_m must be a finite positive number")

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

    cfg = PipelineConfig(
        data_root=data_root,
        device="cuda",
        async_ingest=False,
    )
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
    predictions: list[tuple[int, dict[int, np.ndarray]]] = []
    started = time.perf_counter()
    for tick in run_pipeline(cfg, max_frames=max_frames):
        predictions.append((
            tick["frame_index"],
            {
                track.gid: track.world_xy.copy()
                for track in tick["global_tracks"]
            },
        ))
        if len(predictions) % 250 == 0:
            print(
                f"Evaluated {len(predictions)} ticks in "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
    elapsed_s = time.perf_counter() - started
    if not predictions:
        raise RuntimeError("The tracking pipeline produced no evaluation ticks")

    best_offset = 0
    best_rmse = float("inf")
    for offset in (0, 1, 2):
        errors = _match_errors(
            predictions[:100], gt_frames, offset, match_gate_m
        )
        if errors:
            rmse = float(np.sqrt(np.mean(np.square(errors))))
            if rmse < best_rmse:
                best_offset = offset
                best_rmse = rmse

    metrics = evaluate_mot(
        predictions,
        gt_frames,
        frame_offset=best_offset,
        match_gate_m=match_gate_m,
    )
    metrics.update({
        "frame_offset": best_offset,
        "match_gate_m": match_gate_m,
        "max_frames": max_frames,
        "evaluated_ticks": len(predictions),
        "elapsed_s": elapsed_s,
        "ticks_per_s": len(predictions) / max(elapsed_s, 1e-9),
        "gpu": torch.cuda.get_device_name(0),
        "cameras": cfg.cameras,
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
    metrics_out: str = "modal-mot-metrics.json",
) -> None:
    """Optionally upload inputs, then render MP4 or evaluate MOT metrics."""
    if upload:
        _upload_assets(Path(data_root).expanduser().resolve(), Path(model).expanduser().resolve())

    selected_cameras = [value.strip() for value in cameras.split(",") if value.strip()]
    if evaluate:
        import json

        local_metrics = Path(metrics_out).expanduser().resolve()
        if local_metrics.suffix.lower() != ".json":
            raise ValueError("--metrics-out must end with .json")
        remote_name = f"{local_metrics.stem}-{uuid.uuid4().hex[:10]}.json"
        print(
            f"Starting Modal T4 evaluation: max_frames={max_frames}, "
            f"match_gate_m={match_gate_m}..."
        )
        result = evaluate_on_t4.remote(
            output_name=remote_name,
            max_frames=max_frames,
            match_gate_m=match_gate_m,
            cameras=selected_cameras or None,
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
    print(f"Starting Modal T4 render: seconds={seconds}, stride={stride}...")
    result = render_on_t4.remote(
        output_name=remote_name,
        seconds=seconds,
        start=start,
        stride=stride,
        cameras=selected_cameras or None,
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
