"""Versioned, validated detector cache metadata for deterministic replays."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import PipelineConfig


CACHE_SCHEMA = "multicam-detection-cache-v3"
SUPPORTED_CACHE_SCHEMAS = {"multicam-detection-cache-v2", CACHE_SCHEMA}


class DetectionCacheError(ValueError):
    """Raised when a detector cache cannot represent the requested replay."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sampled_file_fingerprint(path: Path, *, sample_bytes: int = 1024 * 1024) -> str:
    """Hash bounded samples from a potentially large video plus its size."""
    size = path.stat().st_size
    offsets = sorted({0, max(0, (size - sample_bytes) // 2), max(0, size - sample_bytes)})
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as source:
        for offset in offsets:
            source.seek(offset)
            digest.update(offset.to_bytes(8, "little"))
            digest.update(source.read(sample_bytes))
    return digest.hexdigest()


def build_detection_cache_metadata(
    cfg: PipelineConfig,
    ticks: int,
) -> dict[str, object]:
    """Fingerprint every input that can change cached detector output."""
    if ticks < 0:
        raise ValueError("ticks must be >= 0")
    model_path = Path(cfg.model_name).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    scene = cfg.scene_dir.resolve()
    videos: dict[str, object] = {}
    for camera in cfg.cameras:
        video = (scene / camera / "video.mp4").resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        stat = video.stat()
        videos[camera] = {
            "path": str(video),
            "bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sample_sha256": sampled_file_fingerprint(video),
        }
    return {
        "schema": CACHE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ticks": int(ticks),
        "row_schema": {
            "detections": "camera-ordered float32[N,5] xyxy+confidence",
            "packets": [
                "camera", "capture_ts_s", "arrival_ts_s", "sequence",
                "source_frame_index",
            ],
        },
        "detector": {
            "model": str(model_path),
            "model_sha256": sha256_file(model_path),
            "imgsz": int(cfg.imgsz),
            "conf_threshold": float(cfg.conf_threshold),
            "person_class_id": int(cfg.person_class_id),
            "device": cfg.device,
        },
        "input": {
            "scene": str(scene),
            "cameras": list(cfg.cameras),
            "stride": int(cfg.stride),
            "camera_start_frame_offsets": dict(cfg.camera_start_frame_offsets),
            "videos": videos,
        },
    }


def configure_from_cache(cfg: PipelineConfig, metadata: dict[str, Any]) -> None:
    """Apply only source ordering/time offsets recorded by a trusted cache."""
    cache_input = _mapping(metadata, "input")
    cfg.cameras = [str(camera) for camera in cache_input.get("cameras", [])]
    cfg.camera_start_frame_offsets = {
        str(camera): int(offset)
        for camera, offset in _mapping(
            cache_input, "camera_start_frame_offsets"
        ).items()
    }


def validate_detection_cache(
    metadata: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    cfg: PipelineConfig,
    *,
    validate_files: bool = True,
) -> None:
    """Reject truncated, malformed or input-incompatible detector caches."""
    schema = metadata.get("schema")
    if schema not in SUPPORTED_CACHE_SCHEMAS:
        raise DetectionCacheError(f"Unsupported detection cache schema: {schema!r}")
    if int(metadata.get("ticks", -1)) != len(rows):
        raise DetectionCacheError("Detection cache tick count does not match its rows")

    cache_input = _mapping(metadata, "input")
    detector = _mapping(metadata, "detector")
    cameras = [str(camera) for camera in cache_input.get("cameras", [])]
    if not cameras or len(cameras) != len(set(cameras)):
        raise DetectionCacheError("Detection cache cameras are missing or duplicated")
    if cameras != list(cfg.cameras):
        raise DetectionCacheError(
            f"Detection cache cameras {cameras!r} do not match replay {cfg.cameras!r}"
        )
    _require_equal("stride", int(cache_input.get("stride", -1)), int(cfg.stride))
    _require_equal("imgsz", int(detector.get("imgsz", -1)), int(cfg.imgsz))
    _require_equal(
        "person_class_id",
        int(detector.get("person_class_id", -1)),
        int(cfg.person_class_id),
    )
    if not np.isclose(
        float(detector.get("conf_threshold", np.nan)),
        float(cfg.conf_threshold),
        rtol=0.0,
        atol=1e-12,
    ):
        raise DetectionCacheError("Detection cache confidence threshold mismatch")

    if validate_files:
        model_path = Path(cfg.model_name).expanduser().resolve()
        if not model_path.is_file():
            raise DetectionCacheError(f"Replay model does not exist: {model_path}")
        if sha256_file(model_path) != detector.get("model_sha256"):
            raise DetectionCacheError("Detection cache model SHA256 mismatch")
        videos = _mapping(cache_input, "videos")
        for camera in cameras:
            cached_video = _mapping(videos, camera)
            video = (cfg.scene_dir / camera / "video.mp4").resolve()
            if not video.is_file():
                raise DetectionCacheError(f"Replay video does not exist: {video}")
            if int(cached_video.get("bytes", -1)) != video.stat().st_size:
                raise DetectionCacheError(f"Detection cache video size mismatch: {camera}")
            expected_sample = cached_video.get("sample_sha256")
            if expected_sample and sampled_file_fingerprint(video) != expected_sample:
                raise DetectionCacheError(
                    f"Detection cache video fingerprint mismatch: {camera}"
                )

    for index, row in enumerate(rows):
        _validate_row(row, cameras, index)


def _validate_row(row: dict[str, Any], cameras: list[str], index: int) -> None:
    packets = _mapping(row, "packets")
    if list(packets) != cameras:
        raise DetectionCacheError(
            f"Cache row {index} packet order/cameras do not match metadata"
        )
    detections = row.get("detections")
    if not isinstance(detections, (list, tuple)) or len(detections) != len(cameras):
        raise DetectionCacheError(
            f"Cache row {index} detection batch has the wrong camera count"
        )
    for camera, values, camera_detections in zip(cameras, packets.values(), detections):
        packet = _mapping_value(values, f"row {index} packet {camera}")
        if packet.get("camera") != camera:
            raise DetectionCacheError(f"Cache row {index} packet camera mismatch")
        for field in ("capture_ts_s", "arrival_ts_s"):
            if not np.isfinite(float(packet.get(field, np.nan))):
                raise DetectionCacheError(f"Cache row {index} has invalid {field}")
        for field in ("sequence", "source_frame_index"):
            if int(packet.get(field, -1)) < 0:
                raise DetectionCacheError(f"Cache row {index} has invalid {field}")
        array = np.asarray(camera_detections)
        if array.ndim != 2 or array.shape[1] != 5:
            raise DetectionCacheError(
                f"Cache row {index} detections must have shape (N, 5)"
            )
        if not np.all(np.isfinite(array)):
            raise DetectionCacheError(f"Cache row {index} detections are not finite")
        if len(array) and (
            np.any(array[:, 2] < array[:, 0])
            or np.any(array[:, 3] < array[:, 1])
            or np.any(array[:, 4] < 0.0)
            or np.any(array[:, 4] > 1.0)
        ):
            raise DetectionCacheError(f"Cache row {index} contains invalid boxes")


def _mapping(parent: dict[str, Any], name: str) -> dict[str, Any]:
    return _mapping_value(parent.get(name), name)


def _mapping_value(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DetectionCacheError(f"Detection cache {name} must be an object")
    return value


def _require_equal(name: str, cached: object, configured: object) -> None:
    if cached != configured:
        raise DetectionCacheError(
            f"Detection cache {name} {cached!r} does not match replay {configured!r}"
        )
