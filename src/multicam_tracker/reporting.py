"""Reproducible benchmark metadata shared by local and Modal evaluators."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
from typing import Any, Callable

import numpy as np

from .config import PipelineConfig


def configure_determinism(seed: int = 0, *, deterministic: bool = True) -> None:
    """Seed CPU/CUDA libraries before detector and tracker construction."""
    if seed < 0:
        raise ValueError("seed must be >= 0")
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import cv2

        cv2.setRNGSeed(seed)
    except (ImportError, OverflowError):
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = not deterministic
            torch.backends.cudnn.deterministic = deterministic
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except ImportError:
        pass


def canonical_replay_tick_bytes(
    tick: dict[str, Any],
) -> tuple[bytes, bytes]:
    """Serialize exact per-tick predictions and assignments deterministically."""

    frame_index = int(tick["frame_index"])
    predictions = [
        [
            int(track.gid),
            [float(value) for value in np.asarray(track.world_xy).reshape(-1)],
        ]
        for track in sorted(tick.get("global_tracks", ()), key=lambda item: item.gid)
    ]

    def affinity_rows(name: str) -> list[list[object]]:
        mapping = tick.get(name, {})
        return [
            [str(camera), int(local_id), int(gid)]
            for (camera, local_id), gid in sorted(
                mapping.items(), key=lambda item: (str(item[0][0]), int(item[0][1]))
            )
        ]

    prediction_record = {
        "frame_index": frame_index,
        "global_tracks": predictions,
    }
    assignment_record = {
        "frame_index": frame_index,
        "local_to_gid": affinity_rows("local_to_gid"),
        "local_to_internal_gid": affinity_rows("local_to_internal_gid"),
        "accepted_fusion_keys": [
            [str(camera), int(local_id)]
            for camera, local_id in sorted(
                tick.get("accepted_fusion_keys", ()),
                key=lambda item: (str(item[0]), int(item[1])),
            )
        ],
        "gid_aliases": [
            [int(source), int(target)]
            for source, target in sorted(tick.get("gid_aliases", {}).items())
        ],
    }

    def encoded(record: dict[str, object]) -> bytes:
        return json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    return encoded(prediction_record), encoded(assignment_record)


def committed_stream_hashes(predictions, source_maps) -> dict[str, str]:
    """Hash the actual evaluator input, including end-of-stream backfill."""
    prediction_hash, assignment_hash = hashlib.sha256(), hashlib.sha256()
    for frame_index, tracks in predictions:
        records = (
            [int(frame_index), [[int(gid), np.asarray(xy, dtype=float).tolist()]
                               for gid, xy in sorted(tracks.items())]],
            [int(frame_index), [[int(gid), sorted([str(camera), int(local)]
                                               for camera, local in keys)]
                               for gid, keys in sorted(source_maps.get(frame_index, {}).items())]],
        )
        for digest, record in zip((prediction_hash, assignment_hash), records):
            payload = json.dumps(record, separators=(",", ":"), allow_nan=False).encode()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return {"committed_prediction_stream_sha256": prediction_hash.hexdigest(),
            "committed_assignment_stream_sha256": assignment_hash.hexdigest()}


def build_run_metadata(
    cfg: PipelineConfig,
    *,
    seed: int = 0,
    deterministic: bool = True,
    repository_root: Path | None = None,
    git_runner: Callable[..., Any] | None = None,
) -> dict[str, object]:
    """Return JSON-safe configuration, code, dependency and input identity."""
    root = (repository_root or Path.cwd()).resolve()
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": _repository_state(root, git_runner=git_runner),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: _package_version(name)
                for name in ("numpy", "scipy", "torch", "ultralytics", "opencv-python")
            },
        },
        "determinism": {
            "seed": int(seed),
            "deterministic_algorithms": bool(deterministic),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        },
        "config": _json_safe(asdict(cfg)),
        "inputs": _input_fingerprint(cfg),
    }


def _repository_state(
    root: Path,
    *,
    git_runner: Callable[..., Any] | None = None,
) -> dict[str, object]:
    runner = subprocess.run if git_runner is None else git_runner

    def git(*args: str) -> str | None:
        try:
            result = runner(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    sha = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    status = git("status", "--porcelain")
    source_fingerprint = _source_tree_fingerprint(root, git)
    if sha is None:
        sha = os.environ.get("SOURCE_GIT_SHA") or None
    if branch is None:
        branch = os.environ.get("SOURCE_GIT_BRANCH") or None
    if status is None:
        dirty_value = os.environ.get("SOURCE_GIT_DIRTY")
        dirty = None if dirty_value is None else dirty_value == "1"
        status_lines: list[str] = []
    else:
        dirty = bool(status)
        status_lines = status.splitlines() if status else []
    return {
        "commit_sha": sha,
        "branch": branch,
        "dirty": dirty,
        "status": status_lines,
        "source_fingerprint": source_fingerprint,
    }


def _source_tree_fingerprint(
    root: Path,
    git: Callable[..., str | None],
) -> dict[str, object]:
    """Hash source/config bytes, including dirty tracked and untracked files."""
    configured = os.environ.get("SOURCE_TREE_SHA256")
    listed = git("ls-files", "--cached", "--others", "--exclude-standard")
    if listed is None:
        return {
            "algorithm": "sha256",
            "sha256": configured or None,
            "source": "environment" if configured else "unavailable",
        }

    candidates: list[tuple[str, Path]] = []
    for relative in listed.splitlines():
        normalized = relative.replace("\\", "/")
        path = root / relative
        if not path.is_file():
            continue
        if not (
            normalized.startswith(("src/", "benchmarks/", "tests/"))
            or normalized in {"modal_app.py", "pyproject.toml", "uv.lock"}
        ):
            continue
        if path.suffix.lower() not in {".py", ".toml", ".lock"}:
            continue
        candidates.append((normalized, path))

    digest = hashlib.sha256()
    total_bytes = 0
    for normalized, path in sorted(candidates):
        encoded = normalized.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(content)
        total_bytes += len(content)
    return {
        "algorithm": "sha256",
        "sha256": digest.hexdigest(),
        "source": "git_worktree",
        "file_count": len(candidates),
        "bytes": total_bytes,
    }


def _input_fingerprint(cfg: PipelineConfig) -> dict[str, object]:
    scene = cfg.scene_dir.resolve()
    model = Path(cfg.model_name).expanduser()
    cameras: dict[str, object] = {}
    for camera in cfg.cameras:
        camera_dir = scene / camera
        calibration_files = sorted(
            path for path in camera_dir.glob("*")
            if path.is_file() and path.name.lower() != "video.mp4"
        )
        cameras[camera] = {
            "video": _file_identity(camera_dir / "video.mp4", hash_contents=False),
            "calibration": [
                _file_identity(path, hash_contents=True)
                for path in calibration_files
            ],
        }
    return {
        "model": _file_identity(model, hash_contents=True),
        "ground_truth": _file_identity(
            scene / "ground_truth.txt", hash_contents=False
        ),
        "scene": str(scene),
        "cameras": cameras,
    }


def _file_identity(path: Path, *, hash_contents: bool) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "exists": False}
    stat = resolved.stat()
    result: dict[str, object] = {
        "path": str(resolved),
        "exists": True,
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_contents:
        digest = hashlib.sha256()
        with resolved.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


def validate_json_metadata(value: dict[str, object]) -> None:
    """Raise if benchmark metadata is not strict-JSON serializable."""
    json.dumps(value, allow_nan=False)
