"""Camera calibration and ground-truth loading.

The key geometric idea of the whole demo lives here: each camera ships a
3x3 **homography** H that maps a point on the warehouse floor, in world
meters (X, Y, 1), to a pixel (u, v, 1) in that camera's image. Inverting H
lets us do the opposite — take the pixel where a person's feet touch the
ground and recover their position on the floor in meters. That shared
"world floor plane" is what makes cross-camera fusion possible.

Dataset files (NVIDIA PhysicalAI-SmartSpaces, MTMC_Tracking_2024):

- ``camera_NNNN/calibration.json``
    * ``"camera projection matrix"`` — 3x4 world->pixel projection P
    * ``"homography matrix"``        — 3x3 world-floor->pixel homography H
      (P with its Z column dropped, i.e. valid for points with Z=0)
- ``calibration_2025_format.json`` (scene level)
    * per-camera ``intrinsicMatrix`` K (3x3) and ``extrinsicMatrix``
      [R|t] (3x4, world->camera) — used to place camera frusta in the 3D view
- ``ground_truth.txt`` — one line per (camera, person, frame):
    ``camera_id person_id frame x y w h worldX worldY``
    with (x, y, w, h) the top-left 2D bbox in pixels and (worldX, worldY)
    the person's position on the floor in meters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CameraCalibration:
    name: str                 # e.g. "camera_0001"
    P: np.ndarray             # 3x4 world -> pixel projection
    H_world_to_image: np.ndarray  # 3x3 floor (X, Y, 1) -> pixel (u, v, 1)
    H_image_to_world: np.ndarray  # 3x3 pixel -> floor, inverse of the above
    K: np.ndarray             # 3x3 intrinsics
    Rt: np.ndarray            # 3x4 extrinsics [R|t], world -> camera
    width: int
    height: int
    fps: float

    def image_to_world(self, pts_uv: np.ndarray) -> np.ndarray:
        """Map pixel coordinates (N, 2) to world floor coordinates (N, 2) in meters."""
        return _apply_homography(self.H_image_to_world, pts_uv)

    def world_to_image(self, pts_xy: np.ndarray) -> np.ndarray:
        """Map world floor coordinates (N, 2) in meters to pixel coordinates (N, 2)."""
        return _apply_homography(self.H_world_to_image, pts_xy)

    def pose_in_world(self) -> tuple[np.ndarray, np.ndarray]:
        """Camera orientation (3x3) and position (3,) in world coordinates.

        The dataset's [R|t] maps world -> camera, so the camera's pose in the
        world is the inverse: orientation R^T, position -R^T @ t.
        """
        R, t = self.Rt[:, :3], self.Rt[:, 3]
        return R.T, -R.T @ t


def _apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    # Lift each 2D point to homogeneous coordinates [x, y, 1], multiply by H,
    # and get back [x', y', w]. That result is only a direction until the
    # perspective divide: the actual point is (x'/w, y'/w). This divide is
    # what makes a homography able to model perspective (a plain 2D affine
    # transform cannot).
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
    mapped = homogeneous @ H.T
    return mapped[:, :2] / mapped[:, 2:3]


def load_calibrations(scene_dir: Path, cameras: list[str]) -> dict[str, CameraCalibration]:
    """Load calibration for each camera, combining both dataset formats."""
    scene_file = scene_dir / "calibration_2025_format.json"
    sensors = {
        s["id"].lower(): s
        for s in json.loads(scene_file.read_text())["sensors"]
        if s.get("type") == "camera"
    }

    calibrations: dict[str, CameraCalibration] = {}
    for cam in cameras:
        per_cam = json.loads((scene_dir / cam / "calibration.json").read_text())
        H = np.array(per_cam["homography matrix"], dtype=np.float64)
        sensor = sensors[cam.lower()]  # dirs are "camera_0001", ids "Camera_0001"
        attrs = {a["name"]: a["value"] for a in sensor.get("attributes", [])}
        calibrations[cam] = CameraCalibration(
            name=cam,
            P=np.array(per_cam["camera projection matrix"], dtype=np.float64),
            H_world_to_image=H,
            H_image_to_world=np.linalg.inv(H),
            K=np.array(sensor["intrinsicMatrix"], dtype=np.float64),
            Rt=np.array(sensor["extrinsicMatrix"], dtype=np.float64),
            width=int(attrs.get("frameWidth", 1920)),
            height=int(attrs.get("frameHeight", 1080)),
            fps=float(attrs.get("fps", 30)),
        )
    return calibrations


def load_ground_truth(scene_dir: Path) -> np.ndarray:
    """Load ground truth as an (N, 9) float array.

    Columns: camera_id, person_id, frame, x, y, w, h, worldX, worldY.
    """
    return np.loadtxt(scene_dir / "ground_truth.txt", dtype=np.float64)


def foot_point(bbox_xywh: np.ndarray) -> np.ndarray:
    """Bottom-center of a top-left-anchored (x, y, w, h) bbox — where feet meet floor."""
    bbox = np.atleast_2d(bbox_xywh)
    return np.stack([bbox[:, 0] + bbox[:, 2] / 2, bbox[:, 1] + bbox[:, 3]], axis=1)
