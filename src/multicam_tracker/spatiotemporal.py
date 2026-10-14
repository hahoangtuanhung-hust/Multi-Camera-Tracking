"""No-ReID spatio-temporal association on a shared ground plane.

This module implements the geometry-only part of Section 3.3 in Huang et al.,
"Enhancing Multi-Camera People Tracking with Anchor-Guided Clustering and
Spatio-Temporal Consistency ID Re-Assignment".  It intentionally contains no
appearance embedding, face feature, or Re-ID dependency.

The processing chain is:

``bbox / ankle keypoints -> image ground point -> homography -> robust
sliding-window position -> one-to-one spatial Hungarian assignment``.

The classes are independent of the detector and local tracker so they can be
used online by :mod:`multicam_tracker.fusion` or offline for ID reassignment.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import exp, log
from typing import Literal, Mapping, Protocol, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


BBoxFormat = Literal["xywh", "xyxy"]
GroundPointSource = Literal["ankles", "bbox"]
TrackletKey = tuple[str, int]


class CalibrationWithImageToWorld(Protocol):
    """Structural type accepted by :meth:`TopDownProjector.from_calibrations`."""

    H_image_to_world: np.ndarray


class ProjectionError(ValueError):
    """Raised when a homography cannot produce a finite top-down point."""


@dataclass(frozen=True)
class GroundPlaneLocation:
    """Image-space ground point and the evidence used to produce it."""

    xy: np.ndarray
    source: GroundPointSource

    def __post_init__(self) -> None:
        point = _point2(self.xy, name="ground-plane location")
        object.__setattr__(self, "xy", point.copy())


def estimate_ground_plane_location(
    detection: np.ndarray | Sequence[float],
    pose_keypoints: np.ndarray | Sequence[Sequence[float]] | None,
    *,
    tau_pose: float = 0.5,
    bbox_format: BBoxFormat = "xywh",
    left_ankle_index: int = 15,
    right_ankle_index: int = 16,
) -> GroundPlaneLocation:
    """Estimate the pixel where a detected person contacts the ground.

    ``pose_keypoints`` follows the usual ``(K, >=3)`` layout ``[x, y, conf]``.
    COCO ankle indices 15 and 16 are the defaults.  Both ankle confidences
    must be strictly greater than ``tau_pose``; otherwise the bottom-centre of
    the bounding box is used.  Extra detection columns, such as confidence or
    class ID, are ignored.
    """
    if not np.isfinite(tau_pose) or tau_pose < 0.0:
        raise ValueError("tau_pose must be finite and >= 0")
    if left_ankle_index < 0 or right_ankle_index < 0:
        raise ValueError("ankle indices must be non-negative")

    bbox = np.asarray(detection, dtype=float).reshape(-1)
    if bbox.size < 4 or not np.all(np.isfinite(bbox[:4])):
        raise ValueError("detection must contain four finite bbox coordinates")

    if bbox_format == "xywh":
        x, y, width, height = bbox[:4]
        if width <= 0.0 or height <= 0.0:
            raise ValueError("xywh detection width and height must be positive")
        fallback = np.array([x + width * 0.5, y + height], dtype=float)
    elif bbox_format == "xyxy":
        x1, y1, x2, y2 = bbox[:4]
        if x2 <= x1 or y2 <= y1:
            raise ValueError("xyxy detection must satisfy x2>x1 and y2>y1")
        fallback = np.array([(x1 + x2) * 0.5, y2], dtype=float)
    else:
        raise ValueError(f"unsupported bbox_format: {bbox_format!r}")

    if pose_keypoints is None:
        return GroundPlaneLocation(fallback, "bbox")
    keypoints = np.asarray(pose_keypoints, dtype=float)
    required = max(left_ankle_index, right_ankle_index)
    if (
        keypoints.ndim != 2
        or keypoints.shape[0] <= required
        or keypoints.shape[1] < 3
    ):
        return GroundPlaneLocation(fallback, "bbox")

    left = keypoints[left_ankle_index, :3]
    right = keypoints[right_ankle_index, :3]
    ankles_are_valid = bool(
        np.all(np.isfinite(left))
        and np.all(np.isfinite(right))
        and left[2] > tau_pose
        and right[2] > tau_pose
    )
    if not ankles_are_valid:
        return GroundPlaneLocation(fallback, "bbox")
    return GroundPlaneLocation((left[:2] + right[:2]) * 0.5, "ankles")


def get_ground_plane_location(
    detection: np.ndarray | Sequence[float],
    pose_keypoints: np.ndarray | Sequence[Sequence[float]] | None,
    *,
    tau_pose: float = 0.5,
    bbox_format: BBoxFormat = "xywh",
    left_ankle_index: int = 15,
    right_ankle_index: int = 16,
) -> np.ndarray:
    """Return only the ``(x, y)`` result of :func:`estimate_ground_plane_location`."""
    return estimate_ground_plane_location(
        detection,
        pose_keypoints,
        tau_pose=tau_pose,
        bbox_format=bbox_format,
        left_ankle_index=left_ankle_index,
        right_ankle_index=right_ankle_index,
    ).xy


class TopDownProjector:
    """Project image ground points into one shared top-down coordinate frame.

    Matrices passed here must map ``image -> world/top-down``.  The perspective
    divide is performed explicitly after multiplying homogeneous coordinates.
    """

    def __init__(
        self,
        image_to_world_homographies: Mapping[str, np.ndarray],
        *,
        denominator_epsilon: float = 1e-9,
    ) -> None:
        if not np.isfinite(denominator_epsilon) or denominator_epsilon <= 0.0:
            raise ValueError("denominator_epsilon must be finite and > 0")
        self.denominator_epsilon = float(denominator_epsilon)
        self.homographies: dict[str, np.ndarray] = {}
        for camera_id, value in image_to_world_homographies.items():
            matrix = np.asarray(value, dtype=float)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise ValueError(
                    f"homography for {camera_id!r} must be a finite 3x3 matrix"
                )
            if np.linalg.matrix_rank(matrix) < 3:
                raise ValueError(f"homography for {camera_id!r} is singular")
            self.homographies[str(camera_id)] = matrix.copy()

    @classmethod
    def from_calibrations(
        cls,
        calibrations: Mapping[str, CalibrationWithImageToWorld],
        *,
        denominator_epsilon: float = 1e-9,
    ) -> TopDownProjector:
        """Construct from this project's ``CameraCalibration`` objects."""
        return cls(
            {
                camera_id: calibration.H_image_to_world
                for camera_id, calibration in calibrations.items()
            },
            denominator_epsilon=denominator_epsilon,
        )

    def project(
        self,
        camera_id: str,
        image_xy: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Project one ``(2,)`` point or a batch ``(N, 2)`` to top-down space."""
        if camera_id not in self.homographies:
            raise KeyError(f"unknown camera homography: {camera_id!r}")
        points = np.asarray(image_xy, dtype=float)
        single = points.ndim == 1
        if single:
            points = points.reshape(1, -1)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("image_xy must have shape (2,) or (N, 2)")
        if not np.all(np.isfinite(points)):
            raise ValueError("image_xy must contain only finite values")

        homogeneous = np.column_stack([points, np.ones(len(points), dtype=float)])
        projected = homogeneous @ self.homographies[camera_id].T
        denominator = projected[:, 2]
        invalid = (~np.isfinite(denominator)) | (
            np.abs(denominator) <= self.denominator_epsilon
        )
        if np.any(invalid):
            rows = np.flatnonzero(invalid).tolist()
            raise ProjectionError(
                f"homography maps point rows {rows} to the projective horizon"
            )
        world_xy = projected[:, :2] / denominator[:, None]
        if not np.all(np.isfinite(world_xy)):
            raise ProjectionError("homography produced non-finite top-down coordinates")
        return world_xy[0].copy() if single else world_xy


@dataclass(frozen=True)
class WorldTrackletObservation:
    """One local-track observation expressed in the common world plane."""

    camera_id: str
    local_id: int
    timestamp_s: float
    world_xy: np.ndarray
    confidence: float = 1.0
    velocity_xy: np.ndarray | None = None
    ground_point_source: GroundPointSource = "bbox"

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        object.__setattr__(
            self, "world_xy", _point2(self.world_xy, name="world_xy").copy()
        )
        if self.velocity_xy is not None:
            object.__setattr__(
                self,
                "velocity_xy",
                _point2(self.velocity_xy, name="velocity_xy").copy(),
            )

    @property
    def key(self) -> TrackletKey:
        return self.camera_id, int(self.local_id)


@dataclass(frozen=True)
class SmoothingResult:
    """Result of inserting one raw world point into a trajectory window."""

    key: TrackletKey
    timestamp_s: float
    raw_xy: np.ndarray
    smoothed_xy: np.ndarray
    velocity_xy: np.ndarray
    acceleration_xy: np.ndarray
    predicted_xy: np.ndarray
    accepted: bool
    sample_count: int
    residual_m: float
    allowed_residual_m: float


@dataclass(frozen=True)
class _TrajectorySample:
    timestamp_s: float
    xy: np.ndarray
    confidence: float


class SlidingWindowTrajectorySmoother:
    """Confidence- and recency-weighted trajectory smoothing with outlier veto.

    Position and velocity come from weighted least-squares over all accepted
    samples.  With at least four distinct timestamps, a quadratic term also
    estimates acceleration.  A new point is checked against that trajectory;
    rejected or out-of-order samples never enter the weighted window.
    """

    def __init__(
        self,
        *,
        window_s: float = 1.0,
        recency_half_life_s: float = 0.35,
        outlier_gate_m: float = 0.75,
        max_speed_mps: float = 3.2,
        max_acceleration_mps2: float = 4.0,
        jump_window_s: float = 1.0,
        max_jump_m: float = 2.0,
    ) -> None:
        values = {
            "window_s": window_s,
            "recency_half_life_s": recency_half_life_s,
            "outlier_gate_m": outlier_gate_m,
            "max_speed_mps": max_speed_mps,
            "max_acceleration_mps2": max_acceleration_mps2,
            "jump_window_s": jump_window_s,
            "max_jump_m": max_jump_m,
        }
        for name, value in values.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        self.window_s = float(window_s)
        self.recency_half_life_s = float(recency_half_life_s)
        self.outlier_gate_m = float(outlier_gate_m)
        self.max_speed_mps = float(max_speed_mps)
        self.max_acceleration_mps2 = float(max_acceleration_mps2)
        self.jump_window_s = float(jump_window_s)
        self.max_jump_m = float(max_jump_m)
        self.histories: dict[TrackletKey, deque[_TrajectorySample]] = {}
        self.metrics: defaultdict[str, int] = defaultdict(int)

    def update(
        self,
        key: TrackletKey,
        timestamp_s: float,
        world_xy: np.ndarray | Sequence[float],
        *,
        confidence: float = 1.0,
    ) -> SmoothingResult:
        """Insert a point and return the current robust weighted estimate."""
        timestamp = float(timestamp_s)
        raw = _point2(world_xy, name="world_xy")
        if not np.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")

        canonical_key = str(key[0]), int(key[1])
        history = self.histories.setdefault(canonical_key, deque())
        if history and timestamp < history[-1].timestamp_s - 1e-9:
            self.metrics["out_of_order_rejected"] += 1
            position, velocity, acceleration = self._weighted_regression(
                history, history[-1].timestamp_s
            )
            velocity, acceleration = self._bounded_motion(
                velocity, acceleration
            )
            return SmoothingResult(
                key=canonical_key,
                timestamp_s=timestamp,
                raw_xy=raw.copy(),
                smoothed_xy=position,
                velocity_xy=velocity,
                acceleration_xy=acceleration,
                predicted_xy=position.copy(),
                accepted=False,
                sample_count=len(history),
                residual_m=float(np.linalg.norm(raw - position)),
                allowed_residual_m=0.0,
            )
        self._prune_history(history, timestamp)

        predicted = raw.copy()
        elapsed = 0.0
        if history:
            elapsed = max(0.0, timestamp - history[-1].timestamp_s)
            position, velocity, acceleration = self._weighted_regression(
                history, history[-1].timestamp_s
            )
            velocity, acceleration = self._bounded_motion(
                velocity, acceleration
            )
            predicted = (
                position
                + velocity * elapsed
                + 0.5 * acceleration * elapsed * elapsed
            )

        residual = float(np.linalg.norm(raw - predicted)) if history else 0.0
        allowed = (
            self.outlier_gate_m
            + self.max_speed_mps * elapsed
            + 0.5 * self.max_acceleration_mps2 * elapsed * elapsed
        )
        jump = (
            float(np.linalg.norm(raw - history[-1].xy)) if history else 0.0
        )
        exceeds_jump_cap = bool(
            history
            and elapsed <= self.jump_window_s + 1e-9
            and jump > self.max_jump_m + 1e-9
        )
        accepted = not history or (
            residual <= allowed + 1e-9 and not exceeds_jump_cap
        )
        if accepted:
            sample = _TrajectorySample(timestamp, raw.copy(), float(confidence))
            if history and abs(timestamp - history[-1].timestamp_s) <= 1e-9:
                history[-1] = sample
            else:
                history.append(sample)
            self.metrics["accepted"] += 1
        else:
            self.metrics["outliers_rejected"] += 1
            if exceeds_jump_cap:
                self.metrics["jump_cap_rejected"] += 1

        smoothed, velocity, acceleration = self._weighted_regression(
            history, timestamp
        )
        velocity, acceleration = self._bounded_motion(velocity, acceleration)
        return SmoothingResult(
            key=canonical_key,
            timestamp_s=timestamp,
            raw_xy=raw.copy(),
            smoothed_xy=smoothed,
            velocity_xy=velocity,
            acceleration_xy=acceleration,
            predicted_xy=predicted.copy(),
            accepted=accepted,
            sample_count=len(history),
            residual_m=residual,
            allowed_residual_m=allowed,
        )

    def reset(self, key: TrackletKey | None = None) -> None:
        """Clear one trajectory or the full bounded state."""
        if key is None:
            self.histories.clear()
            return
        self.histories.pop((str(key[0]), int(key[1])), None)

    def prune(self, reference_ts_s: float) -> None:
        """Prune inactive keys and samples against a global event watermark."""
        reference = float(reference_ts_s)
        if not np.isfinite(reference):
            raise ValueError("reference_ts_s must be finite")
        for key, history in list(self.histories.items()):
            self._prune_history(history, reference)
            if not history:
                del self.histories[key]
                self.metrics["histories_pruned"] += 1
        self.metrics["retained_histories"] = len(self.histories)
        self.metrics["retained_samples"] = sum(
            len(history) for history in self.histories.values()
        )

    def _weighted_regression(
        self, history: deque[_TrajectorySample], reference_ts_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not history:
            raise RuntimeError("cannot smooth an empty trajectory")
        timestamps = np.asarray(
            [sample.timestamp_s for sample in history], dtype=float
        )
        times = timestamps - float(reference_ts_s)
        ages = np.maximum(0.0, float(reference_ts_s) - timestamps)
        recency = np.exp(-log(2.0) * ages / self.recency_half_life_s)
        confidence = np.array(
            [max(sample.confidence, 1e-3) for sample in history], dtype=float
        )
        weights = recency * confidence
        points = np.stack([sample.xy for sample in history])
        distinct = len(np.unique(np.round(timestamps, decimals=9)))
        degree = 2 if distinct >= 4 else (1 if distinct >= 2 else 0)
        if degree == 0:
            return (
                np.average(points, axis=0, weights=weights),
                np.zeros(2, dtype=float),
                np.zeros(2, dtype=float),
            )
        columns = [np.ones(len(times)), times]
        if degree == 2:
            columns.append(0.5 * times * times)
        design = np.column_stack(columns)
        root_weight = np.sqrt(weights)[:, None]
        coefficients, _, rank, _ = np.linalg.lstsq(
            design * root_weight,
            points * root_weight,
            rcond=None,
        )
        if rank < design.shape[1]:
            # Degenerate timestamp layouts fall back to a stable weighted mean.
            return (
                np.average(points, axis=0, weights=weights),
                np.zeros(2, dtype=float),
                np.zeros(2, dtype=float),
            )
        position = coefficients[0]
        velocity = coefficients[1]
        acceleration = (
            coefficients[2] if degree == 2 else np.zeros(2, dtype=float)
        )
        return position.copy(), velocity.copy(), acceleration.copy()

    def _bounded_motion(
        self, velocity: np.ndarray, acceleration: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        velocity = np.asarray(velocity, dtype=float).copy()
        acceleration = np.asarray(acceleration, dtype=float).copy()
        speed = float(np.linalg.norm(velocity))
        if speed > self.max_speed_mps:
            velocity *= self.max_speed_mps / speed
        acceleration_norm = float(np.linalg.norm(acceleration))
        if acceleration_norm > self.max_acceleration_mps2:
            acceleration *= self.max_acceleration_mps2 / acceleration_norm
        return velocity, acceleration

    def _prune_history(
        self, history: deque[_TrajectorySample], reference_ts_s: float
    ) -> None:
        cutoff = reference_ts_s - self.window_s
        while history and history[0].timestamp_s < cutoff - 1e-9:
            history.popleft()
            self.metrics["samples_pruned"] += 1


def build_spatial_cost_matrix(
    left_world_xy: np.ndarray | Sequence[Sequence[float]],
    right_world_xy: np.ndarray | Sequence[Sequence[float]],
    *,
    squared: bool = False,
) -> np.ndarray:
    """Return all pairwise Euclidean ground-plane distances.

    Set ``squared=True`` to obtain the squared-distance form used by Equation
    (4) of the paper.  Hungarian association normally uses metric distances.
    """
    left = _points2(left_world_xy, name="left_world_xy")
    right = _points2(right_world_xy, name="right_world_xy")
    if not len(left) or not len(right):
        return np.empty((len(left), len(right)), dtype=float)
    delta = left[:, None, :] - right[None, :, :]
    squared_distance = np.sum(delta * delta, axis=2)
    return squared_distance if squared else np.sqrt(squared_distance)


@dataclass(frozen=True)
class SpatialMatch:
    """One accepted cross-camera assignment."""

    left_key: TrackletKey
    right_key: TrackletKey
    left_index: int
    right_index: int
    cost: float
    spatial_distance_m: float
    time_delta_s: float
    assignment_margin: float


@dataclass(frozen=True)
class AssociationResult:
    """One-to-one assignment plus unmatched inputs for lifecycle handling."""

    matches: tuple[SpatialMatch, ...]
    unmatched_left: tuple[int, ...]
    unmatched_right: tuple[int, ...]
    cost_matrix: np.ndarray


class SpatialConsistencyMatcher:
    """Hungarian cross-camera association using only space and event time."""

    def __init__(
        self,
        *,
        max_distance_m: float = 2.0,
        max_time_delta_s: float = 0.25,
        temporal_weight_m_per_s: float = 0.25,
        min_assignment_margin: float = 0.0,
    ) -> None:
        if not np.isfinite(max_distance_m) or max_distance_m <= 0.0:
            raise ValueError("max_distance_m must be finite and > 0")
        if not np.isfinite(max_time_delta_s) or max_time_delta_s < 0.0:
            raise ValueError("max_time_delta_s must be finite and >= 0")
        if not np.isfinite(temporal_weight_m_per_s) or temporal_weight_m_per_s < 0.0:
            raise ValueError("temporal_weight_m_per_s must be finite and >= 0")
        if not np.isfinite(min_assignment_margin) or min_assignment_margin < 0.0:
            raise ValueError("min_assignment_margin must be finite and >= 0")
        self.max_distance_m = float(max_distance_m)
        self.max_time_delta_s = float(max_time_delta_s)
        self.temporal_weight_m_per_s = float(temporal_weight_m_per_s)
        self.min_assignment_margin = float(min_assignment_margin)

    def build_cost_matrix(
        self,
        left: Sequence[WorldTrackletObservation],
        right: Sequence[WorldTrackletObservation],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build gated total, spatial, and temporal pairwise matrices."""
        total = np.full((len(left), len(right)), np.inf, dtype=float)
        spatial = np.full_like(total, np.inf)
        temporal = np.full_like(total, np.inf)
        for row, left_item in enumerate(left):
            for column, right_item in enumerate(right):
                if left_item.camera_id == right_item.camera_id:
                    continue
                dt = abs(left_item.timestamp_s - right_item.timestamp_s)
                if dt > self.max_time_delta_s + 1e-9:
                    continue
                reference_ts = max(
                    left_item.timestamp_s, right_item.timestamp_s
                )
                left_xy = self._predict_to(left_item, reference_ts)
                right_xy = self._predict_to(right_item, reference_ts)
                distance = float(np.linalg.norm(left_xy - right_xy))
                if distance > self.max_distance_m + 1e-9:
                    continue
                spatial[row, column] = distance
                temporal[row, column] = dt
                total[row, column] = (
                    distance + self.temporal_weight_m_per_s * dt
                )
        return total, spatial, temporal

    def match(
        self,
        left: Sequence[WorldTrackletObservation],
        right: Sequence[WorldTrackletObservation],
    ) -> AssociationResult:
        """Run one-to-one Hungarian assignment on all feasible pairs."""
        total, spatial, temporal = self.build_cost_matrix(left, right)
        if not len(left) or not len(right) or not np.any(np.isfinite(total)):
            return AssociationResult(
                (), tuple(range(len(left))), tuple(range(len(right))), total
            )

        finite_values = total[np.isfinite(total)]
        sentinel = max(1e6, float(np.max(finite_values)) + 1e6)
        assignment_cost = np.where(np.isfinite(total), total, sentinel)
        rows, columns = linear_sum_assignment(assignment_cost)
        matches: list[SpatialMatch] = []
        matched_left: set[int] = set()
        matched_right: set[int] = set()
        for row, column in zip(rows, columns):
            if not np.isfinite(total[row, column]):
                continue
            alternatives = [
                total[row, other]
                for other in range(total.shape[1])
                if other != column and np.isfinite(total[row, other])
            ] + [
                total[other, column]
                for other in range(total.shape[0])
                if other != row and np.isfinite(total[other, column])
            ]
            margin = min(alternatives, default=float("inf")) - total[row, column]
            if margin + 1e-9 < self.min_assignment_margin:
                continue
            matches.append(
                SpatialMatch(
                    left_key=left[row].key,
                    right_key=right[column].key,
                    left_index=int(row),
                    right_index=int(column),
                    cost=float(total[row, column]),
                    spatial_distance_m=float(spatial[row, column]),
                    time_delta_s=float(temporal[row, column]),
                    assignment_margin=float(margin),
                )
            )
            matched_left.add(int(row))
            matched_right.add(int(column))
        return AssociationResult(
            matches=tuple(matches),
            unmatched_left=tuple(
                index for index in range(len(left)) if index not in matched_left
            ),
            unmatched_right=tuple(
                index for index in range(len(right)) if index not in matched_right
            ),
            cost_matrix=total.copy(),
        )

    @staticmethod
    def _predict_to(
        observation: WorldTrackletObservation, reference_ts_s: float
    ) -> np.ndarray:
        if observation.velocity_xy is None:
            return observation.world_xy
        dt = max(0.0, reference_ts_s - observation.timestamp_s)
        return observation.world_xy + observation.velocity_xy * dt


def spatial_consistency_distance(
    target_world_xy: np.ndarray | Sequence[float],
    peer_world_xy: np.ndarray | Sequence[Sequence[float]],
    *,
    outlier_mad_scale: float = 3.5,
) -> float:
    """Mean squared cross-view distance after robust peer outlier removal."""
    target = _point2(target_world_xy, name="target_world_xy")
    peers = _points2(peer_world_xy, name="peer_world_xy")
    if not len(peers):
        return float("inf")
    if not np.isfinite(outlier_mad_scale) or outlier_mad_scale <= 0.0:
        raise ValueError("outlier_mad_scale must be finite and > 0")
    inliers = _robust_point_inliers(peers, outlier_mad_scale)
    error = peers[inliers] - target
    return float(np.mean(np.sum(error * error, axis=1)))


def reassignment_confidence(
    current_spatial_distance: float,
    proposed_spatial_distance: float,
) -> float:
    """Equation (5): relative consistency gain for ``current -> proposed``."""
    current = float(current_spatial_distance)
    proposed = float(proposed_spatial_distance)
    if not np.isfinite(current) or current <= 0.0:
        raise ValueError("current_spatial_distance must be finite and > 0")
    if not np.isfinite(proposed) or proposed < 0.0:
        raise ValueError("proposed_spatial_distance must be finite and >= 0")
    return 1.0 - proposed / current


class SpatioTemporalAssociator:
    """Facade from local detections to No-ReID cross-camera assignments."""

    def __init__(
        self,
        projector: TopDownProjector,
        *,
        smoother: SlidingWindowTrajectorySmoother | None = None,
        matcher: SpatialConsistencyMatcher | None = None,
        tau_pose: float = 0.5,
    ) -> None:
        if not np.isfinite(tau_pose) or tau_pose < 0.0:
            raise ValueError("tau_pose must be finite and >= 0")
        self.projector = projector
        self.smoother = smoother or SlidingWindowTrajectorySmoother()
        self.matcher = matcher or SpatialConsistencyMatcher()
        self.tau_pose = float(tau_pose)

    def observe(
        self,
        *,
        camera_id: str,
        local_id: int,
        timestamp_s: float,
        detection: np.ndarray | Sequence[float],
        pose_keypoints: np.ndarray | Sequence[Sequence[float]] | None = None,
        confidence: float = 1.0,
        bbox_format: BBoxFormat = "xyxy",
    ) -> WorldTrackletObservation:
        """Project and smooth one confirmed single-camera track observation."""
        ground = estimate_ground_plane_location(
            detection,
            pose_keypoints,
            tau_pose=self.tau_pose,
            bbox_format=bbox_format,
        )
        raw_world = self.projector.project(camera_id, ground.xy)
        smoothed = self.smoother.update(
            (camera_id, int(local_id)),
            timestamp_s,
            raw_world,
            confidence=confidence,
        )
        return WorldTrackletObservation(
            camera_id=camera_id,
            local_id=int(local_id),
            timestamp_s=float(timestamp_s),
            world_xy=smoothed.smoothed_xy,
            confidence=float(confidence),
            velocity_xy=smoothed.velocity_xy,
            ground_point_source=ground.source,
        )

    def associate(
        self,
        left: Sequence[WorldTrackletObservation],
        right: Sequence[WorldTrackletObservation],
    ) -> AssociationResult:
        """Associate two camera observation sets without appearance features."""
        return self.matcher.match(left, right)


def _point2(value: np.ndarray | Sequence[float], *, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite array with shape (2,)")
    return point


def _points2(
    value: np.ndarray | Sequence[Sequence[float]], *, name: str
) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be a finite array with shape (N, 2)")
    return points


def _robust_point_inliers(points: np.ndarray, scale: float) -> np.ndarray:
    if len(points) <= 2:
        return np.ones(len(points), dtype=bool)
    center = np.median(points, axis=0)
    radial = np.linalg.norm(points - center, axis=1)
    median = float(np.median(radial))
    mad = float(np.median(np.abs(radial - median)))
    robust_sigma = 1.4826 * mad
    threshold = median + scale * max(robust_sigma, 1e-9)
    inliers = radial <= threshold + 1e-9
    if not np.any(inliers):
        inliers[int(np.argmin(radial))] = True
    return inliers
