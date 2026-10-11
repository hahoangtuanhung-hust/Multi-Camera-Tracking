"""Short metadata-only trajectory history and cross-camera matching.

No appearance embedding or image is retained here. A tracklet is just a
bounded sequence of timestamped ground-plane positions, velocities and
detector confidence values.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import exp
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import PipelineConfig

TrackletKey = tuple[str, int]


@dataclass
class TimedWorldPoint:
    timestamp_s: float
    xy: np.ndarray
    velocity: np.ndarray
    confidence: float
    velocity_supplied: bool = False


@dataclass
class TrackletHistory:
    camera: str
    local_id: int
    points: deque[TimedWorldPoint] = field(default_factory=deque)
    _arrays: tuple[np.ndarray, ...] | None = field(default=None, init=False, repr=False)
    _bounds: tuple[np.ndarray, np.ndarray] | None = field(default=None, init=False, repr=False)

    def invalidate(self) -> None:
        self._arrays = None
        self._bounds = None

    def arrays(self) -> tuple[np.ndarray, ...]:
        """Materialize metadata once per revision, not once per offset sample."""
        if self._arrays is None:
            self._arrays = (
                np.array([point.timestamp_s for point in self.points]),
                np.stack([point.xy for point in self.points]),
                np.stack([point.velocity for point in self.points]),
                np.array([point.confidence for point in self.points]),
            )
        return self._arrays

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self._bounds is None:
            xy = self.arrays()[1]
            self._bounds = xy.min(axis=0), xy.max(axis=0)
        return self._bounds

    @property
    def key(self) -> TrackletKey:
        return self.camera, self.local_id

    @property
    def start_s(self) -> float:
        return self.points[0].timestamp_s

    @property
    def end_s(self) -> float:
        return self.points[-1].timestamp_s

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s) if self.points else 0.0


class TrackletStore:
    """Bounded event-time store keyed by ``(camera, local_id)``."""

    def __init__(self, cfg: PipelineConfig):
        self.history_s = cfg.tracklet_history_s
        self.late_drop_s = cfg.tracklet_late_drop_s
        self.min_duration_s = cfg.tracklet_min_duration_s
        self.histories: dict[TrackletKey, TrackletHistory] = {}
        self.max_seen_timestamp_s = float("-inf")
        self.metrics: defaultdict[str, int] = defaultdict(int)
        self._last_pruned_s = float("-inf")

    def advance(self, timestamp_s: float) -> None:
        """Advance retention even on frames containing no detections."""
        if not np.isfinite(timestamp_s):
            raise ValueError("Tracklet timestamp must be finite")
        self.max_seen_timestamp_s = max(self.max_seen_timestamp_s, float(timestamp_s))
        self._prune()

    def append(
        self,
        camera: str,
        local_id: int,
        timestamp_s: float,
        xy: np.ndarray,
        confidence: float,
        velocity: np.ndarray | None = None,
    ) -> bool:
        timestamp_s = float(timestamp_s)
        if not np.isfinite(timestamp_s):
            raise ValueError("Tracklet timestamp must be finite")
        if timestamp_s < self.max_seen_timestamp_s - min(self.late_drop_s, self.history_s):
            self.metrics["late_dropped"] += 1
            return False
        if timestamp_s < self.max_seen_timestamp_s:
            self.metrics["out_of_order_accepted"] += 1
        self.max_seen_timestamp_s = max(self.max_seen_timestamp_s, timestamp_s)

        key = camera, int(local_id)
        history = self.histories.get(key)
        if history is None:
            history = self.histories[key] = TrackletHistory(camera, int(local_id))
        supplied_velocity = None if velocity is None else np.asarray(velocity, dtype=float).copy()
        point = TimedWorldPoint(
            timestamp_s=timestamp_s,
            xy=np.asarray(xy, dtype=float).copy(),
            velocity=np.zeros(2, dtype=float) if supplied_velocity is None else supplied_velocity,
            confidence=float(confidence),
            velocity_supplied=supplied_velocity is not None,
        )

        history.invalidate()
        if not history.points or timestamp_s > history.end_s + 1e-9:
            # The normal path is O(1). Keep velocities supplied by the local
            # tracker, or extend the existing EMA without rescanning history.
            if supplied_velocity is None and history.points:
                previous = history.points[-1]
                dt = timestamp_s - previous.timestamp_s
                if dt > 1e-6:
                    raw = (point.xy - previous.xy) / dt
                    point.velocity = raw if len(history.points) == 1 else .5 * raw + .5 * previous.velocity
            history.points.append(point)
            self.metrics["points_stored"] += 1
        else:
            # Only late/replaced points require ordered insertion and repair.
            values = list(history.points)
            for index, existing in enumerate(values):
                if abs(existing.timestamp_s - timestamp_s) < 1e-9:
                    values[index] = point
                    break
            else:
                values.append(point)
                values.sort(key=lambda item: item.timestamp_s)
                self.metrics["points_stored"] += 1
            history.points = deque(values)
            self._recompute_velocities(history)
        self._prune()
        return True

    @staticmethod
    def _recompute_velocities(
        history: TrackletHistory,
    ) -> None:
        points = list(history.points)
        smoothed = np.zeros(2, dtype=float)
        for index, point in enumerate(points):
            if index == 0:
                if not point.velocity_supplied:
                    point.velocity = np.zeros(2, dtype=float)
                continue
            previous = points[index - 1]
            dt = point.timestamp_s - previous.timestamp_s
            if dt <= 1e-6:
                continue
            raw = (point.xy - previous.xy) / dt
            smoothed = raw if index == 1 else 0.5 * raw + 0.5 * previous.velocity
            if not point.velocity_supplied:
                point.velocity = smoothed.copy()

    def _prune(self) -> None:
        if self.max_seen_timestamp_s <= self._last_pruned_s:
            return
        self._last_pruned_s = self.max_seen_timestamp_s
        cutoff = self.max_seen_timestamp_s - self.history_s
        empty: list[TrackletKey] = []
        for key, history in self.histories.items():
            while history.points and history.points[0].timestamp_s < cutoff:
                history.points.popleft()
                history.invalidate()
                self.metrics["points_pruned"] += 1
                self.metrics["points_stored"] -= 1
            if not history.points:
                empty.append(key)
        for key in empty:
            del self.histories[key]

    def get(self, key: TrackletKey) -> TrackletHistory | None:
        return self.histories.get(key)

    def eligible(self) -> list[TrackletHistory]:
        return [
            history
            for history in self.histories.values()
            if len(history.points) >= 2 and history.duration_s >= self.min_duration_s
        ]

    @property
    def approximate_bytes(self) -> int:
        # 1 timestamp + 2 xy + 2 velocity + 1 confidence, all float64.
        return self.metrics["points_stored"] * 6 * 8


@dataclass(frozen=True)
class MatchResult:
    left_key: TrackletKey
    right_key: TrackletKey
    accepted: bool
    cost: float
    delta_s: float
    overlap_s: float
    position_rmse_m: float
    velocity_rmse_mps: float
    direction_cosine: float
    confidence: float
    margin: float = float("inf")
    reason: str = ""


class TrackletMatcher:
    """Compare trajectory shape in a common timeline without appearance."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.metrics: defaultdict[str, int] = defaultdict(int)

    def _spatially_plausible(self, left: TrackletHistory, right: TrackletHistory) -> bool:
        """Cheap O(points) reject before the offset scan/interpolation.

        A clock shift changes *when* a path is traversed, not its world-space
        footprint. Non-overlapping expanded footprints therefore cannot match.
        """
        left_min, left_max = left.bounds()
        right_min, right_max = right.bounds()
        separation = np.maximum(0.0, np.maximum(left_min - right_max, right_min - left_max))
        return float(np.linalg.norm(separation)) <= self.cfg.tracklet_position_gate_m * 1.5

    @staticmethod
    def _interpolate(history: TrackletHistory, times: np.ndarray, shift_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source_t, xy, velocity, confidence = history.arrays()
        source_t = source_t + shift_s if shift_s else source_t
        sampled_xy = np.column_stack([
            np.interp(times, source_t, xy[:, axis]) for axis in range(2)
        ])
        sampled_velocity = np.column_stack([
            np.interp(times, source_t, velocity[:, axis]) for axis in range(2)
        ])
        sampled_confidence = np.interp(times, source_t, confidence)
        return sampled_xy, sampled_velocity, sampled_confidence

    def compare(
        self,
        left: TrackletHistory,
        right: TrackletHistory,
        delta_s: float,
    ) -> MatchResult:
        """Shift right timestamps by ``delta_s`` and compare with left."""
        start = max(left.start_s, right.start_s + delta_s)
        end = min(left.end_s, right.end_s + delta_s)
        overlap = end - start
        if overlap + 1e-9 < self.cfg.tracklet_min_overlap_s:
            return MatchResult(
                left.key, right.key, False, float("inf"), delta_s, max(0.0, overlap),
                float("inf"), float("inf"), -1.0, 0.0, reason="insufficient_overlap",
            )

        count = max(2, int(np.floor(overlap * self.cfg.tracklet_resample_hz)) + 1)
        times = np.linspace(start, end, count)
        left_xy, left_velocity, left_conf = self._interpolate(left, times, 0.0)
        right_xy, right_velocity, right_conf = self._interpolate(right, times, delta_s)

        position_error = np.linalg.norm(left_xy - right_xy, axis=1)
        velocity_error = np.linalg.norm(left_velocity - right_velocity, axis=1)
        # A clipped RMS is robust to a few noisy homography/detection points.
        pos_clip = max(self.cfg.tracklet_position_gate_m * 2.0, 1e-6)
        vel_clip = max(self.cfg.tracklet_velocity_gate_mps * 2.0, 1e-6)
        position_rmse = float(np.sqrt(np.mean(np.minimum(position_error, pos_clip) ** 2)))
        velocity_rmse = float(np.sqrt(np.mean(np.minimum(velocity_error, vel_clip) ** 2)))

        left_speed = np.linalg.norm(left_velocity, axis=1)
        right_speed = np.linalg.norm(right_velocity, axis=1)
        moving = (left_speed > 0.08) & (right_speed > 0.08)
        if np.any(moving):
            cosine = np.sum(left_velocity[moving] * right_velocity[moving], axis=1)
            cosine /= left_speed[moving] * right_speed[moving]
            direction_cosine = float(np.median(cosine))
        else:
            direction_cosine = 1.0  # direction is not evidence for stationary tracks

        mean_confidence = float(np.mean(np.minimum(left_conf, right_conf)))
        direction_penalty = max(0.0, 1.0 - direction_cosine) / max(
            1e-6, 1.0 - self.cfg.tracklet_direction_cosine
        )
        cost = (
            self.cfg.tracklet_position_weight * position_rmse / self.cfg.tracklet_position_gate_m
            + self.cfg.tracklet_velocity_weight * velocity_rmse / self.cfg.tracklet_velocity_gate_mps
            + self.cfg.tracklet_direction_weight * direction_penalty
            + self.cfg.tracklet_confidence_weight * (1.0 - mean_confidence)
        )
        accepted = (
            position_rmse <= self.cfg.tracklet_position_gate_m
            and velocity_rmse <= self.cfg.tracklet_velocity_gate_mps
            and direction_cosine >= self.cfg.tracklet_direction_cosine
        )
        reason = "" if accepted else "trajectory_gate"
        confidence = mean_confidence * exp(-max(0.0, cost))
        return MatchResult(
            left.key, right.key, accepted, float(cost), float(delta_s), float(overlap),
            position_rmse, velocity_rmse, direction_cosine, float(confidence), reason=reason,
        )

    def best_offset(
        self,
        left: TrackletHistory,
        right: TrackletHistory,
        center_s: float = 0.0,
        radius_s: float | None = None,
    ) -> MatchResult:
        radius = self.cfg.max_camera_offset_s if radius_s is None else max(0.0, radius_s)
        fine_step = max(0.01, self.cfg.offset_scan_step_s)
        coarse_step = max(0.25, fine_step * 4.0)
        lower = max(-self.cfg.max_camera_offset_s, center_s - radius)
        upper = min(self.cfg.max_camera_offset_s, center_s + radius)
        coarse_offsets = np.arange(lower, upper + coarse_step * 0.5, coarse_step)
        if not len(coarse_offsets):
            coarse_offsets = np.array([np.clip(center_s, lower, upper)])
        coarse_results = [self.compare(left, right, float(delta)) for delta in coarse_offsets]
        coarse_best = min(coarse_results, key=lambda result: result.cost)

        # Refine only around the best coarse basin. This preserves a bounded
        # global search while cutting steady CPU work by roughly 2–3x.
        refine_lower = max(lower, coarse_best.delta_s - coarse_step)
        refine_upper = min(upper, coarse_best.delta_s + coarse_step)
        fine_offsets = np.arange(refine_lower, refine_upper + fine_step * 0.5, fine_step)
        fine_results = [self.compare(left, right, float(delta)) for delta in fine_offsets]
        return min(coarse_results + fine_results, key=lambda result: result.cost)

    def assign(
        self,
        left: Iterable[TrackletHistory],
        right: Iterable[TrackletHistory],
        center_s: float = 0.0,
        radius_s: float | None = None,
    ) -> list[MatchResult]:
        left_items, right_items = list(left), list(right)
        if not left_items or not right_items:
            return []
        matrix = np.full((len(left_items), len(right_items)), 1e6, dtype=float)
        details: dict[tuple[int, int], MatchResult] = {}
        for row, left_history in enumerate(left_items):
            for column, right_history in enumerate(right_items):
                self.metrics["candidate_pairs"] += 1
                if not self._spatially_plausible(left_history, right_history):
                    self.metrics["spatial_pruned"] += 1
                    continue
                result = self.best_offset(left_history, right_history, center_s, radius_s)
                details[row, column] = result
                if result.accepted:
                    matrix[row, column] = result.cost
                else:
                    self.metrics["trajectory_rejected"] += 1
                    if result.overlap_s < self.cfg.tracklet_min_overlap_s:
                        self.metrics["overlap_gate_failed"] += 1
                    if result.position_rmse_m > self.cfg.tracklet_position_gate_m:
                        self.metrics["position_gate_failed"] += 1
                    if result.velocity_rmse_mps > self.cfg.tracklet_velocity_gate_mps:
                        self.metrics["velocity_gate_failed"] += 1
                    if result.direction_cosine < self.cfg.tracklet_direction_cosine:
                        self.metrics["direction_gate_failed"] += 1
        rows, columns = linear_sum_assignment(matrix)
        accepted: list[MatchResult] = []
        for row, column in zip(rows, columns):
            if (row, column) not in details:
                continue
            result = details[row, column]
            if not result.accepted or matrix[row, column] >= 1e6:
                continue
            alternatives = [
                matrix[row, other] for other in range(matrix.shape[1]) if other != column
            ] + [
                matrix[other, column] for other in range(matrix.shape[0]) if other != row
            ]
            second = min(alternatives, default=float("inf"))
            margin = second - result.cost
            if margin + 1e-9 < self.cfg.tracklet_assignment_margin:
                self.metrics["ambiguity_margin_failed"] += 1
                continue
            accepted.append(MatchResult(**{**result.__dict__, "margin": float(margin)}))
            self.metrics["assignments_accepted"] += 1
        return accepted


@dataclass(frozen=True)
class ClockEstimate:
    offset_s: float = 0.0
    scale: float = 1.0
    confidence: float = 0.0
    samples: int = 0


class CameraClockOffsetEstimator:
    """Robust camera-level affine clock estimates from accepted trajectories."""

    def __init__(self, cfg: PipelineConfig, reference_camera: str):
        self.cfg = cfg
        self.reference_camera = reference_camera
        self._samples: defaultdict[str, deque[tuple[float, float, float]]] = defaultdict(
            lambda: deque(maxlen=64)
        )
        self._estimates: dict[str, ClockEstimate] = {
            reference_camera: ClockEstimate(0.0, 1.0, 1.0, 1)
        }

    def estimate(self, camera: str) -> ClockEstimate:
        return self._estimates.get(camera, ClockEstimate())

    def correction_at(self, camera: str, camera_timestamp_s: float) -> float:
        estimate = self.estimate(camera)
        return (estimate.scale - 1.0) * camera_timestamp_s + estimate.offset_s

    def relative_shift(self, left_camera: str, right_camera: str, timestamp_s: float) -> float:
        return self.correction_at(right_camera, timestamp_s) - self.correction_at(left_camera, timestamp_s)

    def observe(
        self,
        camera: str,
        camera_timestamp_s: float,
        correction_s: float,
        match_confidence: float,
    ) -> ClockEstimate:
        if camera == self.reference_camera:
            return self._estimates[camera]
        if match_confidence < self.cfg.offset_min_confidence:
            return self.estimate(camera)
        if abs(correction_s) > self.cfg.max_camera_offset_s:
            return self.estimate(camera)

        samples = self._samples[camera]
        if len(samples) >= 4:
            values = np.array([sample[1] for sample in samples])
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            tolerance = max(0.15, 4.0 * 1.4826 * mad)
            if abs(correction_s - median) > tolerance:
                return self.estimate(camera)
        samples.append((float(camera_timestamp_s), float(correction_s), float(match_confidence)))

        times = np.array([sample[0] for sample in samples])
        corrections = np.array([sample[1] for sample in samples])
        weights = np.array([sample[2] for sample in samples])
        center = float(np.average(times, weights=weights))
        if len(samples) >= 3 and np.ptp(times) >= self.cfg.offset_min_drift_span_s:
            slope, centered_intercept = np.polyfit(times - center, corrections, 1, w=weights)
        else:
            slope, centered_intercept = 0.0, float(np.median(corrections))
        max_slope = self.cfg.offset_max_drift_ppm * 1e-6
        slope = float(np.clip(slope, -max_slope, max_slope))
        intercept = float(centered_intercept - slope * center)
        previous = self.estimate(camera)
        alpha = 1.0 if previous.samples == 0 else self.cfg.offset_ewma_alpha
        offset = (1.0 - alpha) * previous.offset_s + alpha * intercept
        scale = (1.0 - alpha) * previous.scale + alpha * (1.0 + slope)
        confidence = min(1.0, len(samples) / 5.0) * float(np.median(weights))
        estimate = ClockEstimate(offset, scale, confidence, len(samples))
        self._estimates[camera] = estimate
        return estimate
