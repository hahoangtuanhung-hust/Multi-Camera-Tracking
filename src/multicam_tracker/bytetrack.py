"""Minimal single-camera multi-object tracker (simplified ByteTrack).

Detection tells you *where people are*; tracking tells you *who is who over
time*. This module maintains a stable local ID per person within one camera
by predicting where each known person will be next (Kalman filter) and
matching predictions to fresh detections (Hungarian algorithm on IoU).

We implement the two ideas that made ByteTrack (ECCV 2022) strong while
staying ~200 readable lines:

1. **Two-stage association.** Match high-confidence detections first; then
   give still-unmatched tracks a second chance against *low*-confidence
   detections. Occluded people usually yield low-confidence boxes, so this
   rescues tracks that a single-pass tracker would drop.
2. **Constant-velocity Kalman filter.** Each track predicts its next box
   from its motion, so matching works even when a person moves quickly or a
   detection flickers for a frame.

The same Kalman filter class is reused by ``fusion.py`` for smoothing
tracks on the world ground plane — one motion model, two altitudes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import PipelineConfig


class ConstantVelocityKalman:
    """Kalman filter with state [z, ż] for any measured vector z.

    Standard textbook filter: `predict()` extrapolates state one tick with
    constant velocity, `update(z)` blends in a new measurement weighted by
    the relative uncertainty of prediction vs. measurement.
    """

    def __init__(self, initial: np.ndarray, process_noise: float, measurement_noise: float):
        m = len(initial)
        self.m = m
        self.x = np.concatenate([initial, np.zeros(m)])  # [positions, velocities]

        self.H = np.hstack([np.eye(m), np.zeros((m, m))])  # we only measure z

        # Discrete white-noise-acceleration model: unknown accelerations are
        # the process noise. For time step dt the standard result is
        #   Q = q * [[dt⁴/4, dt³/2],
        #            [dt³/2, dt²  ]]   (per measured dimension)
        # and with dt = 1 tick those factors become 0.25, 0.5 and 1.
        self.process_noise = process_noise
        self.R = np.eye(m) * measurement_noise

        self.P = np.eye(2 * m) * measurement_noise
        self.P[m:, m:] *= 100.0  # initial velocity is unknown
        self._transition = np.eye(2 * m)
        self._process_cov = np.zeros((2 * m, 2 * m))
        self._indices = np.arange(m)
        self._identity = np.eye(2 * m)

    def predict(
        self, dt: float = 1.0, process_noise_scale: float = 1.0
    ) -> np.ndarray:
        """Predict by an actual time delta (seconds or legacy ticks)."""
        if not np.isfinite(dt) or dt < 0:
            raise ValueError(f"Kalman dt must be non-negative, got {dt}")
        if not np.isfinite(process_noise_scale) or process_noise_scale < 0:
            raise ValueError(
                f"Kalman process-noise scale must be non-negative, got {process_noise_scale}"
            )
        if dt == 0:
            return self.position
        transition = self._transition
        index = self._indices
        transition[index, index + self.m] = dt
        q = self.process_noise * process_noise_scale
        process_cov = self._process_cov
        process_cov[index, index] = q * dt**4 / 4
        process_cov[index, index + self.m] = q * dt**3 / 2
        process_cov[index + self.m, index] = q * dt**3 / 2
        process_cov[index + self.m, index + self.m] = q * dt**2
        self.x = transition @ self.x
        self.P = transition @ self.P @ transition.T + process_cov
        return self.x[: self.m]

    def update(
        self,
        z: np.ndarray,
        measurement_noise_scale: float = 1.0,
        nis_threshold: float | None = None,
    ) -> None:
        """Correct the state, robustly down-weighting a statistical outlier.

        ``nis_threshold`` is a normalized-innovation-squared threshold. Above
        it, measurement covariance is inflated for this update only; the
        observation still contributes, but a merged/partial bbox cannot pull
        the predicted walking direction across another person in one frame.
        """
        if not np.isfinite(measurement_noise_scale) or measurement_noise_scale <= 0:
            raise ValueError(
                "Kalman measurement-noise scale must be positive, "
                f"got {measurement_noise_scale}"
            )
        innovation = z - self.H @ self.x
        measurement_cov = self.R * measurement_noise_scale
        S = self.H @ self.P @ self.H.T + measurement_cov
        if nis_threshold is not None and nis_threshold > 0:
            nis = float(innovation @ np.linalg.solve(S, innovation))
            if np.isfinite(nis) and nis > nis_threshold:
                inflation = min(nis / nis_threshold, 100.0)
                measurement_cov = measurement_cov * inflation
                S = self.H @ self.P @ self.H.T + measurement_cov
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (self._identity - K @ self.H) @ self.P

    def projected_measurement(
        self,
        dt: float = 0.0,
        process_noise_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return future measurement mean/covariance without mutating state.

        Fusion uses this for statistically gated fragment reconnection. A
        longer gap naturally grows covariance, while an implausible position
        can still be rejected by its normalized innovation distance.
        """
        if not np.isfinite(dt) or dt < 0:
            raise ValueError(f"Kalman dt must be non-negative, got {dt}")
        if not np.isfinite(process_noise_scale) or process_noise_scale < 0:
            raise ValueError(
                "Kalman process-noise scale must be non-negative, "
                f"got {process_noise_scale}"
            )
        transition = np.eye(2 * self.m)
        index = np.arange(self.m)
        transition[index, index + self.m] = dt
        q = self.process_noise * process_noise_scale
        process_cov = np.zeros_like(self.P)
        process_cov[index, index] = q * dt**4 / 4
        process_cov[index, index + self.m] = q * dt**3 / 2
        process_cov[index + self.m, index] = q * dt**3 / 2
        process_cov[index + self.m, index + self.m] = q * dt**2
        predicted_state = transition @ self.x
        predicted_covariance = transition @ self.P @ transition.T + process_cov
        mean = self.H @ predicted_state
        covariance = self.H @ predicted_covariance @ self.H.T + self.R
        return mean, covariance

    @property
    def position(self) -> np.ndarray:
        return self.x[: self.m]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[self.m:]


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of (N, 4) xyxy boxes."""
    a = boxes_a[:, None, :]  # (A, 1, 4)
    b = boxes_b[None, :, :]  # (1, B, 4)
    inter_w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    inter_h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = inter_w * inter_h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


def match_by_iou(tracks_xyxy: np.ndarray, dets_xyxy: np.ndarray, min_iou: float):
    """Hungarian matching on IoU. Returns (pairs, unmatched_tracks, unmatched_dets)."""
    if len(tracks_xyxy) == 0 or len(dets_xyxy) == 0:
        return [], list(range(len(tracks_xyxy))), list(range(len(dets_xyxy)))
    iou = iou_matrix(tracks_xyxy, dets_xyxy)
    # Preserve maximum total overlap before rejecting weak pairs. Forcing
    # maximum match count can replace one strong identity match with two
    # ambiguous weak matches, increasing ID switches in crossing tracks.
    rows, cols = linear_sum_assignment(-iou)
    pairs = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= min_iou]
    matched_t = {r for r, _ in pairs}
    matched_d = {c for _, c in pairs}
    return (
        pairs,
        [t for t in range(len(tracks_xyxy)) if t not in matched_t],
        [d for d in range(len(dets_xyxy)) if d not in matched_d],
    )


def match_tracks_motion(
    tracks: list["Track"],
    detections: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Associate boxes with IoU plus statistically normalized motion.

    The Kalman states have already been predicted to the current capture
    timestamp.  Active tracks still require the normal IoU gate.  A confirmed
    track that was missed recently may instead pass the NIS and size gates,
    which is the inexpensive reactivation path needed after a short
    occlusion.  The cost matrix is solved globally so detection ordering never
    determines identity ownership.
    """
    if not tracks or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))

    detections = np.asarray(detections, dtype=float).reshape((-1, 5))
    track_boxes = np.stack([track.xyxy for track in tracks])
    overlap = iou_matrix(track_boxes, detections[:, :4])
    sentinel = 1e6
    cost = np.full(overlap.shape, sentinel, dtype=float)
    gate = max(float(cfg.track_association_nis_gate), 1e-9)
    max_size_change = max(float(cfg.track_association_max_size_log_change), 1e-9)

    for row, track in enumerate(tracks):
        predicted, covariance = track.kf.projected_measurement()
        try:
            covariance_inverse = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            continue
        for column, detection in enumerate(detections):
            measurement = _xyxy_to_box_state(detection)
            if np.any(measurement[2:] <= 0.0):
                continue
            innovation = measurement - predicted
            nis = float(innovation @ covariance_inverse @ innovation)
            if not np.isfinite(nis):
                continue

            predicted_size = np.maximum(predicted[2:], 1.0)
            measured_size = np.maximum(measurement[2:], 1.0)
            size_change = float(
                np.mean(np.abs(np.log(measured_size / predicted_size)))
            )
            recently_lost = (
                track.confirmed
                and track.misses > 0
                and track.missed_seconds <= cfg.track_motion_reactivation_s + 1e-9
            )
            passes_iou = overlap[row, column] >= cfg.track_min_iou
            if not passes_iou and not recently_lost:
                continue
            # Preserve ByteTrack's reliable active-track candidate set.  Hard
            # statistical/scale gates are for the low-IoU reactivation path;
            # on a visible track they would turn harmless detector box jitter
            # into a new local ID even though overlap is still strong.
            if recently_lost and (nis > gate or size_change > max_size_change):
                continue

            direction_penalty = 0.0
            if track.last_measurement is not None:
                velocity = np.asarray(track.kf.velocity[:2], dtype=float)
                displacement = measurement[:2] - track.last_measurement[:2]
                speed = float(np.linalg.norm(velocity))
                travel = float(np.linalg.norm(displacement))
                if speed > cfg.track_motion_min_speed and travel > 1e-6:
                    cosine = float(np.dot(velocity, displacement) / (speed * travel))
                    direction_penalty = 0.5 * (1.0 - np.clip(cosine, -1.0, 1.0))

            cost[row, column] = (
                cfg.track_association_iou_weight * (1.0 - overlap[row, column])
                + cfg.track_association_nis_weight * min(nis / gate, 1.0)
                + cfg.track_association_size_weight
                * min(size_change / max_size_change, 1.0)
                + cfg.track_association_direction_weight * direction_penalty
            )

    rows, columns = linear_sum_assignment(cost)
    pairs = [
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if cost[row, column] < sentinel
    ]
    matched_tracks = {row for row, _ in pairs}
    matched_detections = {column for _, column in pairs}
    return (
        pairs,
        [index for index in range(len(tracks)) if index not in matched_tracks],
        [index for index in range(len(detections)) if index not in matched_detections],
    )


def match_low_confidence_motion(
    tracks: list,
    detections: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian BYTE recovery with IoU, covariance, size and direction gates.

    This operates in image space: metric speed limits are deliberately left to
    world-plane fusion after homography projection. Low-confidence detections
    can update an existing track but are never returned as birth candidates.
    """
    if (
        not tracks
        or len(detections) == 0
        or not cfg.track_detection_low_conf_recovery_enabled
    ):
        return [], list(range(len(tracks))), list(range(len(detections)))

    detections = np.asarray(detections, dtype=float).reshape((-1, 5))
    track_boxes = np.stack([
        np.asarray(
            track.predicted_xyxy
            if hasattr(track, "predicted_xyxy")
            else track.xyxy,
            dtype=float,
        )
        for track in tracks
    ])
    overlap = iou_matrix(track_boxes, detections[:, :4])
    sentinel = 1e6
    cost = np.full(overlap.shape, sentinel, dtype=float)
    nis_gate = max(float(cfg.track_association_nis_gate), 1e-9)
    size_gate = max(float(cfg.track_association_max_size_log_change), 1e-9)

    for row, track in enumerate(tracks):
        predicted, covariance = track.kf.projected_measurement(0.0)
        for column, detection in enumerate(detections):
            if overlap[row, column] < cfg.track_min_iou:
                continue
            measurement = _xyxy_to_box_state(detection)
            if np.any(measurement[2:] <= 0.0):
                continue
            nis = 0.0
            size_change = 0.0
            if cfg.track_motion_gate_enabled:
                innovation = measurement - predicted
                try:
                    nis = float(innovation @ np.linalg.solve(covariance, innovation))
                except np.linalg.LinAlgError:
                    continue
                predicted_size = np.maximum(predicted[2:], 1.0)
                measured_size = np.maximum(measurement[2:], 1.0)
                size_change = float(
                    np.mean(np.abs(np.log(measured_size / predicted_size)))
                )
                if (
                    not np.isfinite(nis)
                    or nis > nis_gate
                    or size_change > size_gate
                ):
                    continue

                velocity = getattr(track, "velocity", None)
                previous = getattr(track, "last_observation", None)
                if velocity is None and hasattr(track, "last_measurement"):
                    velocity = np.asarray(track.kf.velocity[:2], dtype=float)
                    previous_state = getattr(track, "last_measurement", None)
                    previous = (
                        None
                        if previous_state is None
                        else _box_state_to_xyxy(previous_state)
                    )
                if velocity is not None and previous is not None:
                    velocity = np.asarray(velocity, dtype=float)[:2]
                    speed = float(np.linalg.norm(velocity))
                    displacement = measurement[:2] - _xyxy_to_box_state(previous)[:2]
                    travel = float(np.linalg.norm(displacement))
                    if speed > cfg.track_motion_min_speed and travel > 1e-6:
                        direction_cosine = float(
                            np.dot(velocity, displacement) / (speed * travel)
                        )
                        if direction_cosine < -0.45:
                            continue

            cost[row, column] = (
                1.0 - overlap[row, column]
                + 0.05 * min(nis / nis_gate, 1.0)
                + 0.05 * min(size_change / size_gate, 1.0)
            )

    rows, columns = linear_sum_assignment(cost)
    pairs = [
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if cost[row, column] < sentinel
    ]
    matched_tracks = {row for row, _column in pairs}
    matched_detections = {column for _row, column in pairs}
    return (
        pairs,
        [index for index in range(len(tracks)) if index not in matched_tracks],
        [index for index in range(len(detections)) if index not in matched_detections],
    )


def required_confirmation_hits(cfg: PipelineConfig) -> int:
    """Return the explicit temporal confirmation threshold.

    ``track_min_hits=1`` remains an intentional opt-out used by low-latency
    deployments and existing tests. Otherwise the dedicated temporal setting
    can only make confirmation stricter, never weaker than ``track_min_hits``.
    """
    if cfg.track_min_hits <= 1:
        return 1
    return max(
        int(cfg.track_min_hits),
        int(cfg.track_detection_temporal_confirmation_frames),
    )


def _xyxy_to_box_state(box: np.ndarray) -> np.ndarray:
    """Convert xyxy to the stable [center_x, center_y, width, height] state."""
    x1, y1, x2, y2 = box[:4]
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])


def _box_state_to_xyxy(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h = box
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class Track:
    track_id: int
    kf: ConstantVelocityKalman
    conf: float
    hits: int = 1
    misses: int = 0
    missed_seconds: float = 0.0
    confirmed: bool = False
    last_measurement: np.ndarray | None = None

    @property
    def xyxy(self) -> np.ndarray:
        """Current best box estimate — the prediction until `kf.update` runs."""
        return _box_state_to_xyxy(self.kf.position)


class ByteTracker:
    """Per-camera tracker: feed detections each tick, get back stable IDs."""

    def __init__(self, cfg: PipelineConfig):
        if cfg.track_detection_temporal_confirmation_frames < 1:
            raise ValueError(
                "track_detection_temporal_confirmation_frames must be >= 1"
            )
        if cfg.track_max_speed_mps <= 0:
            raise ValueError("track_max_speed_mps must be > 0")
        self.cfg = cfg
        self.tracks: list[Track] = []
        self._next_id = 1
        self._last_timestamp_s: float | None = None
        self.metrics: defaultdict[str, int] = defaultdict(int)
        # Cleared every update: diagnostics stay O(decisions in one tick) and
        # are streamed by demo.py through the shared bounded JSONL writer.
        self.last_events: list[dict] = []
        self._diagnostics_enabled = cfg.identity_diagnostics_path is not None

    def _record_event(self, event: str, track: Track, **fields) -> None:
        if not self._diagnostics_enabled:
            return
        self.last_events.append({
            "event": event,
            "local_track_id": track.track_id,
            "state": (
                "Lost" if track.confirmed and track.misses > 0
                else "Confirmed" if track.confirmed
                else "Tentative"
            ),
            "hits": track.hits,
            "misses": track.misses,
            "missed_seconds": track.missed_seconds,
            "bbox_xyxy": track.xyxy.copy(),
            "confidence": track.conf,
            **fields,
        })

    def update(self, detections: np.ndarray, timestamp_s: float | None = None) -> list[Track]:
        """Advance one tick with (N, 5) [x1, y1, x2, y2, conf] detections.

        Returns the confirmed tracks (stable IDs, Kalman-smoothed boxes).
        """
        cfg = self.cfg
        self.last_events.clear()
        detections = np.asarray(detections, dtype=float).reshape((-1, 5))
        if timestamp_s is not None and not np.isfinite(timestamp_s):
            raise ValueError("ByteTracker timestamp must be finite")
        if timestamp_s is None:
            dt = 1.0
        elif self._last_timestamp_s is None:
            dt = 0.0
        else:
            dt = float(timestamp_s) - self._last_timestamp_s
            if dt < -1e-9:
                raise ValueError(
                    f"ByteTracker timestamps must be monotonic: "
                    f"{timestamp_s} < {self._last_timestamp_s}"
                )
            dt = max(0.0, dt)
        if timestamp_s is not None:
            self._last_timestamp_s = float(timestamp_s)
            # A camera may be absent for many ticks. Expire before matching,
            # otherwise its first returning detection resurrects an old ID.
            retained = []
            for track in self.tracks:
                if track.missed_seconds + dt <= cfg.track_max_age_s:
                    retained.append(track)
                else:
                    self.metrics["tracks_removed_timeout"] += 1
                    self._record_event(
                        "local_removed",
                        track,
                        reason="capture_time_timeout_before_association",
                        elapsed_since_last_match_s=track.missed_seconds + dt,
                    )
            self.tracks = retained

        # 1. Predict where every known track should be this tick. Until a
        #    track is matched and updated, `track.xyxy` is this prediction.
        for track in self.tracks:
            timebase_scale = 1.0
            if timestamp_s is not None:
                timebase_scale = (
                    cfg.track_kf_time_process_noise
                    / max(cfg.track_kf_process_noise, 1e-9)
                )
            track.kf.predict(
                dt,
                process_noise_scale=(
                    timebase_scale * self._box_noise_scale(track.kf.position[3])
                ),
            )

        high = detections[detections[:, 4] >= cfg.track_high_conf]
        low = detections[detections[:, 4] < cfg.track_high_conf]
        # 2. Stage one: high-confidence detections vs. all tracks.
        if cfg.track_motion_association_enabled:
            pairs, unmatched_tracks, unmatched_high = match_tracks_motion(
                self.tracks, high, cfg
            )
        else:
            track_boxes = (
                np.stack([t.xyxy for t in self.tracks])
                if self.tracks else np.empty((0, 4))
            )
            pairs, unmatched_tracks, unmatched_high = match_by_iou(
                track_boxes, high[:, :4], cfg.track_min_iou
        )
        for t_idx, d_idx in pairs:
            track = self.tracks[t_idx]
            was_lost = track.misses > 0
            predicted_bbox = track.xyxy.copy() if self._diagnostics_enabled else None
            was_confirmed = track.confirmed if self._diagnostics_enabled else False
            if was_lost:
                self.metrics["reactivations"] += 1
            self.metrics["high_confidence_matches"] += 1
            self._mark_matched(track, high[d_idx])
            if self._diagnostics_enabled:
                overlap = float(iou_matrix(
                    predicted_bbox[None, :], high[d_idx:d_idx + 1, :4]
                )[0, 0])
                self._record_event(
                    "local_match",
                    track,
                    stage="high_confidence",
                    association="motion" if cfg.track_motion_association_enabled else "iou",
                    iou=overlap,
                    association_cost=1.0 - overlap,
                    minimum_iou=cfg.track_min_iou,
                    predicted_bbox_xyxy=predicted_bbox,
                    detection_bbox_xyxy=high[d_idx, :4].copy(),
                    detection_confidence=float(high[d_idx, 4]),
                    reactivated=was_lost,
                    became_confirmed=not was_confirmed and track.confirmed,
                )

        # 3. Stage two (the "byte" trick): leftover tracks get a second
        #    chance against low-confidence detections — typically occluded
        #    people whose boxes the detector wasn't sure about. (These
        #    tracks were not updated in stage one, so their boxes are
        #    still the stage-one predictions.)
        leftover = [self.tracks[index] for index in unmatched_tracks]
        pairs2, still_unmatched, _ = match_low_confidence_motion(
            leftover, low, cfg
        )
        for local_t, d_idx in pairs2:
            track = self.tracks[unmatched_tracks[local_t]]
            was_lost = track.misses > 0
            predicted_bbox = track.xyxy.copy() if self._diagnostics_enabled else None
            was_confirmed = track.confirmed if self._diagnostics_enabled else False
            if was_lost:
                self.metrics["reactivations"] += 1
            self.metrics["low_confidence_matches"] += 1
            self._mark_matched(track, low[d_idx])
            if self._diagnostics_enabled:
                overlap = float(iou_matrix(
                    predicted_bbox[None, :], low[d_idx:d_idx + 1, :4]
                )[0, 0])
                self._record_event(
                    "local_match",
                    track,
                    stage="low_confidence",
                    association="motion_gated_low_confidence",
                    iou=overlap,
                    association_cost=1.0 - overlap,
                    minimum_iou=cfg.track_min_iou,
                    predicted_bbox_xyxy=predicted_bbox,
                    detection_bbox_xyxy=low[d_idx, :4].copy(),
                    detection_confidence=float(low[d_idx, 4]),
                    reactivated=was_lost,
                    became_confirmed=not was_confirmed and track.confirmed,
                )
        unmatched_tracks = [unmatched_tracks[i] for i in still_unmatched]

        # 4. Unmatched tracks age; tentative ones die immediately,
        #    confirmed ones survive up to max_age ticks (occlusion tolerance).
        for t_idx in unmatched_tracks:
            track = self.tracks[t_idx]
            track.misses += 1
            track.missed_seconds += dt
            self._record_event(
                "local_miss",
                track,
                reason="no_association",
            )
        retained = []
        for track in self.tracks:
            keep = (
                (
                    track.missed_seconds <= cfg.track_max_age_s
                    if timestamp_s is not None
                    else track.misses <= cfg.track_max_age
                )
                if track.confirmed
                else track.misses <= 0
            )
            if keep:
                retained.append(track)
                continue
            reason = "tentative_first_miss" if not track.confirmed else "timeout"
            self.metrics["tracks_removed_tentative" if not track.confirmed else "tracks_removed_timeout"] += 1
            self._record_event("local_removed", track, reason=reason)
        self.tracks = retained

        # 5. Every unmatched high-confidence detection births a new track.
        for d_idx in unmatched_high:
            det = high[d_idx]
            measurement = _xyxy_to_box_state(det)
            track = Track(
                track_id=self._next_id,
                kf=ConstantVelocityKalman(
                    measurement,
                    process_noise=cfg.track_kf_process_noise,
                    measurement_noise=cfg.track_kf_measurement_noise,
                ),
                conf=float(det[4]),
                confirmed=required_confirmation_hits(cfg) <= 1,
                last_measurement=measurement.copy(),
            )
            self.tracks.append(track)
            self._next_id += 1
            self.metrics["tracks_created"] += 1
            self._record_event(
                "local_birth",
                track,
                reason="unmatched_high_confidence_detection",
                detection_bbox_xyxy=det[:4].copy(),
                detection_confidence=float(det[4]),
                birth_threshold=cfg.track_high_conf,
            )

        self.metrics["peak_retained_tracks"] = max(
            self.metrics["peak_retained_tracks"], len(self.tracks)
        )
        return [t for t in self.tracks if t.confirmed and t.misses == 0]

    def _mark_matched(self, track: Track, det: np.ndarray) -> None:
        confidence_scale = 1.0 + self.cfg.track_kf_confidence_noise_gain * (
            1.0 - float(np.clip(det[4], 0.0, 1.0))
        )
        measurement_scale = self._box_noise_scale(det[3] - det[1]) * confidence_scale
        measurement = _xyxy_to_box_state(det)
        track.kf.update(
            measurement,
            measurement_noise_scale=measurement_scale,
            nis_threshold=self.cfg.track_kf_robust_nis_threshold,
        )
        track.conf = float(det[4])
        track.hits += 1
        track.misses = 0
        track.missed_seconds = 0.0
        track.last_measurement = measurement.copy()
        if track.hits >= required_confirmation_hits(self.cfg):
            track.confirmed = True

    def _box_noise_scale(self, height_px: float) -> float:
        if not self.cfg.track_kf_adaptive_noise:
            return 1.0
        reference = max(self.cfg.track_kf_reference_height_px, 1e-6)
        scale = (max(float(height_px), 1.0) / reference) ** 2
        return float(np.clip(
            scale,
            self.cfg.track_kf_noise_scale_min,
            self.cfg.track_kf_noise_scale_max,
        ))
