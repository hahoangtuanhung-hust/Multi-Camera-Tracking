"""Observation-Centric SORT for per-camera, appearance-free tracking.

This implementation keeps the project's timestamp-aware Kalman model and
public track interface, while replacing ByteTrack's primary association with
the three motion-only mechanisms from OC-SORT:

* observation-centric momentum (OCM) during primary association;
* observation-centric recovery (OCR) using the last real bounding box; and
* observation-centric re-update (ORU) with virtual observations after a gap.

No appearance embedding, ReID model, or face identity is used.  The optional
low-confidence second pass is the BYTE extension supported by OC-SORT; it can
be disabled with ``ocsort_use_byte`` without changing the tracker backend.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .bytetrack import (
    ConstantVelocityKalman,
    _box_state_to_xyxy,
    _xyxy_to_box_state,
    iou_matrix,
    match_by_iou,
)
from .config import PipelineConfig


def _box_center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])


def _unit_direction(previous: np.ndarray, current: np.ndarray) -> np.ndarray | None:
    displacement = _box_center(current) - _box_center(previous)
    norm = float(np.linalg.norm(displacement))
    if not np.isfinite(norm) or norm <= 1e-9:
        return None
    return displacement / norm


@dataclass
class OCTrack:
    """One OC-SORT trajectory with bounded observation history."""

    track_id: int
    kf: ConstantVelocityKalman
    conf: float
    hits: int = 1
    misses: int = 0
    missed_seconds: float = 0.0
    confirmed: bool = False
    age: int = 0
    velocity: np.ndarray | None = None
    last_observation: np.ndarray | None = None
    last_observation_ts: float | None = None
    observations: deque[tuple[int, np.ndarray]] = field(default_factory=deque)
    last_observed_state: np.ndarray | None = None
    last_observed_covariance: np.ndarray | None = None

    @property
    def predicted_xyxy(self) -> np.ndarray:
        """Kalman prediction used for association at the current tick."""
        state = self.kf.position.copy()
        state[2:] = np.maximum(state[2:], 1e-3)
        return _box_state_to_xyxy(state)

    @property
    def xyxy(self) -> np.ndarray:
        """Observed box for output; prediction only while the track is lost.

        OC-SORT intentionally publishes the latest real observation. Using a
        re-updated Kalman box here would add spatial lag to the foot point and
        can turn a correct identity into a world-plane false negative.
        """
        if self.misses == 0 and self.last_observation is not None:
            return self.last_observation[:4].copy()
        return self.predicted_xyxy


def _previous_observation(track: OCTrack, delta_t: int) -> np.ndarray | None:
    """Return the observation at least ``delta_t`` ages back, or the oldest."""
    if not track.observations:
        return None
    target_age = track.age - max(1, int(delta_t))
    for age, observation in reversed(track.observations):
        if age <= target_age:
            return observation
    return track.observations[0][1]


def associate_observation_centric(
    tracks: list[OCTrack],
    detections: np.ndarray,
    min_iou: float,
    inertia: float,
    delta_t: int,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian association using IoU plus observed direction consistency."""
    if not tracks or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))

    detections = np.asarray(detections, dtype=float).reshape((-1, 5))
    predicted = np.stack([track.predicted_xyxy for track in tracks])
    overlap = iou_matrix(predicted, detections[:, :4])
    score = overlap.copy()

    for row, track in enumerate(tracks):
        previous = _previous_observation(track, delta_t)
        if previous is None or track.velocity is None:
            continue
        candidate_centers = np.column_stack((
            (detections[:, 0] + detections[:, 2]) * 0.5,
            (detections[:, 1] + detections[:, 3]) * 0.5,
        ))
        displacement = candidate_centers - _box_center(previous)
        norms = np.linalg.norm(displacement, axis=1)
        valid = norms > 1e-9
        directions = np.zeros_like(displacement)
        directions[valid] = displacement[valid] / norms[valid, None]
        cosine = np.clip(directions @ track.velocity, -1.0, 1.0)
        # Same normalized angular preference as OC-SORT: +0.5 when aligned,
        # zero when perpendicular, and -0.5 for the opposite direction.
        angular = (np.pi / 2.0 - np.abs(np.arccos(cosine))) / np.pi
        angular[~valid] = 0.0
        score[row] += float(inertia) * angular * detections[:, 4]

    rows, columns = linear_sum_assignment(-score)
    pairs = [
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if overlap[row, column] >= min_iou
    ]
    matched_tracks = {row for row, _ in pairs}
    matched_detections = {column for _, column in pairs}
    return (
        pairs,
        [index for index in range(len(tracks)) if index not in matched_tracks],
        [index for index in range(len(detections)) if index not in matched_detections],
    )


class OCSortTracker:
    """Timestamp-aware OC-SORT backend compatible with ``run_pipeline``."""

    def __init__(self, cfg: PipelineConfig):
        if cfg.ocsort_delta_t < 1:
            raise ValueError("ocsort_delta_t must be >= 1")
        if cfg.ocsort_inertia < 0:
            raise ValueError("ocsort_inertia must be >= 0")
        if cfg.ocsort_oru_min_misses < 1:
            raise ValueError("ocsort_oru_min_misses must be >= 1")
        self.cfg = cfg
        self.tracks: list[OCTrack] = []
        self._next_id = 1
        self._last_timestamp_s: float | None = None
        self.metrics: defaultdict[str, int] = defaultdict(int)
        self.last_events: list[dict] = []
        self._diagnostics_enabled = cfg.identity_diagnostics_path is not None

    def _record_event(self, event: str, track: OCTrack, **fields) -> None:
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
            "tracker": "ocsort",
            **fields,
        })

    def update(
        self, detections: np.ndarray, timestamp_s: float | None = None
    ) -> list[OCTrack]:
        """Advance one camera tick using ``[x1, y1, x2, y2, confidence]``."""
        cfg = self.cfg
        self.last_events.clear()
        detections = np.asarray(detections, dtype=float).reshape((-1, 5))
        if len(detections):
            valid = (
                np.all(np.isfinite(detections), axis=1)
                & (detections[:, 2] > detections[:, 0])
                & (detections[:, 3] > detections[:, 1])
            )
            detections = detections[valid]

        if timestamp_s is not None and not np.isfinite(timestamp_s):
            raise ValueError("OCSortTracker timestamp must be finite")
        if timestamp_s is None:
            dt = 1.0
        elif self._last_timestamp_s is None:
            dt = 0.0
        else:
            dt = float(timestamp_s) - self._last_timestamp_s
            if dt < -1e-9:
                raise ValueError(
                    "OCSortTracker timestamps must be monotonic: "
                    f"{timestamp_s} < {self._last_timestamp_s}"
                )
            dt = max(0.0, dt)
        if timestamp_s is not None:
            self._last_timestamp_s = float(timestamp_s)
            self._expire_before_association(dt)

        for track in self.tracks:
            # Do not let an uncertain scale velocity invert the box.
            for position_index, velocity_index in ((2, 6), (3, 7)):
                if track.kf.x[position_index] + dt * track.kf.x[velocity_index] <= 1e-3:
                    track.kf.x[velocity_index] = 0.0
            track.kf.predict(
                dt,
                process_noise_scale=self._process_noise_scale(track.kf.position[3], timestamp_s),
            )
            track.kf.x[2:4] = np.maximum(track.kf.x[2:4], 1e-3)
            track.age += 1

        high = detections[detections[:, 4] >= cfg.track_high_conf]
        low = detections[detections[:, 4] < cfg.track_high_conf]

        pairs, unmatched_tracks, unmatched_high = associate_observation_centric(
            self.tracks,
            high,
            cfg.track_min_iou,
            cfg.ocsort_inertia,
            cfg.ocsort_delta_t,
        )
        for track_index, detection_index in pairs:
            self.metrics["high_confidence_matches"] += 1
            self.metrics["ocm_matches"] += 1
            self._apply_match(
                self.tracks[track_index],
                high[detection_index],
                dt,
                timestamp_s,
                stage="high_confidence",
                association="ocsort_observation_momentum",
            )

        # OC-SORT officially supports this optional BYTE extension. It keeps a
        # partially occluded person alive without allowing low-score births.
        if cfg.ocsort_use_byte and unmatched_tracks and len(low):
            leftover = [self.tracks[index] for index in unmatched_tracks]
            boxes = np.stack([track.predicted_xyxy for track in leftover])
            pairs_low, still_unmatched, _ = match_by_iou(
                boxes, low[:, :4], cfg.track_min_iou
            )
            for local_track_index, detection_index in pairs_low:
                track = self.tracks[unmatched_tracks[local_track_index]]
                self.metrics["low_confidence_matches"] += 1
                self.metrics["byte_extension_matches"] += 1
                self._apply_match(
                    track,
                    low[detection_index],
                    dt,
                    timestamp_s,
                    stage="low_confidence",
                    association="ocsort_byte_extension",
                )
            unmatched_tracks = [
                unmatched_tracks[index] for index in still_unmatched
            ]

        # OCR compares remaining detections with the last *real* observation.
        # It is deliberately after predicted-state association so it repairs a
        # drifting Kalman trajectory without overriding a clear primary match.
        if (
            cfg.ocsort_observation_recovery_enabled
            and unmatched_tracks
            and unmatched_high
        ):
            recovery_tracks = [self.tracks[index] for index in unmatched_tracks]
            valid_local_indices = [
                index for index, track in enumerate(recovery_tracks)
                if track.last_observation is not None
            ]
            if valid_local_indices:
                last_boxes = np.stack([
                    recovery_tracks[index].last_observation[:4]
                    for index in valid_local_indices
                ])
                remaining_high = high[unmatched_high]
                recovery_pairs, still_recovery_tracks, still_high = match_by_iou(
                    last_boxes, remaining_high[:, :4], cfg.track_min_iou
                )
                for recovery_index, high_index in recovery_pairs:
                    local_index = valid_local_indices[recovery_index]
                    track = recovery_tracks[local_index]
                    detection_index = unmatched_high[high_index]
                    self.metrics["high_confidence_matches"] += 1
                    self.metrics["ocr_matches"] += 1
                    self._apply_match(
                        track,
                        high[detection_index],
                        dt,
                        timestamp_s,
                        stage="high_confidence",
                        association="ocsort_last_observation_recovery",
                    )
                matched_local = {
                    valid_local_indices[recovery_index]
                    for recovery_index, _ in recovery_pairs
                }
                unmatched_tracks = [
                    track_index
                    for local_index, track_index in enumerate(unmatched_tracks)
                    if local_index not in matched_local
                ]
                unmatched_high = [unmatched_high[index] for index in still_high]

        for track_index in unmatched_tracks:
            track = self.tracks[track_index]
            track.misses += 1
            track.missed_seconds += dt
            self._record_event("local_miss", track, reason="no_association")

        retained: list[OCTrack] = []
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
            tentative = not track.confirmed
            metric = "tracks_removed_tentative" if tentative else "tracks_removed_timeout"
            self.metrics[metric] += 1
            self._record_event(
                "local_removed",
                track,
                reason="tentative_first_miss" if tentative else "timeout",
            )
        self.tracks = retained

        for detection_index in unmatched_high:
            self._create_track(high[detection_index], timestamp_s)

        self.metrics["peak_retained_tracks"] = max(
            self.metrics["peak_retained_tracks"], len(self.tracks)
        )
        return [track for track in self.tracks if track.confirmed and track.misses == 0]

    def _expire_before_association(self, dt: float) -> None:
        retained: list[OCTrack] = []
        for track in self.tracks:
            elapsed = track.missed_seconds + dt
            if elapsed <= self.cfg.track_max_age_s:
                retained.append(track)
                continue
            self.metrics["tracks_removed_timeout"] += 1
            self._record_event(
                "local_removed",
                track,
                reason="capture_time_timeout_before_association",
                elapsed_since_last_match_s=elapsed,
            )
        self.tracks = retained

    def _apply_match(
        self,
        track: OCTrack,
        detection: np.ndarray,
        dt: float,
        timestamp_s: float | None,
        *,
        stage: str,
        association: str,
    ) -> None:
        was_lost = track.misses > 0
        was_confirmed = track.confirmed
        predicted_bbox = track.predicted_xyxy.copy()
        overlap = float(iou_matrix(
            predicted_bbox[None, :], detection[None, :4]
        )[0, 0])

        previous = _previous_observation(track, self.cfg.ocsort_delta_t)
        if previous is not None:
            direction = _unit_direction(previous, detection)
            if direction is not None:
                track.velocity = direction

        if (
            was_lost
            and self.cfg.ocsort_oru_enabled
            and track.misses >= self.cfg.ocsort_oru_min_misses
            and track.last_observed_state is not None
            and track.last_observed_covariance is not None
            and track.last_observation is not None
        ):
            self._observation_centric_reupdate(track, detection, timestamp_s)
            self.metrics["oru_reupdates"] += 1
        else:
            self._kalman_update(track, detection)

        if was_lost:
            self.metrics["reactivations"] += 1
        track.conf = float(detection[4])
        track.hits += 1
        track.misses = 0
        track.missed_seconds = 0.0
        if track.hits >= self.cfg.track_min_hits:
            track.confirmed = True
        track.last_observation = detection.copy()
        track.last_observation_ts = (
            float(timestamp_s) if timestamp_s is not None else None
        )
        track.observations.append((track.age, detection.copy()))
        while len(track.observations) > self.cfg.ocsort_delta_t + 2:
            track.observations.popleft()
        track.last_observed_state = track.kf.x.copy()
        track.last_observed_covariance = track.kf.P.copy()
        self._record_event(
            "local_match",
            track,
            stage=stage,
            association=association,
            iou=overlap,
            association_cost=1.0 - overlap,
            minimum_iou=self.cfg.track_min_iou,
            predicted_bbox_xyxy=predicted_bbox,
            detection_bbox_xyxy=detection[:4].copy(),
            detection_confidence=float(detection[4]),
            reactivated=was_lost,
            became_confirmed=not was_confirmed and track.confirmed,
        )

    def _observation_centric_reupdate(
        self,
        track: OCTrack,
        detection: np.ndarray,
        timestamp_s: float | None,
    ) -> None:
        """Replay a virtual observed trajectory over an occlusion (ORU)."""
        track.kf.x = track.last_observed_state.copy()
        track.kf.P = track.last_observed_covariance.copy()
        start = _xyxy_to_box_state(track.last_observation)
        end = _xyxy_to_box_state(detection)
        steps = max(1, track.misses + 1)
        if timestamp_s is not None and track.last_observation_ts is not None:
            total_dt = max(0.0, float(timestamp_s) - track.last_observation_ts)
        else:
            total_dt = float(steps)
        step_dt = total_dt / steps
        for step in range(1, steps + 1):
            alpha = step / steps
            virtual = start + alpha * (end - start)
            track.kf.predict(
                step_dt,
                process_noise_scale=self._process_noise_scale(
                    virtual[3], timestamp_s
                ),
            )
            if step == steps:
                self._kalman_update(track, detection)
            else:
                track.kf.update(
                    virtual,
                    measurement_noise_scale=self._box_noise_scale(virtual[3]),
                    nis_threshold=self.cfg.track_kf_robust_nis_threshold,
                )

    def _kalman_update(self, track: OCTrack, detection: np.ndarray) -> None:
        measurement = _xyxy_to_box_state(detection)
        confidence_scale = 1.0 + self.cfg.track_kf_confidence_noise_gain * (
            1.0 - float(np.clip(detection[4], 0.0, 1.0))
        )
        track.kf.update(
            measurement,
            measurement_noise_scale=(
                self._box_noise_scale(measurement[3]) * confidence_scale
            ),
            nis_threshold=self.cfg.track_kf_robust_nis_threshold,
        )

    def _create_track(
        self, detection: np.ndarray, timestamp_s: float | None
    ) -> None:
        measurement = _xyxy_to_box_state(detection)
        kf = ConstantVelocityKalman(
            measurement,
            process_noise=self.cfg.track_kf_process_noise,
            measurement_noise=self.cfg.track_kf_measurement_noise,
        )
        track = OCTrack(
            track_id=self._next_id,
            kf=kf,
            conf=float(detection[4]),
            confirmed=self.cfg.track_min_hits <= 1,
            last_observation=detection.copy(),
            last_observation_ts=(
                float(timestamp_s) if timestamp_s is not None else None
            ),
            observations=deque([(0, detection.copy())]),
            last_observed_state=kf.x.copy(),
            last_observed_covariance=kf.P.copy(),
        )
        self._next_id += 1
        self.tracks.append(track)
        self.metrics["tracks_created"] += 1
        self._record_event(
            "local_birth",
            track,
            birth_threshold=self.cfg.track_high_conf,
            detection_bbox_xyxy=detection[:4].copy(),
            detection_confidence=float(detection[4]),
        )

    def _process_noise_scale(
        self, height: float, timestamp_s: float | None
    ) -> float:
        scale = self._box_noise_scale(height)
        if timestamp_s is not None:
            scale *= self.cfg.track_kf_time_process_noise / max(
                self.cfg.track_kf_process_noise, 1e-9
            )
        return scale

    def _box_noise_scale(self, height: float) -> float:
        if not self.cfg.track_kf_adaptive_noise:
            return 1.0
        reference = max(self.cfg.track_kf_reference_height_px, 1e-6)
        ratio = max(float(height), 1.0) / reference
        return float(np.clip(
            ratio * ratio,
            self.cfg.track_kf_noise_scale_min,
            self.cfg.track_kf_noise_scale_max,
        ))
