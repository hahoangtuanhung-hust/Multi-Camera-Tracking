"""Hybrid-SORT (no ReID) per-camera tracker.

Implements the appearance-free variant of Hybrid-SORT (Zhao et al., CVPR 2023,
"Hybrid-SORT: Weak Cues Matter for Online Multi-Object Tracking").

The key idea over plain OC-SORT is a **two-round** primary association:

Round 1 — High-confidence detections vs. Kalman predictions (standard IoU).
    Unmatched tracks that still have a recent real observation proceed to…

Round 2 — Same unmatched tracks vs. same unmatched detections using a
    **blended** cost that linearly combines:
      * IoU(Kalman_prediction, detection)  — motion continuity
      * IoU(last_real_observation, detection) — temporal proximity / re-entry

    The mix weight ``hybridsort_alpha`` controls this blend:
      alpha=0 → pure Kalman IoU (degenerates to vanilla SORT)
      alpha=1 → pure last-observation IoU
      default: 0.40 — empirically good for occlusion recovery without
                       false merges at crossings

After the two-round primary stage the optional BYTE and OCR rescue passes from
OC-SORT are preserved, so Hybrid-SORT is a strict superset of OC-SORT.

No appearance embedding, ReID model, or face identity is used.
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
from .ocsort import _box_center, _previous_observation, _unit_direction


# ---------------------------------------------------------------------------
# Track dataclass
# ---------------------------------------------------------------------------

@dataclass
class HybridTrack:
    """One Hybrid-SORT trajectory with bounded observation history."""

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

        Hybrid-SORT publishes the last real observation just like OC-SORT,
        to avoid foot-point spatial lag from Kalman smoothing.
        """
        if self.misses == 0 and self.last_observation is not None:
            return self.last_observation[:4].copy()
        return self.predicted_xyxy


# ---------------------------------------------------------------------------
# Two-round Hybrid-SORT association
# ---------------------------------------------------------------------------

def associate_hybrid(
    tracks: list[HybridTrack],
    detections: np.ndarray,
    min_iou: float,
    alpha: float,
    inertia: float,
    delta_t: int,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Two-round primary association combining Kalman-IoU and observation-IoU.

    Round 1: standard IoU against predicted boxes (all tracks, high dets).
    Round 2: blended cost for still-unmatched tracks that have a real last
             observation available.

    Returns (pairs, unmatched_track_indices, unmatched_detection_indices).
    """
    if not tracks or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))

    detections = np.asarray(detections, dtype=float).reshape((-1, 5))
    predicted = np.stack([track.predicted_xyxy for track in tracks])

    # ---- Round 1: Kalman-IoU + OC-SORT direction inertia (unchanged) ------
    iou_pred = iou_matrix(predicted, detections[:, :4])
    score_r1 = iou_pred.copy()
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
        angular = (np.pi / 2.0 - np.abs(np.arccos(cosine))) / np.pi
        angular[~valid] = 0.0
        score_r1[row] += float(inertia) * angular * detections[:, 4]

    rows1, cols1 = linear_sum_assignment(-score_r1)
    pairs_r1 = [
        (int(r), int(c))
        for r, c in zip(rows1, cols1)
        if iou_pred[r, c] >= min_iou
    ]
    matched_tracks_r1 = {r for r, _ in pairs_r1}
    matched_dets_r1 = {c for _, c in pairs_r1}

    unmatched_track_r1 = [i for i in range(len(tracks)) if i not in matched_tracks_r1]
    unmatched_det_r1 = [i for i in range(len(detections)) if i not in matched_dets_r1]

    if not unmatched_track_r1 or not unmatched_det_r1:
        return (
            pairs_r1,
            unmatched_track_r1,
            unmatched_det_r1,
        )

    # ---- Round 2: blended (Kalman-IoU + last-observation-IoU) -------------
    # Only tracks that have a real previous observation participate.
    eligible_local = [
        local_i
        for local_i, track_i in enumerate(unmatched_track_r1)
        if tracks[track_i].last_observation is not None
    ]
    if not eligible_local or not unmatched_det_r1:
        return pairs_r1, unmatched_track_r1, unmatched_det_r1

    eligible_track_indices = [unmatched_track_r1[li] for li in eligible_local]
    remaining_dets = detections[unmatched_det_r1]

    # IoU between Kalman prediction and remaining detections
    pred_sub = np.stack([tracks[i].predicted_xyxy for i in eligible_track_indices])
    iou_pred_sub = iou_matrix(pred_sub, remaining_dets[:, :4])

    # IoU between last real observation and remaining detections
    obs_sub = np.stack([tracks[i].last_observation[:4] for i in eligible_track_indices])
    iou_obs_sub = iou_matrix(obs_sub, remaining_dets[:, :4])

    # Blended IoU: (1-alpha)*kalman_iou + alpha*observation_iou
    blended = (1.0 - float(alpha)) * iou_pred_sub + float(alpha) * iou_obs_sub
    # Gate: at least one of the two IoU components must be >= min_iou
    gate_mask = (iou_pred_sub >= min_iou) | (iou_obs_sub >= min_iou)
    blended[~gate_mask] = 0.0

    rows2, cols2 = linear_sum_assignment(-blended)
    pairs_r2_local = [
        (int(r), int(c))
        for r, c in zip(rows2, cols2)
        if gate_mask[r, c]
    ]

    # Re-map local → global indices
    pairs_r2 = [
        (eligible_track_indices[r], unmatched_det_r1[c])
        for r, c in pairs_r2_local
    ]
    matched_tracks_r2 = {r for r, _ in pairs_r2}
    matched_dets_r2 = {c for _, c in pairs_r2}

    final_pairs = pairs_r1 + pairs_r2
    final_unmatched_tracks = [
        i for i in unmatched_track_r1 if i not in matched_tracks_r2
    ]
    final_unmatched_dets = [
        i for i in unmatched_det_r1 if i not in matched_dets_r2
    ]
    return final_pairs, final_unmatched_tracks, final_unmatched_dets


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class HybridSortTracker:
    """Timestamp-aware Hybrid-SORT (no ReID) backend compatible with run_pipeline."""

    def __init__(self, cfg: PipelineConfig) -> None:
        if cfg.ocsort_delta_t < 1:
            raise ValueError("ocsort_delta_t must be >= 1")
        if cfg.ocsort_inertia < 0:
            raise ValueError("ocsort_inertia must be >= 0")
        if not (0.0 <= cfg.hybridsort_alpha <= 1.0):
            raise ValueError("hybridsort_alpha must be in [0, 1]")
        if cfg.ocsort_oru_min_misses < 1:
            raise ValueError("ocsort_oru_min_misses must be >= 1")
        self.cfg = cfg
        self.tracks: list[HybridTrack] = []
        self._next_id = 1
        self._last_timestamp_s: float | None = None
        self.metrics: defaultdict[str, int] = defaultdict(int)
        self.last_events: list[dict] = []
        self._diagnostics_enabled = cfg.identity_diagnostics_path is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, detections: np.ndarray, timestamp_s: float | None = None
    ) -> list[HybridTrack]:
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
            raise ValueError("HybridSortTracker timestamp must be finite")
        if timestamp_s is None:
            dt = 1.0
        elif self._last_timestamp_s is None:
            dt = 0.0
        else:
            dt = float(timestamp_s) - self._last_timestamp_s
            if dt < -1e-9:
                raise ValueError(
                    "HybridSortTracker timestamps must be monotonic: "
                    f"{timestamp_s} < {self._last_timestamp_s}"
                )
            dt = max(0.0, dt)
        if timestamp_s is not None:
            self._last_timestamp_s = float(timestamp_s)
            self._expire_before_association(dt)

        # Kalman predict all tracks
        for track in self.tracks:
            for pos_idx, vel_idx in ((2, 6), (3, 7)):
                if track.kf.x[pos_idx] + dt * track.kf.x[vel_idx] <= 1e-3:
                    track.kf.x[vel_idx] = 0.0
            track.kf.predict(
                dt,
                process_noise_scale=self._process_noise_scale(track.kf.position[3], timestamp_s),
            )
            track.kf.x[2:4] = np.maximum(track.kf.x[2:4], 1e-3)
            track.age += 1

        high = detections[detections[:, 4] >= cfg.track_high_conf]
        low = detections[detections[:, 4] < cfg.track_high_conf]

        # ---- Two-round primary association (the Hybrid-SORT addition) ------
        pairs, unmatched_tracks, unmatched_high = associate_hybrid(
            self.tracks,
            high,
            cfg.track_min_iou,
            cfg.hybridsort_alpha,
            cfg.ocsort_inertia,
            cfg.ocsort_delta_t,
        )
        for track_index, detection_index in pairs:
            self.metrics["high_confidence_matches"] += 1
            self.metrics["hybrid_primary_matches"] += 1
            self._apply_match(
                self.tracks[track_index],
                high[detection_index],
                dt,
                timestamp_s,
                stage="high_confidence",
                association="hybrid_sort_two_round",
            )

        # ---- Optional BYTE extension (low-confidence rescue) ---------------
        if cfg.ocsort_use_byte and unmatched_tracks and len(low):
            leftover = [self.tracks[i] for i in unmatched_tracks]
            boxes = np.stack([t.predicted_xyxy for t in leftover])
            pairs_low, still_unmatched, _ = match_by_iou(
                boxes, low[:, :4], cfg.track_min_iou
            )
            for local_i, det_i in pairs_low:
                track = self.tracks[unmatched_tracks[local_i]]
                self.metrics["low_confidence_matches"] += 1
                self.metrics["byte_extension_matches"] += 1
                self._apply_match(
                    track, low[det_i], dt, timestamp_s,
                    stage="low_confidence", association="hybrid_byte_extension",
                )
            unmatched_tracks = [unmatched_tracks[i] for i in still_unmatched]

        # ---- OCR: last-observation recovery for still-unmatched HighDets --
        if (
            cfg.ocsort_observation_recovery_enabled
            and unmatched_tracks
            and unmatched_high
        ):
            recovery_tracks = [self.tracks[i] for i in unmatched_tracks]
            valid_local = [
                li for li, t in enumerate(recovery_tracks)
                if t.last_observation is not None
            ]
            if valid_local:
                last_boxes = np.stack([
                    recovery_tracks[li].last_observation[:4]
                    for li in valid_local
                ])
                remaining_high = high[unmatched_high]
                rec_pairs, still_rec, still_high = match_by_iou(
                    last_boxes, remaining_high[:, :4], cfg.track_min_iou
                )
                for rec_i, high_i in rec_pairs:
                    li = valid_local[rec_i]
                    track = recovery_tracks[li]
                    det_i = unmatched_high[high_i]
                    self.metrics["high_confidence_matches"] += 1
                    self.metrics["ocr_matches"] += 1
                    self._apply_match(
                        track, high[det_i], dt, timestamp_s,
                        stage="high_confidence",
                        association="hybrid_last_observation_recovery",
                    )
                matched_local = {valid_local[rec_i] for rec_i, _ in rec_pairs}
                unmatched_tracks = [
                    ti for li, ti in enumerate(unmatched_tracks)
                    if li not in matched_local
                ]
                unmatched_high = [unmatched_high[i] for i in still_high]

        # ---- Age missed tracks ---------------------------------------------
        for track_index in unmatched_tracks:
            track = self.tracks[track_index]
            track.misses += 1
            track.missed_seconds += dt
            self._record_event("local_miss", track, reason="no_association")

        # ---- Remove timed-out / tentative tracks --------------------------
        retained: list[HybridTrack] = []
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
            self.metrics["tracks_removed_tentative" if tentative else "tracks_removed_timeout"] += 1
            self._record_event(
                "local_removed", track,
                reason="tentative_first_miss" if tentative else "timeout",
            )
        self.tracks = retained

        # ---- Birth new tracks from unmatched high-confidence detections ----
        for det_i in unmatched_high:
            self._create_track(high[det_i], timestamp_s)

        self.metrics["peak_retained_tracks"] = max(
            self.metrics["peak_retained_tracks"], len(self.tracks)
        )
        return [t for t in self.tracks if t.confirmed and t.misses == 0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expire_before_association(self, dt: float) -> None:
        retained: list[HybridTrack] = []
        for track in self.tracks:
            elapsed = track.missed_seconds + dt
            if elapsed <= self.cfg.track_max_age_s:
                retained.append(track)
                continue
            self.metrics["tracks_removed_timeout"] += 1
            self._record_event(
                "local_removed", track,
                reason="capture_time_timeout_before_association",
                elapsed_since_last_match_s=elapsed,
            )
        self.tracks = retained

    def _apply_match(
        self,
        track: HybridTrack,
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
        overlap = float(iou_matrix(predicted_bbox[None, :], detection[None, :4])[0, 0])

        # Update direction velocity from observation history (OC-SORT style)
        previous = _previous_observation(track, self.cfg.ocsort_delta_t)
        if previous is not None:
            direction = _unit_direction(previous, detection)
            if direction is not None:
                track.velocity = direction

        # ORU: virtual Kalman re-update over the occlusion gap
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
        track.last_observation_ts = float(timestamp_s) if timestamp_s is not None else None
        track.observations.append((track.age, detection.copy()))
        while len(track.observations) > self.cfg.ocsort_delta_t + 2:
            track.observations.popleft()
        track.last_observed_state = track.kf.x.copy()
        track.last_observed_covariance = track.kf.P.copy()
        self._record_event(
            "local_match", track,
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
        track: HybridTrack,
        detection: np.ndarray,
        timestamp_s: float | None,
    ) -> None:
        """Replay a virtual observed trajectory over an occlusion (ORU from OC-SORT)."""
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
                process_noise_scale=self._process_noise_scale(virtual[3], timestamp_s),
            )
            if step == steps:
                self._kalman_update(track, detection)
            else:
                track.kf.update(
                    virtual,
                    measurement_noise_scale=self._box_noise_scale(virtual[3]),
                    nis_threshold=self.cfg.track_kf_robust_nis_threshold,
                )

    def _kalman_update(self, track: HybridTrack, detection: np.ndarray) -> None:
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

    def _create_track(self, detection: np.ndarray, timestamp_s: float | None) -> None:
        measurement = _xyxy_to_box_state(detection)
        kf = ConstantVelocityKalman(
            measurement,
            process_noise=self.cfg.track_kf_process_noise,
            measurement_noise=self.cfg.track_kf_measurement_noise,
        )
        track = HybridTrack(
            track_id=self._next_id,
            kf=kf,
            conf=float(detection[4]),
            confirmed=self.cfg.track_min_hits <= 1,
            last_observation=detection.copy(),
            last_observation_ts=float(timestamp_s) if timestamp_s is not None else None,
            observations=deque([(0, detection.copy())]),
            last_observed_state=kf.x.copy(),
            last_observed_covariance=kf.P.copy(),
        )
        self._next_id += 1
        self.tracks.append(track)
        self.metrics["tracks_created"] += 1
        self._record_event(
            "local_birth", track,
            birth_threshold=self.cfg.track_high_conf,
            detection_bbox_xyxy=detection[:4].copy(),
            detection_confidence=float(detection[4]),
        )

    def _process_noise_scale(self, height: float, timestamp_s: float | None) -> float:
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

    def _record_event(self, event: str, track: HybridTrack, **fields) -> None:
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
            "tracker": "hybridsort",
            **fields,
        })
