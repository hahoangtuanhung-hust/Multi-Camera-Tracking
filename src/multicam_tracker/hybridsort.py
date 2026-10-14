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
    match_low_confidence_motion,
    match_by_iou,
    required_confirmation_hits,
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
    max_obs_gap_s: float = 0.80,
    min_blended_iou: float = 0.25,
    timestamp_s: float | None = None,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Two-round primary association combining Kalman-IoU and observation-IoU.

    Round 1: standard IoU against predicted boxes (all tracks, high dets).
    Round 2: blended cost for recently lost tracks with observation history.
             Includes scale/aspect-ratio gating, direction consistency,
             and temporal decay to prevent identity theft.

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
        return pairs_r1, unmatched_track_r1, unmatched_det_r1

    # ---- Round 2: guarded blended (Kalman-IoU + last-observation-IoU) -----
    # Only tracks with recent observations are eligible. Stale tracks (> max_obs_gap_s)
    # are excluded to prevent identity theft when another person crosses the area.
    eligible_local = []
    for local_i, track_i in enumerate(unmatched_track_r1):
        track = tracks[track_i]
        if track.last_observation is None:
            continue
        gap_s = (
            float(timestamp_s) - track.last_observation_ts
            if timestamp_s is not None and track.last_observation_ts is not None
            else track.missed_seconds
        )
        if gap_s <= max_obs_gap_s:
            eligible_local.append(local_i)

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

    num_tracks = len(eligible_track_indices)
    num_dets = len(remaining_dets)
    blended = np.zeros((num_tracks, num_dets), dtype=float)
    valid_mask = np.zeros((num_tracks, num_dets), dtype=bool)

    for r, track_i in enumerate(eligible_track_indices):
        track = tracks[track_i]
        gap_s = (
            float(timestamp_s) - track.last_observation_ts
            if timestamp_s is not None and track.last_observation_ts is not None
            else track.missed_seconds
        )
        # Piecewise temporal decay: for short occlusions (<= 0.4s) weight is full;
        # then decays smoothly to 0 at max_obs_gap_s.
        if gap_s <= 0.40:
            decay = 1.0
        else:
            decay = max(0.0, 1.0 - (gap_s - 0.40) / max(max_obs_gap_s - 0.40, 1e-6))
        effective_alpha = float(alpha) * decay

        obs_box = track.last_observation[:4]
        obs_w = max(1.0, obs_box[2] - obs_box[0])
        obs_h = max(1.0, obs_box[3] - obs_box[1])
        obs_center = _box_center(obs_box)

        for c in range(num_dets):
            det_box = remaining_dets[c, :4]
            det_w = max(1.0, det_box[2] - det_box[0])
            det_h = max(1.0, det_box[3] - det_box[1])

            # Scale / aspect-ratio consistency check: reject severe box size mismatch
            scale_w = det_w / obs_w
            scale_h = det_h / obs_h
            if scale_w < 0.45 or scale_w > 2.2 or scale_h < 0.45 or scale_h > 2.2:
                continue

            # Motion direction check: reject detections moving backwards from established velocity
            if track.velocity is not None:
                disp = _box_center(det_box) - obs_center
                norm = float(np.linalg.norm(disp))
                if norm > 15.0:
                    unit_disp = disp / norm
                    cos_dir = float(unit_disp @ track.velocity)
                    if cos_dir < -0.45:
                        continue

            pred_iou = iou_pred_sub[r, c]
            obs_iou = iou_obs_sub[r, c]
            score = (1.0 - effective_alpha) * pred_iou + effective_alpha * obs_iou

            # Robust gating: accept match if blended score meets threshold, or
            # strong observation IoU meets threshold with sufficient blended confidence
            is_valid = (
                (score >= min_blended_iou)
                or (obs_iou >= max(min_iou, 0.30) and score >= min_blended_iou * 0.50)
                or (pred_iou >= min_iou and obs_iou >= 0.15)
            )
            if is_valid:
                blended[r, c] = max(score, obs_iou * effective_alpha)
                valid_mask[r, c] = True

    if not np.any(valid_mask):
        return pairs_r1, unmatched_track_r1, unmatched_det_r1

    rows2, cols2 = linear_sum_assignment(-blended)
    pairs_r2_local = [
        (int(r), int(c))
        for r, c in zip(rows2, cols2)
        if valid_mask[r, c] and blended[r, c] > 0.0
    ]

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


def find_ambiguous_detections(
    tracks: list[HybridTrack],
    detections: np.ndarray,
    *,
    min_iou: float,
    margin: float,
    min_area_ratio: float = 1.0,
) -> dict[int, tuple[int, ...]]:
    """Return detections that cannot safely choose between active owners.

    A high-confidence box is ambiguous when it overlaps at least two confirmed,
    currently visible track predictions, the two best IoUs differ by no more
    than ``margin``, and its area is expanded relative to both owner boxes.
    This is intentionally conservative: normal same-size detections, recently
    lost tracks and tentative tracks cannot suppress a legitimate observation.

    The returned mapping is ``detection_index -> candidate_track_indices``.
    Callers can let those established tracks coast for one tick while also
    preventing a merged detection from creating a duplicate local identity.
    """
    detections = np.asarray(detections, dtype=float).reshape((-1, 5))
    eligible = [
        index
        for index, track in enumerate(tracks)
        if track.confirmed and track.misses == 0
    ]
    if len(eligible) < 2 or len(detections) == 0:
        return {}

    predicted = np.stack([tracks[index].predicted_xyxy for index in eligible])
    overlap = iou_matrix(predicted, detections[:, :4])
    ambiguous: dict[int, tuple[int, ...]] = {}
    for detection_index in range(len(detections)):
        scores = overlap[:, detection_index]
        candidate_rows = np.flatnonzero(scores >= min_iou)
        if len(candidate_rows) < 2:
            continue
        ordered_rows = candidate_rows[np.argsort(scores[candidate_rows])[::-1]]
        if float(scores[ordered_rows[0]] - scores[ordered_rows[1]]) > margin:
            continue
        detection = detections[detection_index, :4]
        detection_area = max(0.0, detection[2] - detection[0]) * max(
            0.0, detection[3] - detection[1]
        )
        owner_areas = []
        for row in ordered_rows[:2]:
            owner = predicted[int(row)]
            owner_areas.append(
                max(0.0, owner[2] - owner[0]) * max(0.0, owner[3] - owner[1])
            )
        if detection_area < min_area_ratio * max(owner_areas):
            continue
        ambiguous[detection_index] = tuple(
            eligible[int(row)] for row in ordered_rows
        )
    return ambiguous


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
        if cfg.hybridsort_max_obs_gap_s <= 0:
            raise ValueError("hybridsort_max_obs_gap_s must be > 0")
        if not (0.0 <= cfg.hybridsort_min_blended_iou <= 1.0):
            raise ValueError("hybridsort_min_blended_iou must be in [0, 1]")
        if not (0.0 <= cfg.hybridsort_ambiguity_min_iou <= 1.0):
            raise ValueError("hybridsort_ambiguity_min_iou must be in [0, 1]")
        if cfg.hybridsort_ambiguity_margin < 0.0:
            raise ValueError("hybridsort_ambiguity_margin must be >= 0")
        if cfg.hybridsort_ambiguity_min_area_ratio < 1.0:
            raise ValueError("hybridsort_ambiguity_min_area_ratio must be >= 1")
        if cfg.ocsort_oru_min_misses < 1:
            raise ValueError("ocsort_oru_min_misses must be >= 1")
        if cfg.track_detection_temporal_confirmation_frames < 1:
            raise ValueError(
                "track_detection_temporal_confirmation_frames must be >= 1"
            )
        if cfg.track_tentative_max_misses < 0:
            raise ValueError("track_tentative_max_misses must be >= 0")
        if cfg.track_max_speed_mps <= 0:
            raise ValueError("track_max_speed_mps must be > 0")
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

        ambiguous_high = (
            find_ambiguous_detections(
                self.tracks,
                high,
                min_iou=cfg.hybridsort_ambiguity_min_iou,
                margin=cfg.hybridsort_ambiguity_margin,
                min_area_ratio=cfg.hybridsort_ambiguity_min_area_ratio,
            )
            if cfg.hybridsort_ambiguity_suppression_enabled
            else {}
        )
        primary_detection_indices = np.asarray(
            [index for index in range(len(high)) if index not in ambiguous_high],
            dtype=int,
        )
        primary_high = high[primary_detection_indices]
        for detection_index, candidate_tracks in ambiguous_high.items():
            self.metrics["ambiguous_detections_suppressed"] += 1
            for track_index in candidate_tracks:
                self._record_event(
                    "local_ambiguous_detection_suppressed",
                    self.tracks[track_index],
                    reason="multiple_established_owners",
                    candidate_count=len(candidate_tracks),
                    detection_bbox_xyxy=high[detection_index, :4].copy(),
                    detection_confidence=float(high[detection_index, 4]),
                )

        # ---- Two-round primary association (the Hybrid-SORT addition) ------
        pairs, unmatched_tracks, unmatched_primary = associate_hybrid(
            self.tracks,
            primary_high,
            cfg.track_min_iou,
            cfg.hybridsort_alpha,
            cfg.ocsort_inertia,
            cfg.ocsort_delta_t,
            max_obs_gap_s=cfg.hybridsort_max_obs_gap_s,
            min_blended_iou=cfg.hybridsort_min_blended_iou,
            timestamp_s=timestamp_s,
        )
        pairs = [
            (track_index, int(primary_detection_indices[detection_index]))
            for track_index, detection_index in pairs
        ]
        unmatched_high = [
            int(primary_detection_indices[detection_index])
            for detection_index in unmatched_primary
        ]
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
            pairs_low, still_unmatched, _ = match_low_confidence_motion(
                leftover, low, cfg
            )
            for local_i, det_i in pairs_low:
                track = self.tracks[unmatched_tracks[local_i]]
                self.metrics["low_confidence_matches"] += 1
                self.metrics["byte_extension_matches"] += 1
                self._apply_match(
                    track, low[det_i], dt, timestamp_s,
                    stage="low_confidence",
                    association="hybrid_motion_gated_byte_extension",
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
                and (
                    (timestamp_s - t.last_observation_ts <= cfg.hybridsort_max_obs_gap_s)
                    if timestamp_s is not None and t.last_observation_ts is not None
                    else t.missed_seconds <= cfg.hybridsort_max_obs_gap_s
                )
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
                else track.misses <= cfg.track_tentative_max_misses
            )
            if keep:
                retained.append(track)
                continue
            tentative = not track.confirmed
            self.metrics["tracks_removed_tentative" if tentative else "tracks_removed_timeout"] += 1
            self._record_event(
                "local_removed", track,
                reason="tentative_miss_budget" if tentative else "timeout",
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
        if track.hits >= required_confirmation_hits(self.cfg):
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
            confirmed=required_confirmation_hits(self.cfg) <= 1,
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
