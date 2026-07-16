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

        self.F = np.eye(2 * m)                # transition: z += ż * dt (dt = 1 tick)
        self.F[:m, m:] = np.eye(m)
        self.H = np.hstack([np.eye(m), np.zeros((m, m))])  # we only measure z

        # Discrete white-noise-acceleration model: unknown accelerations are
        # the process noise. For time step dt the standard result is
        #   Q = q * [[dt⁴/4, dt³/2],
        #            [dt³/2, dt²  ]]   (per measured dimension)
        # and with dt = 1 tick those factors become 0.25, 0.5 and 1.
        q = process_noise
        self.Q = np.block([
            [np.eye(m) * q * 0.25, np.eye(m) * q * 0.5],
            [np.eye(m) * q * 0.5,  np.eye(m) * q],
        ])
        self.R = np.eye(m) * measurement_noise

        self.P = np.eye(2 * m) * measurement_noise
        self.P[m:, m:] *= 100.0  # initial velocity is unknown

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[: self.m]

    def update(self, z: np.ndarray) -> None:
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(2 * self.m) - K @ self.H) @ self.P

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
    rows, cols = linear_sum_assignment(-iou)  # maximize total IoU
    pairs = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= min_iou]
    matched_t = {r for r, _ in pairs}
    matched_d = {c for _, c in pairs}
    return (
        pairs,
        [t for t in range(len(tracks_xyxy)) if t not in matched_t],
        [d for d in range(len(dets_xyxy)) if d not in matched_d],
    )


def _xyxy_to_cxcywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box[:4]
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])


def _cxcywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h = box
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class Track:
    track_id: int
    kf: ConstantVelocityKalman
    conf: float
    hits: int = 1
    misses: int = 0
    confirmed: bool = False

    @property
    def xyxy(self) -> np.ndarray:
        """Current best box estimate — the prediction until `kf.update` runs."""
        return _cxcywh_to_xyxy(self.kf.position)


class ByteTracker:
    """Per-camera tracker: feed detections each tick, get back stable IDs."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: np.ndarray) -> list[Track]:
        """Advance one tick with (N, 5) [x1, y1, x2, y2, conf] detections.

        Returns the confirmed tracks (stable IDs, Kalman-smoothed boxes).
        """
        cfg = self.cfg

        # 1. Predict where every known track should be this tick. Until a
        #    track is matched and updated, `track.xyxy` is this prediction.
        for track in self.tracks:
            track.kf.predict()

        high = detections[detections[:, 4] >= cfg.track_high_conf]
        low = detections[detections[:, 4] < cfg.track_high_conf]
        track_boxes = (
            np.stack([t.xyxy for t in self.tracks])
            if self.tracks else np.empty((0, 4))
        )

        # 2. Stage one: high-confidence detections vs. all tracks.
        pairs, unmatched_tracks, unmatched_high = match_by_iou(
            track_boxes, high[:, :4], cfg.track_min_iou
        )
        for t_idx, d_idx in pairs:
            self._mark_matched(self.tracks[t_idx], high[d_idx])

        # 3. Stage two (the "byte" trick): leftover tracks get a second
        #    chance against low-confidence detections — typically occluded
        #    people whose boxes the detector wasn't sure about. (These
        #    tracks were not updated in stage one, so their boxes are
        #    still the stage-one predictions.)
        leftover_boxes = (
            np.stack([self.tracks[t].xyxy for t in unmatched_tracks])
            if unmatched_tracks else np.empty((0, 4))
        )
        pairs2, still_unmatched, _ = match_by_iou(leftover_boxes, low[:, :4], cfg.track_min_iou)
        for local_t, d_idx in pairs2:
            self._mark_matched(self.tracks[unmatched_tracks[local_t]], low[d_idx])
        unmatched_tracks = [unmatched_tracks[i] for i in still_unmatched]

        # 4. Unmatched tracks age; tentative ones die immediately,
        #    confirmed ones survive up to max_age ticks (occlusion tolerance).
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].misses += 1
        self.tracks = [
            t for t in self.tracks
            if t.misses <= (cfg.track_max_age if t.confirmed else 0)
        ]

        # 5. Every unmatched high-confidence detection births a new track.
        for d_idx in unmatched_high:
            det = high[d_idx]
            self.tracks.append(Track(
                track_id=self._next_id,
                kf=ConstantVelocityKalman(
                    _xyxy_to_cxcywh(det),
                    process_noise=cfg.track_kf_process_noise,
                    measurement_noise=cfg.track_kf_measurement_noise,
                ),
                conf=float(det[4]),
            ))
            self._next_id += 1

        return [t for t in self.tracks if t.confirmed and t.misses == 0]

    def _mark_matched(self, track: Track, det: np.ndarray) -> None:
        track.kf.update(_xyxy_to_cxcywh(det))
        track.conf = float(det[4])
        track.hits += 1
        track.misses = 0
        if track.hits >= self.cfg.track_min_hits:
            track.confirmed = True
