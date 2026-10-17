"""Small, dependency-free person appearance descriptors for recovery events.

This is deliberately not a face identifier or a neural ReID system.  It keeps
only a normalized, short-lived colour descriptor computed from the inner body
crop.  The descriptor is useful as a veto/tie-breaker when geometry becomes
degenerate in a stationary crowd, while the normal hot path remains unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np


def normalize_descriptor(descriptor: np.ndarray | None) -> np.ndarray | None:
    """Return a finite L2-normalized descriptor, or ``None`` if invalid."""
    if descriptor is None:
        return None
    value = np.asarray(descriptor, dtype=np.float32).reshape(-1)
    if value.size == 0 or not np.all(np.isfinite(value)):
        return None
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        return None
    return value / norm


def appearance_similarity(
    first: np.ndarray | None, second: np.ndarray | None
) -> float | None:
    """Cosine similarity for normalized appearance descriptors."""
    left = normalize_descriptor(first)
    right = normalize_descriptor(second)
    if left is None or right is None or left.shape != right.shape:
        return None
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def update_appearance_prototype(
    current: np.ndarray | None,
    observation: np.ndarray | None,
    *,
    alpha: float,
) -> np.ndarray | None:
    """Update a bounded EMA prototype without retaining image pixels."""
    observed = normalize_descriptor(observation)
    if observed is None:
        return normalize_descriptor(current)
    previous = normalize_descriptor(current)
    if previous is None or previous.shape != observed.shape:
        return observed.copy()
    mixed = (1.0 - float(alpha)) * previous + float(alpha) * observed
    return normalize_descriptor(mixed)


def extract_person_descriptor(
    frame_bgr: np.ndarray | None,
    bbox_xyxy: np.ndarray,
) -> np.ndarray | None:
    """Extract a compact three-stripe HSV histogram from one person crop.

    The box is inset horizontally to reduce neighbouring-person/background
    leakage.  Three vertical stripes retain coarse clothing layout while
    remaining cheap enough to compute only at birth/reappearance events.
    """
    if frame_bgr is None:
        return None
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] < 3 or frame.size == 0:
        return None
    height, width = frame.shape[:2]
    if height < 4 or width < 4:
        return None
    box = np.asarray(bbox_xyxy, dtype=float).reshape(-1)
    if box.size < 4 or not np.all(np.isfinite(box[:4])):
        return None
    x1, y1, x2, y2 = box[:4]
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    if box_width < 4.0 or box_height < 8.0:
        return None
    x1 += 0.15 * box_width
    x2 -= 0.15 * box_width
    y1 += 0.03 * box_height
    y2 -= 0.03 * box_height
    left = max(0, min(width, int(np.floor(x1))))
    right = max(0, min(width, int(np.ceil(x2))))
    top = max(0, min(height, int(np.floor(y1))))
    bottom = max(0, min(height, int(np.ceil(y2))))
    if right - left < 3 or bottom - top < 6:
        return None
    crop = frame[top:bottom, left:right, :3]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    stripe_edges = np.linspace(0, len(hsv), 4, dtype=int)
    parts: list[np.ndarray] = []
    for start, end in zip(stripe_edges[:-1], stripe_edges[1:]):
        stripe = hsv[start:end]
        if stripe.size == 0:
            return None
        # Very dark pixels carry little stable clothing colour information.
        mask = np.where(stripe[:, :, 2] >= 24, 255, 0).astype(np.uint8)
        if int(np.count_nonzero(mask)) < 8:
            mask = None
        histogram = cv2.calcHist(
            [stripe], [0, 1], mask, [12, 4], [0, 180, 0, 256]
        ).reshape(-1)
        total = float(histogram.sum())
        if total <= 0:
            return None
        # Hellinger transform makes histogram cosine distance less dominated
        # by one large bin and is robust to moderate illumination changes.
        parts.append(np.sqrt(histogram / total))
    return normalize_descriptor(np.concatenate(parts))


def extract_person_descriptors(
    frame_bgr: np.ndarray | None,
    detections: np.ndarray,
) -> list[np.ndarray | None]:
    """Extract descriptors in exactly the same order as ``detections``."""
    values = np.asarray(detections, dtype=float).reshape((-1, 5))
    return [extract_person_descriptor(frame_bgr, item[:4]) for item in values]
