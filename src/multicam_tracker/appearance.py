"""Same-camera appearance: default HSV and opt-in trained TorchScript ReID.

Descriptors are motion-gated veto/tie-breakers, not standalone identifiers.
Neural weights must be provided explicitly; the default path stays HSV-only.
"""

from __future__ import annotations

import cv2
import numpy as np
from functools import lru_cache
from pathlib import Path
import hashlib


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


@lru_cache(maxsize=2)
def _load_scripted_reid(path: str, sha256: str, device: str):
    """Load a user-supplied, trained TorchScript ReID encoder; never download."""
    import torch
    checkpoint = Path(path)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError("ReID checkpoint SHA-256 mismatch")
    return torch.jit.load(str(checkpoint), map_location=device).eval()


class PersonAppearanceBackend:
    """Same-camera descriptor adapter with quality gating and batched inference.

    TorchScript contract: ImageNet-normalized RGB float32 NCHW, 256x128 input,
    finite NxD output trained for person ReID. A random/classification backbone
    is not substituted for missing trained weights.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.appearance_backend not in {"hsv", "torchscript"}:
            raise ValueError("appearance_backend must be hsv or torchscript")
        if cfg.appearance_gallery_size < 1 or cfg.appearance_min_crop_height < 8:
            raise ValueError("invalid appearance gallery/crop size")
        if not 0 < cfg.appearance_max_crop_overlap <= 1:
            raise ValueError("invalid appearance overlap gate")
        self.model = None
        self.inference_batches = self.inference_crops = self.quality_rejections = 0
        if cfg.appearance_backend == "torchscript":
            if not cfg.appearance_model_path or not cfg.appearance_model_sha256:
                raise ValueError("trained ReID checkpoint path and SHA-256 are required")
            self.model = _load_scripted_reid(str(Path(cfg.appearance_model_path).resolve()),
                                             cfg.appearance_model_sha256, cfg.device)

    def extract(self, frame, detections, *, event=True):
        detections = np.asarray(detections, dtype=float).reshape((-1, 5))
        if self.cfg.appearance_event_triggered and not event:
            return [None] * len(detections)
        if self.model is None and not self.cfg.appearance_quality_filter_enabled:
            return extract_person_descriptors(frame, detections)
        output = [None] * len(detections)
        if frame is None or frame.ndim != 3:
            return output
        height, width = frame.shape[:2]
        boxes = detections[:, :4]
        crops, indices = [], []
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            area = max(0., x2-x1) * max(0., y2-y1)
            overlap = 0.
            for other_index, other in enumerate(boxes):
                if other_index == index:
                    continue
                intersection = np.maximum(np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2]), 0.)
                overlap = max(overlap, float(np.prod(intersection)) / max(area, 1.))
            if (not np.all(np.isfinite(box)) or y2-y1 < self.cfg.appearance_min_crop_height
                    or x2-x1 < 8 or min(x1, y1) < 0 or x2 > width or y2 > height
                    or overlap > self.cfg.appearance_max_crop_overlap):
                self.quality_rejections += 1
                continue
            if self.model is None:
                output[index] = extract_person_descriptor(frame, box)
            else:
                crop = frame[int(y1):int(y2), int(x1):int(x2), :3]
                crops.append(cv2.resize(crop, (128, 256))[:, :, ::-1].copy())
                indices.append(index)
        if crops:
            import torch
            values = np.stack(crops).astype(np.float32).transpose(0, 3, 1, 2) / 255.
            values = (values - np.array([.485, .456, .406], dtype=np.float32)[None, :, None, None]) / np.array([.229, .224, .225], dtype=np.float32)[None, :, None, None]
            with torch.inference_mode():
                embeddings = self.model(torch.from_numpy(values).to(self.cfg.device))
            if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2 or embeddings.shape[0] != len(crops):
                raise ValueError("ReID encoder must return a tensor shaped [N, D]")
            if not bool(torch.isfinite(embeddings).all()):
                raise ValueError("ReID encoder produced non-finite embeddings")
            self.inference_batches += 1
            self.inference_crops += len(crops)
            for index, embedding in zip(indices, embeddings.float().cpu().numpy()):
                output[index] = normalize_descriptor(embedding)
        return output


def gallery_similarity(gallery, descriptor):
    """Best trusted prototype, never a cross-camera comparison."""
    scores = [appearance_similarity(item, descriptor) for item in gallery]
    scores = [score for score in scores if score is not None]
    return max(scores) if scores else None
