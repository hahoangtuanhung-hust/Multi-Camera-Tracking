"""Person detection with YOLO26 (batched across cameras).

One inference call per tick with all camera frames batched together — the
same trick DeepStream uses (nvinfer batches streams into one TensorRT
call). Batching matters more than raw model speed: it amortizes the
per-call overhead of the GPU pipeline across cameras.

YOLO26 is end-to-end (NMS-free): the network directly outputs final boxes,
which removes the traditional non-maximum-suppression postprocessing step.
On Apple Silicon it runs on the GPU via PyTorch's MPS backend.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from .config import PipelineConfig


class PersonDetector:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.model_name)

    def detect_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Detect people in a batch of BGR frames.

        Returns one ``(N, 5)`` array per frame: ``[x1, y1, x2, y2, conf]``.
        """
        results = self.model(
            frames,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf_threshold,
            classes=[self.cfg.person_class_id],
            device=self.cfg.device,
            verbose=False,
        )
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                detections.append(np.empty((0, 5), dtype=np.float32))
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()[:, None]
            detections.append(np.hstack([xyxy, conf]).astype(np.float32))
        return detections
