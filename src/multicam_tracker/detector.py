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
import cv2
import torch
from ultralytics import YOLO

from .config import PipelineConfig


class PersonDetector:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        if cfg.cpu_threads < 1 or cfg.opencv_threads < 1:
            raise ValueError("CPU and OpenCV thread counts must be >= 1")
        if cfg.device == "cpu":
            cv2.setNumThreads(cfg.opencv_threads)
            torch.set_num_threads(cfg.cpu_threads)
        self.model = YOLO(cfg.model_name)
        if cfg.device == "cpu":
            # Ultralytics select_device resets the process thread count when
            # the predictor is first created. Restore our budget before each
            # inference, without initializing a second model or dummy batch.
            self.model.add_callback("on_predict_start", self._configure_cpu)

    @property
    def _cuda(self) -> bool:
        return str(self.cfg.device).split(":", 1)[0].lower() == "cuda"

    def _configure_cpu(self, _predictor) -> None:
        if torch.get_num_threads() != self.cfg.cpu_threads:
            torch.set_num_threads(self.cfg.cpu_threads)

    def detect_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Detect people in a batch of BGR frames.

        Returns one ``(N, 5)`` array per frame: ``[x1, y1, x2, y2, conf]``.
        """
        if not frames:
            return []
        # Ultralytics already wraps prediction in smart_inference_mode.
        results = self.model(
            frames,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf_threshold,
            classes=[self.cfg.person_class_id],
            device=self.cfg.device,
            quantize=16 if self._cuda and self.cfg.detector_fp16 else None,
            compile=self.cfg.detector_compile if self._cuda else False,
            verbose=False,
        )
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                detections.append(np.empty((0, 5), dtype=np.float32))
                continue
            # predict (not model.track) returns xyxy/conf/cls. Transfer once,
            # and copy only the compact metadata so results/images can expire.
            detections.append(boxes.data[:, :5].cpu().numpy().astype(np.float32, copy=True))
        return detections
