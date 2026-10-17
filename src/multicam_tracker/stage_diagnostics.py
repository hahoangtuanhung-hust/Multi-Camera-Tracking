"""Per-camera, per-window diagnostics across the full tracking pipeline."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from .mot_metrics import evaluate_bbox_mot


STAGES = ("detector", "local", "projection", "fusion", "public", "emitted")
CONFIDENCE_THRESHOLDS = (0.15, 0.20, 0.25, 0.50)


@dataclass(frozen=True)
class BboxGroundTruth:
    bbox_xyxy: np.ndarray
    world_xy: np.ndarray


def load_bbox_ground_truth(
    scene_dir: Path,
    cameras: list[str],
    camera_frame_offsets: dict[str, int] | None = None,
    *,
    max_frame: int | None = None,
) -> dict[int, dict[str, dict[int, BboxGroundTruth]]]:
    """Load logical-frame bbox/world GT without materializing the 100 MB file."""
    camera_names = {int(name.split("_")[1]): name for name in cameras}
    offsets = {
        int(name.split("_")[1]): int(offset)
        for name, offset in (camera_frame_offsets or {}).items()
    }
    result: defaultdict[
        int, defaultdict[str, dict[int, BboxGroundTruth]]
    ] = defaultdict(lambda: defaultdict(dict))
    with (scene_dir / "ground_truth.txt").open("r", encoding="utf-8") as source:
        for line in source:
            values = line.split()
            if len(values) < 9:
                continue
            camera_id = int(values[0])
            camera = camera_names.get(camera_id)
            if camera is None:
                continue
            logical_frame = int(values[2]) - offsets.get(camera_id, 0)
            if logical_frame < 0:
                continue
            if max_frame is not None and logical_frame >= max_frame:
                continue
            person_id = int(values[1])
            x, y, width, height = map(float, values[3:7])
            result[logical_frame][camera][person_id] = BboxGroundTruth(
                bbox_xyxy=np.array([x, y, x + width, y + height], dtype=float),
                world_xy=np.array([float(values[7]), float(values[8])], dtype=float),
            )
    return {
        frame: {camera: dict(people) for camera, people in by_camera.items()}
        for frame, by_camera in result.items()
    }


class PipelineStageDiagnostics:
    """Aggregate GT-to-output retention with bounded event evidence."""

    def __init__(
        self,
        ground_truth: dict[int, dict[str, dict[int, BboxGroundTruth]]],
        *,
        stride: int = 1,
        iou_threshold: float = 0.5,
        segment_frames: int = 500,
        max_failure_events: int = 2_000,
        pipeline_detection_threshold: float | None = None,
        confidence_source_floor: float = 0.0,
    ) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        if segment_frames < 1:
            raise ValueError("segment_frames must be >= 1")
        self.ground_truth = ground_truth
        self.stride = stride
        self.iou_threshold = iou_threshold
        self.segment_frames = segment_frames
        self.max_failure_events = max_failure_events
        self.pipeline_detection_threshold = pipeline_detection_threshold
        self.confidence_source_floor = float(confidence_source_floor)
        self.overall: Counter = Counter()
        self.by_camera: defaultdict[str, Counter] = defaultdict(Counter)
        self.by_segment: defaultdict[str, Counter] = defaultdict(Counter)
        self.confidence_thresholds: dict[float, Counter] = {
            threshold: Counter()
            for threshold in CONFIDENCE_THRESHOLDS
            if threshold + 1e-9 >= self.confidence_source_floor
        }
        self.confidence_thresholds_by_camera: defaultdict[
            str, dict[float, Counter]
        ] = defaultdict(lambda: {
            threshold: Counter() for threshold in self.confidence_thresholds
        })
        self.bbox_heights: defaultdict[str, Counter] = defaultdict(Counter)
        self.failure_events: list[dict[str, object]] = []
        self.failure_events_dropped = 0
        self.local_predictions: defaultdict[
            str, list[tuple[int, dict[int, np.ndarray]]]
        ] = defaultdict(list)
        self.local_ground_truth: defaultdict[
            str, dict[int, dict[int, np.ndarray]]
        ] = defaultdict(dict)
        self.per_gt: defaultdict[tuple[str, int], dict[str, object]] = defaultdict(
            self._new_gt_record
        )

    @staticmethod
    def _new_gt_record() -> dict[str, object]:
        return {
            "visible_frames": 0,
            **{stage: 0 for stage in STAGES},
            "local_ids": set(),
            "public_gids": set(),
            "first_frame": None,
            "last_frame": None,
        }

    def update(
        self,
        tick: dict[str, object],
        *,
        confidence_detections_by_camera: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        detections_by_camera = tick.get("detections", {})
        confidence_detections_by_camera = (
            detections_by_camera
            if confidence_detections_by_camera is None
            else confidence_detections_by_camera
        )
        accepted_keys = set(tick.get("accepted_fusion_keys", set()))
        public_affinity = tick.get("local_to_gid", {})
        emitted_gids = {int(track.gid) for track in tick["global_tracks"]}
        current_public_gids = set()

        for camera, packet in tick["packets"].items():
            logical_frame = int(packet.sequence) * self.stride
            gt = self.ground_truth.get(logical_frame, {}).get(camera, {})
            gt_ids = list(gt)
            gt_boxes = _boxes(entry.bbox_xyxy for entry in gt.values())

            detections = np.asarray(
                detections_by_camera.get(camera, np.empty((0, 5))), dtype=float
            ).reshape((-1, 5))
            confidence_detections = np.asarray(
                confidence_detections_by_camera.get(camera, np.empty((0, 5))),
                dtype=float,
            ).reshape((-1, 5))
            local_tracks = list(tick["camera_tracks"].get(camera, []))
            observations = [
                observation
                for observation in tick["observations"]
                if observation.camera == camera
            ]
            fused_observations = [
                observation
                for observation in observations
                if (camera, int(observation.local_id)) in accepted_keys
            ]
            public_observations = [
                observation
                for observation in fused_observations
                if (camera, int(observation.local_id)) in public_affinity
            ]
            emitted_observations = [
                observation
                for observation in public_observations
                if int(public_affinity[(camera, int(observation.local_id))])
                in emitted_gids
            ]
            current_public_gids.update(
                int(public_affinity[(camera, int(observation.local_id))])
                for observation in public_observations
            )

            stage_boxes = {
                "detector": detections[:, :4],
                "local": _boxes(track.xyxy for track in local_tracks),
                "projection": _boxes(obs.bbox_xyxy for obs in observations),
                "fusion": _boxes(obs.bbox_xyxy for obs in fused_observations),
                "public": _boxes(obs.bbox_xyxy for obs in public_observations),
                "emitted": _boxes(obs.bbox_xyxy for obs in emitted_observations),
            }
            matches = {
                stage: _match_boxes(gt_boxes, boxes, self.iou_threshold)
                for stage, boxes in stage_boxes.items()
            }
            segment_start = (logical_frame // self.segment_frames) * self.segment_frames
            segment = f"{segment_start}-{segment_start + self.segment_frames - 1}"
            groups = (self.overall, self.by_camera[camera], self.by_segment[segment])
            for group in groups:
                group["frames"] += 1
                group["gt_instances"] += len(gt_boxes)
                for stage, boxes in stage_boxes.items():
                    matched = len(matches[stage])
                    group[f"{stage}_tp"] += matched
                    group[f"{stage}_fp"] += len(boxes) - matched
                    group[f"{stage}_fn"] += len(gt_boxes) - matched

            for threshold, counter in self.confidence_thresholds.items():
                selected = confidence_detections[
                    confidence_detections[:, 4] >= threshold
                ]
                selected_matches = _match_boxes(
                    gt_boxes, selected[:, :4], self.iou_threshold
                )
                for target in (
                    counter,
                    self.confidence_thresholds_by_camera[camera][threshold],
                ):
                    target["tp"] += len(selected_matches)
                    target["fp"] += len(selected) - len(selected_matches)
                    target["fn"] += len(gt_boxes) - len(selected_matches)

            detected_gt = set(matches["detector"])
            for index, gt_id in enumerate(gt_ids):
                height_bin = _height_bin(gt_boxes[index])
                self.bbox_heights[height_bin]["gt"] += 1
                self.bbox_heights[height_bin]["detected"] += index in detected_gt
                record = self.per_gt[(camera, int(gt_id))]
                record["visible_frames"] += 1
                record["first_frame"] = (
                    logical_frame
                    if record["first_frame"] is None
                    else min(int(record["first_frame"]), logical_frame)
                )
                record["last_frame"] = logical_frame
                present = {stage: index in stage_match for stage, stage_match in matches.items()}
                for stage, value in present.items():
                    record[stage] += int(value)

                if not present["detector"]:
                    loss_stage = "detector"
                elif not present["local"]:
                    loss_stage = "detector_to_local"
                elif not present["projection"]:
                    loss_stage = "local_to_projection"
                elif not present["fusion"]:
                    loss_stage = "projection_to_fusion"
                elif not present["public"]:
                    loss_stage = "fusion_to_public"
                elif not present["emitted"]:
                    loss_stage = "public_to_emitted"
                else:
                    loss_stage = "complete"
                for group in groups:
                    group[f"stage_loss_{loss_stage}"] += 1

                local_index = matches["local"].get(index)
                if local_index is not None:
                    record["local_ids"].add(int(local_tracks[local_index].track_id))
                public_index = matches["public"].get(index)
                if public_index is not None:
                    observation = public_observations[public_index]
                    record["public_gids"].add(int(public_affinity[
                        (camera, int(observation.local_id))
                    ]))

                if loss_stage != "complete":
                    if len(self.failure_events) < self.max_failure_events:
                        self.failure_events.append({
                            "frame": logical_frame,
                            "camera": camera,
                            "gt_id": int(gt_id),
                            "first_failed_stage": loss_stage,
                            "bbox_xyxy": gt_boxes[index].tolist(),
                        })
                    else:
                        self.failure_events_dropped += 1

            projected_matches = matches["projection"]
            for gt_index, observation_index in projected_matches.items():
                error = float(np.linalg.norm(
                    gt[gt_ids[gt_index]].world_xy
                    - np.asarray(observations[observation_index].world_xy, dtype=float)
                ))
                for group in groups:
                    group["projection_error_count"] += 1
                    group["projection_squared_error_m2"] += error * error

            self.local_predictions[camera].append((
                logical_frame,
                {
                    int(track.track_id): np.asarray(track.xyxy, dtype=float).copy()
                    for track in local_tracks
                },
            ))
            self.local_ground_truth[camera][logical_frame] = {
                int(person_id): entry.bbox_xyxy.copy()
                for person_id, entry in gt.items()
            }

        self.overall["emitted_global_without_current_public_observation"] += len(
            emitted_gids - current_public_gids
        )

    def finalize(self) -> dict[str, object]:
        local_metrics = {
            camera: evaluate_bbox_mot(
                predictions,
                self.local_ground_truth[camera],
                iou_threshold=self.iou_threshold,
            )
            for camera, predictions in sorted(self.local_predictions.items())
        }
        worst_gt = []
        for (camera, gt_id), record in self.per_gt.items():
            visible = int(record["visible_frames"])
            item = {
                "camera": camera,
                "gt_id": gt_id,
                "visible_frames": visible,
                "first_frame": record["first_frame"],
                "last_frame": record["last_frame"],
                **{
                    f"{stage}_coverage": float(record[stage]) / max(visible, 1)
                    for stage in STAGES
                },
                "local_ids": sorted(record["local_ids"]),
                "local_id_count": len(record["local_ids"]),
                "public_gids": sorted(record["public_gids"]),
                "public_gid_count": len(record["public_gids"]),
            }
            worst_gt.append(item)
        worst_gt.sort(key=lambda item: (
            item["emitted_coverage"],
            -item["public_gid_count"],
            -item["local_id_count"],
            item["camera"],
            item["gt_id"],
        ))
        return {
            "iou_threshold": self.iou_threshold,
            "segment_frames": self.segment_frames,
            "pipeline_detection_threshold": self.pipeline_detection_threshold,
            "confidence_source_floor": self.confidence_source_floor,
            "overall": _summarize(self.overall),
            "by_camera": {
                camera: _summarize(counter)
                for camera, counter in sorted(self.by_camera.items())
            },
            "by_segment": {
                segment: _summarize(counter)
                for segment, counter in sorted(
                    self.by_segment.items(),
                    key=lambda item: int(item[0].split("-")[0]),
                )
            },
            "detection_by_confidence_threshold": {
                f"{threshold:.2f}": _precision_recall(counter)
                for threshold, counter in self.confidence_thresholds.items()
            },
            "detection_by_confidence_threshold_and_camera": {
                camera: {
                    f"{threshold:.2f}": _precision_recall(counter)
                    for threshold, counter in counters.items()
                }
                for camera, counters in sorted(
                    self.confidence_thresholds_by_camera.items()
                )
            },
            "detection_recall_by_bbox_height": {
                name: {
                    "gt": int(counter["gt"]),
                    "detected": int(counter["detected"]),
                    "recall": float(counter["detected"] / max(counter["gt"], 1)),
                }
                for name, counter in sorted(self.bbox_heights.items())
            },
            "local_tracking_by_camera": local_metrics,
            "worst_gt_camera_trajectories": worst_gt[:25],
            "failure_events": self.failure_events,
            "failure_events_dropped": self.failure_events_dropped,
            "failure_events_truncated": self.failure_events_dropped > 0,
        }


def _boxes(values: Iterable[np.ndarray]) -> np.ndarray:
    return np.asarray(list(values), dtype=float).reshape((-1, 4))


def _match_boxes(
    gt_boxes: np.ndarray,
    candidate_boxes: np.ndarray,
    threshold: float,
) -> dict[int, int]:
    if not len(gt_boxes) or not len(candidate_boxes):
        return {}
    iou = _iou_matrix(gt_boxes, candidate_boxes)
    rows, columns = linear_sum_assignment(-iou)
    return {
        int(row): int(column)
        for row, column in zip(rows, columns)
        if iou[row, column] >= threshold
    }


def _iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    top_left = np.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = np.minimum(left[:, None, 2:], right[None, :, 2:])
    wh = np.maximum(0.0, bottom_right - top_left)
    intersection = wh[..., 0] * wh[..., 1]
    left_area = np.maximum(0.0, left[:, 2] - left[:, 0]) * np.maximum(
        0.0, left[:, 3] - left[:, 1]
    )
    right_area = np.maximum(0.0, right[:, 2] - right[:, 0]) * np.maximum(
        0.0, right[:, 3] - right[:, 1]
    )
    union = left_area[:, None] + right_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > np.finfo(float).eps,
    )


def _height_bin(box: np.ndarray) -> str:
    height = max(0.0, float(box[3] - box[1]))
    if height < 64.0:
        return "lt_64px"
    if height < 128.0:
        return "64_127px"
    return "ge_128px"


def _precision_recall(counter: Counter) -> dict[str, int | float]:
    tp, fp, fn = (int(counter[name]) for name in ("tp", "fp", "fn"))
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
    }


def _summarize(counter: Counter) -> dict[str, object]:
    result: dict[str, object] = {
        key: int(value) if float(value).is_integer() else float(value)
        for key, value in counter.items()
        if key != "projection_squared_error_m2"
    }
    for stage in STAGES:
        result[stage] = _precision_recall(Counter({
            "tp": counter[f"{stage}_tp"],
            "fp": counter[f"{stage}_fp"],
            "fn": counter[f"{stage}_fn"],
        }))
    error_count = int(counter["projection_error_count"])
    result["projection_rmse_m"] = float(np.sqrt(
        counter["projection_squared_error_m2"] / max(error_count, 1)
    ))
    result["stage_loss_gt_instances"] = {
        name: int(counter[f"stage_loss_{name}"])
        for name in (
            "detector",
            "detector_to_local",
            "local_to_projection",
            "projection_to_fusion",
            "fusion_to_public",
            "public_to_emitted",
            "complete",
        )
    }
    return result
