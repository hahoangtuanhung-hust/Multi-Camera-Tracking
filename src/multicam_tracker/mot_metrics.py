"""Standard MOT metrics for world-plane point trajectories.

The metric definitions follow the official TrackEval implementations of
Identity, CLEAR and HOTA. Detection similarity is adapted from bounding-box
IoU to a ground-plane point similarity::

    similarity = max(0, 1 - distance_m / match_gate_m)

Thus CLEAR/Identity accept points strictly inside ``match_gate_m`` and HOTA
averages the standard 19 alpha thresholds from 0.05 through 0.95.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate_mot(
    predictions: list[tuple[int, dict[int, np.ndarray]]],
    ground_truth: dict[int, dict[int, np.ndarray]],
    frame_offset: int = 0,
    match_gate_m: float = 1.0,
) -> dict[str, float | int | list[tuple[int, int, int, int, int]]]:
    """Evaluate one global-ID sequence with Identity, CLEAR and HOTA metrics."""
    if not np.isfinite(match_gate_m) or match_gate_m <= 0:
        raise ValueError(f"match_gate_m must be positive, got {match_gate_m}")
    sequence = _prepare_sequence(
        predictions, ground_truth, frame_offset, match_gate_m
    )
    identity = _identity_metrics(sequence)
    clear = _clear_metrics(sequence)
    hota = _hota_metrics(sequence)
    return {**identity, **clear, **hota}


def _prepare_sequence(predictions, ground_truth, frame_offset, match_gate_m):
    gt_values = {
        int(gt_id)
        for frame_index, _ in predictions
        for gt_id in ground_truth.get(frame_index + frame_offset, {})
    }
    tracker_values = {
        int(tracker_id)
        for _, frame_predictions in predictions
        for tracker_id in frame_predictions
    }
    gt_lookup = {value: index for index, value in enumerate(sorted(gt_values))}
    tracker_lookup = {
        value: index for index, value in enumerate(sorted(tracker_values))
    }

    frames = []
    for frame_index, frame_predictions in predictions:
        frame_gt = ground_truth.get(frame_index + frame_offset, {})
        gt_original = np.asarray(list(frame_gt), dtype=int)
        tracker_original = np.asarray(list(frame_predictions), dtype=int)
        gt_ids = np.asarray([gt_lookup[value] for value in gt_original], dtype=int)
        tracker_ids = np.asarray(
            [tracker_lookup[value] for value in tracker_original], dtype=int
        )
        gt_xy = np.asarray(
            [frame_gt[value] for value in gt_original], dtype=float
        ).reshape((-1, 2))
        tracker_xy = np.asarray(
            [frame_predictions[value] for value in tracker_original], dtype=float
        ).reshape((-1, 2))
        if len(gt_xy) and len(tracker_xy):
            distance = np.linalg.norm(
                gt_xy[:, None, :] - tracker_xy[None, :, :], axis=2
            )
            similarity = np.maximum(0.0, 1.0 - distance / match_gate_m)
        else:
            distance = np.empty((len(gt_xy), len(tracker_xy)), dtype=float)
            similarity = np.empty_like(distance)
        frames.append({
            "frame_index": int(frame_index),
            "gt_ids": gt_ids,
            "tracker_ids": tracker_ids,
            "gt_original": gt_original,
            "tracker_original": tracker_original,
            "distance": distance,
            "similarity": similarity,
        })
    return {
        "frames": frames,
        "num_gt_ids": len(gt_lookup),
        "num_tracker_ids": len(tracker_lookup),
        "num_gt_dets": sum(len(frame["gt_ids"]) for frame in frames),
        "num_tracker_dets": sum(len(frame["tracker_ids"]) for frame in frames),
        "tracker_original_ids": np.asarray(sorted(tracker_values), dtype=int),
    }


def _identity_metrics(sequence):
    num_gt_ids = sequence["num_gt_ids"]
    num_tracker_ids = sequence["num_tracker_ids"]
    num_gt_dets = sequence["num_gt_dets"]
    num_tracker_dets = sequence["num_tracker_dets"]
    if not num_gt_dets or not num_tracker_dets:
        idtp = 0
        idfn = num_gt_dets
        idfp = num_tracker_dets
    else:
        potential = np.zeros((num_gt_ids, num_tracker_ids), dtype=float)
        gt_count = np.zeros(num_gt_ids, dtype=float)
        tracker_count = np.zeros(num_tracker_ids, dtype=float)
        for frame in sequence["frames"]:
            gt_ids = frame["gt_ids"]
            tracker_ids = frame["tracker_ids"]
            rows, columns = np.nonzero(frame["similarity"] > 0.0)
            potential[gt_ids[rows], tracker_ids[columns]] += 1
            gt_count[gt_ids] += 1
            tracker_count[tracker_ids] += 1

        size = num_gt_ids + num_tracker_ids
        large = 1e10
        false_positive = np.zeros((size, size), dtype=float)
        false_negative = np.zeros((size, size), dtype=float)
        false_positive[num_gt_ids:, :num_tracker_ids] = large
        false_negative[:num_gt_ids, num_tracker_ids:] = large
        for gt_id in range(num_gt_ids):
            false_negative[gt_id, :num_tracker_ids] = gt_count[gt_id]
            false_negative[gt_id, num_tracker_ids + gt_id] = gt_count[gt_id]
        for tracker_id in range(num_tracker_ids):
            false_positive[:num_gt_ids, tracker_id] = tracker_count[tracker_id]
            false_positive[num_gt_ids + tracker_id, tracker_id] = tracker_count[tracker_id]
        false_negative[:num_gt_ids, :num_tracker_ids] -= potential
        false_positive[:num_gt_ids, :num_tracker_ids] -= potential
        rows, columns = linear_sum_assignment(false_negative + false_positive)
        idfn = int(round(float(false_negative[rows, columns].sum())))
        idfp = int(round(float(false_positive[rows, columns].sum())))
        idtp = int(num_gt_dets - idfn)

    idr = idtp / max(idtp + idfn, 1)
    idp = idtp / max(idtp + idfp, 1)
    idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1)
    return {
        "IDTP": idtp,
        "IDFN": idfn,
        "IDFP": idfp,
        "IDR": idr,
        "IDP": idp,
        "IDF1": idf1,
    }


def _clear_metrics(sequence):
    previous_match = np.full(sequence["num_gt_ids"], -1, dtype=int)
    previous_frame_match = np.full(sequence["num_gt_ids"], -1, dtype=int)
    previous_frame_index: dict[int, int] = {}
    tp = fp = fn = id_switches = 0
    similarity_sum = 0.0
    distance_sum = 0.0
    squared_distance_sum = 0.0
    switch_events: list[tuple[int, int, int, int, int]] = []

    for frame in sequence["frames"]:
        gt_ids = frame["gt_ids"]
        tracker_ids = frame["tracker_ids"]
        similarity = frame["similarity"]
        if not len(gt_ids):
            fp += len(tracker_ids)
            previous_frame_match[:] = -1
            continue
        if not len(tracker_ids):
            fn += len(gt_ids)
            previous_frame_match[:] = -1
            continue

        continuation = (
            tracker_ids[None, :] == previous_frame_match[gt_ids, None]
        )
        score = 1000.0 * continuation + similarity
        score[similarity <= 0.0] = 0.0
        rows, columns = linear_sum_assignment(-score)
        matched = score[rows, columns] > 0.0
        rows = rows[matched]
        columns = columns[matched]
        matched_gt = gt_ids[rows]
        matched_tracker = tracker_ids[columns]

        for row, column, gt_id, tracker_id in zip(
            rows, columns, matched_gt, matched_tracker
        ):
            old_tracker = previous_match[gt_id]
            if old_tracker >= 0 and old_tracker != tracker_id:
                id_switches += 1
                switch_events.append((
                    int(frame["gt_original"][row]),
                    previous_frame_index.get(int(gt_id), frame["frame_index"]),
                    int(frame["frame_index"]),
                    int(sequence["tracker_original_ids"][old_tracker]),
                    int(sequence["tracker_original_ids"][tracker_id]),
                ))
            previous_match[gt_id] = tracker_id
            previous_frame_index[int(gt_id)] = int(frame["frame_index"])

        previous_frame_match[:] = -1
        previous_frame_match[matched_gt] = matched_tracker
        match_distance = frame["distance"][rows, columns]
        tp += len(rows)
        fn += len(gt_ids) - len(rows)
        fp += len(tracker_ids) - len(rows)
        similarity_sum += float(similarity[rows, columns].sum())
        distance_sum += float(match_distance.sum())
        squared_distance_sum += float(np.square(match_distance).sum())

    gt_detections = tp + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(gt_detections, 1)
    mota = 1.0 - (fn + fp + id_switches) / max(gt_detections, 1)
    motp = similarity_sum / max(tp, 1)
    motp_distance = distance_sum / max(tp, 1)
    position_rmse = float(np.sqrt(squared_distance_sum / max(tp, 1)))
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "IDSW": id_switches,
        "MOTA": mota,
        "MOTP": motp,
        "MOTP_distance_m": motp_distance,
        "position_RMSE_m": position_rmse,
        "switch_events": switch_events,
    }


def _hota_metrics(sequence):
    alphas = np.arange(0.05, 0.99, 0.05)
    num_gt_ids = sequence["num_gt_ids"]
    num_tracker_ids = sequence["num_tracker_ids"]
    gt_count = np.zeros((num_gt_ids, 1), dtype=float)
    tracker_count = np.zeros((1, num_tracker_ids), dtype=float)
    potential = np.zeros((num_gt_ids, num_tracker_ids), dtype=float)

    for frame in sequence["frames"]:
        gt_ids = frame["gt_ids"]
        tracker_ids = frame["tracker_ids"]
        similarity = frame["similarity"]
        if len(gt_ids) and len(tracker_ids):
            denominator = (
                similarity.sum(axis=0)[None, :]
                + similarity.sum(axis=1)[:, None]
                - similarity
            )
            normalized = np.divide(
                similarity,
                denominator,
                out=np.zeros_like(similarity),
                where=denominator > np.finfo(float).eps,
            )
            potential[gt_ids[:, None], tracker_ids[None, :]] += normalized
        gt_count[gt_ids] += 1
        tracker_count[0, tracker_ids] += 1

    alignment_denominator = gt_count + tracker_count - potential
    alignment = np.divide(
        potential,
        alignment_denominator,
        out=np.zeros_like(potential),
        where=alignment_denominator > np.finfo(float).eps,
    )
    true_positive = np.zeros(len(alphas), dtype=float)
    false_negative = np.zeros(len(alphas), dtype=float)
    false_positive = np.zeros(len(alphas), dtype=float)
    localization_sum = np.zeros(len(alphas), dtype=float)
    match_counts = [np.zeros_like(potential) for _ in alphas]

    for frame in sequence["frames"]:
        gt_ids = frame["gt_ids"]
        tracker_ids = frame["tracker_ids"]
        similarity = frame["similarity"]
        if not len(gt_ids):
            false_positive += len(tracker_ids)
            continue
        if not len(tracker_ids):
            false_negative += len(gt_ids)
            continue
        score = alignment[gt_ids[:, None], tracker_ids[None, :]] * similarity
        rows, columns = linear_sum_assignment(-score)
        for index, alpha in enumerate(alphas):
            accepted = similarity[rows, columns] >= alpha - np.finfo(float).eps
            accepted_rows = rows[accepted]
            accepted_columns = columns[accepted]
            count = len(accepted_rows)
            true_positive[index] += count
            false_negative[index] += len(gt_ids) - count
            false_positive[index] += len(tracker_ids) - count
            if count:
                localization_sum[index] += float(
                    similarity[accepted_rows, accepted_columns].sum()
                )
                match_counts[index][
                    gt_ids[accepted_rows], tracker_ids[accepted_columns]
                ] += 1

    association = np.zeros(len(alphas), dtype=float)
    association_recall = np.zeros(len(alphas), dtype=float)
    association_precision = np.zeros(len(alphas), dtype=float)
    for index, matches in enumerate(match_counts):
        pair_union = gt_count + tracker_count - matches
        pair_accuracy = matches / np.maximum(1.0, pair_union)
        pair_recall = matches / np.maximum(1.0, gt_count)
        pair_precision = matches / np.maximum(1.0, tracker_count)
        association[index] = (
            np.sum(matches * pair_accuracy) / max(true_positive[index], 1.0)
        )
        association_recall[index] = (
            np.sum(matches * pair_recall) / max(true_positive[index], 1.0)
        )
        association_precision[index] = (
            np.sum(matches * pair_precision) / max(true_positive[index], 1.0)
        )

    detection = true_positive / np.maximum(
        1.0, true_positive + false_negative + false_positive
    )
    localization = localization_sum / np.maximum(1.0, true_positive)
    hota = np.sqrt(detection * association)
    return {
        "HOTA": float(np.mean(hota)),
        "DetA": float(np.mean(detection)),
        "AssA": float(np.mean(association)),
        "LocA": float(np.mean(localization)),
        "AssRe": float(np.mean(association_recall)),
        "AssPr": float(np.mean(association_precision)),
    }
