"""Standard MOT metrics for world-plane point trajectories.

The metric definitions follow the official TrackEval implementations of
Identity, CLEAR and HOTA. Detection similarity is adapted from bounding-box
IoU to a ground-plane point similarity::

    similarity = max(0, 1 - distance_m / match_gate_m)

Thus CLEAR/Identity accept points strictly inside ``match_gate_m`` and HOTA
averages the standard 19 alpha thresholds from 0.05 through 0.95.
"""

from __future__ import annotations

from collections import defaultdict

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


def evaluate_mot_slices(
    predictions: list[tuple[int, dict[int, np.ndarray]]],
    ground_truth: dict[int, dict[int, np.ndarray]],
    *,
    frame_offset: int = 0,
    match_gate_m: float = 1.0,
    cumulative_frames: tuple[int, ...] | list[int] = (
        500, 1000, 1500, 2000, 4000
    ),
    window_frames: int = 500,
) -> dict[str, list[dict[str, object]]]:
    """Evaluate from-start prefixes and independent windows from one run.

    Identity metrics are sequence-global.  Reporting both views makes a GID
    rename across a window boundary visible instead of accidentally resetting
    the GT-to-predicted-ID assignment for every benchmark job.
    """
    if window_frames < 0:
        raise ValueError("window_frames must be >= 0")
    if any(int(value) <= 0 for value in cumulative_frames):
        raise ValueError("cumulative frame counts must be positive")
    if not predictions:
        return {"cumulative_metrics": [], "metric_windows": []}

    counts = {
        min(int(value), len(predictions))
        for value in cumulative_frames
        if int(value) <= len(predictions)
    }
    counts.add(len(predictions))
    cumulative = [
        _metric_slice(
            predictions[:count], ground_truth, frame_offset, match_gate_m
        )
        for count in sorted(counts)
    ]
    windows: list[dict[str, object]] = []
    if window_frames:
        for start in range(0, len(predictions), window_frames):
            windows.append(_metric_slice(
                predictions[start:start + window_frames],
                ground_truth,
                frame_offset,
                match_gate_m,
            ))
    return {"cumulative_metrics": cumulative, "metric_windows": windows}


def identity_trajectory_diagnostics(
    predictions: list[tuple[int, dict[int, np.ndarray]]],
    ground_truth: dict[int, dict[int, np.ndarray]],
    *,
    frame_offset: int = 0,
    match_gate_m: float = 1.0,
    long_gap_frames: int = 90,
    crossing_distance_m: float = 1.5,
) -> dict[str, object]:
    """Describe long-sequence GID fragmentation for every ground-truth ID.

    Matching deliberately follows CLEAR's continuity-first assignment.  The
    resulting diagnostics therefore explain the same switch events reported
    by :func:`evaluate_mot`, while retaining the full per-person GID lifetime
    distribution needed to diagnose cumulative-IDF1 collapse.
    """
    if long_gap_frames < 1:
        raise ValueError("long_gap_frames must be >= 1")
    if not np.isfinite(crossing_distance_m) or crossing_distance_m <= 0:
        raise ValueError("crossing_distance_m must be finite and positive")
    sequence = _prepare_sequence(
        predictions, ground_truth, frame_offset, match_gate_m
    )
    people: defaultdict[int, dict[str, object]] = defaultdict(lambda: {
        "visible_frames": 0,
        "matched_frames": 0,
        "matches": defaultdict(lambda: {
            "matched_frames": 0,
            "first_frame": None,
            "last_frame": None,
        }),
        "switch_count": 0,
        "longest_missing_gap": 0,
    })
    previous_gid: dict[int, int] = {}
    previous_frame: dict[int, int] = {}
    long_gap_attempts = long_gap_successes = crossing_switches = switches = 0

    for frame in sequence["frames"]:
        frame_index = int(frame["frame_index"])
        gt_original = frame["gt_original"]
        tracker_original = frame["tracker_original"]
        for gt_id in gt_original:
            people[int(gt_id)]["visible_frames"] += 1
        if not len(gt_original) or not len(tracker_original):
            continue

        continuation = np.asarray([
            previous_gid.get(int(gt_id), -1) for gt_id in gt_original
        ], dtype=int)[:, None] == tracker_original[None, :]
        score = 1000.0 * continuation + frame["similarity"]
        score[frame["similarity"] <= 0.0] = 0.0
        rows, columns = linear_sum_assignment(-score)
        accepted = score[rows, columns] > 0.0
        rows, columns = rows[accepted], columns[accepted]

        frame_gt = ground_truth.get(frame_index + frame_offset, {})
        for row, column in zip(rows, columns):
            gt_id = int(gt_original[row])
            gid = int(tracker_original[column])
            record = people[gt_id]
            record["matched_frames"] += 1
            gid_record = record["matches"][gid]
            gid_record["matched_frames"] += 1
            if gid_record["first_frame"] is None:
                gid_record["first_frame"] = frame_index
            gid_record["last_frame"] = frame_index

            old_gid = previous_gid.get(gt_id)
            old_frame = previous_frame.get(gt_id)
            if old_frame is not None:
                missing_gap = max(0, frame_index - old_frame - 1)
                record["longest_missing_gap"] = max(
                    int(record["longest_missing_gap"]), missing_gap
                )
                if frame_index - old_frame > long_gap_frames:
                    long_gap_attempts += 1
                    if old_gid == gid:
                        long_gap_successes += 1
            if old_gid is not None and old_gid != gid:
                switches += 1
                record["switch_count"] += 1
                point = frame_gt.get(gt_id)
                if point is not None:
                    neighbor_distances = [
                        float(np.linalg.norm(point - other))
                        for other_id, other in frame_gt.items()
                        if int(other_id) != gt_id
                    ]
                    if (
                        neighbor_distances
                        and min(neighbor_distances) <= crossing_distance_m
                    ):
                        crossing_switches += 1
            previous_gid[gt_id] = gid
            previous_frame[gt_id] = frame_index

    per_gt: list[dict[str, object]] = []
    gid_counts: list[int] = []
    coverages: list[float] = []
    wrong_id_duration = 0
    for gt_id, record in sorted(people.items()):
        matched = int(record["matched_frames"])
        visible = int(record["visible_frames"])
        predicted_gids = [
            {
                "gid": int(gid),
                "first_frame": int(details["first_frame"]),
                "last_frame": int(details["last_frame"]),
                "matched_frames": int(details["matched_frames"]),
            }
            for gid, details in sorted(
                record["matches"].items(),
                key=lambda item: (-item[1]["matched_frames"], item[0]),
            )
        ]
        dominant = max(
            (item["matched_frames"] for item in predicted_gids), default=0
        )
        coverage = dominant / max(visible, 1)
        gid_count = len(predicted_gids)
        gid_counts.append(gid_count)
        coverages.append(coverage)
        wrong_id_duration += max(0, matched - dominant)
        per_gt.append({
            "gt_id": int(gt_id),
            "predicted_gids": predicted_gids,
            "gid_count": gid_count,
            "visible_frames": visible,
            "matched_frames": matched,
            "dominant_gid_coverage": float(coverage),
            "switch_count": int(record["switch_count"]),
            "longest_missing_gap": int(record["longest_missing_gap"]),
        })

    lifetimes: list[dict[str, int]] = []
    for gid in sorted({gid for _, values in predictions for gid in values}):
        frames = [frame for frame, values in predictions if gid in values]
        lifetimes.append({
            "gid": int(gid),
            "first_frame": int(frames[0]),
            "last_frame": int(frames[-1]),
            "observed_frames": len(frames),
            "span_frames": int(frames[-1] - frames[0] + 1),
        })

    return {
        "per_gt": per_gt,
        "predicted_gids_per_gt": _distribution_summary(gid_counts),
        "dominant_gid_coverage": _distribution_summary(coverages),
        "gid_churn_per_1000_frames": float(
            1000.0 * sum(max(0, count - 1) for count in gid_counts)
            / max(len(predictions), 1)
        ),
        "wrong_id_duration_frames": int(wrong_id_duration),
        "long_gap_reconnect_attempts": int(long_gap_attempts),
        "long_gap_reconnect_successes": int(long_gap_successes),
        "long_gap_reconnect_success_rate": float(
            long_gap_successes / max(long_gap_attempts, 1)
        ),
        "crossing_switches": int(crossing_switches),
        "crossing_switch_rate": float(crossing_switches / max(switches, 1)),
        "public_gid_lifetimes": lifetimes,
    }


def build_identity_error_taxonomy(
    metrics: dict[str, object],
    identity_diagnostics: dict[str, object],
    pipeline_metrics: dict[str, object],
    *,
    switch_diagnostics: list[dict[str, object]] | None = None,
    track_exists_not_output: int = 0,
) -> dict[str, dict[str, object]]:
    """Map available evaluator/runtime evidence to the ten failure classes.

    Some failure causes cannot be uniquely recovered from fused world points.
    Such values are explicitly marked as proxies instead of pretending that
    every false negative originated in the detector.
    """
    switches = switch_diagnostics or []
    long_attempts = int(identity_diagnostics.get("long_gap_reconnect_attempts", 0))
    long_successes = int(identity_diagnostics.get("long_gap_reconnect_successes", 0))
    trajectory_rejections = int(
        pipeline_metrics.get("trajectory_matches_rejected", 0)
    )
    publication_rejections = int(
        pipeline_metrics.get("new_gid_trajectory_rejected", 0)
    )
    return {
        "detector_miss": {
            "count": int(metrics.get("FN", 0)),
            "evidence": "upper_bound_proxy",
            "note": "Fused FN also includes local/fusion/output misses.",
        },
        "tentative_local_track_deleted": {
            "count": int(pipeline_metrics.get("local_tracks_removed_tentative", 0)),
            "evidence": "runtime_counter",
        },
        "local_switch_at_crossing": {
            "count": sum(
                bool(record.get("near_crossing"))
                and record.get("classification") == "local_fragmentation"
                for record in switches
            ),
            "evidence": "evaluator_switch_context" if switches else "unavailable",
        },
        "local_fragmentation_after_occlusion": {
            "count": sum(
                record.get("classification") == "local_fragmentation"
                and int(record.get("gap_frames", 0)) > 1
                for record in switches
            ),
            "evidence": "evaluator_switch_context" if switches else "unavailable",
        },
        "global_reassignment_local_stable": {
            "count": sum(
                record.get("classification") == "global_changed_local_stable"
                for record in switches
            ),
            "evidence": "evaluator_switch_context" if switches else "unavailable",
        },
        "dormant_reconnect_failed": {
            "count": max(0, long_attempts - long_successes),
            "evidence": "trajectory_matching",
        },
        "motion_gate_rejection": {
            "count": int(pipeline_metrics.get("fusion_motion_jump_rejected", 0)),
            "evidence": "runtime_counter",
        },
        "trajectory_verification_rejection": {
            "count": trajectory_rejections + publication_rejections,
            "evidence": "runtime_counter",
            "details": {
                "association": trajectory_rejections,
                "publication": publication_rejections,
            },
        },
        "entry_publication_deferred": {
            "count": int(pipeline_metrics.get("new_gid_entry_deferred", 0)),
            "evidence": "runtime_counter",
        },
        "track_exists_not_output": {
            "count": int(track_exists_not_output),
            "evidence": "per_tick_observation_vs_public_affinity",
        },
    }


def _metric_slice(predictions, ground_truth, frame_offset, match_gate_m):
    metrics = evaluate_mot(
        predictions,
        ground_truth,
        frame_offset=frame_offset,
        match_gate_m=match_gate_m,
    )
    return {
        "start_frame": int(predictions[0][0]),
        "end_frame": int(predictions[-1][0]),
        "ticks": len(predictions),
        **{name: value for name, value in metrics.items() if name != "switch_events"},
    }


def _distribution_summary(values) -> dict[str, float | int]:
    samples = np.asarray(list(values), dtype=float)
    if not len(samples):
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(len(samples)),
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "p95": float(np.percentile(samples, 95)),
        "max": float(np.max(samples)),
    }


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
    tp = fp = fn = id_switches = fragmentations = 0
    tracked_before = np.zeros(sequence["num_gt_ids"], dtype=bool)
    tracked_previous_frame = np.zeros(sequence["num_gt_ids"], dtype=bool)
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
            tracked_previous_frame[:] = False
            continue
        if not len(tracker_ids):
            fn += len(gt_ids)
            previous_frame_match[:] = -1
            tracked_previous_frame[:] = False
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
        fragmentations += int(np.sum(
            tracked_before[matched_gt] & ~tracked_previous_frame[matched_gt]
        ))
        tracked_before[matched_gt] = True
        tracked_previous_frame[:] = False
        tracked_previous_frame[matched_gt] = True

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
        "Frag": fragmentations,
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
