"""Cross-camera fusion with event-time trajectory reassociation.

The legacy synchronous geometry path remains available when observations do
not carry timestamps. Timestamped observations first receive provisional
global IDs, then bounded tracklet history can merge those IDs after enough
motion evidence exists. No appearance, ReID or face feature is used.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .bytetrack import ConstantVelocityKalman
from .config import PipelineConfig
from .tracklets import CameraClockOffsetEstimator, MatchResult, TrackletKey, TrackletMatcher, TrackletStore


@dataclass
class CameraObservation:
    """One confirmed local track projected to the shared ground plane."""

    camera: str
    local_id: int
    world_xy: np.ndarray
    bbox_xyxy: np.ndarray
    conf: float
    capture_ts_s: float | None = None
    world_velocity: np.ndarray | None = None
    arrival_ts_s: float | None = None


@dataclass
class GlobalTrack:
    gid: int
    kf: ConstantVelocityKalman
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    cameras: set[str] = field(default_factory=set)
    trail: list[np.ndarray] = field(default_factory=list)
    created_at_s: float | None = None
    state_timestamp_s: float | None = None
    last_seen_ts_s: float | None = None
    last_arrival_ts_s: float | None = None
    emitted: bool = False

    @property
    def world_xy(self) -> np.ndarray:
        return self.kf.position


class GlobalTrackManager:
    """Fuse observations and maintain canonical, alias-safe global IDs."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.tracks: dict[int, GlobalTrack] = {}
        self.affinity: dict[TrackletKey, int] = {}
        self.aliases: dict[int, int] = {}
        self._alias_created_s: dict[int, float] = {}
        self.members: defaultdict[int, set[TrackletKey]] = defaultdict(set)
        self.tracklets = TrackletStore(cfg)
        reference = cfg.cameras[0] if cfg.cameras else "reference"
        self.clock_estimator = CameraClockOffsetEstimator(cfg, reference)
        self.matcher = TrackletMatcher(cfg)
        self.metrics: defaultdict[str, int | float] = defaultdict(int)
        self._next_gid = 1
        self._event_mode_started = False
        self._max_event_ts_s = float("-inf")
        self._max_arrival_ts_s = float("-inf")
        self._last_match_run_s = float("-inf")
        self._instant_merge_votes: dict[tuple[int, int], tuple[int, float]] = {}

    def resolve_gid(self, gid: int) -> int:
        """Resolve and path-compress a possibly retired alias."""
        path: list[int] = []
        while gid in self.aliases and self.aliases[gid] != gid:
            path.append(gid)
            gid = self.aliases[gid]
        for old in path:
            self.aliases[old] = gid
        return gid

    def step(
        self,
        observations: list[CameraObservation],
        timestamp_s: float | None = None,
        arrival_timestamp_s: float | None = None,
    ) -> list[GlobalTrack]:
        has_event_time = any(obs.capture_ts_s is not None for obs in observations)
        if self.cfg.timestamp_fusion_enabled and (
            has_event_time or self._event_mode_started or timestamp_s is not None
        ):
            self._event_mode_started = True
            return self._step_event_time(observations, timestamp_s, arrival_timestamp_s)
        return self._step_legacy(observations)

    def _step_event_time(
        self,
        observations: list[CameraObservation],
        timestamp_s: float | None,
        arrival_timestamp_s: float | None,
    ) -> list[GlobalTrack]:
        for track in self.tracks.values():
            track.cameras = set()

        ordered = sorted(
            (obs for obs in observations if obs.capture_ts_s is not None),
            key=lambda obs: float(obs.capture_ts_s),
        )
        if timestamp_s is not None:
            self._max_event_ts_s = max(self._max_event_ts_s, float(timestamp_s))
        if arrival_timestamp_s is not None:
            self._max_arrival_ts_s = max(self._max_arrival_ts_s, float(arrival_timestamp_s))
        if ordered:
            self._max_event_ts_s = max(self._max_event_ts_s, float(ordered[-1].capture_ts_s))
        if np.isfinite(self._max_event_ts_s):
            self.tracklets.advance(self._max_event_ts_s)
        changed_keys: set[TrackletKey] = set()
        accepted: list[CameraObservation] = []
        for obs in ordered:
            event_ts = float(obs.capture_ts_s)
            arrival_ts = float(obs.arrival_ts_s) if obs.arrival_ts_s is not None else event_ts
            self._max_event_ts_s = max(self._max_event_ts_s, event_ts)
            self._max_arrival_ts_s = max(self._max_arrival_ts_s, arrival_ts)
            key = obs.camera, int(obs.local_id)
            stored = self.tracklets.append(
                obs.camera,
                obs.local_id,
                event_ts,
                obs.world_xy,
                obs.conf,
                obs.world_velocity,
            )
            if not stored:
                self.metrics["late_observations_dropped"] += 1
                continue
            accepted.append(obs)
            changed_keys.add(key)
            gid = self.affinity.get(key)
            if gid is None or self.resolve_gid(gid) not in self.tracks:
                gid = self._find_local_reconnect(obs)
                if gid is None:
                    gid = self._birth_provisional(obs)
                    self.metrics["provisional_ids_created"] += 1
                else:
                    self.affinity[key] = gid
                    self.members[gid].add(key)
                    self._update_track(self.tracks[gid], obs)
                    self.metrics["local_fragments_reconnected"] += 1
            else:
                gid = self.resolve_gid(gid)
                self.affinity[key] = gid
                self._update_track(self.tracks[gid], obs)
            self.members[gid].add(key)

        if (
            changed_keys
            and self._max_event_ts_s - self._last_match_run_s
            >= self.cfg.tracklet_match_interval_s
        ):
            self._match_tracklets()
            self._last_match_run_s = self._max_event_ts_s
        if self.cfg.fusion_instant_merge_enabled and accepted:
            self._instant_geometry_merge(accepted)

        self._expire_event_tracks()
        self._canonicalize_affinity()
        self._prune_identity_metadata()
        visible: list[GlobalTrack] = []
        for gid, track in sorted(self.tracks.items()):
            if self.resolve_gid(gid) != gid or not track.confirmed or track.last_seen_ts_s is None:
                continue
            output_age = self._max_arrival_ts_s - (
                track.last_arrival_ts_s
                if track.last_arrival_ts_s is not None else track.last_seen_ts_s
            )
            if output_age <= self.cfg.fusion_output_grace_s + 1e-9:
                track.emitted = True
                track.trail.append(track.world_xy.copy())
                self._trim_trail(track)
                visible.append(track)
        self.metrics["active_global_tracks"] = len(visible)
        self.metrics["tracklet_points"] = self.tracklets.metrics["points_stored"]
        self.metrics["tracklet_bytes"] = self.tracklets.approximate_bytes
        return visible

    def _birth_provisional(self, obs: CameraObservation) -> int:
        gid = self._next_gid
        self._next_gid += 1
        event_ts = float(obs.capture_ts_s)
        track = GlobalTrack(
            gid=gid,
            kf=ConstantVelocityKalman(
                np.asarray(obs.world_xy, dtype=float),
                process_noise=self.cfg.fusion_kf_process_noise,
                measurement_noise=self.cfg.fusion_kf_measurement_noise,
            ),
            confirmed=self.cfg.fusion_min_hits <= 1,
            cameras={obs.camera},
            created_at_s=event_ts,
            state_timestamp_s=event_ts,
            last_seen_ts_s=event_ts,
            last_arrival_ts_s=(
                float(obs.arrival_ts_s) if obs.arrival_ts_s is not None else event_ts
            ),
        )
        self.tracks[gid] = track
        key = obs.camera, int(obs.local_id)
        self.affinity[key] = gid
        self.members[gid].add(key)
        return gid

    def _find_local_reconnect(self, obs: CameraObservation) -> int | None:
        """Reconnect a new local ID to a recently ended track in that camera."""
        event_ts = float(obs.capture_ts_s)
        spatial_candidates: list[tuple[float, int, np.ndarray]] = []
        for gid, track in self.tracks.items():
            gid = self.resolve_gid(gid)
            if gid not in self.tracks or obs.camera in track.cameras:
                continue
            same_camera_histories = [
                self.tracklets.get(key)
                for key in self.members[gid]
                if key[0] == obs.camera
            ]
            same_camera_histories = [history for history in same_camera_histories if history]
            if not same_camera_histories:
                continue
            latest_end = max(history.end_s for history in same_camera_histories)
            gap_s = event_ts - latest_end
            if gap_s <= 1e-6 or gap_s > self.cfg.fusion_max_age_s:
                continue
            state_ts = track.state_timestamp_s if track.state_timestamp_s is not None else event_ts
            prediction_dt = max(0.0, event_ts - state_ts)
            predicted_xy = track.world_xy + track.kf.velocity * prediction_dt
            distance = float(np.linalg.norm(predicted_xy - obs.world_xy))
            if distance > self.cfg.fusion_local_reconnect_gate_m:
                continue
            spatial_candidates.append((distance, gid, track.kf.velocity.copy()))
        candidates: list[tuple[float, float, int]] = []
        for distance, gid, track_velocity in spatial_candidates:
            motion_cost = self._velocity_pair_cost(
                track_velocity,
                obs.world_velocity,
                require_evidence=False,
                weight_s=self.cfg.fusion_local_reconnect_velocity_weight_s,
            )
            if motion_cost is None:
                self.metrics["local_reconnect_motion_gate_failed"] += 1
                continue
            if motion_cost > 0:
                self.metrics["local_reconnect_motion_penalties"] += 1
            candidates.append((distance + motion_cost, distance, gid))
        if not candidates:
            return None
        candidates.sort()
        best_cost, _best_distance, best_gid = candidates[0]
        if len(candidates) > 1:
            margin = candidates[1][0] - best_cost
            if margin + 1e-9 < self.cfg.fusion_instant_margin_m:
                self.metrics["local_reconnect_ambiguous"] += 1
                return None
        return best_gid

    def _update_track(self, track: GlobalTrack, obs: CameraObservation) -> None:
        event_ts = float(obs.capture_ts_s)
        if track.state_timestamp_s is None or event_ts >= track.state_timestamp_s - 1e-9:
            dt = 0.0 if track.state_timestamp_s is None else max(0.0, event_ts - track.state_timestamp_s)
            track.kf.predict(dt)
            track.kf.update(np.asarray(obs.world_xy, dtype=float))
            track.state_timestamp_s = event_ts
        else:
            # Keep late metadata for reassociation, but never rewind the live KF.
            self.metrics["out_of_sequence_kf_updates_skipped"] += 1
        track.last_seen_ts_s = max(track.last_seen_ts_s or event_ts, event_ts)
        arrival_ts = float(obs.arrival_ts_s) if obs.arrival_ts_s is not None else event_ts
        track.last_arrival_ts_s = max(track.last_arrival_ts_s or arrival_ts, arrival_ts)
        track.hits += 1
        track.misses = 0
        track.cameras.add(obs.camera)
        if track.hits >= self.cfg.fusion_min_hits:
            track.confirmed = True

    def _match_tracklets(self) -> None:
        by_camera: defaultdict[str, list] = defaultdict(list)
        for history in self.tracklets.eligible():
            if history.key in self.affinity:
                by_camera[history.camera].append(history)

        for left_camera, right_camera in combinations(sorted(by_camera), 2):
            left_items = by_camera[left_camera]
            right_items = by_camera[right_camera]
            if self.cfg.single_person_mode:
                left_items = [max(left_items, key=lambda history: history.duration_s)]
                right_items = [max(right_items, key=lambda history: history.duration_s)]
            pair_time = max(
                max(history.end_s for history in left_items),
                max(history.end_s for history in right_items),
            )
            center = self.clock_estimator.relative_shift(left_camera, right_camera, pair_time)
            left_clock = self.clock_estimator.estimate(left_camera)
            right_clock = self.clock_estimator.estimate(right_camera)
            clocks_known = (
                left_clock.confidence >= self.cfg.offset_min_confidence
                and right_clock.confidence >= self.cfg.offset_min_confidence
            )
            radius = self.cfg.offset_refine_window_s if clocks_known else self.cfg.max_camera_offset_s
            results = self.matcher.assign(left_items, right_items, center, radius)
            if results:
                median_delta = float(np.median([result.delta_s for result in results]))
                representative = min(
                    results, key=lambda result: abs(result.delta_s - median_delta)
                )
                aggregate = MatchResult(**{
                    **representative.__dict__,
                    "delta_s": median_delta,
                    "confidence": float(np.median([
                        result.confidence for result in results
                    ])),
                })
                # One robust camera-pair sample per matching cycle. Multiple
                # people in one frame are correlated evidence, not independent
                # clock measurements.
                self._learn_pair_offset(left_camera, right_camera, aggregate)
            for result in results:
                left_gid = self.resolve_gid(self.affinity[result.left_key])
                right_gid = self.resolve_gid(self.affinity[result.right_key])
                if left_gid == right_gid:
                    continue
                if self.merge_gids(left_gid, right_gid):
                    self.metrics["trajectory_matches_accepted"] += 1

    def _learn_pair_offset(
        self,
        left_camera: str,
        right_camera: str,
        result: MatchResult,
    ) -> None:
        left_history = self.tracklets.get(result.left_key)
        right_history = self.tracklets.get(result.right_key)
        if left_camera == self.clock_estimator.reference_camera:
            self.clock_estimator.observe(
                right_camera, right_history.end_s, result.delta_s, result.confidence
            )
        elif right_camera == self.clock_estimator.reference_camera:
            self.clock_estimator.observe(
                left_camera, left_history.end_s, -result.delta_s, result.confidence
            )
        else:
            left_estimate = self.clock_estimator.estimate(left_camera)
            right_estimate = self.clock_estimator.estimate(right_camera)
            if left_estimate.confidence >= right_estimate.confidence:
                correction = self.clock_estimator.correction_at(left_camera, left_history.end_s)
                self.clock_estimator.observe(
                    right_camera, right_history.end_s, correction + result.delta_s, result.confidence
                )
            else:
                correction = self.clock_estimator.correction_at(right_camera, right_history.end_s)
                self.clock_estimator.observe(
                    left_camera, left_history.end_s, correction - result.delta_s, result.confidence
                )

    def _instant_geometry_merge(self, observations: list[CameraObservation]) -> None:
        """Merge short/noisy tracks after camera clocks share one event time.

        Trajectory evidence remains the bootstrap and safety layer. Once a
        camera clock is confident (or raw capture times already agree), this
        inexpensive path recovers the strong precision of synchronous
        geometry without ever comparing delayed frames as simultaneous.
        """
        by_camera: defaultdict[str, list[CameraObservation]] = defaultdict(list)
        for observation in observations:
            by_camera[observation.camera].append(observation)
        for left_camera, right_camera in combinations(sorted(by_camera), 2):
            left_items = by_camera[left_camera]
            right_items = by_camera[right_camera]
            left_near_peer = self._near_same_camera_peer(left_items)
            right_near_peer = self._near_same_camera_peer(right_items)
            left_clock = self.clock_estimator.estimate(left_camera)
            right_clock = self.clock_estimator.estimate(right_camera)
            clocks_known = (
                left_clock.confidence >= self.cfg.offset_min_confidence
                and right_clock.confidence >= self.cfg.offset_min_confidence
            )
            spatial = np.full((len(left_items), len(right_items)), 1e6, dtype=float)
            for row, left in enumerate(left_items):
                for column, right in enumerate(right_items):
                    left_ts = float(left.capture_ts_s)
                    right_ts = float(right.capture_ts_s)
                    if clocks_known:
                        left_ts += self.clock_estimator.correction_at(left_camera, left_ts)
                        right_ts += self.clock_estimator.correction_at(right_camera, right_ts)
                    if abs(left_ts - right_ts) > self.cfg.fusion_event_time_gate_s:
                        continue
                    world_distance = float(np.linalg.norm(left.world_xy - right.world_xy))
                    if world_distance > self.cfg.fusion_merge_m:
                        continue
                    spatial[row, column] = world_distance

            # Motion is mandatory only where position offers competing
            # assignments. Clear one-to-one pairs still merge immediately;
            # this prevents provisional IDs from being displayed and renamed
            # later merely because another person is nearby in the scene.
            row_candidates = np.sum(spatial < 1e6, axis=1)
            column_candidates = np.sum(spatial < 1e6, axis=0)
            cost = np.full_like(spatial, 1e6)
            ambiguous_assignment = np.zeros_like(spatial, dtype=bool)
            for row, left in enumerate(left_items):
                for column, right in enumerate(right_items):
                    if spatial[row, column] >= 1e6:
                        continue
                    requires_motion = bool(
                        left_near_peer[row]
                        or right_near_peer[column]
                        or row_candidates[row] > 1
                        or column_candidates[column] > 1
                    )
                    ambiguous_assignment[row, column] = requires_motion
                    motion_cost = 0.0
                    if self.cfg.fusion_instant_velocity_weight_s > 0:
                        motion_cost = self._velocity_pair_cost(
                            self._stable_observation_velocity(left),
                            self._stable_observation_velocity(right),
                            require_evidence=False,
                        )
                        if motion_cost is None:
                            self.metrics["instant_motion_gate_failed"] += 1
                            continue
                        if motion_cost > 0:
                            self.metrics["instant_motion_penalties"] += 1
                    cost[row, column] = spatial[row, column] + motion_cost
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns):
                value = cost[row, column]
                if value >= 1e6:
                    continue
                alternatives = [
                    cost[row, other]
                    for other in range(cost.shape[1]) if other != column
                ] + [
                    cost[other, column]
                    for other in range(cost.shape[0]) if other != row
                ]
                margin = min(alternatives, default=float("inf")) - value
                if margin + 1e-9 < self.cfg.fusion_instant_margin_m:
                    self.metrics["instant_geometry_ambiguous"] += 1
                    continue
                if ambiguous_assignment[row, column] and self._strongly_opposite_motion(
                    left_items[row], right_items[column]
                ):
                    # Do not immediately try the second-best spatial pairing.
                    # Waiting until the people separate is safer than making an
                    # irreversible cross-person merge during this frame.
                    self.metrics["instant_opposite_direction_vetoes"] += 1
                    continue
                left_key = left_camera, left_items[row].local_id
                right_key = right_camera, right_items[column].local_id
                left_gid = self.resolve_gid(self.affinity[left_key])
                right_gid = self.resolve_gid(self.affinity[right_key])
                if left_gid == right_gid:
                    continue
                pair = tuple(sorted((left_gid, right_gid)))
                evidence_ts = max(
                    float(left_items[row].capture_ts_s),
                    float(right_items[column].capture_ts_s),
                )
                count, previous_ts = self._instant_merge_votes.get(
                    pair, (0, float("-inf"))
                )
                if evidence_ts - previous_ts > self.cfg.fusion_instant_confirmation_window_s:
                    count = 0
                count += 1
                self._instant_merge_votes[pair] = (count, evidence_ts)
                confirmations = (
                    max(1, self.cfg.fusion_instant_confirmations)
                    if ambiguous_assignment[row, column]
                    else 1
                )
                if count < confirmations:
                    self.metrics["instant_merges_deferred_for_confirmation"] += 1
                    continue
                if self.merge_gids(left_gid, right_gid):
                    for candidate in list(self._instant_merge_votes):
                        if left_gid in candidate or right_gid in candidate:
                            del self._instant_merge_votes[candidate]
                    canonical = self.tracks[self.resolve_gid(left_gid)]
                    # Late geometry can repair identity, but must not apply an
                    # older measurement to the current Kalman state.
                    measurement_ts = min(float(left_items[row].capture_ts_s),
                                         float(right_items[column].capture_ts_s))
                    if canonical.state_timestamp_s is None or measurement_ts >= canonical.state_timestamp_s - 1e-9:
                        canonical.kf.update(
                            (left_items[row].world_xy + right_items[column].world_xy) / 2.0
                        )
                    self.metrics["instant_geometry_merges"] += 1

        if np.isfinite(self._max_event_ts_s):
            cutoff = (
                self._max_event_ts_s
                - self.cfg.fusion_instant_confirmation_window_s
            )
            for pair, (_, seen_ts) in list(self._instant_merge_votes.items()):
                if seen_ts < cutoff:
                    del self._instant_merge_votes[pair]

    def _near_same_camera_peer(
        self, observations: list[CameraObservation]
    ) -> list[bool]:
        """Mark observations whose identity is position-ambiguous right now."""
        count = len(observations)
        if count < 2 or self.cfg.fusion_crossing_guard_m <= 0:
            return [False] * count
        xy = np.stack([np.asarray(obs.world_xy, dtype=float) for obs in observations])
        distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
        np.fill_diagonal(distance, np.inf)
        return list(np.min(distance, axis=1) <= self.cfg.fusion_crossing_guard_m)

    def _instant_motion_cost(
        self, left: CameraObservation, right: CameraObservation
    ) -> float | None:
        """Return a motion penalty, or reject a direction-incompatible pair."""
        return self._velocity_pair_cost(
            self._stable_observation_velocity(left),
            self._stable_observation_velocity(right),
        )

    def _strongly_opposite_motion(
        self, left: CameraObservation, right: CameraObservation
    ) -> bool:
        left_velocity = self._stable_observation_velocity(left)
        right_velocity = self._stable_observation_velocity(right)
        if left_velocity is None or right_velocity is None:
            return False
        left_velocity = np.asarray(left_velocity, dtype=float)
        right_velocity = np.asarray(right_velocity, dtype=float)
        if not np.all(np.isfinite(left_velocity)) or not np.all(np.isfinite(right_velocity)):
            return False
        left_speed = float(np.linalg.norm(left_velocity))
        right_speed = float(np.linalg.norm(right_velocity))
        if min(left_speed, right_speed) < self.cfg.fusion_instant_min_speed_mps:
            return False
        cosine = float(
            np.dot(left_velocity, right_velocity) / (left_speed * right_speed)
        )
        return cosine < self.cfg.fusion_instant_opposite_direction_cosine

    def _stable_observation_velocity(
        self, observation: CameraObservation
    ) -> np.ndarray | None:
        """Estimate motion over a short path instead of trusting one frame.

        A local ID can briefly jitter or swap at an occlusion. Its instantaneous
        finite-difference velocity then points toward the other person exactly
        when fusion needs motion most. The displacement over a bounded history
        window preserves the pre-crossing direction for a few frames and acts
        as inexpensive trajectory hysteresis.
        """
        history = self.tracklets.get((observation.camera, int(observation.local_id)))
        if history is None or len(history.points) < 2:
            return observation.world_velocity
        event_ts = float(observation.capture_ts_s)
        points = [
            point for point in history.points
            if point.timestamp_s <= event_ts + 1e-9
        ]
        if len(points) < 2:
            return observation.world_velocity
        latest = points[-1]
        minimum_span = max(0.0, self.cfg.fusion_motion_min_span_s)
        candidates = [
            point for point in points[:-1]
            if latest.timestamp_s - point.timestamp_s >= minimum_span - 1e-9
        ]
        if not candidates:
            return observation.world_velocity
        target_ts = latest.timestamp_s - max(
            minimum_span, self.cfg.fusion_motion_window_s
        )
        reference = min(
            candidates,
            key=lambda point: abs(point.timestamp_s - target_ts),
        )
        span_s = latest.timestamp_s - reference.timestamp_s
        if span_s <= 1e-9:
            return observation.world_velocity
        velocity = (latest.xy - reference.xy) / span_s
        if not np.all(np.isfinite(velocity)):
            return observation.world_velocity
        self.metrics["stable_motion_estimates_used"] += 1
        return velocity

    def _velocity_pair_cost(
        self,
        left_velocity_value: np.ndarray | None,
        right_velocity_value: np.ndarray | None,
        *,
        require_evidence: bool = False,
        weight_s: float | None = None,
    ) -> float | None:
        if left_velocity_value is None or right_velocity_value is None:
            return None if require_evidence else 0.0
        left_velocity = np.asarray(left_velocity_value, dtype=float)
        right_velocity = np.asarray(right_velocity_value, dtype=float)
        if not np.all(np.isfinite(left_velocity)) or not np.all(np.isfinite(right_velocity)):
            return None if require_evidence else 0.0
        left_speed = float(np.linalg.norm(left_velocity))
        right_speed = float(np.linalg.norm(right_velocity))
        if min(left_speed, right_speed) < self.cfg.fusion_instant_min_speed_mps:
            return None if require_evidence else 0.0
        velocity_error = float(np.linalg.norm(left_velocity - right_velocity))
        direction_cosine = float(
            np.dot(left_velocity, right_velocity) / (left_speed * right_speed)
        )
        # Use motion as a soft assignment preference. A hard rejection causes
        # fragmentation when homography-derived velocity is noisy; a capped
        # penalty still makes the same-direction candidate win at crossings
        # while preserving the sole spatially plausible match.
        clipped_velocity_error = min(
            velocity_error,
            max(self.cfg.fusion_instant_velocity_gate_mps * 2.0, 1e-6),
        )
        direction_penalty = max(
            0.0,
            self.cfg.fusion_instant_direction_cosine - direction_cosine,
        )
        if weight_s is None:
            weight_s = self.cfg.fusion_instant_velocity_weight_s
        return weight_s * (
            clipped_velocity_error + direction_penalty
        )

    def merge_gids(self, first_gid: int, second_gid: int) -> bool:
        """Merge IDs into the oldest canonical ID; safe and idempotent."""
        first = self.resolve_gid(first_gid)
        second = self.resolve_gid(second_gid)
        if first == second:
            return True
        if first not in self.tracks or second not in self.tracks:
            return False
        if self._has_same_camera_conflict(first, second):
            self.metrics["same_camera_merge_conflicts"] += 1
            return False

        canonical = min((first, second), key=self._canonical_merge_priority)
        retired = second if canonical == first else first
        keep = self.tracks[canonical]
        lose = self.tracks[retired]
        was_late = keep.emitted or lose.emitted
        if keep.emitted and not lose.emitted:
            self.metrics["published_id_preserved_on_merge"] += 1
        if (
            lose.state_timestamp_s is not None
            and (keep.state_timestamp_s is None or lose.state_timestamp_s > keep.state_timestamp_s)
        ):
            keep.kf.x = lose.kf.x.copy()
            keep.kf.P = lose.kf.P.copy()
            keep.state_timestamp_s = lose.state_timestamp_s
        keep.last_seen_ts_s = max(
            value for value in (keep.last_seen_ts_s, lose.last_seen_ts_s) if value is not None
        )
        keep.last_arrival_ts_s = max(
            value
            for value in (keep.last_arrival_ts_s, lose.last_arrival_ts_s)
            if value is not None
        )
        keep.created_at_s = min(
            value for value in (keep.created_at_s, lose.created_at_s) if value is not None
        )
        keep.hits += lose.hits
        keep.confirmed = keep.confirmed or lose.confirmed or keep.hits >= self.cfg.fusion_min_hits
        keep.cameras.update(lose.cameras)
        keep.trail.extend(lose.trail)
        self._trim_trail(keep)

        self.aliases[retired] = canonical
        self._alias_created_s[retired] = self._max_event_ts_s
        self.members[canonical].update(self.members.pop(retired, set()))
        del self.tracks[retired]
        self._canonicalize_affinity()
        self.metrics["global_id_merges"] += 1
        if was_late:
            self.metrics["late_reassociations"] += 1
        return True

    def _canonical_merge_priority(self, gid: int) -> tuple[bool, float, int]:
        """Prefer an already published ID, then the oldest track.

        Merging two provisional identities is internal bookkeeping. Once one
        identity has reached an external consumer, however, renaming it causes
        a visible Global ID jump even if the reassociation itself is correct.
        """
        track = self.tracks[gid]
        created_at = (
            float(track.created_at_s)
            if track.created_at_s is not None
            else float("inf")
        )
        return (not track.emitted, created_at, gid)

    def _trim_trail(self, track: GlobalTrack) -> None:
        if self.cfg.trail_length <= 0:
            track.trail.clear()
        else:
            del track.trail[:-self.cfg.trail_length]

    def _has_same_camera_conflict(self, first_gid: int, second_gid: int) -> bool:
        for first_key in self.members[first_gid]:
            for second_key in self.members[second_gid]:
                if first_key == second_key or first_key[0] != second_key[0]:
                    continue
                first = self.tracklets.get(first_key)
                second = self.tracklets.get(second_key)
                if first is None or second is None:
                    continue
                overlap = min(first.end_s, second.end_s) - max(first.start_s, second.start_s)
                if overlap >= -1e-9:
                    return True
        return False

    def _canonicalize_affinity(self) -> None:
        for key, gid in list(self.affinity.items()):
            self.affinity[key] = self.resolve_gid(gid)

    def _prune_identity_metadata(self) -> None:
        """Retain retired aliases only for the supported late-packet window.

        Active local IDs already map directly to canonical IDs. An external
        consumer needing archival alias resolution must persist merge events.
        """
        for key, gid in list(self.affinity.items()):
            if self.tracklets.get(key) is None or gid not in self.tracks:
                del self.affinity[key]
                self.members[gid].discard(key)
        for gid in list(self.members):
            if gid not in self.tracks:
                del self.members[gid]
        # Compress all paths before deleting intermediate aliases.
        for gid in list(self.aliases):
            self.resolve_gid(gid)
        cutoff = self._max_event_ts_s - max(self.cfg.tracklet_history_s, self.cfg.tracklet_late_drop_s)
        for gid, created in list(self._alias_created_s.items()):
            if created < cutoff or self.aliases.get(gid) not in self.tracks:
                self.aliases.pop(gid, None)
                del self._alias_created_s[gid]

    def _expire_event_tracks(self) -> None:
        if not np.isfinite(self._max_event_ts_s):
            return
        retention_s = max(self.cfg.fusion_max_age_s, self.cfg.tracklet_history_s)
        for gid in list(self.tracks):
            track = self.tracks[gid]
            if track.last_seen_ts_s is None:
                continue
            if self._max_event_ts_s - track.last_seen_ts_s <= retention_s:
                continue
            del self.tracks[gid]
            self.members.pop(gid, None)
            for key, mapped_gid in list(self.affinity.items()):
                if self.resolve_gid(mapped_gid) == gid:
                    del self.affinity[key]
            self.metrics["global_tracks_expired"] += 1

    def metrics_snapshot(self) -> dict[str, int | float | dict[str, dict[str, float]]]:
        snapshot: dict = dict(self.metrics)
        snapshot.update({
            "tracklet_late_dropped": self.tracklets.metrics["late_dropped"],
            "tracklet_out_of_order_accepted": self.tracklets.metrics["out_of_order_accepted"],
            "tracklet_points": self.tracklets.metrics["points_stored"],
            "tracklet_bytes": self.tracklets.approximate_bytes,
            "retained_global_tracks": len(self.tracks),
            "retained_aliases": len(self.aliases),
            "retained_affinities": len(self.affinity),
            "clock_estimates": {
                camera: {
                    "offset_s": estimate.offset_s,
                    "scale": estimate.scale,
                    "confidence": estimate.confidence,
                }
                for camera in self.cfg.cameras
                for estimate in [self.clock_estimator.estimate(camera)]
            },
            "matcher": dict(self.matcher.metrics),
        })
        return snapshot

    # ------------------------------------------------------------------
    # Original synchronous/tick path, retained as an explicit fallback.
    def _step_legacy(self, observations: list[CameraObservation]) -> list[GlobalTrack]:
        cfg = self.cfg
        for track in self.tracks.values():
            track.kf.predict()
            track.cameras = set()

        assigned: dict[int, list[CameraObservation]] = {}
        pending: list[CameraObservation] = []
        for obs in observations:
            gid = self.affinity.get((obs.camera, obs.local_id))
            if gid is not None:
                gid = self.resolve_gid(gid)
            if gid is not None and gid in self.tracks:
                drift = np.linalg.norm(self.tracks[gid].world_xy - obs.world_xy)
                if drift < 2 * cfg.fusion_gate_m:
                    assigned.setdefault(gid, []).append(obs)
                    continue
                del self.affinity[(obs.camera, obs.local_id)]
            pending.append(obs)

        clusters = self._cluster(pending)
        gids = list(self.tracks)
        if clusters and gids:
            predicted = np.stack([self.tracks[gid].world_xy for gid in gids])
            centroids = np.stack([
                np.mean([obs.world_xy for obs in cluster], axis=0) for cluster in clusters
            ])
            distance = np.linalg.norm(predicted[:, None] - centroids[None, :], axis=2)
            rows, columns = linear_sum_assignment(distance)
            for row, column in zip(rows, columns):
                if distance[row, column] < cfg.fusion_gate_m:
                    taken = {obs.camera for obs in assigned.get(gids[row], [])}
                    novel = [obs for obs in clusters[column] if obs.camera not in taken]
                    if novel:
                        assigned.setdefault(gids[row], []).extend(novel)
                    clusters[column] = None
            clusters = [cluster for cluster in clusters if cluster is not None]

        for gid, obs_list in assigned.items():
            track = self.tracks[gid]
            track.kf.update(np.mean([obs.world_xy for obs in obs_list], axis=0))
            track.hits += 1
            track.misses = 0
            if track.hits >= cfg.fusion_min_hits:
                track.confirmed = True
            for obs in obs_list:
                track.cameras.add(obs.camera)
                key = obs.camera, obs.local_id
                self.affinity[key] = gid
                self.members[gid].add(key)

        for cluster in clusters:
            centroid = np.mean([obs.world_xy for obs in cluster], axis=0)
            gid = self._next_gid
            self._next_gid += 1
            track = GlobalTrack(
                gid=gid,
                kf=ConstantVelocityKalman(
                    centroid,
                    process_noise=cfg.fusion_kf_process_noise,
                    measurement_noise=cfg.fusion_kf_measurement_noise,
                ),
                cameras={obs.camera for obs in cluster},
            )
            self.tracks[gid] = track
            for obs in cluster:
                key = obs.camera, obs.local_id
                self.affinity[key] = gid
                self.members[gid].add(key)

        for gid in list(self.tracks):
            track = self.tracks[gid]
            if not track.cameras:
                track.misses += 1
                if track.misses > (cfg.fusion_max_age if track.confirmed else 0):
                    del self.tracks[gid]
                    self.members.pop(gid, None)
                    self.affinity = {key: value for key, value in self.affinity.items() if value != gid}

        visible = [track for track in self.tracks.values() if track.confirmed and track.misses == 0]
        for track in visible:
            track.trail.append(track.world_xy.copy())
            self._trim_trail(track)
        return visible

    def _cluster(self, observations: list[CameraObservation]) -> list[list[CameraObservation]]:
        clusters = [[obs] for obs in observations]
        merged = True
        while merged:
            merged = False
            for index in range(len(clusters)):
                for other in range(index + 1, len(clusters)):
                    first, second = clusters[index], clusters[other]
                    cameras = {obs.camera for obs in first}
                    if any(obs.camera in cameras for obs in second):
                        continue
                    distance = np.linalg.norm(
                        np.mean([obs.world_xy for obs in first], axis=0)
                        - np.mean([obs.world_xy for obs in second], axis=0)
                    )
                    if distance < self.cfg.fusion_merge_m:
                        clusters[index] = first + second
                        del clusters[other]
                        merged = True
                        break
                if merged:
                    break
        return clusters


def foot_point_xyxy(bbox_xyxy: np.ndarray) -> np.ndarray:
    """Bottom-center pixel of an xyxy box."""
    x1, _y1, x2, y2 = bbox_xyxy[:4]
    return np.array([(x1 + x2) / 2, y2])
