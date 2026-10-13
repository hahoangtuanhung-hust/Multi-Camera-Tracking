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
    near_boundary: bool = False


@dataclass
class TrackEndpoint:
    """Latest compact observation retained for safe fragment reconnection."""

    timestamp_s: float
    xy: np.ndarray
    velocity: np.ndarray | None
    near_boundary: bool


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
    emitted_at_s: float | None = None
    camera_endpoints: dict[str, TrackEndpoint] = field(default_factory=dict)

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
        self._duplicate_merge_votes: dict[tuple[int, int], tuple[int, float]] = {}
        self._duplicate_suppressed_gids: set[int] = set()
        self._fragment_reconnect_votes: dict[
            TrackletKey, tuple[int, int, float]
        ] = {}

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
        self._duplicate_suppressed_gids.clear()
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
        records: list[tuple[CameraObservation, TrackletKey, int, bool]] = []
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
            is_new = gid is None or self.resolve_gid(gid) not in self.tracks
            if is_new:
                gid = self._birth_provisional(obs)
                self.metrics["provisional_ids_created"] += 1
            else:
                gid = self.resolve_gid(gid)
            records.append((obs, key, gid, is_new))

        active_by_camera: defaultdict[str, set[int]] = defaultdict(set)
        for obs, _key, gid, _is_new in records:
            active_by_camera[obs.camera].add(self.resolve_gid(gid))
        reconnects = self._batch_local_reconnect(records, active_by_camera)

        for obs, key, initial_gid, is_new in records:
            event_ts = float(obs.capture_ts_s)
            gid = self.resolve_gid(initial_gid)
            reconnect = reconnects.get(key)
            if reconnect is not None:
                candidate, requires_confirmation = reconnect
                candidate = self.resolve_gid(candidate)
                if (
                    not requires_confirmation
                    and key not in self._fragment_reconnect_votes
                ):
                    gid = self._commit_fragment_reconnect(key, gid, candidate)
                else:
                    reconnected = self._record_fragment_reconnect_vote(
                        key, gid, candidate, event_ts
                    )
                    if reconnected is not None:
                        gid = reconnected
            elif key in self._fragment_reconnect_votes:
                del self._fragment_reconnect_votes[key]

            gid = self.resolve_gid(gid)
            self.affinity[key] = gid
            if not is_new:
                self._update_track(self.tracks[gid], obs)
            self.members[gid].add(key)

        self.metrics["peak_retained_global_tracks"] = max(
            int(self.metrics["peak_retained_global_tracks"]), len(self.tracks)
        )

        if (
            changed_keys
            and self._max_event_ts_s - self._last_match_run_s
            >= self.cfg.tracklet_match_interval_s
        ):
            self._match_tracklets()
            self._last_match_run_s = self._max_event_ts_s
        if self.cfg.fusion_instant_merge_enabled and accepted:
            self._instant_geometry_merge(accepted)
        self._consolidate_duplicate_tracks()

        self._expire_event_tracks()
        self._canonicalize_affinity()
        self._prune_identity_metadata()
        visible: list[GlobalTrack] = []
        for gid, track in sorted(self.tracks.items()):
            if self.resolve_gid(gid) != gid or not track.confirmed or track.last_seen_ts_s is None:
                continue
            if gid in self._duplicate_suppressed_gids:
                self.metrics["duplicate_outputs_suppressed"] += 1
                continue
            if (
                self.cfg.fusion_provisional_buffer_enabled
                and track.created_at_s is not None
                and self._max_event_ts_s - track.created_at_s
                + 1e-9 < self.cfg.fusion_provisional_min_age_s
            ):
                self.metrics["provisional_publication_deferred"] += 1
                continue
            output_age = self._max_arrival_ts_s - (
                track.last_arrival_ts_s
                if track.last_arrival_ts_s is not None else track.last_seen_ts_s
            )
            if output_age <= self.cfg.fusion_output_grace_s + 1e-9:
                if not track.emitted:
                    track.emitted_at_s = self._max_event_ts_s
                    self.metrics["provisional_ids_published"] += 1
                    self.metrics["unique_published_gids"] += 1
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
            camera_endpoints={obs.camera: self._endpoint_from_observation(obs)},
        )
        self.tracks[gid] = track
        key = obs.camera, int(obs.local_id)
        self.affinity[key] = gid
        self.members[gid].add(key)
        return gid

    @staticmethod
    def _endpoint_from_observation(obs: CameraObservation) -> TrackEndpoint:
        velocity = (
            None
            if obs.world_velocity is None
            else np.asarray(obs.world_velocity, dtype=float).copy()
        )
        return TrackEndpoint(
            timestamp_s=float(obs.capture_ts_s),
            xy=np.asarray(obs.world_xy, dtype=float).copy(),
            velocity=velocity,
            near_boundary=bool(obs.near_boundary),
        )

    def _find_local_reconnect(
        self,
        obs: CameraObservation,
        *,
        excluded_gid: int | None = None,
        excluded_gids: set[int] | None = None,
    ) -> tuple[int, bool] | None:
        """Find a motion- and covariance-compatible ended same-camera ID."""
        candidates = self._local_reconnect_candidates(
            obs,
            excluded_gid=excluded_gid,
            excluded_gids=excluded_gids,
        )
        if not candidates:
            return None
        candidates.sort()
        best_cost, _best_distance, best_gid, best_is_long = candidates[0]
        if len(candidates) > 1:
            margin = candidates[1][0] - best_cost
            if margin + 1e-9 < self.cfg.fusion_instant_margin_m:
                self.metrics["local_reconnect_ambiguous"] += 1
                return None
        requires_confirmation = best_is_long or len(candidates) > 1
        return best_gid, requires_confirmation

    def _local_reconnect_candidates(
        self,
        obs: CameraObservation,
        *,
        excluded_gid: int | None = None,
        excluded_gids: set[int] | None = None,
    ) -> list[tuple[float, float, int, bool]]:
        """Return every statistically plausible same-camera predecessor."""
        event_ts = float(obs.capture_ts_s)
        excluded = excluded_gids or set()
        observation_velocity = self._stable_observation_velocity(obs)
        candidates: list[tuple[float, float, int, bool]] = []
        for gid, track in self.tracks.items():
            gid = self.resolve_gid(gid)
            if (
                gid not in self.tracks
                or gid == excluded_gid
                or gid in excluded
                or obs.camera in track.cameras
            ):
                continue
            endpoint = track.camera_endpoints.get(obs.camera)
            if endpoint is None:
                continue
            gap_s = event_ts - endpoint.timestamp_s
            if gap_s <= 1e-6 or gap_s > self.cfg.fusion_local_reconnect_long_window_s:
                continue
            is_long_gap = gap_s > self.cfg.fusion_max_age_s
            if is_long_gap and not (
                endpoint.near_boundary and obs.near_boundary
            ):
                self.metrics["long_reconnect_boundary_rejected"] += 1
                continue

            state_ts = (
                track.state_timestamp_s
                if track.state_timestamp_s is not None
                else endpoint.timestamp_s
            )
            prediction_dt = max(0.0, event_ts - state_ts)
            predicted_xy, innovation_covariance = track.kf.projected_measurement(
                prediction_dt
            )
            if (
                gap_s > self.cfg.fusion_local_reconnect_window_s
                and endpoint.velocity is not None
                and np.all(np.isfinite(endpoint.velocity))
            ):
                # For a long hidden interval the last camera-track velocity is
                # usually better established than the global KF's early
                # velocity state. Keep the KF covariance as uncertainty, but
                # extrapolate the mean from that stable endpoint trajectory.
                predicted_xy = endpoint.xy + endpoint.velocity * gap_s
            innovation = np.asarray(obs.world_xy, dtype=float) - predicted_xy
            distance = float(np.linalg.norm(innovation))
            if distance > self.cfg.fusion_local_reconnect_gate_m:
                continue
            nis = 0.0
            if self.cfg.fusion_local_reconnect_covariance_enabled:
                try:
                    nis = float(
                        innovation
                        @ np.linalg.solve(innovation_covariance, innovation)
                    )
                except np.linalg.LinAlgError:
                    self.metrics["local_reconnect_singular_covariance"] += 1
                    continue
                if (
                    not np.isfinite(nis)
                    or nis > self.cfg.fusion_local_reconnect_nis_gate
                ):
                    self.metrics["local_reconnect_covariance_gate_failed"] += 1
                    continue

            motion_cost = self._velocity_pair_cost(
                endpoint.velocity,
                observation_velocity,
                require_evidence=is_long_gap,
                weight_s=self.cfg.fusion_local_reconnect_velocity_weight_s,
            )
            if motion_cost is None:
                self.metrics["local_reconnect_motion_gate_failed"] += 1
                continue
            if is_long_gap and not self._reconnect_motion_compatible(
                endpoint.velocity, observation_velocity
            ):
                self.metrics["long_reconnect_motion_rejected"] += 1
                continue
            if motion_cost > 0:
                self.metrics["local_reconnect_motion_penalties"] += 1
            # Distance remains in metres for the ambiguity margin. NIS is a
            # small tie-breaker that favours the more certain prediction.
            cost = distance + motion_cost + 0.01 * np.sqrt(max(nis, 0.0))
            candidates.append((cost, distance, gid, is_long_gap))
        return candidates

    def _batch_local_reconnect(
        self,
        records: list[tuple[CameraObservation, TrackletKey, int, bool]],
        active_by_camera: dict[str, set[int]],
    ) -> dict[TrackletKey, tuple[int, bool]]:
        """Assign ended fragments to canonical GIDs without input-order bias."""
        eligible = [
            record
            for record in records
            if record[3]
            or record[1] in self._fragment_reconnect_votes
            or not self.tracks[self.resolve_gid(record[2])].emitted
        ]
        if not eligible:
            return {}

        if not self.cfg.fusion_batch_reconnect_enabled:
            result: dict[TrackletKey, tuple[int, bool]] = {}
            claimed: set[int] = set()
            for obs, key, gid, _is_new in eligible:
                match = self._find_local_reconnect(
                    obs,
                    excluded_gid=self.resolve_gid(gid),
                    excluded_gids=claimed | active_by_camera.get(obs.camera, set()),
                )
                if match is not None:
                    result[key] = match
                    claimed.add(self.resolve_gid(match[0]))
            return result

        row_candidates: list[list[tuple[float, float, int, bool]]] = []
        candidate_gids: set[int] = set()
        for obs, _key, gid, _is_new in eligible:
            candidates = self._local_reconnect_candidates(
                obs,
                excluded_gid=self.resolve_gid(gid),
                excluded_gids=active_by_camera.get(obs.camera, set()),
            )
            row_candidates.append(candidates)
            candidate_gids.update(candidate[2] for candidate in candidates)
        if not candidate_gids:
            return {}

        columns = sorted(candidate_gids)
        column_by_gid = {gid: index for index, gid in enumerate(columns)}
        sentinel = 1e6
        matrix = np.full((len(eligible), len(columns)), sentinel, dtype=float)
        details: dict[tuple[int, int], tuple[float, float, int, bool]] = {}
        for row, candidates in enumerate(row_candidates):
            for candidate in candidates:
                column = column_by_gid[candidate[2]]
                if candidate[0] < matrix[row, column]:
                    matrix[row, column] = candidate[0]
                    details[row, column] = candidate

        rows, assigned_columns = linear_sum_assignment(matrix)
        assignments: dict[TrackletKey, tuple[int, bool]] = {}
        for row, column in zip(rows, assigned_columns):
            if matrix[row, column] >= sentinel or (row, column) not in details:
                continue
            candidate = details[row, column]
            alternatives = [
                matrix[row, other]
                for other in range(matrix.shape[1])
                if other != column
            ] + [
                matrix[other, column]
                for other in range(matrix.shape[0])
                if other != row
            ]
            margin = min(alternatives, default=float("inf")) - matrix[row, column]
            ambiguous = margin + 1e-9 < self.cfg.fusion_instant_margin_m
            if ambiguous:
                self.metrics["local_reconnect_ambiguous"] += 1
            requires_confirmation = candidate[3] or ambiguous or len(row_candidates[row]) > 1
            assignments[eligible[row][1]] = (candidate[2], requires_confirmation)
            self.metrics["batch_reconnect_assignments"] += 1
        return assignments

    def _reconnect_motion_compatible(
        self,
        previous_velocity: np.ndarray | None,
        current_velocity: np.ndarray | None,
    ) -> bool:
        if previous_velocity is None or current_velocity is None:
            return False
        previous = np.asarray(previous_velocity, dtype=float)
        current = np.asarray(current_velocity, dtype=float)
        if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
            return False
        previous_speed = float(np.linalg.norm(previous))
        current_speed = float(np.linalg.norm(current))
        if min(previous_speed, current_speed) < self.cfg.fusion_instant_min_speed_mps:
            return False
        if (
            np.linalg.norm(previous - current)
            > self.cfg.fusion_local_reconnect_velocity_gate_mps
        ):
            return False
        cosine = float(
            np.dot(previous, current) / (previous_speed * current_speed)
        )
        return cosine >= self.cfg.fusion_local_reconnect_direction_cosine

    def _record_fragment_reconnect_vote(
        self,
        key: TrackletKey,
        current_gid: int,
        candidate_gid: int,
        event_ts: float,
    ) -> int | None:
        previous = self._fragment_reconnect_votes.get(key)
        count = 1
        if previous is not None:
            previous_gid, previous_count, previous_ts = previous
            if (
                self.resolve_gid(previous_gid) == self.resolve_gid(candidate_gid)
                and event_ts - previous_ts
                <= self.cfg.fusion_local_reconnect_confirmation_window_s + 1e-9
            ):
                count = previous_count + 1
        self._fragment_reconnect_votes[key] = (candidate_gid, count, event_ts)
        if count < max(1, self.cfg.fusion_local_reconnect_confirmations):
            self.metrics["local_reconnects_deferred_for_confirmation"] += 1
            return None
        return self._commit_fragment_reconnect(key, current_gid, candidate_gid)

    def _commit_fragment_reconnect(
        self,
        key: TrackletKey,
        current_gid: int,
        candidate_gid: int,
    ) -> int:
        current = self.resolve_gid(current_gid)
        candidate = self.resolve_gid(candidate_gid)
        reconnected_before_publish = (
            current in self.tracks and not self.tracks[current].emitted
        )
        if current != candidate and self.merge_gids(candidate, current):
            canonical = self.resolve_gid(candidate)
            self.affinity[key] = canonical
            self.members[canonical].add(key)
            self.metrics["local_fragments_reconnected"] += 1
            if reconnected_before_publish:
                self.metrics["provisional_ids_reconnected_before_publish"] += 1
        else:
            canonical = self.resolve_gid(current)
        self._fragment_reconnect_votes.pop(key, None)
        return canonical

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
        previous_endpoint = track.camera_endpoints.get(obs.camera)
        if (
            previous_endpoint is None
            or event_ts >= previous_endpoint.timestamp_s - 1e-9
        ):
            track.camera_endpoints[obs.camera] = self._endpoint_from_observation(obs)
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
                left_track = self.tracks[left_gid]
                right_track = self.tracks[right_gid]
                published_pair = left_track.emitted and right_track.emitted
                motion_compatible = self._observation_motion_compatible(
                    left_items[row],
                    right_items[column],
                    self.cfg.fusion_instant_velocity_gate_mps,
                )
                soft_published_motion = False
                if (
                    published_pair
                    and not motion_compatible
                    and self.cfg.fusion_published_motion_gate_enabled
                ):
                    if (
                        not self.cfg.fusion_adaptive_hysteresis_enabled
                        or ambiguous_assignment[row, column]
                    ):
                        self.metrics["published_merge_motion_rejected"] += 1
                        continue
                    # A sole spatial candidate can have noisy homography-derived
                    # velocity.  Require repeated spatial ownership evidence
                    # rather than permanently rejecting it from one sample.
                    soft_published_motion = True
                    self.metrics["published_merge_motion_softened"] += 1
                count = self._record_pair_vote(
                    self._instant_merge_votes,
                    pair,
                    evidence_ts,
                    self.cfg.fusion_instant_confirmation_window_s,
                )
                confirmations = (
                    max(1, self.cfg.fusion_instant_confirmations)
                    if ambiguous_assignment[row, column]
                    else 1
                )
                if (
                    self.cfg.fusion_adaptive_hysteresis_enabled
                    and ambiguous_assignment[row, column]
                ):
                    confirmations = max(confirmations, 2)
                if published_pair:
                    confirmations = max(
                        confirmations,
                        self.cfg.fusion_published_merge_confirmations,
                    )
                if soft_published_motion:
                    confirmations = max(confirmations, 2)
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

    @staticmethod
    def _record_pair_vote(
        votes: dict[tuple[int, int], tuple[int, float]],
        pair: tuple[int, int],
        timestamp_s: float,
        window_s: float,
    ) -> int:
        """Count only consecutive ownership evidence for either GID."""
        for competing in list(votes):
            if competing != pair and set(competing) & set(pair):
                del votes[competing]
        count, previous_ts = votes.get(pair, (0, float("-inf")))
        if timestamp_s - previous_ts > window_s + 1e-9:
            count = 0
        count += 1
        votes[pair] = (count, timestamp_s)
        return count

    def _track_motion_compatible(
        self,
        left: GlobalTrack,
        right: GlobalTrack,
        velocity_gate_mps: float,
    ) -> bool:
        left_velocity = np.asarray(left.kf.velocity, dtype=float)
        right_velocity = np.asarray(right.kf.velocity, dtype=float)
        left_speed = float(np.linalg.norm(left_velocity))
        right_speed = float(np.linalg.norm(right_velocity))
        if min(left_speed, right_speed) < self.cfg.fusion_instant_min_speed_mps:
            return True
        if float(np.linalg.norm(left_velocity - right_velocity)) > velocity_gate_mps:
            return False
        cosine = float(
            np.dot(left_velocity, right_velocity) / (left_speed * right_speed)
        )
        return cosine >= self.cfg.fusion_instant_direction_cosine

    def _observation_motion_compatible(
        self,
        left: CameraObservation,
        right: CameraObservation,
        velocity_gate_mps: float,
    ) -> bool:
        left_value = self._stable_observation_velocity(left)
        right_value = self._stable_observation_velocity(right)
        if left_value is None or right_value is None:
            return True
        left_velocity = np.asarray(left_value, dtype=float)
        right_velocity = np.asarray(right_value, dtype=float)
        left_speed = float(np.linalg.norm(left_velocity))
        right_speed = float(np.linalg.norm(right_velocity))
        if min(left_speed, right_speed) < self.cfg.fusion_instant_min_speed_mps:
            return True
        if float(np.linalg.norm(left_velocity - right_velocity)) > velocity_gate_mps:
            return False
        cosine = float(
            np.dot(left_velocity, right_velocity) / (left_speed * right_speed)
        )
        return cosine >= self.cfg.fusion_instant_direction_cosine

    def _consolidate_duplicate_tracks(self) -> None:
        """Merge repeated close global tracks after stable one-to-one evidence.

        Unlike the immediate camera-pair matcher, this compares current global
        states. It catches duplicates created by delayed cameras while refusing
        tracks that currently share a camera, have overlapping same-camera
        histories, or disagree in motion.
        """
        if (
            self.cfg.fusion_duplicate_merge_m <= 0
            or not np.isfinite(self._max_event_ts_s)
        ):
            return
        active = [
            gid
            for gid, track in self.tracks.items()
            if self.resolve_gid(gid) == gid
            and track.confirmed
            and track.last_seen_ts_s is not None
            and self._max_event_ts_s - track.last_seen_ts_s
            <= self.cfg.fusion_output_grace_s + 1e-9
        ]
        candidates: list[tuple[float, int, int]] = []
        for first_gid, second_gid in combinations(active, 2):
            first = self.tracks[first_gid]
            second = self.tracks[second_gid]
            if first.cameras & second.cameras:
                continue
            distance = float(np.linalg.norm(first.world_xy - second.world_xy))
            if distance > self.cfg.fusion_duplicate_merge_m:
                continue
            if self._has_same_camera_conflict(first_gid, second_gid):
                continue
            if not self._track_motion_compatible(
                first,
                second,
                self.cfg.fusion_duplicate_velocity_gate_mps,
            ):
                self.metrics["duplicate_motion_rejected"] += 1
                continue
            candidates.append((distance, first_gid, second_gid))

        used: set[int] = set()
        for _distance, first_gid, second_gid in sorted(candidates):
            if first_gid in used or second_gid in used:
                continue
            if first_gid not in self.tracks or second_gid not in self.tracks:
                continue
            used.update((first_gid, second_gid))
            pair = tuple(sorted((first_gid, second_gid)))
            count = self._record_pair_vote(
                self._duplicate_merge_votes,
                pair,
                self._max_event_ts_s,
                self.cfg.fusion_duplicate_confirmation_window_s,
            )
            if count < max(1, self.cfg.fusion_duplicate_confirmations):
                self.metrics["duplicate_merges_deferred"] += 1
                if (
                    self.cfg.fusion_duplicate_suppression_enabled
                    and count
                    >= max(
                        1,
                        self.cfg.fusion_duplicate_suppression_min_confirmations,
                    )
                ):
                    canonical = min(
                        (first_gid, second_gid),
                        key=self._canonical_merge_priority,
                    )
                    suppressed = (
                        second_gid if canonical == first_gid else first_gid
                    )
                    self._duplicate_suppressed_gids.add(suppressed)
                continue
            if self.merge_gids(first_gid, second_gid):
                self.metrics["duplicate_global_tracks_merged"] += 1
                for candidate in list(self._duplicate_merge_votes):
                    if first_gid in candidate or second_gid in candidate:
                        del self._duplicate_merge_votes[candidate]

        cutoff = (
            self._max_event_ts_s
            - self.cfg.fusion_duplicate_confirmation_window_s
        )
        for pair, (_, seen_ts) in list(self._duplicate_merge_votes.items()):
            if seen_ts < cutoff:
                del self._duplicate_merge_votes[pair]

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
        emitted_times = [
            value
            for value in (keep.emitted_at_s, lose.emitted_at_s)
            if value is not None
        ]
        keep.emitted_at_s = min(emitted_times) if emitted_times else None
        keep.cameras.update(lose.cameras)
        for camera, endpoint in lose.camera_endpoints.items():
            current = keep.camera_endpoints.get(camera)
            if current is None or endpoint.timestamp_s > current.timestamp_s:
                keep.camera_endpoints[camera] = endpoint
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
                self._fragment_reconnect_votes.pop(key, None)
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
        retention_s = max(
            self.cfg.fusion_max_age_s,
            self.cfg.tracklet_history_s,
            self.cfg.fusion_identity_retention_s,
        )
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
                    self._fragment_reconnect_votes.pop(key, None)
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
