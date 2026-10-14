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
from .diagnostics import IdentityEventWriter
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


@dataclass(frozen=True)
class MotionFeasibilityResult:
    """Result of an event-time world-plane physical feasibility check."""

    feasible: bool
    reason: str
    dt_s: float
    distance_m: float
    allowed_distance_m: float
    uncertainty_m: float
    predicted_xy: np.ndarray


@dataclass(frozen=True)
class TrajectoryMotion:
    """Compact velocity/acceleration summary over a bounded trajectory."""

    velocity: np.ndarray
    acceleration: np.ndarray
    samples: int
    duration_s: float


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
    birth_near_boundary: bool = False
    birth_world_xy: np.ndarray | None = None
    retirement_deferred: bool = False
    dormant: bool = False
    dormant_since_s: float | None = None
    camera_endpoints: dict[str, TrackEndpoint] = field(default_factory=dict)

    @property
    def world_xy(self) -> np.ndarray:
        return self.kf.position


class GlobalTrackManager:
    """Fuse observations and maintain canonical, alias-safe global IDs."""

    def __init__(self, cfg: PipelineConfig):
        if cfg.fusion_provisional_min_age_s < 0:
            raise ValueError("fusion_provisional_min_age_s must be >= 0")
        if cfg.fusion_provisional_buffer_bootstrap_s < 0:
            raise ValueError("fusion_provisional_buffer_bootstrap_s must be >= 0")
        if cfg.fusion_fixed_population_size < 0:
            raise ValueError("fusion_fixed_population_size must be >= 0")
        if cfg.fusion_new_gid_bootstrap_s < 0:
            raise ValueError("fusion_new_gid_bootstrap_s must be >= 0")
        if cfg.fusion_forced_retire_s < 0:
            raise ValueError("fusion_forced_retire_s must be >= 0")
        if cfg.identity_diagnostics_max_events < 0:
            raise ValueError("identity_diagnostics_max_events must be >= 0")
        if cfg.clock_max_pair_delta_spread_s < 0:
            raise ValueError("clock_max_pair_delta_spread_s must be >= 0")
        if cfg.fusion_dormant_retention_s < 0:
            raise ValueError("fusion_dormant_retention_s must be >= 0")
        if cfg.fusion_kinematic_match_gate_m < 0:
            raise ValueError("fusion_kinematic_match_gate_m must be >= 0")
        if cfg.fusion_kinematic_direction_weight < 0:
            raise ValueError("fusion_kinematic_direction_weight must be >= 0")
        if cfg.fusion_kinematic_speed_weight < 0:
            raise ValueError("fusion_kinematic_speed_weight must be >= 0")
        if cfg.fusion_anti_pingpong_frames < 1:
            raise ValueError("fusion_anti_pingpong_frames must be >= 1")
        if cfg.fusion_anti_pingpong_distance_m < 0:
            raise ValueError("fusion_anti_pingpong_distance_m must be >= 0")
        if cfg.fusion_trajectory_min_points < 2:
            raise ValueError("fusion_trajectory_min_points must be >= 2")
        if cfg.fusion_trajectory_min_duration_s < 0:
            raise ValueError("fusion_trajectory_min_duration_s must be >= 0")
        if cfg.fusion_trajectory_max_jump_m <= 0:
            raise ValueError("fusion_trajectory_max_jump_m must be > 0")
        if cfg.fusion_trajectory_max_speed_mps <= 0:
            raise ValueError("fusion_trajectory_max_speed_mps must be > 0")
        if cfg.fusion_jump_window_s <= 0:
            raise ValueError("fusion_jump_window_s must be > 0")
        if cfg.fusion_max_jump_m <= 0:
            raise ValueError("fusion_max_jump_m must be > 0")
        if cfg.fusion_max_speed_mps <= 0:
            raise ValueError("fusion_max_speed_mps must be > 0")
        if cfg.fusion_max_acceleration_mps2 < 0:
            raise ValueError("fusion_max_acceleration_mps2 must be >= 0")
        if cfg.fusion_motion_uncertainty_sigma < 0:
            raise ValueError("fusion_motion_uncertainty_sigma must be >= 0")
        if cfg.fusion_motion_measurement_tolerance_m < 0:
            raise ValueError("fusion_motion_measurement_tolerance_m must be >= 0")
        if cfg.fusion_trajectory_window_s <= 0:
            raise ValueError("fusion_trajectory_window_s must be > 0")
        trajectory_weights = (
            cfg.fusion_trajectory_position_weight,
            cfg.fusion_trajectory_velocity_weight,
            cfg.fusion_trajectory_direction_weight,
            cfg.fusion_trajectory_acceleration_weight,
        )
        if any(weight < 0 for weight in trajectory_weights):
            raise ValueError("fusion trajectory weights must be >= 0")
        if not -1.0 <= cfg.fusion_trajectory_min_direction_cosine <= 1.0:
            raise ValueError(
                "fusion_trajectory_min_direction_cosine must be in [-1, 1]"
            )
        self.cfg = cfg
        self._diagnostics = IdentityEventWriter(
            cfg.identity_diagnostics_path,
            max_events=cfg.identity_diagnostics_max_events,
            flush_every=cfg.identity_diagnostics_flush_every,
        )
        self.tracks: dict[int, GlobalTrack] = {}
        # Dormant identities remain canonical tracks because they still occupy
        # a closed-world population slot.  This explicit pool indexes that
        # subset for lifecycle accounting and kinematic wake-up.
        self.dormant_tracks: dict[int, GlobalTrack] = {}
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
        self._next_provisional_gid = -1
        self._event_mode_started = False
        self._event_start_ts_s: float | None = None
        self._max_event_ts_s = float("-inf")
        self._max_arrival_ts_s = float("-inf")
        self._last_match_run_s = float("-inf")
        self._instant_merge_votes: dict[tuple[int, int], tuple[int, float]] = {}
        self._duplicate_merge_votes: dict[tuple[int, int], tuple[int, float]] = {}
        self._duplicate_suppressed_gids: set[int] = set()
        self._fragment_reconnect_votes: dict[
            TrackletKey, tuple[int, int, float]
        ] = {}
        self._anti_pingpong_locks: dict[TrackletKey, tuple[int, int, float]] = {}
        self._identity_event(
            "run_start",
            cameras=list(cfg.cameras),
            tracklet_history_s=cfg.tracklet_history_s,
            fusion_max_age_s=cfg.fusion_max_age_s,
            fusion_identity_retention_s=cfg.fusion_identity_retention_s,
        )

    def _identity_event(
        self,
        event: str,
        *,
        event_ts_s: float | None = None,
        arrival_ts_s: float | None = None,
        **fields,
    ) -> None:
        if not self._diagnostics.enabled:
            return
        if event_ts_s is None and np.isfinite(self._max_event_ts_s):
            event_ts_s = self._max_event_ts_s
        if arrival_ts_s is None and np.isfinite(self._max_arrival_ts_s):
            arrival_ts_s = self._max_arrival_ts_s
        self._diagnostics.write(
            event,
            event_ts_s=event_ts_s,
            arrival_ts_s=arrival_ts_s,
            **fields,
        )

    def close(self) -> None:
        self._diagnostics.close()

    def record_diagnostic_event(self, event: str, **fields) -> None:
        """Append a non-fusion event to the shared bounded diagnostics log."""
        self._identity_event(event, **fields)

    def resolve_gid(self, gid: int) -> int:
        """Resolve and path-compress a possibly retired alias."""
        path: list[int] = []
        while gid in self.aliases and self.aliases[gid] != gid:
            path.append(gid)
            gid = self.aliases[gid]
        for old in path:
            self.aliases[old] = gid
        return gid

    def _motion_feasibility(
        self,
        track: GlobalTrack,
        observed_xy: np.ndarray,
        timestamp_s: float,
    ) -> MotionFeasibilityResult:
        """Check whether a measurement is reachable from a track's KF state.

        The short-window absolute cap is the final guard against covariance
        inflation making a teleport statistically plausible. Out-of-sequence
        observations are retained as trajectory metadata but never rewind the
        live Kalman state, so they intentionally bypass a forward-time gate.
        """
        observed = np.asarray(observed_xy, dtype=float)
        fallback = np.asarray(track.world_xy, dtype=float).copy()
        if observed.shape != fallback.shape or not np.all(np.isfinite(observed)):
            return MotionFeasibilityResult(
                False, "invalid_observation", 0.0, float("inf"), 0.0, 0.0, fallback
            )
        state_ts = (
            track.state_timestamp_s
            if track.state_timestamp_s is not None
            else track.last_seen_ts_s
        )
        if state_ts is None:
            return MotionFeasibilityResult(
                True, "uninitialized_time", 0.0, 0.0, float("inf"), 0.0, fallback
            )
        raw_dt = float(timestamp_s) - float(state_ts)
        if raw_dt < -1e-9:
            return MotionFeasibilityResult(
                True,
                "out_of_sequence",
                raw_dt,
                float(np.linalg.norm(observed - fallback)),
                float("inf"),
                0.0,
                fallback,
            )
        dt = max(0.0, raw_dt)
        try:
            predicted, covariance = track.kf.projected_measurement(dt)
            predicted = np.asarray(predicted, dtype=float)
            covariance = np.asarray(covariance, dtype=float)
            if (
                predicted.shape != observed.shape
                or covariance.shape != (len(observed), len(observed))
                or not np.all(np.isfinite(predicted))
                or not np.all(np.isfinite(covariance))
            ):
                raise ValueError("non-finite projected state")
            symmetric_covariance = 0.5 * (covariance + covariance.T)
            largest_variance = max(
                0.0, float(np.max(np.linalg.eigvalsh(symmetric_covariance)))
            )
        except (ValueError, np.linalg.LinAlgError):
            return MotionFeasibilityResult(
                False,
                "invalid_prediction",
                dt,
                float("inf"),
                0.0,
                0.0,
                fallback,
            )

        uncertainty = self.cfg.fusion_motion_uncertainty_sigma * np.sqrt(
            largest_variance
        )
        physical_limit = (
            self.cfg.fusion_max_speed_mps * dt
            + 0.5 * self.cfg.fusion_max_acceleration_mps2 * dt * dt
            + uncertainty
            + self.cfg.fusion_motion_measurement_tolerance_m
        )
        allowed = float(physical_limit)
        if dt <= self.cfg.fusion_jump_window_s + 1e-9:
            allowed = min(allowed, self.cfg.fusion_max_jump_m)
        distance = float(np.linalg.norm(observed - predicted))
        feasible = bool(
            np.isfinite(distance)
            and np.isfinite(allowed)
            and distance <= allowed + 1e-9
        )
        return MotionFeasibilityResult(
            feasible,
            "within_gate" if feasible else "physical_jump",
            dt,
            distance,
            allowed,
            float(uncertainty),
            predicted.copy(),
        )

    def _record_motion_feasibility(
        self,
        track: GlobalTrack,
        observation: CameraObservation,
        result: MotionFeasibilityResult,
    ) -> None:
        if result.feasible:
            if result.reason == "within_gate":
                self.metrics["fusion_motion_gate_accepted"] += 1
            return
        self.metrics["fusion_motion_jump_rejected"] += 1
        self._identity_event(
            "motion_gate_rejected",
            event_ts_s=float(observation.capture_ts_s),
            arrival_ts_s=(
                float(observation.arrival_ts_s)
                if observation.arrival_ts_s is not None
                else None
            ),
            gid=track.gid,
            camera=observation.camera,
            local_id=int(observation.local_id),
            reason=result.reason,
            dt_s=result.dt_s,
            distance_m=result.distance_m,
            allowed_distance_m=result.allowed_distance_m,
            uncertainty_m=result.uncertainty_m,
            predicted_xy=result.predicted_xy,
            observed_xy=observation.world_xy,
        )

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
        if self._event_start_ts_s is None:
            start_candidates = [float(obs.capture_ts_s) for obs in ordered]
            if timestamp_s is not None:
                start_candidates.append(float(timestamp_s))
            if start_candidates:
                self._event_start_ts_s = min(start_candidates)
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
            mapped_gid = self.affinity.get(key)
            if mapped_gid is not None:
                mapped_gid = self.resolve_gid(mapped_gid)
            if mapped_gid is not None and mapped_gid in self.tracks:
                feasibility = self._motion_feasibility(
                    self.tracks[mapped_gid], obs.world_xy, event_ts
                )
                self._record_motion_feasibility(
                    self.tracks[mapped_gid], obs, feasibility
                )
                if not feasibility.feasible:
                    # A rejected observation must not contaminate tracklet
                    # history, ownership, or the live Kalman state.
                    # It still breaks any pending consecutive-evidence run:
                    # otherwise a teleport between two valid proposals would
                    # incorrectly count as uninterrupted ownership evidence.
                    self._fragment_reconnect_votes.pop(key, None)
                    self._anti_pingpong_locks.pop(key, None)
                    continue
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
            gid = mapped_gid
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
            pingpong_blocked = False
            if reconnect is not None:
                candidate, requires_confirmation = reconnect
                candidate = self.resolve_gid(candidate)
                if (
                    self.cfg.fusion_anti_pingpong_enabled
                    and gid in self.tracks
                    and candidate in self.tracks
                    and gid != candidate
                ):
                    cur_track = self.tracks[gid]
                    cand_track = self.tracks[candidate]
                    if (
                        cur_track.emitted
                        and cand_track.emitted
                        and not cur_track.dormant
                        and not cand_track.dormant
                        and np.linalg.norm(cur_track.world_xy - cand_track.world_xy)
                        < self.cfg.fusion_anti_pingpong_distance_m
                    ):
                        lock_target, lock_count, lock_ts = self._anti_pingpong_locks.get(
                            key, (None, 0, float("-inf"))
                        )
                        consecutive = (
                            lock_target == candidate
                            and event_ts >= lock_ts - 1e-9
                            and event_ts - lock_ts
                            <= self.cfg.fusion_local_reconnect_confirmation_window_s
                            + 1e-9
                        )
                        new_count = lock_count + 1 if consecutive else 1
                        if new_count >= self.cfg.fusion_anti_pingpong_frames:
                            self._anti_pingpong_locks.pop(key, None)
                        else:
                            self._anti_pingpong_locks[key] = (
                                candidate,
                                new_count,
                                event_ts,
                            )
                            self.metrics["anti_pingpong_switches_blocked"] += 1
                            pingpong_blocked = True
                            reconnect = None

            if reconnect is not None:
                if (
                    not requires_confirmation
                    and key not in self._fragment_reconnect_votes
                ):
                    gid = self._commit_fragment_reconnect(
                        key, gid, candidate, event_ts=event_ts
                    )
                else:
                    reconnected = self._record_fragment_reconnect_vote(
                        key, gid, candidate, event_ts
                    )
                    if reconnected is not None:
                        gid = reconnected
            else:
                self._fragment_reconnect_votes.pop(key, None)
                # A missing or competing proposal breaks the required run of
                # consecutive ownership evidence.
                if not pingpong_blocked:
                    self._anti_pingpong_locks.pop(key, None)

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
        if self.cfg.fusion_dormant_retention_s > 0:
            self._match_dormant_tracks(self._max_event_ts_s)
        self._promote_ready_provisionals()
        self._prune_identity_metadata()
        self._update_population_metrics()
        visible: list[GlobalTrack] = []
        for gid, track in sorted(self.tracks.items()):
            if self.resolve_gid(gid) != gid or not track.confirmed or track.last_seen_ts_s is None:
                continue
            if self.cfg.fusion_deferred_gid_allocation_enabled and gid < 0:
                continue
            if gid in self._duplicate_suppressed_gids:
                self.metrics["duplicate_outputs_suppressed"] += 1
                continue
            if self._publication_buffer_active(track):
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
                    self._identity_event(
                        "publish",
                        gid=gid,
                        created_at_s=track.created_at_s,
                        age_s=(
                            None
                            if track.created_at_s is None
                            else self._max_event_ts_s - track.created_at_s
                        ),
                        cameras=sorted(track.cameras),
                        covariance_trace=float(np.trace(track.kf.P)),
                    )
                track.emitted = True
                track.trail.append(track.world_xy.copy())
                self._trim_trail(track)
                visible.append(track)
        self.metrics["active_global_tracks"] = len(visible)
        self.metrics["tracklet_points"] = self.tracklets.metrics["points_stored"]
        self.metrics["tracklet_bytes"] = self.tracklets.approximate_bytes
        return visible

    def _publication_buffer_active(self, track: GlobalTrack) -> bool:
        """True while a post-bootstrap birth is still under ID review."""
        if (
            not self.cfg.fusion_provisional_buffer_enabled
            or track.created_at_s is None
        ):
            return False
        in_bootstrap = (
            self.cfg.fusion_provisional_buffer_bootstrap_s > 0
            and self._event_start_ts_s is not None
            and track.created_at_s - self._event_start_ts_s
            <= self.cfg.fusion_provisional_buffer_bootstrap_s + 1e-9
        )
        if in_bootstrap:
            return False
        return (
            self._max_event_ts_s - track.created_at_s + 1e-9
            < self.cfg.fusion_provisional_min_age_s
        )

    def _birth_provisional(self, obs: CameraObservation) -> int:
        if self.cfg.fusion_deferred_gid_allocation_enabled:
            gid = self._next_provisional_gid
            self._next_provisional_gid -= 1
        else:
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
            birth_near_boundary=bool(obs.near_boundary),
            birth_world_xy=np.asarray(obs.world_xy, dtype=float).copy(),
            camera_endpoints={obs.camera: self._endpoint_from_observation(obs)},
        )
        self.tracks[gid] = track
        key = obs.camera, int(obs.local_id)
        self.affinity[key] = gid
        self.members[gid].add(key)
        self._identity_event(
            "birth",
            event_ts_s=event_ts,
            arrival_ts_s=track.last_arrival_ts_s,
            camera=obs.camera,
            local_id=int(obs.local_id),
            gid=gid,
            xy=obs.world_xy,
            velocity=obs.world_velocity,
            near_boundary=bool(obs.near_boundary),
            provisional=not track.emitted,
            covariance_trace=float(np.trace(track.kf.P)),
        )
        return gid

    def _promote_ready_provisionals(self) -> None:
        """Allocate public GIDs only after the bounded identity decision."""
        if not self.cfg.fusion_deferred_gid_allocation_enabled:
            return
        public_slots = sum(
            gid > 0 and self.resolve_gid(gid) == gid for gid in self.tracks
        )
        pending = sorted(
            (
                track for gid, track in self.tracks.items()
                if gid < 0 and self.resolve_gid(gid) == gid
            ),
            key=lambda track: (
                float(track.created_at_s) if track.created_at_s is not None else float("inf"),
                track.gid,
            ),
        )
        for track in pending:
            if not track.confirmed or track.created_at_s is None or track.last_seen_ts_s is None:
                continue
            age_s = self._max_event_ts_s - track.created_at_s
            if age_s + 1e-9 < self.cfg.fusion_provisional_min_age_s:
                self.metrics["provisional_publication_deferred"] += 1
                continue
            last_arrival = (
                track.last_arrival_ts_s
                if track.last_arrival_ts_s is not None else track.last_seen_ts_s
            )
            if self._max_arrival_ts_s - last_arrival > self.cfg.fusion_output_grace_s + 1e-9:
                continue
            in_bootstrap = (
                self._event_start_ts_s is not None
                and track.created_at_s - self._event_start_ts_s
                <= self.cfg.fusion_new_gid_bootstrap_s + 1e-9
            )
            if (
                (self.cfg.fusion_new_gid_entry_required or bool(self.cfg.fusion_world_entry_zones))
                and not in_bootstrap
            ):
                if self.cfg.fusion_world_entry_zones:
                    has_entry = self._has_world_entry_evidence(track)
                else:
                    has_entry = track.birth_near_boundary
                if not has_entry:
                    self.metrics["new_gid_entry_deferred"] += 1
                    continue
            capacity = self.cfg.fusion_fixed_population_size
            if capacity > 0 and public_slots >= capacity:
                self.metrics["new_gid_capacity_deferred"] += 1
                continue
            if (
                (
                    self.cfg.fusion_trajectory_verification_enabled
                    or bool(self.cfg.fusion_world_entry_zones)
                )
                and not in_bootstrap
                and not self._verify_trajectory_plausibility(track)
            ):
                self.metrics["new_gid_trajectory_rejected"] += 1
                continue
            self._promote_provisional_gid(track.gid)
            public_slots += 1

    def _verify_trajectory_plausibility(self, track: GlobalTrack) -> bool:
        """Verify that a provisional track has a physically consistent, logically valid
        trajectory before allocating an externally visible Global ID."""
        total_points = track.hits
        total_duration = (
            float(track.last_seen_ts_s - track.created_at_s)
            if track.last_seen_ts_s is not None and track.created_at_s is not None
            else 0.0
        )
        max_jump = 0.0
        speeds: list[float] = []

        # Check all member tracklet histories if available
        for key in self.members.get(track.gid, ()):
            history = self.tracklets.get(key)
            if history is not None and history.points:
                total_points = max(total_points, len(history.points))
                total_duration = max(total_duration, history.duration_s)
                times, xy, vel, conf = history.arrays()
                if len(xy) >= 2:
                    dt = np.diff(times)
                    dist = np.linalg.norm(np.diff(xy, axis=0), axis=1)
                    valid_dt = dt > 1e-4
                    if np.any(valid_dt):
                        inst_speeds = dist[valid_dt] / dt[valid_dt]
                        speeds.extend(inst_speeds.tolist())
                        max_jump = max(max_jump, float(np.max(dist)))

        # Also check the track's world trail if present
        if len(track.trail) >= 2:
            trail_arr = np.asarray(track.trail, dtype=float)
            trail_jumps = np.linalg.norm(np.diff(trail_arr, axis=0), axis=1)
            if len(trail_jumps):
                max_jump = max(max_jump, float(np.max(trail_jumps)))

        # 1. Trajectory length and points check
        if total_points < self.cfg.fusion_trajectory_min_points:
            return False
        if total_duration + 1e-9 < self.cfg.fusion_trajectory_min_duration_s:
            return False

        # 2. Reject sudden teleportation / jump glitches
        if max_jump > self.cfg.fusion_trajectory_max_jump_m:
            return False

        # 3. Reject any unrealistic pedestrian-speed segment. An average can
        # hide a single teleport between otherwise stationary observations.
        if speeds and float(np.max(speeds)) > self.cfg.fusion_trajectory_max_speed_mps:
            return False

        # 4. Speed from Kalman filter velocity
        kf_speed = float(np.linalg.norm(track.kf.velocity))
        if kf_speed > self.cfg.fusion_trajectory_max_speed_mps:
            return False

        return True

    def _ordered_track_samples(
        self,
        track: GlobalTrack,
        *,
        reference_ts_s: float | None = None,
        window_s: float | None = None,
    ) -> list[tuple[float, np.ndarray]]:
        """Return timestamped world observations, averaging camera duplicates."""
        raw_samples: list[tuple[float, np.ndarray]] = []
        for key in self.members.get(track.gid, ()):
            history = self.tracklets.get(key)
            if history is None:
                continue
            raw_samples.extend(
                (float(point.timestamp_s), np.asarray(point.xy, dtype=float).copy())
                for point in history.points
            )
        if track.created_at_s is not None and track.birth_world_xy is not None:
            raw_samples.append(
                (
                    float(track.created_at_s),
                    np.asarray(track.birth_world_xy, dtype=float).copy(),
                )
            )
        for endpoint in track.camera_endpoints.values():
            raw_samples.append(
                (float(endpoint.timestamp_s), np.asarray(endpoint.xy, dtype=float).copy())
            )
        if not raw_samples:
            return []

        grouped: defaultdict[float, list[np.ndarray]] = defaultdict(list)
        for timestamp, point in raw_samples:
            if np.isfinite(timestamp) and np.all(np.isfinite(point)):
                grouped[timestamp].append(point)
        samples = [
            (timestamp, np.mean(points, axis=0))
            for timestamp, points in sorted(grouped.items())
        ]
        if not samples:
            return []
        reference = (
            samples[-1][0] if reference_ts_s is None else float(reference_ts_s)
        )
        if window_s is not None:
            cutoff = reference - max(0.0, float(window_s))
            samples = [
                sample
                for sample in samples
                if cutoff - 1e-9 <= sample[0] <= reference + 1e-9
            ]
        return [(timestamp, point.copy()) for timestamp, point in samples]

    def _ordered_track_positions(self, track: GlobalTrack) -> list[np.ndarray]:
        """Return de-duplicated observed positions in temporal order."""
        samples = self._ordered_track_samples(track)
        positions = [point.copy() for _timestamp, point in samples]
        if not positions:
            positions = [np.asarray(point, dtype=float).copy() for point in track.trail]

        deduplicated: list[np.ndarray] = []
        for point in positions:
            if not deduplicated or not np.allclose(
                deduplicated[-1], point, atol=1e-6
            ):
                deduplicated.append(point)
        return deduplicated

    def _trajectory_motion(
        self, track: GlobalTrack, reference_ts_s: float
    ) -> TrajectoryMotion:
        samples = self._ordered_track_samples(
            track,
            reference_ts_s=reference_ts_s,
            window_s=self.cfg.fusion_trajectory_window_s,
        )
        if len(samples) < 2:
            velocity = self._latest_track_velocity(track)
            return TrajectoryMotion(
                np.asarray(velocity, dtype=float),
                np.zeros(2, dtype=float),
                len(samples),
                0.0,
            )

        segment_velocities: list[tuple[float, np.ndarray]] = []
        for (left_ts, left_xy), (right_ts, right_xy) in zip(samples, samples[1:]):
            dt = right_ts - left_ts
            if dt <= 1e-6:
                continue
            segment_velocities.append((right_ts, (right_xy - left_xy) / dt))
        if not segment_velocities:
            velocity = self._latest_track_velocity(track)
            acceleration = np.zeros(2, dtype=float)
        else:
            velocity = segment_velocities[-1][1]
            acceleration = np.zeros(2, dtype=float)
            if len(segment_velocities) >= 2:
                acceleration_dt = (
                    segment_velocities[-1][0] - segment_velocities[-2][0]
                )
                if acceleration_dt > 1e-6:
                    acceleration = (
                        segment_velocities[-1][1] - segment_velocities[-2][1]
                    ) / acceleration_dt
        return TrajectoryMotion(
            np.asarray(velocity, dtype=float),
            np.asarray(acceleration, dtype=float),
            len(samples),
            max(0.0, samples[-1][0] - samples[0][0]),
        )

    def _latest_track_velocity(
        self, track: GlobalTrack, positions: list[np.ndarray] | None = None
    ) -> np.ndarray:
        if track.camera_endpoints:
            endpoint = max(
                track.camera_endpoints.values(), key=lambda item: item.timestamp_s
            )
            if endpoint.velocity is not None:
                velocity = np.asarray(endpoint.velocity, dtype=float)
                if np.all(np.isfinite(velocity)) and np.linalg.norm(velocity) > 1e-6:
                    return velocity
        velocity = np.asarray(track.kf.velocity, dtype=float)
        if np.all(np.isfinite(velocity)) and np.linalg.norm(velocity) > 1e-6:
            return velocity
        if positions is None:
            positions = self._ordered_track_positions(track)
        if len(positions) >= 2:
            return positions[-1] - positions[-2]
        return np.zeros(2, dtype=float)

    def _has_world_entry_evidence(self, track: GlobalTrack) -> bool:
        """Require a birth in an entry zone followed by motion into the scene.

        Entry zones describe door footprints, not the entire room. Therefore a
        trajectory crossing from its birth zone to a subsequently observed
        non-zone point defines the inward normal without another calibration
        parameter. The latest velocity must agree with that inward direction.
        """
        positions = self._ordered_track_positions(track)
        if not positions:
            return False
        birth = (
            np.asarray(track.birth_world_xy, dtype=float)
            if track.birth_world_xy is not None
            else positions[0]
        )
        velocity = self._latest_track_velocity(track, positions)
        for zone in self.cfg.fusion_world_entry_zones:
            if not self._point_in_world_zones(birth, [zone]):
                continue
            inside_to_outside = next(
                (
                    point - birth
                    for point in positions[1:]
                    if not self._point_in_world_zones(point, [zone])
                ),
                None,
            )
            if inside_to_outside is None:
                continue
            norm = float(np.linalg.norm(inside_to_outside))
            if norm <= 1e-6:
                continue
            inward_normal = inside_to_outside / norm
            if float(np.dot(velocity, inward_normal)) > 1e-6:
                return True
        return False

    def _has_world_exit_evidence(self, track: GlobalTrack) -> bool:
        """Require an interior-to-exit-zone crossing with outward motion."""
        positions = self._ordered_track_positions(track)
        if len(positions) < 2:
            return False
        final = positions[-1]
        velocity = self._latest_track_velocity(track, positions)
        for zone in self.cfg.fusion_world_exit_zones:
            if not self._point_in_world_zones(final, [zone]):
                continue
            previous_interior = next(
                (
                    point
                    for point in reversed(positions[:-1])
                    if not self._point_in_world_zones(point, [zone])
                ),
                None,
            )
            if previous_interior is None:
                continue
            outward = final - previous_interior
            norm = float(np.linalg.norm(outward))
            if norm <= 1e-6:
                continue
            outward_normal = outward / norm
            if float(np.dot(velocity, outward_normal)) > 1e-6:
                return True
        return False

    @staticmethod
    def _point_in_world_zones(
        point: np.ndarray | tuple[float, float],
        zones: list,
    ) -> bool:
        """Check if (x, y) point is inside any defined world zone.

        Supported zone formats:
        - Box: [xmin, xmax, ymin, ymax]
        - Polygon: [(x1, y1), (x2, y2), ...]
        """
        if not zones:
            return False
        x, y = float(point[0]), float(point[1])
        for zone in zones:
            coordinates = np.asarray(zone, dtype=float)
            if coordinates.ndim == 1 and coordinates.shape == (4,):
                xmin, xmax, ymin, ymax = coordinates
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    return True
            elif (
                coordinates.ndim == 2
                and coordinates.shape[0] >= 3
                and coordinates.shape[1] == 2
            ):
                # Point-in-polygon ray-casting algorithm
                n = len(coordinates)
                inside = False
                p1x, p1y = coordinates[0]
                for i in range(1, n + 1):
                    p2x, p2y = coordinates[i % n]
                    if min(p1y, p2y) < y <= max(p1y, p2y):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                    p1x, p1y = p2x, p2y
                if inside:
                    return True
        return False

    def _match_dormant_tracks(self, event_ts: float) -> None:
        """Kinematically match unpromoted provisional tracks against dormant published tracks.

        When a person is temporarily occluded inside the scene interior, their public GID
        becomes dormant. Instead of birthing a redundant new GID when a tracklet reappears,
        this forward-predicts the dormant track's motion (position + direction + speed)
        and reconnects the provisional track.
        """
        if not self.tracks or not np.isfinite(event_ts):
            return

        dormant_gids = [
            gid
            for gid, track in sorted(self.dormant_tracks.items())
            if gid > 0
            and self.resolve_gid(gid) == gid
            and gid in self.tracks
            and track.emitted
            and track.last_seen_ts_s is not None
            and track.dormant
            and event_ts - track.last_seen_ts_s
            <= self.cfg.fusion_dormant_retention_s + 1e-9
        ]
        if not dormant_gids:
            return

        provisional_gids = [
            gid
            for gid, track in sorted(self.tracks.items())
            if gid < 0
            and self.resolve_gid(gid) == gid
            and track.last_seen_ts_s is not None
            and abs(event_ts - track.last_seen_ts_s) <= self.cfg.fusion_output_grace_s + 0.5
            and (
                not self.cfg.fusion_world_entry_zones
                or track.birth_world_xy is None
                or not self._point_in_world_zones(
                    track.birth_world_xy, self.cfg.fusion_world_entry_zones
                )
            )
        ]
        if not provisional_gids:
            return

        gate_m = self.cfg.fusion_kinematic_match_gate_m
        sentinel = 1e6

        cost_matrix = np.full(
            (len(dormant_gids), len(provisional_gids)), sentinel, dtype=float
        )
        spatial_matrix = np.full_like(cost_matrix, sentinel)
        evidence_by_pair: dict[tuple[int, int], dict[str, float]] = {}

        for i, d_gid in enumerate(dormant_gids):
            d_track = self.tracks[d_gid]
            d_motion = self._trajectory_motion(d_track, event_ts)
            d_speed = float(np.linalg.norm(d_motion.velocity))

            for j, p_gid in enumerate(provisional_gids):
                if self._has_same_camera_conflict(d_gid, p_gid):
                    continue
                p_track = self.tracks[p_gid]
                p_samples = self._ordered_track_samples(p_track)
                p_pos = p_samples[-1][1] if p_samples else p_track.world_xy
                feasibility = self._motion_feasibility(
                    d_track, p_pos, event_ts
                )
                if not feasibility.feasible:
                    self.metrics["fusion_motion_jump_rejected"] += 1
                    self.metrics["trajectory_matches_rejected"] += 1
                    continue
                self.metrics["fusion_motion_gate_accepted"] += 1
                dist = feasibility.distance_m
                if dist > gate_m:
                    self.metrics["trajectory_matches_rejected"] += 1
                    continue

                p_motion = self._trajectory_motion(p_track, event_ts)
                p_speed = float(np.linalg.norm(p_motion.velocity))
                if max(d_speed, p_speed) > self.cfg.fusion_max_speed_mps:
                    self.metrics["trajectory_speed_rejected"] += 1
                    self.metrics["trajectory_matches_rejected"] += 1
                    continue
                if (
                    p_motion.samples >= 3
                    and np.linalg.norm(p_motion.acceleration)
                    > self.cfg.fusion_max_acceleration_mps2
                ):
                    self.metrics["trajectory_acceleration_rejected"] += 1
                    self.metrics["trajectory_matches_rejected"] += 1
                    continue

                dir_cost = 0.0
                if d_speed >= 0.15 and p_speed >= 0.15:
                    dot = float(
                        np.dot(d_motion.velocity, p_motion.velocity)
                    )
                    cos_sim = dot / (d_speed * p_speed)
                    cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
                    if (
                        cos_sim
                        < self.cfg.fusion_trajectory_min_direction_cosine
                    ):
                        self.metrics["trajectory_direction_rejected"] += 1
                        self.metrics["trajectory_matches_rejected"] += 1
                        continue
                    dir_cost = max(0.0, 1.0 - cos_sim)

                speed_diff = abs(d_speed - p_speed)
                acceleration_diff = float(
                    np.linalg.norm(
                        d_motion.acceleration - p_motion.acceleration
                    )
                )
                cost = (
                    self.cfg.fusion_trajectory_position_weight * dist
                    + self.cfg.fusion_trajectory_velocity_weight * speed_diff
                    + self.cfg.fusion_trajectory_direction_weight * dir_cost
                    + self.cfg.fusion_trajectory_acceleration_weight
                    * acceleration_diff
                )
                cost_matrix[i, j] = cost
                spatial_matrix[i, j] = dist
                evidence_by_pair[i, j] = {
                    "position_cost": float(dist),
                    "velocity_cost": float(speed_diff),
                    "direction_cost": float(dir_cost),
                    "acceleration_cost": float(acceleration_diff),
                    "allowed_distance_m": feasibility.allowed_distance_m,
                }

        if not np.any(cost_matrix < sentinel):
            return

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            if cost >= sentinel or spatial_matrix[r, c] > gate_m:
                continue
            alternatives = [
                cost_matrix[r, other]
                for other in range(cost_matrix.shape[1])
                if other != c and cost_matrix[r, other] < sentinel
            ] + [
                cost_matrix[other, c]
                for other in range(cost_matrix.shape[0])
                if other != r and cost_matrix[other, c] < sentinel
            ]
            margin = min(alternatives, default=float("inf")) - cost
            if margin + 1e-9 < self.cfg.fusion_instant_margin_m:
                self.metrics["trajectory_matches_rejected"] += 1
                self.metrics["trajectory_matches_ambiguous"] += 1
                continue
            d_gid = dormant_gids[r]
            p_gid = provisional_gids[c]
            dormant_track = self.tracks[d_gid]
            dormant_state = dormant_track.kf.x.copy()
            dormant_covariance = dormant_track.kf.P.copy()
            dormant_state_ts = dormant_track.state_timestamp_s
            provisional_samples = self._ordered_track_samples(self.tracks[p_gid])
            observed_position = (
                provisional_samples[-1][1]
                if provisional_samples
                else self.tracks[p_gid].world_xy.copy()
            )

            dormant_last_seen = self.tracks[d_gid].last_seen_ts_s
            dt_val = max(
                0.0,
                event_ts
                - (event_ts if dormant_last_seen is None else dormant_last_seen),
            )
            if self.merge_gids(
                d_gid,
                p_gid,
                reason="kinematic_dormant_reconnect",
                evidence={
                    "cost": float(cost),
                    "dt_s": float(dt_val),
                    "assignment_margin": float(margin),
                    **evidence_by_pair.get((r, c), {}),
                },
            ):
                canonical_gid = self.resolve_gid(d_gid)
                self.dormant_tracks.pop(d_gid, None)
                self.dormant_tracks.pop(p_gid, None)
                if canonical_gid in self.tracks:
                    canonical_track = self.tracks[canonical_gid]
                    # Wake the public identity from its own predicted state,
                    # then correct it with the latest provisional observation.
                    # This retains the dormant trajectory/covariance instead
                    # of blindly replacing it with a short fragment's KF.
                    canonical_track.kf.x = dormant_state
                    canonical_track.kf.P = dormant_covariance
                    prediction_dt = max(
                        0.0,
                        event_ts
                        - (
                            event_ts
                            if dormant_state_ts is None
                            else dormant_state_ts
                        ),
                    )
                    canonical_track.kf.predict(prediction_dt)
                    canonical_track.kf.update(observed_position)
                    canonical_track.state_timestamp_s = event_ts
                    canonical_track.dormant = False
                    canonical_track.dormant_since_s = None
                    canonical_track.retirement_deferred = False
                self.metrics["dormant_reconnects_accepted"] += 1
                self.metrics["trajectory_matches_accepted"] += 1

    def _promote_provisional_gid(self, provisional_gid: int) -> int:
        provisional_gid = self.resolve_gid(provisional_gid)
        if provisional_gid >= 0 or provisional_gid not in self.tracks:
            return provisional_gid
        public_gid = self._next_gid
        self._next_gid += 1
        track = self.tracks.pop(provisional_gid)
        track.gid = public_gid
        self.tracks[public_gid] = track
        self.members[public_gid].update(self.members.pop(provisional_gid, set()))
        self.aliases[provisional_gid] = public_gid
        self._alias_created_s[provisional_gid] = self._max_event_ts_s
        self._canonicalize_affinity()
        # Votes are cheap and short-lived. Dropping them avoids carrying a
        # stale provisional pair across the externally visible rename.
        self._instant_merge_votes.clear()
        self._duplicate_merge_votes.clear()
        self.metrics["public_gid_allocations"] += 1
        return public_gid

    def public_affinity_snapshot(self) -> dict[TrackletKey, int]:
        """Return only associations safe to expose to viewers/consumers."""
        if (
            not self.cfg.fusion_deferred_gid_allocation_enabled
            and not self.cfg.fusion_provisional_buffer_enabled
        ):
            return dict(self.affinity)
        published: dict[TrackletKey, int] = {}
        for key, mapped_gid in self.affinity.items():
            gid = self.resolve_gid(mapped_gid)
            track = self.tracks.get(gid)
            if gid > 0 and track is not None and track.emitted:
                published[key] = gid
        return published

    def _update_population_metrics(self) -> None:
        pending = sum(
            gid < 0 and self.resolve_gid(gid) == gid for gid in self.tracks
        )
        public = sum(
            gid > 0 and self.resolve_gid(gid) == gid for gid in self.tracks
        )
        self.metrics["pending_provisional_tracks"] = pending
        self.metrics["public_identity_slots"] = public
        self.metrics["dormant_identity_count"] = len(self.dormant_tracks)
        self.metrics["peak_pending_provisional_tracks"] = max(
            int(self.metrics["peak_pending_provisional_tracks"]), pending
        )
        self.metrics["peak_public_identity_slots"] = max(
            int(self.metrics["peak_public_identity_slots"]), public
        )

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
        candidate_details = [
            {
                "gid": gid,
                "cost": cost,
                "distance_m": distance,
                "long_gap": is_long,
                "emitted": self.tracks[gid].emitted,
                "covariance_trace": float(np.trace(self.tracks[gid].kf.P)),
            }
            for cost, distance, gid, is_long in candidates[:3]
        ]
        if len(candidates) > 1:
            margin = candidates[1][0] - best_cost
            if margin + 1e-9 < self.cfg.fusion_instant_margin_m:
                self.metrics["local_reconnect_ambiguous"] += 1
                self._identity_event(
                    "reconnect_rejected",
                    event_ts_s=float(obs.capture_ts_s),
                    arrival_ts_s=(
                        float(obs.arrival_ts_s)
                        if obs.arrival_ts_s is not None else None
                    ),
                    camera=obs.camera,
                    local_id=int(obs.local_id),
                    current_gid=excluded_gid,
                    reason="ambiguity_margin",
                    margin=margin,
                    candidates=candidate_details,
                    xy=obs.world_xy,
                    velocity=self._stable_observation_velocity(obs),
                )
                return None
        published_owner = self.tracks[best_gid].emitted
        requires_confirmation = (
            best_is_long
            or len(candidates) > 1
            or (
                self.cfg.fusion_gid_ownership_hysteresis_enabled
                and published_owner
            )
        )
        self._identity_event(
            "reconnect_candidate",
            event_ts_s=float(obs.capture_ts_s),
            arrival_ts_s=(
                float(obs.arrival_ts_s) if obs.arrival_ts_s is not None else None
            ),
            camera=obs.camera,
            local_id=int(obs.local_id),
            current_gid=excluded_gid,
            selected_gid=best_gid,
            published_owner=published_owner,
            requires_confirmation=requires_confirmation,
            candidates=candidate_details,
            xy=obs.world_xy,
            velocity=self._stable_observation_velocity(obs),
        )
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

            feasibility = self._motion_feasibility(track, obs.world_xy, event_ts)
            self._record_motion_feasibility(track, obs, feasibility)
            if not feasibility.feasible:
                self.metrics["trajectory_matches_rejected"] += 1
                continue
            state_ts = (
                track.state_timestamp_s
                if track.state_timestamp_s is not None
                else endpoint.timestamp_s
            )
            prediction_dt = max(0.0, event_ts - state_ts)
            _predicted_xy, innovation_covariance = track.kf.projected_measurement(
                prediction_dt
            )
            predicted_xy = feasibility.predicted_xy
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

            if endpoint.velocity is not None and observation_velocity is not None:
                previous_velocity = np.asarray(endpoint.velocity, dtype=float)
                current_velocity = np.asarray(observation_velocity, dtype=float)
                previous_speed = float(np.linalg.norm(previous_velocity))
                current_speed = float(np.linalg.norm(current_velocity))
                if max(previous_speed, current_speed) > self.cfg.fusion_max_speed_mps:
                    self.metrics["trajectory_speed_rejected"] += 1
                    self.metrics["trajectory_matches_rejected"] += 1
                    continue
                if (
                    min(previous_speed, current_speed)
                    >= self.cfg.fusion_instant_min_speed_mps
                ):
                    direction_cosine = float(
                        np.dot(previous_velocity, current_velocity)
                        / (previous_speed * current_speed)
                    )
                    if (
                        direction_cosine
                        < self.cfg.fusion_trajectory_min_direction_cosine
                    ):
                        # Preserve the historical aggregate metric while the
                        # more precise counter records that this is now a hard
                        # trajectory veto rather than merely a soft penalty.
                        self.metrics["local_reconnect_motion_penalties"] += 1
                        self.metrics["trajectory_direction_rejected"] += 1
                        self.metrics["trajectory_matches_rejected"] += 1
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
            published_owner = self.tracks[candidate[2]].emitted
            requires_confirmation = (
                candidate[3]
                or ambiguous
                or len(row_candidates[row]) > 1
                or (
                    self.cfg.fusion_gid_ownership_hysteresis_enabled
                    and published_owner
                )
            )
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
        return self._commit_fragment_reconnect(
            key, current_gid, candidate_gid, event_ts=event_ts
        )

    def _commit_fragment_reconnect(
        self,
        key: TrackletKey,
        current_gid: int,
        candidate_gid: int,
        *,
        event_ts: float | None = None,
    ) -> int:
        current = self.resolve_gid(current_gid)
        candidate = self.resolve_gid(candidate_gid)
        reconnected_before_publish = (
            current in self.tracks and not self.tracks[current].emitted
        )
        if current != candidate and self.merge_gids(
            candidate,
            current,
            reason="local_fragment_reconnect",
            evidence={"camera": key[0], "local_id": key[1]},
        ):
            canonical = self.resolve_gid(candidate)
            self.affinity[key] = canonical
            self.members[canonical].add(key)
            self.metrics["local_fragments_reconnected"] += 1
            self.metrics["trajectory_matches_accepted"] += 1
            if reconnected_before_publish:
                self.metrics["provisional_ids_reconnected_before_publish"] += 1
            self._identity_event(
                "reconnect",
                event_ts_s=event_ts,
                camera=key[0],
                local_id=key[1],
                previous_gid=current,
                candidate_gid=candidate,
                canonical_gid=canonical,
                provisional_before_publish=reconnected_before_publish,
            )
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
        track.dormant = False
        track.dormant_since_s = None
        track.retirement_deferred = False
        self.dormant_tracks.pop(track.gid, None)
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
            clock_delta_spread_s = (
                float(np.ptp([result.delta_s for result in results]))
                if len(results) > 1
                else 0.0
            )
            clock_delta_consistent = (
                clock_delta_spread_s
                <= self.cfg.clock_max_pair_delta_spread_s + 1e-9
            )
            if results and clock_delta_consistent:
                median_delta = float(np.median([
                    result.delta_s for result in results
                ]))
                representative = min(
                    results,
                    key=lambda result: abs(result.delta_s - median_delta),
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
            elif results and not clock_delta_consistent:
                self.metrics["clock_updates_rejected_delta_spread"] += 1
                self._identity_event(
                    "clock_update_rejected",
                    event_ts_s=pair_time,
                    left_camera=left_camera,
                    right_camera=right_camera,
                    reason="inconsistent_pair_delta",
                    matches=len(results),
                    delta_spread_s=clock_delta_spread_s,
                    maximum_spread_s=self.cfg.clock_max_pair_delta_spread_s,
                )
            for result in results:
                left_gid = self.resolve_gid(self.affinity[result.left_key])
                right_gid = self.resolve_gid(self.affinity[result.right_key])
                if left_gid == right_gid:
                    continue
                if self.merge_gids(
                    left_gid,
                    right_gid,
                    reason="tracklet_trajectory",
                    evidence={
                        "left_key": result.left_key,
                        "right_key": result.right_key,
                        "cost": result.cost,
                        "margin": result.margin,
                        "delta_s": result.delta_s,
                        "overlap_s": result.overlap_s,
                        "position_rmse_m": result.position_rmse_m,
                        "velocity_rmse_mps": result.velocity_rmse_mps,
                        "direction_cosine": result.direction_cosine,
                        "confidence": result.confidence,
                    },
                ):
                    self.metrics["trajectory_matches_accepted"] += 1

    def _learn_pair_offset(
        self,
        left_camera: str,
        right_camera: str,
        result: MatchResult,
    ) -> None:
        left_history = self.tracklets.get(result.left_key)
        right_history = self.tracklets.get(result.right_key)
        before_left = self.clock_estimator.estimate(left_camera)
        before_right = self.clock_estimator.estimate(right_camera)
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
        after_left = self.clock_estimator.estimate(left_camera)
        after_right = self.clock_estimator.estimate(right_camera)
        estimator_updated = (
            after_left.samples > before_left.samples
            or after_right.samples > before_right.samples
        )
        if not estimator_updated:
            self.metrics["clock_updates_rejected_by_estimator"] += 1
        self._identity_event(
            "clock_update" if estimator_updated else "clock_update_rejected",
            event_ts_s=max(left_history.end_s, right_history.end_s),
            left_camera=left_camera,
            right_camera=right_camera,
            reason=None if estimator_updated else "estimator_confidence_or_outlier",
            delta_s=result.delta_s,
            confidence=result.confidence,
            position_rmse_m=result.position_rmse_m,
            velocity_rmse_mps=result.velocity_rmse_mps,
            before={
                left_camera: before_left.__dict__,
                right_camera: before_right.__dict__,
            },
            after={
                left_camera: after_left.__dict__,
                right_camera: after_right.__dict__,
            },
            correction_s={
                left_camera: self.clock_estimator.correction_at(
                    left_camera, left_history.end_s
                ),
                right_camera: self.clock_estimator.correction_at(
                    right_camera, right_history.end_s
                ),
            },
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
                    self._identity_event(
                        "instant_merge_rejected",
                        event_ts_s=max(
                            float(left_items[row].capture_ts_s),
                            float(right_items[column].capture_ts_s),
                        ),
                        left_key=(left_camera, left_items[row].local_id),
                        right_key=(right_camera, right_items[column].local_id),
                        reason="ambiguity_margin",
                        distance_m=float(spatial[row, column]),
                        cost=float(value),
                        margin=float(margin),
                        left_clock_correction_s=self.clock_estimator.correction_at(
                            left_camera, float(left_items[row].capture_ts_s)
                        ),
                        right_clock_correction_s=self.clock_estimator.correction_at(
                            right_camera, float(right_items[column].capture_ts_s)
                        ),
                    )
                    continue
                if ambiguous_assignment[row, column] and self._strongly_opposite_motion(
                    left_items[row], right_items[column]
                ):
                    # Do not immediately try the second-best spatial pairing.
                    # Waiting until the people separate is safer than making an
                    # irreversible cross-person merge during this frame.
                    self.metrics["instant_opposite_direction_vetoes"] += 1
                    self._identity_event(
                        "instant_merge_rejected",
                        event_ts_s=max(
                            float(left_items[row].capture_ts_s),
                            float(right_items[column].capture_ts_s),
                        ),
                        left_key=(left_camera, left_items[row].local_id),
                        right_key=(right_camera, right_items[column].local_id),
                        reason="opposite_motion",
                        distance_m=float(spatial[row, column]),
                        margin=float(margin),
                    )
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
                if self.merge_gids(
                    left_gid,
                    right_gid,
                    reason="instant_geometry",
                    evidence={
                        "left_key": left_key,
                        "right_key": right_key,
                        "distance_m": float(spatial[row, column]),
                        "cost": float(value),
                        "margin": float(margin),
                        "motion_compatible": motion_compatible,
                        "confirmations": confirmations,
                        "votes": count,
                    },
                ):
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
            if self.merge_gids(
                first_gid,
                second_gid,
                reason="duplicate_consolidation",
                evidence={"distance_m": float(_distance), "votes": count},
            ):
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

    def merge_gids(
        self,
        first_gid: int,
        second_gid: int,
        *,
        reason: str = "explicit",
        evidence: dict | None = None,
    ) -> bool:
        """Merge IDs into the oldest canonical ID; safe and idempotent."""
        first = self.resolve_gid(first_gid)
        second = self.resolve_gid(second_gid)
        if first == second:
            return True
        if first not in self.tracks or second not in self.tracks:
            return False
        if self._has_same_camera_conflict(first, second):
            self.metrics["same_camera_merge_conflicts"] += 1
            self._identity_event(
                "merge_rejected",
                first_gid=first,
                second_gid=second,
                reason=reason,
                rejection="same_camera_overlap",
                evidence=evidence or {},
            )
            return False

        canonical = min((first, second), key=self._canonical_merge_priority)
        retired = second if canonical == first else first
        keep = self.tracks[canonical]
        lose = self.tracks[retired]
        keep_created_at = keep.created_at_s
        lose_created_at = lose.created_at_s
        was_late = keep.emitted or lose.emitted
        published_pair = keep.emitted and lose.emitted
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
        if keep.birth_world_xy is None or (
            lose.birth_world_xy is not None
            and lose_created_at is not None
            and (keep_created_at is None or lose_created_at < keep_created_at)
        ):
            keep.birth_world_xy = (
                None
                if lose.birth_world_xy is None
                else np.asarray(lose.birth_world_xy, dtype=float).copy()
            )
        keep.hits += lose.hits
        keep.confirmed = keep.confirmed or lose.confirmed or keep.hits >= self.cfg.fusion_min_hits
        emitted_times = [
            value
            for value in (keep.emitted_at_s, lose.emitted_at_s)
            if value is not None
        ]
        keep.emitted_at_s = min(emitted_times) if emitted_times else None
        keep.birth_near_boundary = (
            keep.birth_near_boundary or lose.birth_near_boundary
        )
        keep.retirement_deferred = (
            keep.retirement_deferred and lose.retirement_deferred
        )
        keep.dormant = keep.dormant and lose.dormant
        dormant_times = [
            value
            for value in (keep.dormant_since_s, lose.dormant_since_s)
            if value is not None
        ]
        keep.dormant_since_s = min(dormant_times) if keep.dormant else None
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
        self.dormant_tracks.pop(first, None)
        self.dormant_tracks.pop(second, None)
        if keep.dormant:
            self.dormant_tracks[canonical] = keep
        self._canonicalize_affinity()
        self.metrics["global_id_merges"] += 1
        if was_late:
            self.metrics["late_reassociations"] += 1
        self._identity_event(
            "merge",
            first_gid=first,
            second_gid=second,
            canonical_gid=canonical,
            retired_gid=retired,
            reason=reason,
            evidence=evidence or {},
            published_pair=published_pair,
            canonical_was_published=keep.emitted,
            retired_was_published=lose.emitted,
            covariance_trace=float(np.trace(keep.kf.P)),
        )
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
                self._anti_pingpong_locks.pop(key, None)
        for gid in list(self.members):
            if gid not in self.tracks:
                del self.members[gid]
        for gid in list(self.dormant_tracks):
            if gid not in self.tracks or not self.tracks[gid].dormant:
                del self.dormant_tracks[gid]
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
        active_retention_s = max(
            self.cfg.fusion_max_age_s, self.cfg.fusion_identity_retention_s
        )
        provisional_retention_s = max(
            active_retention_s, self.cfg.tracklet_history_s
        )
        for gid in list(self.tracks):
            track = self.tracks[gid]
            if track.last_seen_ts_s is None:
                continue
            age_s = self._max_event_ts_s - track.last_seen_ts_s
            public_identity = gid > 0 and track.emitted
            closed_world_lifecycle = public_identity and (
                self.cfg.fusion_retire_requires_exit_boundary
                or bool(self.cfg.fusion_world_exit_zones)
                or self.cfg.fusion_dormant_retention_s > 0
            )

            if not closed_world_lifecycle:
                if age_s <= provisional_retention_s:
                    continue
                exit_evidence = False
            else:
                if age_s <= active_retention_s:
                    continue
                exit_evidence = self._has_exit_evidence(track)
                if exit_evidence:
                    self.metrics["confirmed_identity_exits"] += 1
                else:
                    dormant_retention_s = self.cfg.fusion_dormant_retention_s
                    if dormant_retention_s <= 0:
                        dormant_retention_s = self.cfg.fusion_forced_retire_s
                    if dormant_retention_s <= 0 or age_s <= dormant_retention_s:
                        track.dormant = True
                        if track.dormant_since_s is None:
                            track.dormant_since_s = self._max_event_ts_s
                        self.dormant_tracks[gid] = track
                        if not track.retirement_deferred:
                            track.retirement_deferred = True
                            self.metrics["retirement_deferred_no_exit"] += 1
                            self._identity_event(
                                "dormant",
                                gid=gid,
                                last_seen_ts_s=track.last_seen_ts_s,
                                age_s=age_s,
                            )
                        continue
                    self.metrics["forced_identity_retirements"] += 1

            del self.tracks[gid]
            self.dormant_tracks.pop(gid, None)
            self.members.pop(gid, None)
            for key, mapped_gid in list(self.affinity.items()):
                if self.resolve_gid(mapped_gid) == gid:
                    del self.affinity[key]
                    self._fragment_reconnect_votes.pop(key, None)
                    self._anti_pingpong_locks.pop(key, None)
            self.metrics["global_tracks_expired"] += 1
            self._identity_event(
                "expiry",
                gid=gid,
                last_seen_ts_s=track.last_seen_ts_s,
                age_s=age_s,
                emitted=track.emitted,
                exit_evidence=exit_evidence,
                cameras=sorted(track.camera_endpoints),
                covariance_trace=float(np.trace(track.kf.P)),
            )

    def _has_exit_evidence(self, track: GlobalTrack) -> bool:
        """True when a final observation was at a camera image boundary or in a world exit zone."""
        if track.last_seen_ts_s is None:
            return False
        if self.cfg.fusion_world_exit_zones:
            return self._has_world_exit_evidence(track)
        latest = [
            endpoint
            for endpoint in track.camera_endpoints.values()
            if abs(endpoint.timestamp_s - track.last_seen_ts_s)
            <= self.cfg.fusion_event_time_gate_s + 1e-9
        ]
        return bool(latest) and any(endpoint.near_boundary for endpoint in latest)

    def metrics_snapshot(self) -> dict[str, int | float | dict[str, dict[str, float]]]:
        snapshot: dict = dict(self.metrics)
        snapshot.update({
            "tracklet_late_dropped": self.tracklets.metrics["late_dropped"],
            "tracklet_out_of_order_accepted": self.tracklets.metrics["out_of_order_accepted"],
            "tracklet_points": self.tracklets.metrics["points_stored"],
            "tracklet_bytes": self.tracklets.approximate_bytes,
            "retained_global_tracks": len(self.tracks),
            "dormant_identity_count": len(self.dormant_tracks),
            "pending_provisional_tracks": sum(
                gid < 0 and self.resolve_gid(gid) == gid for gid in self.tracks
            ),
            "public_identity_slots": sum(
                gid > 0 and self.resolve_gid(gid) == gid for gid in self.tracks
            ),
            "retained_aliases": len(self.aliases),
            "retained_affinities": len(self.affinity),
            "identity_events_written": self._diagnostics.events_written,
            "identity_events_dropped": self._diagnostics.events_dropped,
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
