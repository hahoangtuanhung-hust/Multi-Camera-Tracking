"""Bounded offline commit for late-bound global identities.

The fusion manager may keep a fragment under a private negative ID until a
later promotion or merge decides its canonical public owner. This module
buffers only small world-position/source snapshots, never video frames, and
applies alias decisions before predictions become immutable evaluation output.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

TrackletKey = tuple[str, int]


@dataclass(frozen=True)
class CommittedIdentityFrame:
    frame_index: int
    capture_ts_s: float
    predictions: dict[int, np.ndarray]
    sources: dict[int, list[TrackletKey]]


@dataclass
class _PendingIdentityFrame:
    frame_index: int
    capture_ts_s: float
    hypotheses: dict[int, np.ndarray]
    sources: dict[int, list[TrackletKey]]


class IdentityCommitBuffer:
    """Canonicalize delayed identity snapshots using accumulated aliases."""

    def __init__(self, delay_s: float = 0.0) -> None:
        if not np.isfinite(delay_s) or delay_s < 0:
            raise ValueError("delay_s must be a finite value >= 0")
        self.delay_s = float(delay_s)
        self._pending: deque[_PendingIdentityFrame] = deque()
        self._aliases: dict[int, int] = {}
        self.metrics: defaultdict[str, int] = defaultdict(int)
        self._decisions: dict[int, dict] = {}
        self._terminal_contract = False
        self._finalizer = None
        self.lifecycle_report: dict = {}
        self.final_pipeline_metrics: dict = {}
        self._frame_outcomes: defaultdict[int, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))

    def push_tick(self, tick: Mapping[str, object]) -> list[CommittedIdentityFrame]:
        """Add one pipeline tick and return frames whose delay has elapsed."""
        frame_index = int(tick["frame_index"])
        capture_ts_s = float(tick["capture_ts_s"])
        self._terminal_contract = bool(tick.get("identity_terminal_contract", False))
        self._finalizer = tick.get("identity_finalizer")
        self._consume_decisions(tick.get("identity_decisions", {}))
        aliases = tick.get("identity_aliases", {})
        if isinstance(aliases, Mapping):
            self._aliases.update({int(k): int(v) for k, v in aliases.items()})

        raw_hypotheses = tick.get("identity_hypotheses")
        if raw_hypotheses is None:
            raw_hypotheses = {
                int(track.gid): np.asarray(track.world_xy, dtype=float).copy()
                for track in tick.get("global_tracks", [])
            }
        hypotheses = {
            int(gid): np.asarray(position, dtype=float).copy()
            for gid, position in raw_hypotheses.items()
        }

        sources: defaultdict[int, list[TrackletKey]] = defaultdict(list)
        internal_affinity = tick.get("local_to_internal_gid", {})
        observations_present = "observations" in tick
        observations = tick.get("observations", [])
        observed_keys = {
            (str(observation.camera), int(observation.local_id))
            for observation in observations
        }
        if isinstance(internal_affinity, Mapping):
            for raw_key, raw_gid in internal_affinity.items():
                key = (str(raw_key[0]), int(raw_key[1]))
                if observations_present and key not in observed_keys:
                    continue
                sources[int(raw_gid)].append(key)

        self._pending.append(_PendingIdentityFrame(
            frame_index=frame_index,
            capture_ts_s=capture_ts_s,
            hypotheses=hypotheses,
            sources=dict(sources),
        ))
        self.metrics["identity_commit_frames_buffered"] += 1
        return self._drain(capture_ts_s, force=False)

    def flush(self) -> list[CommittedIdentityFrame]:
        """Commit every remaining frame using the final known alias graph."""
        if callable(self._finalizer):
            final = self._finalizer()
            self._aliases.update(final["identity_aliases"])
            self._consume_decisions(final["identity_decisions"])
            self.lifecycle_report = final["identity_lifecycle"]
            self.final_pipeline_metrics = final["metrics"]
        result = self._drain(float("inf"), force=True)
        for record in self.lifecycle_report.get("records", []):
            record["frame_outcomes"] = dict(self._frame_outcomes.get(record["provisional_gid"], {}))
        return result

    def _consume_decisions(self, decisions) -> None:
        for gid, decision in decisions.items():
            self._decisions[int(gid)] = dict(decision)
            if decision.get("public_gid") is not None:
                self._aliases[int(gid)] = int(decision["public_gid"])

    def _drain(
        self, watermark_s: float, *, force: bool
    ) -> list[CommittedIdentityFrame]:
        committed: list[CommittedIdentityFrame] = []
        while self._pending:
            frame = self._pending[0]
            if (
                not force
                and watermark_s - frame.capture_ts_s + 1e-9 < self.delay_s
            ):
                break
            self._pending.popleft()
            committed.append(self._canonicalize(frame))
        return committed

    def _resolve_gid(self, gid: int) -> int:
        path: list[int] = []
        current = int(gid)
        seen: set[int] = set()
        while current in self._aliases and current not in seen:
            seen.add(current)
            path.append(current)
            current = int(self._aliases[current])
        for alias in path:
            self._aliases[alias] = current
        return current

    def _canonicalize(
        self, frame: _PendingIdentityFrame
    ) -> CommittedIdentityFrame:
        grouped_positions: defaultdict[int, list[tuple[int, np.ndarray]]] = (
            defaultdict(list)
        )
        for internal_gid, position in frame.hypotheses.items():
            canonical_gid = self._resolve_gid(internal_gid)
            if canonical_gid <= 0:
                decision = self._decisions.get(internal_gid) or self._decisions.get(canonical_gid)
                if decision and decision["outcome"] == "REJECTED_FALSE_TRACK":
                    self.metrics["identity_commit_intentionally_rejected_hypotheses"] += 1
                    self._frame_outcomes[internal_gid]["rejected"] += 1
                    continue
                if self._terminal_contract:
                    # A missing decision is a contract violation, not a score
                    # improvement obtained by renaming an unresolved counter.
                    raise RuntimeError(f"No terminal decision for private GID {internal_gid} at frame {frame.frame_index}")
                self.metrics["identity_commit_unresolved_hypotheses_dropped"] += 1
                self._frame_outcomes[internal_gid]["unresolved"] += 1
                continue
            if internal_gid < 0:
                self.metrics["identity_commit_backfilled_hypotheses"] += 1
                self._frame_outcomes[internal_gid]["backfilled"] += 1
            grouped_positions[canonical_gid].append((internal_gid, position))

        predictions: dict[int, np.ndarray] = {}
        for canonical_gid, candidates in grouped_positions.items():
            direct = next(
                (position for gid, position in candidates if gid == canonical_gid),
                None,
            )
            if direct is not None:
                predictions[canonical_gid] = direct.copy()
            else:
                predictions[canonical_gid] = np.mean(
                    [position for _, position in candidates], axis=0
                )
            if len(candidates) > 1:
                self.metrics["identity_commit_duplicate_hypotheses_collapsed"] += (
                    len(candidates) - 1
                )

        source_sets: defaultdict[int, set[TrackletKey]] = defaultdict(set)
        for internal_gid, keys in frame.sources.items():
            canonical_gid = self._resolve_gid(internal_gid)
            if canonical_gid > 0:
                source_sets[canonical_gid].update(keys)
        sources = {
            gid: sorted(keys)
            for gid, keys in source_sets.items()
        }
        self.metrics["identity_commit_frames_committed"] += 1
        return CommittedIdentityFrame(
            frame_index=frame.frame_index,
            capture_ts_s=frame.capture_ts_s,
            predictions=predictions,
            sources=sources,
        )


def append_committed_frames(
    frames: Iterable[CommittedIdentityFrame],
    predictions: list[tuple[int, dict[int, np.ndarray]]],
    source_maps: dict[int, dict[int, list[TrackletKey]]],
) -> None:
    """Append committed frames to the evaluator's existing containers."""
    for frame in frames:
        predictions.append((frame.frame_index, frame.predictions))
        source_maps[frame.frame_index] = frame.sources
