"""Unique private-identity accounting and opt-in bounded terminal association.

The ledger distinguishes identity outcomes from attempts and observations.
Only the active ledger and a bounded terminal-report ring are retained. Decision
deltas are drained by consumers; they are never inferred from counter totals.
"""
from __future__ import annotations

from collections import Counter, deque
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from .appearance import appearance_similarity

if TYPE_CHECKING:
    from .fusion import GlobalTrackManager


class IdentityLifecycle:
    def __init__(self, limit: int = 10000):
        self.pending: dict[int, dict] = {}
        self.completed: deque[dict] = deque(maxlen=limit)
        self.decisions: dict[int, dict] = {}
        self.counts: Counter = Counter()

    def birth(self, gid, timestamp, source):
        self.pending[gid] = {
            "provisional_gid": gid, "internal_gid": gid,
            "created_ts_s": timestamp, "last_seen_ts_s": timestamp,
            "sources": [list(source)], "state": "PRIVATE_TRACKLET",
            "candidates": [], "confirmations": 0, "defer_reason": None,
        }
        self.counts["created_unique"] += 1

    def finish(self, gid, outcome, timestamp, reason, public_gid=None):
        # Idempotence is essential: terminal counters are unique identities.
        for original, record in list(self.pending.items()):
            if record["internal_gid"] != gid:
                continue
            self.pending.pop(original)
            record.update(state=outcome, decision_ts_s=timestamp,
                          decision_reason=reason, public_gid=public_gid)
            self.completed.append(record)
            self.decisions[original] = {
                "outcome": outcome, "public_gid": public_gid, "reason": reason,
            }
            self.counts[outcome] += 1

    def merge(self, retired, canonical, timestamp, reason):
        if canonical > 0:
            self.finish(retired, "BOUND_EXISTING_GID", timestamp, reason, canonical)
        else:
            for record in self.pending.values():
                if record["internal_gid"] == retired:
                    record["internal_gid"] = canonical

    def annotate(self, manager):
        for record in self.pending.values():
            track = manager.tracks.get(record["internal_gid"])
            if track is not None:
                record["last_seen_ts_s"] = track.last_seen_ts_s
                sources = {tuple(key) for key in record["sources"]}
                sources.update(manager.members[track.gid])
                record["sources"] = [list(key) for key in sorted(sources)]

    def evidence(self, gid, candidates, reason, confirmations=0):
        for record in self.pending.values():
            if record["internal_gid"] == gid:
                record.update(state="BIND_PENDING", candidates=candidates,
                              defer_reason=reason, confirmations=confirmations)

    def drain(self):
        result, self.decisions = self.decisions, {}
        return result

    def report(self):
        terminal_count = sum(value for key, value in self.counts.items()
                             if key != "created_unique")
        return {
            "unique_counts": dict(self.counts), "pending_unique": len(self.pending),
            "conservation_ok": self.counts["created_unique"] == terminal_count + len(self.pending),
            "records_omitted": max(0, terminal_count - len(self.completed)),
            "records": list(self.completed) + list(self.pending.values()),
        }


class TerminalBinding:
    """Evidence-gated assignment; expiry alone never proves a new person."""
    def __init__(self, manager: GlobalTrackManager):
        self.m = manager
        self.votes: dict[int, tuple[int, int, float]] = {}

    def _pair(self, owner, private):
        m, cfg = self.m, self.m.cfg
        timestamp = float(private.last_seen_ts_s)
        samples = m._ordered_track_samples(private)
        point = samples[-1][1] if samples else private.world_xy
        evidence = {"candidate_gid": owner.gid,
                    "gap_s": timestamp - float(owner.last_seen_ts_s),
                    "entry_evidence": bool(private.birth_near_boundary)}
        if m._has_same_camera_conflict(owner.gid, private.gid):
            return None, {**evidence, "rejection": "same_camera_conflict"}
        # Do not extrapolate backwards from a future owner observation.
        if timestamp < float(owner.state_timestamp_s) - cfg.fusion_event_time_gate_s:
            return None, {**evidence, "rejection": "owner_newer_than_fragment"}
        feasible = m._motion_feasibility(owner, point, timestamp)
        evidence.update(distance_m=feasible.distance_m,
                        allowed_distance_m=feasible.allowed_distance_m,
                        motion_reason=feasible.reason)
        if not feasible.feasible or feasible.distance_m > cfg.fusion_kinematic_match_gate_m:
            return None, {**evidence, "rejection": "motion"}
        mean, covariance = owner.kf.projected_measurement(
            max(0.0, timestamp - float(owner.state_timestamp_s)))
        innovation = point - mean
        try:
            nis = float(innovation @ np.linalg.solve(covariance, innovation))
        except np.linalg.LinAlgError:
            return None, {**evidence, "rejection": "singular_covariance"}
        evidence["nis"] = nis if np.isfinite(nis) else None
        if not np.isfinite(nis) or nis > cfg.fusion_terminal_nis_gate:
            return None, {**evidence, "rejection": "nis"}
        left, right = m._trajectory_motion(owner, timestamp), m._trajectory_motion(private, timestamp)
        ls, rs = float(np.linalg.norm(left.velocity)), float(np.linalg.norm(right.velocity))
        direction = 0.0
        if min(ls, rs) >= cfg.fusion_stationary_speed_mps:
            cosine = float(np.clip(np.dot(left.velocity, right.velocity) / (ls * rs), -1, 1))
            evidence["direction_cosine"] = cosine
            if cosine < cfg.fusion_trajectory_min_direction_cosine:
                return None, {**evidence, "rejection": "direction"}
            direction = 1.0 - cosine
        acceleration = 0.0
        if left.acceleration_reliable and right.acceleration_reliable:
            acceleration = float(np.linalg.norm(left.acceleration - right.acceleration))
        if max(ls, rs) > cfg.fusion_max_speed_mps:
            return None, {**evidence, "rejection": "speed"}
        similarities = []
        for camera in sorted(owner.appearance_by_camera.keys() & private.appearance_by_camera.keys()):
            score = appearance_similarity(owner.appearance_by_camera[camera], private.appearance_by_camera[camera])
            if score is not None:
                similarities.append(score)
        similarity = min(similarities) if similarities else None
        evidence.update(appearance_similarity=similarity, velocity_error=abs(ls-rs),
                        acceleration_error=acceleration)
        if similarity is not None and similarity < cfg.fusion_appearance_conflict_min_similarity:
            return None, {**evidence, "rejection": "appearance_conflict"}
        cost = (0.45 * feasible.distance_m / max(cfg.fusion_kinematic_match_gate_m, 1e-9)
                + 0.15 * nis / cfg.fusion_terminal_nis_gate
                + 0.15 * abs(ls-rs) / cfg.fusion_max_speed_mps
                + 0.10 * direction + 0.05 * min(acceleration, 2.0)
                + 0.05 * min(abs(evidence["gap_s"]) / max(cfg.fusion_dormant_retention_s, 1e-9), 1.0)
                + (0.05 * (1-similarity) if similarity is not None else 0.0))
        evidence["cost"] = float(cost)
        return float(cost), evidence

    def step(self, *, final=False):
        m, cfg = self.m, self.m.cfg
        now = m._max_event_ts_s
        if not np.isfinite(now):
            return
        m.lifecycle.annotate(m)
        private = [t for gid, t in sorted(m.tracks.items()) if gid < 0]
        owners = [t for gid, t in sorted(m.tracks.items())
                  if gid > 0 and t.emitted and t.last_seen_ts_s is not None
                  and now-t.last_seen_ts_s <= max(cfg.fusion_identity_retention_s, cfg.fusion_dormant_retention_s)]
        valid = {t.gid: t.confirmed and m._verify_trajectory_plausibility(t) for t in private}
        matrix = np.full((len(private), len(owners)), 1e6)
        details: dict[int, list[dict]] = {}
        for row, track in enumerate(private):
            candidates = []
            for col, owner in enumerate(owners):
                cost, evidence = self._pair(owner, track)
                candidates.append(evidence)
                if cost is not None:
                    matrix[row, col] = cost
            details[track.gid] = candidates
        selected = {}
        if cfg.fusion_terminal_batch_enabled and matrix.size:
            rows, columns = linear_sum_assignment(matrix)
            for row, col in zip(rows, columns):
                cost = matrix[row, col]
                if cost >= 1e6 or not valid[private[row].gid]:
                    continue
                alternatives = [matrix[row, k] for k in range(len(owners)) if k != col and matrix[row,k] < 1e6]
                alternatives += [matrix[k, col] for k in range(len(private)) if k != row and matrix[k,col] < 1e6]
                margin = min(alternatives, default=float("inf")) - cost
                gid, owner_gid = private[row].gid, owners[col].gid
                evidence = next(e for e in details[gid] if e["candidate_gid"] == owner_gid)
                evidence["assignment_margin"] = float(margin) if np.isfinite(margin) else None
                if cost > cfg.fusion_terminal_max_cost or margin < cfg.fusion_terminal_margin:
                    continue
                # Confirmation is based on new observations, not repeated calls
                # or an end-of-stream flush of the same observation.
                event_ts = float(private[row].last_seen_ts_s)
                previous = self.votes.get(gid)
                count = 1
                if previous and previous[0] == owner_gid:
                    if event_ts == previous[2]:
                        count = previous[1]
                    elif 0 < event_ts-previous[2] <= cfg.fusion_terminal_confirmation_window_s:
                        count = previous[1]+1
                self.votes[gid] = owner_gid, count, event_ts
                selected[gid] = owner_gid, count, evidence
        for gid in list(self.votes):
            if gid not in selected:
                del self.votes[gid]
        for row, track in enumerate(private):
            gid = track.gid
            if gid not in m.tracks:
                continue
            proposal = selected.get(gid)
            confirmations = proposal[1] if proposal else 0
            plausible_owner = bool(np.any(matrix[row] < 1e6))
            age = now - float(track.created_at_s)
            deadline = cfg.fusion_late_identity_max_defer_s + cfg.fusion_terminal_grace_s
            m.lifecycle.evidence(gid, details[gid], "owner_evidence" if plausible_owner else "birth_evidence", confirmations)
            if proposal and confirmations >= cfg.fusion_terminal_confirmations:
                owner_gid, _, evidence = proposal
                if m.merge_gids(owner_gid, gid, reason="terminal_batch_binding", evidence=evidence):
                    m.metrics["terminal_bindings_accepted"] += 1
                    self.votes.pop(gid, None)
                    continue
            if plausible_owner and not final and age < deadline - 1e-9:
                m.metrics["terminal_binding_defer_attempts"] += 1
                continue
            if not final and age < cfg.fusion_provisional_min_age_s - 1e-9:
                continue
            capacity = cfg.fusion_fixed_population_size
            capacity_ok = capacity <= 0 or sum(g > 0 for g in m.tracks) < capacity
            entry_ok = (m._has_world_entry_evidence(track) if cfg.fusion_world_entry_zones
                        else track.birth_near_boundary or not cfg.fusion_new_gid_entry_required)
            if valid[gid] and not plausible_owner and entry_ok and capacity_ok:
                m._promote_provisional_gid(gid)
                continue
            # A malformed but still young trajectory can acquire evidence.
            if not final and age < deadline - 1e-9:
                continue
            reason = ("ambiguous_owner_at_deadline" if plausible_owner else
                      "capacity_limit" if not capacity_ok else
                      "entry_evidence_missing" if not entry_ok else
                      "insufficient_or_invalid_trajectory")
            m.lifecycle.finish(gid, "REJECTED_FALSE_TRACK", now, reason)
            # REJECTED_FALSE_TRACK is the requested schema label, not GT truth:
            # ambiguity/insufficient evidence may also censor a real person.
            m._identity_event("terminal_identity_rejected", gid=gid, reason=reason)
            m.tracks.pop(gid, None)
            m.members.pop(gid, None)
            m.dormant_tracks.pop(gid, None)
            for key, mapped in list(m.affinity.items()):
                if m.resolve_gid(mapped) == gid:
                    del m.affinity[key]
            self.votes.pop(gid, None)
