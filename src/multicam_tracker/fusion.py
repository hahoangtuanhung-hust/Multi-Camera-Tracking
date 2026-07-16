"""Cross-camera fusion: many per-camera tracks -> one global ID per person.

This is the heart of the demo — the part DeepStream calls Multi-View 3D
Tracking (MV3DT). The insight: once every camera projects its tracks onto
the *same* world ground plane (via its homography), "is camera A's track 7
the same person as camera B's track 3?" becomes a simple geometry question:
**are they standing in the same spot?**

Each tick:

1. **Sticky re-association** — a (camera, local_id) pair that already
   belongs to a global track keeps it, as long as geometry still agrees.
   This is the cheap, reliable path that handles ~95% of observations.
   (MV3DT's analog: peers negotiate and then *keep* a global ID.)
2. **Clustering** — remaining observations that stand near each other on
   the floor (and come from *different* cameras) are grouped: they are
   probably the same physical person seen from several views.
3. **Hungarian assignment** — clusters are matched to existing global
   tracks by predicted world distance, gated so far-away matches are
   rejected. (MV3DT's analog: tracklet matching by RMSE distance.)
4. **Kalman fusion + lifecycle** — matched tracks fuse all their cameras'
   measurements (mean -> Kalman update); unmatched clusters birth tentative
   tracks; unseen tracks age and die. Births need `fusion_min_hits`
   consecutive support so single-frame ghosts never show up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .bytetrack import ConstantVelocityKalman
from .config import PipelineConfig


@dataclass
class CameraObservation:
    """One confirmed per-camera track, projected to the world floor."""
    camera: str
    local_id: int
    world_xy: np.ndarray  # (2,) meters on the ground plane
    bbox_xyxy: np.ndarray
    conf: float


@dataclass
class GlobalTrack:
    gid: int
    kf: ConstantVelocityKalman
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    cameras: set[str] = field(default_factory=set)   # views supporting it this tick
    trail: list[np.ndarray] = field(default_factory=list)

    @property
    def world_xy(self) -> np.ndarray:
        return self.kf.position


class GlobalTrackManager:
    """Fuses per-camera observations into globally consistent tracks."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.tracks: dict[int, GlobalTrack] = {}
        # (camera, local_id) -> gid. This mapping IS a fusion output: it is
        # what lets the viewer paint per-camera boxes in global-ID colors.
        self.affinity: dict[tuple[str, int], int] = {}
        self._next_gid = 1

    def step(self, observations: list[CameraObservation]) -> list[GlobalTrack]:
        """Advance one tick. Returns confirmed global tracks."""
        cfg = self.cfg
        for track in self.tracks.values():
            track.kf.predict()
            track.cameras = set()

        # -- 1. Sticky re-association ------------------------------------
        # An observation whose (camera, local_id) already maps to a live
        # global track keeps it — unless it has drifted implausibly far,
        # which usually means the local tracker swapped IDs.
        assigned: dict[int, list[CameraObservation]] = {}
        pending: list[CameraObservation] = []
        for obs in observations:
            gid = self.affinity.get((obs.camera, obs.local_id))
            if gid is not None and gid in self.tracks:
                drift = np.linalg.norm(self.tracks[gid].world_xy - obs.world_xy)
                if drift < 2 * cfg.fusion_gate_m:
                    assigned.setdefault(gid, []).append(obs)
                    continue
                del self.affinity[(obs.camera, obs.local_id)]  # stale link
            pending.append(obs)

        # -- 2. Cluster the rest by floor proximity ----------------------
        clusters = self._cluster(pending)

        # -- 3. Hungarian: clusters vs. global tracks --------------------
        gids = list(self.tracks)
        if clusters and gids:
            predicted = np.stack([self.tracks[g].world_xy for g in gids])
            centroids = np.stack([
                np.mean([o.world_xy for o in cluster], axis=0) for cluster in clusters
            ])
            dist = np.linalg.norm(predicted[:, None] - centroids[None, :], axis=2)
            rows, cols = linear_sum_assignment(dist)
            for r, c in zip(rows, cols):
                if dist[r, c] < cfg.fusion_gate_m:
                    # Keep the one-observation-per-camera invariant: if the
                    # track already got a sticky observation from a camera,
                    # drop that camera's duplicate from the cluster.
                    taken = {o.camera for o in assigned.get(gids[r], [])}
                    novel = [o for o in clusters[c] if o.camera not in taken]
                    if novel:
                        assigned.setdefault(gids[r], []).extend(novel)
                    clusters[c] = None
            clusters = [c for c in clusters if c is not None]

        # -- 4a. Update matched tracks with fused measurements -----------
        for gid, obs_list in assigned.items():
            track = self.tracks[gid]
            fused = np.mean([o.world_xy for o in obs_list], axis=0)
            track.kf.update(fused)
            track.hits += 1
            track.misses = 0
            if track.hits >= cfg.fusion_min_hits:
                track.confirmed = True
            for obs in obs_list:
                track.cameras.add(obs.camera)
                self.affinity[(obs.camera, obs.local_id)] = gid

        # -- 4b. Unmatched clusters birth tentative tracks ---------------
        for cluster in clusters:
            centroid = np.mean([o.world_xy for o in cluster], axis=0)
            gid = self._next_gid
            self._next_gid += 1
            track = GlobalTrack(
                gid=gid,
                kf=ConstantVelocityKalman(
                    centroid,
                    process_noise=cfg.fusion_kf_process_noise,
                    measurement_noise=cfg.fusion_kf_measurement_noise,
                ),
                cameras={o.camera for o in cluster},
            )
            self.tracks[gid] = track
            for obs in cluster:
                self.affinity[(obs.camera, obs.local_id)] = gid

        # -- 4c. Lifecycle: age out unseen tracks ------------------------
        for gid in list(self.tracks):
            track = self.tracks[gid]
            if not track.cameras:
                track.misses += 1
                # Tentative tracks die instantly; confirmed ones tolerate
                # fusion_max_age ticks of total occlusion.
                if track.misses > (cfg.fusion_max_age if track.confirmed else 0):
                    del self.tracks[gid]
                    self.affinity = {k: g for k, g in self.affinity.items() if g != gid}

        visible = [t for t in self.tracks.values() if t.confirmed and t.misses == 0]
        for track in visible:
            track.trail.append(track.world_xy.copy())
            del track.trail[: -cfg.trail_length]
        return visible

    def _cluster(self, observations: list[CameraObservation]) -> list[list[CameraObservation]]:
        """Greedily merge observations closer than `fusion_merge_m` on the floor.

        Two observations from the *same* camera are never merged — one camera
        reports each person at most once.
        """
        clusters = [[obs] for obs in observations]
        merged = True
        while merged:
            merged = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    ci, cj = clusters[i], clusters[j]
                    cameras_i = {o.camera for o in ci}
                    if any(o.camera in cameras_i for o in cj):
                        continue
                    dist = np.linalg.norm(
                        np.mean([o.world_xy for o in ci], axis=0)
                        - np.mean([o.world_xy for o in cj], axis=0)
                    )
                    if dist < self.cfg.fusion_merge_m:
                        clusters[i] = ci + cj
                        del clusters[j]
                        merged = True
                        break
                if merged:
                    break
        return clusters


def foot_point_xyxy(bbox_xyxy: np.ndarray) -> np.ndarray:
    """Bottom-center of an xyxy bbox — the pixel where the person meets the floor."""
    x1, y1, x2, y2 = bbox_xyxy[:4]
    return np.array([(x1 + x2) / 2, y2])
