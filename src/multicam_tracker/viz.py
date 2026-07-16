"""Rerun visualization: camera grid + live bird's-eye 3D world view.

Layout (built as a Rerun blueprint): the world view on the left — camera
frusta placed from the dataset's extrinsics, plus every fused track as a
colored dot with a fading trail — and the synchronized camera grid on the
right, where each detection box is painted in its *global* track color.
The same person wearing the same color in every panel is the money shot.

Entity paths:
    /world/cameras/{cam}      pinhole frusta (static)
    /world/floor              light reference grid (static)
    /world/tracks/{gid}       fused position + trail per global track
    /{cam}/image              camera frame (JPEG-compressed)
    /{cam}/boxes              per-camera boxes, colored by global ID
    /metrics/fps              pipeline throughput
"""

from __future__ import annotations

import colorsys

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .calibration import CameraCalibration
from .config import PipelineConfig


def color_for_id(gid: int) -> tuple[int, int, int]:
    """Deterministic, well-separated color per ID (golden-ratio hue walk)."""
    hue = (gid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def build_blueprint(cameras: list[str]) -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/world", name="World (bird's-eye)"),
            rrb.Grid(*[
                rrb.Spatial2DView(origin=f"/{cam}", name=cam) for cam in cameras
            ]),
            column_shares=[1.0, 1.2],
        ),
        collapse_panels=True,
    )


class RerunLogger:
    def __init__(
        self,
        cfg: PipelineConfig,
        calibrations: dict[str, CameraCalibration],
        save_path: str | None = None,
    ):
        self.cfg = cfg
        self._prev_gids: set[int] = set()  # gids logged last tick, for Clear()
        # Either spawn the interactive viewer or record to a .rrd file.
        rr.init("multicam_tracker", spawn=save_path is None)
        if save_path:
            rr.save(save_path)
        rr.send_blueprint(build_blueprint(list(calibrations)))

        rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self._log_floor_grid(calibrations)
        for cam, calib in calibrations.items():
            orientation, position = calib.pose_in_world()
            rr.log(
                f"/world/cameras/{cam}",
                rr.Transform3D(translation=position, mat3x3=orientation),
                static=True,
            )
            rr.log(
                f"/world/cameras/{cam}",
                rr.Pinhole(
                    image_from_camera=calib.K,
                    resolution=[calib.width, calib.height],
                    camera_xyz=rr.ViewCoordinates.RDF,  # OpenCV convention
                    image_plane_distance=1.5,
                ),
                static=True,
            )

    def _log_floor_grid(self, calibrations: dict[str, CameraCalibration]) -> None:
        """A light 1 m grid spanning the cameras' shared field of view."""
        corners = []
        for calib in calibrations.values():
            w, h = calib.width, calib.height
            image_rim = np.array([[w / 2, h], [0, h], [w, h], [0, h * 0.6], [w, h * 0.6]])
            corners.append(calib.image_to_world(image_rim))
        # Points near a camera's horizon can project absurdly far away;
        # clamp so one wide shot cannot blow up the grid extent.
        corners = np.clip(np.vstack(corners), -60, 60)
        (x0, y0), (x1, y1) = corners.min(axis=0) - 2, corners.max(axis=0) + 2
        lines = [[[x, y0, 0], [x, y1, 0]] for x in np.arange(np.floor(x0), x1, 1.0)]
        lines += [[[x0, y, 0], [x1, y, 0]] for y in np.arange(np.floor(y0), y1, 1.0)]
        rr.log(
            "/world/floor",
            rr.LineStrips3D(lines, colors=[[80, 80, 90]], radii=0.004),
            static=True,
        )

    def log_tick(
        self,
        frame_index: int,
        frames: dict[str, np.ndarray],
        camera_tracks: dict[str, list],   # cam -> list[bytetrack.Track]
        local_to_gid: dict[tuple[str, int], int],
        global_tracks: list,              # list[fusion.GlobalTrack]
        fps: float | None = None,
    ) -> None:
        rr.set_time("frame", sequence=frame_index)

        for cam, frame in frames.items():
            ok, jpeg = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality]
            )
            if ok:
                rr.log(f"/{cam}/image", rr.EncodedImage(contents=jpeg.tobytes(), media_type="image/jpeg"))

            tracks = camera_tracks.get(cam, [])
            boxes, colors, labels = [], [], []
            for track in tracks:
                gid = local_to_gid.get((cam, track.track_id))
                if gid is None:
                    continue  # not yet part of a confirmed global track
                x1, y1, x2, y2 = track.xyxy
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                colors.append(color_for_id(gid))
                labels.append(f"#{gid}")
            rr.log(
                f"/{cam}/boxes",
                rr.Boxes2D(array=np.array(boxes).reshape(-1, 4), array_format=rr.Box2DFormat.XYWH,
                           colors=colors or None, labels=labels or None),
            )

        # World view: one dot + trail per confirmed global track.
        seen_gids = set()
        for track in global_tracks:
            seen_gids.add(track.gid)
            color = color_for_id(track.gid)
            x, y = track.world_xy
            rr.log(
                f"/world/tracks/{track.gid}/dot",
                rr.Points3D([[x, y, 0.05]], colors=[color], radii=0.18,
                            labels=[f"#{track.gid} ({len(track.cameras)} cams)"]),
            )
            if len(track.trail) >= 2:
                trail = [[px, py, 0.05] for px, py in track.trail]
                rr.log(
                    f"/world/tracks/{track.gid}/trail",
                    rr.LineStrips3D([trail], colors=[color], radii=0.03),
                )
        # Clear entities of tracks that died so they vanish from the view.
        for gid in self._prev_gids - seen_gids:
            rr.log(f"/world/tracks/{gid}", rr.Clear(recursive=True))
        self._prev_gids = seen_gids

        if fps is not None:
            rr.log("/metrics/fps", rr.Scalars(fps))
