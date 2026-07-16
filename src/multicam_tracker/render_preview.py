"""Render a shareable MP4/GIF preview: camera grid + bird's-eye map.

The live Rerun viewer is the primary experience, but a video file is what
you can embed in a README or post online. This composites each tick into
a single frame — cameras on the left, world map on the right — using only
OpenCV drawing, then writes an MP4 (always) and optionally a GIF.

Run: ``uv run render-preview --start 0 --seconds 20``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig
from .demo import run_pipeline
from .viz import color_for_id

CELL_W, CELL_H = 480, 270          # camera cell in the grid
MAP_W = 540                        # bird's-eye panel width


class BevRenderer:
    """Draws world-plane tracks onto a top-down map image."""

    def __init__(self, calibrations: dict, height: int):
        # Fit the map to the cameras' combined coverage.
        pts = []
        for calib in calibrations.values():
            w, h = calib.width, calib.height
            rim = np.array([[0, h], [w, h], [w / 2, h], [0, h * 0.55], [w, h * 0.55]])
            pts.append(calib.image_to_world(rim))
        # Clamp horizon-adjacent projections so the map extent stays sane.
        pts = np.clip(np.vstack(pts), -60, 60)
        (self.x0, self.y0) = pts.min(axis=0) - 1.5
        (x1, y1) = pts.max(axis=0) + 1.5
        self.scale = min(MAP_W / (x1 - self.x0), height / (y1 - self.y0))
        self.size = (MAP_W, height)
        self.calibrations = calibrations

    def to_px(self, xy: np.ndarray) -> tuple[int, int]:
        u = (xy[0] - self.x0) * self.scale
        v = self.size[1] - (xy[1] - self.y0) * self.scale  # world +Y = map up
        return int(u), int(v)

    def render(self, global_tracks: list) -> np.ndarray:
        img = np.full((self.size[1], self.size[0], 3), 24, dtype=np.uint8)
        # 1 m grid
        for x in np.arange(np.floor(self.x0), self.x0 + self.size[0] / self.scale + 1):
            u, _ = self.to_px(np.array([x, self.y0]))
            cv2.line(img, (u, 0), (u, self.size[1]), (45, 45, 45), 1)
        for y in np.arange(np.floor(self.y0), self.y0 + self.size[1] / self.scale + 1):
            _, v = self.to_px(np.array([self.x0, y]))
            cv2.line(img, (0, v), (self.size[0], v), (45, 45, 45), 1)
        # camera positions
        for calib in self.calibrations.values():
            _, pos = calib.pose_in_world()
            u, v = self.to_px(pos[:2])
            cv2.drawMarker(img, (u, v), (200, 200, 200), cv2.MARKER_TRIANGLE_UP, 14, 2)
        # tracks: trail + dot + id
        for track in global_tracks:
            r, g, b = color_for_id(track.gid)
            bgr = (b, g, r)
            trail = [self.to_px(p) for p in track.trail]
            for p, q in zip(trail, trail[1:]):
                cv2.line(img, p, q, bgr, 2, cv2.LINE_AA)
            u, v = self.to_px(track.world_xy)
            cv2.circle(img, (u, v), 7, bgr, -1, cv2.LINE_AA)
            cv2.putText(img, str(track.gid), (u + 9, v + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1, cv2.LINE_AA)
        cv2.putText(img, "world ground plane (1 m grid)", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
        return img


def composite(tick: dict, bev: BevRenderer, cameras: list[str]) -> np.ndarray:
    cells = []
    for cam in cameras:
        src_h, src_w = tick["frames"][cam].shape[:2]
        frame = cv2.resize(tick["frames"][cam], (CELL_W, CELL_H))
        sx, sy = CELL_W / src_w, CELL_H / src_h
        for track in tick["camera_tracks"].get(cam, []):
            gid = tick["local_to_gid"].get((cam, track.track_id))
            if gid is None:
                continue
            r, g, b = color_for_id(gid)
            x1, y1, x2, y2 = track.xyxy
            p1 = (int(x1 * sx), int(y1 * sy))
            p2 = (int(x2 * sx), int(y2 * sy))
            cv2.rectangle(frame, p1, p2, (b, g, r), 2)
            cv2.putText(frame, str(gid), (p1[0], p1[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 2, cv2.LINE_AA)
        cv2.putText(frame, cam, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (230, 230, 230), 1, cv2.LINE_AA)
        cells.append(frame)

    rows = [np.hstack(cells[i:i + 2]) for i in range(0, len(cells), 2)]
    if len(cells) % 2:  # pad an odd camera count with a black cell
        rows[-1] = np.hstack([cells[-1], np.zeros_like(cells[-1])])
    grid = np.vstack(rows)
    map_img = bev.render(tick["global_tracks"])
    map_img = cv2.resize(map_img, (MAP_W, grid.shape[0]))
    return np.hstack([grid, map_img])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("preview.mp4"))
    parser.add_argument("--gif", type=Path, default=None, help="Also write a GIF here")
    parser.add_argument("--start", type=int, default=0, help="Skip this many source frames")
    parser.add_argument("--seconds", type=float, default=20.0, help="Length of the clip")
    parser.add_argument("--gif-fps", type=int, default=10)
    args = parser.parse_args()

    cfg = PipelineConfig(data_root=args.data_root)
    cfg.cameras = sorted(p.name for p in cfg.scene_dir.glob("camera_*") if p.is_dir())
    end_frame = args.start + int(args.seconds * 30)

    writer = None
    gif_frames = []
    bev = None
    tick_count = 0
    for tick in run_pipeline(cfg, max_frames=end_frame):
        if tick["frame_index"] < args.start:
            continue
        tick_count += 1
        if bev is None:
            grid_h = CELL_H * ((len(cfg.cameras) + 1) // 2)
            bev = BevRenderer(tick["calibrations"], grid_h)
        frame = composite(tick, bev, cfg.cameras)
        if writer is None:
            fps = 30 / cfg.stride
            writer = cv2.VideoWriter(
                str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                (frame.shape[1], frame.shape[0]),
            )
        writer.write(frame)
        if args.gif and tick_count % 3 == 1:  # keep every 3rd tick for the GIF
            gif_frames.append(cv2.cvtColor(
                cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2)),
                cv2.COLOR_BGR2RGB,
            ))
    if writer:
        writer.release()
        print(f"Wrote {args.out}")

    if args.gif and gif_frames:
        from PIL import Image
        images = [Image.fromarray(f) for f in gif_frames]
        images[0].save(
            args.gif, save_all=True, append_images=images[1:],
            duration=int(1000 / args.gif_fps), loop=0, optimize=True,
        )
        print(f"Wrote {args.gif} ({args.gif.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
