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

from .config import (
    DEFAULT_TRACKING_PROFILE,
    TRACKING_PROFILE_OVERRIDES,
    build_effective_config,
)
from .demo import run_pipeline
from .local_tracker import LOCAL_TRACKERS
from .viz import color_for_id

CELL_W, CELL_H = 480, 270          # camera cell in the grid
MAP_W = 540                        # bird's-eye panel width


def _draw_dashed_rectangle(
    image: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    color: tuple[int, int, int],
    *,
    thickness: int = 1,
    dash: int = 8,
) -> None:
    """Draw a predicted/occluded box without implying an observation."""
    x1, y1 = first
    x2, y2 = second
    segments = (
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    )
    for start, end in segments:
        length = int(round(np.hypot(end[0] - start[0], end[1] - start[1])))
        if length <= 0:
            continue
        for offset in range(0, length, 2 * dash):
            stop = min(length, offset + dash)
            alpha0 = offset / length
            alpha1 = stop / length
            p0 = (
                int(round(start[0] + alpha0 * (end[0] - start[0]))),
                int(round(start[1] + alpha0 * (end[1] - start[1]))),
            )
            p1 = (
                int(round(start[0] + alpha1 * (end[0] - start[0]))),
                int(round(start[1] + alpha1 * (end[1] - start[1]))),
            )
            cv2.line(image, p0, p1, color, thickness, cv2.LINE_AA)


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
            observed = bool(getattr(track, "cameras", set()))
            cv2.circle(
                img, (u, v), 7, bgr, -1 if observed else 2, cv2.LINE_AA
            )
            label = str(track.gid) if observed else f"{track.gid} OCC"
            cv2.putText(img, label, (u + 9, v + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1, cv2.LINE_AA)
        cv2.putText(img, "world ground plane (1 m grid)", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
        return img


def composite(tick: dict, bev: BevRenderer, cameras: list[str]) -> np.ndarray:
    cells = []
    for cam in cameras:
        source = tick["frames"].get(cam)
        if source is None:
            frame = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
            cv2.putText(frame, f"{cam}: no frame", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (200, 200, 200), 1)
            cells.append(frame)
            continue
        src_h, src_w = source.shape[:2]
        frame = cv2.resize(source, (CELL_W, CELL_H))
        sx, sy = CELL_W / src_w, CELL_H / src_h
        camera_render_tracks = tick.get(
            "camera_render_tracks", tick["camera_tracks"]
        )
        for track in camera_render_tracks.get(cam, []):
            gid = tick["local_to_gid"].get((cam, track.track_id))
            if gid is None:
                continue
            r, g, b = color_for_id(gid)
            x1, y1, x2, y2 = track.xyxy
            p1 = (int(x1 * sx), int(y1 * sy))
            p2 = (int(x2 * sx), int(y2 * sy))
            occluded = getattr(track, "misses", 0) > 0
            if occluded:
                _draw_dashed_rectangle(frame, p1, p2, (b, g, r))
            else:
                cv2.rectangle(frame, p1, p2, (b, g, r), 2)
            label = str(gid) if not occluded else f"{gid} OCC"
            cv2.putText(frame, label, (p1[0], p1[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r),
                        1 if occluded else 2, cv2.LINE_AA)
        cv2.putText(frame, cam, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (230, 230, 230), 1, cv2.LINE_AA)
        cells.append(frame)

    rows = [np.hstack(cells[i:i + 2]) for i in range(0, len(cells), 2)]
    if len(cells) % 2:  # pad an odd camera count with a black cell
        rows[-1] = np.hstack([cells[-1], np.zeros_like(cells[-1])])
    grid = np.vstack(rows)
    map_img = bev.render(tick.get("global_render_tracks", tick["global_tracks"]))
    if map_img.shape[:2] != (grid.shape[0], MAP_W):
        map_img = cv2.resize(map_img, (MAP_W, grid.shape[0]))
    return np.hstack([grid, map_img])


def render_video(
    *,
    data_root: Path,
    out: Path,
    start: int = 0,
    seconds: float = 20.0,
    stride: int = 1,
    device: str = "cpu",
    model_name: str | None = None,
    cameras: list[str] | None = None,
    tracking_profile: str = DEFAULT_TRACKING_PROFILE,
    local_tracker: str | None = None,
    expected_people: int | None = None,
    detector_fp16: bool = False,
    detector_compile: bool = False,
    gif: Path | None = None,
    gif_fps: int = 10,
) -> Path:
    """Run the tracking pipeline and write a camera-grid + BEV MP4.

    This programmatic entrypoint is shared by the local CLI and Modal's T4
    worker. Keeping one renderer avoids cloud/local output drift.
    """
    data_root = Path(data_root)
    out = Path(out)
    gif = Path(gif) if gif is not None else None
    if start < 0:
        raise ValueError("start must be >= 0")
    if seconds <= 0:
        raise ValueError("seconds must be > 0")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if gif_fps < 1:
        raise ValueError("gif_fps must be >= 1")
    if expected_people is not None and expected_people < 0:
        raise ValueError("expected_people must be >= 0")

    cfg = build_effective_config(
        data_root=data_root,
        stride=stride,
        device=device,
        async_ingest=False,
        fusion_output_time_policy="event",
        profile=tracking_profile,
        local_tracker=local_tracker,
    )
    if expected_people is not None:
        cfg.fusion_fixed_population_size = expected_people
    cfg.detector_fp16 = detector_fp16
    cfg.detector_compile = detector_compile
    if model_name is not None:
        cfg.model_name = model_name
    available = sorted(p.name for p in cfg.scene_dir.glob("camera_*") if p.is_dir())
    cfg.cameras = list(cameras) if cameras else available
    missing = sorted(set(cfg.cameras) - set(available))
    if missing:
        raise FileNotFoundError(f"Camera folders not found: {', '.join(missing)}")
    if not cfg.cameras:
        raise FileNotFoundError(f"No camera folders found under {cfg.scene_dir}")

    out.parent.mkdir(parents=True, exist_ok=True)
    if gif is not None:
        gif.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(cfg.scene_dir / cfg.cameras[0] / "video.mp4"))
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not capture.isOpened() or not np.isfinite(source_fps) or source_fps <= 0:
            raise RuntimeError("Cannot determine source video FPS")
    finally:
        capture.release()
    fps = source_fps / cfg.stride
    end_frame = start + int(round(seconds * source_fps))

    writer = None
    gif_stream = None
    next_gif_s = 0.0
    bev = None
    tick_count = 0
    try:
        for tick in run_pipeline(cfg, max_frames=end_frame):
            if tick["frame_index"] < start:
                continue
            tick_count += 1
            if bev is None:
                grid_h = CELL_H * ((len(cfg.cameras) + 1) // 2)
                bev = BevRenderer(tick["calibrations"], grid_h)
            frame = composite(tick, bev, cfg.cameras)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                    (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"OpenCV could not open MP4 writer for {out}")
            writer.write(frame)
            if gif is not None and (tick_count - 1) / fps + 1e-9 >= next_gif_s:
                # Encode one GIF frame at a time: no clip-sized image list.
                from PIL import GifImagePlugin, Image
                rgb = cv2.cvtColor(
                    cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2)),
                    cv2.COLOR_BGR2RGB,
                )
                paletted = Image.fromarray(rgb).quantize(colors=256)
                if gif_stream is None:
                    gif_stream = gif.open("wb")
                    header, _ = GifImagePlugin.getheader(paletted, info={"loop": 0})
                    for chunk in header:
                        gif_stream.write(chunk)
                for chunk in GifImagePlugin.getdata(
                    paletted, duration=round(1000 / min(gif_fps, fps)),
                    include_color_table=True,
                ):
                    gif_stream.write(chunk)
                next_gif_s += 1.0 / min(gif_fps, fps)
    finally:
        if writer is not None:
            writer.release()
        if gif_stream is not None:
            gif_stream.write(b";")
            gif_stream.close()

    if tick_count == 0 or not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError("No output frames were rendered; check --start and input videos")
    print(f"Wrote {out}")

    if gif is not None and gif_stream is not None:
        print(f"Wrote {gif} ({gif.stat().st_size / 1e6:.1f} MB)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("preview.mp4"))
    parser.add_argument("--gif", type=Path, default=None, help="Also write a GIF here")
    parser.add_argument("--start", type=int, default=0, help="Skip this many source frames")
    parser.add_argument("--seconds", type=float, default=20.0, help="Length of the clip")
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth source frame (default: 1)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                        help="Inference device (default: cpu)")
    parser.add_argument(
        "--local-tracker",
        choices=LOCAL_TRACKERS,
        default=None,
        help="Explicit per-camera tracker override (default: selected profile)",
    )
    parser.add_argument("--model", default=None,
                        help="YOLO weights path/name (default from PipelineConfig)")
    parser.add_argument("--cameras", nargs="+", default=None, metavar="CAM")
    parser.add_argument(
        "--profile",
        choices=sorted(TRACKING_PROFILE_OVERRIDES),
        default=DEFAULT_TRACKING_PROFILE,
    )
    parser.add_argument("--expected-people", type=int, default=None)
    parser.add_argument("--gif-fps", type=int, default=10)
    args = parser.parse_args()

    render_video(
        data_root=args.data_root,
        out=args.out,
        start=args.start,
        seconds=args.seconds,
        stride=args.stride,
        device=args.device,
        model_name=args.model,
        cameras=args.cameras,
        tracking_profile=args.profile,
        local_tracker=args.local_tracker,
        expected_people=args.expected_people,
        gif=args.gif,
        gif_fps=args.gif_fps,
    )


if __name__ == "__main__":
    main()
