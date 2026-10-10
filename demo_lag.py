import argparse
import time
from pathlib import Path
from typing import Iterator
import numpy as np

# Thêm thư mục src vào path để có thể import các module của dự án gốc
import sys
sys.path.insert(0, str(Path("src").resolve()))

from multicam_tracker.video import SyncedVideoReader
from multicam_tracker.calibration import load_calibrations
from multicam_tracker.config import PipelineConfig
from multicam_tracker.detector import PersonDetector
from multicam_tracker.fusion import CameraObservation, GlobalTrackManager, foot_point_xyxy
from multicam_tracker.viz import RerunLogger

class LagSimulatedVideoReader(SyncedVideoReader):
    """Kế thừa bộ đọc gốc để chèn thêm tính năng giả lập lag (đóng băng hình ảnh)."""
    def __init__(self, scene_dir: Path, cameras: list[str], stride: int, 
                 lag_cam: str, lag_start: int, lag_end: int):
        super().__init__(scene_dir, cameras, stride=stride)
        self.lag_cam = lag_cam
        self.lag_start = lag_start
        self.lag_end = lag_end

                # --- THÊM ĐOẠN NÀY ĐỂ TEST CẮT FRAME ---
        frames_to_skip = 90  # Cắt bỏ 90 frame (tương đương đi trước 3 giây)
        print(f"✂️ Đang cắt bỏ {frames_to_skip} frame ở đầu của {self.lag_cam}...")
        
        if self.lag_cam in self.captures:
            cap = self.captures[self.lag_cam]
            for _ in range(frames_to_skip):
                cap.grab() # grab() đọc xả băng rất nhanh mà không tốn CPU giải mã màu
        # --------------------------------------

    def __iter__(self) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        frame_index = 0
        last_frames = {}
        
        while True:
            frames = {}
            for cam, capture in self.captures.items():
                is_lagging = (cam == self.lag_cam and self.lag_start <= frame_index <= self.lag_end)
                
                if is_lagging and cam in last_frames:
                    # Đóng băng hình ảnh: Không đọc frame mới, dùng frame cũ
                    frames[cam] = last_frames[cam]
                else:
                    ok, frame = capture.read()
                    if not ok:
                        return
                    frames[cam] = frame
                    last_frames[cam] = frame
                    
            yield frame_index, frames

            for _ in range(self.stride - 1):
                for cam, capture in self.captures.items():
                    # Vẫn cho cuộn băng ngầm ở dưới để khi hết lag hình ảnh sẽ nhảy cóc (fast-forward)
                    if not capture.grab():
                        return
            frame_index += self.stride

def run_lag_pipeline(cfg: PipelineConfig, max_frames: int | None, 
                     lag_cam: str, lag_start: int, lag_end: int):
    """Vòng lặp chính của Pipeline, y hệt demo.py nhưng dùng LagSimulatedVideoReader"""
    calibrations = load_calibrations(cfg.scene_dir, cfg.cameras)
    reader = LagSimulatedVideoReader(cfg.scene_dir, cfg.cameras, cfg.stride, lag_cam, lag_start, lag_end)
    detector = PersonDetector(cfg)
    
    from multicam_tracker.bytetrack import ByteTracker
    trackers = {cam: ByteTracker(cfg) for cam in cfg.cameras}
    fusion = GlobalTrackManager(cfg)

    try:
        for frame_index, frames in reader:
            if max_frames is not None and frame_index >= max_frames:
                break

            detections = detector.detect_batch([frames[cam] for cam in cfg.cameras])
            camera_tracks = {
                cam: trackers[cam].update(dets)
                for cam, dets in zip(cfg.cameras, detections)
            }

            observations = [
                CameraObservation(
                    camera=cam,
                    local_id=track.track_id,
                    world_xy=calibrations[cam].image_to_world(foot_point_xyxy(track.xyxy))[0],
                    bbox_xyxy=track.xyxy,
                    conf=track.conf,
                )
                for cam, tracks in camera_tracks.items()
                for track in tracks
            ]
            global_tracks = fusion.step(observations)

            yield {
                "frame_index": frame_index,
                "frames": frames,
                "camera_tracks": camera_tracks,
                "global_tracks": global_tracks,
                "local_to_gid": dict(fusion.affinity),
                "calibrations": calibrations,
            }
    finally:
        reader.close()

def main():
    parser = argparse.ArgumentParser(description="Demo multi-camera tracking với giả lập mạng lag.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--stride", type=int, default=3, help="Mức độ bỏ qua frame (default: 3)")
    parser.add_argument("--device", default="cpu", choices=["mps", "cpu", "cuda"])
    
    # Tham số điều khiển Lag
    parser.add_argument("--lag-camera", type=str, default="camera_0001", help="Camera muốn gây lag (vd: camera_0001)")
    parser.add_argument("--lag-start", type=int, default=150, help="Frame bắt đầu bị đứng hình")
    parser.add_argument("--lag-end", type=int, default=300, help="Frame mạng ổn định lại")
    
    args = parser.parse_args()

    cfg = PipelineConfig(data_root=args.data_root)
    cfg.cameras = sorted(p.name for p in cfg.scene_dir.glob("camera_*") if p.is_dir())
    cfg.stride = args.stride
    cfg.device = args.device

    logger = None
    fps_smoothed = None
    t_prev = time.perf_counter()
    
    print("="*60)
    print(f"BẬT GIẢ LẬP ĐỨNG HÌNH CHO: [{args.lag_camera}]")
    print(f"Từ frame: {args.lag_start}  -> Đến frame: {args.lag_end}")
    print("="*60)
    
    for tick in run_lag_pipeline(cfg, None, args.lag_camera, args.lag_start, args.lag_end):
        if logger is None:
            logger = RerunLogger(cfg, tick["calibrations"])

        now = time.perf_counter()
        instant = 1.0 / max(now - t_prev, 1e-6)
        fps_smoothed = instant if fps_smoothed is None else 0.9 * fps_smoothed + 0.1 * instant
        t_prev = now

        logger.log_tick(
            tick["frame_index"], tick["frames"], tick["camera_tracks"],
            tick["local_to_gid"], tick["global_tracks"], fps=fps_smoothed,
        )
        
        # In ra log có kèm cảnh báo trạng thái mạng
        status_msg = " 🔴 [NETWORK LAGGING]" if args.lag_start <= tick["frame_index"] <= args.lag_end else ""
        print(f"frame {tick['frame_index']:5d} | {len(tick['global_tracks'])} người{status_msg}")

if __name__ == "__main__":
    main()
