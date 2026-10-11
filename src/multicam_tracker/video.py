"""Timestamp-aware local-file and live multi-camera readers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic, sleep

import cv2
import numpy as np

from .timebase import ClockModel, FrameBatch, FramePacket, TimestampQuality


def open_video(source: str, decoder_threads: int = 1, live: bool = False):
    """FFmpeg decoders have a thread budget separate from OpenCV operations."""
    params = [cv2.CAP_PROP_N_THREADS, decoder_threads]
    if live:
        params += [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                   cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]
    capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
    if not capture.isOpened() and not live:
        capture.release()
        capture = cv2.VideoCapture(source)
    return capture


class SyncedVideoReader:
    """Iterate over synchronized frames from N camera videos.

    Yields ``(frame_index, {camera: BGR image})`` tuples, advancing all
    videos in lockstep and skipping ``stride - 1`` frames between ticks.
    """

    def __init__(
        self,
        scene_dir: Path,
        cameras: list[str],
        stride: int = 1,
        frame_offsets: dict[str, int] | None = None,
        clock_models: dict[str, ClockModel] | None = None,
        decoder_threads: int = 1,
    ):
        if not cameras or stride < 1:
            raise ValueError("At least one camera and stride >= 1 are required")
        self.cameras = cameras
        self.stride = stride
        self.frame_offsets = frame_offsets or {}
        self.clock_models = clock_models or {}
        self.captures: dict[str, cv2.VideoCapture] = {}
        self._source_indices: dict[str, int] = {}
        self._sequences: dict[str, int] = {camera: 0 for camera in cameras}
        self._fps: dict[str, float] = {}
        for cam in cameras:
            path = scene_dir / cam / "video.mp4"
            capture = open_video(str(path), decoder_threads)
            if not capture.isOpened():
                capture.release()
                self.close()
                raise FileNotFoundError(f"Cannot open {path} — run `download-data` first")
            self.captures[cam] = capture
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            self._fps[cam] = fps if np.isfinite(fps) and fps > 0 else 30.0
            offset = max(0, int(self.frame_offsets.get(cam, 0)))
            for _ in range(offset):
                if not capture.grab():
                    break
            self._source_indices[cam] = offset

        first = self.captures[cameras[0]]
        self.fps = self._fps[cameras[0]]
        self.frame_count = int(first.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        """Legacy lockstep API retained for existing callers."""
        for batch in self.iter_batches():
            first = next(iter(batch.packets.values()))
            yield first.sequence * self.stride, {
                camera: packet.image for camera, packet in batch.packets.items()
            }

    def iter_batches(self) -> Iterator[FrameBatch]:
        """Yield lockstep batches carrying per-camera event timestamps."""
        active = set(self.cameras)
        while active:
            packets: dict[str, FramePacket] = {}
            for cam, capture in self.captures.items():
                if cam not in active:
                    continue
                ok, frame = capture.read()
                if not ok:
                    active.remove(cam)
                    continue
                source_index = self._source_indices[cam]
                # A configured frame offset trims/alines the source; it is not
                # a residual camera-clock offset. Keep the original index for
                # audit, but restart capture event time at the trimmed epoch.
                raw_ts_s = self._sequences[cam] * self.stride / self._fps[cam]
                clock = self.clock_models.get(cam, ClockModel())
                packets[cam] = FramePacket(
                    camera=cam,
                    image=frame,
                    capture_ts_s=clock.normalize(raw_ts_s),
                    arrival_ts_s=monotonic(),
                    sequence=self._sequences[cam],
                    source_frame_index=source_index,
                    timestamp_quality=TimestampQuality.FRAME_INDEX,
                )
                self._sequences[cam] += 1
                self._source_indices[cam] += self.stride
            if packets:
                yield FrameBatch(packets=packets, emitted_at_s=monotonic())

            # Cheap skip: grab() decodes but skips retrieval/conversion.
            for _ in range(self.stride - 1):
                for cam in list(active):
                    capture = self.captures[cam]
                    if not capture.grab():
                        active.remove(cam)

    def close(self) -> None:
        for capture in self.captures.values():
            capture.release()


class AsyncVideoReader:
    """Read each live/file source independently into a bounded FIFO queue.

    A slow or disconnected camera cannot block healthy streams. If a consumer
    falls behind, only that camera's producer waits for queue capacity. No
    decoded frames are discarded to make an overloaded CPU appear real time.
    Live transports may still drop upstream; use a recorder for lossless replay.
    """

    def __init__(
        self,
        sources: dict[str, str | Path],
        stride: int = 1,
        queue_size: int = 2,
        microbatch_ms: float = 20.0,
        clock_models: dict[str, ClockModel] | None = None,
        frame_offsets: dict[str, int] | None = None,
        decoder_threads: int = 1,
    ):
        if not sources or stride < 1 or queue_size < 1:
            raise ValueError("Sources, stride >= 1 and queue_size >= 1 are required")
        self.cameras = list(sources)
        self.stride = stride
        self.microbatch_s = max(0.001, microbatch_ms / 1000.0)
        self.clock_models = clock_models or {}
        self.frame_offsets = frame_offsets or {}
        self.decoder_threads = decoder_threads
        self.queues = {cam: Queue(maxsize=max(1, queue_size)) for cam in sources}
        self.dropped_frames = {cam: 0 for cam in sources}
        self.errors: dict[str, str] = {}
        self._done = {cam: False for cam in sources}
        self._stop = Event()
        self._threads: list[Thread] = []
        for camera, source in sources.items():
            thread = Thread(
                target=self._read_camera,
                args=(camera, str(source)),
                name=f"capture-{camera}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def _read_camera(self, camera: str, source: str) -> None:
        capture = None
        is_live = source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))
        try:
            capture = open_video(source, self.decoder_threads, live=is_live)
            if not capture.isOpened():
                raise OSError(f"Cannot open camera {camera}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            fps = fps if np.isfinite(fps) and fps > 0 else 30.0
            source_index = max(0, int(self.frame_offsets.get(camera, 0)))
            trimmed_start_index = source_index
            for _ in range(source_index):
                if self._stop.is_set() or not capture.grab():
                    return
            sequence = 0
            clock = self.clock_models.get(camera, ClockModel())
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    if is_live:
                        self.errors[camera] = "Stream disconnected or read timeout"
                    break
                arrival = monotonic()
                pts_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                if not is_live:
                    raw_ts = (source_index - trimmed_start_index) / fps
                    quality = TimestampQuality.FRAME_INDEX
                elif np.isfinite(pts_ms) and pts_ms > 0:
                    raw_ts = pts_ms / 1000.0
                    quality = TimestampQuality.SOURCE_PTS
                else:
                    raw_ts = arrival
                    quality = TimestampQuality.ARRIVAL_FALLBACK
                packet = FramePacket(
                    camera=camera,
                    image=frame,
                    capture_ts_s=clock.normalize(raw_ts),
                    arrival_ts_s=arrival,
                    sequence=sequence,
                    source_frame_index=source_index,
                    timestamp_quality=quality,
                )
                queue = self.queues[camera]
                while not self._stop.is_set():
                    try:
                        queue.put(packet, timeout=0.05)
                        break
                    except Full:
                        continue
                sequence += 1
                source_index += self.stride
                for _ in range(self.stride - 1):
                    if not capture.grab():
                        return
        except Exception as exc:
            self.errors[camera] = f"{type(exc).__name__}: {exc}"
        finally:
            if capture is not None:
                capture.release()
            self._done[camera] = True

    def iter_batches(self) -> Iterator[FrameBatch]:
        while not self._stop.is_set():
            packets: dict[str, FramePacket] = {}
            deadline = monotonic() + self.microbatch_s
            while monotonic() < deadline:
                for camera, queue in self.queues.items():
                    if camera in packets:
                        continue
                    try:
                        packets[camera] = queue.get_nowait()
                    except Empty:
                        pass
                active = {
                    camera for camera, queue in self.queues.items()
                    if not self._done[camera] or not queue.empty()
                }
                if active and active.issubset(packets):
                    break
                sleep(min(0.001 if not packets else 0.0005, max(0.0, deadline - monotonic())))
            if packets:
                yield FrameBatch(
                    packets=packets,
                    emitted_at_s=monotonic(),
                    dropped_frames=dict(self.dropped_frames),
                )
            elif all(self._done.values()) and all(q.empty() for q in self.queues.values()):
                if len(self.errors) == len(self.cameras):
                    raise RuntimeError(f"All camera readers failed: {self.errors}")
                return

    def close(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        for queue in self.queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                except Empty:
                    break
