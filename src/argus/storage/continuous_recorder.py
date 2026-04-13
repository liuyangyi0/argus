"""Continuous 24/7 video recording with 4-hour segment rotation.

Each camera gets a dedicated background thread that encodes frames
into H.264 MP4 segments.  Segments rotate at a configurable interval
(default 4 h).  Frame pushes are non-blocking — frames are dropped
silently when the queue is full, matching the best-effort policy for
continuous recording.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()

_SENTINEL = None  # pushed to queue to signal stop


@dataclass
class RecordingSegment:
    """Metadata for a single recording segment."""

    camera_id: str
    segment_path: Path
    start_time: datetime
    end_time: datetime | None = None
    frame_count: int = 0
    file_size_bytes: int = 0


class ContinuousRecorder:
    """Per-camera continuous recorder with automatic segment rotation.

    Parameters
    ----------
    camera_id:
        Unique camera identifier.
    output_dir:
        Root directory for segment files.
    segment_duration_hours:
        Maximum segment length before rotation.
    encoding_crf:
        H.264 CRF quality (higher = smaller files).
    encoding_preset:
        libx264 preset.
    encoding_fps:
        Target FPS for output (may differ from capture FPS).
    resolution:
        (width, height) of output video.
    """

    def __init__(
        self,
        camera_id: str,
        output_dir: Path,
        segment_duration_hours: float = 4.0,
        encoding_crf: int = 26,
        encoding_preset: str = "veryfast",
        encoding_fps: int = 10,
        resolution: tuple[int, int] = (1920, 1080),
    ) -> None:
        self._camera_id = camera_id
        self._output_dir = Path(output_dir)
        self._segment_duration_s = segment_duration_hours * 3600.0
        self._crf = encoding_crf
        self._preset = encoding_preset
        self._fps = encoding_fps
        self._width, self._height = resolution

        self._queue: queue.Queue[tuple[np.ndarray, float] | None] = queue.Queue(
            maxsize=60,
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._current_segment: RecordingSegment | None = None
        self._writer: cv2.VideoWriter | None = None
        self._segment_start_ts: float = 0.0
        self._recording = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._recording_loop,
            name=f"cont-rec-{self._camera_id}",
            daemon=True,
        )
        self._thread.start()
        self._recording = True
        logger.info(
            "continuous_recorder.started",
            camera_id=self._camera_id,
            segment_hours=self._segment_duration_s / 3600,
        )

    def stop(self) -> None:
        if not self._recording:
            return
        self._stop_event.set()
        # Push sentinel to unblock queue.get()
        try:
            self._queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._recording = False
        logger.info("continuous_recorder.stopped", camera_id=self._camera_id)

    def push_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """Non-blocking frame push.  Drops frame if queue is full."""
        if not self._recording:
            return
        try:
            self._queue.put_nowait((frame, timestamp))
        except queue.Full:
            pass  # best-effort — drop frame silently

    @property
    def current_segment(self) -> RecordingSegment | None:
        with self._lock:
            return self._current_segment

    @property
    def is_recording(self) -> bool:
        return self._recording

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recording_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if item is _SENTINEL:
                    break
                frame, timestamp = item
                self._ensure_segment(timestamp)
                try:
                    self._write_frame(frame)
                except Exception:
                    logger.warning(
                        "continuous_recorder.write_frame_failed",
                        camera_id=self._camera_id,
                        exc_info=True,
                    )
        except Exception:
            logger.exception(
                "continuous_recorder.loop_error",
                camera_id=self._camera_id,
            )
        finally:
            self._close_segment()

    def _ensure_segment(self, timestamp: float) -> None:
        elapsed = timestamp - self._segment_start_ts
        if self._writer is None or elapsed >= self._segment_duration_s:
            self._close_segment()
            self._open_segment(timestamp)

    def _open_segment(self, timestamp: float) -> None:
        now = datetime.now(tz=timezone.utc)
        date_dir = self._output_dir / now.strftime("%Y-%m-%d") / self._camera_id
        date_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self._camera_id}_{now.strftime('%H_%M')}.mp4"
        seg_path = date_dir / filename

        # Try H.264 (avc1), fall back to mp4v
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(
            str(seg_path), fourcc, self._fps, (self._width, self._height),
        )
        if not writer.isOpened():
            writer.release()  # Release failed writer before fallback
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(seg_path), fourcc, self._fps, (self._width, self._height),
            )

        self._writer = writer
        self._segment_start_ts = timestamp
        with self._lock:
            self._current_segment = RecordingSegment(
                camera_id=self._camera_id,
                segment_path=seg_path,
                start_time=now,
            )
        logger.info(
            "continuous_recorder.segment_opened",
            camera_id=self._camera_id,
            path=str(seg_path),
        )

    def _write_frame(self, frame: np.ndarray) -> None:
        if self._writer is None or not self._writer.isOpened():
            return
        h, w = frame.shape[:2]
        if (w, h) != (self._width, self._height):
            frame = cv2.resize(frame, (self._width, self._height))
        self._writer.write(frame)
        if self._current_segment is not None:
            self._current_segment.frame_count += 1

    def _close_segment(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        with self._lock:
            seg = self._current_segment
            if seg is not None:
                seg.end_time = datetime.now(tz=timezone.utc)
                if seg.segment_path.exists():
                    seg.file_size_bytes = seg.segment_path.stat().st_size
                logger.info(
                    "continuous_recorder.segment_closed",
                    camera_id=self._camera_id,
                    frames=seg.frame_count,
                    size_mb=round(seg.file_size_bytes / 1_048_576, 1),
                )
            self._current_segment = None


class ContinuousRecordingManager:
    """Manages ContinuousRecorder instances for all cameras."""

    def __init__(
        self,
        output_dir: Path,
        segment_duration_hours: float = 4.0,
        encoding_crf: int = 26,
        encoding_preset: str = "veryfast",
        encoding_fps: int = 10,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._segment_hours = segment_duration_hours
        self._crf = encoding_crf
        self._preset = encoding_preset
        self._fps = encoding_fps
        self._recorders: dict[str, ContinuousRecorder] = {}

    def start_camera(
        self, camera_id: str, resolution: tuple[int, int] = (1920, 1080),
    ) -> None:
        if camera_id in self._recorders:
            return
        rec = ContinuousRecorder(
            camera_id=camera_id,
            output_dir=self._output_dir,
            segment_duration_hours=self._segment_hours,
            encoding_crf=self._crf,
            encoding_preset=self._preset,
            encoding_fps=self._fps,
            resolution=resolution,
        )
        rec.start()
        self._recorders[camera_id] = rec

    def stop_camera(self, camera_id: str) -> None:
        rec = self._recorders.pop(camera_id, None)
        if rec is not None:
            rec.stop()

    def stop_all(self) -> None:
        for cam_id in list(self._recorders):
            self.stop_camera(cam_id)

    def push_frame(
        self, camera_id: str, frame: np.ndarray, timestamp: float,
    ) -> None:
        rec = self._recorders.get(camera_id)
        if rec is not None:
            rec.push_frame(frame, timestamp)

    def get_segments(
        self,
        camera_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[RecordingSegment]:
        """List recording segments for a camera from disk."""
        cam_dir = self._output_dir
        segments: list[RecordingSegment] = []
        if not cam_dir.exists():
            return segments

        for date_dir in sorted(cam_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            sub = date_dir / camera_id
            if not sub.is_dir():
                continue
            for mp4 in sorted(sub.glob("*.mp4")):
                mtime = datetime.fromtimestamp(mp4.stat().st_mtime, tz=timezone.utc)
                if start and mtime < start:
                    continue
                if end and mtime > end:
                    continue
                segments.append(RecordingSegment(
                    camera_id=camera_id,
                    segment_path=mp4,
                    start_time=mtime,
                    file_size_bytes=mp4.stat().st_size,
                ))
        return segments

    def get_storage_stats(self) -> dict:
        """Return storage usage statistics."""
        total_bytes = 0
        total_segments = 0
        per_camera: dict[str, int] = {}

        if self._output_dir.exists():
            for mp4 in self._output_dir.rglob("*.mp4"):
                size = mp4.stat().st_size
                total_bytes += size
                total_segments += 1
                parts = mp4.parts
                # Try to extract camera_id from path
                for i, p in enumerate(parts):
                    if p == self._output_dir.name and i + 2 < len(parts):
                        cam = parts[i + 2]
                        per_camera[cam] = per_camera.get(cam, 0) + size
                        break

        return {
            "total_bytes": total_bytes,
            "total_gb": round(total_bytes / (1024**3), 2),
            "total_segments": total_segments,
            "per_camera_bytes": per_camera,
            "recording_cameras": list(self._recorders.keys()),
        }
