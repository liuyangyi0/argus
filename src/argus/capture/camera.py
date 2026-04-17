"""Single camera capture with reconnection resilience."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from threading import Event
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argus.capture.frame_buffer import LatestFrameBuffer

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()

_BACKEND_NAMES = {
    getattr(cv2, "CAP_DSHOW", -1): "dshow",
    getattr(cv2, "CAP_MSMF", -1): "msmf",
    getattr(cv2, "CAP_FFMPEG", -1): "ffmpeg",
}


@dataclass
class FrameData:
    """A captured frame with metadata."""

    frame: np.ndarray
    camera_id: str
    timestamp: float  # monotonic time
    frame_number: int
    resolution: tuple[int, int]  # (width, height)
    is_nir: bool = False  # True when frame was captured under NIR strobe


@dataclass
class CameraState:
    """Tracks camera connection state."""

    connected: bool = False
    last_frame_time: float = 0.0
    total_frames: int = 0
    reconnect_count: int = 0
    error: str | None = None


class CameraCapture:
    """Captures frames from a single camera source with auto-reconnection.

    Supports RTSP streams, USB cameras (by device index), and video files.
    Handles disconnections gracefully with exponential backoff reconnection.
    """

    def __init__(
        self,
        camera_id: str,
        source: str,
        protocol: str = "rtsp",
        fps_target: int = 5,
        resolution: tuple[int, int] = (1920, 1080),
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = -1,
    ):
        self.camera_id = camera_id
        self.source = source
        self.protocol = protocol
        self.fps_target = fps_target
        self.resolution = resolution
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        self._cap: cv2.VideoCapture | None = None
        self._state = CameraState()
        self._stop_event = Event()
        self._frame_interval = 1.0 / fps_target if fps_target > 0 else 0
        self._frame_count = 0

        # Non-blocking reconnection state
        self._reconnecting = False
        self._reconnect_lock = threading.Lock()

        # Async capture thread buffer (initialised by start_capture_thread)
        self._frame_buffer: LatestFrameBuffer | None = None

    @property
    def state(self) -> CameraState:
        return self._state

    def _usb_backend_candidates(self) -> list[tuple[int | None, str]]:
        """Return preferred OpenCV backends for USB cameras."""
        if sys.platform.startswith("win"):
            candidates = []
            if hasattr(cv2, "CAP_DSHOW"):
                candidates.append((cv2.CAP_DSHOW, "dshow"))
            if hasattr(cv2, "CAP_MSMF"):
                candidates.append((cv2.CAP_MSMF, "msmf"))
            candidates.append((None, "default"))
            return candidates
        return [(None, "default")]

    def _open_capture(self) -> tuple[cv2.VideoCapture | None, str | None]:
        """Open a VideoCapture with protocol-appropriate backend fallbacks."""
        if self.protocol == "usb":
            source_index = int(self.source)
            for backend, backend_name in self._usb_backend_candidates():
                cap = (
                    cv2.VideoCapture(source_index)
                    if backend is None
                    else cv2.VideoCapture(source_index, backend)
                )
                if cap is not None and cap.isOpened():
                    return cap, backend_name
                if cap is not None:
                    cap.release()
                logger.warning(
                    "camera.backend_open_failed",
                    camera_id=self.camera_id,
                    source=self.source,
                    backend=backend_name,
                )
            return None, None

        if self.protocol == "file":
            return cv2.VideoCapture(self.source), "default"

        return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG), "ffmpeg"

    def connect(self) -> bool:
        """Establish connection to the camera source."""
        try:
            self._cap, backend_name = self._open_capture()

            if self._cap is None or not self._cap.isOpened():
                self._state.connected = False
                self._state.error = f"Failed to open source: {self.source}"
                logger.error(
                    "camera.connect_failed",
                    camera_id=self.camera_id,
                    source=self.source,
                    protocol=self.protocol,
                )
                return False

            # Set timeouts to prevent frozen streams from blocking threads
            self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

            # Minimise internal OpenCV buffer to reduce stale-frame accumulation
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Set resolution for live sources
            if self.protocol != "file":
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            self._state.connected = True
            self._state.error = None
            logger.info(
                "camera.connected",
                camera_id=self.camera_id,
                source=self.source,
                protocol=self.protocol,
                backend=backend_name,
            )
            return True

        except Exception as e:
            self._state.connected = False
            self._state.error = str(e)
            logger.error("camera.connect_error", camera_id=self.camera_id, error=str(e))
            return False

    def read(self) -> FrameData | None:
        """Read a single frame from the camera.

        Returns None if the frame should be skipped (fps decimation)
        or if the camera is disconnected.
        """
        if self._cap is None or not self._state.connected:
            return None

        # FPS throttle: sleep until next frame interval to avoid busy-read loop.
        # This MUST happen before _cap.read() — decoding frames only to discard
        # them wastes CPU, especially for video files at 25-30fps with a low
        # fps_target (e.g. 5).
        if self._frame_interval > 0:
            now = time.monotonic()
            elapsed = now - self._state.last_frame_time
            remaining = self._frame_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

        ret, frame = self._cap.read()
        # CRIT-05: Copy immediately to decouple from VideoCapture's internal buffer
        if ret and frame is not None:
            frame = frame.copy()
        if not ret or frame is None:
            # For video files: loop back to beginning instead of disconnecting
            if self.protocol == "file" and self._cap is not None:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    frame = frame.copy()
                    logger.debug("camera.file_loop", camera_id=self.camera_id)
                    # Fall through to normal processing
                else:
                    self._state.connected = False
                    self._state.error = "Read failed"
                    return None
            else:
                self._state.connected = False
                self._state.error = "Read failed"
                return None

        self._frame_count += 1

        self._state.last_frame_time = time.monotonic()
        self._state.total_frames += 1

        h, w = frame.shape[:2]
        return FrameData(
            frame=frame,
            camera_id=self.camera_id,
            timestamp=now,
            frame_number=self._state.total_frames,
            resolution=(w, h),
        )

    # ------------------------------------------------------------------
    # Async capture thread (Frigate-inspired latest-frame pattern)
    # ------------------------------------------------------------------

    def start_capture_thread(self) -> None:
        """Start a dedicated capture thread that keeps the latest frame ready.

        The capture thread continuously reads from the video source and
        overwrites a :class:`LatestFrameBuffer`.  The detection pipeline
        calls :meth:`read_latest` to get the most recent frame without
        blocking on the network or accumulating stale frames.
        """
        from argus.capture.frame_buffer import LatestFrameBuffer

        if self._frame_buffer is not None:
            return  # already started
        self._frame_buffer = LatestFrameBuffer()
        t = threading.Thread(
            target=self._capture_loop,
            name=f"capture-{self.camera_id}",
            daemon=True,
        )
        t.start()
        logger.info("camera.capture_thread_started", camera_id=self.camera_id)

    def _capture_loop(self) -> None:
        """Background loop: read frames and store the latest one."""
        while not self._stop_event.is_set():
            frame_data = self.read()
            if frame_data is not None:
                self._frame_buffer.put(frame_data)
            else:
                # Camera disconnected or FPS throttle — brief yield
                self._stop_event.wait(0.01)

    def read_latest(self) -> FrameData | None:
        """Return the most recently captured frame.

        Requires :meth:`start_capture_thread` to have been called.
        Falls back to :meth:`read` if no capture thread is running.
        """
        if self._frame_buffer is None:
            return self.read()
        return self._frame_buffer.get(timeout=5.0)

    def request_reconnect(self) -> None:
        """Request a non-blocking reconnection attempt.

        Spawns a background thread to handle exponential backoff reconnection
        so the calling thread (e.g. the detection pipeline) is not blocked.
        """
        with self._reconnect_lock:
            if self._reconnecting:
                return  # already trying
            self._reconnecting = True

        thread = threading.Thread(
            target=self._reconnect_background,
            name=f"reconnect-{self.camera_id}",
            daemon=True,
        )
        thread.start()

    def _reconnect_background(self) -> None:
        """Background reconnection with exponential backoff."""
        try:
            self.release()
            delay = self.reconnect_delay
            attempt = 0

            while not self._stop_event.is_set():
                if self.max_reconnect_attempts >= 0 and attempt >= self.max_reconnect_attempts:
                    logger.error(
                        "camera.reconnect_exhausted",
                        camera_id=self.camera_id,
                        attempts=attempt,
                    )
                    break

                attempt += 1
                self._state.reconnect_count += 1
                logger.info(
                    "camera.reconnecting",
                    camera_id=self.camera_id,
                    attempt=attempt,
                    delay=delay,
                )

                self._stop_event.wait(delay)
                if self._stop_event.is_set():
                    break

                if self.connect():
                    logger.info(
                        "camera.reconnected",
                        camera_id=self.camera_id,
                        attempts=attempt,
                    )
                    break

                # Exponential backoff, max 60 seconds
                delay = min(delay * 2, 60.0)
        finally:
            with self._reconnect_lock:
                self._reconnecting = False

    def reconnect(self) -> bool:
        """Attempt to reconnect with exponential backoff (blocking).

        Note: For non-blocking reconnection from pipeline threads,
        use ``request_reconnect()`` instead.
        """
        self.release()

        attempt = 0
        delay = self.reconnect_delay

        while not self._stop_event.is_set():
            if self.max_reconnect_attempts >= 0 and attempt >= self.max_reconnect_attempts:
                logger.error(
                    "camera.reconnect_exhausted",
                    camera_id=self.camera_id,
                    attempts=attempt,
                )
                return False

            attempt += 1
            self._state.reconnect_count += 1
            logger.info(
                "camera.reconnecting",
                camera_id=self.camera_id,
                attempt=attempt,
                delay=delay,
            )

            self._stop_event.wait(delay)
            if self._stop_event.is_set():
                return False

            if self.connect():
                return True

            # Exponential backoff, max 60 seconds
            delay = min(delay * 2, 60.0)

        return False

    def release(self) -> None:
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._state.connected = False

    def stop(self) -> None:
        """Signal the camera to stop (used to break reconnection loops)."""
        self._stop_event.set()
        self.release()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.stop()
