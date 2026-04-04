"""Single camera capture with reconnection resilience."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from threading import Event

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class FrameData:
    """A captured frame with metadata."""

    frame: np.ndarray
    camera_id: str
    timestamp: float  # monotonic time
    frame_number: int
    resolution: tuple[int, int]  # (width, height)


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

    @property
    def state(self) -> CameraState:
        return self._state

    def connect(self) -> bool:
        """Establish connection to the camera source."""
        try:
            if self.protocol == "usb":
                self._cap = cv2.VideoCapture(int(self.source))
            elif self.protocol == "file":
                self._cap = cv2.VideoCapture(self.source)
            else:  # rtsp
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

            if self._cap is None or not self._cap.isOpened():
                self._state.connected = False
                self._state.error = f"Failed to open source: {self.source}"
                logger.error("camera.connect_failed", camera_id=self.camera_id, source=self.source)
                return False

            # Set timeouts to prevent frozen streams from blocking threads
            self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

            # Set resolution for live sources
            if self.protocol != "file":
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            self._state.connected = True
            self._state.error = None
            logger.info("camera.connected", camera_id=self.camera_id, source=self.source)
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

        ret, frame = self._cap.read()
        if not ret or frame is None:
            # For video files: loop back to beginning instead of disconnecting
            if self.protocol == "file" and self._cap is not None:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if ret and frame is not None:
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
        now = time.monotonic()

        # FPS decimation: skip frames to match target FPS
        if self._frame_interval > 0:
            elapsed = now - self._state.last_frame_time
            if elapsed < self._frame_interval:
                return None

        self._state.last_frame_time = now
        self._state.total_frames += 1

        h, w = frame.shape[:2]
        return FrameData(
            frame=frame,
            camera_id=self.camera_id,
            timestamp=now,
            frame_number=self._state.total_frames,
            resolution=(w, h),
        )

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
