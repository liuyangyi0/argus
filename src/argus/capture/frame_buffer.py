"""Thread-safe single-frame buffer for decoupling capture from detection.

Inspired by Frigate NVR's shared-memory frame slot pattern.  A dedicated
capture thread continuously reads from the video source and overwrites the
buffer with the latest frame.  The detection thread reads from the buffer
at its own pace — always getting the most recent frame, never a stale one.

This eliminates the "slow-motion" effect caused by detection blocking capture:
the TCP/RTSP buffer is drained continuously by the capture thread regardless
of how long detection takes.
"""

from __future__ import annotations

import threading
import time

import structlog

from argus.capture.camera import FrameData

logger = structlog.get_logger()


class LatestFrameBuffer:
    """Thread-safe single-frame buffer.

    - ``put()`` is called by the capture thread.  It overwrites the previous
      frame and signals the detection thread.  Never blocks.
    - ``get()`` is called by the detection thread.  It blocks until a new
      frame is available (or *timeout* expires) and returns it.

    Frames that the detection thread does not consume are not "dropped" in the
    traditional sense — they were fully read from the network (so the TCP
    buffer stays drained), but the detection pipeline only needs to analyse
    the most recent one because intermediate frames have no new information
    when the camera position has not changed.
    """

    def __init__(self) -> None:
        self._frame: FrameData | None = None
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._frames_replaced: int = 0

    def put(self, frame: FrameData) -> None:
        """Store *frame*, replacing any unconsumed previous frame."""
        with self._lock:
            if self._frame is not None:
                self._frames_replaced += 1
            self._frame = frame
        self._new_frame.set()

    def get(self, timeout: float = 5.0) -> FrameData | None:
        """Block until a frame is available, then return it.

        Returns ``None`` on timeout (camera may have disconnected).
        """
        if not self._new_frame.wait(timeout):
            return None
        with self._lock:
            frame = self._frame
            self._frame = None
        self._new_frame.clear()
        return frame

    @property
    def frames_replaced(self) -> int:
        """Number of frames overwritten before detection could consume them."""
        return self._frames_replaced
