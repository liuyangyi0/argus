"""Shared capture interface for detection pipeline inputs."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from argus.capture.camera import CameraState, FrameData


@runtime_checkable
class CaptureAdapter(Protocol):
    """Minimal camera capture interface used by ``DetectionPipeline``."""

    camera_id: str
    source: str
    protocol: str

    @property
    def state(self) -> CameraState:
        """Current connection state for health and reconnection checks."""
        ...

    def connect(self) -> bool:
        """Open the capture source."""
        ...

    def read(self) -> FrameData | None:
        """Read one frame directly from the capture source."""
        ...

    def read_latest(self) -> FrameData | None:
        """Read the freshest frame, using an internal buffer when available."""
        ...

    def start_capture_thread(self) -> None:
        """Start background latest-frame capture when supported."""
        ...

    def request_reconnect(self) -> None:
        """Schedule a reconnect attempt without blocking the caller."""
        ...

    def stop(self) -> None:
        """Stop capture and release resources."""
        ...
