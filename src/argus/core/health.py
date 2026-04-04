"""System health monitoring.

Tracks the health of all cameras, pipelines, and system resources.
Provides a unified health status for the dashboard and alerting.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from enum import Enum

import structlog

logger = structlog.get_logger()


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # some cameras offline
    UNHEALTHY = "unhealthy"  # all cameras offline or critical error


@dataclass
class CameraHealth:
    """Health info for a single camera."""

    camera_id: str
    connected: bool = False
    frames_captured: int = 0
    avg_latency_ms: float = 0.0
    reconnect_count: int = 0
    last_frame_time: float = 0.0
    error: str | None = None


@dataclass
class SystemHealth:
    """Overall system health snapshot."""

    status: HealthStatus
    uptime_seconds: float
    cameras: list[CameraHealth]
    total_alerts: int = 0
    platform: str = ""
    python_version: str = ""


class HealthMonitor:
    """Monitors system and camera health.

    Aggregates health data from all camera pipelines and provides
    a unified view for the dashboard.
    """

    def __init__(self):
        self._start_time = time.monotonic()
        self._camera_health: dict[str, CameraHealth] = {}
        self._total_alerts = 0

    def update_camera(
        self,
        camera_id: str,
        connected: bool,
        frames_captured: int = 0,
        avg_latency_ms: float = 0.0,
        reconnect_count: int = 0,
        error: str | None = None,
    ) -> None:
        """Update health data for a camera."""
        self._camera_health[camera_id] = CameraHealth(
            camera_id=camera_id,
            connected=connected,
            frames_captured=frames_captured,
            avg_latency_ms=avg_latency_ms,
            reconnect_count=reconnect_count,
            last_frame_time=time.monotonic(),
            error=error,
        )

    def record_alert(self) -> None:
        """Increment the total alert counter."""
        self._total_alerts += 1

    def get_health(self) -> SystemHealth:
        """Get the current system health snapshot."""
        cameras = list(self._camera_health.values())
        connected_count = sum(1 for c in cameras if c.connected)

        if not cameras:
            status = HealthStatus.UNHEALTHY
        elif connected_count == len(cameras):
            status = HealthStatus.HEALTHY
        elif connected_count > 0:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY

        return SystemHealth(
            status=status,
            uptime_seconds=time.monotonic() - self._start_time,
            cameras=cameras,
            total_alerts=self._total_alerts,
            platform=platform.platform(),
            python_version=platform.python_version(),
        )

    def check_stale_cameras(self, max_stale_seconds: float = 30.0) -> list[str]:
        """Find cameras that haven't produced frames recently."""
        now = time.monotonic()
        stale = []
        for camera_id, health in self._camera_health.items():
            if health.connected and health.last_frame_time > 0:
                elapsed = now - health.last_frame_time
                if elapsed > max_stale_seconds:
                    stale.append(camera_id)
        return stale
