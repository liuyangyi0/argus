"""System health monitoring.

Tracks the health of all cameras, pipelines, and system resources.
Provides a unified health status for the dashboard and alerting.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Callable

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
    observed_since: float = 0.0
    capture_state: str = "waiting"
    scene_guard_usable: bool | None = None
    scene_guard_state: str | None = None
    scene_guard_reason: str | None = None
    model_ready: bool | None = None
    model_error: str | None = None
    pipeline_mode: str | None = "active"
    detection_state: str = "waiting"
    detection_reason: str | None = "runtime_status_unavailable"


def runtime_detection_health(
    scene_guard: dict | None, model_statuses: list[dict],
    *, pipeline_mode: str | None = None,
) -> dict:
    """Summarize detached runtime diagnostics without masking unknown state."""
    guard = scene_guard if isinstance(scene_guard, dict) else {}
    models = [
        model for model in model_statuses
        if model.get("backend") != "disabled"
    ]
    failures = [
        model for model in models
        if model.get("loaded") is not True
        or model.get("consecutive_failures", 0) > 0
        or model.get("backend") in {"none", "ssim-fallback"}
    ]
    model_ready = not failures if models else None
    model_error = None
    if failures:
        failed = failures[0]
        model_error = str(failed.get("last_error") or f"{failed.get('name', 'model')}_not_ready")
    return {
        "scene_guard_usable": guard.get("usable") if isinstance(guard.get("usable"), bool) else None,
        "scene_guard_state": guard.get("state"),
        "scene_guard_reason": guard.get("reason"),
        "model_ready": model_ready,
        "model_error": model_error,
        "pipeline_mode": pipeline_mode,
    }


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

    def __init__(
        self,
        on_change: Callable[[str, dict], None] | None = None,
        feedback_manager: object | None = None,
        degradation_manager: object | None = None,
    ):
        self._start_time = time.monotonic()
        self._camera_health: dict[str, CameraHealth] = {}
        self._total_alerts = 0
        self._on_change = on_change
        self._feedback_manager = feedback_manager
        self._degradation_manager = degradation_manager
        self._last_status: str = ""
        self._last_connected_count: int = -1
        self._last_readiness_signature: tuple = ()
        self._degradation_feedback_sent: set[str] = set()  # cameras already reported

    def update_camera(
        self,
        camera_id: str,
        connected: bool,
        frames_captured: int = 0,
        avg_latency_ms: float = 0.0,
        reconnect_count: int = 0,
        error: str | None = None,
        *,
        scene_guard_usable: bool | None = None,
        scene_guard_state: str | None = None,
        scene_guard_reason: str | None = None,
        model_ready: bool | None = None,
        model_error: str | None = None,
        pipeline_mode: str | None = "active",
    ) -> None:
        """Update health data for a camera."""
        prev = self._camera_health.get(camera_id)
        was_connected = prev.connected if prev else True
        now = time.monotonic()
        # Polling a connected flag is not evidence of a newly captured frame.
        # A reset counter must advance again before it renews freshness.
        advanced = frames_captured > (prev.frames_captured if prev else 0)
        last_frame_time = now if advanced else (prev.last_frame_time if prev else 0.0)
        observed_since = prev.observed_since if prev and was_connected == connected else now

        self._camera_health[camera_id] = CameraHealth(
            camera_id=camera_id,
            connected=connected,
            frames_captured=frames_captured,
            avg_latency_ms=avg_latency_ms,
            reconnect_count=reconnect_count,
            last_frame_time=last_frame_time,
            error=error,
            observed_since=observed_since,
            scene_guard_usable=scene_guard_usable,
            scene_guard_state=scene_guard_state,
            scene_guard_reason=scene_guard_reason,
            model_ready=model_ready,
            model_error=model_error,
            pipeline_mode=pipeline_mode,
        )
        self._notify_change()

        # Passive feedback: camera disconnection or error
        if self._feedback_manager and not connected and was_connected:
            if camera_id not in self._degradation_feedback_sent:
                self._degradation_feedback_sent.add(camera_id)
                try:
                    self._feedback_manager.submit_passive_feedback(
                        camera_id=camera_id,
                        zone_id="all",
                        source="health",
                        notes=f"Camera disconnected. error={error or 'none'}",
                    )
                except Exception:
                    logger.debug("health.feedback_failed", camera_id=camera_id)
        elif connected and camera_id in self._degradation_feedback_sent:
            # Camera recovered — allow future feedback
            self._degradation_feedback_sent.discard(camera_id)

        # UX v2 §5: Report degradation events to global bar
        if self._degradation_manager is not None:
            if not connected and was_connected:
                try:
                    self._degradation_manager.report(
                        category="rtsp_broken",
                        camera_id=camera_id,
                        camera=camera_id,
                        n=reconnect_count,
                        max=10,
                        t=0,
                    )
                except Exception:
                    logger.debug("health.degradation_report_failed", camera_id=camera_id)
            elif connected and not was_connected:
                try:
                    self._degradation_manager.resolve_by_category(
                        category="rtsp_broken", camera_id=camera_id
                    )
                except Exception:
                    logger.debug("health.degradation_resolve_failed", camera_id=camera_id)

    def retain_cameras(self, camera_ids: Iterable[str]) -> None:
        """Forget removed configurations without hiding configured offline cameras.

        The main loop supplies an authoritative configuration snapshot, not
        only running pipelines. Otherwise a deleted camera remains stale in
        readiness forever, or an offline configured camera disappears.
        """
        retained = set(camera_ids)
        removed = set(self._camera_health) - retained
        for camera_id in removed:
            self._camera_health.pop(camera_id, None)
            self._degradation_feedback_sent.discard(camera_id)
        if removed:
            self._notify_change()

    def record_alert(self) -> None:
        """Increment the total alert counter."""
        self._total_alerts += 1
        self._notify_change()

    def _notify_change(self) -> None:
        """Notify WebSocket subscribers of health state change.

        Only broadcasts when status or connected camera count actually changes.
        """
        if not self._on_change:
            return
        try:
            h = self.get_health()
            connected = sum(1 for c in h.cameras if c.connected)
            readiness_signature = tuple(
                (c.camera_id, c.capture_state, c.pipeline_mode, c.detection_state, c.detection_reason)
                for c in h.cameras
            )
            if (
                h.status.value == self._last_status
                and connected == self._last_connected_count
                and readiness_signature == self._last_readiness_signature
            ):
                return  # no meaningful change
            self._last_status = h.status.value
            self._last_connected_count = connected
            self._last_readiness_signature = readiness_signature
            self._on_change("health", {
                "status": h.status.value,
                "uptime_seconds": round(h.uptime_seconds, 1),
                "total_alerts": h.total_alerts,
                "cameras": [
                    {
                        "camera_id": c.camera_id,
                        "connected": c.connected,
                        "frames_captured": c.frames_captured,
                        "avg_latency_ms": round(c.avg_latency_ms, 1),
                        "reconnect_count": c.reconnect_count,
                        "capture_state": c.capture_state,
                        "detection_state": c.detection_state,
                        "detection_reason": c.detection_reason,
                        "pipeline_mode": c.pipeline_mode,
                    }
                    for c in h.cameras
                ],
            })
        except Exception as e:
            logger.debug("health.notify_failed", error=str(e))

    def get_health(self) -> SystemHealth:
        """Get the current system health snapshot."""
        cameras = [self._camera_snapshot(c) for c in list(self._camera_health.values())]
        connected_count = sum(
            1 for c in cameras if c.connected and c.capture_state != "stale"
        )

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

    def get_camera_health(self, camera_id: str) -> dict | None:
        """Get a single camera health record as a dashboard-friendly dict."""
        health = self._camera_health.get(camera_id)
        if health is None:
            return None

        return asdict(self._camera_snapshot(health))

    @staticmethod
    def _camera_snapshot(health: CameraHealth, max_stale_seconds: float = 30.0) -> CameraHealth:
        now = time.monotonic()
        if not health.connected:
            capture_state = "disconnected"
        elif now - max(health.last_frame_time, health.observed_since) > max_stale_seconds:
            capture_state = "stale"
        elif health.last_frame_time < health.observed_since or health.frames_captured == 0:
            capture_state = "waiting"
        else:
            capture_state = "receiving"

        if capture_state != "receiving":
            detection_state = "failed" if capture_state in {"stale", "disconnected"} else "waiting"
            reason = f"capture_{capture_state}"
        elif health.scene_guard_usable is False:
            detection_state, reason = "blocked", health.scene_guard_reason or "scene_guard_blocked"
        elif health.pipeline_mode in {"collection", "training", "learning"}:
            detection_state = "blocked"
            reason = (
                "learning_alerts_suppressed"
                if health.pipeline_mode == "learning"
                else f"pipeline_mode_{health.pipeline_mode}"
            )
        elif health.pipeline_mode not in {"active", "maintenance"}:
            detection_state, reason = "waiting", "runtime_mode_unavailable"
        elif health.model_ready is False:
            detection_state, reason = "failed", health.model_error or "model_not_ready"
        elif health.scene_guard_usable is None or health.model_ready is None:
            detection_state, reason = "waiting", "runtime_status_unavailable"
        else:
            detection_state, reason = "ready", None
        return replace(
            health,
            capture_state=capture_state,
            detection_state=detection_state,
            detection_reason=reason,
        )

    def get_detection_readiness(self) -> dict:
        """Detection readiness is independent from process liveness."""
        cameras = [asdict(self._camera_snapshot(c)) for c in list(self._camera_health.values())]
        ready = bool(cameras) and all(c["detection_state"] == "ready" for c in cameras)
        return {
            "ready": ready,
            "status": "ready" if ready else "not_ready",
            "reasons": [
                f"{c['camera_id']}:{c['detection_reason']}"
                for c in cameras if c["detection_state"] != "ready"
            ] if cameras else ["no_cameras"],
            "cameras": cameras,
        }

    def check_stale_cameras(self, max_stale_seconds: float = 30.0) -> list[str]:
        """Find cameras that haven't produced frames recently."""
        return [
            camera_id for camera_id, health in list(self._camera_health.items())
            if self._camera_snapshot(health, max_stale_seconds).capture_state == "stale"
        ]
