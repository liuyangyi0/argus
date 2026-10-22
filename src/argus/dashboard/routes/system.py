"""System status, overview, and administration routes (Chinese UI).

Consolidates: overview, system info, config, backup, audit, reports, users
into a single tabbed interface.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict

from argus.core.metrics import METRICS
from argus.dashboard.api_response import ApiEnvelope, api_error, api_success, api_typed_success

router = APIRouter()


class LivenessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alive: bool
    status: Literal["alive"]
    uptime_seconds: float | None


class CameraReadinessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_id: str
    connected: bool
    frames_captured: int
    avg_latency_ms: float
    reconnect_count: int
    last_frame_time: float
    observed_since: float
    error: str | None
    capture_state: Literal["receiving", "waiting", "stale", "disconnected"]
    scene_guard_usable: bool | None
    scene_guard_state: str | None
    scene_guard_reason: str | None
    model_ready: bool | None
    model_error: str | None
    pipeline_mode: str | None
    detection_state: Literal["ready", "blocked", "failed", "waiting"]
    detection_reason: str | None


class DetectionReadinessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: bool
    status: Literal["ready", "not_ready"]
    reasons: list[str]
    cameras: list[CameraReadinessResponse]


@router.get("/mode_summary")
def mode_summary(request: Request):
    """痛点 4: aggregate pipeline_mode across cameras for the navbar badge.

    Returns ``{cameras: {id: mode}, global_state}`` where global_state is
    one of "normal" | "capturing" | "training" | "maintenance":
    - training when ANY camera is in TRAINING (most disruptive — GPU busy)
    - capturing when any camera is in COLLECTION (samples being recorded)
    - maintenance when any camera is in MAINTENANCE
    - normal otherwise (all ACTIVE / LEARNING)
    """
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if camera_manager is None:
        return api_success({"cameras": {}, "global_state": "normal"})
    modes = camera_manager.get_all_pipeline_modes()
    if "training" in modes.values():
        global_state = "training"
    elif "collection" in modes.values():
        global_state = "capturing"
    elif "maintenance" in modes.values():
        global_state = "maintenance"
    else:
        global_state = "normal"
    return api_success({"cameras": modes, "global_state": global_state})


@router.get("/health")
def health(request: Request):
    """JSON health endpoint for monitoring tools."""
    health_monitor = request.app.state.health_monitor
    if not health_monitor:
        return api_success({"status": "unknown"})

    h = health_monitor.get_health()
    return api_success({
        "status": h.status.value,
        "uptime_seconds": round(h.uptime_seconds, 1),
        "total_alerts": h.total_alerts,
        "cameras": [
            {
                "camera_id": c.camera_id,
                "connected": c.connected,
                "frames_captured": c.frames_captured,
                "avg_latency_ms": round(c.avg_latency_ms, 1),
                "capture_state": c.capture_state,
                "detection_state": c.detection_state,
                "detection_reason": c.detection_reason,
                "pipeline_mode": c.pipeline_mode,
            }
            for c in h.cameras
        ],
        "platform": h.platform,
        "python_version": h.python_version,
    })


@router.get("/health/live", response_model=ApiEnvelope[LivenessResponse])
def liveness(request: Request):
    """Process liveness; scene blocking must never trigger restart loops."""
    monitor = getattr(request.app.state, "health_monitor", None)
    uptime = monitor.get_health().uptime_seconds if monitor is not None else None
    return api_typed_success({
        "alive": True,
        "status": "alive",
        "uptime_seconds": round(uptime, 1) if uptime is not None else None,
    }, LivenessResponse, headers={"Cache-Control": "no-store"})


@router.get(
    "/health/ready",
    response_model=ApiEnvelope[DetectionReadinessResponse],
    responses={503: {"model": ApiEnvelope[DetectionReadinessResponse]}},
)
def detection_readiness(request: Request):
    """Detection readiness, including capture freshness and safety gates."""
    monitor = getattr(request.app.state, "health_monitor", None)
    payload = monitor.get_detection_readiness() if monitor is not None else {
        "ready": False,
        "status": "not_ready",
        "reasons": ["health_monitor_unavailable"],
        "cameras": [],
    }
    validated = DetectionReadinessResponse.model_validate(payload).model_dump(mode="json")
    if not validated["ready"]:
        return api_error(
            50300, "检测尚未就绪", 503, data=validated,
            headers={"Cache-Control": "no-store"},
        )
    return api_typed_success(
        validated, DetectionReadinessResponse, headers={"Cache-Control": "no-store"},
    )


def _system_snapshots(camera_manager) -> list[dict]:
    """Read detached per-camera status through CameraManager's public API."""
    getter = getattr(camera_manager, "get_system_status_snapshots", None)
    if not callable(getter):
        return []
    snapshots = getter()
    return (
        [item for item in snapshots if isinstance(item, dict)]
        if isinstance(snapshots, list)
        else []
    )


def _snapshot_section(camera_manager, section: str) -> list[dict]:
    result: list[dict] = []
    for snapshot in _system_snapshots(camera_manager):
        payload = snapshot.get(section)
        if not isinstance(payload, dict):
            continue
        result.append({"camera_id": snapshot.get("camera_id"), **payload})
    return result


@router.get("/drift")
async def drift_status(request: Request):
    """Per-camera drift detection status."""
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if not camera_manager:
        return api_success({"cameras": []})

    result = _snapshot_section(camera_manager, "drift")
    return api_success({"cameras": result})


@router.get("/camera-health")
async def camera_health(request: Request):
    """Per-camera health check details (5-check analysis)."""
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if not camera_manager:
        return api_success({"cameras": []})

    result = _snapshot_section(camera_manager, "camera_health")
    return api_success({"cameras": result})


@router.get("/ensemble")
async def ensemble_status(request: Request):
    """Per-camera ensemble detector status."""
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if not camera_manager:
        return api_success({"cameras": []})

    result = _snapshot_section(camera_manager, "ensemble")
    return api_success({"cameras": result})


@router.get("/metrics")
def prometheus_metrics():
    """Prometheus-compatible metrics endpoint for scraping."""
    return Response(
        content=METRICS.generate(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/anomaly-degradation")
def anomaly_degradation(request: Request):
    """Aggregate anomaly head degradation status across all pipelines.

    Reports whether the per-camera anomaly detector has fallen into the
    simplex-only fallback mode (see ``DetectionPipeline._enter_anomaly_degraded``).
    The frontend uses this to drive the global degradation banner — operators
    must know that alert severity has been auto-downgraded.

    Returns ``{anomaly: {degraded, reason, since, cameras: [...]}}``. The
    aggregate ``degraded`` is True iff *any* camera is degraded; the per-camera
    list lets the system overview page surface details.
    """
    camera_manager = getattr(request.app.state, "camera_manager", None)
    snapshots = _system_snapshots(camera_manager) if camera_manager is not None else []
    if not snapshots:
        return api_success({
            "anomaly": {
                "degraded": False,
                "reason": "pipeline_not_ready",
                "since": None,
                "cameras": [],
            },
        })

    cameras_status = []
    aggregate_degraded = False
    aggregate_reason: str | None = None
    aggregate_since: float | None = None
    for snapshot in snapshots:
        cam_id = snapshot.get("camera_id")
        degradation = snapshot.get("anomaly_degradation")
        if not isinstance(degradation, dict):
            degradation = {}
        is_degraded = degradation.get("degraded") is True
        reason = degradation.get("reason")
        since = degradation.get("since")
        cameras_status.append({
            "camera_id": cam_id,
            "degraded": bool(is_degraded),
            "reason": reason,
            "since": since,
        })
        if is_degraded and not aggregate_degraded:
            aggregate_degraded = True
            aggregate_reason = reason
            aggregate_since = since

    return api_success({
        "anomaly": {
            "degraded": aggregate_degraded,
            "reason": aggregate_reason,
            "since": aggregate_since,
            "cameras": cameras_status,
        },
    })
