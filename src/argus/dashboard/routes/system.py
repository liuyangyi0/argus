"""System status, overview, and administration routes (Chinese UI).

Consolidates: overview, system info, config, backup, audit, reports, users
into a single tabbed interface.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

from argus.core.metrics import METRICS
from argus.dashboard.api_response import api_success

router = APIRouter()


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
            }
            for c in h.cameras
        ],
        "platform": h.platform,
        "python_version": h.python_version,
    })


def _collect_from_pipelines(camera_manager, attr_name: str, extractor, default_fn):
    """Iterate pipelines and collect per-camera data with fallback defaults."""
    result = []
    for cam_id, pipeline in camera_manager._pipelines.items():
        component = getattr(pipeline, attr_name, None)
        extracted = extractor(component) if component else None
        if extracted is not None:
            extracted["camera_id"] = cam_id
            result.append(extracted)
        else:
            d = default_fn()
            d["camera_id"] = cam_id
            result.append(d)
    return result


@router.get("/drift")
async def drift_status(request: Request):
    """Per-camera drift detection status."""
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if not camera_manager:
        return api_success({"cameras": []})

    def _extract_drift(detector):
        if not hasattr(detector, "get_status"):
            return None
        st = detector.get_status()
        return {
            "is_drifted": st.is_drifted,
            "ks_statistic": round(st.ks_statistic, 4),
            "p_value": round(st.p_value, 6),
            "reference_mean": round(st.reference_mean, 4),
            "current_mean": round(st.current_mean, 4),
            "samples_collected": st.samples_collected,
            "last_check_time": st.last_check_time,
        }

    result = _collect_from_pipelines(
        camera_manager, "_drift_detector", _extract_drift,
        lambda: {"is_drifted": False, "ks_statistic": 0, "p_value": 1,
                 "reference_mean": 0, "current_mean": 0, "samples_collected": 0,
                 "last_check_time": 0},
    )
    return api_success({"cameras": result})


@router.get("/camera-health")
async def camera_health(request: Request):
    """Per-camera health check details (5-check analysis)."""
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if not camera_manager:
        return api_success({"cameras": []})

    def _extract_health(analyzer):
        hr = getattr(analyzer, "last_result", None)
        if not hr:
            return None
        return {
            "is_frozen": hr.is_frozen,
            "sharpness_score": round(hr.sharpness_score, 2),
            "displacement_px": round(hr.displacement_px, 1),
            "is_flash": hr.is_flash,
            "gain_drift_pct": round(hr.gain_drift_pct, 1),
            "suppress_detection": hr.suppress_detection,
            "warnings": hr.warnings,
        }

    result = _collect_from_pipelines(
        camera_manager, "_health_analyzer", _extract_health,
        lambda: {"is_frozen": False, "sharpness_score": 0, "displacement_px": 0,
                 "is_flash": False, "gain_drift_pct": 0, "suppress_detection": False,
                 "warnings": []},
    )
    return api_success({"cameras": result})


@router.get("/ensemble")
async def ensemble_status(request: Request):
    """Per-camera ensemble detector status."""
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if not camera_manager:
        return api_success({"cameras": []})

    def _extract_ensemble(detector):
        if not hasattr(detector, "get_status"):
            return None
        return detector.get_status()

    result = _collect_from_pipelines(
        camera_manager, "_ensemble_detector", _extract_ensemble,
        lambda: {"model_count": 0, "loaded": False, "method": "none",
                 "model_paths": []},
    )
    return api_success({"cameras": result})


@router.get("/metrics")
def prometheus_metrics():
    """Prometheus-compatible metrics endpoint for scraping."""
    return Response(
        content=METRICS.generate(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
