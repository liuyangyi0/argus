"""Physics trajectory and localization API routes."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from argus.dashboard.api_response import api_not_found, api_success

logger = structlog.get_logger()
router = APIRouter()


@router.get("/{alert_id}/trajectory")
async def get_alert_trajectory(request: Request, alert_id: str) -> JSONResponse:
    """Get trajectory data for a specific alert."""
    db = getattr(request.app.state, "database", None)
    if db is None:
        return api_not_found("Database not available")

    with db.get_session() as session:
        from argus.storage.models import AlertRecord
        from sqlalchemy import select

        alert = session.scalar(
            select(AlertRecord).where(AlertRecord.alert_id == alert_id)
        )
        if alert is None:
            return api_not_found(f"Alert {alert_id} not found")

        return api_success({
            "alert_id": alert_id,
            "speed_ms": alert.speed_ms,
            "speed_px_per_sec": alert.speed_px_per_sec,
            "trajectory_model": alert.trajectory_model,
            "origin": {
                "x_mm": alert.origin_x_mm,
                "y_mm": alert.origin_y_mm,
                "z_mm": alert.origin_z_mm,
            } if alert.origin_x_mm is not None else None,
            "landing": {
                "x_mm": alert.landing_x_mm,
                "y_mm": alert.landing_y_mm,
                "z_mm": alert.landing_z_mm,
            } if alert.landing_x_mm is not None else None,
            "classification": {
                "label": alert.classification_label,
                "confidence": alert.classification_confidence,
            } if alert.classification_label else None,
        })


@router.get("/{camera_id}/active-tracks")
async def get_active_tracks(request: Request, camera_id: str) -> JSONResponse:
    """Get currently tracked objects with speed/trajectory data for a camera."""
    manager = getattr(request.app.state, "camera_manager", None)
    if manager is None:
        return api_not_found("Camera manager not available")

    runner = manager.get_runner(camera_id) if hasattr(manager, "get_runner") else None
    if runner is None:
        return api_not_found(f"Camera {camera_id} not found")

    # Get the pipeline's temporal tracker state
    pipeline = getattr(runner, "_pipeline", None)
    if pipeline is None:
        return api_success({"camera_id": camera_id, "tracks": []})

    tracker = getattr(pipeline, "_temporal_tracker", None)
    if tracker is None:
        return api_success({"camera_id": camera_id, "tracks": []})

    tracks = []
    for tid, state in tracker._tracks.items():
        tracks.append({
            "track_id": state.track_id,
            "centroid_x": round(state.centroid_x, 1),
            "centroid_y": round(state.centroid_y, 1),
            "consecutive_frames": state.consecutive_frames,
            "max_score": round(state.max_score, 3),
            "area_px": state.area_px,
            "trajectory_length": len(state.trajectory_history),
        })

    return api_success({"camera_id": camera_id, "tracks": tracks})


@router.get("/{alert_id}/localization")
async def get_alert_localization(request: Request, alert_id: str) -> JSONResponse:
    """Get origin/landing point estimates for an alert."""
    db = getattr(request.app.state, "database", None)
    if db is None:
        return api_not_found("Database not available")

    with db.get_session() as session:
        from argus.storage.models import AlertRecord
        from sqlalchemy import select

        alert = session.scalar(
            select(AlertRecord).where(AlertRecord.alert_id == alert_id)
        )
        if alert is None:
            return api_not_found(f"Alert {alert_id} not found")

        has_origin = alert.origin_x_mm is not None
        has_landing = alert.landing_x_mm is not None

        return api_success({
            "alert_id": alert_id,
            "has_localization": has_origin or has_landing,
            "trajectory_model": alert.trajectory_model,
            "origin": {
                "x_mm": round(alert.origin_x_mm, 1),
                "y_mm": round(alert.origin_y_mm, 1),
                "z_mm": round(alert.origin_z_mm, 1),
            } if has_origin else None,
            "landing": {
                "x_mm": round(alert.landing_x_mm, 1),
                "y_mm": round(alert.landing_y_mm, 1),
                "z_mm": round(alert.landing_z_mm, 1),
            } if has_landing else None,
            "speed_ms": alert.speed_ms,
        })
