"""Physics trajectory and localization API routes."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from argus.dashboard.api_contracts import AlertTrajectoryResponse
from argus.dashboard.api_response import (
    ApiEnvelope,
    api_forbidden,
    api_not_found,
    api_success,
    api_typed_success,
)
from argus.dashboard.auth import require_permission

logger = structlog.get_logger()
router = APIRouter()


@router.get(
    "/{alert_id}/trajectory",
    response_model=ApiEnvelope[AlertTrajectoryResponse],
)
async def get_alert_trajectory(request: Request, alert_id: str) -> JSONResponse:
    """Get trajectory data for a specific alert.

    Returns both the primary-track scalars (mm-space, matches ORM fields) and
    the full list of per-track fits (with pixel-space origin/landing so the
    frontend can overlay them on video). ``trajectories`` is always a list and
    is empty when no fit was persisted.
    """
    if not require_permission(request, "view_replay"):
        return api_forbidden("权限不足")

    db = getattr(request.app.state, "database", None)
    if db is None:
        return api_not_found("Database not available")

    with db.get_session() as session:
        from argus.storage.models import AlertRecord
        from sqlalchemy import select
        import json as _json

        alert = session.scalar(
            select(AlertRecord).where(AlertRecord.alert_id == alert_id)
        )
        if alert is None:
            return api_not_found(f"Alert {alert_id} not found")

        trajectories: list = []
        if alert.trajectories_json:
            try:
                parsed = _json.loads(alert.trajectories_json)
                if isinstance(parsed, list):
                    trajectories = parsed
            except Exception:
                trajectories = []

        points = None
        if alert.trajectory_points:
            try:
                points = _json.loads(alert.trajectory_points)
            except Exception:
                points = None

        primary = None
        if alert.trajectory_model is not None:
            primary = {
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
            }

        return api_typed_success(
            {
                "alert_id": alert_id,
                # Legacy flat fields — kept so older clients do not break
                "speed_ms": alert.speed_ms,
                "speed_px_per_sec": alert.speed_px_per_sec,
                "trajectory_model": alert.trajectory_model,
                "points": points,
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
                # New multi-track view
                "primary": primary,
                "trajectories": trajectories,
                "classification": {
                    "label": alert.classification_label,
                    "confidence": alert.classification_confidence,
                } if alert.classification_label else None,
            },
            AlertTrajectoryResponse,
        )


@router.get("/{camera_id}/active-tracks")
async def get_active_tracks(request: Request, camera_id: str) -> JSONResponse:
    """Get currently tracked objects with speed/trajectory data for a camera."""
    manager = getattr(request.app.state, "camera_manager", None)
    if manager is None:
        return api_not_found("Camera manager not available")

    get_tracks = getattr(manager, "get_active_tracks_snapshot", None)
    if not callable(get_tracks):
        logger.error("physics.camera_runtime_contract_missing", camera_id=camera_id)
        return api_not_found(f"Camera {camera_id} not found")

    tracks = get_tracks(camera_id)
    if tracks is None:
        return api_not_found(f"Camera {camera_id} not found")

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
