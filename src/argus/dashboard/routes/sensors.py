"""External sensor fusion API routes.

Generic endpoint that lets any external sensor push a short-lived
multiplier for a ``(camera_id, zone_id)`` pair. The grader consults the
fusion store during severity computation.

Endpoints:
- ``POST   /api/sensors/signal``  — upsert an entry
- ``GET    /api/sensors/signals`` — list currently active (non-expired) entries
- ``DELETE /api/sensors/signal``  — remove an entry
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from argus.dashboard.api_response import (
    api_not_found,
    api_success,
    api_unavailable,
    api_validation_error,
)

logger = structlog.get_logger()
router = APIRouter()


class SensorSignalRequest(BaseModel):
    """Payload for ``POST /api/sensors/signal``."""

    camera_id: str = Field(description="Camera ID or '*' for a global signal")
    zone_id: str = Field(description="Zone ID or '*' for a camera-wide signal")
    multiplier: float = Field(description="Severity bias (0.1-5.0, 1.0 = neutral)")
    valid_for_s: float | None = Field(
        default=None,
        description="Signal TTL in seconds. Falls back to sensor_fusion.default_valid_for_s",
    )


class SensorSignalDeleteRequest(BaseModel):
    """Payload for ``DELETE /api/sensors/signal``."""

    camera_id: str
    zone_id: str


def _get_fusion(request: Request):
    """Fetch the app-wide SensorFusion instance or ``None`` if unavailable."""
    return getattr(request.app.state, "sensor_fusion", None)


@router.post("/signal")
async def set_signal(request: Request, payload: SensorSignalRequest) -> JSONResponse:
    """Upsert an external sensor signal.

    Returns 503 if fusion was never configured on this app, 400 on
    validation errors (including out-of-range multiplier).
    """
    fusion = _get_fusion(request)
    if fusion is None:
        return api_unavailable("传感器融合未配置")

    try:
        fusion.set_signal(
            camera_id=payload.camera_id,
            zone_id=payload.zone_id,
            multiplier=payload.multiplier,
            valid_for_s=payload.valid_for_s,
        )
    except ValueError as exc:
        return api_validation_error(str(exc))

    logger.info(
        "sensors.signal_set",
        camera_id=payload.camera_id,
        zone_id=payload.zone_id,
        multiplier=payload.multiplier,
        valid_for_s=payload.valid_for_s,
    )
    return api_success({"applied": True})


@router.get("/signals")
async def list_signals(request: Request) -> JSONResponse:
    """Return all currently-active sensor signals."""
    fusion = _get_fusion(request)
    if fusion is None:
        return api_unavailable("传感器融合未配置")

    return api_success({
        "enabled": bool(getattr(fusion, "enabled", False)),
        "signals": fusion.active_signals(),
    })


@router.delete("/signal")
async def delete_signal(
    request: Request, payload: SensorSignalDeleteRequest
) -> JSONResponse:
    """Remove a sensor signal."""
    fusion = _get_fusion(request)
    if fusion is None:
        return api_unavailable("传感器融合未配置")

    removed = fusion.remove_signal(payload.camera_id, payload.zone_id)
    if not removed:
        return api_not_found(
            f"信号不存在: camera_id={payload.camera_id} zone_id={payload.zone_id}",
        )

    logger.info(
        "sensors.signal_removed",
        camera_id=payload.camera_id,
        zone_id=payload.zone_id,
    )
    return api_success({"removed": True})
