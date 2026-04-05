"""Model management API routes (C4)."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from argus.storage.model_registry import ModelRegistry
from argus.storage.models import ModelRecord

logger = structlog.get_logger()

router = APIRouter()


def _get_registry(request: Request) -> ModelRegistry | None:
    """Get ModelRegistry from app state, or None if unavailable."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return None
    session_factory = getattr(db, "get_session", None)
    if session_factory is None:
        return None
    return ModelRegistry(session_factory=session_factory)


@router.get("/json")
async def list_models(request: Request, camera_id: str | None = None):
    """List all registered models, optionally filtered by camera_id."""
    registry = _get_registry(request)
    if registry is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    models = registry.list_models(camera_id=camera_id)
    return JSONResponse({
        "models": [m.to_dict() for m in models],
        "total": len(models),
    })


@router.post("/{version_id}/activate")
async def activate_model(request: Request, version_id: str):
    """Activate a specific model version."""
    registry = _get_registry(request)
    if registry is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    try:
        registry.activate(version_id)
        return JSONResponse({"status": "ok", "activated": version_id})
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=404)


@router.post("/{version_id}/rollback")
async def rollback_model(request: Request, version_id: str):
    """Rollback to the previous model for the same camera as version_id.

    The version_id is used to identify the camera; the actual rollback
    activates the previous model for that camera.
    """
    registry = _get_registry(request)
    if registry is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    # Look up the camera_id from the version_id
    models = registry.list_models()
    camera_id = None
    for m in models:
        if m.model_version_id == version_id:
            camera_id = m.camera_id
            break

    if camera_id is None:
        return JSONResponse({"error": f"Model version not found: {version_id}"}, status_code=404)

    previous = registry.rollback(camera_id)
    if previous is None:
        return JSONResponse(
            {"error": "No previous model to rollback to"},
            status_code=404,
        )

    return JSONResponse({
        "status": "ok",
        "activated": previous.model_version_id,
        "camera_id": camera_id,
    })
