"""Model management API routes (C4)."""

from __future__ import annotations

from pathlib import Path

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from argus.storage.model_registry import ModelRegistry
from argus.storage.models import ModelRecord

logger = structlog.get_logger()

router = APIRouter()

MAX_BATCH_SIZE = 100


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


@router.post("/batch-inference")
async def batch_inference(request: Request):
    """Score multiple images against a camera's active anomaly model.

    Request body: { camera_id: str, image_paths: list[str] }
    Returns: { results: [{path, score, is_anomalous, error?}], total, scored }
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    camera_id = body.get("camera_id")
    image_paths = body.get("image_paths", [])

    if not camera_id:
        return JSONResponse({"error": "camera_id is required"}, status_code=400)
    if not image_paths:
        return JSONResponse({"error": "image_paths must be a non-empty list"}, status_code=400)
    if len(image_paths) > MAX_BATCH_SIZE:
        return JSONResponse(
            {"error": f"Maximum {MAX_BATCH_SIZE} images per batch"},
            status_code=400,
        )

    # Get the camera's pipeline to access the detector
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if camera_manager is None:
        return JSONResponse({"error": "Camera manager not available"}, status_code=503)

    pipeline = camera_manager._pipelines.get(camera_id)
    if pipeline is None:
        return JSONResponse(
            {"error": f"Camera '{camera_id}' not running or not found"},
            status_code=404,
        )

    detector = getattr(pipeline, "_detector", None)
    if detector is None:
        return JSONResponse(
            {"error": f"No anomaly detector available for camera '{camera_id}'"},
            status_code=503,
        )

    import cv2

    results = []
    scored = 0
    threshold = getattr(detector, "threshold", 0.5)

    for img_path in image_paths:
        entry = {"path": img_path}
        try:
            p = Path(img_path)
            if not p.exists():
                entry["error"] = "File not found"
                results.append(entry)
                continue

            frame = cv2.imread(str(p))
            if frame is None:
                entry["error"] = "Could not read image"
                results.append(entry)
                continue

            prediction = detector.predict(frame)
            score = float(prediction.get("score", 0.0)) if isinstance(prediction, dict) else float(prediction)
            entry["score"] = round(score, 4)
            entry["is_anomalous"] = score >= threshold
            scored += 1
        except Exception as e:
            entry["error"] = str(e)

        results.append(entry)

    logger.info(
        "batch_inference.complete",
        camera_id=camera_id,
        total=len(image_paths),
        scored=scored,
    )

    return JSONResponse({
        "results": results,
        "total": len(image_paths),
        "scored": scored,
    })
