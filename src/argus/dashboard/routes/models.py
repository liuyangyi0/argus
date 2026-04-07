"""Model management API routes (C4)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from argus.dashboard.model_runtime import (
    activate_model_version,
    get_registry,
    rollback_camera_model,
    sync_active_camera_model,
)
from argus.storage.model_registry import ModelRegistry
from argus.storage.models import ModelRecord, ModelStage
from argus.storage.release_pipeline import (
    ReleasePipeline,
    StageTransitionError,
)

logger = structlog.get_logger()

router = APIRouter()

MAX_BATCH_SIZE = 100


def _get_registry(request: Request) -> ModelRegistry | None:
    """Get ModelRegistry from app state, or None if unavailable."""
    return get_registry(request)


def _get_release_pipeline(request: Request) -> ReleasePipeline | None:
    """Get or create ReleasePipeline singleton from app state."""
    cached = getattr(request.app.state, "_release_pipeline", None)
    if cached is not None:
        return cached

    db = getattr(request.app.state, "database", None) or getattr(request.app.state, "db", None)
    if db is None:
        return None
    session_factory = getattr(db, "get_session", None)
    if session_factory is None:
        return None
    config = getattr(request.app.state, "config", None)
    kwargs = {}
    if config:
        anomaly_cfg = getattr(config, "anomaly", None)
        if anomaly_cfg:
            kwargs["min_shadow_days"] = getattr(anomaly_cfg, "min_shadow_days", 3)
            kwargs["min_canary_days"] = getattr(anomaly_cfg, "min_canary_days", 7)
    pipeline = ReleasePipeline(session_factory=session_factory, **kwargs)
    request.app.state._release_pipeline = pipeline
    return pipeline


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
    if _get_registry(request) is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    try:
        triggered_by = "dashboard"
        try:
            body = await request.json()
            if isinstance(body, dict):
                triggered_by = body.get("triggered_by", triggered_by)
        except Exception:
            pass

        record, runtime_synced = activate_model_version(
            request,
            version_id,
            triggered_by=triggered_by,
        )
        return JSONResponse({
            "status": "ok",
            "activated": record.model_version_id,
            "camera_id": record.camera_id,
            "runtime_synced": runtime_synced,
        })
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

    previous, runtime_synced = rollback_camera_model(
        request,
        camera_id,
        triggered_by="dashboard",
    )
    if previous is None:
        return JSONResponse(
            {"error": "No previous model to rollback to"},
            status_code=404,
        )

    return JSONResponse({
        "status": "ok",
        "activated": previous.model_version_id,
        "camera_id": camera_id,
        "runtime_synced": runtime_synced,
    })


def _run_batch_inference(
    image_paths: list[str], detector: object,
) -> tuple[list[dict], int]:
    """Run inference on a batch of images (blocking, runs in thread pool)."""
    import cv2

    results: list[dict] = []
    scored = 0
    threshold = getattr(detector, "threshold", 0.5)

    for img_path in image_paths:
        entry: dict = {"path": img_path}
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

    return results, scored


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

    # Get the camera's anomaly detector via public API
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if camera_manager is None:
        return JSONResponse({"error": "Camera manager not available"}, status_code=503)

    det_status = camera_manager.get_detector_status(camera_id)
    if det_status is None:
        return JSONResponse(
            {"error": f"Camera '{camera_id}' not running or not found"},
            status_code=404,
        )

    pipeline = camera_manager._pipelines.get(camera_id)
    detector = getattr(pipeline, "_anomaly_detector", None) if pipeline else None
    if detector is None:
        return JSONResponse(
            {"error": f"No anomaly detector available for camera '{camera_id}'"},
            status_code=503,
        )

    results, scored = await asyncio.to_thread(
        _run_batch_inference, image_paths, detector,
    )

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


# ── Release pipeline endpoints ──


@router.post("/{version_id}/promote")
async def promote_model(request: Request, version_id: str):
    """Promote a model to the next release stage.

    Body: { target_stage, triggered_by, reason?, canary_camera_id? }
    """
    pipeline = _get_release_pipeline(request)
    if pipeline is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    target_stage = body.get("target_stage")
    triggered_by = body.get("triggered_by")
    if not target_stage or not triggered_by:
        return JSONResponse(
            {"error": "target_stage and triggered_by are required"},
            status_code=400,
        )

    try:
        record = pipeline.transition(
            model_version_id=version_id,
            target_stage=target_stage,
            triggered_by=triggered_by,
            reason=body.get("reason"),
            canary_camera_id=body.get("canary_camera_id"),
        )
        return JSONResponse({
            "status": "ok",
            "model": record.to_dict(),
            "runtime_synced": (
                sync_active_camera_model(request, record.camera_id)
                if target_stage == ModelStage.PRODUCTION.value
                else False
            ),
        })
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except StageTransitionError as e:
        return JSONResponse({"error": str(e)}, status_code=409)


@router.post("/{version_id}/retire")
async def retire_model(request: Request, version_id: str):
    """Retire a model (any stage → retired)."""
    pipeline = _get_release_pipeline(request)
    if pipeline is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        body = {}

    triggered_by = body.get("triggered_by", "system")

    try:
        record = pipeline.transition(
            model_version_id=version_id,
            target_stage=ModelStage.RETIRED.value,
            triggered_by=triggered_by,
            reason=body.get("reason", "manual retirement"),
        )
        return JSONResponse({
            "status": "ok",
            "model": record.to_dict(),
        })
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except StageTransitionError as e:
        return JSONResponse({"error": str(e)}, status_code=409)


@router.delete("/{version_id}")
async def delete_model(request: Request, version_id: str):
    """Delete a model version (only non-active, non-production models)."""
    registry = _get_registry(request)
    if registry is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    record = registry.get_by_version_id(version_id)
    if record is None:
        return JSONResponse({"error": f"Model not found: {version_id}"}, status_code=404)

    if record.is_active:
        return JSONResponse({"error": "无法删除当前激活的模型"}, status_code=400)
    if record.stage == ModelStage.PRODUCTION.value:
        return JSONResponse({"error": "无法删除生产中的模型"}, status_code=400)

    # Delete model files from disk
    if record.model_path:
        import shutil
        model_dir = Path(record.model_path)
        if model_dir.exists():
            if model_dir.is_dir():
                shutil.rmtree(model_dir, ignore_errors=True)
            else:
                model_dir.unlink(missing_ok=True)
            logger.info("model.files_deleted", path=str(model_dir))

    # Delete from database
    registry.delete_model(version_id)
    logger.info("model.deleted", model_version_id=version_id)

    return JSONResponse({"status": "ok", "deleted": version_id})


@router.get("/{version_id}/stage-history")
async def stage_history(request: Request, version_id: str):
    """Get version event history for a specific model."""
    registry = _get_registry(request)
    if registry is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    events = registry.get_version_events(model_version_id=version_id)
    return JSONResponse({
        "events": [e.to_dict() for e in events],
        "total": len(events),
    })


@router.get("/events/list")
async def list_version_events(
    request: Request,
    camera_id: str | None = None,
    limit: int = 50,
):
    """Query global version transition events."""
    registry = _get_registry(request)
    if registry is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    events = registry.get_version_events(camera_id=camera_id, limit=limit)
    return JSONResponse({
        "events": [e.to_dict() for e in events],
        "total": len(events),
    })


@router.get("/{version_id}/shadow-report")
async def shadow_report(
    request: Request,
    version_id: str,
    camera_id: str | None = None,
    days: int = 7,
):
    """Get shadow inference comparison report."""
    pipeline = _get_release_pipeline(request)
    if pipeline is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    stats = pipeline.get_shadow_stats(
        shadow_version_id=version_id,
        camera_id=camera_id,
        days=days,
    )
    return JSONResponse(stats)


# ── Backbone management endpoints ──


@router.get("/backbone/status")
async def backbone_status(request: Request):
    """Get current backbone loading status."""
    from argus.anomaly.backbone_manager import BackboneManager

    manager = BackboneManager.get_instance()
    return JSONResponse({
        "version": manager.version,
        "loaded": manager.is_loaded,
    })


@router.post("/backbone/upgrade")
async def backbone_upgrade(request: Request):
    """Upgrade the shared backbone.

    Body: { backbone_path: str, version: str, triggered_by: str }
    """
    from pathlib import Path

    from argus.anomaly.backbone_manager import BackboneManager

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    backbone_path = body.get("backbone_path")
    version = body.get("version")
    triggered_by = body.get("triggered_by", "system")

    if not backbone_path or not version:
        return JSONResponse(
            {"error": "backbone_path and version are required"},
            status_code=400,
        )

    manager = BackboneManager.get_instance()
    success = await asyncio.to_thread(manager.upgrade, Path(backbone_path), version)

    if success:
        return JSONResponse({
            "status": "ok",
            "version": version,
        })
    else:
        return JSONResponse(
            {"error": "Backbone upgrade failed"},
            status_code=500,
        )
