"""Model management API routes (C4)."""

from __future__ import annotations

import asyncio
import base64
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from argus.dashboard.api_response import (
    api_conflict,
    api_internal_error,
    api_not_found,
    api_success,
    api_unavailable,
    api_validation_error,
)
from argus.dashboard.model_runtime import (
    activate_model_version,
    get_registry,
    rollback_camera_model,
    sync_active_camera_model,
)
from argus.storage.model_registry import ModelRegistry
from argus.storage.models import ModelRecord, ModelStage, ShadowInferenceLog
from argus.storage.release_pipeline import (
    ReleasePipeline,
    StageTransitionError,
)

logger = structlog.get_logger()

router = APIRouter()

MAX_BATCH_SIZE = 100

# Dedicated thread pool for CPU-intensive model inference (not asyncio default pool)
_AB_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ab-compare")


def _get_registry(request: Request) -> ModelRegistry | None:
    """Get ModelRegistry from app state, or None if unavailable."""
    return get_registry(request)


def _get_session_factory(request: Request):
    """Get SQLAlchemy session factory from app state, or None."""
    db = getattr(request.app.state, "database", None) or getattr(request.app.state, "db", None)
    if db is None:
        return None
    return getattr(db, "get_session", None)


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


@router.get("/status")
def models_status(request: Request) -> JSONResponse:
    """Aggregate health status of every detector across all cameras.

    Returns a flat list of :class:`~argus.core.model_status.ModelStatus`
    records (anomaly + YOLO per camera). The frontend polls this every few
    seconds so operators can see "cam1 anomaly is failing" without having
    to grep server logs.
    """
    manager = getattr(request.app.state, "camera_manager", None)
    if manager is None:
        return api_success({"models": []})

    out: list[dict] = []
    runners = getattr(manager, "_runners", None) or {}
    for _cid, runner in runners.items():
        pipeline = getattr(runner, "_pipeline", None)
        if pipeline is None:
            continue
        anomaly = getattr(pipeline, "_anomaly_detector", None)
        if anomaly is not None and hasattr(anomaly, "status"):
            out.append(anomaly.status.to_dict())
        yolo = getattr(pipeline, "_object_detector", None)
        if yolo is not None and hasattr(yolo, "status"):
            out.append(yolo.status.to_dict())
    return api_success({"models": out})


@router.get("/json")
async def list_models(request: Request, camera_id: str | None = None):
    """List all registered models, optionally filtered by camera_id."""
    registry = _get_registry(request)
    if registry is None:
        return api_unavailable("数据库不可用")

    models = registry.list_models(camera_id=camera_id)
    return api_success({
        "models": [m.to_dict() for m in models],
        "total": len(models),
    })


@router.post("/{version_id}/activate")
async def activate_model(request: Request, version_id: str):
    """Activate a specific model version."""
    if _get_registry(request) is None:
        return api_unavailable("数据库不可用")

    try:
        triggered_by = "dashboard"
        try:
            body = await request.json()
            if isinstance(body, dict):
                triggered_by = body.get("triggered_by", triggered_by)
        except Exception:
            logger.debug("models.request_body_parse_failed", exc_info=True)

        record, runtime_synced = activate_model_version(
            request,
            version_id,
            triggered_by=triggered_by,
        )
        return api_success({
            "activated": record.model_version_id,
            "camera_id": record.camera_id,
            "runtime_synced": runtime_synced,
        })
    except ValueError as e:
        # P1 fix (2026-05): registry rejects activation of candidate models.
        # Distinguish "not found" from "stage gate" so the UI can guide users
        # through the promotion flow instead of pretending the version vanished.
        msg = str(e)
        if "stage" in msg.lower() or "shadow" in msg.lower() or "production" in msg.lower():
            return api_validation_error(msg)
        return api_not_found(msg)


@router.post("/{version_id}/rollback")
async def rollback_model(request: Request, version_id: str):
    """Rollback to the previous model for the same camera as version_id.

    The version_id is used to identify the camera; the actual rollback
    activates the previous model for that camera.
    """
    registry = _get_registry(request)
    if registry is None:
        return api_unavailable("数据库不可用")

    # Look up the camera_id from the version_id
    record = registry.get_by_version_id(version_id)
    if record is None:
        return api_not_found(f"模型版本不存在: {version_id}")
    camera_id = record.camera_id

    previous, runtime_synced = rollback_camera_model(
        request,
        camera_id,
        triggered_by="dashboard",
    )
    if previous is None:
        return api_not_found("没有可回滚的历史模型")

    return api_success({
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
            score = float(prediction.anomaly_score) if hasattr(prediction, "anomaly_score") else (
                float(prediction.get("score", 0.0)) if isinstance(prediction, dict) else float(prediction)
            )
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
        return api_validation_error("无效的JSON请求")

    camera_id = body.get("camera_id")
    image_paths = body.get("image_paths", [])

    if not camera_id:
        return api_validation_error("camera_id is required")
    if not image_paths:
        return api_validation_error("image_paths must be a non-empty list")
    if len(image_paths) > MAX_BATCH_SIZE:
        return api_validation_error(f"Maximum {MAX_BATCH_SIZE} images per batch")

    # Get the camera's anomaly detector via public API
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if camera_manager is None:
        return api_unavailable("相机管理器不可用")

    det_status = camera_manager.get_detector_status(camera_id)
    if det_status is None:
        return api_not_found(f"相机 '{camera_id}' 未运行或不存在")

    pipeline = camera_manager._pipelines.get(camera_id)
    detector = getattr(pipeline, "_anomaly_detector", None) if pipeline else None
    if detector is None:
        return api_unavailable(f"相机 '{camera_id}' 的异常检测器不可用")

    results, scored = await asyncio.to_thread(
        _run_batch_inference, image_paths, detector,
    )

    logger.info(
        "batch_inference.complete",
        camera_id=camera_id,
        total=len(image_paths),
        scored=scored,
    )

    return api_success({
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
        return api_unavailable("数据库不可用")

    try:
        body = await request.json()
    except Exception:
        return api_validation_error("无效的JSON请求")

    target_stage = body.get("target_stage")
    triggered_by = body.get("triggered_by")
    if not target_stage or not triggered_by:
        return api_validation_error("target_stage and triggered_by are required")

    try:
        record = pipeline.transition(
            model_version_id=version_id,
            target_stage=target_stage,
            triggered_by=triggered_by,
            reason=body.get("reason"),
            canary_camera_id=body.get("canary_camera_id"),
        )
        return api_success({
            "model": record.to_dict(),
            "runtime_synced": (
                sync_active_camera_model(request, record.camera_id)
                if target_stage == ModelStage.PRODUCTION.value
                else False
            ),
        })
    except ValueError as e:
        return api_validation_error(str(e))
    except StageTransitionError as e:
        return api_conflict(str(e))


@router.post("/{version_id}/retire")
async def retire_model(request: Request, version_id: str):
    """Retire a model (any stage → retired)."""
    pipeline = _get_release_pipeline(request)
    if pipeline is None:
        return api_unavailable("数据库不可用")

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
        return api_success({
            "model": record.to_dict(),
        })
    except ValueError as e:
        return api_not_found(str(e))
    except StageTransitionError as e:
        return api_conflict(str(e))


@router.delete("/{version_id}")
def delete_model(request: Request, version_id: str):
    """Delete a model version (only non-active, non-production models)."""
    registry = _get_registry(request)
    if registry is None:
        return api_unavailable("数据库不可用")

    record = registry.get_by_version_id(version_id)
    if record is None:
        return api_not_found(f"模型不存在: {version_id}")

    if record.is_active:
        return api_validation_error("无法删除当前激活的模型")

    if record.stage in (ModelStage.SHADOW.value, ModelStage.CANARY.value, ModelStage.PRODUCTION.value):
        return api_validation_error(
            f"无法删除处于 {record.stage} 阶段的模型，请先退役"
        )

    # Delete model files from disk — only if no other model shares the same path
    if record.model_path:
        other_models = [
            m for m in registry.list_models()
            if m.model_path == record.model_path
            and m.model_version_id != version_id
        ]
        if not other_models:
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

    return api_success({"deleted": version_id})


def _get_trainer(request: Request):
    """Construct a ModelTrainer from app state config."""
    from argus.anomaly.baseline import BaselineManager
    from argus.anomaly.trainer import ModelTrainer

    config = getattr(request.app.state, "config", None)
    if config is None:
        return None
    baselines_dir = str(config.storage.baselines_dir)
    models_dir = str(Path(config.storage.baselines_dir).parent / "models")
    exports_dir = str(Path(config.storage.baselines_dir).parent / "exports")
    bm = BaselineManager(baselines_dir=baselines_dir)
    return ModelTrainer(
        baseline_manager=bm,
        models_dir=models_dir,
        exports_dir=exports_dir,
    )


def _get_baseline_manager(request: Request):
    """Get or create a BaselineManager from app state or config."""
    baseline_mgr = getattr(request.app.state, "baseline_manager", None)
    if baseline_mgr is not None:
        return baseline_mgr
    config = getattr(request.app.state, "config", None)
    if config:
        from argus.anomaly.baseline import BaselineManager
        return BaselineManager(baselines_dir=str(config.storage.baselines_dir))
    return None


@router.post("/{version_id}/reexport")
async def reexport_model(request: Request, version_id: str):
    """Re-export model from checkpoint, fitting PostProcessor MinMax.

    Body: { export_format?: str, quantization?: str }
    """
    registry = _get_registry(request)
    if registry is None:
        return api_unavailable("数据库不可用")

    record = registry.get_by_version_id(version_id)
    if record is None:
        return api_not_found(f"模型不存在: {version_id}")

    if not record.model_path:
        return api_validation_error("模型没有关联的文件路径")

    model_dir = Path(record.model_path)
    if not model_dir.exists():
        return api_not_found(f"模型目录不存在: {record.model_path}")

    try:
        body = await request.json()
    except Exception:
        body = {}

    export_format = body.get("export_format", "openvino")
    quantization = body.get("quantization", "fp16")

    if export_format not in {"openvino", "onnx", "torch"}:
        return api_validation_error(f"不支持的导出格式: {export_format}")
    if quantization not in {"fp16", "fp32", "int8"}:
        return api_validation_error(f"不支持的量化方式: {quantization}")

    trainer = _get_trainer(request)
    if trainer is None:
        return api_unavailable("配置不可用")

    result = await asyncio.to_thread(
        trainer.reexport_model,
        model_dir=model_dir,
        export_format=export_format,
        quantization=quantization,
        model_type=record.model_type,
    )

    if result.get("status") == "error":
        return api_internal_error(result["error"])

    return api_success(result)


@router.post("/{version_id}/recalibrate")
async def recalibrate_model(request: Request, version_id: str):
    """Re-calibrate score normalization using baseline images.

    Runs conformal calibration + sigmoid recalibration, persists to calibration.json.
    """
    registry = _get_registry(request)
    if registry is None:
        return api_unavailable("数据库不可用")

    record = registry.get_by_version_id(version_id)
    if record is None:
        return api_not_found(f"模型不存在: {version_id}")

    if not record.model_path:
        return api_validation_error("模型没有关联的文件路径")

    model_dir = Path(record.model_path)

    # Extract zone_id from model_path (data/models/{camera_id}/{zone_id})
    zone_id = model_dir.parts[-1] if len(model_dir.parts) >= 2 else "default"

    baseline_mgr = _get_baseline_manager(request)
    if baseline_mgr is None:
        return api_unavailable("基线管理器不可用")

    baseline_dir = baseline_mgr.get_baseline_dir(record.camera_id, zone_id)
    if not baseline_dir.is_dir():
        return api_not_found(f"基线目录不存在: {baseline_dir}")

    trainer = _get_trainer(request)
    if trainer is None:
        return api_unavailable("配置不可用")

    result = await asyncio.to_thread(
        trainer.recalibrate_model,
        model_dir=model_dir,
        baseline_dir=baseline_dir,
        camera_id=record.camera_id,
        zone_id=zone_id,
    )

    if result.get("status") == "error":
        return api_internal_error(result["error"])

    return api_success(result)


@router.get("/{version_id}/stage-history")
async def stage_history(request: Request, version_id: str):
    """Get version event history for a specific model."""
    registry = _get_registry(request)
    if registry is None:
        return api_unavailable("数据库不可用")

    events = registry.get_version_events(model_version_id=version_id)
    return api_success({
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
        return api_unavailable("数据库不可用")

    events = registry.get_version_events(camera_id=camera_id, limit=limit)
    return api_success({
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
        return api_unavailable("数据库不可用")

    stats = pipeline.get_shadow_stats(
        shadow_version_id=version_id,
        camera_id=camera_id,
        days=days,
    )
    return api_success(stats)


# ── A/B Comparison endpoints ──


def _query_shadow_logs(session, version_id: str, camera_id: str | None, days: int):
    """Build a filtered ShadowInferenceLog query (shared by ab-scores/ab-distribution)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    query = (
        session.query(ShadowInferenceLog)
        .filter(ShadowInferenceLog.shadow_version_id == version_id)
        .filter(ShadowInferenceLog.timestamp >= cutoff)
    )
    if camera_id:
        query = query.filter(ShadowInferenceLog.camera_id == camera_id)
    return query


@router.get("/{version_id}/ab-scores")
async def ab_compare_scores(
    request: Request,
    version_id: str,
    camera_id: str | None = None,
    days: int = 7,
    limit: int = 500,
):
    """Get time-series shadow vs production scores for A/B comparison chart."""
    session_factory = _get_session_factory(request)
    if session_factory is None:
        return api_unavailable("数据库不可用")

    with session_factory() as session:
        logs = list(
            _query_shadow_logs(session, version_id, camera_id, days)
            .order_by(ShadowInferenceLog.timestamp.desc())
            .limit(limit)
            .all()
        )

    logs.reverse()

    scores = [
        {
            "t": log.timestamp.isoformat() if log.timestamp else None,
            "shadow": round(log.shadow_score, 4),
            "production": round(log.production_score, 4) if log.production_score is not None else None,
            "shadow_alert": log.shadow_would_alert,
            "prod_alert": log.production_alerted,
        }
        for log in logs
    ]

    return api_success({
        "scores": scores,
        "total": len(scores),
        "shadow_version_id": version_id,
    })


@router.get("/{version_id}/ab-distribution")
async def ab_compare_distribution(
    request: Request,
    version_id: str,
    camera_id: str | None = None,
    days: int = 7,
    bins: int = 20,
):
    """Get score distribution histograms for shadow vs production."""
    session_factory = _get_session_factory(request)
    if session_factory is None:
        return api_unavailable("数据库不可用")

    with session_factory() as session:
        # Only fetch the two score columns, not full ORM objects
        rows = (
            _query_shadow_logs(session, version_id, camera_id, days)
            .with_entities(ShadowInferenceLog.shadow_score, ShadowInferenceLog.production_score)
            .limit(50000)
            .all()
        )

    if not rows:
        return api_success({
            "bin_edges": [],
            "shadow_counts": [],
            "production_counts": [],
            "total_samples": 0,
        })

    bin_edges = [round(i / bins, 4) for i in range(bins + 1)]
    shadow_counts = [0] * bins
    prod_counts = [0] * bins

    for shadow_score, prod_score in rows:
        shadow_counts[min(int(shadow_score * bins), bins - 1)] += 1
        if prod_score is not None:
            prod_counts[min(int(prod_score * bins), bins - 1)] += 1

    return api_success({
        "bin_edges": bin_edges,
        "shadow_counts": shadow_counts,
        "production_counts": prod_counts,
        "total_samples": len(rows),
    })


# ── Shadow detector cache (LRU, max 3 entries) ──
from collections import OrderedDict

_shadow_detector_cache: OrderedDict[str, object] = OrderedDict()
_shadow_cache_lock = threading.Lock()


def _get_or_load_shadow_detector(shadow_model_path: Path, shadow_version_id: str, threshold: float):
    """Get cached shadow detector or load it. Thread-safe, LRU eviction (max 3)."""
    with _shadow_cache_lock:
        cached = _shadow_detector_cache.get(shadow_version_id)
        if cached is not None:
            # Move to end for LRU ordering
            _shadow_detector_cache.move_to_end(shadow_version_id)  # type: ignore[attr-defined]
            return cached

    from argus.anomaly.detector import AnomalibDetector

    det = AnomalibDetector(model_path=shadow_model_path, threshold=threshold)

    with _shadow_cache_lock:
        if len(_shadow_detector_cache) >= 3:
            # Evict least-recently-used (first item)
            evicted_key, _ = _shadow_detector_cache.popitem(last=False)  # type: ignore[call-arg]
            logger.debug("ab_compare.cache_evict", evicted=evicted_key)
        _shadow_detector_cache[shadow_version_id] = det

    logger.info("ab_compare.shadow_loaded", version_id=shadow_version_id)
    return det


def _run_ab_heatmap(
    frame: np.ndarray,
    production_detector: object,
    shadow_model_path: Path,
    shadow_version_id: str,
    shadow_threshold: float,
) -> dict:
    """Run both models on a frame and produce heatmap images (blocking)."""
    result: dict = {}

    # --- Production model ---
    t0 = time.perf_counter()
    try:
        prod_pred = production_detector.predict(frame)
        result["production_score"] = round(float(prod_pred.anomaly_score), 4)
        result["production_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        if prod_pred.anomaly_map is not None:
            heatmap = _anomaly_map_to_jpeg(prod_pred.anomaly_map, frame.shape[:2])
            result["production_heatmap"] = base64.b64encode(heatmap).decode()
        else:
            result["production_heatmap"] = None
    except Exception as e:
        result["production_error"] = str(e)

    # --- Shadow model (cached) ---
    t1 = time.perf_counter()
    try:
        shadow_det = _get_or_load_shadow_detector(
            shadow_model_path, shadow_version_id, shadow_threshold,
        )
        shadow_pred = shadow_det.predict(frame)
        result["shadow_score"] = round(float(shadow_pred.anomaly_score), 4)
        result["shadow_latency_ms"] = round((time.perf_counter() - t1) * 1000, 1)

        if shadow_pred.anomaly_map is not None:
            heatmap = _anomaly_map_to_jpeg(shadow_pred.anomaly_map, frame.shape[:2])
            result["shadow_heatmap"] = base64.b64encode(heatmap).decode()
        else:
            result["shadow_heatmap"] = None
    except Exception as e:
        result["shadow_error"] = str(e)

    # --- Original frame thumbnail ---
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    result["original_frame"] = base64.b64encode(buf.tobytes()).decode()

    return result


def _anomaly_map_to_jpeg(anomaly_map: np.ndarray, target_hw: tuple[int, int]) -> bytes:
    """Convert an anomaly map (float or uint8) to a JET-colored JPEG thumbnail."""
    amap = anomaly_map
    if amap.ndim == 3:
        amap = amap[:, :, 0]

    # Normalize to 0-255
    if amap.dtype != np.uint8:
        amap = np.clip(amap, 0, 1)
        amap = (amap * 255).astype(np.uint8)

    # Resize to match frame dimensions
    h, w = target_hw
    if amap.shape[:2] != (h, w):
        amap = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)

    colored = cv2.applyColorMap(amap, cv2.COLORMAP_JET)
    _, buf = cv2.imencode(".jpg", colored, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


@router.post("/ab-live-compare")
async def ab_live_compare(request: Request):
    """Run both production and shadow model on a live camera frame.

    Body: { camera_id: str, shadow_version_id: str }
    Returns: base64 encoded heatmaps for production and shadow, plus scores.
    """
    try:
        body = await request.json()
    except Exception:
        return api_validation_error("无效的JSON请求")

    camera_id = body.get("camera_id")
    shadow_version_id = body.get("shadow_version_id")

    if not camera_id or not shadow_version_id:
        return api_validation_error("camera_id and shadow_version_id are required")

    # Get camera frame
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if camera_manager is None:
        return api_unavailable("相机管理器不可用")

    frame = camera_manager.get_latest_frame(camera_id)
    if frame is None:
        return api_not_found(f"相机 '{camera_id}' 无可用帧")

    # Get production detector
    pipeline = camera_manager.get_pipeline(camera_id)
    production_detector = getattr(pipeline, "_anomaly_detector", None) if pipeline else None
    if production_detector is None:
        return api_unavailable(f"相机 '{camera_id}' 的异常检测器不可用")

    # Get shadow model path from registry
    registry = _get_registry(request)
    if registry is None:
        return api_unavailable("数据库不可用")

    shadow_record = registry.get_by_version_id(shadow_version_id)
    if shadow_record is None:
        return api_not_found(f"影子模型不存在: {shadow_version_id}")
    if not shadow_record.model_path:
        return api_validation_error("影子模型没有关联的文件路径")

    shadow_model_path = Path(shadow_record.model_path)
    if not shadow_model_path.exists():
        return api_not_found(f"影子模型文件不存在: {shadow_record.model_path}")

    shadow_threshold = 0.5
    if shadow_record.calibration_thresholds:
        import json
        try:
            cal = json.loads(shadow_record.calibration_thresholds)
            shadow_threshold = cal.get("threshold", 0.5)
        except Exception:
            logger.warning("models.calibration_thresholds_parse_failed", exc_info=True)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _AB_EXECUTOR,
        _run_ab_heatmap, frame, production_detector,
        shadow_model_path, shadow_version_id, shadow_threshold,
    )

    return api_success(result)


# ── Backbone management endpoints ──


@router.get("/backbone/status")
async def backbone_status(request: Request):
    """Get current backbone loading status."""
    from argus.anomaly.backbone_manager import BackboneManager

    manager = BackboneManager.get_instance()
    return api_success({
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
        return api_validation_error("无效的JSON请求")

    backbone_path = body.get("backbone_path")
    version = body.get("version")
    triggered_by = body.get("triggered_by", "system")

    if not backbone_path or not version:
        return api_validation_error("backbone_path and version are required")

    manager = BackboneManager.get_instance()
    success = await asyncio.to_thread(manager.upgrade, Path(backbone_path), version)

    if success:
        return api_success({
            "version": version,
        })
    else:
        return api_internal_error("Backbone升级失败")


@router.get("/threshold-preview")
def threshold_preview(
    request: Request,
    camera_id: str | None = None,
    threshold: float = 0.5,
    days: int = 7,
) -> JSONResponse:
    """Preview detection metrics at a given anomaly threshold.

    Uses historical inference records to compute:
    - Total frames
    - Frames above threshold (would trigger alert)
    - Alert rate (fraction of frames above threshold)
    - Score percentiles (p50, p90, p95, p99)

    This allows operators to tune the threshold with real data feedback.
    """
    from sqlalchemy import select, func as sa_func
    from argus.storage.models import InferenceRecord

    db = request.app.state.db
    if db is None:
        return api_unavailable("数据库未初始化")

    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()

    with db.get_session() as session:
        base = select(InferenceRecord).where(InferenceRecord.timestamp >= cutoff_ts)
        if camera_id:
            base = base.where(InferenceRecord.camera_id == camera_id)

        # Get all scores for this camera in the time window
        score_stmt = select(InferenceRecord.anomaly_score).where(
            InferenceRecord.timestamp >= cutoff_ts
        )
        if camera_id:
            score_stmt = score_stmt.where(InferenceRecord.camera_id == camera_id)

        scores = [row[0] for row in session.execute(score_stmt).all()]

    if not scores:
        return api_success(data={
            "total_frames": 0,
            "above_threshold": 0,
            "alert_rate": 0.0,
            "threshold": threshold,
            "percentiles": {},
            "message": "暂无推理记录",
        })

    scores_arr = np.array(scores)
    total = len(scores_arr)
    above = int(np.sum(scores_arr >= threshold))
    alert_rate = above / total if total > 0 else 0.0

    percentiles = {
        "p50": float(np.percentile(scores_arr, 50)),
        "p75": float(np.percentile(scores_arr, 75)),
        "p90": float(np.percentile(scores_arr, 90)),
        "p95": float(np.percentile(scores_arr, 95)),
        "p99": float(np.percentile(scores_arr, 99)),
        "max": float(np.max(scores_arr)),
        "min": float(np.min(scores_arr)),
        "mean": float(np.mean(scores_arr)),
    }

    # Generate histogram for score distribution
    hist_counts, hist_edges = np.histogram(scores_arr, bins=50, range=(0.0, 1.0))
    histogram = [
        {"bin_start": float(hist_edges[i]), "bin_end": float(hist_edges[i + 1]), "count": int(hist_counts[i])}
        for i in range(len(hist_counts))
    ]

    return api_success(data={
        "total_frames": total,
        "above_threshold": above,
        "alert_rate": round(alert_rate, 6),
        "threshold": threshold,
        "percentiles": percentiles,
        "histogram": histogram,
        "days": days,
        "camera_id": camera_id,
    })
