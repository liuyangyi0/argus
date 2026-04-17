"""Baseline and model management routes (Chinese UI)."""

from __future__ import annotations

import asyncio
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import Response

from argus.dashboard.api_response import (
    api_conflict,
    api_forbidden,
    api_internal_error,
    api_not_found,
    api_success,
    api_unavailable,
    api_validation_error,
)
from argus.capture.baseline_job import run_baseline_capture_job, BaselineCaptureJobConfig
from argus.config.schema import BaselineCaptureConfig
from argus.core.model_discovery import resolve_runtime_model_path
from argus.dashboard.forms import htmx_toast_headers, parse_request_form
from argus.dashboard.model_runtime import activate_model_version, find_registered_model_by_path

logger = structlog.get_logger()

router = APIRouter()

_DEFAULT_ZONE_ID = "default"


def _camera_baseline_root(
    baselines_dir: Path,
    camera_id: str,
    zone_id: str = _DEFAULT_ZONE_ID,
) -> Path:
    """Return the default-zone root for a camera, with legacy flat-layout fallback."""
    zone_root = baselines_dir / camera_id / zone_id
    if zone_root.is_dir():
        return zone_root
    return baselines_dir / camera_id


def _iter_camera_baseline_versions(
    baselines_dir: Path,
    camera_id: str,
    zone_id: str = _DEFAULT_ZONE_ID,
):
    """Yield version directories for one camera's default baseline zone."""
    base_dir = _camera_baseline_root(baselines_dir, camera_id, zone_id)
    if not base_dir.exists():
        return
    for version_dir in sorted(base_dir.iterdir()):
        if version_dir.is_dir():
            yield version_dir


def _resolve_baseline_version_dir(
    baselines_dir: Path,
    camera_id: str,
    version: str,
    zone_id: str = _DEFAULT_ZONE_ID,
) -> Path:
    """Resolve a version directory for the default zone, including legacy flat paths."""
    return _camera_baseline_root(baselines_dir, camera_id, zone_id) / version


def _read_capture_meta(version_dir: Path) -> dict:
    """Load capture metadata if present."""
    import json as _json

    meta_path = version_dir / "capture_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return _json.loads(meta_path.read_text())
    except (ValueError, OSError):
        return {}


def _count_version_images(version_dir: Path) -> int:
    """Count images stored in a baseline version directory."""
    return len(list(version_dir.glob("*.png"))) + len(list(version_dir.glob("*.jpg")))


def _reset_current_marker_after_delete(base_dir: Path, deleted_version: str) -> None:
    """Update current.txt when the referenced version has been deleted."""
    marker = base_dir / "current.txt"
    if not marker.exists():
        return
    current_version = marker.read_text().strip()
    if current_version != deleted_version:
        return

    remaining_versions = sorted([path for path in base_dir.iterdir() if path.is_dir()])
    if remaining_versions:
        marker.write_text(remaining_versions[-1].name)
        return

    marker.unlink(missing_ok=True)


@router.get("/list/json")
def baseline_list_json(request: Request):
    """JSON API: list baselines by camera with version info."""
    config = request.app.state.config
    if not config:
        return api_unavailable("配置不可用")

    baselines_dir = Path(config.storage.baselines_dir)
    if not baselines_dir.exists():
        return api_success({"baselines": []})

    baselines = []
    for cam_dir in sorted(baselines_dir.iterdir()):
        if not cam_dir.is_dir():
            continue
        camera_id = cam_dir.name
        for ver_dir in _iter_camera_baseline_versions(baselines_dir, camera_id):
            image_count = _count_version_images(ver_dir)
            if not image_count:
                continue
            meta = _read_capture_meta(ver_dir)
            lifecycle = _get_lifecycle(request)
            version_state = None
            if lifecycle:
                ver_rec = lifecycle.get_version(camera_id, _DEFAULT_ZONE_ID, ver_dir.name)
                version_state = ver_rec.state if ver_rec else None
            baselines.append({
                "camera_id": camera_id,
                "version": ver_dir.name,
                "image_count": image_count,
                "session_label": meta.get("session_label", ""),
                "status": "ready",
                "state": version_state,
            })

    return api_success({"baselines": baselines})


@router.get("/models/json")
async def models_list_json(request: Request):
    """JSON API: list trained models found in data/models/."""
    config = request.app.state.config
    if not config:
        return api_unavailable("配置不可用")

    models_dir = Path(config.storage.models_dir)
    if not models_dir.exists():
        return api_success({"models": []})

    result = []
    for cam_dir in sorted(models_dir.iterdir()):
        if not cam_dir.is_dir():
            continue
        camera_id = cam_dir.name

        for pattern, fmt in [("model.xml", "openvino"), ("model.onnx", "onnx"),
                              ("model.pt", "pytorch"), ("model.ckpt", "checkpoint")]:
            for model_file in sorted(cam_dir.rglob(pattern),
                                     key=lambda p: p.stat().st_mtime, reverse=True):
                mtime = time.strftime("%Y-%m-%dT%H:%M:%S",
                                      time.localtime(model_file.stat().st_mtime))
                size_mb = model_file.stat().st_size / (1024 * 1024)
                rel_path = str(model_file.relative_to(Path(".")))
                result.append({
                    "camera_id": camera_id,
                    "format": fmt,
                    "size_mb": round(size_mb, 1),
                    "trained_at": mtime,
                    "model_path": rel_path,
                })

    return api_success({"models": result})


@router.delete("/models/by-path")
async def delete_model_by_path(request: Request):
    """Delete a trained model file by its path (for filesystem-only models)."""
    config = request.app.state.config
    if not config:
        return api_unavailable("配置不可用")

    body = await request.json()
    model_path = body.get("model_path")
    if not model_path:
        return api_validation_error("model_path is required")

    target = Path(model_path).resolve()
    models_dir = Path(config.storage.models_dir).resolve()
    if not str(target).startswith(str(models_dir)):
        return api_forbidden("路径不在模型目录下")

    if not target.exists():
        return api_not_found(f"文件不存在: {model_path}")

    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
    else:
        target.unlink(missing_ok=True)
    logger.info("model.file_deleted", path=model_path)

    return api_success({"deleted": model_path})


@router.get("/training-history/json")
async def training_history_json(request: Request):
    """JSON API: training history with quality grades."""
    database = getattr(request.app.state, "database", None)
    if not database:
        return api_success({"records": []})

    camera_id = request.query_params.get("camera_id")
    records = database.get_training_history(camera_id=camera_id, limit=50)
    return api_success({"records": [r.to_dict() for r in records]})


@router.get("/training-history/{record_id}/metrics")
async def training_history_metrics(request: Request, record_id: int):
    """Phase 2: per-record P/R curve + raw scores/labels for threshold slider.

    Returns scores/labels as two aligned arrays (length ≤ a few hundred typically),
    plus a pre-computed PR curve. The frontend can recompute P/R/F1 at any
    threshold locally by thresholding the scores — no model re-run needed.
    """
    database = getattr(request.app.state, "database", None)
    if not database:
        return api_unavailable("数据库不可用")

    record = database.get_training_record(record_id)
    if record is None:
        return api_not_found(f"训练记录不存在: {record_id}")

    if not record.val_scores_json or not record.val_labels_json:
        return api_success({
            "record_id": record_id,
            "has_labeled_eval": False,
            "message": "本次训练没有真实标注评估数据",
        })

    import json as _json
    try:
        scores = _json.loads(record.val_scores_json)
        labels = _json.loads(record.val_labels_json)
    except Exception:
        logger.warning("baseline.metrics_json_parse_failed", record_id=record_id, exc_info=True)
        return api_internal_error("评估数据解析失败")

    if len(scores) != len(labels):
        return api_internal_error(
            f"评估数据长度不一致: scores={len(scores)} vs labels={len(labels)}",
        )

    from argus.anomaly.metrics import (
        compute_pr_curve,
        evaluate_at_threshold,
        find_optimal_threshold,
    )

    import numpy as np
    y_scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(labels, dtype=np.int64)

    threshold_used = record.threshold_recommended if record.threshold_recommended else 0.7
    metrics_at_threshold = evaluate_at_threshold(y_true, y_scores, threshold_used)
    pr_curve = compute_pr_curve(y_true, y_scores)
    optimal_threshold = find_optimal_threshold(y_true, y_scores, target="f1")

    cm_stored = None
    if record.val_confusion_matrix:
        try:
            cm_stored = _json.loads(record.val_confusion_matrix)
        except Exception:
            cm_stored = None

    return api_success({
        "record_id": record_id,
        "has_labeled_eval": True,
        "scores": scores,
        "labels": labels,
        "threshold_used": threshold_used,
        "metrics_at_threshold": metrics_at_threshold,
        "pr_curve": pr_curve,
        "optimal_f1_threshold": optimal_threshold,
        "stored_confusion_matrix": cm_stored,
        "stored_metrics": {
            "precision": record.val_precision,
            "recall": record.val_recall,
            "f1": record.val_f1,
            "auroc": record.val_auroc,
            "pr_auc": record.val_pr_auc,
        },
        "sample_count": len(scores),
    })


@router.delete("/version")
async def delete_baseline_version(request: Request):
    """Delete an entire baseline version directory and its lifecycle record."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    camera_id = data.get("camera_id", "")
    zone_id = data.get("zone_id", _DEFAULT_ZONE_ID)
    version = data.get("version", "")
    user = data.get("user", "operator")

    if not camera_id:
        camera_id = request.query_params.get("camera_id", "")
    if not version:
        version = request.query_params.get("version", "")
    if zone_id == _DEFAULT_ZONE_ID:
        zone_id = request.query_params.get("zone_id", zone_id)
    if user == "operator":
        user = request.query_params.get("user", user)

    if not camera_id or not version:
        return api_validation_error("camera_id and version are required")

    config = request.app.state.config
    if not config:
        return api_unavailable("配置不可用")

    baselines_dir = Path(config.storage.baselines_dir)
    base_dir = _camera_baseline_root(baselines_dir, camera_id, zone_id)
    version_dir = _resolve_baseline_version_dir(baselines_dir, camera_id, version, zone_id)
    if not version_dir.exists() or not version_dir.is_dir():
        return api_not_found("基线版本不存在")

    lifecycle = _get_lifecycle(request)
    if lifecycle is not None:
        version_record = lifecycle.get_version(camera_id, zone_id, version)
        if version_record is not None and version_record.state == "active":
            return api_validation_error("生产中的基线版本不能直接删除，请先退役")

    await asyncio.to_thread(shutil.rmtree, version_dir)
    _reset_current_marker_after_delete(base_dir, version)

    if lifecycle is not None:
        client_ip = request.client.host if request.client else ""
        lifecycle.delete_version(camera_id, zone_id, version, user=user, ip_address=client_ip)

    logger.info(
        "baseline.version_dir_deleted",
        camera_id=camera_id,
        zone_id=zone_id,
        version=version,
        user=user,
    )
    return api_success({"camera_id": camera_id, "version": version})


@router.post("/version/delete")
async def delete_baseline_version_post(request: Request):
    """POST alias for version deletion to avoid DELETE-body compatibility issues."""
    return await delete_baseline_version(request)


@router.post("/deploy")
async def deploy_model(request: Request):
    """Deploy a trained model to a camera (hot-reload)."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("服务不可用")

    data = await request.json()
    camera_id = data.get("camera_id", "")
    model_path = data.get("model_path", "")

    if not camera_id or not model_path:
        return api_validation_error("缺少参数")

    # Path safety
    allowed_roots = [Path("data/models").resolve(), Path("data/exports").resolve()]
    resolved = Path(model_path).resolve()
    if not any(str(resolved).startswith(str(root)) for root in allowed_roots):
        return api_validation_error("模型路径无效")
    if not resolved.exists():
        return api_not_found("模型文件不存在")

    if camera_id not in camera_manager._pipelines:
        return api_not_found(f"摄像头 {camera_id} 不存在")

    resolved_runtime_model = resolve_runtime_model_path(resolved, camera_id)
    if resolved_runtime_model is None:
        return api_not_found("模型目录中未找到可部署的模型文件 (.xml/.pt)")
    resolved = resolved_runtime_model
    logger.info("deploy.resolved_model_file", path=str(resolved))

    registry_record = find_registered_model_by_path(
        request,
        Path(model_path),
        camera_id=camera_id,
    )
    if registry_record is None:
        registry_record = find_registered_model_by_path(
            request,
            resolved,
            camera_id=camera_id,
        )
    if registry_record is not None:
        _, success = activate_model_version(
            request,
            registry_record.model_version_id,
            triggered_by="dashboard",
        )
    else:
        success = camera_manager.reload_model(camera_id, str(resolved))

    if success:
        audit = getattr(request.app.state, "audit_logger", None)
        client_ip = request.client.host if request.client else ""
        if audit:
            audit.log(
                user="operator",
                action="deploy_model",
                target_type="camera",
                target_id=camera_id,
                detail=model_path,
                ip_address=client_ip,
            )
        return api_success(
            {
                "model_version_id": registry_record.model_version_id if registry_record else None,
            },
            headers=htmx_toast_headers("模型已部署"),
        )
    return api_internal_error("模型部署失败")


# ── Advanced Baseline Capture Job Endpoints ──


@router.post("/job")
async def start_capture_job(request: Request):
    """Start an advanced baseline capture job with strategy selection."""
    app = request.app
    camera_manager = app.state.camera_manager
    task_manager = app.state.task_manager
    config = app.state.config
    if not camera_manager or not task_manager or not config:
        return api_unavailable("服务不可用")

    form = await parse_request_form(request)
    camera_id = form.get("camera_id", "")
    session_label = form.get("session_label", "daytime")

    if not camera_id:
        return api_validation_error("请选择摄像头")

    if "count" in form or "interval" in form:
        target_frames = int(form.get("count", 100))
        interval = float(form.get("interval", 2.0))
        duration_hours = max(target_frames * interval / 3600, 1 / 3600)
        sampling_strategy = "uniform"
        diversity_threshold = config.baseline_capture.diversity_threshold
        frames_per_period = config.baseline_capture.frames_per_period
    else:
        target_frames = int(form.get("target_frames", 1000))
        duration_hours = float(form.get("duration_hours", 24.0))
        sampling_strategy = form.get("sampling_strategy", config.baseline_capture.default_strategy)
        diversity_threshold = float(
            form.get("diversity_threshold", config.baseline_capture.diversity_threshold)
        )
        frames_per_period = int(
            form.get("frames_per_period", config.baseline_capture.frames_per_period)
        )

    job_config = BaselineCaptureJobConfig(
        camera_id=camera_id,
        session_label=session_label,
        target_frames=target_frames,
        duration_hours=duration_hours,
        sampling_strategy=sampling_strategy,
        storage_path=str(config.storage.baselines_dir),
        quality_config=config.capture_quality,
        pause_on_anomaly_lock=config.baseline_capture.pause_on_anomaly_lock,
        diversity_threshold=diversity_threshold,
        dino_backbone=config.baseline_capture.dino_backbone,
        dino_image_size=config.baseline_capture.dino_image_size,
        active_sleep_min_seconds=config.baseline_capture.active_sleep_min_seconds,
        active_cpu_threads=config.baseline_capture.active_cpu_threads,
        schedule_periods=dict(config.baseline_capture.schedule_periods),
        frames_per_period=frames_per_period,
        post_capture_review=config.baseline_capture.post_capture_review,
        review_flag_percentile=config.baseline_capture.review_flag_percentile,
        models_dir=str(config.storage.models_dir),
        exports_dir=str(config.models.anomalib_export_dir),
    )

    pause_ev = threading.Event()
    pause_ev.set()  # not paused
    abort_ev = threading.Event()
    lifecycle = getattr(app.state, "baseline_lifecycle", None)

    try:
        task_id = task_manager.submit(
            "baseline_capture",
            run_baseline_capture_job,
            camera_id=camera_id,
            job_config=job_config,
            camera_manager=camera_manager,
            pause_event=pause_ev,
            abort_event=abort_ev,
            lifecycle=lifecycle,
        )
        # Sync events so TaskManager.pause_task/abort_task work
        task_info = task_manager.get_task(task_id)
        if task_info:
            task_info.pause_event = pause_ev
            task_info.abort_event = abort_ev
    except RuntimeError as e:
        return api_conflict(str(e))

    return api_success({"task_id": task_id})


@router.post("/capture")
async def start_capture_legacy(request: Request):
    """Backward-compatible alias for quick capture clients still posting to /capture."""
    return await start_capture_job(request)


@router.post("/job/{task_id}/pause")
async def pause_capture_job(task_id: str, request: Request):
    """Pause a running baseline capture job."""
    task_manager = request.app.state.task_manager
    if task_manager.pause_task(task_id):
        return api_success({"task_id": task_id, "status": "paused"})
    return api_not_found("任务不存在或未在运行")


@router.post("/job/{task_id}/resume")
async def resume_capture_job(task_id: str, request: Request):
    """Resume a paused baseline capture job."""
    task_manager = request.app.state.task_manager
    if task_manager.resume_task(task_id):
        return api_success({"task_id": task_id, "status": "resumed"})
    return api_not_found("任务不存在或未暂停")


@router.post("/job/{task_id}/abort")
async def abort_capture_job(task_id: str, request: Request):
    """Abort a running or paused baseline capture job."""
    task_manager = request.app.state.task_manager
    if task_manager.abort_task(task_id):
        return api_success({"task_id": task_id, "status": "aborting"})
    return api_not_found("任务不存在或未激活")


@router.get("/job/{task_id}")
async def get_capture_job_status(task_id: str, request: Request):
    """Get the status of a baseline capture job."""
    task_manager = request.app.state.task_manager
    task = task_manager.get_task(task_id)
    if task is None:
        return api_not_found("任务不存在")
    return api_success({
        "task_id": task.task_id,
        "task_type": task.task_type,
        "camera_id": task.camera_id,
        "status": task.status.value,
        "progress": task.progress,
        "message": task.message,
        "error": task.error,
        "result": task.result if task.status.value in ("complete", "aborted", "failed") else None,
    })


@router.post("/compare")
async def compare_models_route(request: Request):
    """Compare two trained models (TRN-008)."""
    database = getattr(request.app.state, "database", None)
    if not database:
        return api_unavailable("数据库不可用")

    data = await request.json()
    old_record_id = data.get("old_record_id")
    new_record_id = data.get("new_record_id")

    if not old_record_id or not new_record_id:
        return api_validation_error("缺少参数")

    old_record = database.get_training_record(int(old_record_id))
    new_record = database.get_training_record(int(new_record_id))

    if not old_record or not new_record:
        return api_not_found("训练记录不存在")

    if not old_record.model_path or not new_record.model_path:
        return api_validation_error("模型路径不存在")

    # Find a validation set (prefer new record's baseline)
    config = request.app.state.config
    baselines_dir = Path(config.storage.baselines_dir)
    val_dir = baselines_dir / new_record.camera_id / new_record.zone_id
    if not val_dir.exists():
        return api_validation_error("验证集目录不存在")

    try:
        from argus.anomaly.baseline import BaselineManager
        from argus.anomaly.trainer import ModelTrainer

        bm = BaselineManager(baselines_dir=str(baselines_dir))
        trainer = ModelTrainer(baseline_manager=bm)

        # Find model files
        old_model = trainer._find_best_model_file(Path(old_record.model_path))
        new_model = trainer._find_best_model_file(Path(new_record.model_path))

        if not old_model or not new_model:
            return api_validation_error("无法找到模型文件")

        result = trainer.compare_models(old_model, new_model, val_dir)
        return api_success(result)
    except Exception as e:
        return api_internal_error(str(e))



# ── Background task functions ──


def _train_model_task(
    progress_callback, *, baselines_dir, models_dir, camera_id, model_type,
    export_format, quantization="fp16", database_url=None, anomaly_config=None,
    resume_from=None, skip_baseline_validation=False,
):
    """Train an anomaly detection model with full validation pipeline.

    Uses ModelTrainer with progress callback for UI updates.
    Saves TrainingRecord to database when available.
    """
    progress_callback(5, "正在验证基线数据...")

    try:
        from argus.anomaly.baseline import BaselineManager
        from argus.anomaly.trainer import ModelTrainer, TrainingStatus
    except ImportError:
        raise RuntimeError("Anomalib 未安装，无法训练模型")

    bm = BaselineManager(baselines_dir=baselines_dir)

    trainer = ModelTrainer(
        baseline_manager=bm,
        models_dir=models_dir,
        exports_dir=str(Path(models_dir).parent / "exports"),
    )

    # Train with progress callback
    fmt = export_format if export_format != "none" else None
    result = trainer.train(
        camera_id=camera_id,
        zone_id="default",
        model_type=model_type,
        export_format=fmt,
        quantization=quantization,
        progress_callback=progress_callback,
        anomaly_config=anomaly_config,
        resume_from=resume_from,
        skip_baseline_validation=skip_baseline_validation,
    )

    if result.status == TrainingStatus.FAILED:
        raise RuntimeError(result.error or "训练失败（未知原因）")

    # Save training record to database
    if database_url:
        try:
            from argus.storage.database import Database
            from argus.storage.model_registry import ModelRegistry
            from argus.storage.models import TrainingRecord

            db = Database(database_url=database_url)
            db.initialize()

            # Get baseline version
            baseline_dir = bm.get_baseline_dir(camera_id, "default")
            baseline_version = baseline_dir.name

            pre_val = result.pre_validation or {}
            output_val = result.output_validation or {}
            val_stats = result.val_stats or {}
            quality = result.quality_report

            # Phase 1: extract real-labeled metrics from ValidationReport when present
            vr = result.validation_report
            val_cm_json = None
            val_scores_json_str = None
            val_labels_json_str = None
            if vr is not None:
                import json as _json
                if getattr(vr, "real_confusion_matrix", None):
                    val_cm_json = _json.dumps(vr.real_confusion_matrix)
                # Phase 2: persist raw scores/labels for frontend threshold slider
                if getattr(vr, "real_scores", None) is not None:
                    val_scores_json_str = _json.dumps([round(float(s), 6) for s in vr.real_scores])
                if getattr(vr, "real_labels", None) is not None:
                    val_labels_json_str = _json.dumps([int(l) for l in vr.real_labels])

            record = TrainingRecord(
                camera_id=camera_id,
                zone_id="default",
                model_type=model_type,
                export_format=fmt,
                baseline_version=baseline_version,
                baseline_count=result.image_count,
                train_count=result.train_count,
                val_count=result.val_count,
                pre_validation_passed=pre_val.get("passed", True),
                corruption_rate=pre_val.get("corruption_rate"),
                near_duplicate_rate=pre_val.get("near_duplicate_rate"),
                brightness_std=pre_val.get("brightness_std"),
                val_score_mean=val_stats.get("mean"),
                val_score_std=val_stats.get("std"),
                val_score_max=val_stats.get("max"),
                val_score_p95=val_stats.get("p95"),
                quality_grade=quality.grade if quality else None,
                threshold_recommended=result.threshold_recommended,
                val_precision=getattr(vr, "real_precision", None) if vr else None,
                val_recall=getattr(vr, "real_recall", None) if vr else None,
                val_f1=getattr(vr, "real_f1", None) if vr else None,
                val_auroc=getattr(vr, "real_auroc", None) if vr else None,
                val_pr_auc=getattr(vr, "real_pr_auc", None) if vr else None,
                val_confusion_matrix=val_cm_json,
                val_real_sample_count=getattr(vr, "real_sample_count", None) if vr else None,
                val_scores_json=val_scores_json_str,
                val_labels_json=val_labels_json_str,
                model_path=result.model_path,
                export_path=str(Path(models_dir).parent / "exports" / camera_id / "default") if fmt else None,
                checkpoint_valid=output_val.get("checkpoint_valid"),
                export_valid=output_val.get("export_valid"),
                smoke_test_passed=output_val.get("smoke_test_passed"),
                inference_latency_ms=output_val.get("inference_latency_ms"),
                status=result.status.value,
                error=result.error,
                duration_seconds=result.duration_seconds,
                trained_at=datetime.now(timezone.utc),
            )
            db.save_training_record(record)

            if result.model_path and not result.model_version_id:
                registry = ModelRegistry(session_factory=db.get_session)
                result.model_version_id = registry.register(
                    model_path=result.model_path,
                    baseline_dir=baseline_dir,
                    camera_id=camera_id,
                    model_type=model_type,
                    training_params={
                        "export_format": fmt,
                        "quantization": quantization,
                        "train_count": result.train_count,
                        "val_count": result.val_count,
                        "quality_grade": result.quality_report.grade if result.quality_report else None,
                    },
                )
            db.close()
        except Exception as e:
            logger.warning("training.record_save_failed", error=str(e))

    grade_str = f" | 质量: {result.quality_report.grade}" if result.quality_report else ""
    threshold_str = f" | 推荐阈值: {result.threshold_recommended:.3f}" if result.threshold_recommended else ""
    progress_callback(100, f"训练完成 — 耗时 {result.duration_seconds:.0f} 秒{grade_str}{threshold_str}")

    return {
        "model_path": result.model_path,
        "model_version_id": result.model_version_id,
        "status": result.status.value,
        "duration": result.duration_seconds,
        "image_count": result.image_count,
        "train_count": result.train_count,
        "val_count": result.val_count,
        "quality_grade": result.quality_report.grade if result.quality_report else None,
        "threshold_recommended": result.threshold_recommended,
    }


def _get_baseline_manager(request: Request):
    """Get or create a BaselineManager from app state or config."""
    baseline_mgr = getattr(request.app.state, "baseline_manager", None)
    if baseline_mgr is not None:
        return baseline_mgr
    config = request.app.state.config
    if config:
        from argus.anomaly.baseline import BaselineManager
        return BaselineManager(baselines_dir=str(config.storage.baselines_dir))
    return None


@router.get("/optimize/preview")
async def optimize_preview(
    request: Request,
    camera_id: str = "",
    zone_id: str = "default",
    target_ratio: float = 0.2,
):
    """Preview how many images would be kept/moved by optimization."""
    if not camera_id:
        return api_validation_error("camera_id is required")

    baseline_mgr = _get_baseline_manager(request)
    if baseline_mgr is None:
        return api_unavailable("基线管理器不可用")

    baseline_dir = baseline_mgr.get_baseline_dir(camera_id, zone_id)
    all_images = sorted(
        list(baseline_dir.glob("*.png")) + list(baseline_dir.glob("*.jpg"))
    )
    total = len(all_images)
    if total == 0:
        return api_success({"total": 0, "keep": 0, "move": 0})

    target_count = max(30, int(total * target_ratio))
    keep = min(target_count, total)
    return api_success({"total": total, "keep": keep, "move": total - keep})


@router.post("/optimize/json")
async def optimize_baseline_json(request: Request):
    """JSON API: Optimize baseline by selecting most diverse subset (A4-2).

    Params (JSON body):
        camera_id: str (required)
        zone_id: str (default "default")
        target_ratio: float (default 0.2)

    Returns: {"selected": N, "moved": M, "backup_dir": "..."}
    """
    data = await request.json()
    camera_id = data.get("camera_id", "")
    zone_id = data.get("zone_id", "default")
    target_ratio = float(data.get("target_ratio", 0.2))

    if not camera_id:
        return api_validation_error("camera_id is required")

    baseline_mgr = _get_baseline_manager(request)
    if baseline_mgr is None:
        return api_unavailable("基线管理器不可用")

    baseline_dir = baseline_mgr.get_baseline_dir(camera_id, zone_id)
    all_images = sorted(
        list(baseline_dir.glob("*.png")) + list(baseline_dir.glob("*.jpg"))
    )
    if not all_images:
        return api_not_found("未找到基线图片")

    target_count = max(30, int(len(all_images) * target_ratio))
    selected = baseline_mgr.diversity_select(baseline_dir, target_count)
    selected_set = set(selected)

    # Move unselected to backup directory
    backup_dir = baseline_dir / "backup"
    backup_dir.mkdir(exist_ok=True)
    moved = 0
    for img_path in all_images:
        if img_path not in selected_set:
            shutil.move(str(img_path), str(backup_dir / img_path.name))
            moved += 1

    logger.info(
        "baseline.optimized",
        camera_id=camera_id,
        zone_id=zone_id,
        selected=len(selected),
        moved=moved,
    )

    # Audit trail
    audit = getattr(request.app.state, "audit_logger", None)
    if audit:
        client_ip = request.client.host if request.client else ""
        audit.log(
            user="operator",
            action="optimize_baseline",
            target_type="baseline",
            target_id=f"{camera_id}/{zone_id}",
            detail=f"保留 {len(selected)} 张, 移除 {moved} 张",
            ip_address=client_ip,
        )

    return api_success({
        "selected": len(selected),
        "moved": moved,
        "backup_dir": str(backup_dir),
    })


# ── Camera Group Endpoints ──


@router.get("/groups/json")
async def camera_groups_json(request: Request):
    """JSON API: list configured camera groups with baseline status."""
    config = request.app.state.config
    if not config:
        return api_success({"groups": []})

    groups = []
    baseline_mgr = _get_baseline_manager(request)
    for grp in getattr(config, "camera_groups", []):
        image_count = 0
        current_version = None
        if baseline_mgr:
            group_dir = baseline_mgr.get_group_baseline_dir(grp.group_id, grp.zone_id)
            if group_dir.is_dir():
                image_count = (
                    len(list(group_dir.glob("*.png")))
                    + len(list(group_dir.glob("*.jpg")))
                )
                current_version = group_dir.name
        groups.append({
            "group_id": grp.group_id,
            "name": grp.name,
            "camera_ids": grp.camera_ids,
            "zone_id": grp.zone_id,
            "image_count": image_count,
            "current_version": current_version,
        })
    return api_success({"groups": groups})


@router.post("/groups/merge")
async def merge_group_baseline(request: Request):
    """Merge baselines from member cameras into a group baseline version."""
    data = await request.json()
    group_id = data.get("group_id", "")
    zone_id = data.get("zone_id", "default")
    target_count = data.get("target_count")

    if not group_id:
        return api_validation_error("group_id is required")

    config = request.app.state.config
    if not config:
        return api_unavailable("配置不可用")

    group_cfg = next(
        (g for g in getattr(config, "camera_groups", []) if g.group_id == group_id),
        None,
    )
    if group_cfg is None:
        return api_not_found(f"分组 {group_id} 不存在")

    baseline_mgr = _get_baseline_manager(request)
    if baseline_mgr is None:
        return api_unavailable("基线管理器不可用")

    version_dir = baseline_mgr.merge_camera_baselines(
        group_id=group_id,
        camera_ids=group_cfg.camera_ids,
        zone_id=zone_id,
        target_count=int(target_count) if target_count else None,
    )

    image_count = (
        len(list(version_dir.glob("*.png")))
        + len(list(version_dir.glob("*.jpg")))
    )

    return api_success({
        "group_id": group_id,
        "version": version_dir.name,
        "image_count": image_count,
    })


# ── False Positive Merge Endpoint ──


@router.post("/merge-fp")
async def merge_false_positives(request: Request):
    """Merge false positive candidate pool into a new baseline version (Draft state)."""
    data = await request.json()
    camera_id = data.get("camera_id", "")
    zone_id = data.get("zone_id", "default")
    max_fp_images = data.get("max_fp_images")

    if not camera_id:
        return api_validation_error("camera_id is required")

    feedback_mgr = getattr(request.app.state, "feedback_manager", None)
    if feedback_mgr is None:
        return api_unavailable("反馈管理器未初始化")

    baseline_mgr = _get_baseline_manager(request)
    if baseline_mgr is None:
        return api_unavailable("基线管理器不可用")

    result = feedback_mgr.merge_fp_into_baseline(
        camera_id=camera_id,
        zone_id=zone_id,
        baseline_manager=baseline_mgr,
        max_fp_images=int(max_fp_images) if max_fp_images else None,
    )

    if "error" in result:
        return api_validation_error(result["error"])

    return api_success(result)


# ── Baseline Lifecycle Endpoints ──


def _get_lifecycle(request: Request):
    """Get BaselineLifecycle from app state."""
    return getattr(request.app.state, "baseline_lifecycle", None)


@router.get("/versions/json")
async def baseline_versions_json(request: Request):
    """JSON API: list baseline versions with lifecycle state."""
    camera_id = request.query_params.get("camera_id", "")
    zone_id = request.query_params.get("zone_id", "default")
    if not camera_id:
        return api_validation_error("camera_id is required")

    lifecycle = _get_lifecycle(request)
    if lifecycle is None:
        return api_unavailable("生命周期管理器未初始化")

    versions = lifecycle.get_versions(camera_id, zone_id)
    return api_success({"versions": [v.to_dict() for v in versions]})


@router.post("/verify")
async def verify_baseline(request: Request):
    """Verify a baseline version (Draft -> Verified)."""
    data = await request.json()
    camera_id = data.get("camera_id", "")
    zone_id = data.get("zone_id", "default")
    version = data.get("version", "")
    verified_by = data.get("verified_by", "")
    verified_by_secondary = data.get("verified_by_secondary")

    if not camera_id or not version or not verified_by:
        return api_validation_error("camera_id, version, verified_by are required")

    lifecycle = _get_lifecycle(request)
    if lifecycle is None:
        return api_unavailable("生命周期管理器未初始化")

    try:
        client_ip = request.client.host if request.client else ""
        record = lifecycle.verify(
            camera_id, zone_id, version,
            verified_by=verified_by,
            verified_by_secondary=verified_by_secondary,
            ip_address=client_ip,
        )
        return api_success({"version": record.to_dict()})
    except Exception as e:
        return api_validation_error(str(e))


@router.post("/activate-baseline")
async def activate_baseline(request: Request):
    """Activate a baseline version (Verified -> Active). Auto-retires previous."""
    data = await request.json()
    camera_id = data.get("camera_id", "")
    zone_id = data.get("zone_id", "default")
    version = data.get("version", "")
    user = data.get("user", "operator")

    if not camera_id or not version:
        return api_validation_error("camera_id and version are required")

    lifecycle = _get_lifecycle(request)
    if lifecycle is None:
        return api_unavailable("生命周期管理器未初始化")

    try:
        client_ip = request.client.host if request.client else ""
        record = lifecycle.activate(
            camera_id, zone_id, version, user=user, ip_address=client_ip,
        )
        return api_success({"version": record.to_dict()})
    except Exception as e:
        return api_validation_error(str(e))


@router.post("/retire")
async def retire_baseline(request: Request):
    """Retire a baseline version (Active -> Retired)."""
    data = await request.json()
    camera_id = data.get("camera_id", "")
    zone_id = data.get("zone_id", "default")
    version = data.get("version", "")
    user = data.get("user", "operator")
    reason = data.get("reason", "")

    if not camera_id or not version:
        return api_validation_error("camera_id and version are required")

    lifecycle = _get_lifecycle(request)
    if lifecycle is None:
        return api_unavailable("生命周期管理器未初始化")

    try:
        client_ip = request.client.host if request.client else ""
        record = lifecycle.retire(
            camera_id, zone_id, version,
            user=user, reason=reason, ip_address=client_ip,
        )
        return api_success({"version": record.to_dict()})
    except Exception as e:
        return api_validation_error(str(e))
