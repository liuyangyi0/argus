# config.py

"""Configuration management API routes (Chinese UI with tabs)."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from argus.core.model_discovery import resolve_runtime_model_path
from argus.dashboard.api_response import (
    api_forbidden,
    api_success,
    api_internal_error,
    api_not_found,
    api_unavailable,
    api_validation_error,
)
from argus.dashboard.forms import htmx_toast_headers, parse_request_form
from argus.dashboard.model_runtime import find_registered_model_by_path

logger = structlog.get_logger()

router = APIRouter()


class ThresholdUpdateRequest(BaseModel):
    anomaly_threshold: float | None = None
    info_threshold: float | None = None
    low_threshold: float | None = None
    medium_threshold: float | None = None
    high_threshold: float | None = None


class ModelReloadRequest(BaseModel):
    camera_id: str
    model_path: str


@router.post("/detection-params")
async def update_detection_params(request: Request):
    """Update detection parameters from form."""
    config = request.app.state.config
    camera_manager = request.app.state.camera_manager
    if not config or not camera_manager:
        return api_unavailable("不可用")

    form = await parse_request_form(request)

    # Update severity thresholds
    try:
        st = config.alerts.severity_thresholds
        if form.get("sev_info"):
            st.info = float(form["sev_info"])
        if form.get("sev_low"):
            st.low = float(form["sev_low"])
        if form.get("sev_medium"):
            st.medium = float(form["sev_medium"])
        if form.get("sev_high"):
            st.high = float(form["sev_high"])
    except (ValueError, Exception) as e:
        return api_validation_error(f"阈值参数无效: {e}")

    # Update temporal
    temp = config.alerts.temporal
    if form.get("temp_frames"):
        temp.min_consecutive_frames = int(form["temp_frames"])
    if form.get("temp_gap"):
        temp.max_gap_seconds = float(form["temp_gap"])
    if form.get("temp_overlap"):
        temp.min_spatial_overlap = float(form["temp_overlap"])

    # Update suppression
    supp = config.alerts.suppression
    if form.get("supp_zone"):
        supp.same_zone_window_seconds = float(form["supp_zone"])

    # Apply to pipelines
    updated = 0
    for cam_cfg in config.cameras:
        pipeline = camera_manager._pipelines.get(cam_cfg.camera_id)
        if not pipeline:
            continue
        if form.get("anomaly_threshold"):
            pipeline.update_thresholds(anomaly_threshold=float(form["anomaly_threshold"]))
        updated += 1

    logger.info("config.detection_params_updated", pipelines=updated)
    audit = getattr(request.app.state, "audit_logger", None)
    client_ip = request.client.host if request.client else ""
    if audit:
        audit.log(
            user="operator",
            action="update_config",
            target_type="detection_params",
            detail=f"更新检测参数，影响 {updated} 条流水线",
            ip_address=client_ip,
        )
    return api_success(
        {"pipelines_updated": updated},
        headers=htmx_toast_headers("检测参数已更新"),
    )


@router.post("/notifications")
async def update_notifications(request: Request):
    """Update email/webhook config."""
    config = request.app.state.config
    if not config:
        return api_unavailable("不可用")

    form = await parse_request_form(request)

    # Webhook settings
    webhook = config.alerts.webhook
    if "webhook_enabled" in form:
        webhook.enabled = form["webhook_enabled"] == "true"
    if "webhook_url" in form:
        webhook.url = form["webhook_url"]
    if "webhook_timeout" in form:
        webhook.timeout = float(form["webhook_timeout"])

    return api_success(
        headers=htmx_toast_headers("通知设置已更新"),
    )




@router.post("/test-webhook")
async def test_webhook(request: Request):
    """Send a test webhook."""
    config = request.app.state.config
    if not config:
        return api_unavailable("不可用")

    webhook = config.alerts.webhook
    if not webhook.url:
        return api_validation_error("请先配置 Webhook URL")

    def _send():
        import httpx
        payload = {
            "alert_id": "TEST-000",
            "type": "test",
            "message": "Argus 系统测试消息",
            "severity": "info",
        }
        with httpx.Client(timeout=webhook.timeout) as client:
            resp = client.post(webhook.url, json=payload)
            resp.raise_for_status()

    try:
        await asyncio.to_thread(_send)
        return api_success(
            headers=htmx_toast_headers("Webhook 测试成功"),
        )
    except Exception as e:
        return api_internal_error(str(e))


@router.get("/storage/info")
def storage_info_json(request: Request):
    """JSON endpoint for storage usage and retention info."""
    config = request.app.state.config
    db = request.app.state.db

    result: dict = {"retention_days": 90, "alert_count": 0}
    if config:
        result["retention_days"] = config.storage.alert_retention_days

    if db:
        try:
            result["alert_count"] = db.get_alert_count()
        except Exception:
            logger.debug("config.alert_count_query_failed", exc_info=True)

    try:
        usage = shutil.disk_usage("data")
        result["disk"] = {
            "total_gb": round(usage.total / (1024**3), 1),
            "used_gb": round(usage.used / (1024**3), 1),
            "free_gb": round(usage.free / (1024**3), 1),
            "percent_used": round(usage.used / usage.total * 100, 1),
        }
    except OSError:
        result["disk"] = None

    return api_success(result)


@router.post("/cleanup")
def cleanup_data(request: Request):
    """Cleanup old alert data. Requires admin role."""
    from argus.dashboard.auth import require_role
    if not require_role(request, "admin"):
        return api_forbidden("需要管理员权限")

    db = request.app.state.db
    config = request.app.state.config
    if not db:
        return api_unavailable("数据库不可用")

    days = config.storage.alert_retention_days if config else 90
    deleted, paths = db.delete_old_alerts(days=days)

    # Delete image files
    removed_files = 0
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
            removed_files += 1
        except Exception:
            logger.debug("config.alert_file_cleanup_failed", path=str(p), exc_info=True)

    return api_success(
        {"deleted": deleted, "files": removed_files},
        headers=htmx_toast_headers(f"已清理 {deleted} 条旧告警"),
    )


@router.post("/thresholds")
async def update_thresholds(request: Request, req: ThresholdUpdateRequest):
    """Hot-update detection thresholds without restart."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("不可用")

    updated = 0
    for pipeline in camera_manager._pipelines.values():
        if req.anomaly_threshold is not None:
            pipeline.update_thresholds(anomaly_threshold=req.anomaly_threshold)
            updated += 1

    return api_success({"pipelines_updated": updated})


@router.post("/reload-model")
async def reload_model(request: Request, req: ModelReloadRequest):
    """Trigger model hot-reload for a specific camera."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("不可用")

    if req.camera_id not in camera_manager._pipelines:
        return api_not_found(f"摄像头 {req.camera_id} 不存在")

    # Sanitize model path
    allowed_roots = [Path("data/models").resolve(), Path("data/exports").resolve()]
    try:
        model_path = Path(req.model_path).resolve()
        if not any(str(model_path).startswith(str(root)) for root in allowed_roots):
            return api_validation_error("模型路径必须在 data/models/ 或 data/exports/ 下")
        if not model_path.exists():
            return api_not_found(f"模型文件不存在: {req.model_path}")
    except (ValueError, OSError):
        return api_validation_error("模型路径无效")

    resolved_runtime_model = resolve_runtime_model_path(model_path, req.camera_id)
    if resolved_runtime_model is None:
        return api_not_found("未找到可部署的模型文件 (.xml/.pt)")

    registry_record = find_registered_model_by_path(
        request,
        model_path,
        camera_id=req.camera_id,
    )
    version_tag = registry_record.model_version_id if registry_record is not None else None
    success = camera_manager.reload_model(
        req.camera_id,
        str(resolved_runtime_model),
        version_tag=version_tag,
    )
    return api_success({"result": "ok" if success else "failed", "camera_id": req.camera_id})


@router.post("/clear-lock/{camera_id}")
async def clear_anomaly_lock(request: Request, camera_id: str):
    """Clear the anomaly region lock for a camera."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("不可用")

    pipeline = camera_manager._pipelines.get(camera_id)
    if not pipeline:
        return api_not_found(f"摄像头 {camera_id} 不存在")

    pipeline.clear_anomaly_lock()
    audit = getattr(request.app.state, "audit_logger", None)
    client_ip = request.client.host if request.client else ""
    if audit:
        audit.log(
            user="operator",
            action="clear_lock",
            target_type="camera",
            target_id=camera_id,
            ip_address=client_ip,
        )
    return HTMLResponse(
        '<span style="color:#4caf50;">锁定已解除</span>',
        headers=htmx_toast_headers("异常锁定已解除"),
    )


@router.post("/camera/{camera_id}/restart")
async def restart_camera(request: Request, camera_id: str):
    """Stop and restart a camera pipeline."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("不可用")

    await asyncio.to_thread(camera_manager.stop_camera, camera_id)
    success = await asyncio.to_thread(camera_manager.start_camera, camera_id)
    return api_success(
        {"result": "ok" if success else "failed"},
        headers=htmx_toast_headers("摄像头已重启"),
    )


@router.post("/reload")
def reload_config(request: Request):
    """Reload configuration from YAML."""
    config_path = getattr(request.app.state, "config_path", None)
    camera_manager = request.app.state.camera_manager
    if not config_path or not camera_manager:
        return api_unavailable("不可用")

    try:
        from argus.config.loader import load_config as _load_config
        new_config = _load_config(config_path)
    except Exception as e:
        return api_validation_error(f"加载配置失败: {e}")

    updated = 0
    for cam_cfg in new_config.cameras:
        pipeline = camera_manager._pipelines.get(cam_cfg.camera_id)
        if not pipeline:
            continue
        pipeline.update_thresholds(anomaly_threshold=cam_cfg.anomaly.threshold)
        if cam_cfg.zones:
            pipeline.update_zones(cam_cfg.zones)
        updated += 1

    request.app.state.config = new_config

    return api_success(
        {"pipelines_updated": updated},
        headers=htmx_toast_headers("配置已重新加载"),
    )


@router.post("/save")
def save_config(request: Request):
    """Save current config to YAML file."""
    config = request.app.state.config
    config_path = getattr(request.app.state, "config_path", None)
    if not config or not config_path:
        return api_unavailable("不可用")

    try:
        from argus.config.loader import save_config as _save
        _save(config, config_path)
    except Exception as e:
        return api_internal_error(f"保存失败: {e}")

    return api_success(
        headers=htmx_toast_headers("配置已保存到文件"),
    )


# ── UX v2 §2.5: Audio alert configuration ──

@router.get("/audio-alerts")
async def get_audio_alerts(request: Request):
    """Return current audio alert settings per severity."""
    config = request.app.state.config
    if not config:
        return api_unavailable("配置不可用")
    dashboard_cfg = getattr(config, "dashboard", None)
    if dashboard_cfg is None:
        from argus.config.schema import DashboardConfig
        dashboard_cfg = DashboardConfig()
    audio_cfg = getattr(dashboard_cfg, "audio_alerts", None)
    if audio_cfg is None:
        from argus.config.schema import AudioAlertConfig
        audio_cfg = AudioAlertConfig()
    return api_success(audio_cfg.model_dump())


@router.put("/audio-alerts")
async def update_audio_alerts(request: Request):
    """Update audio alert settings.

    Request body: {low: {enabled, sound, voice_template},
                   medium: {...}, high: {...}}
    """
    config = request.app.state.config
    if not config:
        return api_unavailable("配置不可用")

    data = await request.json()

    from argus.config.schema import AudioAlertConfig
    try:
        new_audio_cfg = AudioAlertConfig(**data)
    except Exception as e:
        return api_validation_error(f"无效配置: {e}")

    dashboard_cfg = getattr(config, "dashboard", None)
    if dashboard_cfg is not None:
        dashboard_cfg.audio_alerts = new_audio_cfg

    return api_success({"audio_alerts": new_audio_cfg.model_dump()})


@router.get("/modules")
async def get_module_states(request: Request):
    """Return current state of all toggleable modules."""
    config = request.app.state.config
    if not config:
        return api_success({})

    def _safe_get(section: str, field: str) -> bool:
        obj = getattr(config, section, None)
        if obj is None:
            return False
        return bool(getattr(obj, field, False))

    return api_success({
        "imaging.enabled": _safe_get("imaging", "enabled"),
        "imaging.polarization_processing": _safe_get("imaging", "polarization_processing"),
        "classifier.enabled": _safe_get("classifier", "enabled"),
        "segmenter.enabled": _safe_get("segmenter", "enabled"),
        "physics.speed_enabled": _safe_get("physics", "speed_enabled"),
        "physics.trajectory_enabled": _safe_get("physics", "trajectory_enabled"),
        "physics.localization_enabled": _safe_get("physics", "localization_enabled"),
        "physics.triangulation_enabled": _safe_get("physics", "triangulation_enabled"),
        "continuous_recording.enabled": _safe_get("continuous_recording", "enabled"),
    })


@router.get("/classifier")
async def get_classifier_config(request: Request):
    """Return the full classifier config + runtime readiness.

    The System "分类器" panel uses this to show operators what the
    classifier is configured to detect before they flip the enable
    switch. Runtime state is derived by introspecting live pipelines:
    ``total_pipelines`` is how many camera pipelines exist, and
    ``pipelines_loaded`` is how many of them have the classifier
    already warmed up. When the toggle is off, both should be zero
    (except the total, which reflects how many cameras are running).
    """
    config = request.app.state.config
    if not config:
        return api_unavailable("配置不可用")

    cfg = getattr(config, "classifier", None)
    if cfg is None:
        return api_unavailable("分类器配置不可用")

    camera_manager = getattr(request.app.state, "camera_manager", None)
    total_pipelines = 0
    pipelines_loaded = 0
    pipelines_attached = 0
    if camera_manager is not None:
        try:
            for cam_status in camera_manager.get_status():
                pipeline = camera_manager.get_pipeline(cam_status.camera_id)
                if pipeline is None:
                    continue
                total_pipelines += 1
                classifier = getattr(pipeline, "_classifier", None)
                if classifier is None:
                    continue
                pipelines_attached += 1
                if getattr(classifier, "_loaded", False):
                    pipelines_loaded += 1
        except Exception:
            logger.debug("classifier.pipeline_status_failed", exc_info=True)

    return api_success({
        "enabled": cfg.enabled,
        "model_name": cfg.model_name,
        "min_anomaly_score_to_classify": cfg.min_anomaly_score_to_classify,
        "vocabulary": list(cfg.vocabulary),
        "high_risk_labels": list(cfg.high_risk_labels),
        "low_risk_labels": list(cfg.low_risk_labels),
        "suppress_labels": list(cfg.suppress_labels),
        "custom_vocabulary_path": cfg.custom_vocabulary_path,
        "runtime": {
            "total_pipelines": total_pipelines,
            "pipelines_attached": pipelines_attached,
            "pipelines_loaded": pipelines_loaded,
        },
    })


@router.get("/segmenter")
async def get_segmenter_config(request: Request):
    """Return the full segmenter config + runtime readiness.

    Mirror of the classifier panel but for SAM2 instance segmentation.
    ``runtime.attached/loaded`` counters tell the UI how many live
    pipelines already have the segmenter warmed up — zero is expected
    whenever the toggle is off (pipelines are only *created* with a
    segmenter when the config had it enabled at startup, which means
    toggling enabled ON here does NOT retroactively attach segmenters
    to running pipelines; that's a "needs restart" warning we show in
    the UI).
    """
    config = request.app.state.config
    if not config:
        return api_unavailable("配置不可用")

    cfg = getattr(config, "segmenter", None)
    if cfg is None:
        return api_unavailable("分割器配置不可用")

    camera_manager = getattr(request.app.state, "camera_manager", None)
    total_pipelines = 0
    pipelines_attached = 0
    pipelines_loaded = 0
    if camera_manager is not None:
        try:
            for cam_status in camera_manager.get_status():
                pipeline = camera_manager.get_pipeline(cam_status.camera_id)
                if pipeline is None:
                    continue
                total_pipelines += 1
                segmenter = getattr(pipeline, "_segmenter", None)
                if segmenter is None:
                    continue
                pipelines_attached += 1
                if getattr(segmenter, "_loaded", False):
                    pipelines_loaded += 1
        except Exception:
            logger.debug("segmenter.pipeline_status_failed", exc_info=True)

    return api_success({
        "enabled": cfg.enabled,
        "model_size": cfg.model_size,
        "max_points": cfg.max_points,
        "min_anomaly_score": cfg.min_anomaly_score,
        "min_mask_area_px": cfg.min_mask_area_px,
        "timeout_seconds": cfg.timeout_seconds,
        "runtime": {
            "total_pipelines": total_pipelines,
            "pipelines_attached": pipelines_attached,
            "pipelines_loaded": pipelines_loaded,
        },
    })


class SegmenterParamsRequest(BaseModel):
    max_points: int | None = None
    min_anomaly_score: float | None = None
    min_mask_area_px: int | None = None
    timeout_seconds: float | None = None


@router.put("/segmenter/params")
async def update_segmenter_params_route(
    request: Request, req: SegmenterParamsRequest,
):
    """Patch the segmenter's runtime-tunable parameters.

    Four knobs accepted: ``max_points`` / ``min_anomaly_score`` (pipeline
    peak-extraction), ``min_mask_area_px`` / ``timeout_seconds`` (segmenter
    itself). Writes to the in-memory ``config.segmenter`` so subsequent
    reads of the "分割器" panel show the new values, and then fans the
    update out to every live pipeline via
    ``camera_manager.update_segmenter_params()``. ``enabled`` and
    ``model_size`` are **not** tunable here — they require a pipeline
    restart.

    Validation: max_points in [1, 32], min_anomaly_score in [0, 1],
    min_mask_area_px ≥ 0, timeout_seconds in (0, 120].
    """
    from argus.dashboard.auth import require_role
    if not require_role(request, "admin", "engineer"):
        return api_forbidden("需要管理员或工程师权限")

    config = request.app.state.config
    if not config:
        return api_unavailable("配置不可用")

    cfg = getattr(config, "segmenter", None)
    if cfg is None:
        return api_unavailable("分割器配置不可用")

    # Validate — all four are optional, only validate the ones provided.
    if req.max_points is not None and not (1 <= req.max_points <= 32):
        return api_validation_error("max_points 必须在 [1, 32] 范围内")
    if req.min_anomaly_score is not None and not (0.0 <= req.min_anomaly_score <= 1.0):
        return api_validation_error("min_anomaly_score 必须在 [0, 1] 范围内")
    if req.min_mask_area_px is not None and req.min_mask_area_px < 0:
        return api_validation_error("min_mask_area_px 不能为负数")
    if req.timeout_seconds is not None and not (0.0 < req.timeout_seconds <= 120.0):
        return api_validation_error("timeout_seconds 必须在 (0, 120] 范围内")

    if req.max_points is not None:
        cfg.max_points = req.max_points
    if req.min_anomaly_score is not None:
        cfg.min_anomaly_score = req.min_anomaly_score
    if req.min_mask_area_px is not None:
        cfg.min_mask_area_px = req.min_mask_area_px
    if req.timeout_seconds is not None:
        cfg.timeout_seconds = req.timeout_seconds

    camera_manager = getattr(request.app.state, "camera_manager", None)
    pushed = 0
    if camera_manager is not None:
        try:
            pushed = camera_manager.update_segmenter_params(
                max_points=req.max_points,
                min_anomaly_score=req.min_anomaly_score,
                min_mask_area_px=req.min_mask_area_px,
                timeout_seconds=req.timeout_seconds,
            )
        except Exception:
            logger.warning("config.segmenter_params_push_failed", exc_info=True)

    logger.info(
        "config.segmenter_params_updated",
        max_points=cfg.max_points,
        min_anomaly_score=cfg.min_anomaly_score,
        min_mask_area_px=cfg.min_mask_area_px,
        timeout_seconds=cfg.timeout_seconds,
        pipelines_updated=pushed,
    )

    return api_success({
        "max_points": cfg.max_points,
        "min_anomaly_score": cfg.min_anomaly_score,
        "min_mask_area_px": cfg.min_mask_area_px,
        "timeout_seconds": cfg.timeout_seconds,
        "pipelines_updated": pushed,
    })


class ClassifierVocabularyRequest(BaseModel):
    vocabulary: list[str]
    high_risk_labels: list[str] | None = None
    low_risk_labels: list[str] | None = None
    suppress_labels: list[str] | None = None


@router.put("/classifier/vocabulary")
async def update_classifier_vocabulary(
    request: Request, req: ClassifierVocabularyRequest,
):
    """Replace the classifier vocabulary + risk-bucket assignments.

    Writes to the in-memory ``config.classifier`` object and pushes the new
    vocabulary to every live pipeline's classifier via
    ``camera_manager.update_classifier_vocabulary()`` — so operators can
    iterate on the FOE list without restarting Argus. Changes are NOT
    persisted to YAML; ``POST /api/config/save`` is a separate step.

    Validation rules: vocabulary must be non-empty, each label a non-empty
    string (trimmed). high/low/suppress lists, if provided, must be subsets
    of the vocabulary — otherwise operators risk labelling things that the
    detector will never emit.
    """
    from argus.dashboard.auth import require_role
    if not require_role(request, "admin", "engineer"):
        return api_forbidden("需要管理员或工程师权限")

    config = request.app.state.config
    if not config:
        return api_unavailable("配置不可用")

    cfg = getattr(config, "classifier", None)
    if cfg is None:
        return api_unavailable("分类器配置不可用")

    vocab_clean: list[str] = []
    seen: set[str] = set()
    for raw in req.vocabulary:
        if not isinstance(raw, str):
            return api_validation_error("vocabulary 必须是字符串列表")
        label = raw.strip()
        if not label:
            continue
        if label in seen:
            continue
        seen.add(label)
        vocab_clean.append(label)
    if not vocab_clean:
        return api_validation_error("词表不能为空")

    def _validate_subset(labels: list[str] | None, field_name: str) -> list[str] | None:
        if labels is None:
            return None
        result: list[str] = []
        for raw in labels:
            if not isinstance(raw, str):
                raise ValueError(f"{field_name} 必须是字符串列表")
            label = raw.strip()
            if not label:
                continue
            if label not in seen:
                raise ValueError(f"{field_name} 包含词表之外的标签: {label}")
            result.append(label)
        return result

    try:
        high = _validate_subset(req.high_risk_labels, "high_risk_labels")
        low = _validate_subset(req.low_risk_labels, "low_risk_labels")
        suppress = _validate_subset(req.suppress_labels, "suppress_labels")
    except ValueError as e:
        return api_validation_error(str(e))

    cfg.vocabulary = vocab_clean
    if high is not None:
        cfg.high_risk_labels = high
    if low is not None:
        cfg.low_risk_labels = low
    if suppress is not None:
        cfg.suppress_labels = suppress

    camera_manager = getattr(request.app.state, "camera_manager", None)
    pushed = 0
    if camera_manager is not None:
        try:
            pushed = camera_manager.update_classifier_vocabulary(vocab_clean)
        except Exception:
            logger.warning("config.classifier_vocab_push_failed", exc_info=True)

    logger.info(
        "config.classifier_vocabulary_updated",
        vocab_size=len(vocab_clean),
        high_count=len(high) if high is not None else None,
        low_count=len(low) if low is not None else None,
        suppress_count=len(suppress) if suppress is not None else None,
        pipelines_updated=pushed,
    )

    return api_success({
        "vocabulary": vocab_clean,
        "high_risk_labels": list(cfg.high_risk_labels),
        "low_risk_labels": list(cfg.low_risk_labels),
        "suppress_labels": list(cfg.suppress_labels),
        "pipelines_updated": pushed,
    })


class ModuleToggleRequest(BaseModel):
    key: str
    value: bool


@router.post("/modules")
async def update_module_toggle(request: Request, req: ModuleToggleRequest):
    """Toggle a module on/off at runtime.

    Accepts key paths like 'imaging.enabled', 'physics.speed_enabled',
    'continuous_recording.enabled', 'classifier.enabled'.
    Requires admin or engineer role.
    """
    from argus.dashboard.auth import require_role
    if not require_role(request, "admin", "engineer"):
        from argus.dashboard.api_response import api_forbidden
        return api_forbidden("需要管理员或工程师权限")

    config = request.app.state.config
    parts = req.key.split(".")
    if len(parts) != 2:
        from argus.dashboard.api_response import api_validation_error
        return api_validation_error(f"Invalid key format: {req.key}")

    section, field = parts
    section_obj = getattr(config, section, None)
    if section_obj is None:
        from argus.dashboard.api_response import api_validation_error
        return api_validation_error(f"Unknown config section: {section}")

    if not hasattr(section_obj, field):
        from argus.dashboard.api_response import api_validation_error
        return api_validation_error(f"Unknown field: {field} in {section}")

    setattr(section_obj, field, req.value)
    logger.info("config.module_toggled", key=req.key, value=req.value)
    return api_success({"key": req.key, "value": req.value})
