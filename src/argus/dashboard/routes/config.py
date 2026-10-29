# config.py

"""Configuration management API routes (Chinese UI with tabs)."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import structlog
from fastapi import APIRouter, Request
from pydantic import BaseModel, StrictBool

from argus.dashboard.api_response import (
    api_conflict,
    api_forbidden,
    api_success,
    api_internal_error,
    api_not_found,
    api_unavailable,
    api_validation_error,
)
from argus.dashboard.auth import current_username, require_permission
from argus.dashboard.forms import htmx_toast_headers, parse_request_form

logger = structlog.get_logger()

router = APIRouter()


def _deny_without_permission(request: Request, permission: str):
    if require_permission(request, permission):
        return None
    return api_forbidden("权限不足")


async def _parse_form_or_json(request: Request) -> dict:
    content_type = request.headers.get("content-type", "")
    mime_type = content_type.split(";", 1)[0].strip().lower()
    if mime_type == "application/json":
        try:
            payload = await request.json()
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}
    return dict(await parse_request_form(request))


def _has_value(data: dict, key: str) -> bool:
    return key in data and data.get(key) not in (None, "")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _runtime_component_summary(camera_manager) -> dict:
    """Read component readiness through CameraManager's public snapshot API."""
    defaults = {
        "total_pipelines": 0,
        "classifier_attached": 0,
        "classifier_loaded": 0,
        "classifier_ready": 0,
        "classifier_degraded": 0,
        "classifier_failed": 0,
        "classifier_details": [],
        "segmenter_attached": 0,
        "segmenter_loaded": 0,
        "segmenter_ready": 0,
        "segmenter_degraded": 0,
        "segmenter_failed": 0,
        "segmenter_details": [],
        "imaging_attached": 0,
        "correlator_present": False,
    }
    if camera_manager is None:
        return defaults
    getter = getattr(camera_manager, "get_runtime_component_summary", None)
    if not callable(getter):
        return defaults
    try:
        summary = getter()
    except Exception:
        logger.debug("config.runtime_component_summary_failed", exc_info=True)
        return defaults
    return {**defaults, **summary} if isinstance(summary, dict) else defaults


def _auxiliary_config_lock(request: Request) -> asyncio.Lock:
    lock = getattr(request.app.state, "module_toggle_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        request.app.state.module_toggle_lock = lock
    return lock


async def _serialized_auxiliary_update(request: Request, operation, payload):
    """Keep ownership until the worker finishes, even if the client leaves.

    Cancelling to_thread does not stop its thread. The independent owned task
    must retain the async lock through runtime application and persistence.
    """
    denied = _deny_without_permission(request, "edit_config")
    if denied:
        return denied
    pending = getattr(request.app.state, "auxiliary_config_tasks", None)
    if pending is None:
        pending = set()
        request.app.state.auxiliary_config_tasks = pending
    if len(pending) >= 32:
        return api_unavailable("辅助模块配置操作繁忙，请稍后重试")

    async def apply():
        async with _auxiliary_config_lock(request):
            return await operation(request, payload)

    task = asyncio.create_task(apply())
    pending.add(task)

    def finished(completed):
        pending.discard(completed)
        if not completed.cancelled() and completed.exception() is not None:
            logger.error("config.auxiliary_update_failed", error=str(completed.exception()))

    task.add_done_callback(finished)
    return await asyncio.shield(task)


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
    """Update detection parameters from JSON or form data."""
    denied = _deny_without_permission(request, "edit_config")
    if denied:
        return denied

    config = request.app.state.config
    camera_manager = request.app.state.camera_manager
    if not config or not camera_manager:
        return api_unavailable("不可用")

    form = await _parse_form_or_json(request)

    # Update severity thresholds
    try:
        st = config.alerts.severity_thresholds
        if _has_value(form, "sev_info"):
            st.info = float(form["sev_info"])
        if _has_value(form, "sev_low"):
            st.low = float(form["sev_low"])
        if _has_value(form, "sev_medium"):
            st.medium = float(form["sev_medium"])
        if _has_value(form, "sev_high"):
            st.high = float(form["sev_high"])
    except (ValueError, Exception) as e:
        return api_validation_error(f"阈值参数无效: {e}")

    # Update temporal
    temp = config.alerts.temporal
    if _has_value(form, "temp_gap"):
        temp.max_gap_seconds = float(form["temp_gap"])
    if _has_value(form, "temp_overlap"):
        temp.min_spatial_overlap = float(form["temp_overlap"])

    # Update suppression
    supp = config.alerts.suppression
    if _has_value(form, "supp_zone"):
        supp.same_zone_window_seconds = float(form["supp_zone"])

    # Determine which sections changed so we can tell the pipelines to
    # refresh their derived caches AND tell the UI which knobs took effect.
    severity_changed = any(
        _has_value(form, k)
        for k in ("sev_info", "sev_low", "sev_medium", "sev_high")
    )
    temporal_changed = any(_has_value(form, k) for k in ("temp_gap", "temp_overlap"))
    suppression_changed = _has_value(form, "supp_zone")
    anomaly_threshold = float(form["anomaly_threshold"]) if _has_value(form, "anomaly_threshold") else None

    # Apply to pipelines
    update_runtime = getattr(camera_manager, "update_pipeline_thresholds", None)
    if not callable(update_runtime):
        return api_unavailable("运行时阈值更新接口不可用")
    updated, section_hot = update_runtime(
        camera_ids=[cam.camera_id for cam in config.cameras],
        anomaly_threshold=anomaly_threshold,
        severity_changed=severity_changed,
        temporal_changed=temporal_changed,
        suppression_changed=suppression_changed,
    )

    logger.info("config.detection_params_updated", pipelines=updated)
    audit = getattr(request.app.state, "audit_logger", None)
    client_ip = request.client.host if request.client else ""
    if audit:
        audit.log(
            user=current_username(request),
            action="update_config",
            target_type="detection_params",
            detail=f"更新检测参数，影响 {updated} 条流水线",
            ip_address=client_ip,
        )

    # 痛点 9: per-section hot-reload status drives the SystemConfigPanel toast.
    def _summary(touched: bool, key: str) -> dict:
        if not touched:
            return {"changed": False, "hot_reloaded": False, "applied": 0, "total": updated}
        return {
            "changed": True,
            "hot_reloaded": section_hot.get(key, False),
            "applied": updated if section_hot.get(key, False) else 0,
            "total": updated,
        }

    return api_success(
        {
            "pipelines_updated": updated,
            "anomaly_threshold": _summary(anomaly_threshold is not None, "anomaly_threshold"),
            "severity": _summary(severity_changed, "severity"),
            "temporal": _summary(temporal_changed, "temporal"),
            "suppression": _summary(suppression_changed, "suppression"),
        },
        headers=htmx_toast_headers("检测参数已更新"),
    )


@router.post("/notifications")
async def update_notifications(request: Request):
    """Update webhook config (email surface was removed in 2026-05)."""
    denied = _deny_without_permission(request, "edit_config")
    if denied:
        return denied

    config = request.app.state.config
    if not config:
        return api_unavailable("不可用")

    form = await _parse_form_or_json(request)

    # Webhook settings
    webhook = config.alerts.webhook
    if "webhook_enabled" in form:
        webhook.enabled = _as_bool(form["webhook_enabled"])
    if "webhook_url" in form:
        webhook.url = str(form["webhook_url"])
    if "webhook_timeout" in form:
        webhook.timeout = float(form["webhook_timeout"])

    return api_success(
        headers=htmx_toast_headers("通知设置已更新"),
    )




@router.post("/test-webhook")
async def test_webhook(request: Request):
    """Send a test webhook."""
    denied = _deny_without_permission(request, "edit_config")
    if denied:
        return denied

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
    denied = _deny_without_permission(request, "edit_config")
    if denied:
        return denied

    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("不可用")

    updated = 0
    if req.anomaly_threshold is not None:
        update_runtime = getattr(camera_manager, "update_pipeline_thresholds", None)
        if not callable(update_runtime):
            return api_unavailable("运行时阈值更新接口不可用")
        updated, _ = update_runtime(anomaly_threshold=req.anomaly_threshold)

    return api_success({"pipelines_updated": updated})


@router.post("/reload-model")
async def reload_model(request: Request, req: ModelReloadRequest):
    """Reject the legacy path-based model hot-reload command.

    A path plus a camera id is not an authorized release identity and cannot
    participate in the desired/applied compare-and-swap transaction.  Model
    changes must enter through ``ModelReleaseService`` so the registry stage,
    artifact digest, SceneGuard profile and running detector change together.
    """
    denied = _deny_without_permission(request, "manage_models")
    if denied:
        return denied

    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("不可用")

    return api_conflict(
        "按路径直接热加载模型的入口已关闭，未修改 registry 或运行时。请先注册 "
        "artifact，再通过 POST /api/models/{version_id}/promote 提交带幂等键、"
        "release revision 与 SceneGuard 绑定的两阶段发布。",
        data={
            "camera_id": req.camera_id,
            "requested_model_path": req.model_path,
            "runtime_changed": False,
            "replacement": "/api/models/{version_id}/promote",
        },
    )


@router.post("/clear-lock/{camera_id}")
async def clear_anomaly_lock(request: Request, camera_id: str):
    """Clear the anomaly region lock for a camera."""
    denied = _deny_without_permission(request, "edit_config")
    if denied:
        return denied

    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("不可用")

    clear_lock = getattr(camera_manager, "clear_camera_anomaly_lock", None)
    if not callable(clear_lock) or clear_lock(camera_id) is not True:
        return api_not_found(f"摄像头 {camera_id} 不存在")
    audit = getattr(request.app.state, "audit_logger", None)
    client_ip = request.client.host if request.client else ""
    if audit:
        audit.log(
            user=current_username(request),
            action="clear_lock",
            target_type="camera",
            target_id=camera_id,
            ip_address=client_ip,
        )
    return api_success({"camera_id": camera_id, "lock_cleared": True})


@router.post("/camera/{camera_id}/restart")
async def restart_camera(request: Request, camera_id: str):
    return await _serialized_auxiliary_update(request, _restart_camera, camera_id)


async def _restart_camera(request: Request, camera_id: str):
    """Stop and restart a camera pipeline."""
    denied = _deny_without_permission(request, "edit_config")
    if denied:
        return denied

    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("不可用")

    restart = getattr(camera_manager, "restart_camera", None)
    if not callable(restart):
        return api_unavailable("安全重启接口不可用")
    try:
        success = await asyncio.to_thread(restart, camera_id)
    except Exception:
        logger.warning("config.camera_restart_failed", camera_id=camera_id, exc_info=True)
        return api_conflict("摄像头清理或重启失败，请核验状态后重试")
    if success is not True:
        return api_conflict("摄像头未完成重启，请检查启动或清理失败状态")
    return api_success(
        {"result": "ok" if success else "failed"},
        headers=htmx_toast_headers("摄像头已重启"),
    )


@router.post("/reload")
async def reload_config(request: Request):
    return await _serialized_auxiliary_update(request, _reload_config, None)


async def _reload_config(request: Request, _payload):
    """Reload configuration from YAML."""
    denied = _deny_without_permission(request, "edit_config")
    if denied:
        return denied

    config_path = getattr(request.app.state, "config_path", None)
    camera_manager = request.app.state.camera_manager
    if not config_path or not camera_manager:
        return api_unavailable("不可用")

    try:
        from argus.config.loader import load_config as _load_config
        new_config = await asyncio.to_thread(_load_config, config_path)
    except Exception as e:
        return api_validation_error(f"加载配置失败: {e}")

    current_config = getattr(request.app.state, "config", None)
    if current_config is None:
        return api_unavailable("当前配置不可用")
    # Training enablement, maintenance approval and pickle provenance are
    # startup-only authority. A generic engineer reload cannot replace them.
    current_training = getattr(current_config, "task_training", None)
    replacement_training = getattr(new_config, "task_training", None)
    if (current_training is None or replacement_training is None
            or current_training.model_dump() != replacement_training.model_dump()):
        return api_conflict("任务训练配置仅能在服务启动时生效，请由运维修改配置并重启服务")
    new_config.task_training = current_training
    # Auxiliary SDKs and their risk policy retain references to these objects.
    # Replacing the app configuration alone would leave stale runtime authority.
    for section in ("classifier", "segmenter", "physics"):
        current = getattr(current_config, section, None)
        replacement = getattr(new_config, section, None)
        if current is None or replacement is None or current.model_dump() != replacement.model_dump():
            return api_conflict("辅助识别配置变更不能通过配置重载生效，请使用专用接口或重启服务")
        setattr(new_config, section, current)

    apply_runtime_config = getattr(
        camera_manager,
        "apply_runtime_camera_config",
        None,
    )
    if not callable(apply_runtime_config):
        return api_unavailable("运行时配置更新接口不可用")
    updated = await asyncio.to_thread(lambda: sum(
        1 for cam_cfg in new_config.cameras if apply_runtime_config(cam_cfg) is True
    ))

    request.app.state.config = new_config

    return api_success(
        {"pipelines_updated": updated},
        headers=htmx_toast_headers("配置已重新加载"),
    )


@router.post("/save")
def save_config(request: Request):
    """Save current config to YAML file."""
    denied = _deny_without_permission(request, "edit_config")
    if denied:
        return denied

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
    denied = _deny_without_permission(request, "edit_config")
    if denied:
        return denied

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
        "cross_camera.enabled": _safe_get("cross_camera", "enabled"),
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
    runtime = _runtime_component_summary(camera_manager)

    return api_success({
        "enabled": cfg.enabled,
        "model_name": cfg.model_name,
        "min_anomaly_score_to_classify": cfg.min_anomaly_score_to_classify,
        "timeout_seconds": cfg.timeout_seconds,
        "min_confidence_to_affect_alert": cfg.min_confidence_to_affect_alert,
        "vocabulary": list(cfg.vocabulary),
        "high_risk_labels": list(cfg.high_risk_labels),
        "low_risk_labels": list(cfg.low_risk_labels),
        "suppress_labels": list(cfg.suppress_labels),
        "custom_vocabulary_path": cfg.custom_vocabulary_path,
        "runtime": {
            "total_pipelines": runtime["total_pipelines"],
            "pipelines_attached": runtime["classifier_attached"],
            "pipelines_loaded": runtime["classifier_loaded"],
            "pipelines_ready": runtime["classifier_ready"],
            "pipelines_degraded": runtime["classifier_degraded"],
            "pipelines_failed": runtime["classifier_failed"],
            "details": runtime["classifier_details"],
        },
    })


@router.get("/segmenter")
async def get_segmenter_config(request: Request):
    """Return the full segmenter config + runtime readiness.

    Attachment, target-model readiness and degraded/failure states are
    separate; an enabled desired configuration is not proof of a loaded SAM2.
    """
    config = request.app.state.config
    if not config:
        return api_unavailable("配置不可用")

    cfg = getattr(config, "segmenter", None)
    if cfg is None:
        return api_unavailable("分割器配置不可用")

    camera_manager = getattr(request.app.state, "camera_manager", None)
    runtime = _runtime_component_summary(camera_manager)

    return api_success({
        "enabled": cfg.enabled,
        "model_size": cfg.model_size,
        "max_points": cfg.max_points,
        "min_anomaly_score": cfg.min_anomaly_score,
        "min_mask_area_px": cfg.min_mask_area_px,
        "timeout_seconds": cfg.timeout_seconds,
        "runtime": {
            "total_pipelines": runtime["total_pipelines"],
            "pipelines_attached": runtime["segmenter_attached"],
            "pipelines_loaded": runtime["segmenter_loaded"],
            "pipelines_ready": runtime["segmenter_ready"],
            "pipelines_degraded": runtime["segmenter_degraded"],
            "pipelines_failed": runtime["segmenter_failed"],
            "details": runtime["segmenter_details"],
        },
    })


@router.get("/imaging")
async def get_imaging_config(request: Request):
    """Return the multi-modal imaging config + runtime status."""
    config = request.app.state.config
    if not config:
        return api_unavailable("配置不可用")

    cfg = getattr(config, "imaging", None)
    if cfg is None:
        return api_unavailable("成像配置不可用")

    # Check runtime status: how many pipelines have imaging processor
    camera_manager = getattr(request.app.state, "camera_manager", None)
    runtime = _runtime_component_summary(camera_manager)

    return api_success({
        "enabled": cfg.enabled,
        "mode": cfg.mode,
        "camera_sdk": cfg.camera_sdk,
        "nir_strobe_enabled": cfg.nir_strobe_enabled,
        "polarization_processing": cfg.polarization_processing,
        "fusion_channels": cfg.fusion_channels,
        "deglare_method": cfg.deglare_method,
        "dolp_threshold": cfg.dolp_threshold,
        "runtime": {
            "total_pipelines": runtime["total_pipelines"],
            "pipelines_with_imaging": runtime["imaging_attached"],
        },
    })


@router.get("/cross-camera")
async def get_cross_camera_config(request: Request):
    """Return the full cross-camera correlation config + runtime readiness.

    Mirror of the segmenter panel. Runtime counters tell the UI how many
    live pipelines were *constructed with* a cross-camera correlator
    (``attached``) — similar to segmenter, toggling ``enabled`` at runtime
    only affects NEW pipelines, so the UI should show a restart warning.
    """
    config = request.app.state.config
    if not config:
        return api_unavailable("配置不可用")

    cfg = getattr(config, "cross_camera", None)
    if cfg is None:
        return api_unavailable("跨相机配置不可用")

    camera_manager = getattr(request.app.state, "camera_manager", None)
    runtime = _runtime_component_summary(camera_manager)

    # Serialize the overlap pairs (homography is a list of 3 lists of 3 floats)
    pairs_payload = []
    for p in cfg.overlap_pairs:
        pairs_payload.append({
            "camera_a": p.camera_a,
            "camera_b": p.camera_b,
            "homography": [list(row) for row in p.homography],
        })

    return api_success({
        "enabled": cfg.enabled,
        "corroboration_threshold": cfg.corroboration_threshold,
        "max_age_seconds": cfg.max_age_seconds,
        "uncorroborated_severity_downgrade": cfg.uncorroborated_severity_downgrade,
        "overlap_pairs": pairs_payload,
        "runtime": {
            "total_pipelines": runtime["total_pipelines"],
            "correlator_present": runtime["correlator_present"],
        },
    })


class CrossCameraPairEntry(BaseModel):
    camera_a: str
    camera_b: str
    homography: list[list[float]]


class CrossCameraUpdateRequest(BaseModel):
    overlap_pairs: list[CrossCameraPairEntry] | None = None
    corroboration_threshold: float | None = None
    max_age_seconds: float | None = None
    uncorroborated_severity_downgrade: int | None = None


@router.put("/cross-camera/pairs")
async def update_cross_camera_config(
    request: Request, req: CrossCameraUpdateRequest,
):
    """Patch cross-camera correlation config at runtime.

    ``overlap_pairs`` replaces the full list when provided.  Scalar params
    (``corroboration_threshold``, ``max_age_seconds``,
    ``uncorroborated_severity_downgrade``) are individually optional.
    Pushes pair/threshold changes to the live correlator via
    ``camera_manager.update_cross_camera_pairs()``.
    """
    from argus.config.schema import CameraOverlapConfig
    from argus.core.correlation import CameraOverlapPair
    denied = _deny_without_permission(request, "edit_config")
    if denied:
        return denied

    config = request.app.state.config
    if not config:
        return api_unavailable("配置不可用")

    cfg = getattr(config, "cross_camera", None)
    if cfg is None:
        return api_unavailable("跨相机配置不可用")

    # ── Validate overlap_pairs ──
    if req.overlap_pairs is not None:
        seen: set[frozenset[str]] = set()
        for p in req.overlap_pairs:
            if p.camera_a == p.camera_b:
                return api_validation_error(
                    f"camera_a 和 camera_b 不能相同 ('{p.camera_a}')"
                )
            if len(p.homography) != 3 or any(
                len(row) != 3 for row in p.homography
            ):
                return api_validation_error(
                    f"homography 必须是 3×3 矩阵 ('{p.camera_a}' → '{p.camera_b}')"
                )
            key = frozenset({p.camera_a, p.camera_b})
            if key in seen:
                return api_validation_error(
                    f"重复的相机对 ('{p.camera_a}' ↔ '{p.camera_b}')"
                )
            seen.add(key)

    # ── Validate scalar params ──
    if req.corroboration_threshold is not None and not (
        0.1 <= req.corroboration_threshold <= 0.9
    ):
        return api_validation_error("corroboration_threshold 必须在 [0.1, 0.9] 范围内")
    if req.max_age_seconds is not None and not (
        1.0 <= req.max_age_seconds <= 30.0
    ):
        return api_validation_error("max_age_seconds 必须在 [1.0, 30.0] 范围内")
    if req.uncorroborated_severity_downgrade is not None and not (
        0 <= req.uncorroborated_severity_downgrade <= 2
    ):
        return api_validation_error("uncorroborated_severity_downgrade 必须在 [0, 2] 范围内")

    # ── Apply to in-memory config ──
    if req.overlap_pairs is not None:
        cfg.overlap_pairs = [
            CameraOverlapConfig(
                camera_a=p.camera_a,
                camera_b=p.camera_b,
                homography=p.homography,
            )
            for p in req.overlap_pairs
        ]
    if req.corroboration_threshold is not None:
        cfg.corroboration_threshold = req.corroboration_threshold
    if req.max_age_seconds is not None:
        cfg.max_age_seconds = req.max_age_seconds
    if req.uncorroborated_severity_downgrade is not None:
        cfg.uncorroborated_severity_downgrade = req.uncorroborated_severity_downgrade

    # ── Hot-push to correlator ──
    correlator_updated = False
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if camera_manager is not None and (
        req.overlap_pairs is not None or req.corroboration_threshold is not None
    ):
        try:
            pairs = [
                CameraOverlapPair(
                    camera_a=p.camera_a,
                    camera_b=p.camera_b,
                    homography=p.homography,
                )
                for p in cfg.overlap_pairs
            ]
            correlator_updated = camera_manager.update_cross_camera_pairs(
                pairs=pairs,
                corroboration_threshold=req.corroboration_threshold,
            )
        except Exception:
            logger.warning("config.cross_camera_pairs_push_failed", exc_info=True)

    logger.info(
        "config.cross_camera_updated",
        pairs=len(cfg.overlap_pairs),
        corroboration_threshold=cfg.corroboration_threshold,
        max_age_seconds=cfg.max_age_seconds,
        correlator_updated=correlator_updated,
    )

    return api_success({
        "overlap_pairs": [
            {"camera_a": p.camera_a, "camera_b": p.camera_b, "homography": p.homography}
            for p in cfg.overlap_pairs
        ],
        "corroboration_threshold": cfg.corroboration_threshold,
        "max_age_seconds": cfg.max_age_seconds,
        "uncorroborated_severity_downgrade": cfg.uncorroborated_severity_downgrade,
        "correlator_updated": correlator_updated,
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
    return await _serialized_auxiliary_update(request, _apply_segmenter_params, req)


async def _apply_segmenter_params(request: Request, req: SegmenterParamsRequest):
    """Patch the segmenter's runtime-tunable parameters.

    Four knobs accepted: ``max_points`` / ``min_anomaly_score`` (pipeline
    peak-extraction), ``min_mask_area_px`` / ``timeout_seconds`` (segmenter
    itself). Writes to the in-memory ``config.segmenter`` so subsequent
    reads of the "分割器" panel show the new values, and then fans the
    update out to every live pipeline via
    ``camera_manager.update_segmenter_params()``. ``enabled`` and
    ``model_size`` are not tunable here. Validation uses the same schema as
    persisted startup configuration, so an accepted update remains reloadable.
    """
    denied = _deny_without_permission(request, "edit_config")
    if denied:
        return denied

    config = request.app.state.config
    if not config:
        return api_unavailable("配置不可用")

    cfg = getattr(config, "segmenter", None)
    if cfg is None:
        return api_unavailable("分割器配置不可用")

    from argus.config.schema import SegmenterConfig

    updates = req.model_dump(exclude_none=True)
    try:
        candidate = SegmenterConfig.model_validate({**cfg.model_dump(), **updates})
    except ValueError as exc:
        return api_validation_error(str(exc))
    for field in updates:
        setattr(cfg, field, getattr(candidate, field))

    camera_manager = getattr(request.app.state, "camera_manager", None)
    pushed = 0
    if camera_manager is not None:
        try:
            pushed = await asyncio.to_thread(camera_manager.update_segmenter_params,
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
    return await _serialized_auxiliary_update(request, _apply_classifier_vocabulary, req)


async def _apply_classifier_vocabulary(request: Request, req: ClassifierVocabularyRequest):
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
    denied = _deny_without_permission(request, "edit_config")
    if denied:
        return denied

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

    active_high = set(high if high is not None else cfg.high_risk_labels) & seen
    active_low = set(low if low is not None else cfg.low_risk_labels) & seen
    active_suppress = set(suppress if suppress is not None else cfg.suppress_labels) & seen
    if active_high & (active_low | active_suppress) or active_low & active_suppress:
        return api_validation_error("同一有效标签不能同时属于高风险、低风险或抑制类别")

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
            pushed = await asyncio.to_thread(camera_manager.update_classifier_vocabulary, vocab_clean)
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
    value: StrictBool


@router.post("/modules")
async def update_module_toggle(request: Request, req: ModuleToggleRequest):
    return await _serialized_auxiliary_update(request, _apply_module_toggle, req)


async def _apply_module_toggle(request: Request, req: ModuleToggleRequest):
    """Toggle a module on/off at runtime.

    Accepts key paths like 'imaging.enabled', 'physics.speed_enabled',
    'continuous_recording.enabled', 'classifier.enabled'.
    Requires admin or engineer role.
    """
    denied = _deny_without_permission(request, "edit_config")
    if denied:
        return denied

    config = request.app.state.config
    if not config:
        return api_unavailable("配置不可用")

    allowed = {
        "imaging.enabled", "imaging.polarization_processing", "classifier.enabled",
        "segmenter.enabled", "cross_camera.enabled", "physics.speed_enabled",
        "physics.trajectory_enabled", "physics.localization_enabled",
        "physics.triangulation_enabled", "continuous_recording.enabled",
    }
    if req.key not in allowed:
        return api_validation_error(f"Unsupported module toggle: {req.key}")
    parts = req.key.split(".")
    if len(parts) != 2:
        return api_validation_error(f"Invalid key format: {req.key}")

    section, field = parts
    section_obj = getattr(config, section, None)
    if section_obj is None:
        return api_validation_error(f"Unknown config section: {section}")

    if not hasattr(section_obj, field):
        return api_validation_error(f"Unknown field: {field} in {section}")

    setattr(section_obj, field, req.value)

    # Try hot-reload into running pipelines
    _hot_reloadable_keys = {
        "classifier.enabled",
        "segmenter.enabled",
        "physics.speed_enabled",
    }
    restart_required = False
    hot_reloaded = 0
    hot_reload_failed = 0
    pipelines_seen = 0
    runtime_checked = False

    if req.key in _hot_reloadable_keys:
        camera_manager = getattr(request.app.state, "camera_manager", None)
        if camera_manager:
            reload_modules = getattr(
                camera_manager,
                "reload_module_for_cameras",
                None,
            )
            if callable(reload_modules):
                try:
                    result = await asyncio.to_thread(
                        reload_modules,
                        [cam.camera_id for cam in getattr(config, "cameras", [])],
                        req.key, req.value,
                    )
                    if isinstance(result, dict):
                        pipelines_seen = int(result.get("seen", 0))
                        hot_reloaded = int(result.get("succeeded", 0))
                        hot_reload_failed = int(result.get("failed", 0))
                        runtime_checked = True
                except Exception:
                    hot_reload_failed = 1
                    logger.warning("config.module_hot_reload_failed", exc_info=True)
        restart_required = not runtime_checked or hot_reload_failed > 0 or hot_reloaded < pipelines_seen
    else:
        # These still need restart (no hot-reload path yet)
        _restart_required_keys = {
            "cross_camera.enabled",
            "imaging.enabled",
            "imaging.polarization_processing",
            "physics.trajectory_enabled",
            "physics.localization_enabled",
            "physics.triangulation_enabled",
            "continuous_recording.enabled",
        }
        restart_required = req.key in _restart_required_keys

    # Auto-persist to YAML so toggle survives restart
    persisted = False
    config_path = getattr(request.app.state, "config_path", None)
    if config_path:
        try:
            from argus.config.loader import save_config as _save
            _save(config, config_path)
            persisted = True
        except Exception as e:
            logger.warning("config.auto_persist_failed", error=str(e))

    logger.info(
        "config.module_toggled",
        key=req.key,
        value=req.value,
        restart_required=restart_required,
        hot_reloaded=hot_reloaded,
        hot_reload_failed=hot_reload_failed,
        pipelines_seen=pipelines_seen,
        persisted=persisted,
    )
    return api_success({
        "key": req.key,
        "value": req.value,
        "restart_required": restart_required,
        "hot_reloaded": hot_reloaded,
        "hot_reloadable": req.key in _hot_reloadable_keys,
        "hot_reload_failed": hot_reload_failed,
        "pipelines_seen": pipelines_seen,
        "persisted": persisted,
        "all_applied": runtime_checked and hot_reload_failed == 0 and hot_reloaded == pipelines_seen,
    })
