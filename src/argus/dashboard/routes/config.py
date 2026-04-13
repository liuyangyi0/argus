# config.py

"""Configuration management API routes (Chinese UI with tabs)."""

from __future__ import annotations

import asyncio
import html
import shutil
from pathlib import Path

import structlog
import yaml
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from argus.core.model_discovery import resolve_runtime_model_path
from argus.dashboard.components import (
    confirm_button,
    empty_state,
    form_group,
    form_select,
    page_header,
    tab_bar,
)
from argus.dashboard.api_response import (
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

_TABS = [
    ("detection", "检测参数", "/api/config/detection"),
    ("notifications", "通知设置", "/api/config/notifications"),
    ("storage", "存储与维护", "/api/config/storage"),
    ("logs", "日志查看", "/api/config/logs"),
    ("cameras", "摄像头控制", "/api/config/cameras-tab"),
]


class ThresholdUpdateRequest(BaseModel):
    anomaly_threshold: float | None = None
    info_threshold: float | None = None
    low_threshold: float | None = None
    medium_threshold: float | None = None
    high_threshold: float | None = None


class ModelReloadRequest(BaseModel):
    camera_id: str
    model_path: str


@router.get("", response_class=HTMLResponse)
async def config_page(request: Request):
    """Configuration page with tabbed interface."""
    # When loaded as HTMX fragment (e.g. inside system page tab), skip page header
    is_htmx = request.headers.get("HX-Request") == "true"
    header = "" if is_htmx else page_header(
        "系统设置",
        "调整检测参数、通知设置和系统维护",
        '<button class="btn btn-primary" hx-post="/api/config/save" hx-swap="none">保存到配置文件</button>',
    )
    tabs = tab_bar(_TABS, "detection")
    return HTMLResponse(f"""
    {header}
    {tabs}
    <div hx-get="/api/config/detection" hx-trigger="load" hx-target="#tab-content" hx-swap="innerHTML"></div>""")


@router.get("/detection", response_class=HTMLResponse)
async def detection_params(request: Request):
    """Detection parameters form."""
    config = request.app.state.config
    if not config:
        return HTMLResponse(empty_state("配置不可用"))

    # Get first camera's config for defaults
    cam = config.cameras[0] if config.cameras else None
    mog2 = cam.mog2 if cam else None
    anomaly = cam.anomaly if cam else None
    alert = config.alerts
    st = alert.severity_thresholds
    temp = alert.temporal
    supp = alert.suppression

    mog2_html = ""
    if mog2:
        mog2_html = f"""
        <div class="card">
            <h3>MOG2 背景建模</h3>
            <div class="form-row">
                {form_group("背景历史帧数", "mog2_history", str(mog2.history), "number", "10-5000", min_val="10", max_val="5000")}
                {form_group("方差阈值", "mog2_var_threshold", str(mog2.var_threshold), "number", "1.0-500.0", min_val="1", max_val="500", step="0.5")}
            </div>
            <div class="form-row">
                {form_group("变化检测阈值", "mog2_change_pct", str(mog2.change_pct_threshold), "number", "像素变化百分比", min_val="0.0001", max_val="0.5", step="0.0001")}
                {form_group("心跳帧间隔", "mog2_heartbeat", str(mog2.heartbeat_frames), "number", "强制全帧检测间隔", min_val="10", max_val="3000")}
            </div>
            <div class="form-row">
                {form_group("锁定阈值", "lock_score", str(mog2.lock_score_threshold), "number", "触发异常锁定的分数", min_val="0.5", max_val="0.99", step="0.01")}
                {form_group("锁定解除帧数", "lock_clear", str(mog2.lock_clear_frames), "number", "连续正常帧数解除锁定", min_val="1", max_val="100")}
            </div>
        </div>"""

    anomaly_html = ""
    if anomaly:
        anomaly_html = f"""
        <div class="card">
            <h3>异常检测模型</h3>
            <div class="form-row">
                {form_group("异常阈值", "anomaly_threshold", str(anomaly.threshold), "number", "0.1-0.99", min_val="0.1", max_val="0.99", step="0.01")}
                {form_select("模型类型", "model_type", [("patchcore","PatchCore"),("efficient_ad","EfficientAD"),("anomalydino","AnomalyDINO")], anomaly.model_type)}
            </div>
            <div class="form-row">
                {form_group("SSIM基线帧数", "ssim_frames", str(anomaly.ssim_baseline_frames), "number", "无模型时的基线采集帧数", min_val="5", max_val="100")}
                {form_group("SSIM灵敏度", "ssim_sensitivity", str(anomaly.ssim_sensitivity), "number", "Sigmoid灵敏度系数", min_val="1", max_val="200", step="1")}
            </div>
        </div>"""

    severity_html = f"""
    <div class="card">
        <h3>告警阈值</h3>
        <p style="color:#8890a0;font-size:13px;margin-bottom:12px;">分数从低到高依次为：提示 &lt; 低 &lt; 中 &lt; 高</p>
        <div class="form-row">
            {form_group("提示 (INFO)", "sev_info", str(st.info), "number", min_val="0.1", max_val="0.99", step="0.01")}
            {form_group("低 (LOW)", "sev_low", str(st.low), "number", min_val="0.1", max_val="0.99", step="0.01")}
        </div>
        <div class="form-row">
            {form_group("中 (MEDIUM)", "sev_medium", str(st.medium), "number", min_val="0.1", max_val="0.99", step="0.01")}
            {form_group("高 (HIGH)", "sev_high", str(st.high), "number", min_val="0.1", max_val="0.99", step="0.01")}
        </div>
    </div>"""

    temporal_html = f"""
    <div class="card">
        <h3>时间确认与抑制</h3>
        <div class="form-row">
            {form_group("最少连续帧数", "temp_frames", str(temp.min_consecutive_frames), "number", "异常需持续N帧才触发告警", min_val="1", max_val="30")}
            {form_group("最大间隔（秒）", "temp_gap", str(temp.max_gap_seconds), "number", "帧间最大时间间隔", min_val="1", max_val="120", step="0.5")}
        </div>
        <div class="form-row">
            {form_group("最小空间重叠(IoU)", "temp_overlap", str(temp.min_spatial_overlap), "number", "连续帧异常区域IoU阈值", min_val="0", max_val="1", step="0.05")}
            {form_group("同区域抑制（秒）", "supp_zone", str(supp.same_zone_window_seconds), "number", "同区域告警去重窗口", min_val="10", max_val="3600")}
        </div>
    </div>"""

    return HTMLResponse(f"""
    <form hx-post="/api/config/detection-params" hx-swap="none">
        {mog2_html}
        {anomaly_html}
        {severity_html}
        {temporal_html}
        <div class="form-actions">
            <button type="submit" class="btn btn-primary">应用参数</button>
            <span class="form-hint" style="line-height:36px;">修改将立即生效，但需手动保存到配置文件</span>
        </div>
    </form>""")


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


@router.get("/notifications", response_class=HTMLResponse)
async def notifications_tab(request: Request):
    """Email and webhook notification settings."""
    config = request.app.state.config
    if not config:
        return HTMLResponse(empty_state("配置不可用"))

    webhook = config.alerts.webhook

    webhook_html = f"""
    <div class="card">
        <h3>Webhook 推送</h3>
        <form hx-post="/api/config/notifications" hx-swap="none">
            <div class="form-row">
                {form_select("启用", "webhook_enabled", [("true","启用"),("false","禁用")], str(webhook.enabled).lower())}
                {form_group("超时（秒）", "webhook_timeout", str(webhook.timeout), "number", min_val="1", max_val="30")}
            </div>
            {form_group("Webhook URL", "webhook_url", webhook.url, placeholder="http://plant-dcs:8080/foe-alerts")}
            <div class="form-actions">
                <button type="submit" class="btn btn-primary">保存 Webhook 设置</button>
                <button type="button" class="btn btn-ghost" hx-post="/api/config/test-webhook" hx-swap="none">发送测试消息</button>
            </div>
        </form>
    </div>"""

    return HTMLResponse(webhook_html)


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


@router.get("/storage", response_class=HTMLResponse)
def storage_tab(request: Request):
    """Storage usage and maintenance."""
    config = request.app.state.config
    db = request.app.state.db

    # Disk usage
    disk_html = ""
    try:
        usage = shutil.disk_usage("data")
        total_gb = usage.total / (1024**3)
        used_gb = usage.used / (1024**3)
        free_gb = usage.free / (1024**3)
        pct = usage.used / usage.total * 100

        bar_color = "#4caf50" if pct < 80 else "#ff9800" if pct < 95 else "#f44336"
        disk_html = f"""
        <div class="card">
            <h3>磁盘使用</h3>
            <div style="margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px;">
                    <span>已使用 {used_gb:.1f} GB / {total_gb:.0f} GB</span>
                    <span>可用 {free_gb:.1f} GB ({100-pct:.0f}%)</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width:{pct:.0f}%;background:{bar_color};"></div>
                </div>
            </div>
        </div>"""
    except OSError:
        logger.debug("config.disk_usage_render_failed", exc_info=True)

    # Retention & cleanup
    retention_days = config.storage.alert_retention_days if config else 90
    alert_count = 0
    if db:
        try:
            alert_count = db.get_alert_count()
        except Exception:
            logger.warning("config.alert_count_query_failed", exc_info=True)

    cleanup_html = f"""
    <div class="card">
        <h3>数据维护</h3>
        <table>
            <tr><td style="color:#8890a0;width:160px;">告警记录总数</td><td>{alert_count}</td></tr>
            <tr><td style="color:#8890a0;">告警保留天数</td><td>{retention_days} 天</td></tr>
        </table>
        <div class="form-actions">
            <button class="btn btn-warning" hx-post="/api/config/cleanup" hx-swap="none"
                    hx-confirm="确定清理超过 {retention_days} 天的旧告警数据？此操作不可撤销。">
                清理旧告警</button>
        </div>
    </div>"""

    return HTMLResponse(disk_html + cleanup_html)


@router.post("/cleanup")
def cleanup_data(request: Request):
    """Cleanup old alert data."""
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


@router.get("/logs", response_class=HTMLResponse)
def logs_tab(request: Request):
    """Tail log file."""
    config = request.app.state.config
    log_dir = Path(config.logging.log_dir) if config else Path("data/logs")
    log_file = log_dir / "argus.log"

    if not log_file.exists():
        return HTMLResponse(empty_state("暂无日志文件", f"日志路径: {log_file}"))

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        tail = lines[-50:]  # Last 50 lines
    except Exception as e:
        return HTMLResponse(empty_state(f"读取日志失败: {e}"))

    log_content = html.escape("".join(tail))
    return HTMLResponse(f"""
    <div class="card" hx-get="/api/config/logs" hx-trigger="every 5s" hx-swap="outerHTML">
        <h3>最近日志（最后 50 行）</h3>
        <pre style="background:#12141c;padding:12px;border-radius:6px;font-size:12px;
                    overflow-x:auto;max-height:500px;overflow-y:auto;line-height:1.6;
                    color:#a0a8b8;font-family:'Consolas','Courier New',monospace;">
{log_content}</pre>
    </div>""")


@router.get("/cameras-tab", response_class=HTMLResponse)
async def cameras_control_tab(request: Request):
    """Camera control panel (restart, lock clear)."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return HTMLResponse(empty_state("摄像头管理器不可用"))

    controls = ""
    for status in camera_manager.get_status():
        cam_id = status.camera_id
        dot = "status-healthy" if status.connected else "status-offline"
        running_text = "运行中" if status.running else "已停止"

        pipeline = camera_manager._pipelines.get(cam_id)
        locked = pipeline._locked if pipeline else False
        lock_html = ""
        if locked:
            lock_html = (
                f'<button class="btn btn-warning btn-sm" '
                f'hx-post="/api/config/clear-lock/{cam_id}" hx-swap="outerHTML">'
                f'解除异常锁定</button>'
            )

        controls += f"""
        <div class="card" style="margin-bottom:8px;">
            <div class="flex-between">
                <span><span class="status-dot {dot}"></span>{cam_id} — {running_text}</span>
                <div class="flex gap-8">
                    {lock_html}
                    <button class="btn btn-ghost btn-sm"
                        hx-post="/api/config/camera/{cam_id}/restart" hx-swap="none">重启</button>
                </div>
            </div>
        </div>"""

    return HTMLResponse(f"""
    <div class="card">
        <h3>摄像头控制</h3>
        {controls if controls else '<p style="color:#616161;">暂无摄像头</p>'}
        <div class="form-actions">
            <button class="btn btn-primary" hx-post="/api/config/reload" hx-swap="none">重新加载配置</button>
        </div>
    </div>""")


# ── Existing endpoints (preserved) ──

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
        "physics.speed_enabled": _safe_get("physics", "speed_enabled"),
        "physics.trajectory_enabled": _safe_get("physics", "trajectory_enabled"),
        "physics.localization_enabled": _safe_get("physics", "localization_enabled"),
        "physics.triangulation_enabled": _safe_get("physics", "triangulation_enabled"),
        "continuous_recording.enabled": _safe_get("continuous_recording", "enabled"),
    })


class ModuleToggleRequest(BaseModel):
    key: str
    value: bool


@router.post("/modules")
async def update_module_toggle(request: Request, req: ModuleToggleRequest):
    """Toggle a module on/off at runtime.

    Accepts key paths like 'imaging.enabled', 'physics.speed_enabled',
    'continuous_recording.enabled', 'classifier.enabled'.
    """
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
