"""Camera management API routes with live preview (Chinese UI)."""

from __future__ import annotations

import asyncio
import time

import cv2
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from argus.dashboard.components import (
    empty_state,
    page_header,
    pipeline_stepper,
    status_dot,
)
from argus.dashboard.forms import htmx_toast_headers, parse_request_form

logger = structlog.get_logger()

# Maximum stream duration in seconds (30 minutes)
_MAX_STREAM_DURATION = 30 * 60

router = APIRouter()


class AddCameraRequest(BaseModel):
    camera_id: str
    name: str
    source: str
    protocol: str = "rtsp"
    fps_target: int = 5
    resolution: list[int] = [1920, 1080]


_STAGE_IDS = ["capture", "review", "training", "deploy", "inference"]
_STAGE_NAMES = {"capture": "采集", "review": "基线审查", "training": "训练", "deploy": "发布", "inference": "推理"}


def _find_camera_config(request: Request, camera_id: str):
    """Find camera config from the running manager first, then persisted app config."""
    camera_manager = getattr(request.app.state, "camera_manager", None)
    if camera_manager is not None:
        config = next((c for c in getattr(camera_manager, "_cameras", []) if c.camera_id == camera_id), None)
        if config is not None:
            return config

    app_config = getattr(request.app.state, "config", None)
    if app_config is not None:
        return next((c for c in getattr(app_config, "cameras", []) if c.camera_id == camera_id), None)
    return None


def _get_lifecycle_stages(request: Request, camera_id: str, *, cam_status=None) -> list[dict]:
    """Determine camera's current lifecycle stage for Pipeline Stepper.

    Args:
        cam_status: Pre-fetched CameraStatus to avoid redundant get_status() calls.
    """
    baseline_mgr = getattr(request.app.state, "baseline_manager", None)
    database = getattr(request.app.state, "database", None)
    camera_manager = getattr(request.app.state, "camera_manager", None)

    # Stage 1: Capture
    baseline_count = 0
    if baseline_mgr:
        baseline_count = baseline_mgr.count_images(camera_id)
    has_baselines = baseline_count > 0

    # Stage 2: Baseline verified (auto-pass for now, future: explicit approval)
    baseline_verified = has_baselines

    # Stage 3: Training
    training_done = False
    training_info = ""
    if database:
        latest = database.get_latest_training(camera_id)
        if latest and latest.status == "complete":
            training_done = True
            training_info = f"等级 {latest.quality_grade}" if latest.quality_grade else ""

    # Stage 4: Deployment
    deployed = False
    if camera_manager:
        try:
            det_status = camera_manager.get_detector_status(camera_id)
            if det_status and det_status.get("mode") == "anomalib":
                deployed = True
        except Exception:
            logger.debug("lifecycle.detector_status_failed", camera_id=camera_id, exc_info=True)

    # Stage 5: Inference running (use pre-fetched status to avoid N+1)
    inferring = False
    if cam_status is not None:
        inferring = bool(cam_status.connected and cam_status.stats and cam_status.stats.frames_analyzed > 0)
    elif camera_manager:
        for s in camera_manager.get_status():
            if s.camera_id == camera_id and s.connected and s.stats and s.stats.frames_analyzed > 0:
                inferring = True
                break

    # Determine status per stage
    stages_done = [has_baselines, baseline_verified, training_done, deployed, inferring]
    first_incomplete = next((i for i, done in enumerate(stages_done) if not done), None)

    def _status(idx, done):
        if done:
            return "completed"
        return "active" if first_incomplete == idx else "pending"

    infos = [
        f"{baseline_count} 帧" if baseline_count else "",
        "已通过" if baseline_verified else "",
        training_info,
        "已部署" if deployed else "",
        "运行中" if inferring else "",
    ]

    return [
        {"id": _STAGE_IDS[i], "name": _STAGE_NAMES[_STAGE_IDS[i]],
         "status": _status(i, stages_done[i]), "info": infos[i]}
        for i in range(5)
    ]


def _find_camera(camera_manager, camera_id: str):
    """Find a single camera status from the manager. Returns None if not found."""
    return next((s for s in camera_manager.get_status() if s.camera_id == camera_id), None)


@router.get("", response_class=HTMLResponse)
async def cameras_page(request: Request):
    """Camera list page — compact cards with lifecycle stage indicator."""
    camera_manager = request.app.state.camera_manager

    if not camera_manager:
        return HTMLResponse(empty_state("摄像头管理器不可用"))

    statuses = camera_manager.get_status()

    camera_cards = ""
    for cam in statuses:
        dot_status = "connected" if cam.connected else "offline"
        status_text = "在线" if cam.connected else "离线"
        border_var = "var(--status-ok)" if cam.connected else "var(--border-default)"

        # Lifecycle stage summary (pass pre-fetched status to avoid N+1)
        stages = _get_lifecycle_stages(request, cam.camera_id, cam_status=cam)
        active_stage = next((s for s in stages if s["status"] == "active"), None)
        stage_label = active_stage["name"] if active_stage else "就绪"

        # Alert count
        alerts_count = cam.stats.alerts_emitted if cam.stats else 0

        # Stats summary
        latency = f"{cam.stats.avg_latency_ms:.0f}ms" if cam.stats and cam.stats.avg_latency_ms > 0 else "—"

        camera_cards += f"""
        <div class="card camera-list-card" style="border-left:3px solid {border_var};"
             hx-get="/api/cameras/{cam.camera_id}/detail" hx-target="#content" hx-swap="innerHTML"
             hx-push-url="/cameras?cam={cam.camera_id}">
            <div class="flex-between mb-8">
                <div>
                    <span style="font-size:var(--text-lg);font-weight:var(--font-semibold);">
                        {status_dot(dot_status)}{cam.camera_id}
                    </span>
                    <span style="color:var(--text-secondary);margin-left:var(--space-2);">{cam.name}</span>
                </div>
                <span class="badge badge-info">{stage_label}</span>
            </div>
            <div class="flex-between" style="color:var(--text-tertiary);font-size:var(--text-sm);">
                <span>{status_text}</span>
                <span>告警 {alerts_count} | 延迟 {latency}</span>
            </div>
        </div>"""

    if not camera_cards:
        camera_cards = empty_state("暂无摄像头", "在系统设置中添加摄像头")

    header = page_header("摄像头", f"{len(statuses)} 台")

    return HTMLResponse(f"""
    <div id="cameras-list" data-ws-topic="cameras" data-ws-refresh-url="/api/cameras"
         hx-get="/api/cameras" hx-trigger="every 30s" hx-swap="outerHTML">
        {header}
        <div class="grid-3">{camera_cards}</div>
    </div>""")


@router.get("/{camera_id}/detail", response_class=HTMLResponse)
async def camera_detail(request: Request, camera_id: str):
    """Camera detail page with Pipeline Stepper and sub-tabs."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return HTMLResponse(empty_state("摄像头管理器不可用"))

    cam = _find_camera(camera_manager, camera_id)
    if cam is None:
        return HTMLResponse(empty_state(f"摄像头 {camera_id} 不存在"))

    # Camera name from config
    cam_name = cam.name or camera_id
    dot_status = "connected" if cam.connected else "offline"
    status_text = "在线" if cam.connected else "离线"

    # Pipeline Stepper
    stages = _get_lifecycle_stages(request, camera_id)
    stepper_html = pipeline_stepper(stages)

    # Next action suggestion
    active_stage = next((s for s in stages if s["status"] == "active"), None)
    next_action = ""
    if active_stage:
        action_map = {
            "capture": ("开始采集基线", f"/api/baseline/capture?camera_id={camera_id}"),
            "review": ("审查基线", f"/api/baseline/list?camera_id={camera_id}"),
            "training": ("开始训练", f"/api/baseline/train?camera_id={camera_id}"),
            "deploy": ("部署模型", f"/api/cameras/{camera_id}/deploy-panel"),
            "inference": ("查看实时状态", f"/api/cameras/{camera_id}/live"),
        }
        label, url = action_map.get(active_stage["id"], ("", ""))
        if label:
            next_action = (
                f'<button class="btn btn-primary" hx-get="{url}" '
                f'hx-target="#cam-tab-content" hx-swap="innerHTML">{label}</button>'
            )

    # Sub-tabs
    detail_tabs = [
        ("live", "实时画面", f"/api/cameras/{camera_id}/live"),
        ("capture", "采集", f"/api/baseline/capture?camera_id={camera_id}"),
        ("baselines", "基线", f"/api/baseline/list?camera_id={camera_id}"),
        ("train", "训练", f"/api/baseline/train?camera_id={camera_id}"),
        ("deploy", "发布", f"/api/cameras/{camera_id}/deploy-panel"),
        ("zones", "检测区域", f"/api/zones?camera_id={camera_id}"),
    ]
    tabs_html = ""
    for tab_id, label, url in detail_tabs:
        tabs_html += (
            f'<button class="tab-item" '
            f'hx-get="{url}" hx-target="#cam-tab-content" hx-swap="innerHTML" '
            f'onclick="document.querySelectorAll(\'.cam-tabs .tab-item\').forEach(t=>t.classList.remove(\'active\'));this.classList.add(\'active\')">'
            f'{label}</button>'
        )

    # Back button
    back_btn = (
        '<button class="btn btn-ghost btn-sm" '
        'hx-get="/api/cameras" hx-target="#content" hx-swap="innerHTML">'
        '&larr; 返回列表</button>'
    )

    return HTMLResponse(f"""
    <div>
        <div class="flex-between mb-16">
            <div class="flex-center gap-12">
                {back_btn}
                <h2 style="font-size:var(--text-2xl);font-weight:var(--font-semibold);">
                    {status_dot(dot_status)}{camera_id}
                    <span style="color:var(--text-secondary);font-weight:var(--font-normal);font-size:var(--text-lg);">
                        {cam_name}
                    </span>
                </h2>
                <span style="color:var(--text-tertiary);font-size:var(--text-sm);">{status_text}</span>
            </div>
            <div class="flex gap-8">{next_action}</div>
        </div>

        {stepper_html}

        <div class="cam-tabs tab-bar" style="margin-top:var(--space-4);">
            {tabs_html}
        </div>
        <div id="cam-tab-content"
             hx-get="/api/cameras/{camera_id}/live"
             hx-trigger="load"
             hx-swap="innerHTML">
            <div class="empty-state"><div class="spinner"></div></div>
        </div>
    </div>""")


@router.get("/{camera_id}/live", response_class=HTMLResponse)
async def camera_live_panel(request: Request, camera_id: str):
    """Live monitoring sub-tab for camera detail page."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return HTMLResponse(empty_state("不可用"))

    cam = _find_camera(camera_manager, camera_id)
    if cam is None:
        return HTMLResponse(empty_state("摄像头未找到"))

    # Live preview
    if cam.connected:
        preview = (
            f'<div class="camera-card">'
            f'<div class="preview">'
            f'<img src="/api/cameras/{camera_id}/stream" style="width:100%;height:auto;" alt="实时画面" '
            f'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\'" />'
            f'<span style="color:var(--text-tertiary);display:none;">无信号</span>'
            f'</div></div>'
        )
    else:
        preview = (
            f'<div class="camera-card"><div class="preview">'
            f'<span style="color:var(--text-tertiary);">离线</span></div></div>'
        )

    # Stats table
    stats_html = ""
    if cam.stats:
        s = cam.stats
        skip_pct = (s.frames_skipped_no_change / max(s.frames_captured, 1) * 100)
        stats_html = f"""
        <div class="card mt-16">
            <h3>运行统计</h3>
            <table>
                <tr><td style="color:var(--text-secondary);">已采集帧</td><td>{s.frames_captured}</td></tr>
                <tr><td style="color:var(--text-secondary);">跳过（无变化）</td><td>{s.frames_skipped_no_change} ({skip_pct:.0f}%)</td></tr>
                <tr><td style="color:var(--text-secondary);">跳过（有人）</td><td>{s.frames_skipped_person}</td></tr>
                <tr><td style="color:var(--text-secondary);">已分析帧</td><td>{s.frames_analyzed}</td></tr>
                <tr><td style="color:var(--text-secondary);">检测到异常</td><td>{s.anomalies_detected}</td></tr>
                <tr><td style="color:var(--text-secondary);">已发出告警</td><td>{s.alerts_emitted}</td></tr>
                <tr><td style="color:var(--text-secondary);">平均延迟</td><td>{s.avg_latency_ms:.1f}ms</td></tr>
            </table>
        </div>"""

    # Action buttons
    actions = ""
    if cam.connected:
        actions = (
            f'<button class="btn btn-warning btn-sm" '
            f'hx-post="/api/cameras/{camera_id}/stop" hx-swap="none" '
            f'hx-confirm="确定停止摄像头？">停止</button>'
            f' <button class="btn btn-ghost btn-sm" '
            f'hx-post="/api/config/camera/{camera_id}/restart" hx-swap="none">重启</button>'
        )
    else:
        actions = (
            f'<button class="btn btn-success btn-sm" '
            f'hx-post="/api/cameras/{camera_id}/start" hx-swap="none">启动</button>'
        )

    return HTMLResponse(f"""
    <div>
        <div class="flex-between mb-8">
            <span style="font-size:var(--text-sm);color:var(--text-secondary);">
                实时画面 · {camera_id}
            </span>
            <div class="flex gap-8">{actions}</div>
        </div>
        {preview}
        {stats_html}
    </div>""")


@router.get("/{camera_id}/deploy-panel", response_class=HTMLResponse)
async def camera_deploy_panel(request: Request, camera_id: str):
    """Model deployment sub-tab for camera detail page."""
    camera_manager = request.app.state.camera_manager
    database = getattr(request.app.state, "database", None)

    # Current detector status
    det_html = '<p style="color:var(--text-tertiary);">检测器状态不可用</p>'
    if camera_manager:
        try:
            det = camera_manager.get_detector_status(camera_id)
            if det:
                mode = det.get("mode", "unknown")
                model_path = det.get("model_path", "—")
                threshold = det.get("threshold", "—")
                det_html = f"""
                <table>
                    <tr><td style="color:var(--text-secondary);">检测模式</td><td>{mode}</td></tr>
                    <tr><td style="color:var(--text-secondary);">模型路径</td><td class="mono">{model_path}</td></tr>
                    <tr><td style="color:var(--text-secondary);">阈值</td><td>{threshold}</td></tr>
                </table>"""
        except Exception:
            logger.debug("deploy_panel.detector_status_failed", camera_id=camera_id, exc_info=True)

    # Latest training info
    train_html = ""
    if database:
        latest = database.get_latest_training(camera_id)
        if latest and latest.status == "complete":
            train_html = f"""
            <div class="card mt-16">
                <h3>最近训练</h3>
                <table>
                    <tr><td style="color:var(--text-secondary);">模型类型</td><td>{latest.model_type}</td></tr>
                    <tr><td style="color:var(--text-secondary);">质量等级</td><td>{latest.quality_grade or '—'}</td></tr>
                    <tr><td style="color:var(--text-secondary);">推荐阈值</td><td>{latest.threshold_recommended or '—'}</td></tr>
                    <tr><td style="color:var(--text-secondary);">训练时间</td><td>{latest.trained_at or '—'}</td></tr>
                </table>
            </div>"""

    return HTMLResponse(f"""
    <div>
        <div class="card">
            <h3>当前部署状态</h3>
            {det_html}
        </div>
        {train_html}
    </div>""")


@router.post("")
async def add_camera(request: Request):
    """Add a new camera configuration."""
    camera_manager = request.app.state.camera_manager
    config = request.app.state.config
    if not camera_manager or not config:
        return JSONResponse({"error": "不可用"}, status_code=503)

    form = await parse_request_form(request)
    camera_id = form.get("camera_id", "").strip()
    name = form.get("name", "").strip()
    source = form.get("source", "").strip()
    protocol = form.get("protocol", "rtsp")
    fps_target = int(form.get("fps_target", 5))
    resolution_str = form.get("resolution", "1920,1080")

    if not camera_id or not name or not source:
        return JSONResponse({"error": "请填写所有必填字段"}, status_code=400)

    manager_cameras = getattr(camera_manager, "_cameras", None)

    # Check for duplicate across config and running manager state
    existing_ids = {c.camera_id for c in config.cameras}
    if isinstance(manager_cameras, list):
        existing_ids.update(c.camera_id for c in manager_cameras)

    if camera_id in existing_ids:
        return JSONResponse({"error": f"摄像头 {camera_id} 已存在"}, status_code=400)

    # Parse resolution
    try:
        res_parts = resolution_str.split(",")
        resolution = (int(res_parts[0]), int(res_parts[1]))
    except (ValueError, IndexError):
        resolution = (1920, 1080)

    # Create camera config
    from argus.config.schema import CameraConfig
    cam_config = CameraConfig(
        camera_id=camera_id,
        name=name,
        source=source,
        protocol=protocol,
        fps_target=fps_target,
        resolution=resolution,
    )

    config.cameras.append(cam_config)
    added_to_manager = False

    try:
        if isinstance(manager_cameras, list) and manager_cameras is not config.cameras:
            manager_cameras.append(cam_config)
            added_to_manager = True

        config_path = getattr(request.app.state, "config_path", None)
        if config_path:
            from argus.config.loader import save_config as _save_config

            _save_config(config, config_path)
    except Exception:
        config.cameras = [camera for camera in config.cameras if camera is not cam_config]
        if added_to_manager and isinstance(manager_cameras, list):
            manager_cameras[:] = [camera for camera in manager_cameras if camera is not cam_config]
        logger.exception("camera.add_failed", camera_id=camera_id)
        return JSONResponse({"error": "摄像头配置保存失败"}, status_code=500)

    # Note: camera is added to config but not started. User must click "start".
    logger.info("camera.added", camera_id=camera_id, source=source)

    # Return the refreshed camera list
    return HTMLResponse(
        headers=htmx_toast_headers("摄像头已添加"),
        content='<div hx-get="/api/cameras" hx-trigger="load" hx-swap="outerHTML"></div>',
    )


@router.post("/{camera_id}/start")
async def start_camera(request: Request, camera_id: str):
    """Start a stopped camera."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return JSONResponse({"error": "不可用"}, status_code=503)

    success = await asyncio.to_thread(camera_manager.start_camera, camera_id)
    if success:
        return JSONResponse(
            {"status": "ok"},
            headers=htmx_toast_headers("摄像头已启动"),
        )
    return JSONResponse({"error": "启动失败"}, status_code=500)


@router.post("/{camera_id}/stop")
async def stop_camera(request: Request, camera_id: str):
    """Stop a running camera."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return JSONResponse({"error": "不可用"}, status_code=503)

    await asyncio.to_thread(camera_manager.stop_camera, camera_id)
    return JSONResponse(
        {"status": "ok"},
        headers=htmx_toast_headers("摄像头已停止"),
    )


@router.get("/usb-devices")
async def usb_devices(request: Request):
    """Probe USB camera indices 0-9 and return available devices."""
    import asyncio

    camera_manager = getattr(request.app.state, "camera_manager", None)
    # Indices already occupied by running USB cameras
    in_use: set[int] = set()
    if camera_manager:
        for cfg in camera_manager._cameras:
            if cfg.protocol == "usb" and cfg.camera_id in camera_manager._threads:
                try:
                    in_use.add(int(cfg.source))
                except (ValueError, TypeError):
                    pass

    def _probe() -> list[dict]:
        results = []
        for idx in range(10):
            if idx in in_use:
                continue
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            try:
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    results.append({
                        "index": idx,
                        "name": f"USB Camera {idx}",
                        "width": w,
                        "height": h,
                    })
            finally:
                cap.release()
        return results

    devices = await asyncio.to_thread(_probe)
    return JSONResponse(devices)


@router.get("/json")
async def cameras_json(request: Request):
    """JSON API for camera status."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return JSONResponse({"cameras": []})

    return JSONResponse({"cameras": [
        {
            "camera_id": s.camera_id,
            "name": s.name,
            "connected": s.connected,
            "running": s.running,
            "stats": {
                "frames_captured": s.stats.frames_captured,
                "frames_analyzed": s.stats.frames_analyzed,
                "anomalies_detected": s.stats.anomalies_detected,
                "alerts_emitted": s.stats.alerts_emitted,
                "avg_latency_ms": round(s.stats.avg_latency_ms, 1),
            } if s.stats else None,
        }
        for s in camera_manager.get_status()
    ]})


@router.get("/{camera_id}/detail/json")
async def camera_detail(request: Request, camera_id: str):
    """Return a detailed camera payload for the detail page."""
    camera_manager = request.app.state.camera_manager
    camera_config = _find_camera_config(request, camera_id)
    if camera_config is None:
        return JSONResponse({"error": f"摄像头 {camera_id} 不存在"}, status_code=404)

    status = None
    if camera_manager is not None:
        status = next((item for item in camera_manager.get_status() if item.camera_id == camera_id), None)

    runner = camera_manager.get_runner_snapshot(camera_id) if camera_manager else None
    detector = camera_manager.get_detector_status(camera_id) if camera_manager else None
    learning = camera_manager.get_learning_progress(camera_id) if camera_manager else None
    pipeline_mode = camera_manager.get_pipeline_mode(camera_id) if camera_manager else None
    anomaly_locked = camera_manager.is_anomaly_locked(camera_id) if camera_manager else False

    health_monitor = getattr(request.app.state, "health_monitor", None)
    health = health_monitor.get_camera_health(camera_id) if health_monitor is not None else None

    stats = None
    if status is not None and status.stats is not None:
        stats = {
            "frames_captured": status.stats.frames_captured,
            "frames_analyzed": status.stats.frames_analyzed,
            "anomalies_detected": status.stats.anomalies_detected,
            "alerts_emitted": status.stats.alerts_emitted,
            "avg_latency_ms": round(status.stats.avg_latency_ms, 1),
        }

    return JSONResponse({
        "camera_id": camera_id,
        "name": camera_config.name,
        "connected": status.connected if status is not None else False,
        "running": status.running if status is not None else False,
        "stats": stats,
        "config": camera_config.model_dump(mode="json"),
        "runtime": {
            "pipeline_mode": pipeline_mode,
            "anomaly_locked": anomaly_locked,
            "learning_progress": learning,
        },
        "runner": {
            "model_ref": runner.model_ref,
            "health_status": runner.health_status,
            "cusum_state": runner.cusum_state,
            "lock_state": runner.lock_state.value,
            "last_heartbeat": runner.last_heartbeat,
            "version_tag": runner.version_tag,
            "degradation_state": runner.degradation_state.value,
            "consecutive_failures": runner.consecutive_failures,
        } if runner is not None else None,
        "detector": detector,
        "health": health,
    })


@router.get("/{camera_id}/runner")
async def camera_runner_snapshot(request: Request, camera_id: str):
    """Get the inference runner state snapshot for a camera (5.1)."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return JSONResponse({"error": "Camera manager not running"}, status_code=503)

    snapshot = camera_manager.get_runner_snapshot(camera_id)
    if snapshot is None:
        return JSONResponse({"error": f"Camera {camera_id} not found"}, status_code=404)

    return JSONResponse({
        "camera_id": snapshot.camera_id,
        "model_ref": snapshot.model_ref,
        "health_status": snapshot.health_status,
        "cusum_state": snapshot.cusum_state,
        "lock_state": snapshot.lock_state.value,
        "last_heartbeat": snapshot.last_heartbeat,
        "version_tag": snapshot.version_tag,
        "degradation_state": snapshot.degradation_state.value,
        "consecutive_failures": snapshot.consecutive_failures,
        "stats": {
            "frames_captured": snapshot.stats.frames_captured,
            "frames_analyzed": snapshot.stats.frames_analyzed,
            "anomalies_detected": snapshot.stats.anomalies_detected,
            "alerts_emitted": snapshot.stats.alerts_emitted,
            "avg_latency_ms": round(snapshot.stats.avg_latency_ms, 1),
        },
    })


@router.get("/{camera_id}/snapshot")
async def camera_snapshot(request: Request, camera_id: str):
    """Get the latest frame from a camera as a JPEG image."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return Response(status_code=503)

    frame = camera_manager.get_latest_frame(camera_id)
    if frame is None:
        return Response(status_code=404)

    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store"},
    )


@router.get("/{camera_id}/stream")
async def camera_stream(request: Request, camera_id: str):
    """MJPEG stream of the latest frames from a camera (~5 FPS)."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return Response(status_code=503)

    def generate_frames():
        start_time = time.monotonic()
        try:
            while True:
                if time.monotonic() - start_time > _MAX_STREAM_DURATION:
                    break
                frame = camera_manager.get_latest_frame(camera_id)
                if frame is not None:
                    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + buffer.tobytes()
                        + b"\r\n"
                    )
                time.sleep(0.2)
        except GeneratorExit:
            pass

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/{camera_id}/heatmap-stream")
async def camera_heatmap_stream(request: Request, camera_id: str):
    """MJPEG stream with anomaly heatmap overlay."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return Response(status_code=503)

    import numpy as np

    def generate_heatmap_frames():
        start_time = time.monotonic()
        try:
            while True:
                if time.monotonic() - start_time > _MAX_STREAM_DURATION:
                    break
                frame = camera_manager.get_latest_frame(camera_id)
                if frame is not None:
                    anomaly_map = camera_manager.get_latest_anomaly_map(camera_id)
                    if anomaly_map is not None:
                        h, w = frame.shape[:2]
                        heatmap = cv2.resize(anomaly_map, (w, h))
                        heatmap_u8 = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
                        heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
                        mask = heatmap > 0.3
                        blended = frame.copy()
                        if mask.any():
                            mask_3ch = np.stack([mask] * 3, axis=-1)
                            blended = np.where(
                                mask_3ch,
                                cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0),
                                frame,
                            )
                        frame = blended

                    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + buffer.tobytes()
                        + b"\r\n"
                    )
                time.sleep(0.2)
        except GeneratorExit:
            pass

    return StreamingResponse(
        generate_heatmap_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Video Wall API (UX v2 §2) ──

@router.get("/wall/status")
async def wall_status(request: Request):
    """Return aggregated status for all cameras in the video wall.

    Response: {cameras: [{camera_id, name, status, model_version,
               current_score, score_sparkline, alert_count_today,
               active_alert, degradation}]}
    """
    from datetime import datetime, timezone

    camera_manager = request.app.state.camera_manager
    if camera_manager is None:
        return JSONResponse({"cameras": []})

    db = getattr(request.app.state, "db", None)
    health_monitor = getattr(request.app.state, "health_monitor", None)

    cameras = []
    for cam_status in camera_manager.get_status():
        cam_id = cam_status.camera_id
        tile: dict = {
            "camera_id": cam_id,
            "name": getattr(cam_status, "name", cam_id),
            "status": "online" if getattr(cam_status, "connected", False) else "offline",
            "model_version": getattr(cam_status, "model_version_id", None),
            "current_score": 0.0,
            "score_sparkline": [],
            "alert_count_today": 0,
            "active_alert": None,
            "degradation": None,
        }

        # Get sparkline from pipeline if available
        pipeline = getattr(cam_status, "pipeline", None)
        if pipeline is not None and hasattr(pipeline, "get_wall_status"):
            wall_data = pipeline.get_wall_status()
            tile["current_score"] = wall_data.get("current_score", 0.0)
            tile["score_sparkline"] = wall_data.get("score_sparkline", [])

        # Today's alert count
        if db is not None:
            try:
                today_count = db.get_alert_count(
                    camera_id=cam_id,
                    since=datetime.now(timezone.utc).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    ),
                )
                tile["alert_count_today"] = today_count
            except Exception:
                pass

        # Active (unresolved) alert
        if db is not None:
            try:
                recent = db.get_alerts(camera_id=cam_id, limit=1)
                if recent and recent[0].workflow_status in ("new", "acknowledged", "investigating"):
                    tile["active_alert"] = {
                        "alert_id": recent[0].alert_id,
                        "severity": recent[0].severity,
                    }
            except Exception:
                pass

        # Degradation status from health monitor
        if health_monitor is not None:
            health = health_monitor.get_camera_health(cam_id)
            if health and not health.get("connected", True):
                tile["degradation"] = "rtsp_broken"
            elif health and health.get("error"):
                tile["degradation"] = "error"

        cameras.append(tile)

    return JSONResponse({"cameras": cameras})
