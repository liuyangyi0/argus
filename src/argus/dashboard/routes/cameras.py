"""Camera management API routes with live preview (Chinese UI)."""

from __future__ import annotations

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
    tab_bar,
)

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


def _get_lifecycle_stages(request: Request, camera_id: str) -> list[dict]:
    """Determine camera's current lifecycle stage for Pipeline Stepper."""
    baseline_mgr = getattr(request.app.state, "baseline_manager", None)
    database = getattr(request.app.state, "database", None)
    camera_manager = request.app.state.camera_manager

    # Stage 1: Capture
    has_baselines = False
    baseline_count = 0
    if baseline_mgr:
        baseline_count = baseline_mgr.count_images(camera_id)
        has_baselines = baseline_count > 0

    # Stage 2: Baseline verified (auto-pass for now)
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
            pass

    # Stage 5: Inference running
    inferring = False
    if camera_manager:
        for s in camera_manager.get_status():
            if s.camera_id == camera_id and s.connected and s.stats and s.stats.frames_analyzed > 0:
                inferring = True

    def _stage(done, active_check):
        if done:
            return "completed"
        if active_check:
            return "active"
        return "pending"

    # Find the first incomplete stage
    first_incomplete = None
    stages_done = [has_baselines, baseline_verified, training_done, deployed, inferring]
    for i, done in enumerate(stages_done):
        if not done:
            first_incomplete = i
            break

    steps = [
        {"name": "采集", "status": "completed" if has_baselines else ("active" if first_incomplete == 0 else "pending"),
         "info": f"{baseline_count} 帧" if baseline_count else ""},
        {"name": "基线审查", "status": "completed" if baseline_verified else ("active" if first_incomplete == 1 else "pending"),
         "info": "已通过" if baseline_verified else ""},
        {"name": "训练", "status": "completed" if training_done else ("active" if first_incomplete == 2 else "pending"),
         "info": training_info},
        {"name": "发布", "status": "completed" if deployed else ("active" if first_incomplete == 3 else "pending"),
         "info": "已部署" if deployed else ""},
        {"name": "推理", "status": "completed" if inferring else ("active" if first_incomplete == 4 else "pending"),
         "info": "运行中" if inferring else ""},
    ]
    return steps


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

        # Lifecycle stage summary
        stages = _get_lifecycle_stages(request, cam.camera_id)
        active_stage = next((s for s in stages if s["status"] == "active"), None)
        stage_label = active_stage["name"] if active_stage else "就绪"

        # Alert count
        alerts_count = cam.stats.alerts_emitted if cam.stats else 0

        # Stats summary
        latency = f"{cam.stats.avg_latency_ms:.0f}ms" if cam.stats and cam.stats.avg_latency_ms > 0 else "—"

        camera_cards += f"""
        <div class="card" style="border-left:3px solid {border_var};cursor:pointer;transition:border-color 0.15s;"
             hx-get="/api/cameras/{cam.camera_id}/detail" hx-target="#content" hx-swap="innerHTML"
             hx-push-url="/cameras?cam={cam.camera_id}"
             onmouseenter="this.style.borderColor='var(--status-info)'"
             onmouseleave="this.style.borderColor='{border_var}'">
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

    # Find camera status
    cam = None
    for s in camera_manager.get_status():
        if s.camera_id == camera_id:
            cam = s
            break

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
            "采集": ("开始采集基线", f"/api/baseline/capture?camera_id={camera_id}"),
            "基线审查": ("审查基线", f"/api/baseline/list?camera_id={camera_id}"),
            "训练": ("开始训练", f"/api/baseline/train?camera_id={camera_id}"),
            "发布": ("部署模型", f"/api/cameras/{camera_id}/deploy-panel"),
            "推理": ("查看实时状态", f"/api/cameras/{camera_id}/live"),
        }
        label, url = action_map.get(active_stage["name"], ("", ""))
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

    cam = None
    for s in camera_manager.get_status():
        if s.camera_id == camera_id:
            cam = s
            break

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
            pass

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

    form = await request.form()
    camera_id = form.get("camera_id", "").strip()
    name = form.get("name", "").strip()
    source = form.get("source", "").strip()
    protocol = form.get("protocol", "rtsp")
    fps_target = int(form.get("fps_target", 5))
    resolution_str = form.get("resolution", "1920,1080")

    if not camera_id or not name or not source:
        return JSONResponse({"error": "请填写所有必填字段"}, status_code=400)

    # Check for duplicate
    existing_ids = [c.camera_id for c in config.cameras]
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
    # Note: camera is added to config but not started. User must click "start".
    logger.info("camera.added", camera_id=camera_id, source=source)

    # Return the refreshed camera list
    return HTMLResponse(
        headers={"HX-Trigger": '{"showToast": {"message": "摄像头已添加", "type": "success"}}'},
        content='<div hx-get="/api/cameras" hx-trigger="load" hx-swap="outerHTML"></div>',
    )


@router.post("/{camera_id}/start")
async def start_camera(request: Request, camera_id: str):
    """Start a stopped camera."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return JSONResponse({"error": "不可用"}, status_code=503)

    success = camera_manager.start_camera(camera_id)
    if success:
        return JSONResponse(
            {"status": "ok"},
            headers={"HX-Trigger": '{"showToast": {"message": "摄像头已启动", "type": "success"}}'},
        )
    return JSONResponse({"error": "启动失败"}, status_code=500)


@router.post("/{camera_id}/stop")
async def stop_camera(request: Request, camera_id: str):
    """Stop a running camera."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return JSONResponse({"error": "不可用"}, status_code=503)

    camera_manager.stop_camera(camera_id)
    return JSONResponse(
        {"status": "ok"},
        headers={"HX-Trigger": '{"showToast": {"message": "摄像头已停止", "type": "success"}}'},
    )


@router.get("/json")
async def cameras_json(request: Request):
    """JSON API for camera status."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return []

    return [
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
    ]


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
