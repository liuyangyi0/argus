"""Camera management API routes with live preview (Chinese UI)."""

from __future__ import annotations

import time

import cv2
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from argus.dashboard.components import empty_state, page_header, status_dot

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


@router.get("", response_class=HTMLResponse)
async def cameras_page(request: Request):
    """Camera list and status page."""
    camera_manager = request.app.state.camera_manager

    if not camera_manager:
        return HTMLResponse(empty_state("摄像头管理器不可用"))

    statuses = camera_manager.get_status()

    # Add camera form (collapsible)
    add_form = """
    <div class="card" id="add-camera-form" style="display:none;">
        <h3>添加摄像头</h3>
        <form hx-post="/api/cameras" hx-target="#cameras-list" hx-swap="outerHTML"
              hx-on::after-request="if(event.detail.successful)this.parentElement.style.display='none'">
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">摄像头ID</label>
                    <input type="text" name="camera_id" class="form-input" placeholder="cam_02" required>
                </div>
                <div class="form-group">
                    <label class="form-label">名称</label>
                    <input type="text" name="name" class="form-input" placeholder="反应堆厂房入口" required>
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">视频源</label>
                    <input type="text" name="source" class="form-input"
                           placeholder="rtsp://admin:pass@192.168.1.100:554/stream" required>
                    <div class="form-hint">支持 RTSP URL、USB 设备号（0, 1）或本地视频文件路径</div>
                </div>
                <div class="form-group">
                    <label class="form-label">协议</label>
                    <select name="protocol" class="form-select">
                        <option value="rtsp">RTSP</option>
                        <option value="usb">USB</option>
                        <option value="file">文件</option>
                    </select>
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">目标帧率</label>
                    <input type="number" name="fps_target" class="form-input" value="5" min="1" max="30">
                </div>
                <div class="form-group">
                    <label class="form-label">分辨率</label>
                    <select name="resolution" class="form-select">
                        <option value="1920,1080">1920x1080</option>
                        <option value="1280,720">1280x720</option>
                        <option value="640,480">640x480</option>
                    </select>
                </div>
            </div>
            <div class="form-actions">
                <button type="submit" class="btn btn-primary">添加摄像头</button>
                <button type="button" class="btn btn-ghost"
                        onclick="document.getElementById('add-camera-form').style.display='none'">取消</button>
            </div>
        </form>
    </div>"""

    camera_cards = ""
    for cam in statuses:
        dot_status = "connected" if cam.connected else "offline"
        status_text = "在线" if cam.connected else "离线"
        border_color = "#2e7d32" if cam.connected else "#424242"

        stats_html = ""
        if cam.stats:
            s = cam.stats
            skip_pct = (
                (s.frames_skipped_no_change / s.frames_captured * 100)
                if s.frames_captured > 0 else 0
            )
            stats_html = f"""
            <table style="font-size:13px;width:100%;">
                <tr><td style="color:#8890a0;">已采集帧</td><td>{s.frames_captured}</td></tr>
                <tr><td style="color:#8890a0;">跳过（无变化）</td><td>{s.frames_skipped_no_change} ({skip_pct:.0f}%)</td></tr>
                <tr><td style="color:#8890a0;">跳过（有人）</td><td>{s.frames_skipped_person}</td></tr>
                <tr><td style="color:#8890a0;">已分析帧</td><td>{s.frames_analyzed}</td></tr>
                <tr><td style="color:#8890a0;">检测到异常</td><td>{s.anomalies_detected}</td></tr>
                <tr><td style="color:#8890a0;">已发出告警</td><td>{s.alerts_emitted}</td></tr>
                <tr><td style="color:#8890a0;">平均延迟</td><td>{s.avg_latency_ms:.1f}ms</td></tr>
            </table>"""
        else:
            stats_html = '<p style="color:#616161;font-size:13px;">暂无统计数据</p>'

        # Live preview
        preview_html = (
            f'<div style="margin-bottom:12px;border-radius:6px;overflow:hidden;'
            f'background:#000;aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;">'
            f'<img src="/api/cameras/{cam.camera_id}/stream" '
            f'style="width:100%;height:auto;" alt="实时画面" '
            f'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\'" />'
            f'<span style="color:#616161;font-size:14px;display:none;">无信号</span>'
            f'</div>'
            if cam.connected else
            f'<div style="margin-bottom:12px;border-radius:6px;background:#000;'
            f'aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;">'
            f'<span style="color:#616161;font-size:14px;">离线</span></div>'
        )

        # Action buttons
        actions = ""
        if cam.connected:
            actions += (
                f'<button class="btn btn-warning btn-sm" '
                f'hx-post="/api/cameras/{cam.camera_id}/stop" hx-swap="none" '
                f'hx-confirm="确定停止摄像头 {cam.camera_id}？">停止</button>'
            )
        else:
            actions += (
                f'<button class="btn btn-success btn-sm" '
                f'hx-post="/api/cameras/{cam.camera_id}/start" hx-swap="none">启动</button>'
            )
        actions += (
            f' <button class="btn btn-ghost btn-sm" '
            f'hx-post="/api/config/camera/{cam.camera_id}/restart" hx-swap="none">重启</button>'
        )

        camera_cards += f"""
        <div class="card" style="border-left:3px solid {border_color};">
            <div class="flex-between mb-8">
                <h3 style="margin:0;">{status_dot(dot_status)}{cam.camera_id} — {cam.name}</h3>
                <div class="flex gap-8">
                    <span style="font-size:12px;color:#8890a0;line-height:28px;">{status_text}</span>
                    {actions}
                </div>
            </div>
            {preview_html}
            {stats_html}
        </div>"""

    if not camera_cards:
        camera_cards = empty_state("暂无摄像头", "点击右上角按钮添加摄像头")

    header = page_header(
        "摄像头管理",
        f"{len(statuses)} 台摄像头",
        '<button class="btn btn-primary" '
        'onclick="document.getElementById(\'add-camera-form\').style.display=\'block\'">添加摄像头</button>',
    )

    return HTMLResponse(f"""
    <div id="cameras-list" data-ws-topic="cameras" data-ws-refresh-url="/api/cameras"
         hx-get="/api/cameras" hx-trigger="every 30s" hx-swap="outerHTML">
        {header}
        {add_form}
        <div class="grid-2">{camera_cards}</div>
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
