"""Zone management API routes (Chinese UI)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from argus.dashboard.components import empty_state, page_header
from argus.dashboard.forms import htmx_toast_headers

router = APIRouter()


class ZoneCreateRequest(BaseModel):
    camera_id: str
    zone_id: str
    name: str
    polygon: list[tuple[int, int]]
    zone_type: str = "include"
    priority: str = "standard"
    anomaly_threshold: float = 0.7


@router.get("", response_class=HTMLResponse)
async def zones_page(request: Request):
    """Zone management page with editor."""
    camera_manager = request.app.state.camera_manager

    cameras_html = ""
    if camera_manager:
        for status in camera_manager.get_status():
            cam_id = status.camera_id
            pipeline = camera_manager._pipelines.get(cam_id)
            zones_list = ""

            if pipeline:
                for zone in pipeline.camera_config.zones:
                    type_badge = (
                        '<span class="badge badge-info">检测区</span>'
                        if zone.zone_type == "include"
                        else '<span class="badge badge-medium">排除区</span>'
                    )
                    priority_labels = {"critical": "关键", "standard": "标准", "low_priority": "低"}
                    pri_text = priority_labels.get(zone.priority.value, zone.priority.value)
                    pts = ", ".join(f"({x},{y})" for x, y in zone.polygon[:3])
                    if len(zone.polygon) > 3:
                        pts += "..."
                    zones_list += f"""
                    <tr>
                        <td>{zone.zone_id}</td>
                        <td>{zone.name}</td>
                        <td>{type_badge}</td>
                        <td>{pri_text}</td>
                        <td style="font-size:11px;color:#8890a0;">{pts}</td>
                        <td>
                            <button class="btn btn-sm btn-danger"
                                hx-delete="/api/zones/{cam_id}/{zone.zone_id}"
                                hx-target="#zones-content" hx-swap="innerHTML"
                                hx-confirm="确定删除区域 {zone.name}？">删除</button>
                        </td>
                    </tr>"""

            if not zones_list:
                zones_list = '<tr><td colspan="6" style="color:#616161;">暂无检测区域</td></tr>'

            cameras_html += f"""
            <div class="card" style="margin-bottom:16px;">
                <div class="flex-between mb-8">
                    <h3 style="margin:0;">{cam_id} — {status.name}</h3>
                    <button class="btn btn-sm btn-primary"
                        onclick="openZoneEditor('{cam_id}')">绘制区域</button>
                </div>
                <table>
                    <thead><tr><th>区域ID</th><th>名称</th><th>类型</th><th>优先级</th><th>多边形</th><th>操作</th></tr></thead>
                    <tbody>{zones_list}</tbody>
                </table>
            </div>"""

    header = page_header("检测区域管理", "定义摄像头画面中的检测区域和排除区域")

    return HTMLResponse(f"""
    <div id="zones-content">
        {header}
        <p style="color:#8890a0;margin-bottom:16px;font-size:13px;">
            绘制 <span style="color:#4caf50;">检测区域</span> 限定异常检测范围。
            绘制 <span style="color:#f44336;">排除区域</span> 忽略特定区域（风扇、灯光等）。
        </p>
        {cameras_html if cameras_html else empty_state("暂无摄像头")}
    </div>

    <!-- Zone Editor Modal -->
    <div id="zone-editor-modal" class="modal-overlay"
         onclick="if(event.target===this)closeZoneEditor()">
        <div class="modal-content" style="max-width:900px;">
            <div class="flex-between mb-16">
                <h3 id="editor-title">绘制检测区域</h3>
                <button class="modal-close" onclick="closeZoneEditor()">&times;</button>
            </div>
            <canvas id="zone-canvas" width="640" height="480"
                style="border:1px solid #2a2d37;cursor:crosshair;display:block;margin:0 auto;"></canvas>
            <div class="flex gap-12 mt-16" style="align-items:center;flex-wrap:wrap;">
                <select id="zone-type" class="form-select" style="width:auto;"
                        onchange="redraw()">
                    <option value="exclude">排除区（红色）</option>
                    <option value="include">检测区（绿色）</option>
                </select>
                <select id="zone-priority" class="form-select" style="width:auto;">
                    <option value="standard">标准优先级</option>
                    <option value="critical">关键优先级 (1.2x)</option>
                    <option value="low_priority">低优先级 (0.8x)</option>
                </select>
                <input id="zone-name" class="form-input" placeholder="区域名称" style="width:auto;flex:1;">
                <button class="btn btn-primary" onclick="saveZone()">保存区域</button>
                <button class="btn btn-ghost" onclick="clearPoints()">清除</button>
            </div>
            <p id="editor-status" style="color:#8890a0;font-size:12px;margin-top:8px;">
                点击图片上的点来绘制多边形顶点，至少需要3个点。完成后点击「保存区域」。
            </p>
        </div>
    </div>""")


@router.post("")
async def create_zone(request: Request, zone_req: ZoneCreateRequest):
    """Add a new zone to a camera's pipeline."""
    from argus.config.schema import ZoneConfig, ZonePriority

    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return JSONResponse({"error": "摄像头管理器不可用"}, status_code=503)

    pipeline = camera_manager._pipelines.get(zone_req.camera_id)
    if not pipeline:
        return JSONResponse({"error": f"摄像头 {zone_req.camera_id} 不存在"}, status_code=404)

    priority_map = {
        "critical": ZonePriority.CRITICAL,
        "standard": ZonePriority.STANDARD,
        "low_priority": ZonePriority.LOW_PRIORITY,
    }

    new_zone = ZoneConfig(
        zone_id=zone_req.zone_id,
        name=zone_req.name,
        polygon=zone_req.polygon,
        zone_type=zone_req.zone_type,
        priority=priority_map.get(zone_req.priority, ZonePriority.STANDARD),
        anomaly_threshold=zone_req.anomaly_threshold,
    )

    current_zones = list(pipeline.camera_config.zones)
    current_zones.append(new_zone)
    pipeline.update_zones(current_zones)

    audit = getattr(request.app.state, "audit_logger", None)
    client_ip = request.client.host if request.client else ""
    if audit:
        audit.log(
            user="operator",
            action="update_zone",
            target_type="zone",
            target_id=f"{zone_req.camera_id}/{zone_req.zone_id}",
            detail=f"添加区域 {zone_req.name}",
            ip_address=client_ip,
        )

    return JSONResponse(
        {"status": "ok", "zone_id": zone_req.zone_id},
        headers=htmx_toast_headers("区域已添加"),
    )


@router.delete("/{camera_id}/{zone_id}")
async def delete_zone(request: Request, camera_id: str, zone_id: str):
    """Remove a zone from a camera's pipeline."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return JSONResponse({"error": "摄像头管理器不可用"}, status_code=503)

    pipeline = camera_manager._pipelines.get(camera_id)
    if not pipeline:
        return JSONResponse({"error": f"摄像头 {camera_id} 不存在"}, status_code=404)

    new_zones = [z for z in pipeline.camera_config.zones if z.zone_id != zone_id]
    pipeline.update_zones(new_zones)

    audit = getattr(request.app.state, "audit_logger", None)
    client_ip = request.client.host if request.client else ""
    if audit:
        audit.log(
            user="operator",
            action="update_zone",
            target_type="zone",
            target_id=f"{camera_id}/{zone_id}",
            detail="删除区域",
            ip_address=client_ip,
        )

    return JSONResponse(
        {"status": "ok"},
        headers=htmx_toast_headers("区域已删除"),
    )


@router.get("/snapshot/{camera_id}")
def camera_snapshot(request: Request, camera_id: str):
    """Get a JPEG snapshot from a camera for the zone editor."""
    import cv2

    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return JSONResponse({"error": "摄像头管理器不可用"}, status_code=503)

    pipeline = camera_manager._pipelines.get(camera_id)
    if not pipeline or not pipeline._camera.state.connected:
        return JSONResponse({"error": "摄像头未连接"}, status_code=503)

    frame_data = pipeline._camera.read()
    if frame_data is None:
        return JSONResponse({"error": "无法获取画面"}, status_code=503)

    _, jpeg = cv2.imencode(".jpg", frame_data.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")
