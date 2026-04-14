"""Zone management API routes (Chinese UI)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

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
