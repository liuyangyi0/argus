"""Zone management API routes (Chinese UI)."""

from __future__ import annotations

import logging
from fastapi import APIRouter, Request
from fastapi.responses import Response
from pydantic import BaseModel

from argus.dashboard.api_response import api_success, api_not_found, api_unavailable
from argus.dashboard.forms import htmx_toast_headers

logger = logging.getLogger(__name__)
router = APIRouter()


def _persist_zones(request: Request, camera_id: str, zones: list) -> None:
    """Persist zone changes for *camera_id* back to the YAML config file.

    Updates both the in-memory ``app.state.config`` and the file on disk.
    Silently skips when ``config_path`` is unavailable (e.g. tests).
    """
    config = getattr(request.app.state, "config", None)
    config_path = getattr(request.app.state, "config_path", None)
    if config is None or config_path is None:
        return

    # Update the matching camera in the in-memory config
    for cam in getattr(config, "cameras", []):
        if cam.camera_id == camera_id:
            cam.zones = list(zones)
            break

    # Write back to disk
    try:
        from argus.config.loader import save_config
        save_config(config, config_path)
    except Exception:
        logger.exception("zone_persist.failed camera_id=%s", camera_id)


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
        return api_unavailable("摄像头管理器不可用")

    pipeline = camera_manager.get_pipeline(zone_req.camera_id)
    if not pipeline:
        return api_not_found(f"摄像头 {zone_req.camera_id} 不存在")

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
    _persist_zones(request, zone_req.camera_id, current_zones)

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

    return api_success(
        {"zone_id": zone_req.zone_id},
        headers=htmx_toast_headers("区域已添加"),
    )


class ZoneBulkItem(BaseModel):
    zone_id: str
    zone_type: str = "include"
    vertices: list[dict]
    priority: str = "standard"
    anomaly_threshold: float = 0.7


@router.put("/{camera_id}")
async def update_zones(request: Request, camera_id: str, payload: list[ZoneBulkItem]):
    """Replace all zones for a camera with the provided list."""
    from argus.config.schema import ZoneConfig, ZonePriority

    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("摄像头管理器不可用")

    pipeline = camera_manager.get_pipeline(camera_id)
    if not pipeline:
        return api_not_found(f"摄像头 {camera_id} 不存在")

    priority_map = {
        "critical": ZonePriority.CRITICAL,
        "standard": ZonePriority.STANDARD,
        "low_priority": ZonePriority.LOW_PRIORITY,
    }

    new_zones = []
    for item in payload:
        polygon = [(v.get("x", 0), v.get("y", 0)) for v in item.vertices]
        new_zones.append(ZoneConfig(
            zone_id=item.zone_id,
            name=item.zone_id,
            polygon=polygon,
            zone_type=item.zone_type,
            priority=priority_map.get(item.priority, ZonePriority.STANDARD),
            anomaly_threshold=item.anomaly_threshold,
        ))

    pipeline.update_zones(new_zones)
    _persist_zones(request, camera_id, new_zones)

    audit = getattr(request.app.state, "audit_logger", None)
    client_ip = request.client.host if request.client else ""
    if audit:
        audit.log(
            user="operator",
            action="update_zone",
            target_type="zone",
            target_id=camera_id,
            detail=f"批量更新区域 ({len(new_zones)} 个)",
            ip_address=client_ip,
        )

    return api_success(
        {"count": len(new_zones)},
        headers=htmx_toast_headers("区域配置已保存"),
    )


@router.delete("/{camera_id}/{zone_id}")
async def delete_zone(request: Request, camera_id: str, zone_id: str):
    """Remove a zone from a camera's pipeline."""
    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("摄像头管理器不可用")

    pipeline = camera_manager.get_pipeline(camera_id)
    if not pipeline:
        return api_not_found(f"摄像头 {camera_id} 不存在")

    new_zones = [z for z in pipeline.camera_config.zones if z.zone_id != zone_id]
    pipeline.update_zones(new_zones)
    _persist_zones(request, camera_id, new_zones)

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

    return api_success(
        {"deleted": True},
        headers=htmx_toast_headers("区域已删除"),
    )


@router.get("/snapshot/{camera_id}")
def camera_snapshot(request: Request, camera_id: str):
    """Get a JPEG snapshot from a camera for the zone editor."""
    import cv2

    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("摄像头管理器不可用")

    pipeline = camera_manager.get_pipeline(camera_id)
    if not pipeline or not pipeline._camera.state.connected:
        return api_unavailable("摄像头未连接")

    frame_data = pipeline._camera.read()
    if frame_data is None:
        return api_unavailable("无法获取画面")

    _, jpeg = cv2.imencode(".jpg", frame_data.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")
