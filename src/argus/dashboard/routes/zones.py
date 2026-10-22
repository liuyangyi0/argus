"""Zone management API routes (Chinese UI)."""

from __future__ import annotations

import logging
from fastapi import APIRouter, Request
from fastapi.responses import Response
from pydantic import BaseModel

from argus.dashboard.api_response import (
    api_forbidden,
    api_not_found,
    api_success,
    api_unavailable,
)
from argus.dashboard.auth import current_username, require_role
from argus.dashboard.forms import htmx_toast_headers

logger = logging.getLogger(__name__)
router = APIRouter()


def _deny_without_zone_edit(request: Request):
    if require_role(request, "admin", "engineer"):
        return None
    return api_forbidden("权限不足")


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
    denied = _deny_without_zone_edit(request)
    if denied:
        return denied

    from argus.config.schema import ZoneConfig, ZonePriority

    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("摄像头管理器不可用")

    get_zones = getattr(camera_manager, "get_camera_zones_snapshot", None)
    update_runtime_zones = getattr(camera_manager, "update_camera_zones", None)
    current_zones = (
        get_zones(zone_req.camera_id)
        if callable(get_zones)
        else None
    )
    if current_zones is None or not callable(update_runtime_zones):
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

    current_zones = list(current_zones)
    current_zones.append(new_zone)
    if update_runtime_zones(zone_req.camera_id, current_zones) is not True:
        return api_not_found(f"摄像头 {zone_req.camera_id} 不存在")
    _persist_zones(request, zone_req.camera_id, current_zones)

    audit = getattr(request.app.state, "audit_logger", None)
    client_ip = request.client.host if request.client else ""
    if audit:
        audit.log(
            user=current_username(request),
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
    denied = _deny_without_zone_edit(request)
    if denied:
        return denied

    from argus.config.schema import ZoneConfig, ZonePriority

    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("摄像头管理器不可用")

    update_runtime_zones = getattr(camera_manager, "update_camera_zones", None)
    if not callable(update_runtime_zones):
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

    if update_runtime_zones(camera_id, new_zones) is not True:
        return api_not_found(f"摄像头 {camera_id} 不存在")
    _persist_zones(request, camera_id, new_zones)

    audit = getattr(request.app.state, "audit_logger", None)
    client_ip = request.client.host if request.client else ""
    if audit:
        audit.log(
            user=current_username(request),
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
    denied = _deny_without_zone_edit(request)
    if denied:
        return denied

    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("摄像头管理器不可用")

    get_zones = getattr(camera_manager, "get_camera_zones_snapshot", None)
    update_runtime_zones = getattr(camera_manager, "update_camera_zones", None)
    current_zones = get_zones(camera_id) if callable(get_zones) else None
    if current_zones is None or not callable(update_runtime_zones):
        return api_not_found(f"摄像头 {camera_id} 不存在")

    new_zones = [z for z in current_zones if z.zone_id != zone_id]
    if update_runtime_zones(camera_id, new_zones) is not True:
        return api_not_found(f"摄像头 {camera_id} 不存在")
    _persist_zones(request, camera_id, new_zones)

    audit = getattr(request.app.state, "audit_logger", None)
    client_ip = request.client.host if request.client else ""
    if audit:
        audit.log(
            user=current_username(request),
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
    """Get the latest cached JPEG frame for the zone editor.

    The detection pipeline is the sole consumer of the capture adapter.  A
    second direct ``read()`` here can steal a frame, block the request thread,
    or race reconnect logic, so the editor only reads the manager's public
    latest-frame cache.
    """
    import cv2

    camera_manager = request.app.state.camera_manager
    if not camera_manager:
        return api_unavailable("摄像头管理器不可用")

    get_latest_frame = getattr(camera_manager, "get_latest_frame", None)
    if not callable(get_latest_frame):
        return api_unavailable("摄像头最新画面接口不可用")

    try:
        frame = get_latest_frame(camera_id)
    except Exception:
        logger.exception("zone_snapshot.latest_frame_failed camera_id=%s", camera_id)
        return api_unavailable("无法读取最新画面")
    if frame is None:
        return api_unavailable("暂无可用的最新画面")

    try:
        encoded, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )
    except (cv2.error, TypeError, ValueError):
        logger.exception("zone_snapshot.encode_failed camera_id=%s", camera_id)
        return api_unavailable("最新画面编码失败")
    if not encoded:
        return api_unavailable("最新画面编码失败")
    return Response(
        content=jpeg.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "private, no-store"},
    )
