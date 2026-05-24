"""Streaming API — go2rtc proxy URLs for WebRTC/MSE/HLS camera streams.

These endpoints provide the frontend with go2rtc connection URLs so that
``<video>`` elements can negotiate WebRTC or MSE playback directly with
the go2rtc process, bypassing Python entirely for video delivery.

When go2rtc is unavailable (binary not found, process crashed), the
endpoints return a ``fallback`` field pointing to the legacy MJPEG stream
so the frontend can degrade gracefully.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from argus.dashboard.api_response import (
    api_success,
    api_not_found,
    api_unavailable,
)
from argus.streaming.preview_gateway import PreviewGateway

router = APIRouter()


def _preview_gateway(request: Request) -> PreviewGateway:
    return PreviewGateway.for_request(request)


def _find_camera_config(request: Request, camera_id: str):
    return _preview_gateway(request).find_camera_config(camera_id)


@router.get("/{camera_id}")
def stream_info(request: Request, camera_id: str):
    """Return streaming URLs for a camera."""
    gateway = _preview_gateway(request)
    cam_config = gateway.find_camera_config(camera_id)
    if cam_config is None:
        return api_not_found(f"摄像头 {camera_id} 不存在")

    return api_success(gateway.streaming_info(request, cam_config))


@router.get("")
def streams_list(request: Request):
    """List all registered streams in go2rtc."""
    go2rtc_running, streams = _preview_gateway(request).list_registered_streams()
    return api_success({"go2rtc": go2rtc_running, "streams": streams})


@router.post("/{camera_id}/register")
def register_stream(request: Request, camera_id: str):
    """Dynamically register a camera stream with go2rtc."""
    gateway = _preview_gateway(request)
    if not gateway.stream_manager_running():
        return api_unavailable("go2rtc 未运行")

    camera_manager = getattr(request.app.state, "camera_manager", None)
    if not camera_manager:
        return api_unavailable("摄像头管理器不可用")

    cam_config = _find_camera_config(request, camera_id)
    if cam_config is None:
        return api_not_found(f"摄像头 {camera_id} 不存在")

    resolution = gateway.ensure_stream_registered(cam_config)
    if resolution is None or not resolution.go2rtc_managed:
        return api_unavailable("该摄像头协议无法注册到 go2rtc")
    return api_success({
        "status": "ok",
        "camera_id": camera_id,
        "runtime_source": resolution.runtime_source,
        "runtime_protocol": resolution.runtime_protocol,
    })
