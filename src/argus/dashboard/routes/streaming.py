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
    api_validation_error,
)

router = APIRouter()


def _get_go2rtc(request: Request):
    return getattr(request.app.state, "go2rtc", None)


def _client_base_url(request: Request, port: int) -> str:
    """Build a base URL using the client-facing host and go2rtc port.

    The go2rtc URLs must be reachable by the browser, not just localhost.
    We take the host from the incoming request (which reflects the real
    server hostname / IP from the browser's perspective) and substitute
    the go2rtc port.
    """
    host = request.headers.get("host", "").split(":")[0] or "127.0.0.1"
    scheme = "https" if request.url.scheme == "https" else "http"
    return f"{scheme}://{host}:{port}"


@router.get("/{camera_id}")
def stream_info(request: Request, camera_id: str):
    """Return streaming URLs for a camera."""
    go2rtc = _get_go2rtc(request)

    if go2rtc is not None and go2rtc.running:
        base = _client_base_url(request, go2rtc.api_port)
        ws_scheme = "wss" if request.url.scheme == "https" else "ws"
        ws_base = f"{ws_scheme}://{request.headers.get('host', '').split(':')[0] or '127.0.0.1'}:{go2rtc.api_port}"

        return api_success({
            "camera_id": camera_id,
            "go2rtc": True,
            "webrtc_ws": f"{ws_base}/api/ws?src={camera_id}",
            "mse_ws": f"{ws_base}/api/ws?src={camera_id}&mode=mse",
            "hls": f"{base}/api/stream.m3u8?src={camera_id}",
            "mjpeg": f"{base}/api/frame.jpeg?src={camera_id}",
            "player": f"{base}/stream.html?src={camera_id}",
            "fallback": f"/api/cameras/{camera_id}/stream",
        })

    return api_success({
        "camera_id": camera_id,
        "go2rtc": False,
        "fallback": f"/api/cameras/{camera_id}/stream",
    })


@router.get("")
def streams_list(request: Request):
    """List all registered streams in go2rtc."""
    go2rtc = _get_go2rtc(request)
    if go2rtc is None or not go2rtc.running:
        return api_success({"go2rtc": False, "streams": {}})

    try:
        streams = go2rtc.list_streams()
    except Exception:
        return api_success({"go2rtc": False, "streams": {}})

    return api_success({"go2rtc": True, "streams": streams})


@router.post("/{camera_id}/register")
def register_stream(request: Request, camera_id: str):
    """Dynamically register a camera stream with go2rtc."""
    go2rtc = _get_go2rtc(request)
    if go2rtc is None or not go2rtc.running:
        return api_unavailable("go2rtc 未运行")

    camera_manager = getattr(request.app.state, "camera_manager", None)
    if not camera_manager:
        return api_unavailable("摄像头管理器不可用")

    cam_config = camera_manager.get_camera_config(camera_id)
    if cam_config is None:
        return api_not_found(f"摄像头 {camera_id} 不存在")

    if getattr(cam_config, "protocol", "rtsp") != "rtsp":
        return api_validation_error(
            f"摄像头 {camera_id} 使用协议 '{cam_config.protocol}'，非 RTSP"
        )

    go2rtc.add_stream(camera_id, cam_config.source)
    return api_success({"status": "ok", "camera_id": camera_id})
