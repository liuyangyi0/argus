"""PreviewGateway contract tests for camera preview migration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from starlette.requests import Request

from argus.config.schema import CameraConfig
from argus.streaming.preview_gateway import PreviewGateway


class FakeGo2RTC:
    def __init__(
        self,
        *,
        running: bool = True,
        api_port: int = 11984,
        rtsp_url: str | None = "rtsp://127.0.0.1:8554/cam_01",
    ) -> None:
        self.running = running
        self.api_port = api_port
        self.rtsp_url = rtsp_url
        self.register_calls: list[tuple[str, str, str]] = []

    def register_camera(self, camera_id: str, source: str, protocol: str) -> str | None:
        self.register_calls.append((camera_id, source, protocol))
        return self.rtsp_url


class FakeCameraManager:
    def __init__(self, frame) -> None:
        self.frame = frame

    def get_latest_frame(self, camera_id: str):
        return self.frame


class FakeStreamRegistry:
    def __init__(self) -> None:
        self.manager = SimpleNamespace(running=True, api_port=11984)
        self.calls: list[tuple[str, bool]] = []

    def ensure_registered(self, camera, *, start_if_needed: bool = False):
        self.calls.append((camera.camera_id, start_if_needed))
        return SimpleNamespace(
            camera_id=camera.camera_id,
            stream_name="planned_cam",
            go2rtc_managed=True,
        )


def _request(state, *, host: str = "edge.local:8080", scheme: str = "http") -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/streaming/cam_01",
            "scheme": scheme,
            "headers": [(b"host", host.encode("ascii"))],
            "server": ("edge.local", 8080),
            "client": ("testclient", 50000),
            "app": SimpleNamespace(state=state),
        }
    )


def _camera() -> CameraConfig:
    return CameraConfig(
        camera_id="cam_01",
        name="Camera 01",
        source="rtsp://camera.local/stream",
        protocol="rtsp",
    )


def test_snapshot_contract_returns_raw_jpeg_with_cache_headers():
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    state = SimpleNamespace(camera_manager=FakeCameraManager(frame))

    response = PreviewGateway(state).snapshot_response("cam_01")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-cache, no-store"
    assert response.body.startswith(b"\xff\xd8")


def test_snapshot_contract_returns_404_when_latest_frame_missing():
    state = SimpleNamespace(camera_manager=FakeCameraManager(None))

    response = PreviewGateway(state).snapshot_response("missing")

    assert response.status_code == 404
    assert "cache-control" not in response.headers


def test_snapshot_contract_returns_503_when_camera_manager_missing():
    response = PreviewGateway(SimpleNamespace()).snapshot_response("cam_01")

    assert response.status_code == 503
    assert "cache-control" not in response.headers


def test_streaming_info_contract_falls_back_when_go2rtc_is_unavailable():
    camera = _camera()
    go2rtc = FakeGo2RTC(running=False)
    state = SimpleNamespace(go2rtc=go2rtc)
    request = _request(state)

    info = PreviewGateway(state).streaming_info(request, camera)

    assert info == {
        "camera_id": "cam_01",
        "go2rtc": False,
        "fallback": "/api/cameras/cam_01/stream",
    }
    assert go2rtc.register_calls == []


def test_streaming_info_contract_returns_go2rtc_urls():
    camera = _camera()
    go2rtc = FakeGo2RTC(api_port=11984)
    state = SimpleNamespace(go2rtc=go2rtc)
    request = _request(state, host="edge.local:8080", scheme="https")

    info = PreviewGateway(state).streaming_info(request, camera)

    assert info == {
        "camera_id": "cam_01",
        "go2rtc": True,
        "webrtc_ws": "wss://edge.local:11984/api/ws?src=cam_01",
        "mse_ws": "wss://edge.local:11984/api/ws?src=cam_01&mode=mse",
        "hls": "https://edge.local:11984/api/stream.m3u8?src=cam_01",
        "mjpeg": "https://edge.local:11984/api/frame.jpeg?src=cam_01",
        "player": "https://edge.local:11984/stream.html?src=cam_01",
        "fallback": "/api/cameras/cam_01/stream",
    }
    assert go2rtc.register_calls == [
        ("cam_01", "rtsp://camera.local/stream", "rtsp"),
    ]


def test_streaming_info_contract_prefers_app_state_stream_registry():
    camera = _camera()
    registry = FakeStreamRegistry()
    go2rtc = FakeGo2RTC(api_port=11984)
    state = SimpleNamespace(stream_registry=registry, go2rtc=go2rtc)
    request = _request(state, host="edge.local:8080")

    info = PreviewGateway(state).streaming_info(request, camera)

    assert info["go2rtc"] is True
    assert info["webrtc_ws"] == "ws://edge.local:11984/api/ws?src=planned_cam"
    assert info["hls"] == "http://edge.local:11984/api/stream.m3u8?src=planned_cam"
    assert registry.calls == [("cam_01", False)]
    assert go2rtc.register_calls == []
