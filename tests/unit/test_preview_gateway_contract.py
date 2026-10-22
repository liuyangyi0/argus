"""PreviewGateway contract tests for camera preview migration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from starlette.requests import Request

from argus.config.schema import CameraConfig
from argus.core.frame_identity import FrameIdentity
from argus.streaming import preview_gateway
from argus.streaming.preview_gateway import PreviewGateway, probe_source_blocking


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
    def __init__(self, frame, identity: FrameIdentity | None = None) -> None:
        self.frame = frame
        self.identity = identity
        self.snapshot_ids: list[str] = []

    def get_latest_frame(self, camera_id: str):
        return self.frame

    def get_latest_frame_with_identity(self, camera_id: str):
        return self.frame, self.identity

    def record_snapshot_access(self, camera_id: str, snapshot_id: str):
        self.snapshot_ids.append(snapshot_id)
        if self.identity is None:
            return None
        self.identity = self.identity.with_updates(snapshot_id=snapshot_id)
        return self.identity


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


def test_probe_source_blocking_opens_usb_numeric_source_as_device_index(monkeypatch):
    calls = []

    class FakeCapture:
        def __init__(self, source, backend=None):
            calls.append((source, backend))

        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((12, 16, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(preview_gateway.cv2, "VideoCapture", FakeCapture)

    result = probe_source_blocking("0", protocol="usb")

    assert result["ok"] is True
    assert calls == [(0, preview_gateway.cv2.CAP_DSHOW)]


def test_probe_source_blocking_keeps_file_numeric_source_as_path(monkeypatch):
    calls = []

    class FakeCapture:
        def __init__(self, source, backend=None):
            calls.append((source, backend))

        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((12, 16, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(preview_gateway.cv2, "VideoCapture", FakeCapture)

    result = probe_source_blocking("0", protocol="file")

    assert result["ok"] is True
    assert calls == [("0", None)]


def test_snapshot_contract_returns_raw_jpeg_with_cache_headers():
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    state = SimpleNamespace(camera_manager=FakeCameraManager(frame))

    response = PreviewGateway(state).snapshot_response("cam_01")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-cache, no-store"
    assert response.body.startswith(b"\xff\xd8")


def test_snapshot_contract_exposes_frame_identity_headers():
    identity = FrameIdentity(
        camera_id="cam_01",
        source_declared="0",
        source_runtime="rtsp://127.0.0.1:8554/cam_01",
        stream_id="cam_01",
        frame_index=123,
        capture_wall_ts=1710000000.123,
        capture_monotonic_ts=55.0,
        pipeline_ingest_ts=1710000000.145,
    )
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    manager = FakeCameraManager(frame, identity)
    state = SimpleNamespace(camera_manager=manager)

    response = PreviewGateway(state).snapshot_response("cam_01")

    assert response.status_code == 200
    assert response.headers["x-argus-frame-index"] == "123"
    assert response.headers["x-argus-capture-ts"] == "1710000000.123000"
    assert response.headers["x-argus-pipeline-ts"] == "1710000000.145000"
    assert response.headers["x-argus-stream-id"] == "cam_01"
    assert response.headers["x-argus-snapshot-id"] == manager.snapshot_ids[0]
    assert response.headers["x-argus-runtime-source-hash"] == identity.runtime_source_hash


def test_snapshot_debug_overlay_draws_identity_when_enabled():
    identity = FrameIdentity(
        camera_id="cam_01",
        source_declared="0",
        source_runtime="rtsp://127.0.0.1:8554/cam_01",
        stream_id="cam_01",
        frame_index=123,
        capture_wall_ts=1710000000.123,
        capture_monotonic_ts=55.0,
        pipeline_ingest_ts=1710000000.145,
    )
    frame = np.zeros((96, 240, 3), dtype=np.uint8)
    manager = FakeCameraManager(frame, identity)
    state = SimpleNamespace(
        camera_manager=manager,
        config=SimpleNamespace(
            debug=SimpleNamespace(frame_identity_overlay=True),
        ),
    )

    response = PreviewGateway(state).snapshot_response("cam_01")
    decoded = preview_gateway.cv2.imdecode(
        np.frombuffer(response.body, dtype=np.uint8),
        preview_gateway.cv2.IMREAD_COLOR,
    )

    assert response.status_code == 200
    assert decoded is not None
    assert int(decoded.max()) > 0


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


def test_streaming_info_contract_uses_external_local_go2rtc_source():
    camera = CameraConfig(
        camera_id="cam_01",
        name="Camera 01",
        source="rtsp://127.0.0.1:8554/validation_camera_01",
        protocol="rtsp",
    )
    state = SimpleNamespace(
        config=SimpleNamespace(
            dashboard=SimpleNamespace(go2rtc_api_port=1984, go2rtc_rtsp_port=8554)
        )
    )
    request = _request(state, host="192.0.2.10:18080")

    info = PreviewGateway(state).streaming_info(request, camera)

    assert info == {
        "camera_id": "cam_01",
        "go2rtc": True,
        "webrtc_ws": "ws://192.0.2.10:18080/ws/stream/cam_01?mode=webrtc",
        "mse_ws": "ws://192.0.2.10:18080/ws/stream/cam_01?mode=mse",
        "fallback": "/api/cameras/cam_01/stream",
    }


def test_streaming_info_contract_uses_mse_for_external_go2rtc_over_loopback():
    camera = CameraConfig(
        camera_id="cam_01",
        name="Camera 01",
        source="rtsp://127.0.0.1:8554/validation_camera_01",
        protocol="rtsp",
    )
    state = SimpleNamespace(
        config=SimpleNamespace(
            dashboard=SimpleNamespace(go2rtc_api_port=1984, go2rtc_rtsp_port=8554)
        )
    )
    request = _request(state, host="127.0.0.1:18081")

    info = PreviewGateway(state).streaming_info(request, camera)

    assert info["go2rtc"] is True
    assert info["webrtc_ws"] == "ws://127.0.0.1:18081/ws/stream/cam_01?mode=webrtc"
    assert info["mse_ws"] == "ws://127.0.0.1:18081/ws/stream/cam_01?mode=mse"


def test_streaming_info_contract_returns_go2rtc_urls():
    camera = _camera()
    go2rtc = FakeGo2RTC(api_port=11984)
    state = SimpleNamespace(go2rtc=go2rtc)
    request = _request(state, host="edge.local:8080", scheme="https")

    info = PreviewGateway(state).streaming_info(request, camera)

    assert info == {
        "camera_id": "cam_01",
        "go2rtc": True,
        "webrtc_ws": "wss://edge.local:8080/ws/stream/cam_01?mode=webrtc",
        "mse_ws": "wss://edge.local:8080/ws/stream/cam_01?mode=mse",
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
    assert info["webrtc_ws"] == "ws://edge.local:8080/ws/stream/cam_01?mode=webrtc"
    assert "11984" not in info["webrtc_ws"]
    assert "planned_cam" not in info["webrtc_ws"]
    assert registry.calls == [("cam_01", False)]
    assert go2rtc.register_calls == []
