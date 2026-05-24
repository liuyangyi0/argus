"""Dashboard streaming contract tests for camera input migration."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from fastapi.testclient import TestClient

from argus.config.schema import ArgusConfig, CameraConfig
from argus.dashboard.app import create_app


class _FakeGo2RTC:
    def __init__(
        self,
        *,
        running: bool = True,
        api_port: int = 11984,
        rtsp_url: str | None = "rtsp://127.0.0.1:8554/cam_01",
    ) -> None:
        self.running = running
        self.api_port = api_port
        self.register_camera = MagicMock(return_value=rtsp_url)
        self.close = MagicMock()


def _client(
    camera: CameraConfig,
    *,
    camera_manager: MagicMock | None = None,
    go2rtc: _FakeGo2RTC | None = None,
) -> TestClient:
    config = ArgusConfig(cameras=[camera])
    return TestClient(
        create_app(
            camera_manager=camera_manager,
            config=config,
            go2rtc_instance=go2rtc or _FakeGo2RTC(running=False),
        ),
    )


def _camera_manager(camera: CameraConfig) -> MagicMock:
    manager = MagicMock()
    manager._cameras = [camera]
    manager.get_status.return_value = []
    manager.get_runner_snapshot.return_value = None
    manager.get_detector_status.return_value = None
    manager.get_learning_progress.return_value = None
    manager.get_pipeline_mode.return_value = None
    manager.is_anomaly_locked.return_value = False
    return manager


def test_streaming_info_contract_returns_go2rtc_urls():
    camera = CameraConfig(
        camera_id="cam_01",
        name="Camera 01",
        source="rtsp://camera.local/stream",
        protocol="rtsp",
    )
    go2rtc = _FakeGo2RTC(
        api_port=11984,
        rtsp_url="rtsp://127.0.0.1:8554/cam_01",
    )
    client = _client(camera, go2rtc=go2rtc)

    response = client.get(
        "/api/streaming/cam_01",
        headers={"host": "edge.local:8080"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert body["data"] == {
        "camera_id": "cam_01",
        "go2rtc": True,
        "webrtc_ws": "ws://edge.local:11984/api/ws?src=cam_01",
        "mse_ws": "ws://edge.local:11984/api/ws?src=cam_01&mode=mse",
        "hls": "http://edge.local:11984/api/stream.m3u8?src=cam_01",
        "mjpeg": "http://edge.local:11984/api/frame.jpeg?src=cam_01",
        "player": "http://edge.local:11984/stream.html?src=cam_01",
        "fallback": "/api/cameras/cam_01/stream",
    }
    go2rtc.register_camera.assert_called_once_with(
        "cam_01",
        "rtsp://camera.local/stream",
        "rtsp",
    )


def test_streaming_info_contract_falls_back_when_go2rtc_unavailable():
    camera = CameraConfig(
        camera_id="cam_01",
        name="Camera 01",
        source="rtsp://camera.local/stream",
        protocol="rtsp",
    )
    go2rtc = _FakeGo2RTC(running=False)
    client = _client(camera, go2rtc=go2rtc)

    response = client.get("/api/streaming/cam_01")

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert body["data"] == {
        "camera_id": "cam_01",
        "go2rtc": False,
        "fallback": "/api/cameras/cam_01/stream",
    }
    go2rtc.register_camera.assert_not_called()


def test_streaming_info_contract_falls_back_when_go2rtc_registration_fails():
    camera = CameraConfig(
        camera_id="usb_cam",
        name="USB Camera",
        source="0",
        protocol="usb",
    )
    go2rtc = _FakeGo2RTC(rtsp_url=None)
    client = _client(camera, go2rtc=go2rtc)

    response = client.get("/api/streaming/usb_cam")

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert body["data"] == {
        "camera_id": "usb_cam",
        "go2rtc": False,
        "fallback": "/api/cameras/usb_cam/stream",
    }
    go2rtc.register_camera.assert_called_once_with("usb_cam", "0", "usb")


def test_usb_go2rtc_runtime_rtsp_does_not_leak_to_config_or_detail_api():
    runtime_rtsp = "rtsp://127.0.0.1:8554/usb_cam"
    camera = CameraConfig(
        camera_id="usb_cam",
        name="USB Camera",
        source="0",
        protocol="usb",
    )
    manager = _camera_manager(camera)
    go2rtc = _FakeGo2RTC(rtsp_url=runtime_rtsp)
    client = _client(camera, camera_manager=manager, go2rtc=go2rtc)

    stream_response = client.get("/api/streaming/usb_cam")
    config_response = client.get("/api/cameras/usb_cam/config")
    detail_response = client.get("/api/cameras/usb_cam/detail/json")

    assert stream_response.status_code == 200
    assert stream_response.json()["data"]["go2rtc"] is True
    assert config_response.status_code == 200
    assert detail_response.status_code == 200

    config_payload = config_response.json()["data"]
    detail_config = detail_response.json()["data"]["config"]
    assert config_payload["source"] == "0"
    assert config_payload["protocol"] == "usb"
    assert detail_config["source"] == "0"
    assert detail_config["protocol"] == "usb"
    assert runtime_rtsp not in json.dumps(config_payload)
    assert runtime_rtsp not in json.dumps(detail_config)


def test_running_usb_connection_probe_uses_latest_frame_not_physical_source(
    monkeypatch,
):
    camera = CameraConfig(
        camera_id="usb_cam",
        name="USB Camera",
        source="0",
        protocol="usb",
    )
    manager = _camera_manager(camera)
    manager.get_status.return_value = [
        SimpleNamespace(
            camera_id="usb_cam",
            name="USB Camera",
            connected=True,
            running=True,
            stats=None,
        ),
    ]
    manager.get_latest_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    def fail_probe(*_args, **_kwargs):
        raise AssertionError("physical USB source should not be reopened")

    monkeypatch.setattr("argus.dashboard.routes.cameras._probe_source_blocking", fail_probe)
    client = _client(camera, camera_manager=manager)

    response = client.post("/api/cameras/usb_cam/test-connection")

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert body["data"] == {
        "ok": True,
        "latency_ms": 0.0,
        "resolution": [640, 480],
        "source": "running_pipeline",
    }
