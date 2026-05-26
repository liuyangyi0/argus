"""Dashboard streaming contract tests for camera input migration."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from fastapi.testclient import TestClient

from argus.config.schema import ArgusConfig, CameraConfig
from argus.dashboard.app import create_app
from argus.streaming.go2rtc_manager import gige_to_go2rtc_source, usb_to_go2rtc_source


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
        self.start = MagicMock(side_effect=self._start)
        self.close = MagicMock()

    def _start(self, *, initial_streams=None) -> None:
        self.running = True


class _RouteCameraManager:
    def __init__(self, camera: CameraConfig) -> None:
        self._cameras = [camera]
        self._source_resolver = None
        self.start_calls: list[str] = []
        self.resolved_sources: list[tuple[str, str]] = []

    def set_source_resolver(self, resolver):
        self._source_resolver = resolver

    def get_camera_config(self, camera_id: str):
        return next(
            (camera for camera in self._cameras if camera.camera_id == camera_id),
            None,
        )

    def start_camera(self, camera_id: str) -> bool:
        self.start_calls.append(camera_id)
        camera = self.get_camera_config(camera_id)
        if camera is not None and self._source_resolver is not None:
            runtime_camera = self._source_resolver(camera)
            self.resolved_sources.append(
                (runtime_camera.source, runtime_camera.protocol),
            )
        return True


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


def _expected_usb_source(camera: CameraConfig) -> str:
    return usb_to_go2rtc_source(
        camera.source,
        resolution=camera.resolution,
        fps=camera.fps_target,
        pixel_format=camera.usb.pixel_format,
    )


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


def test_file_streaming_info_uses_mjpeg_fallback_even_when_go2rtc_running():
    camera = CameraConfig(
        camera_id="file_cam",
        name="File Camera",
        source="data/dev/demo.avi",
        protocol="file",
    )
    go2rtc = _FakeGo2RTC(running=True)
    client = _client(camera, go2rtc=go2rtc)

    response = client.get("/api/streaming/file_cam")

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert body["data"] == {
        "camera_id": "file_cam",
        "go2rtc": False,
        "fallback": "/api/cameras/file_cam/stream",
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
    go2rtc.register_camera.assert_called_once_with(
        "usb_cam",
        _expected_usb_source(camera),
        "rtsp",
    )


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


def test_start_usb_camera_uses_go2rtc_runtime_source_without_mutating_config():
    runtime_rtsp = "rtsp://127.0.0.1:8554/usb_cam"
    camera = CameraConfig(
        camera_id="usb_cam",
        name="USB Camera",
        source="0",
        protocol="usb",
    )
    manager = _RouteCameraManager(camera)
    go2rtc = _FakeGo2RTC(rtsp_url=runtime_rtsp)
    client = _client(camera, camera_manager=manager, go2rtc=go2rtc)

    response = client.post("/api/cameras/usb_cam/start")

    assert response.status_code == 200
    go2rtc.register_camera.assert_called_once_with(
        "usb_cam",
        _expected_usb_source(camera),
        "rtsp",
    )
    assert manager.start_calls == ["usb_cam"]
    assert manager.resolved_sources == [(runtime_rtsp, "rtsp")]
    assert camera.source == "0"
    assert camera.protocol == "usb"


def test_start_usb_camera_falls_back_to_original_source_when_go2rtc_not_running():
    camera = CameraConfig(
        camera_id="usb_cam",
        name="USB Camera",
        source="0",
        protocol="usb",
    )
    manager = _RouteCameraManager(camera)
    go2rtc = _FakeGo2RTC(running=False)
    client = _client(camera, camera_manager=manager, go2rtc=go2rtc)

    response = client.post("/api/cameras/usb_cam/start")

    assert response.status_code == 200
    go2rtc.register_camera.assert_not_called()
    assert manager.start_calls == ["usb_cam"]
    assert manager.resolved_sources == [("0", "usb")]


def test_dashboard_lifespan_starts_go2rtc_before_usb_camera_start():
    runtime_rtsp = "rtsp://127.0.0.1:8554/usb_cam"
    camera = CameraConfig(
        camera_id="usb_cam",
        name="USB Camera",
        source="0",
        protocol="usb",
    )
    manager = _RouteCameraManager(camera)
    go2rtc = _FakeGo2RTC(running=False, rtsp_url=runtime_rtsp)
    config = ArgusConfig(cameras=[camera])

    with TestClient(
        create_app(
            camera_manager=manager,
            config=config,
            go2rtc_instance=go2rtc,
        )
    ) as client:
        response = client.post("/api/cameras/usb_cam/start")

    assert response.status_code == 200
    go2rtc.start.assert_called_once_with(initial_streams=None)
    go2rtc.register_camera.assert_called_once_with(
        "usb_cam",
        _expected_usb_source(camera),
        "rtsp",
    )
    assert manager.start_calls == ["usb_cam"]
    assert manager.resolved_sources == [(runtime_rtsp, "rtsp")]
    assert camera.source == "0"
    assert camera.protocol == "usb"


def test_dashboard_lifespan_preloads_gige_go2rtc_initial_stream():
    camera = CameraConfig(
        camera_id="gige_cam",
        name="GigE Camera",
        source="192.168.1.20",
        protocol="gige",
    )
    camera.gige.capture_script = "scripts/gige_capture.ps1"
    go2rtc = _FakeGo2RTC(
        running=False,
        rtsp_url="rtsp://127.0.0.1:8554/gige_cam",
    )
    config = ArgusConfig(cameras=[camera])

    with TestClient(create_app(config=config, go2rtc_instance=go2rtc)):
        pass

    go2rtc.start.assert_called_once_with(
        initial_streams={
            "gige_cam": gige_to_go2rtc_source("scripts/gige_capture.ps1"),
        },
    )
    go2rtc.register_camera.assert_called_once_with(
        "gige_cam",
        "192.168.1.20",
        "gige",
    )
