"""Camera API flow tests for local file-video development sources."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import asyncio

import pytest
from fastapi.testclient import TestClient

from argus.capture.manager import CameraManager
from argus.config.loader import load_config
from argus.config.schema import AlertConfig, ArgusConfig, CameraConfig
from argus.dashboard.app import create_app
from argus.streaming.preview_gateway import PreviewGateway
from scripts.create_dev_video import create_dev_video


def _wait_until(label: str, predicate, *, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.2)
    raise AssertionError(f"Timed out waiting for {label}")


def test_file_camera_can_be_added_started_and_previewed_through_api(tmp_path):
    """A local dev video should exercise the same Cameras API path as the UI."""

    video_path = tmp_path / "dev_camera.avi"
    create_dev_video(
        video_path,
        width=160,
        height=120,
        fps=5,
        seconds=3,
        anomaly_start_s=1.0,
    )

    config_path = tmp_path / "config.yaml"
    config = ArgusConfig(cameras=[])
    manager = CameraManager([], AlertConfig())
    app = create_app(
        camera_manager=manager,
        config=config,
        config_path=str(config_path),
        alerts_dir=str(tmp_path / "alerts"),
    )
    client = TestClient(app)

    try:
        add_response = client.post(
            "/api/cameras",
            data={
                "camera_id": "dev_file",
                "name": "Dev file camera",
                "source": str(video_path),
                "protocol": "file",
                "fps_target": "5",
                "resolution": "160,120",
            },
        )

        assert add_response.status_code == 200
        assert [camera.camera_id for camera in config.cameras] == ["dev_file"]
        added = config.cameras[0]
        assert added.protocol == "file"
        assert added.source == str(video_path)
        assert added.resolution == (160, 120)

        # Keep this contract free from network/model downloads; the purpose is
        # to prove camera add/start/preview, not model loading.
        added.person_filter.model_name = str(tmp_path / "missing-yolo.pt")

        saved_config = load_config(config_path)
        assert saved_config.cameras[0].protocol == "file"
        assert Path(saved_config.cameras[0].source) == video_path

        start_response = client.post("/api/cameras/dev_file/start")
        assert start_response.status_code == 200

        camera_row = _wait_until(
            "file camera to capture frames",
            lambda: next(
                (
                    row
                    for row in client.get("/api/cameras/json").json()["data"]["cameras"]
                    if row["camera_id"] == "dev_file"
                    and row["connected"]
                    and row["running"]
                    and row["stats"]["frames_captured"] >= 2
                ),
                None,
            ),
        )
        assert camera_row["stats"]["frames_captured"] >= 2

        snapshot = client.get("/api/cameras/dev_file/snapshot")
        assert snapshot.status_code == 200
        assert snapshot.headers["content-type"] == "image/jpeg"
        assert snapshot.content.startswith(b"\xff\xd8")
    finally:
        manager.stop_all()


@pytest.mark.asyncio
async def test_file_camera_latest_frame_stream_emits_mjpeg_frame(tmp_path):
    video_path = tmp_path / "dev_camera.avi"
    create_dev_video(
        video_path,
        width=160,
        height=120,
        fps=5,
        seconds=3,
        anomaly_start_s=1.0,
    )

    camera = CameraConfig(
        camera_id="dev_file",
        name="Dev file camera",
        source=str(video_path),
        protocol="file",
        fps_target=5,
        resolution=(160, 120),
    )
    camera.person_filter.model_name = str(tmp_path / "missing-yolo.pt")
    manager = CameraManager([camera], AlertConfig())
    stream_executor = ThreadPoolExecutor(max_workers=1)

    class _Request:
        def __init__(self) -> None:
            self.calls = 0

        async def is_disconnected(self) -> bool:
            self.calls += 1
            return self.calls > 1

    try:
        assert manager.start_camera("dev_file") is True
        _wait_until(
            "file camera latest frame",
            lambda: manager.get_latest_frame("dev_file") is not None,
        )

        gateway = PreviewGateway(
            SimpleNamespace(camera_manager=manager),
            stream_executor=stream_executor,
            stream_semaphore=asyncio.Semaphore(1),
            max_stream_duration=1.0,
        )
        response = gateway.latest_frame_stream_response(_Request(), "dev_file")
        chunk = await anext(response.body_iterator)

        assert response.media_type.startswith("multipart/x-mixed-replace")
        assert b"--frame\r\n" in chunk
        assert b"Content-Type: image/jpeg" in chunk
        assert b"\xff\xd8" in chunk
    finally:
        manager.stop_all()
        stream_executor.shutdown(wait=False, cancel_futures=True)


def test_connection_probe_preserves_saved_camera_protocol(tmp_path, monkeypatch):
    calls: list[tuple[object, float, str | None]] = []

    def fake_probe(source, timeout, protocol=None):
        calls.append((source, timeout, protocol))
        return {"ok": True, "source": "fake"}

    monkeypatch.setattr("argus.dashboard.routes.cameras._probe_source_blocking", fake_probe)

    camera = CameraConfig(
        camera_id="file_zero",
        name="File named zero",
        source="0",
        protocol="file",
    )
    config = ArgusConfig(cameras=[camera])
    manager = CameraManager([camera], AlertConfig())
    app = create_app(
        camera_manager=manager,
        config=config,
        alerts_dir=str(tmp_path / "alerts"),
    )
    client = TestClient(app)

    try:
        response = client.post("/api/cameras/file_zero/test-connection")
    finally:
        manager.stop_all()

    assert response.status_code == 200
    assert response.json()["data"]["ok"] is True
    assert calls == [("0", 5.0, "file")]


def test_draft_connection_probe_uses_selected_protocol(tmp_path, monkeypatch):
    calls: list[tuple[object, float, str | None]] = []

    def fake_probe(source, timeout, protocol=None):
        calls.append((source, timeout, protocol))
        return {"ok": True, "source": "fake"}

    monkeypatch.setattr("argus.dashboard.routes.cameras._probe_source_blocking", fake_probe)

    app = create_app(
        camera_manager=CameraManager([], AlertConfig()),
        config=ArgusConfig(cameras=[]),
        alerts_dir=str(tmp_path / "alerts"),
    )
    client = TestClient(app)

    response = client.post(
        "/api/cameras/test-connection-draft",
        json={"source": "0", "protocol": "file"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["ok"] is True
    assert calls == [("0", 5.0, "file")]
