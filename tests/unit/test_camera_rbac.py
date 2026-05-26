from __future__ import annotations

from fastapi.testclient import TestClient

from argus.config.schema import ArgusConfig, AuthConfig, CameraConfig
from argus.core.pipeline import PipelineMode
from argus.dashboard.app import create_app
from argus.dashboard.auth import create_session_token


class _CameraManager:
    def __init__(self, cameras: list[CameraConfig]) -> None:
        self._cameras = list(cameras)
        self.added: list[str] = []
        self.removed: list[str] = []
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.modes: list[tuple[str, PipelineMode]] = []

    def add_camera_config(self, camera: CameraConfig) -> None:
        self.added.append(camera.camera_id)
        self._cameras.append(camera)

    def remove_camera_config(self, camera_id: str) -> None:
        self.removed.append(camera_id)
        self._cameras = [camera for camera in self._cameras if camera.camera_id != camera_id]

    def start_camera(self, camera_id: str) -> bool:
        self.started.append(camera_id)
        return True

    def stop_camera(self, camera_id: str) -> None:
        self.stopped.append(camera_id)

    def get_status(self) -> list[object]:
        return []

    def get_pipeline_mode(self, camera_id: str) -> PipelineMode:
        return PipelineMode.ACTIVE

    def set_pipeline_mode(self, camera_id: str, mode: PipelineMode) -> bool:
        self.modes.append((camera_id, mode))
        return True


def _client_for_role(role: str, tmp_path):
    camera = CameraConfig(
        camera_id="cam_01",
        name="Camera 01",
        source="rtsp://example.test/stream",
        protocol="rtsp",
    )
    config = ArgusConfig(
        auth=AuthConfig(enabled=True, api_token="camera-rbac-token"),
        cameras=[camera],
    )
    manager = _CameraManager([camera])
    app = create_app(
        camera_manager=manager,
        config=config,
        alerts_dir=str(tmp_path / "alerts"),
    )
    client = TestClient(app)
    token = create_session_token(f"{role}-user", role, app.state.session_secret)
    client.cookies.set("argus_session", token)
    return client, manager, config


def test_operator_cannot_change_camera_configuration(tmp_path):
    client, manager, config = _client_for_role("operator", tmp_path)

    add_resp = client.post(
        "/api/cameras",
        data={
            "camera_id": "cam_02",
            "name": "Camera 02",
            "source": "rtsp://example.test/stream2",
            "protocol": "rtsp",
        },
    )
    update_resp = client.put("/api/cameras/cam_01", data={"name": "Edited"})
    delete_resp = client.delete("/api/cameras/cam_01")

    assert add_resp.status_code == 403
    assert update_resp.status_code == 403
    assert delete_resp.status_code == 403
    assert [camera.camera_id for camera in config.cameras] == ["cam_01"]
    assert manager.added == []
    assert manager.removed == []


def test_engineer_can_add_camera_and_operate_pipeline(tmp_path):
    client, manager, config = _client_for_role("engineer", tmp_path)

    add_resp = client.post(
        "/api/cameras",
        data={
            "camera_id": "cam_02",
            "name": "Camera 02",
            "source": "rtsp://example.test/stream2",
            "protocol": "rtsp",
        },
    )
    start_resp = client.post("/api/cameras/cam_01/start")
    stop_resp = client.post("/api/cameras/cam_01/stop")
    mode_resp = client.post("/api/cameras/cam_01/mode", json={"mode": "maintenance"})

    assert add_resp.status_code == 200
    assert start_resp.status_code == 200
    assert stop_resp.status_code == 200
    assert mode_resp.status_code == 200
    assert [camera.camera_id for camera in config.cameras] == ["cam_01", "cam_02"]
    assert manager.added == ["cam_02"]
    assert manager.started == ["cam_01"]
    assert manager.stopped == ["cam_01"]
    assert manager.modes == [("cam_01", PipelineMode.MAINTENANCE)]


def test_engineer_cannot_delete_camera_configuration(tmp_path):
    client, manager, config = _client_for_role("engineer", tmp_path)

    response = client.delete("/api/cameras/cam_01")

    assert response.status_code == 403
    assert [camera.camera_id for camera in config.cameras] == ["cam_01"]
    assert manager.removed == []


def test_operator_can_operate_existing_camera(tmp_path):
    client, manager, _config = _client_for_role("operator", tmp_path)

    start_resp = client.post("/api/cameras/cam_01/start")
    stop_resp = client.post("/api/cameras/cam_01/stop")
    mode_resp = client.post("/api/cameras/cam_01/mode", json={"mode": "learning"})

    assert start_resp.status_code == 200
    assert stop_resp.status_code == 200
    assert mode_resp.status_code == 200
    assert manager.started == ["cam_01"]
    assert manager.stopped == ["cam_01"]
    assert manager.modes == [("cam_01", PipelineMode.LEARNING)]
