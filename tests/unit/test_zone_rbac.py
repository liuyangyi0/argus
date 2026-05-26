from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from argus.config.schema import ArgusConfig, AuthConfig, CameraConfig
from argus.dashboard.app import create_app
from argus.dashboard.auth import create_session_token


class _Pipeline:
    def __init__(self) -> None:
        self.camera_config = SimpleNamespace(zones=[])
        self.updated_zones: list[object] | None = None

    def update_zones(self, zones: list[object]) -> None:
        self.updated_zones = zones
        self.camera_config.zones = zones


class _CameraManager:
    def __init__(self, pipeline: _Pipeline) -> None:
        self.pipeline = pipeline

    def get_pipeline(self, camera_id: str):
        return self.pipeline if camera_id == "cam_01" else None


def _client_for_role(role: str, tmp_path):
    camera = CameraConfig(
        camera_id="cam_01",
        name="Camera 01",
        source="rtsp://example.test/stream",
    )
    config = ArgusConfig(
        auth=AuthConfig(enabled=True, api_token="zone-rbac-token"),
        cameras=[camera],
    )
    pipeline = _Pipeline()
    app = create_app(
        camera_manager=_CameraManager(pipeline),
        config=config,
        alerts_dir=str(tmp_path / "alerts"),
    )
    client = TestClient(app)
    token = create_session_token(f"{role}-user", role, app.state.session_secret)
    client.cookies.set("argus_session", token)
    return client, pipeline


def _zone_payload():
    return [
        {
            "zone_id": "zone_a",
            "zone_type": "include",
            "vertices": [
                {"x": 0, "y": 0},
                {"x": 10, "y": 0},
                {"x": 10, "y": 10},
            ],
            "priority": "standard",
            "anomaly_threshold": 0.7,
        },
    ]


def test_operator_cannot_mutate_zones(tmp_path):
    client, pipeline = _client_for_role("operator", tmp_path)

    create_resp = client.post(
        "/api/zones",
        json={
            "camera_id": "cam_01",
            "zone_id": "zone_a",
            "name": "Zone A",
            "polygon": [[0, 0], [10, 0], [10, 10]],
        },
    )
    update_resp = client.put("/api/zones/cam_01", json=_zone_payload())
    delete_resp = client.delete("/api/zones/cam_01/zone_a")

    assert create_resp.status_code == 403
    assert update_resp.status_code == 403
    assert delete_resp.status_code == 403
    assert pipeline.updated_zones is None


def test_engineer_can_update_zones(tmp_path):
    client, pipeline = _client_for_role("engineer", tmp_path)

    response = client.put("/api/zones/cam_01", json=_zone_payload())

    assert response.status_code == 200
    assert response.json()["data"]["count"] == 1
    assert pipeline.updated_zones is not None
    assert pipeline.updated_zones[0].zone_id == "zone_a"
