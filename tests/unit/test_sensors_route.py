"""Tests for the /api/sensors/* routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from argus.config.schema import ArgusConfig, SensorFusionConfig
from argus.dashboard.app import create_app


@pytest.fixture
def client(tmp_path) -> TestClient:
    """TestClient with fusion explicitly enabled."""
    config = ArgusConfig(
        sensor_fusion=SensorFusionConfig(enabled=True, default_valid_for_s=60.0),
    )
    app = create_app(
        alerts_dir=str(tmp_path / "alerts"),
        config=config,
    )
    return TestClient(app)


class TestSensorSignalRoute:
    def test_post_get_delete_flow(self, client: TestClient) -> None:
        """POST a signal → GET lists it → DELETE removes it."""
        # POST
        resp = client.post(
            "/api/sensors/signal",
            json={
                "camera_id": "cam1",
                "zone_id": "z1",
                "multiplier": 1.8,
                "valid_for_s": 30.0,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["applied"] is True

        # GET — should contain our signal
        resp = client.get("/api/sensors/signals")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["enabled"] is True
        signals = body["data"]["signals"]
        assert len(signals) == 1
        entry = signals[0]
        assert entry["camera_id"] == "cam1"
        assert entry["zone_id"] == "z1"
        assert entry["multiplier"] == pytest.approx(1.8)
        assert entry["remaining_s"] > 0

        # DELETE
        resp = client.request(
            "DELETE",
            "/api/sensors/signal",
            json={"camera_id": "cam1", "zone_id": "z1"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["removed"] is True

        # GET again — list should now be empty
        resp = client.get("/api/sensors/signals")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["signals"] == []

    def test_post_validation_error_on_bad_multiplier(self, client: TestClient) -> None:
        """multiplier out of [0.1, 5.0] → 400 validation error."""
        resp = client.post(
            "/api/sensors/signal",
            json={"camera_id": "cam1", "zone_id": "z1", "multiplier": 10.0},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["code"] == 40000
        assert "multiplier" in body["msg"].lower()

    def test_delete_nonexistent_returns_404(self, client: TestClient) -> None:
        resp = client.request(
            "DELETE",
            "/api/sensors/signal",
            json={"camera_id": "cam_ghost", "zone_id": "zone_ghost"},
        )
        assert resp.status_code == 404
        body = resp.json()
        assert body["code"] == 40400

    def test_wildcard_camera_signal(self, client: TestClient) -> None:
        """A ('*', '*') signal is listable and uses the wildcard key."""
        resp = client.post(
            "/api/sensors/signal",
            json={"camera_id": "*", "zone_id": "*", "multiplier": 2.0},
        )
        assert resp.status_code == 200
        resp = client.get("/api/sensors/signals")
        assert resp.status_code == 200
        entries = resp.json()["data"]["signals"]
        assert len(entries) == 1
        assert entries[0]["camera_id"] == "*"
        assert entries[0]["zone_id"] == "*"
        assert entries[0]["multiplier"] == pytest.approx(2.0)

    def test_default_valid_for_s_applied_when_omitted(self, client: TestClient) -> None:
        """Omitting valid_for_s uses the configured default (60s)."""
        resp = client.post(
            "/api/sensors/signal",
            json={"camera_id": "cam1", "zone_id": "z1", "multiplier": 1.2},
        )
        assert resp.status_code == 200
        resp = client.get("/api/sensors/signals")
        entries = resp.json()["data"]["signals"]
        assert len(entries) == 1
        # remaining_s should be close to default_valid_for_s (60) but less
        assert 0 < entries[0]["remaining_s"] <= 60.0
