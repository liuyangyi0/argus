from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from argus.config.schema import ArgusConfig
from argus.core.health import HealthMonitor
from argus.dashboard.app import create_app
from argus.storage.database import Database


def _client(config: ArgusConfig, camera_manager: object | None, db, health, alerts_dir):
    app = create_app(
        database=db,
        camera_manager=camera_manager,
        health_monitor=health,
        alerts_dir=str(alerts_dir),
        config=config,
    )
    return TestClient(app)


@pytest.fixture
def db(tmp_path):
    database = Database(database_url=f"sqlite:///{tmp_path / 'config-api.db'}")
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def health():
    return HealthMonitor()


@pytest.fixture
def alerts_dir(tmp_path):
    path = tmp_path / "alerts"
    path.mkdir()
    return path


def test_detection_params_accept_json_payload(db, health, alerts_dir):
    config = ArgusConfig()
    config.cameras = [SimpleNamespace(camera_id="cam_01")]

    pipeline = MagicMock()
    pipeline.update_thresholds.return_value = {
        "anomaly_threshold": True,
        "severity": True,
        "temporal": True,
        "suppression": True,
    }
    camera_manager = SimpleNamespace(_pipelines={"cam_01": pipeline})
    client = _client(config, camera_manager, db, health, alerts_dir)

    response = client.post(
        "/api/config/detection-params",
        json={
            "anomaly_threshold": 0.42,
            "sev_info": 0.0,
            "sev_low": 0.35,
            "temp_gap": 1.5,
            "supp_zone": 3.0,
        },
    )

    assert response.status_code == 200
    assert config.alerts.severity_thresholds.info == 0.0
    assert config.alerts.severity_thresholds.low == 0.35
    assert config.alerts.temporal.max_gap_seconds == 1.5
    assert config.alerts.suppression.same_zone_window_seconds == 3.0
    pipeline.update_thresholds.assert_called_once_with(
        anomaly_threshold=0.42,
        severity_changed=True,
        temporal_changed=True,
        suppression_changed=True,
    )
    data = response.json()["data"]
    assert data["anomaly_threshold"]["hot_reloaded"] is True
    assert data["severity"]["changed"] is True


def test_notifications_accept_json_payload(db, health, alerts_dir):
    config = ArgusConfig()
    client = _client(config, None, db, health, alerts_dir)

    response = client.post(
        "/api/config/notifications",
        json={
            "webhook_enabled": True,
            "webhook_url": "https://example.test/hook",
            "webhook_timeout": 7,
        },
    )

    assert response.status_code == 200
    assert config.alerts.webhook.enabled is True
    assert config.alerts.webhook.url == "https://example.test/hook"
    assert config.alerts.webhook.timeout == 7

    response = client.post(
        "/api/config/notifications",
        json={"webhook_enabled": False},
    )

    assert response.status_code == 200
    assert config.alerts.webhook.enabled is False
