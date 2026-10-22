"""Tests for alert evidence fields in dashboard alert routes."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from argus.dashboard.app import create_app
from argus.storage.database import Database


@pytest.fixture
def db(tmp_path):
    database = Database(database_url=f"sqlite:///{tmp_path / 'argus.db'}")
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def client(db, tmp_path):
    alerts_dir = tmp_path / "alerts"
    alerts_dir.mkdir()
    app = create_app(database=db, alerts_dir=str(alerts_dir))
    return TestClient(app)


def _seed_alert(db: Database) -> None:
    db.save_alert(
        alert_id="ALT-route-evidence",
        timestamp=datetime.now(timezone.utc),
        camera_id="cam-01",
        zone_id="pump-room",
        severity="medium",
        anomaly_score=0.86,
        detection_type="water_leak",
        evidence_summary="water_leak, severity=medium, score=0.8600",
        evidence_json={
            "detection_type": "water_leak",
            "score": 0.86,
            "frame_number": 17,
        },
        suppression_reason=None,
        model_version_id="model-prod-1",
    )


def test_alerts_json_includes_evidence_fields(client: TestClient, db: Database) -> None:
    _seed_alert(db)

    response = client.get("/api/alerts/json")
    assert response.status_code == 200
    alert = response.json()["data"]["alerts"][0]

    assert alert["detection_type"] == "water_leak"
    assert alert["evidence_summary"] == "water_leak, severity=medium, score=0.8600"
    assert alert["evidence_json"]["frame_number"] == 17
    assert alert["suppression_reason"] is None
    assert alert["model_version_id"] == "model-prod-1"


def test_alert_detail_includes_evidence_fields(client: TestClient, db: Database) -> None:
    _seed_alert(db)

    response = client.get("/api/alerts/ALT-route-evidence/detail")
    assert response.status_code == 200
    detail = response.json()["data"]

    assert detail["detection_type"] == "water_leak"
    assert detail["evidence_json"]["detection_type"] == "water_leak"
    assert detail["evidence_json"]["frame_number"] == 17
    assert detail["model_version_id"] == "model-prod-1"
