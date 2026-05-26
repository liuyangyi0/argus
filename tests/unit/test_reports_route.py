"""Tests for reports routes."""

import csv
import io
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from argus.core.health import HealthMonitor
from argus.dashboard.app import create_app
from argus.storage.database import Database
from argus.storage.models import AlertRecordingRecord


@pytest.fixture
def db(tmp_path):
    """Create a temporary database for reports tests."""
    database = Database(database_url=f"sqlite:///{tmp_path / 'test.db'}")
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def client(db, tmp_path):
    alerts_dir = tmp_path / "alerts"
    alerts_dir.mkdir()
    app = create_app(
        database=db,
        health_monitor=HealthMonitor(),
        alerts_dir=str(alerts_dir),
    )
    return TestClient(app)


def _save_alert(
    db,
    alert_id: str,
    *,
    snapshot: bool = False,
    heatmap: bool = False,
    days_ago: int = 0,
    severity: str = "high",
    false_positive: bool = False,
):
    timestamp = datetime.now(tz=timezone.utc) - timedelta(days=days_ago)
    db.save_alert(
        alert_id=alert_id,
        timestamp=timestamp,
        camera_id="cam_01",
        zone_id="zone_a",
        severity=severity,
        anomaly_score=0.95,
        snapshot_path=f"/tmp/{alert_id}.jpg" if snapshot else None,
        heatmap_path=f"/tmp/{alert_id}_heatmap.jpg" if heatmap else None,
    )
    if false_positive:
        db.mark_false_positive(alert_id, notes="report test")
    return timestamp


def _save_recording(db, alert_id: str, trigger: datetime):
    trigger_ts = trigger.timestamp()
    db.save_alert_recording(
        AlertRecordingRecord(
            alert_id=alert_id,
            camera_id="cam_01",
            severity="high",
            recording_path=f"/tmp/{alert_id}.mp4",
            start_timestamp=trigger_ts - 5,
            end_timestamp=trigger_ts + 10,
            trigger_timestamp=trigger_ts,
            frame_count=150,
            fps=10,
            file_size_bytes=2048,
        )
    )


def test_reports_json_evidence_empty(client):
    response = client.get("/api/reports/json")

    assert response.status_code == 200
    evidence = response.json()["data"]["evidence"]
    assert evidence == {
        "total_alerts": 0,
        "alerts_with_snapshot": 0,
        "alerts_with_heatmap": 0,
        "alerts_with_recording": 0,
        "evidence_complete_count": 0,
        "snapshot_rate": 0,
        "heatmap_rate": 0,
        "recording_rate": 0,
        "evidence_complete_rate": 0,
    }


def test_reports_json_counts_snapshot_heatmap_and_recording(client, db):
    complete_at = _save_alert(db, "ALT-complete", snapshot=True, heatmap=True)
    _save_recording(db, "ALT-complete", complete_at)
    _save_alert(db, "ALT-snapshot-only", snapshot=True)
    recording_only_at = _save_alert(db, "ALT-recording-only")
    _save_recording(db, "ALT-recording-only", recording_only_at)
    no_heatmap_at = _save_alert(db, "ALT-snapshot-recording", snapshot=True)
    _save_recording(db, "ALT-snapshot-recording", no_heatmap_at)

    response = client.get("/api/reports/json")

    assert response.status_code == 200
    evidence = response.json()["data"]["evidence"]
    assert evidence["total_alerts"] == 4
    assert evidence["alerts_with_snapshot"] == 3
    assert evidence["alerts_with_heatmap"] == 1
    assert evidence["alerts_with_recording"] == 3
    assert evidence["evidence_complete_count"] == 1
    assert evidence["snapshot_rate"] == 75.0
    assert evidence["heatmap_rate"] == 25.0
    assert evidence["recording_rate"] == 75.0
    assert evidence["evidence_complete_rate"] == 25.0


def test_reports_json_days_scopes_stats_and_evidence(client, db):
    recent_at = _save_alert(db, "ALT-recent", snapshot=True, heatmap=True)
    _save_recording(db, "ALT-recent", recent_at)
    old_at = _save_alert(db, "ALT-old", snapshot=True, heatmap=True, days_ago=100)
    _save_recording(db, "ALT-old", old_at)

    response = client.get("/api/reports/json?days=30")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total_alerts"] == 1
    assert data["by_severity"] == {"high": 1, "medium": 0, "low": 0, "info": 0}
    assert data["evidence"]["total_alerts"] == 1
    assert data["evidence"]["alerts_with_snapshot"] == 1
    assert data["evidence"]["alerts_with_heatmap"] == 1
    assert data["evidence"]["alerts_with_recording"] == 1


def test_reports_distribution_days_scope(client, db):
    recent_at = _save_alert(db, "ALT-recent", snapshot=True)
    _save_recording(db, "ALT-recent", recent_at)
    _save_alert(db, "ALT-old", snapshot=True, days_ago=100)

    severity = client.get("/api/reports/severity-dist/json?days=30")
    camera = client.get("/api/reports/camera-dist/json?days=30")

    assert severity.status_code == 200
    assert severity.json()["data"] == {"high": 1, "medium": 0, "low": 0, "info": 0}
    assert camera.status_code == 200
    assert camera.json()["data"]["cameras"] == [{"camera_id": "cam_01", "count": 1}]


def test_reports_trends_scope_recent_alerts(client, db):
    _save_alert(db, "ALT-recent-fp", severity="medium", false_positive=True)
    _save_alert(db, "ALT-old-fp", severity="high", false_positive=True, days_ago=100)

    daily = client.get("/api/reports/daily-trend/json?days=7")
    fp = client.get("/api/reports/fp-trend/json?days=7")

    assert daily.status_code == 200
    daily_data = daily.json()["data"]
    assert len(daily_data["labels"]) == 7
    assert sum(daily_data["medium"]) == 1
    assert sum(daily_data["high"]) == 0

    assert fp.status_code == 200
    fp_data = fp.json()["data"]
    assert len(fp_data["labels"]) == 7
    assert len(fp_data["rates"]) == 7
    assert fp_data["rates"][-1] == 100.0


def test_compliance_csv_includes_period_evidence(client, db):
    recent_at = _save_alert(db, "ALT-recent", snapshot=True, heatmap=True)
    _save_recording(db, "ALT-recent", recent_at)
    old_at = _save_alert(db, "ALT-old", snapshot=True, days_ago=100)
    _save_recording(db, "ALT-old", old_at)

    response = client.get("/api/reports/compliance?days=30&format=csv")

    assert response.status_code == 200
    text = response.content.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    assert ["期间告警总数", "1"] in rows
    assert ["高", "1"] in rows
    assert "## 证据完整性" in text
    assert "触发截图,1,100.0" in text
    assert "热力图,1,100.0" in text
    assert "Replay录像,1,100.0" in text
    assert "完整证据,1,100.0" in text


def test_compliance_report_rejects_unknown_format(client):
    response = client.get("/api/reports/compliance?days=30&format=xlsx")

    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] != 0
    assert "csv" in payload["msg"]
    assert "pdf" in payload["msg"]
