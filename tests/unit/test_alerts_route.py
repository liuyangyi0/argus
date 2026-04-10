"""Tests for the alert JSON API routes (/api/alerts/*)."""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from argus.core.health import HealthMonitor
from argus.dashboard.app import create_app
from argus.storage.database import Database
from argus.storage.models import AlertRecord


# ── Fixtures (mirrors test_dashboard.py pattern) ──


@pytest.fixture
def db(tmp_path):
    database = Database(database_url=f"sqlite:///{tmp_path / 'test.db'}")
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def health():
    monitor = HealthMonitor()
    monitor.update_camera("cam_01", connected=True, frames_captured=500, avg_latency_ms=15.3)
    return monitor


@pytest.fixture
def alerts_dir(tmp_path):
    d = tmp_path / "alerts"
    d.mkdir()
    return d


@pytest.fixture
def client(db, health, alerts_dir):
    app = create_app(database=db, health_monitor=health, alerts_dir=str(alerts_dir))
    return TestClient(app)


def _insert_alert(db: Database, alert_id: str = "test-alert-001", **overrides):
    """Helper to insert a test alert record."""
    defaults = {
        "alert_id": alert_id,
        "camera_id": "cam_01",
        "timestamp": datetime(2026, 4, 10, 12, 0, 0),
        "zone_id": "zone_a",
        "severity": "high",
        "anomaly_score": 0.95,
        "snapshot_path": "test.jpg",
        "heatmap_path": "test_hm.jpg",
        "workflow_status": "new",
    }
    defaults.update(overrides)
    with db.get_session() as session:
        alert = AlertRecord(**defaults)
        session.add(alert)
        session.commit()


# ── GET /api/alerts/json ──


class TestAlertsJsonEndpoint:
    def test_get_alerts_json_empty(self, client):
        """No alerts in DB should return empty list."""
        resp = client.get("/api/alerts/json")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["alerts"] == []
        assert body["data"]["total"] == 0

    def test_get_alerts_json_with_data(self, client, db):
        """Inserted alert should appear in JSON response."""
        _insert_alert(db, "alert-abc")
        resp = client.get("/api/alerts/json")
        assert resp.status_code == 200
        body = resp.json()
        alerts = body["data"]["alerts"]
        assert len(alerts) == 1
        assert alerts[0]["alert_id"] == "alert-abc"
        assert alerts[0]["severity"] == "high"
        assert body["data"]["total"] == 1

    def test_get_alerts_filter_by_camera(self, client, db):
        """Filter by camera_id should return only matching alerts."""
        _insert_alert(db, "a1", camera_id="cam_01")
        _insert_alert(db, "a2", camera_id="cam_02")
        resp = client.get("/api/alerts/json?camera_id=cam_01")
        body = resp.json()
        alerts = body["data"]["alerts"]
        assert len(alerts) == 1
        assert alerts[0]["camera_id"] == "cam_01"

    def test_get_alerts_filter_by_severity(self, client, db):
        """Filter by severity should return only matching alerts."""
        _insert_alert(db, "s1", severity="high")
        _insert_alert(db, "s2", severity="low")
        _insert_alert(db, "s3", severity="high")
        resp = client.get("/api/alerts/json?severity=low")
        body = resp.json()
        alerts = body["data"]["alerts"]
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "low"

    def test_get_alerts_json_limit(self, client, db):
        """Limit parameter should cap the number of returned alerts."""
        for i in range(5):
            _insert_alert(db, f"lim-{i}")
        resp = client.get("/api/alerts/json?limit=2")
        body = resp.json()
        assert len(body["data"]["alerts"]) == 2
        # total should still reflect all alerts
        assert body["data"]["total"] == 5


# ── POST /api/alerts/{alert_id}/acknowledge ──


class TestAcknowledgeAlert:
    def test_acknowledge_alert(self, client, db):
        """Acknowledging an existing alert should succeed."""
        _insert_alert(db, "ack-001")
        resp = client.post("/api/alerts/ack-001/acknowledge")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["alert_id"] == "ack-001"

        # Verify in DB
        alerts = db.get_alerts()
        assert alerts[0].acknowledged is True

    def test_acknowledge_nonexistent_alert(self, client):
        """Acknowledging a non-existent alert should return 404."""
        resp = client.post("/api/alerts/no-such-alert/acknowledge")
        assert resp.status_code == 404
        body = resp.json()
        assert body["code"] == 40400


# ── POST /api/alerts/bulk-acknowledge ──


class TestBulkAcknowledge:
    def test_bulk_acknowledge(self, client, db):
        """Bulk acknowledge should mark multiple alerts."""
        _insert_alert(db, "bulk-1")
        _insert_alert(db, "bulk-2")
        _insert_alert(db, "bulk-3")
        resp = client.post(
            "/api/alerts/bulk-acknowledge",
            json={"alert_ids": ["bulk-1", "bulk-2"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["count"] == 2

        # Verify third alert is still unacknowledged
        alerts = db.get_alerts()
        ack_map = {a.alert_id: a.acknowledged for a in alerts}
        assert ack_map["bulk-1"] is True
        assert ack_map["bulk-2"] is True
        assert ack_map["bulk-3"] is False

    def test_bulk_acknowledge_with_invalid_ids(self, client, db):
        """Bulk acknowledge should skip non-existent IDs gracefully."""
        _insert_alert(db, "exists-1")
        resp = client.post(
            "/api/alerts/bulk-acknowledge",
            json={"alert_ids": ["exists-1", "does-not-exist"]},
        )
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["count"] == 1


# ── DELETE /api/alerts/{alert_id} ──


class TestDeleteAlert:
    def test_delete_alert(self, client, db):
        """Deleting an existing alert should remove it."""
        _insert_alert(db, "del-001")
        resp = client.delete("/api/alerts/del-001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["alert_id"] == "del-001"

        # Verify it's gone
        assert db.get_alert_count() == 0

    def test_delete_nonexistent_alert(self, client):
        """Deleting a non-existent alert should return 404."""
        resp = client.delete("/api/alerts/no-such-alert")
        assert resp.status_code == 404
        body = resp.json()
        assert body["code"] == 40400


# ── GET /api/alerts/timeline ──


class TestTimeline:
    def test_timeline_returns_data(self, client, db):
        """Timeline endpoint should return 200 for a valid date with alerts."""
        _insert_alert(db, "tl-1", timestamp=datetime(2026, 4, 10, 8, 30, 0))
        resp = client.get("/api/alerts/timeline?date=2026-04-10")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]["cameras"]) >= 1

    def test_timeline_invalid_date(self, client):
        """Invalid date format should return 400."""
        resp = client.get("/api/alerts/timeline?date=not-a-date")
        assert resp.status_code == 400
        body = resp.json()
        assert body["code"] == 40000

    def test_timeline_empty_day(self, client):
        """Timeline with no alerts on that day should still return 200."""
        resp = client.get("/api/alerts/timeline?date=2020-01-01")
        assert resp.status_code == 200


# ── POST /api/alerts/{alert_id}/false-positive ──


class TestFalsePositive:
    def test_mark_false_positive(self, client, db):
        """Marking an alert as false positive should succeed."""
        _insert_alert(db, "fp-001")
        resp = client.post("/api/alerts/fp-001/false-positive")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0

        # Verify in DB
        alerts = db.get_alerts()
        assert alerts[0].false_positive is True

    def test_mark_false_positive_nonexistent(self, client):
        """Marking a non-existent alert as false positive should return 404."""
        resp = client.post("/api/alerts/nonexistent/false-positive")
        assert resp.status_code == 404


# ── POST /api/alerts/bulk-delete ──


class TestBulkDelete:
    def test_bulk_delete(self, client, db):
        """Bulk delete should remove multiple alerts."""
        _insert_alert(db, "bd-1")
        _insert_alert(db, "bd-2")
        _insert_alert(db, "bd-3")
        resp = client.post(
            "/api/alerts/bulk-delete",
            json={"alert_ids": ["bd-1", "bd-3"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["count"] == 2
        assert db.get_alert_count() == 1
