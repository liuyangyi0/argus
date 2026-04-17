"""Tests for the multi-camera storyboard endpoint (/api/replay/storyboard/{alert_id})."""

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from argus.core.health import HealthMonitor
from argus.dashboard.app import create_app
from argus.storage.database import Database
from argus.storage.models import AlertRecord


# ── Fixtures (mirror the test_alerts_route pattern) ──


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


def _insert(db: Database, **fields) -> str:
    """Insert one alert with sensible defaults and return its alert_id."""
    defaults = {
        "alert_id": "a-001",
        "camera_id": "cam_01",
        "zone_id": "zone_a",
        "severity": "high",
        "anomaly_score": 0.9,
        "timestamp": datetime(2026, 4, 10, 12, 0, 0),
        "workflow_status": "new",
    }
    defaults.update(fields)
    with db.get_session() as session:
        rec = AlertRecord(**defaults)
        session.add(rec)
        session.commit()
    return defaults["alert_id"]


# ── /api/replay/storyboard/{alert_id} ──


class TestStoryboardEndpoint:
    def test_not_found(self, client):
        resp = client.get("/api/replay/storyboard/does-not-exist")
        assert resp.status_code == 404
        body = resp.json()
        assert body["code"] == 40400

    def test_single_alert_returns_itself(self, client, db):
        """When there is no correlation partner and no event group, the
        storyboard collapses to a one-item list with offset 0."""
        _insert(db, alert_id="solo-001", camera_id="cam_01")
        resp = client.get("/api/replay/storyboard/solo-001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        data = body["data"]
        assert data["primary_alert_id"] == "solo-001"
        assert data["count"] == 1
        assert len(data["cameras"]) == 1
        cam = data["cameras"][0]
        assert cam["alert_id"] == "solo-001"
        assert cam["camera_id"] == "cam_01"
        assert cam["trigger_offset_s"] == 0.0
        assert cam["video_url"].endswith("/solo-001/video")
        assert cam["metadata_url"].endswith("/solo-001/metadata")
        assert cam["signals_url"].endswith("/solo-001/signals")

    def test_three_correlated_cameras(self, client, db):
        """A: cam1 (primary). B: cam2 (correlation_partner=cam1, +0.3s).
        C: cam3, same event_group_id. All three should be returned with
        offsets relative to A."""
        base = datetime(2026, 4, 10, 12, 0, 0)
        group = "grp-xyz"
        # A = primary
        _insert(
            db,
            alert_id="A",
            camera_id="cam1",
            timestamp=base,
            event_group_id=group,
            correlation_partner="cam2",
        )
        # B on cam2 at +0.3s, pointing back at cam1
        _insert(
            db,
            alert_id="B",
            camera_id="cam2",
            timestamp=base + timedelta(milliseconds=300),
            correlation_partner="cam1",
        )
        # C on cam3 in the same event group
        _insert(
            db,
            alert_id="C",
            camera_id="cam3",
            timestamp=base + timedelta(seconds=1),
            event_group_id=group,
        )

        resp = client.get("/api/replay/storyboard/A")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        data = body["data"]
        assert data["primary_alert_id"] == "A"
        assert data["count"] == 3
        # First entry must be the primary
        assert data["cameras"][0]["alert_id"] == "A"
        assert data["cameras"][0]["trigger_offset_s"] == pytest.approx(0.0)

        by_alert = {c["alert_id"]: c for c in data["cameras"]}
        assert set(by_alert.keys()) == {"A", "B", "C"}
        assert by_alert["B"]["camera_id"] == "cam2"
        assert by_alert["B"]["trigger_offset_s"] == pytest.approx(0.3, abs=1e-6)
        assert by_alert["C"]["camera_id"] == "cam3"
        assert by_alert["C"]["trigger_offset_s"] == pytest.approx(1.0, abs=1e-6)

    def test_correlation_partner_outside_window_is_ignored(self, client, db):
        base = datetime(2026, 4, 10, 12, 0, 0)
        _insert(db, alert_id="P", camera_id="cam1", timestamp=base, correlation_partner="cam2")
        # 10 seconds away — outside the ±5 s window
        _insert(db, alert_id="Q", camera_id="cam2", timestamp=base + timedelta(seconds=10))

        resp = client.get("/api/replay/storyboard/P")
        data = resp.json()["data"]
        assert data["count"] == 1
        assert data["cameras"][0]["alert_id"] == "P"

    def test_event_group_skips_same_camera(self, client, db):
        """Two alerts from the same camera in one event_group must not both
        appear — storyboard is about *different* cameras."""
        base = datetime(2026, 4, 10, 12, 0, 0)
        _insert(db, alert_id="P", camera_id="cam1", timestamp=base, event_group_id="g1")
        _insert(
            db,
            alert_id="P2",
            camera_id="cam1",
            timestamp=base + timedelta(seconds=0.5),
            event_group_id="g1",
        )
        _insert(
            db,
            alert_id="Q",
            camera_id="cam2",
            timestamp=base + timedelta(seconds=0.4),
            event_group_id="g1",
        )

        resp = client.get("/api/replay/storyboard/P")
        data = resp.json()["data"]
        ids = {c["alert_id"] for c in data["cameras"]}
        assert ids == {"P", "Q"}

    def test_caps_at_four_cameras(self, client, db):
        base = datetime(2026, 4, 10, 12, 0, 0)
        _insert(db, alert_id="P", camera_id="cam1", timestamp=base, event_group_id="g1")
        for i in range(2, 8):
            _insert(
                db,
                alert_id=f"C{i}",
                camera_id=f"cam{i}",
                timestamp=base + timedelta(milliseconds=100 * i),
                event_group_id="g1",
            )

        resp = client.get("/api/replay/storyboard/P")
        data = resp.json()["data"]
        assert data["count"] == 4
        assert len(data["cameras"]) == 4
