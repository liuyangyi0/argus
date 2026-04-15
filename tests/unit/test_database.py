"""Tests for the database module."""

from datetime import datetime, timedelta, timezone

import pytest

from argus.storage.database import Database


@pytest.fixture
def db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"
    database = Database(database_url=f"sqlite:///{db_path}")
    database.initialize()
    yield database
    database.close()


class TestDatabase:
    def test_initialize_creates_tables(self, db):
        """Database should be initialized and usable."""
        assert db._engine is not None

    def test_save_and_retrieve_alert(self, db):
        """Should save and retrieve an alert."""
        db.save_alert(
            alert_id="ALT-000001",
            timestamp=datetime.now(tz=timezone.utc),
            camera_id="cam_01",
            zone_id="zone_a",
            severity="medium",
            anomaly_score=0.87,
        )

        alerts = db.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].alert_id == "ALT-000001"
        assert alerts[0].camera_id == "cam_01"
        assert alerts[0].severity == "medium"

    def test_filter_alerts_by_camera(self, db):
        """Should filter alerts by camera ID."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-1", now, "cam_01", "z1", "low", 0.75)
        db.save_alert("ALT-2", now, "cam_02", "z1", "high", 0.96)
        db.save_alert("ALT-3", now, "cam_01", "z2", "medium", 0.88)

        cam1_alerts = db.get_alerts(camera_id="cam_01")
        assert len(cam1_alerts) == 2

        cam2_alerts = db.get_alerts(camera_id="cam_02")
        assert len(cam2_alerts) == 1
        assert cam2_alerts[0].alert_id == "ALT-2"

    def test_filter_alerts_by_severity(self, db):
        """Should filter alerts by severity."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-1", now, "cam_01", "z1", "low", 0.75)
        db.save_alert("ALT-2", now, "cam_01", "z1", "high", 0.96)

        high_alerts = db.get_alerts(severity="high")
        assert len(high_alerts) == 1
        assert high_alerts[0].alert_id == "ALT-2"

    def test_acknowledge_alert(self, db):
        """Should mark an alert as acknowledged."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-1", now, "cam_01", "z1", "medium", 0.88)

        result = db.acknowledge_alert("ALT-1", "operator_wang")
        assert result is True

        alerts = db.get_alerts()
        assert alerts[0].acknowledged is True
        assert alerts[0].acknowledged_by == "operator_wang"

    def test_acknowledge_nonexistent_alert(self, db):
        """Should return False for nonexistent alert."""
        result = db.acknowledge_alert("ALT-NONE", "operator")
        assert result is False

    def test_mark_false_positive(self, db):
        """Should mark an alert as false positive."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-1", now, "cam_01", "z1", "low", 0.72)

        result = db.mark_false_positive("ALT-1", notes="Light reflection")
        assert result is True

        alerts = db.get_alerts()
        assert alerts[0].false_positive is True
        assert alerts[0].notes == "Light reflection"

    def test_get_alert_count(self, db):
        """Should return correct alert count."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-1", now, "cam_01", "z1", "low", 0.75)
        db.save_alert("ALT-2", now, "cam_01", "z1", "high", 0.96)
        db.save_alert("ALT-3", now, "cam_02", "z1", "low", 0.71)

        assert db.get_alert_count() == 3
        assert db.get_alert_count(camera_id="cam_01") == 2
        assert db.get_alert_count(severity="high") == 1

    def test_get_alert_count_since_filter(self, db):
        """``since`` should only count alerts at or after the given timestamp."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-OLD", now - timedelta(days=2), "cam_01", "z1", "low", 0.70)
        db.save_alert("ALT-NEW", now, "cam_01", "z1", "high", 0.95)

        cutoff = now - timedelta(hours=1)
        assert db.get_alert_count(since=cutoff) == 1
        assert db.get_alert_count(camera_id="cam_01", since=cutoff) == 1

    def test_get_wall_status_batch(self, db):
        """Batch query should return today's count + latest-if-active per camera in a single pass.

        Semantics match the old per-camera ``get_alerts(limit=1)`` pattern: the
        *latest* alert is looked up, then reported under ``active`` only if its
        workflow_status is still live. A resolved-latest hides older active
        alerts for that camera.
        """
        now = datetime.now(tz=timezone.utc)
        yesterday = now - timedelta(days=1)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # cam_01: latest is A-3 (active, high) → should be reported
        db.save_alert("A-1", yesterday, "cam_01", "z1", "low", 0.70)
        db.save_alert("A-2", now - timedelta(minutes=30), "cam_01", "z1", "medium", 0.85)
        db.save_alert("A-3", now, "cam_01", "z1", "high", 0.95)
        db.update_alert_workflow("A-2", "resolved")  # older than A-3, irrelevant

        # cam_02: only alert is false_positive → active is None
        db.save_alert("B-1", now, "cam_02", "z1", "low", 0.72)
        db.update_alert_workflow("B-1", "false_positive")

        # cam_03: only yesterday's alert, still active → reported even though count=0
        db.save_alert("C-1", yesterday, "cam_03", "z1", "low", 0.60)

        # cam_05: latest is resolved, earlier one is active → active is None (old semantic)
        db.save_alert("E-1", now - timedelta(minutes=30), "cam_05", "z1", "high", 0.95)
        db.save_alert("E-2", now, "cam_05", "z1", "low", 0.70)
        db.update_alert_workflow("E-2", "resolved")

        result = db.get_wall_status_batch(
            ["cam_01", "cam_02", "cam_03", "cam_04", "cam_05"],
            since=today_start,
        )

        assert set(result.keys()) == {"cam_01", "cam_02", "cam_03", "cam_04", "cam_05"}

        # cam_01: today count = 2 (A-2, A-3 both today), active alert = A-3
        assert result["cam_01"]["count"] == 2
        assert result["cam_01"]["active"] == {"alert_id": "A-3", "severity": "high"}

        # cam_02: today count = 1, but no active alert
        assert result["cam_02"]["count"] == 1
        assert result["cam_02"]["active"] is None

        # cam_03: nothing today, but C-1 (still active) is the latest overall
        assert result["cam_03"]["count"] == 0
        assert result["cam_03"]["active"] == {"alert_id": "C-1", "severity": "low"}

        # cam_04: unknown camera → zero/None
        assert result["cam_04"]["count"] == 0
        assert result["cam_04"]["active"] is None

        # cam_05: latest (E-2) is resolved, so active is None (hides older E-1)
        assert result["cam_05"]["count"] == 2
        assert result["cam_05"]["active"] is None

    def test_get_wall_status_batch_empty_camera_list(self, db):
        """Empty ``camera_ids`` should return an empty dict without querying."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("A-1", now, "cam_01", "z1", "low", 0.70)

        result = db.get_wall_status_batch([], since=now - timedelta(hours=1))
        assert result == {}

    def test_alert_to_dict(self, db):
        """AlertRecord.to_dict should produce a serializable dict."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-1", now, "cam_01", "z1", "medium", 0.85)

        alerts = db.get_alerts()
        d = alerts[0].to_dict()
        assert d["alert_id"] == "ALT-1"
        assert d["severity"] == "medium"
        assert isinstance(d["anomaly_score"], float)

    def test_pagination(self, db):
        """Should support offset and limit."""
        now = datetime.now(tz=timezone.utc)
        for i in range(10):
            db.save_alert(f"ALT-{i:03d}", now, "cam_01", "z1", "low", 0.72)

        page1 = db.get_alerts(limit=3, offset=0)
        assert len(page1) == 3

        page2 = db.get_alerts(limit=3, offset=3)
        assert len(page2) == 3

        # Different alerts
        assert page1[0].alert_id != page2[0].alert_id

    def test_get_alert_single(self, db):
        """Should retrieve a single alert by ID."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-001", now, "cam_01", "z1", "high", 0.96)

        alert = db.get_alert("ALT-001")
        assert alert is not None
        assert alert.alert_id == "ALT-001"

        # Non-existent
        assert db.get_alert("NOPE") is None

    def test_delete_old_alerts(self, db):
        """Should delete alerts older than N days."""
        old_time = datetime.now(tz=timezone.utc) - timedelta(days=100)
        recent_time = datetime.now(tz=timezone.utc)

        db.save_alert("ALT-OLD-1", old_time, "cam_01", "z1", "low", 0.72,
                       snapshot_path="/tmp/old1.jpg")
        db.save_alert("ALT-OLD-2", old_time, "cam_01", "z1", "medium", 0.88)
        db.save_alert("ALT-NEW", recent_time, "cam_01", "z1", "high", 0.96)

        count, paths = db.delete_old_alerts(days=90)
        assert count == 2
        assert "/tmp/old1.jpg" in paths

        # Only the recent alert should remain
        remaining = db.get_alerts()
        assert len(remaining) == 1
        assert remaining[0].alert_id == "ALT-NEW"

    def test_delete_old_alerts_none_to_delete(self, db):
        """Should handle case where no alerts are old enough."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-1", now, "cam_01", "z1", "low", 0.72)

        count, paths = db.delete_old_alerts(days=90)
        assert count == 0
        assert paths == []
        assert db.get_alert_count() == 1
