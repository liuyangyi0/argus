"""Tests for the alert dispatcher module."""

import time
from datetime import datetime, timezone

import numpy as np
import pytest

from argus.alerts.dispatcher import AlertDispatcher
from argus.alerts.grader import Alert
from argus.config.schema import AlertConfig, AlertSeverity
from argus.storage.database import Database


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    database = Database(database_url=f"sqlite:///{db_path}")
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def dispatcher(db, tmp_path):
    config = AlertConfig()
    return AlertDispatcher(config=config, database=db, alerts_dir=tmp_path / "alerts")


def make_alert(
    alert_id: str = "ALT-000001",
    severity: AlertSeverity = AlertSeverity.MEDIUM,
    score: float = 0.88,
    with_snapshot: bool = False,
    with_heatmap: bool = False,
) -> Alert:
    return Alert(
        alert_id=alert_id,
        camera_id="cam_01",
        zone_id="zone_a",
        severity=severity,
        anomaly_score=score,
        timestamp=time.monotonic(),
        frame_number=42,
        snapshot=np.zeros((100, 100, 3), dtype=np.uint8) if with_snapshot else None,
        heatmap=np.random.rand(100, 100).astype(np.float32) if with_heatmap else None,
    )


class TestAlertDispatcher:
    def test_dispatch_saves_to_database(self, dispatcher, db):
        """Dispatching an alert should persist it in the database."""
        alert = make_alert()
        dispatcher.dispatch(alert)

        alerts = db.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].alert_id == "ALT-000001"
        assert alerts[0].severity == "medium"

    def test_dispatch_saves_snapshot(self, dispatcher, tmp_path):
        """Should save snapshot image to disk."""
        alert = make_alert(with_snapshot=True)
        dispatcher.dispatch(alert)

        # Check that a snapshot file was created somewhere under alerts dir
        jpg_files = list((tmp_path / "alerts").rglob("*_snapshot.jpg"))
        assert len(jpg_files) == 1

    def test_dispatch_saves_heatmap(self, dispatcher, tmp_path):
        """Should save heatmap image to disk."""
        alert = make_alert(with_heatmap=True)
        dispatcher.dispatch(alert)

        jpg_files = list((tmp_path / "alerts").rglob("*_heatmap.jpg"))
        assert len(jpg_files) == 1

    def test_dispatch_without_images(self, dispatcher, db):
        """Should work fine without snapshot or heatmap."""
        alert = make_alert()
        dispatcher.dispatch(alert)

        alerts = db.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].snapshot_path is None
        assert alerts[0].heatmap_path is None

    def test_dispatch_multiple_alerts(self, dispatcher, db):
        """Should handle multiple alerts."""
        for i in range(5):
            alert = make_alert(alert_id=f"ALT-{i:06d}")
            dispatcher.dispatch(alert)

        assert db.get_alert_count() == 5
