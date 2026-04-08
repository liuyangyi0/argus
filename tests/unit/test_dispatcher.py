"""Tests for the alert dispatcher module."""

import time
from datetime import datetime, timezone

import numpy as np
import pytest

from argus.alerts.dispatcher import AlertDispatcher
from argus.alerts.grader import Alert
from argus.config.schema import AlertConfig, AlertSeverity, WebhookConfig
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
    d = AlertDispatcher(config=config, database=db, alerts_dir=tmp_path / "alerts")
    yield d
    d.close()


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
        dispatcher.flush_db_queue()

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
        dispatcher.flush_db_queue()

        alerts = db.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].snapshot_path is None
        assert alerts[0].heatmap_path is None

    def test_dispatch_multiple_alerts(self, dispatcher, db):
        """Should handle multiple alerts."""
        for i in range(5):
            alert = make_alert(alert_id=f"ALT-{i:06d}")
            dispatcher.dispatch(alert)
        dispatcher.flush_db_queue()

        assert db.get_alert_count() == 5


class TestWebhookDispatch:
    """Tests for the webhook dispatch channel."""

    def test_webhook_posts_to_url(self, db, tmp_path):
        """Webhook should POST alert payload to configured URL."""
        from unittest.mock import MagicMock, patch

        config = AlertConfig(
            webhook=WebhookConfig(enabled=True, url="http://localhost:9999/alerts"),
        )
        d = AlertDispatcher(config=config, database=db, alerts_dir=tmp_path / "alerts")
        try:
            # Ensure circuit breaker is closed
            d._circuit_breaker.record_success()

            alert = make_alert()

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            d._http_client = mock_client

            d.dispatch(alert)
            time.sleep(1.0)

            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://localhost:9999/alerts"
            payload = call_args[1]["json"]
            assert payload["alert_id"] == "ALT-000001"
            assert payload["severity"] == "medium"
        finally:
            d.close()

    def test_webhook_circuit_breaker_blocks_after_failures(self, db, tmp_path):
        """Circuit breaker should open after consecutive failures."""
        from unittest.mock import MagicMock

        config = AlertConfig(
            webhook=WebhookConfig(enabled=True, url="http://localhost:9999/alerts"),
            circuit_breaker_threshold=2,
            circuit_breaker_timeout=60.0,
        )
        d = AlertDispatcher(config=config, database=db, alerts_dir=tmp_path / "alerts")
        try:
            # Record enough failures to open the circuit breaker
            for _ in range(3):
                d._circuit_breaker.record_failure()

            assert not d._circuit_breaker.allow_request()

            alert = make_alert()
            d.dispatch(alert)
            time.sleep(0.2)

            # Webhook queue should be empty since CB blocked the request
            assert d._webhook_queue.qsize() == 0
        finally:
            d.close()

    def test_webhook_retry_on_failure(self, db, tmp_path):
        """Webhook should retry with backoff on transient failure."""
        from unittest.mock import MagicMock

        config = AlertConfig(
            webhook=WebhookConfig(enabled=True, url="http://localhost:9999/alerts"),
        )
        d = AlertDispatcher(config=config, database=db, alerts_dir=tmp_path / "alerts")
        try:
            mock_client = MagicMock()
            # First call raises, second succeeds
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_client.post.side_effect = [
                ConnectionError("timeout"),
                mock_response,
            ]
            d._http_client = mock_client

            # Reset circuit breaker to ensure clean state
            d._circuit_breaker.record_success()

            alert = make_alert()
            d.dispatch(alert)
            time.sleep(3.0)  # Allow time for retry with backoff

            assert mock_client.post.call_count == 2
        finally:
            d.close()


class TestDiskSpaceCheck:
    """Tests for disk space safety checks."""

    def test_fail_closed_on_check_error(self, db, tmp_path):
        """Disk check failure should fail-closed (skip image save)."""
        from unittest.mock import patch

        config = AlertConfig()
        d = AlertDispatcher(config=config, database=db, alerts_dir=tmp_path / "alerts")
        try:
            # Force cache expiry
            d._disk_space_checked_at = 0.0

            with patch("shutil.disk_usage", side_effect=OSError("Permission denied")):
                result = d._check_disk_space()

            assert result is False  # fail-closed
        finally:
            d.close()

    def test_low_disk_space_skips_images(self, db, tmp_path):
        """When disk space is low, images should be skipped but DB record persisted."""
        from unittest.mock import patch, MagicMock
        from collections import namedtuple

        DiskUsage = namedtuple("DiskUsage", ["total", "used", "free"])

        config = AlertConfig()
        d = AlertDispatcher(config=config, database=db, alerts_dir=tmp_path / "alerts")
        try:
            d._disk_space_checked_at = 0.0

            with patch("shutil.disk_usage", return_value=DiskUsage(
                total=100 * 1024 * 1024,
                used=99 * 1024 * 1024,
                free=1 * 1024 * 1024,  # 1 MB < 500 MB threshold
            )):
                alert = make_alert(with_snapshot=True, with_heatmap=True)
                d.dispatch(alert)
                d.flush_db_queue()

            # DB record should exist
            alerts = db.get_alerts()
            assert len(alerts) == 1
            # But no images saved
            jpg_files = list((tmp_path / "alerts").rglob("*.jpg"))
            assert len(jpg_files) == 0
        finally:
            d.close()
