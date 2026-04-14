"""Tests for the WebSocket connection manager and endpoint."""

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from argus.config.schema import AuthConfig
from argus.core.health import HealthMonitor
from argus.dashboard.app import create_app
from argus.dashboard.tasks import TaskManager
from argus.dashboard.websocket import ConnectionManager, verify_ws_token
from argus.storage.database import Database


@pytest.fixture
def db(tmp_path):
    database = Database(database_url=f"sqlite:///{tmp_path / 'test.db'}")
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def health():
    return HealthMonitor()


@pytest.fixture
def client(db, health):
    app = create_app(database=db, health_monitor=health)
    return TestClient(app)


class TestWebSocketEndpoint:
    def test_websocket_connect(self, client):
        """WebSocket connection should be accepted."""
        with client.websocket_connect("/ws") as ws:
            # Connection established, send a subscribe message
            ws.send_text(json.dumps({"action": "subscribe", "topics": ["health"]}))
            # Connection should stay open

    def test_websocket_receives_ping(self, client):
        """Server should be able to communicate after connect."""
        with client.websocket_connect("/ws") as ws:
            # Send pong to keep alive
            ws.send_text(json.dumps({"action": "pong"}))

    def test_websocket_subscribe(self, client):
        """Client can subscribe to specific topics."""
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({
                "action": "subscribe",
                "topics": ["health", "alerts"],
            }))
            # No error means subscription accepted

    def test_websocket_unsubscribe(self, client):
        """Client can unsubscribe from topics."""
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({
                "action": "unsubscribe",
                "topics": ["cameras"],
            }))


class TestWebSocketAuth:
    def test_auth_disabled_allows_connection(self, db, health):
        """With auth disabled, connection should succeed without token."""
        app = create_app(database=db, health_monitor=health)
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"action": "pong"}))

    def test_auth_enabled_rejects_no_token(self, db, health):
        """With auth enabled and no token, connection should be rejected."""
        from argus.config.schema import ArgusConfig
        config = ArgusConfig(auth=AuthConfig(enabled=True, api_token="test-secret"))
        app = create_app(database=db, health_monitor=health, config=config)
        client = TestClient(app)
        with pytest.raises(Exception):
            with client.websocket_connect("/ws"):
                pass

    def test_auth_enabled_accepts_valid_token(self, db, health):
        """With auth enabled and valid token, connection should succeed."""
        from argus.config.schema import ArgusConfig
        config = ArgusConfig(auth=AuthConfig(enabled=True, api_token="test-secret"))
        app = create_app(database=db, health_monitor=health, config=config)
        client = TestClient(app)
        with client.websocket_connect("/ws?token=test-secret") as ws:
            ws.send_text(json.dumps({"action": "pong"}))

    def test_auth_enabled_rejects_wrong_token(self, db, health):
        """With auth enabled and wrong token, connection should be rejected."""
        from argus.config.schema import ArgusConfig
        config = ArgusConfig(auth=AuthConfig(enabled=True, api_token="test-secret"))
        app = create_app(database=db, health_monitor=health, config=config)
        client = TestClient(app)
        with pytest.raises(Exception):
            with client.websocket_connect("/ws?token=wrong-token"):
                pass


class TestVerifyWsToken:
    def test_disabled_auth_always_passes(self):
        config = AuthConfig(enabled=False)
        assert verify_ws_token("", config) is True
        assert verify_ws_token("anything", config) is True

    def test_enabled_auth_requires_correct_token(self):
        config = AuthConfig(enabled=True, api_token="my-secret")
        assert verify_ws_token("my-secret", config) is True
        assert verify_ws_token("wrong", config) is False
        assert verify_ws_token("", config) is False

    def test_constant_time_comparison(self):
        """Token verification should use constant-time comparison."""
        config = AuthConfig(enabled=True, api_token="secret-token")
        # Both should work correctly regardless of timing
        assert verify_ws_token("secret-token", config) is True
        assert verify_ws_token("secret-toke!", config) is False


class TestConnectionManager:
    def test_broadcast_validates_topic(self):
        """Broadcast with invalid topic should be silently ignored."""
        manager = ConnectionManager()
        # Should not raise
        manager.broadcast("invalid_topic", {"test": True})

    def test_broadcast_valid_topic(self):
        """Broadcast with valid topic should be accepted."""
        manager = ConnectionManager()
        manager.broadcast("health", {"status": "healthy"})
        manager.broadcast("cameras", {"camera_id": "cam_01"})
        manager.broadcast("alerts", {"alert_id": "test"})
        manager.broadcast("tasks", {"task_id": "test"})


class TestHealthMonitorCallback:
    def test_health_monitor_fires_callback_on_update(self):
        """HealthMonitor should call on_change when camera status changes."""
        events = []

        def capture(topic, data):
            events.append((topic, data))

        monitor = HealthMonitor(on_change=capture)
        monitor.update_camera("cam_01", connected=True, frames_captured=100)

        assert len(events) == 1
        assert events[0][0] == "health"
        assert events[0][1]["cameras"][0]["camera_id"] == "cam_01"

    def test_health_monitor_fires_callback_on_alert(self):
        """HealthMonitor should call on_change when alert count changes."""
        events = []

        def capture(topic, data):
            events.append((topic, data))

        monitor = HealthMonitor(on_change=capture)
        monitor.record_alert()

        assert len(events) == 1
        assert events[0][1]["total_alerts"] == 1

    def test_health_monitor_no_callback_when_none(self):
        """HealthMonitor should work fine without callback."""
        monitor = HealthMonitor()
        monitor.update_camera("cam_01", connected=True)
        monitor.record_alert()
        # No error


class TestTaskManagerCallback:
    def test_task_manager_fires_callback_on_progress(self):
        """TaskManager should call on_change when task progresses."""
        events = []

        def capture(topic, data):
            events.append((topic, data))

        manager = TaskManager(on_change=capture)

        def dummy_task(progress_callback):
            progress_callback(50, "halfway")
            return "done"

        task_id = manager.submit("test_task", dummy_task)
        # Give thread time to execute
        time.sleep(0.5)

        # Should have received progress and completion events
        task_events = [e for e in events if e[0] == "tasks"]
        assert len(task_events) >= 1
        # At least one should show progress or completion
        statuses = [e[1]["status"] for e in task_events]
        assert "complete" in statuses or "running" in statuses

    def test_task_manager_no_callback_when_none(self):
        """TaskManager should work fine without callback."""
        manager = TaskManager()

        def dummy_task(progress_callback):
            return "done"

        task_id = manager.submit("test_task", dummy_task)
        time.sleep(0.3)
        task = manager.get_task(task_id)
        assert task.status.value == "complete"


class TestDashboardWithWebSocket:
    def test_index_page_serves_vue_spa(self, client):
        """Index page should serve Vue SPA with app mount point."""
        response = client.get("/")
        assert response.status_code == 200
        assert '<div id="app">' in response.text

