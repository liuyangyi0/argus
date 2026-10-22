"""Tests for the WebSocket connection manager and endpoint."""

import asyncio
import json
import time
from pathlib import Path

import janus
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect, WebSocketState

from argus.config.schema import AuthConfig
from argus.core.health import HealthMonitor
from argus.dashboard.app import create_app
from argus.dashboard.auth import create_session_token, hash_password
from argus.dashboard.tasks import TaskManager
from argus.dashboard.websocket import (
    ConnectionManager,
    allowed_topics_for_role,
    resolve_ws_identity,
    verify_ws_token,
)
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


class _FakeWebSocket:
    def __init__(self):
        self.client_state = WebSocketState.CONNECTED
        self.sent: list[str] = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def send_text(self, message: str):
        self.sent.append(message)

    async def close(self, **_kwargs):
        self.client_state = WebSocketState.DISCONNECTED


class _RevisionChangingWebSocket(_FakeWebSocket):
    def __init__(self):
        super().__init__()
        self.manager = None
        self.client = None
        self.subscriptions_during_ack = None

    async def send_text(self, message: str):
        if not self.sent and self.manager is not None and self.client is not None:
            self.subscriptions_during_ack = set(self.client.subscriptions)
            with self.manager._event_lock:
                self.manager._resource_revisions[("alerts", "ALT-race")] = 2
                self.manager._topic_revisions["alerts"] = 2
            await asyncio.sleep(0)
        await super().send_text(message)


class _ConcurrentSendWebSocket(_FakeWebSocket):
    def __init__(self):
        super().__init__()
        self.active_sends = 0
        self.max_active_sends = 0

    async def send_text(self, message: str):
        self.active_sends += 1
        self.max_active_sends = max(self.max_active_sends, self.active_sends)
        await asyncio.sleep(0.01)
        self.sent.append(message)
        self.active_sends -= 1


class _HangingSendWebSocket(_FakeWebSocket):
    def __init__(self):
        super().__init__()
        self.release = asyncio.Event()

    async def send_text(self, message: str):
        await self.release.wait()
        self.sent.append(message)


class _SignallingWebSocket(_FakeWebSocket):
    def __init__(self):
        super().__init__()
        self.message_sent = asyncio.Event()

    async def send_text(self, message: str):
        self.sent.append(message)
        self.message_sent.set()


class _BlockingAcceptWebSocket(_FakeWebSocket):
    def __init__(self):
        super().__init__()
        self.accept_started = asyncio.Event()
        self.release_accept = asyncio.Event()

    async def accept(self):
        self.accept_started.set()
        await self.release_accept.wait()
        await super().accept()


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
            ack = ws.receive_json()
            assert ack["schema_version"] == 1
            assert ack["type"] == "subscribe_ack"
            assert ack["topic"] == "control"
            assert ack["data"]["subscriptions"] == ["alerts", "health"]

    def test_websocket_unsubscribe(self, client):
        """Client can unsubscribe from topics."""
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({
                "action": "subscribe",
                "topics": ["cameras"],
            }))
            ws.receive_json()
            ws.send_text(json.dumps({
                "action": "unsubscribe",
                "topics": ["cameras"],
            }))
            ack = ws.receive_json()
            assert ack["type"] == "unsubscribe_ack"
            assert ack["data"]["subscriptions"] == []


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

    def test_session_websocket_requires_same_origin(self, db, health):
        from argus.config.schema import ArgusConfig

        db.create_user("alice", hash_password("pw-alice"), "viewer", "Alice")
        config = ArgusConfig(auth=AuthConfig(enabled=True, api_token="test-secret"))
        app = create_app(database=db, health_monitor=health, config=config)
        client = TestClient(app)
        client.cookies.set(
            "argus_session",
            create_session_token("alice", "viewer", app.state.session_secret),
        )

        with client.websocket_connect(
            "/ws",
            headers={"origin": "http://testserver"},
        ) as ws:
            ws.send_text(json.dumps({"action": "pong"}))

        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                "/ws",
                headers={"origin": "https://cross-origin.invalid"},
            ):
                pass
        assert exc_info.value.code == 4403

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

    def test_session_identity_retains_role_for_topic_rbac(self):
        config = AuthConfig(enabled=True, api_token="api-secret")
        token = create_session_token("alice", "viewer", "session-secret")

        identity = resolve_ws_identity(token, config, "session-secret")

        assert identity == {
            "username": "alice",
            "role": "viewer",
            "auth_kind": "session",
        }
        assert "alerts" in allowed_topics_for_role("viewer")
        assert "tasks" not in allowed_topics_for_role("viewer")

    def test_session_identity_refreshes_role_and_rejects_disabled_user(self, db):
        config = AuthConfig(enabled=True, api_token="api-secret")
        db.create_user("alice", hash_password("pw-alice"), "admin")
        token = create_session_token("alice", "admin", "session-secret")

        db.update_user("alice", role="viewer")
        assert resolve_ws_identity(
            token,
            config,
            "session-secret",
            db,
        ) == {
            "username": "alice",
            "role": "viewer",
            "auth_kind": "session",
        }

        db.update_user("alice", active=False)
        assert resolve_ws_identity(token, config, "session-secret", db) is None


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

    @pytest.mark.asyncio
    async def test_new_connection_starts_with_empty_subscription(self):
        manager = ConnectionManager()
        websocket = _FakeWebSocket()

        client = await manager.accept(websocket, "client-1", role="viewer")

        assert websocket.accepted is True
        assert client.subscriptions == set()
        assert client.role == "viewer"
        await manager.disconnect("client-1")

    @pytest.mark.asyncio
    async def test_connection_defaults_to_least_privileged_role(self):
        manager = ConnectionManager()
        websocket = _FakeWebSocket()

        client = await manager.accept(websocket, "client-1")

        assert client.role == "viewer"
        await manager.disconnect("client-1")

    @pytest.mark.asyncio
    async def test_pending_handshake_reserves_connection_capacity(self):
        manager = ConnectionManager(max_connections=1)
        first_socket = _BlockingAcceptWebSocket()
        first_accept = asyncio.create_task(manager.accept(
            first_socket,
            "client-first",
        ))
        await first_socket.accept_started.wait()

        second_socket = _FakeWebSocket()
        with pytest.raises(WebSocketDisconnect):
            await manager.accept(second_socket, "client-second")
        assert second_socket.client_state == WebSocketState.DISCONNECTED

        first_socket.release_accept.set()
        await first_accept
        assert manager.connection_count == 1
        await manager.disconnect("client-first")

    @pytest.mark.asyncio
    async def test_disconnect_principal_closes_only_matching_user_sockets(self):
        manager = ConnectionManager()
        alice_socket = _FakeWebSocket()
        bob_socket = _FakeWebSocket()
        await manager.accept(
            alice_socket,
            "client-alice",
            role="admin",
            username="alice",
            identity_resolver=lambda: {"username": "alice", "role": "admin"},
        )
        await manager.accept(
            bob_socket,
            "client-bob",
            role="viewer",
            username="bob",
            identity_resolver=lambda: {"username": "bob", "role": "viewer"},
        )

        assert await manager.disconnect_principal("alice") == 1

        assert alice_socket.client_state == WebSocketState.DISCONNECTED
        assert bob_socket.client_state == WebSocketState.CONNECTED
        assert manager.connection_count == 1
        await manager.disconnect("client-bob")

    @pytest.mark.asyncio
    async def test_subscribe_revalidates_changed_session_identity(self):
        manager = ConnectionManager()
        websocket = _FakeWebSocket()
        identity = {"username": "alice", "role": "admin"}
        client = await manager.accept(
            websocket,
            "client-alice",
            role="admin",
            username="alice",
            identity_resolver=lambda: dict(identity),
        )
        identity["role"] = "viewer"

        await manager._handle_subscribe(client, {
            "action": "subscribe",
            "topics": ["models"],
        })

        assert websocket.client_state == WebSocketState.DISCONNECTED
        assert manager.connection_count == 0
        assert websocket.sent == []

    @pytest.mark.asyncio
    async def test_subscribe_ack_enforces_rbac_and_requests_revision_resync(self):
        manager = ConnectionManager()
        websocket = _FakeWebSocket()
        client = await manager.accept(websocket, "client-1", role="viewer")
        manager._topic_revisions["alerts"] = 4

        await manager._handle_subscribe(client, {
            "action": "subscribe",
            "topics": ["alerts", "tasks", "unknown"],
            "revisions": {"alerts": 1},
        })

        ack, resync = [json.loads(message) for message in websocket.sent]
        assert ack["type"] == "subscribe_ack"
        assert ack["data"]["server_epoch"] == manager._server_epoch
        assert ack["data"]["subscriptions"] == ["alerts"]
        assert ack["data"]["unauthorized_topics"] == ["tasks"]
        assert ack["data"]["invalid_topics"] == ["unknown"]
        assert resync["type"] == "resync_required"
        assert resync["data"]["topics"] == [{
            "topic": "alerts",
            "client_revision": 1,
            "server_revision": 4,
            "reason": "revision_gap",
        }]

    @pytest.mark.asyncio
    async def test_subscribe_ack_precedes_delivery_and_resyncs_an_ack_window_race(self):
        manager = ConnectionManager()
        websocket = _RevisionChangingWebSocket()
        client = await manager.accept(websocket, "client-race", role="viewer")
        websocket.manager = manager
        websocket.client = client
        manager._resource_revisions[("alerts", "ALT-race")] = 1
        manager._topic_revisions["alerts"] = 1

        await manager._handle_subscribe(client, {
            "action": "subscribe",
            "topics": ["alerts"],
            "resource_revisions": {"alerts": {"ALT-race": 1}},
        })

        ack, resync = [json.loads(message) for message in websocket.sent]
        assert websocket.subscriptions_during_ack == set()
        assert client.subscriptions == {"alerts"}
        assert ack["type"] == "subscribe_ack"
        assert ack["data"]["resource_revisions"] == {"alerts": {"ALT-race": 1}}
        assert resync["type"] == "resync_required"
        assert resync["data"]["topics"] == [{
            "topic": "alerts",
            "resource_id": "ALT-race",
            "client_revision": 1,
            "server_revision": 2,
            "reason": "revision_changed_during_subscribe",
        }]

    @pytest.mark.asyncio
    async def test_ack_window_race_updates_an_existing_gap_to_latest_revision(self):
        manager = ConnectionManager()
        websocket = _RevisionChangingWebSocket()
        client = await manager.accept(websocket, "client-race-gap", role="viewer")
        websocket.manager = manager
        websocket.client = client
        manager._resource_revisions[("alerts", "ALT-race")] = 1
        manager._topic_revisions["alerts"] = 1

        await manager._handle_subscribe(client, {
            "action": "subscribe",
            "topics": ["alerts"],
            "resource_revisions": {"alerts": {"ALT-race": 0}},
        })

        _, resync = [json.loads(message) for message in websocket.sent]
        assert resync["data"]["topics"] == [{
            "topic": "alerts",
            "resource_id": "ALT-race",
            "client_revision": 0,
            "server_revision": 2,
            "reason": "revision_changed_during_subscribe",
        }]

    @pytest.mark.asyncio
    async def test_client_send_lock_serializes_control_frames(self):
        manager = ConnectionManager()
        websocket = _ConcurrentSendWebSocket()
        client = await manager.accept(websocket, "client-send-lock", role="viewer")

        await asyncio.gather(
            manager._send_control(client, "first", {}),
            manager._send_control(client, "second", {}),
        )

        assert websocket.max_active_sends == 1
        assert [json.loads(item)["type"] for item in websocket.sent] == [
            "first",
            "second",
        ]

    @pytest.mark.asyncio
    async def test_client_send_timeout_releases_the_send_lock(self):
        manager = ConnectionManager(send_timeout_seconds=0.01)
        websocket = _HangingSendWebSocket()
        client = await manager.accept(websocket, "client-hanging", role="viewer")

        with pytest.raises(TimeoutError):
            await manager._send_control(client, "blocked", {})

        assert client.send_lock.locked() is False
        websocket.client_state = WebSocketState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_preselected_delivery_is_dropped_after_subscription_changes(self):
        manager = ConnectionManager()
        websocket = _FakeWebSocket()
        client = await manager.accept(websocket, "client-subscription-race", role="viewer")
        client.subscriptions = {"alerts"}
        selected_generation = client.subscription_generation

        # Hold the per-client send lock so the delivery has already been
        # selected by dispatch but cannot send until the subscription changes.
        await client.send_lock.acquire()
        delivery = asyncio.create_task(manager._send_topic_text(
            client,
            '{"topic":"alerts"}',
            "alerts",
            selected_generation,
        ))
        await asyncio.sleep(0)
        client.subscriptions.clear()
        client.subscription_generation += 1
        client.send_lock.release()

        await delivery
        assert websocket.sent == []
        await manager.disconnect("client-subscription-race")

    @pytest.mark.asyncio
    async def test_slow_client_does_not_delay_delivery_to_healthy_client(self):
        manager = ConnectionManager(send_timeout_seconds=0.2)
        queue = janus.Queue(maxsize=10)
        manager._queue = queue
        slow = await manager.accept(
            _HangingSendWebSocket(),
            "client-slow",
            role="viewer",
        )
        fast_socket = _SignallingWebSocket()
        fast = await manager.accept(fast_socket, "client-fast", role="viewer")
        slow.subscriptions = {"alerts"}
        fast.subscriptions = {"alerts"}
        dispatch = asyncio.create_task(manager._dispatch_loop(queue))

        assert manager.broadcast("alerts", {"alert_id": "ALT-fast"}) is True
        await asyncio.wait_for(fast_socket.message_sent.wait(), timeout=0.05)

        assert json.loads(fast_socket.sent[0])["data"] == {
            "alert_id": "ALT-fast",
        }
        dispatch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await dispatch
        manager._connections.clear()
        queue.close()
        await queue.wait_closed()
        manager._queue = None

    @pytest.mark.asyncio
    async def test_broadcast_envelope_deduplicates_event_and_stale_revision(self):
        manager = ConnectionManager()
        manager._queue = janus.Queue(maxsize=10)

        manager.broadcast(
            "alerts",
            {"alert_id": "ALT-1"},
            event_type="alert.created",
            event_id="evt-1",
            revision=1,
        )
        manager.broadcast(
            "alerts",
            {"alert_id": "ALT-1"},
            event_id="evt-1",
            revision=2,
        )
        manager.broadcast(
            "alerts",
            {"alert_id": "ALT-1"},
            event_id="evt-2",
            revision=1,
        )

        assert manager._queue.sync_q.qsize() == 1
        event = manager._queue.sync_q.get_nowait()
        assert event["schema_version"] == 1
        assert event["event_id"] == "evt-1"
        assert event["topic"] == "alerts"
        assert event["type"] == "alert.created"
        assert event["resource_id"] == "ALT-1"
        assert event["revision"] == 1
        assert event["occurred_at"]
        assert event["data"] == {"alert_id": "ALT-1"}

        manager._queue.close()
        await manager._queue.wait_closed()

    @pytest.mark.asyncio
    async def test_queue_overflow_is_reported_and_durable_retry_is_accepted(self):
        manager = ConnectionManager()
        manager._queue = janus.Queue(maxsize=1)

        assert manager.broadcast(
            "alerts", {"alert_id": "ALT-1"}, event_id="evt-1"
        ) is True
        assert manager.broadcast(
            "alerts", {"alert_id": "ALT-2"}, event_id="evt-2"
        ) is False

        manager._queue.sync_q.get_nowait()
        assert manager.broadcast(
            "alerts", {"alert_id": "ALT-2"}, event_id="evt-2"
        ) is True
        retried = manager._queue.sync_q.get_nowait()
        assert retried["event_id"] == "evt-2"
        # Revision 1 was intentionally consumed by the overflow.  Clients see
        # this gap and request a REST resync instead of trusting stale state.
        assert retried["revision"] == 2

        manager._queue.close()
        await manager._queue.wait_closed()
        manager._queue = None

    @pytest.mark.asyncio
    async def test_queue_overflow_accepts_same_explicit_revision_retry(self):
        manager = ConnectionManager()
        manager._queue = janus.Queue(maxsize=1)

        assert manager.broadcast(
            "alerts",
            {"alert_id": "ALT-filler"},
            event_id="evt-filler",
            revision=1,
        ) is True
        assert manager.broadcast(
            "alerts",
            {"alert_id": "ALT-retry"},
            event_id="evt-explicit-retry",
            revision=7,
        ) is False

        manager._queue.sync_q.get_nowait()
        assert manager.broadcast(
            "alerts",
            {"alert_id": "ALT-retry"},
            event_id="evt-explicit-retry",
            revision=7,
        ) is True
        retried = manager._queue.sync_q.get_nowait()
        assert retried["event_id"] == "evt-explicit-retry"
        assert retried["revision"] == 7

        manager._queue.close()
        await manager._queue.wait_closed()
        manager._queue = None

    @pytest.mark.asyncio
    async def test_queue_retry_is_obsolete_after_newer_resource_event(self):
        manager = ConnectionManager()
        manager._queue = janus.Queue(maxsize=1)

        assert manager.broadcast(
            "alerts",
            {"alert_id": "ALT-filler"},
            event_id="evt-filler",
            revision=1,
        ) is True
        assert manager.broadcast(
            "alerts",
            {"alert_id": "ALT-retry"},
            event_id="evt-obsolete",
            revision=7,
        ) is False

        manager._queue.sync_q.get_nowait()
        assert manager.broadcast(
            "alerts",
            {"alert_id": "ALT-retry", "status": "newer"},
            event_id="evt-newer",
            revision=8,
        ) is True
        newer = manager._queue.sync_q.get_nowait()
        assert newer["revision"] == 8

        # A durable outbox may retry the older event after capacity returns.
        # Treat it as successfully superseded without sending it out of order.
        assert manager.broadcast(
            "alerts",
            {"alert_id": "ALT-retry"},
            event_id="evt-obsolete",
            revision=7,
        ) is True
        assert manager._queue.sync_q.empty()
        assert manager._resource_revisions[("alerts", "ALT-retry")] == 8

        manager._queue.close()
        await manager._queue.wait_closed()
        manager._queue = None

    @pytest.mark.asyncio
    async def test_evicted_resource_reenters_above_topic_high_water(self):
        manager = ConnectionManager()
        manager._resource_revision_limit = 1
        manager._queue = janus.Queue(maxsize=10)

        assert manager.broadcast("alerts", {"alert_id": "ALT-A"}) is True
        assert manager.broadcast("alerts", {"alert_id": "ALT-B"}) is True
        assert manager.broadcast("alerts", {"alert_id": "ALT-A"}) is True

        events = [manager._queue.sync_q.get_nowait() for _ in range(3)]
        assert [
            (event["resource_id"], event["revision"])
            for event in events
        ] == [("ALT-A", 1), ("ALT-B", 2), ("ALT-A", 3)]

        manager._queue.close()
        await manager._queue.wait_closed()
        manager._queue = None

    @pytest.mark.asyncio
    async def test_closed_queue_is_reported_without_raising(self):
        manager = ConnectionManager()
        queue = janus.Queue(maxsize=1)
        manager._queue = queue
        queue.close()

        assert manager.broadcast("alerts", {"alert_id": "ALT-stop"}) is False

        await queue.wait_closed()
        manager._queue = None

    @pytest.mark.asyncio
    async def test_revisions_are_ordered_per_topic_and_resource(self):
        manager = ConnectionManager()
        manager._queue = janus.Queue(maxsize=10)

        assert manager.broadcast(
            "cameras",
            {"camera_id": "cam-01"},
            event_id="cam-01-r1",
            revision=1,
        ) is True
        assert manager.broadcast(
            "cameras",
            {"camera_id": "cam-02"},
            event_id="cam-02-r1",
            revision=1,
        ) is True
        assert manager.broadcast(
            "cameras",
            {"camera_id": "cam-01"},
            event_id="cam-01-stale",
            revision=1,
        ) is True

        first = manager._queue.sync_q.get_nowait()
        second = manager._queue.sync_q.get_nowait()
        assert [
            (first["resource_id"], first["revision"]),
            (second["resource_id"], second["revision"]),
        ] == [
            ("cam-01", 1),
            ("cam-02", 2),
        ]
        assert manager._queue.sync_q.empty()

        manager._queue.close()
        await manager._queue.wait_closed()
        manager._queue = None

    @pytest.mark.asyncio
    async def test_resource_revision_ack_requests_only_mismatched_resource_resync(self):
        manager = ConnectionManager()
        websocket = _FakeWebSocket()
        client = await manager.accept(websocket, "client-1", role="viewer")
        manager._resource_revisions[("alerts", "ALT-1")] = 4
        manager._topic_revisions["alerts"] = 4

        await manager._handle_subscribe(client, {
            "action": "subscribe",
            "topics": ["alerts"],
            "resource_revisions": {"alerts": {"ALT-1": 2}},
        })

        ack, resync = [json.loads(message) for message in websocket.sent]
        assert ack["data"]["resource_revisions"] == {"alerts": {"ALT-1": 4}}
        assert resync["data"]["topics"] == [{
            "topic": "alerts",
            "resource_id": "ALT-1",
            "client_revision": 2,
            "server_revision": 4,
            "reason": "revision_gap",
        }]


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

        manager.submit("test_task", dummy_task)
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
        """Serve the Vue SPA only when the separately-built dist exists."""
        response = client.get("/")
        index = Path(__file__).resolve().parents[2] / "web" / "dist" / "index.html"
        if not index.is_file():
            assert response.status_code == 404
            assert response.json()["code"] == 40400
            return
        assert response.status_code == 200
        assert '<div id="app">' in response.text
