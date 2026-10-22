"""WebSocket connection manager for real-time dashboard updates.

Bridges synchronous data sources (camera threads, alert dispatcher) to
async WebSocket clients via a ``janus`` sync/async queue — replacing the
fragile ``call_soon_threadsafe`` approach.

Topics:
- health:        System health and camera status changes
- cameras:       Camera statistics updates
- alerts:        New alert notifications
- tasks:         Background task progress updates
- wall:          Video wall aggregated score/status (UX v2 §2)
- degradation:   Degradation events new/resolved (UX v2 §5)
- heatmap:       Per-camera anomaly heatmap data for frontend Canvas overlay
- audio_alert:   Audio alert playback events
- models:        Model lifecycle events (activation success/failure, reload notices)
- model_release: Release-pipeline stage transitions (candidate→shadow→canary→production)
- system_errors: Runtime error events from the unified error channel (dispatcher,
                 pipeline, release_pipeline, gige_capture, ...) — used by the
                 dashboard to aggregate and display backend failures.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections import OrderedDict, deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

import janus
import structlog
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

if TYPE_CHECKING:
    from argus.config.schema import AuthConfig
    from argus.storage.database import Database

logger = structlog.get_logger()

WS_SCHEMA_VERSION = 1
CONTROL_TOPIC = "control"

VALID_TOPICS = frozenset({
    "health", "cameras", "alerts", "tasks", "wall", "degradation", "heatmap",
    "audio_alert", "models", "model_release", "system_degradation",
    "system_errors",
})

TOPIC_PERMISSIONS: dict[str, str] = {
    "health": "read_system",
    "cameras": "read_cameras",
    "alerts": "read_alerts",
    "tasks": "manage_training",
    "wall": "read_cameras",
    "degradation": "read_degradation",
    "heatmap": "read_cameras",
    "audio_alert": "read_alerts",
    "models": "read_models",
    "model_release": "read_models",
    "system_degradation": "read_degradation",
    "system_errors": "read_system",
}


def allowed_topics_for_role(role: str) -> frozenset[str]:
    """Resolve read-only topic access from the existing HTTP RBAC matrix."""
    from argus.dashboard.auth import PERMISSION_MAP

    permissions = set(PERMISSION_MAP.get(role, []))
    if "*" in permissions:
        return VALID_TOPICS
    return frozenset(
        topic
        for topic, permission in TOPIC_PERMISSIONS.items()
        if permission in permissions
    )


def make_ws_event(
    *,
    topic: str,
    event_type: str,
    data: dict[str, Any],
    resource_id: str,
    revision: int,
    event_id: str | None = None,
    occurred_at: str | None = None,
) -> dict[str, Any]:
    """Build a versioned event while retaining legacy ``topic``/``data`` keys."""
    return {
        "schema_version": WS_SCHEMA_VERSION,
        "event_id": event_id or str(uuid.uuid4()),
        "topic": topic,
        "type": event_type,
        "resource_id": resource_id,
        "revision": revision,
        "occurred_at": occurred_at or datetime.now(timezone.utc).isoformat(),
        "data": data,
    }


def _infer_resource_id(topic: str, data: dict[str, Any]) -> str:
    for key in (
        "resource_id",
        "camera_id",
        "alert_id",
        "task_id",
        "model_id",
        "operation_id",
        "id",
    ):
        value = data.get(key)
        if value is not None and str(value):
            return str(value)
    return f"{topic}:global"


class _ClientConnection:
    """Represents a single WebSocket client."""

    __slots__ = (
        "client_id",
        "websocket",
        "subscriptions",
        "last_pong",
        "role",
        "username",
        "identity_resolver",
        "subscription_generation",
        "send_lock",
    )

    def __init__(
        self,
        client_id: str,
        websocket: WebSocket,
        subscriptions: set[str],
        role: str,
        username: str = "",
        identity_resolver: Callable[[], dict[str, str] | None] | None = None,
    ):
        self.client_id = client_id
        self.websocket = websocket
        self.subscriptions = subscriptions
        self.last_pong = time.monotonic()
        self.role = role
        self.username = username
        self.identity_resolver = identity_resolver
        self.subscription_generation = 0
        self.send_lock = asyncio.Lock()


class ConnectionManager:
    """Manages WebSocket connections with topic-based subscriptions.

    Thread-safe: the ``broadcast`` method can be called from **any** thread.
    Events are enqueued into the sync side of a ``janus`` queue and drained
    by an asyncio task on the event-loop side.
    """

    def __init__(
        self,
        heartbeat_seconds: int = 30,
        max_connections: int = 100,
        send_timeout_seconds: float = 1.0,
    ):
        self._connections: dict[str, _ClientConnection] = {}
        self._pending_accepts = 0
        self._heartbeat_seconds = heartbeat_seconds
        self._max_connections = max_connections
        self._send_timeout_seconds = max(float(send_timeout_seconds), 0.01)

        # Initialised in start() when the event loop is running
        self._queue: janus.Queue[dict[str, Any]] | None = None
        self._broadcast_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._event_lock = threading.Lock()
        # Event revisions are scoped to a resource.  Topic high-water marks are
        # retained only for read-only compatibility with older subscribers.
        self._topic_revisions: dict[str, int] = {}
        self._resource_revisions: OrderedDict[tuple[str, str], int] = OrderedDict()
        self._resource_revision_limit = 4096
        self._recent_event_ids: deque[str] = deque()
        self._recent_event_id_set: set[str] = set()
        self._recent_event_limit = 4096
        # Queue-full retries reserve their transport revision under event_id.
        # Without this, retrying an explicit revision is reported as stale even
        # though the event never reached the dispatcher.
        self._pending_event_revisions: OrderedDict[
            str, tuple[str, str, int]
        ] = OrderedDict()
        self._pending_event_limit = 4096
        self._server_epoch = str(uuid.uuid4())

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the broadcast dispatcher loop."""
        queue: janus.Queue[dict[str, Any]] = janus.Queue(maxsize=1000)
        with self._event_lock:
            self._queue = queue
        self._broadcast_task = asyncio.create_task(self._dispatch_loop(queue))
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("ws.manager_started", max_connections=self._max_connections)

    async def stop(self) -> None:
        """Stop the manager and close all connections."""
        # Detach first so producer threads fail closed instead of racing a
        # queue which is being shut down underneath them.
        with self._event_lock:
            queue = self._queue
            self._queue = None

        for task in (self._broadcast_task, self._heartbeat_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        clients = list(self._connections.values())
        if clients:
            await asyncio.gather(*(self._close_client(client) for client in clients))
        self._connections.clear()

        if queue is not None:
            queue.close()
            await queue.wait_closed()

        logger.info("ws.manager_stopped")

    # ── Client management ────────────────────────────────────────────

    async def accept(
        self,
        websocket: WebSocket,
        client_id: str,
        role: str = "viewer",
        username: str = "",
        identity_resolver: Callable[[], dict[str, str] | None] | None = None,
    ) -> _ClientConnection:
        """Accept a new WebSocket connection."""
        if len(self._connections) + self._pending_accepts >= self._max_connections:
            await websocket.close(code=1013, reason="Max connections reached")
            raise WebSocketDisconnect(code=1013)

        # Reserve capacity before awaiting the ASGI accept handshake. Multiple
        # handshakes can otherwise all observe the same pre-accept count.
        self._pending_accepts += 1
        try:
            await websocket.accept()
        finally:
            self._pending_accepts -= 1

        client = _ClientConnection(
            client_id=client_id,
            websocket=websocket,
            subscriptions=set(),
            role=role,
            username=username,
            identity_resolver=identity_resolver,
        )
        self._connections[client_id] = client
        logger.info("ws.client_connected", client_id=client_id, total=len(self._connections))
        return client

    async def disconnect(self, client_id: str) -> None:
        """Remove a client connection."""
        client = self._connections.pop(client_id, None)
        if client:
            await self._close_client(client)
            logger.info("ws.client_disconnected", client_id=client_id, total=len(self._connections))

    async def disconnect_principal(self, username: str) -> int:
        """Close all live sockets for a user whose identity just changed."""
        client_ids = [
            client_id
            for client_id, client in list(self._connections.items())
            if client.username == username and client.identity_resolver is not None
        ]
        if client_ids:
            await asyncio.gather(*(
                self.disconnect(client_id)
                for client_id in client_ids
            ))
        return len(client_ids)

    # ── Thread-safe broadcast ────────────────────────────────────────

    def broadcast(
        self,
        topic: str,
        data: dict,
        *,
        event_type: str | None = None,
        resource_id: str | None = None,
        revision: int | None = None,
        event_id: str | None = None,
        occurred_at: str | None = None,
    ) -> bool:
        """Thread-safe broadcast: enqueue an event for all subscribers.

        Can be called from **any** thread (camera threads, alert dispatcher,
        etc.).  Uses the synchronous side of a ``janus`` queue so there is
        no need for ``call_soon_threadsafe`` or direct event-loop access. The
        two-positional-argument form remains compatible with existing callers;
        metadata can be supplied through keyword-only arguments.
        """
        if topic not in VALID_TOPICS:
            return False
        resolved_resource_id = resource_id or _infer_resource_id(topic, data)
        resolved_event_id = event_id or str(uuid.uuid4())

        with self._event_lock:
            queue = self._queue
            if queue is None:
                return False
            if resolved_event_id in self._recent_event_id_set:
                logger.debug("ws.duplicate_event_ignored", event_id=resolved_event_id)
                return True

            revision_key = (topic, resolved_resource_id)
            pending = self._pending_event_revisions.get(resolved_event_id)
            if pending is not None and pending[:2] != revision_key:
                logger.warning(
                    "ws.pending_event_identity_mismatch",
                    event_id=resolved_event_id,
                    expected_topic=pending[0],
                    expected_resource_id=pending[1],
                    topic=topic,
                    resource_id=resolved_resource_id,
                )
                return False

            resource_known = revision_key in self._resource_revisions
            current_revision = self._resource_revisions.get(revision_key, 0)
            if pending is not None:
                resolved_revision = pending[2]
            elif revision is not None:
                resolved_revision = (
                    revision
                    if resource_known
                    else max(revision, self._topic_revisions.get(topic, 0) + 1)
                )
            elif resource_known:
                resolved_revision = current_revision + 1
            else:
                # A resource evicted from the bounded LRU must not restart at
                # revision 1 within the same server epoch. Starting above the
                # topic high-water lets old clients accept the re-entry and
                # detect a gap for authoritative REST reconciliation.
                resolved_revision = self._topic_revisions.get(topic, 0) + 1

            if pending is not None and current_revision > resolved_revision:
                # A newer event for the same resource was accepted while this
                # queue-full event waited for retry. The newer revision is the
                # authoritative state; do not enqueue out of order or lower the
                # in-memory cursor.
                self._pending_event_revisions.pop(resolved_event_id, None)
                self._remember_event_id_locked(resolved_event_id)
                logger.info(
                    "ws.pending_event_superseded",
                    topic=topic,
                    resource_id=resolved_resource_id,
                    revision=resolved_revision,
                    current_revision=current_revision,
                )
                return True
            if pending is None and resolved_revision <= current_revision:
                logger.debug(
                    "ws.stale_revision_ignored",
                    topic=topic,
                    revision=resolved_revision,
                    current_revision=current_revision,
                )
                return True

            self._resource_revisions[revision_key] = resolved_revision
            self._resource_revisions.move_to_end(revision_key)
            if len(self._resource_revisions) > self._resource_revision_limit:
                self._resource_revisions.popitem(last=False)
            self._topic_revisions[topic] = max(
                self._topic_revisions.get(topic, 0),
                resolved_revision,
            )
            event = make_ws_event(
                topic=topic,
                event_type=event_type or f"{topic}.updated",
                data=data,
                resource_id=resolved_resource_id,
                revision=resolved_revision,
                event_id=resolved_event_id,
                occurred_at=occurred_at,
            )

            try:
                queue.sync_q.put_nowait(event)
            except (janus.SyncQueueFull, janus.SyncQueueShutDown):
                # Keep the advanced revision so clients can observe a gap and
                # request REST resynchronisation.  Do not remember event_id:
                # a durable outbox may retry the same event after capacity is
                # available.
                self._pending_event_revisions[resolved_event_id] = (
                    topic,
                    resolved_resource_id,
                    resolved_revision,
                )
                self._pending_event_revisions.move_to_end(resolved_event_id)
                if len(self._pending_event_revisions) > self._pending_event_limit:
                    self._pending_event_revisions.popitem(last=False)
                logger.warning(
                    "ws.broadcast_queue_unavailable",
                    topic=topic,
                    revision=resolved_revision,
                )
                return False

            self._pending_event_revisions.pop(resolved_event_id, None)
            self._remember_event_id_locked(resolved_event_id)

        return True

    def _remember_event_id_locked(self, event_id: str) -> None:
        """Record an accepted idempotency key while ``_event_lock`` is held."""
        if len(self._recent_event_ids) >= self._recent_event_limit:
            expired = self._recent_event_ids.popleft()
            self._recent_event_id_set.discard(expired)
        self._recent_event_ids.append(event_id)
        self._recent_event_id_set.add(event_id)

    # ── Client message handler ───────────────────────────────────────

    async def handle_client(self, client: _ClientConnection) -> None:
        """Handle incoming messages from a client (subscriptions, pong)."""
        try:
            while True:
                raw = await client.websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(msg, dict):
                    continue

                action = msg.get("action")
                if action == "subscribe":
                    await self._handle_subscribe(client, msg)
                elif action == "unsubscribe":
                    await self._handle_unsubscribe(client, msg)
                elif action == "pong":
                    client.last_pong = time.monotonic()

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug("ws.client_error", client_id=client.client_id, error=str(e))

    async def _handle_subscribe(
        self,
        client: _ClientConnection,
        msg: dict[str, Any],
    ) -> None:
        if not self._client_identity_is_current(client):
            await self.disconnect(client.client_id)
            return
        client.subscription_generation += 1
        raw_topics = msg.get("topics", [])
        requested = {
            topic
            for topic in raw_topics
            if isinstance(topic, str)
        } if isinstance(raw_topics, list) else set()
        allowed = allowed_topics_for_role(client.role)
        accepted = requested & VALID_TOPICS & allowed
        invalid = requested - VALID_TOPICS
        unauthorized = (requested & VALID_TOPICS) - allowed
        # Pause delivery until subscribe_ack has established the revision
        # snapshot. Events published while the ack is in flight are detected
        # by the second snapshot below and repaired through REST resync.
        client.subscriptions = set()

        raw_resource_revisions = msg.get("resource_revisions")
        resource_revisions_supplied = isinstance(raw_resource_revisions, dict)
        client_resource_revisions = (
            raw_resource_revisions if resource_revisions_supplied else {}
        )
        raw_revisions = msg.get("revisions", {})
        client_revisions = raw_revisions if isinstance(raw_revisions, dict) else {}
        server_topic_revisions, server_resource_revisions = self._revision_snapshot(
            accepted
        )
        gaps: list[dict[str, Any]] = []
        for topic in sorted(accepted):
            server_resources = server_resource_revisions.get(topic, {})
            if resource_revisions_supplied:
                raw_topic_resources = client_resource_revisions.get(topic, {})
                client_resources = (
                    raw_topic_resources
                    if isinstance(raw_topic_resources, dict)
                    else {}
                )
                # Client-only resources were intentionally evicted from the
                # server's bounded cursor. Their revisions remain valid within
                # this epoch, so absence is not a reset signal.
                for resource_id in sorted(server_resources):
                    client_revision = client_resources.get(resource_id, 0)
                    server_revision = server_resources.get(resource_id, 0)
                    if (
                        isinstance(client_revision, int)
                        and not isinstance(client_revision, bool)
                        and client_revision != server_revision
                    ):
                        gaps.append({
                            "topic": topic,
                            "resource_id": resource_id,
                            "client_revision": client_revision,
                            "server_revision": server_revision,
                            "reason": (
                                "server_revision_reset"
                                if client_revision > server_revision
                                else "revision_gap"
                            ),
                        })
            else:
                # Compatibility for pre-resource clients.  This value is not
                # used to order v1 events and can be removed after migration.
                client_revision = client_revisions.get(topic)
                server_revision = server_topic_revisions.get(topic, 0)
                if isinstance(client_revision, int) and client_revision != server_revision:
                    gaps.append({
                        "topic": topic,
                        "client_revision": client_revision,
                        "server_revision": server_revision,
                        "reason": (
                            "server_revision_reset"
                            if client_revision > server_revision
                            else "revision_gap"
                        ),
                    })

        await self._send_control(
            client,
            "subscribe_ack",
            {
                "subscriptions": sorted(accepted),
                "invalid_topics": sorted(invalid),
                "unauthorized_topics": sorted(unauthorized),
                "server_epoch": self._server_epoch,
                "revisions": server_topic_revisions,
                "resource_revisions": server_resource_revisions,
            },
        )
        client.subscriptions = set(accepted)

        # A broadcast thread may have advanced revisions while send_text was
        # awaiting the ack. The client was deliberately unsubscribed during
        # that window, so explicitly request authoritative REST reconciliation.
        latest_topics, latest_resources = self._revision_snapshot(accepted)
        gap_by_key = {
            (gap.get("topic"), gap.get("resource_id")): gap
            for gap in gaps
        }
        if resource_revisions_supplied:
            for topic in sorted(accepted):
                before = server_resource_revisions.get(topic, {})
                after = latest_resources.get(topic, {})
                for resource_id in sorted(set(before) | set(after)):
                    key = (topic, resource_id)
                    if before.get(resource_id, 0) == after.get(resource_id, 0):
                        continue
                    existing_gap = gap_by_key.get(key)
                    if existing_gap is not None:
                        existing_gap["server_revision"] = after.get(resource_id, 0)
                        existing_gap["reason"] = "revision_changed_during_subscribe"
                        continue
                    gap = {
                        "topic": topic,
                        "resource_id": resource_id,
                        "client_revision": before.get(resource_id, 0),
                        "server_revision": after.get(resource_id, 0),
                        "reason": "revision_changed_during_subscribe",
                    }
                    gaps.append(gap)
                    gap_by_key[key] = gap
        else:
            for topic in sorted(accepted):
                key = (topic, None)
                if (
                    server_topic_revisions.get(topic, 0)
                    == latest_topics.get(topic, 0)
                ):
                    continue
                existing_gap = gap_by_key.get(key)
                if existing_gap is not None:
                    existing_gap["server_revision"] = latest_topics.get(topic, 0)
                    existing_gap["reason"] = "revision_changed_during_subscribe"
                    continue
                gap = {
                    "topic": topic,
                    "client_revision": server_topic_revisions.get(topic, 0),
                    "server_revision": latest_topics.get(topic, 0),
                    "reason": "revision_changed_during_subscribe",
                }
                gaps.append(gap)
                gap_by_key[key] = gap
        if gaps:
            await self._send_control(
                client,
                "resync_required",
                {"reason": "revision_mismatch", "topics": gaps},
            )

    def _revision_snapshot(
        self,
        topics: set[str],
    ) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
        with self._event_lock:
            topic_revisions = {
                topic: self._topic_revisions.get(topic, 0)
                for topic in sorted(topics)
            }
            resource_revisions = {
                topic: {
                    resource_id: revision
                    for (event_topic, resource_id), revision
                    in self._resource_revisions.items()
                    if event_topic == topic
                }
                for topic in sorted(topics)
            }
        return topic_revisions, resource_revisions

    @staticmethod
    def _client_identity_is_current(client: _ClientConnection) -> bool:
        if client.identity_resolver is None:
            return True
        try:
            identity = client.identity_resolver()
        except Exception:
            logger.warning(
                "ws.identity_revalidation_failed",
                client_id=client.client_id,
                username=client.username,
            )
            return False
        if identity is None:
            return False
        return (
            identity.get("username") == client.username
            and identity.get("role") == client.role
        )

    async def _handle_unsubscribe(
        self,
        client: _ClientConnection,
        msg: dict[str, Any],
    ) -> None:
        raw_topics = msg.get("topics", [])
        topics = {
            topic
            for topic in raw_topics
            if isinstance(topic, str)
        } if isinstance(raw_topics, list) else set()
        client.subscription_generation += 1
        client.subscriptions -= topics
        await self._send_control(
            client,
            "unsubscribe_ack",
            {"subscriptions": sorted(client.subscriptions)},
        )

    async def _send_control(
        self,
        client: _ClientConnection,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        event = make_ws_event(
            topic=CONTROL_TOPIC,
            event_type=event_type,
            data=data,
            resource_id=client.client_id,
            revision=0,
        )
        await self._send_text(
            client,
            json.dumps(event, ensure_ascii=False, default=str),
        )

    async def _send_text(
        self,
        client: _ClientConnection,
        message: str,
    ) -> None:
        """Serialize one client send and bound lock/backpressure latency."""

        async def send_locked() -> None:
            async with client.send_lock:
                await client.websocket.send_text(message)

        await asyncio.wait_for(send_locked(), timeout=self._send_timeout_seconds)

    async def _send_topic_text(
        self,
        client: _ClientConnection,
        message: str,
        topic: str,
        subscription_generation: int,
    ) -> None:
        """Send only if the subscription selected by dispatch is still live."""

        async def send_locked() -> None:
            async with client.send_lock:
                if (
                    client.subscription_generation != subscription_generation
                    or topic not in client.subscriptions
                ):
                    return
                await client.websocket.send_text(message)

        await asyncio.wait_for(send_locked(), timeout=self._send_timeout_seconds)

    # ── Internal loops ───────────────────────────────────────────────

    async def _dispatch_loop(self, queue: janus.Queue[dict[str, Any]]) -> None:
        """Drain the async side of the janus queue and broadcast."""
        async_q = queue.async_q

        while True:
            event = await async_q.get()
            topic = str(event["topic"])
            message = json.dumps(event, ensure_ascii=False, default=str)

            async def deliver(
                client_id: str,
                client: _ClientConnection,
                subscription_generation: int,
            ) -> str | None:
                try:
                    if client.websocket.client_state == WebSocketState.CONNECTED:
                        await self._send_topic_text(
                            client,
                            message,
                            topic,
                            subscription_generation,
                        )
                    return None
                except Exception:
                    logger.debug(
                        "ws.broadcast_send_failed",
                        client_id=client_id,
                        exc_info=True,
                    )
                    return client_id

            deliveries = []
            for client_id, client in list(self._connections.items()):
                if topic not in client.subscriptions:
                    continue
                deliveries.append(deliver(
                    client_id,
                    client,
                    client.subscription_generation,
                ))

            dead_clients = [
                client_id
                for client_id in await asyncio.gather(*deliveries)
                if client_id is not None
            ] if deliveries else []
            if dead_clients:
                await asyncio.gather(*(
                    self.disconnect(client_id)
                    for client_id in dead_clients
                ))

    async def _heartbeat_loop(self) -> None:
        """Send periodic pings and disconnect stale clients."""
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            now = time.monotonic()

            async def heartbeat(
                client_id: str,
                client: _ClientConnection,
            ) -> str | None:
                if not self._client_identity_is_current(client):
                    return client_id
                # If no pong received within 2x heartbeat interval, consider dead
                if (
                    client.last_pong > 0
                    and (now - client.last_pong) > self._heartbeat_seconds * 2
                ):
                    return client_id

                try:
                    if client.websocket.client_state == WebSocketState.CONNECTED:
                        heartbeat_event = make_ws_event(
                            topic="ping",
                            event_type="heartbeat_ping",
                            data={},
                            resource_id=client.client_id,
                            revision=0,
                        )
                        await self._send_text(client, json.dumps(heartbeat_event))
                    return None
                except Exception:
                    logger.debug(
                        "ws.heartbeat_send_failed",
                        client_id=client_id,
                        exc_info=True,
                    )
                    return client_id

            heartbeats = [
                heartbeat(client_id, client)
                for client_id, client in list(self._connections.items())
            ]
            dead_clients = [
                client_id
                for client_id in await asyncio.gather(*heartbeats)
                if client_id is not None
            ] if heartbeats else []
            for client_id in dead_clients:
                logger.info("ws.heartbeat_timeout", client_id=client_id)
            if dead_clients:
                await asyncio.gather(*(
                    self.disconnect(client_id)
                    for client_id in dead_clients
                ))

    async def _close_client(self, client: _ClientConnection) -> None:
        """Safely close a client WebSocket."""
        try:
            if client.websocket.client_state == WebSocketState.CONNECTED:

                async def close_locked() -> None:
                    async with client.send_lock:
                        await client.websocket.close()

                await asyncio.wait_for(
                    close_locked(),
                    timeout=self._send_timeout_seconds,
                )
        except Exception:
            logger.debug("ws.client_close_failed", exc_info=True)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


def resolve_ws_identity(
    token: str,
    auth_config: AuthConfig,
    session_secret: str = "",
    database: Database | None = None,
) -> dict[str, str] | None:
    """Authenticate a WebSocket and return the identity used for topic RBAC."""
    from argus.dashboard.auth import (
        refresh_session_identity,
        verify_session_token,
        verify_token,
    )

    if not auth_config.enabled:
        return {"username": "system", "role": "admin", "auth_kind": "system"}

    if verify_token(token, auth_config):
        return {"username": "api", "role": "admin", "auth_kind": "api_token"}

    if session_secret and token:
        max_age = auth_config.session_timeout_minutes * 60
        user_info = verify_session_token(token, session_secret, max_age)
        if user_info is not None:
            identity = refresh_session_identity(user_info, database)
            if identity is not None:
                identity["auth_kind"] = "session"
            return identity

    return None


def verify_ws_token(
    token: str,
    auth_config: AuthConfig,
    session_secret: str = "",
    database: Database | None = None,
) -> bool:
    """Backward-compatible boolean WebSocket authentication helper."""
    return resolve_ws_identity(
        token,
        auth_config,
        session_secret,
        database,
    ) is not None
