"""WebSocket connection manager for real-time dashboard updates.

Bridges synchronous data sources (camera threads, alert dispatcher) to
async WebSocket clients via a ``janus`` sync/async queue — replacing the
fragile ``call_soon_threadsafe`` approach.

Topics:
- health:      System health and camera status changes
- cameras:     Camera statistics updates
- alerts:      New alert notifications
- tasks:       Background task progress updates
- wall:        Video wall aggregated score/status (UX v2 §2)
- degradation: Degradation events new/resolved (UX v2 §5)
- heatmap:     Per-camera anomaly heatmap data for frontend Canvas overlay
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

import janus
import structlog
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

if TYPE_CHECKING:
    from argus.config.schema import AuthConfig

logger = structlog.get_logger()

VALID_TOPICS = frozenset({
    "health", "cameras", "alerts", "tasks", "wall", "degradation", "heatmap",
})


class _ClientConnection:
    """Represents a single WebSocket client."""

    __slots__ = ("client_id", "websocket", "subscriptions", "last_pong")

    def __init__(self, client_id: str, websocket: WebSocket, subscriptions: set[str]):
        self.client_id = client_id
        self.websocket = websocket
        self.subscriptions = subscriptions
        self.last_pong = time.monotonic()


class ConnectionManager:
    """Manages WebSocket connections with topic-based subscriptions.

    Thread-safe: the ``broadcast`` method can be called from **any** thread.
    Events are enqueued into the sync side of a ``janus`` queue and drained
    by an asyncio task on the event-loop side.
    """

    def __init__(self, heartbeat_seconds: int = 30, max_connections: int = 100):
        self._connections: dict[str, _ClientConnection] = {}
        self._heartbeat_seconds = heartbeat_seconds
        self._max_connections = max_connections

        # Initialised in start() when the event loop is running
        self._queue: janus.Queue[tuple[str, dict]] | None = None
        self._broadcast_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the broadcast dispatcher loop."""
        self._queue = janus.Queue(maxsize=1000)
        self._broadcast_task = asyncio.create_task(self._dispatch_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("ws.manager_started", max_connections=self._max_connections)

    async def stop(self) -> None:
        """Stop the manager and close all connections."""
        for task in (self._broadcast_task, self._heartbeat_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        for client in list(self._connections.values()):
            await self._close_client(client)
        self._connections.clear()

        if self._queue is not None:
            self._queue.close()
            await self._queue.wait_closed()
            self._queue = None

        logger.info("ws.manager_stopped")

    # ── Client management ────────────────────────────────────────────

    async def accept(
        self,
        websocket: WebSocket,
        client_id: str,
    ) -> _ClientConnection:
        """Accept a new WebSocket connection."""
        if len(self._connections) >= self._max_connections:
            await websocket.close(code=1013, reason="Max connections reached")
            raise WebSocketDisconnect(code=1013)

        await websocket.accept()

        client = _ClientConnection(
            client_id=client_id,
            websocket=websocket,
            subscriptions=set(VALID_TOPICS),  # subscribe to all by default
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

    # ── Thread-safe broadcast ────────────────────────────────────────

    def broadcast(self, topic: str, data: dict) -> None:
        """Thread-safe broadcast: enqueue an event for all subscribers.

        Can be called from **any** thread (camera threads, alert dispatcher,
        etc.).  Uses the synchronous side of a ``janus`` queue so there is
        no need for ``call_soon_threadsafe`` or direct event-loop access.
        """
        if topic not in VALID_TOPICS:
            return
        if self._queue is None:
            return

        try:
            self._queue.sync_q.put_nowait((topic, data))
        except janus.SyncQueueFull:
            logger.warning("ws.broadcast_queue_full", topic=topic)

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

                action = msg.get("action")
                if action == "subscribe":
                    topics = msg.get("topics", [])
                    client.subscriptions = {t for t in topics if t in VALID_TOPICS}
                elif action == "unsubscribe":
                    topics = msg.get("topics", [])
                    client.subscriptions -= set(topics)
                elif action == "pong":
                    client.last_pong = time.monotonic()

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug("ws.client_error", client_id=client.client_id, error=str(e))

    # ── Internal loops ───────────────────────────────────────────────

    async def _dispatch_loop(self) -> None:
        """Drain the async side of the janus queue and broadcast."""
        assert self._queue is not None
        async_q = self._queue.async_q

        while True:
            topic, data = await async_q.get()

            message = json.dumps({
                "topic": topic,
                "data": data,
                "timestamp": time.time(),
            }, ensure_ascii=False, default=str)

            dead_clients: list[str] = []
            for client_id, client in self._connections.items():
                if topic not in client.subscriptions:
                    continue
                try:
                    if client.websocket.client_state == WebSocketState.CONNECTED:
                        await client.websocket.send_text(message)
                except Exception:
                    dead_clients.append(client_id)

            for client_id in dead_clients:
                await self.disconnect(client_id)

    async def _heartbeat_loop(self) -> None:
        """Send periodic pings and disconnect stale clients."""
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            now = time.monotonic()
            dead_clients: list[str] = []

            for client_id, client in self._connections.items():
                # If no pong received within 2x heartbeat interval, consider dead
                if client.last_pong > 0 and (now - client.last_pong) > self._heartbeat_seconds * 2:
                    dead_clients.append(client_id)
                    continue

                try:
                    if client.websocket.client_state == WebSocketState.CONNECTED:
                        await client.websocket.send_text(
                            json.dumps({"topic": "ping", "timestamp": time.time()})
                        )
                except Exception:
                    dead_clients.append(client_id)

            for client_id in dead_clients:
                logger.info("ws.heartbeat_timeout", client_id=client_id)
                await self.disconnect(client_id)

    async def _close_client(self, client: _ClientConnection) -> None:
        """Safely close a client WebSocket."""
        try:
            if client.websocket.client_state == WebSocketState.CONNECTED:
                await client.websocket.close()
        except Exception:
            pass  # close errors are expected during shutdown

    @property
    def connection_count(self) -> int:
        return len(self._connections)


def verify_ws_token(token: str, auth_config: AuthConfig) -> bool:
    """Verify a WebSocket authentication token.

    Delegates to the shared verify_token() in auth module.
    """
    from argus.dashboard.auth import verify_token
    return verify_token(token, auth_config)
