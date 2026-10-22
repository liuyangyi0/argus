"""Authenticated same-origin WebSocket bridge for browser video streams."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urlencode, urlsplit

import structlog
from fastapi import WebSocket
from websockets.asyncio.client import connect

from argus.streaming.preview_gateway import Go2RTCProxyTarget

logger = structlog.get_logger()

_ALLOWED_MODES = frozenset({"webrtc", "mse"})
_MAX_MESSAGE_BYTES = 16 * 1024 * 1024


def is_same_origin_websocket(browser: WebSocket) -> bool:
    """Validate a browser WebSocket Origin against its public request URL."""
    origin = browser.headers.get("origin", "").strip()
    host = browser.headers.get("host", "").strip()
    if not origin or not host:
        return False
    parsed = urlsplit(origin)
    if parsed.username is not None or parsed.password is not None:
        return False
    expected_scheme = "https" if browser.url.scheme == "wss" else "http"
    return (
        parsed.scheme.lower() == expected_scheme
        and parsed.netloc.lower() == host.lower()
        and parsed.path in ("", "/")
        and not parsed.query
        and not parsed.fragment
    )


def build_go2rtc_websocket_uri(target: Go2RTCProxyTarget, mode: str) -> str:
    """Build a loopback-only upstream URI from a validated camera target."""
    if mode not in _ALLOWED_MODES:
        raise ValueError(f"unsupported stream mode: {mode}")
    if not target.stream_name:
        raise ValueError("stream name is required")
    if not 1 <= target.api_port <= 65535:
        raise ValueError("go2rtc API port is out of range")

    query = {"src": target.stream_name}
    if mode == "mse":
        query["mode"] = "mse"
    return f"ws://127.0.0.1:{target.api_port}/api/ws?{urlencode(query)}"


async def bridge_websockets(
    browser: WebSocket,
    upstream: Any,
    *,
    max_message_bytes: int = _MAX_MESSAGE_BYTES,
) -> None:
    """Copy frames bidirectionally until either side disconnects.

    This helper is separate from connection establishment so the framing,
    cancellation, and size-limit behaviour can be fault-injection tested.
    """

    async def browser_to_upstream() -> None:
        while True:
            message = await browser.receive()
            message_type = message.get("type")
            if message_type == "websocket.disconnect":
                return
            if message_type != "websocket.receive":
                continue
            payload = message.get("text")
            if payload is None:
                payload = message.get("bytes")
            if payload is None:
                continue
            size = len(payload.encode("utf-8")) if isinstance(payload, str) else len(payload)
            if size > max_message_bytes:
                await browser.close(code=1009, reason="Stream message too large")
                return
            await upstream.send(payload)

    async def upstream_to_browser() -> None:
        async for payload in upstream:
            if isinstance(payload, str):
                await browser.send_text(payload)
            else:
                data = bytes(payload)
                if len(data) > max_message_bytes:
                    await browser.close(code=1009, reason="Stream message too large")
                    return
                await browser.send_bytes(data)

    tasks = {
        asyncio.create_task(browser_to_upstream()),
        asyncio.create_task(upstream_to_browser()),
    }
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        # The bridge itself can be cancelled when authorization is revoked or
        # the downstream connection closes.  Always reap both pump tasks so
        # neither side keeps reading from a socket after its owner is gone.
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def proxy_go2rtc_websocket(
    browser: WebSocket,
    target: Go2RTCProxyTarget,
    mode: str,
) -> None:
    """Connect to local go2rtc and bridge an already-authorized browser."""
    upstream_uri = build_go2rtc_websocket_uri(target, mode)
    logger.info(
        "stream_proxy.connect",
        camera_id=target.camera_id,
        mode=mode,
        api_port=target.api_port,
    )
    async with connect(
        upstream_uri,
        open_timeout=5,
        close_timeout=2,
        max_size=_MAX_MESSAGE_BYTES,
        ping_interval=20,
        ping_timeout=20,
    ) as upstream:
        await bridge_websockets(browser, upstream)
