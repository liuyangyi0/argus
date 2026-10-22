"""Tests for the authenticated same-origin go2rtc WebSocket bridge."""

from __future__ import annotations

import asyncio

import pytest

from argus.streaming.preview_gateway import Go2RTCProxyTarget
from argus.streaming.websocket_proxy import (
    bridge_websockets,
    build_go2rtc_websocket_uri,
    is_same_origin_websocket,
)


def _target(**overrides) -> Go2RTCProxyTarget:
    values = {
        "camera_id": "cam/01",
        "stream_name": "line one/cam&01",
        "api_port": 11984,
    }
    values.update(overrides)
    return Go2RTCProxyTarget(**values)


def test_build_upstream_uri_is_loopback_and_encodes_stream_name():
    uri = build_go2rtc_websocket_uri(_target(), "mse")

    assert uri == (
        "ws://127.0.0.1:11984/api/ws?"
        "src=line+one%2Fcam%2601&mode=mse"
    )
    assert "cam/01" not in uri


def test_build_webrtc_uri_does_not_add_mse_mode():
    uri = build_go2rtc_websocket_uri(_target(), "webrtc")

    assert uri == "ws://127.0.0.1:11984/api/ws?src=line+one%2Fcam%2601"


class _OriginWebSocket:
    def __init__(self, origin: str | None, host: str, scheme: str = "ws"):
        self.headers = {"host": host}
        if origin is not None:
            self.headers["origin"] = origin
        self.url = type("Url", (), {"scheme": scheme})()


@pytest.mark.parametrize(
    ("browser", "expected"),
    [
        (_OriginWebSocket("http://edge.local:8080", "edge.local:8080"), True),
        (_OriginWebSocket("https://edge.local", "edge.local", "wss"), True),
        (_OriginWebSocket("https://evil.local", "edge.local", "wss"), False),
        (_OriginWebSocket("http://edge.local", "edge.local", "wss"), False),
        (_OriginWebSocket(None, "edge.local"), False),
        (_OriginWebSocket("http://edge.local/path", "edge.local"), False),
    ],
)
def test_same_origin_websocket_validation(browser, expected):
    assert is_same_origin_websocket(browser) is expected


@pytest.mark.parametrize(
    ("target", "mode"),
    [
        (_target(api_port=0), "mse"),
        (_target(api_port=65536), "webrtc"),
        (_target(stream_name=""), "mse"),
        (_target(), "hls"),
    ],
)
def test_build_upstream_uri_rejects_invalid_targets(target, mode):
    with pytest.raises(ValueError):
        build_go2rtc_websocket_uri(target, mode)


class _Browser:
    def __init__(self, messages: list[dict], receive_gate: asyncio.Event | None = None):
        self.messages = list(messages)
        self.receive_gate = receive_gate
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []
        self.closed: list[tuple[int, str]] = []

    async def receive(self):
        if self.messages:
            return self.messages.pop(0)
        if self.receive_gate is not None:
            await self.receive_gate.wait()
        return {"type": "websocket.disconnect"}

    async def send_text(self, payload: str):
        self.sent_text.append(payload)

    async def send_bytes(self, payload: bytes):
        self.sent_bytes.append(payload)
        if self.receive_gate is not None:
            self.receive_gate.set()

    async def close(self, code: int, reason: str):
        self.closed.append((code, reason))


class _Upstream:
    def __init__(self, payloads: list[str | bytes]):
        self.payloads = list(payloads)
        self.sent: list[str | bytes] = []
        self._wait_forever = asyncio.Event()

    async def send(self, payload: str | bytes):
        self.sent.append(payload)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.payloads:
            return self.payloads.pop(0)
        await self._wait_forever.wait()
        raise StopAsyncIteration


class _CancellationAwareBrowser(_Browser):
    def __init__(self):
        super().__init__([])
        self.receive_started = asyncio.Event()
        self.receive_cancelled = asyncio.Event()
        self._wait_forever = asyncio.Event()

    async def receive(self):
        self.receive_started.set()
        try:
            await self._wait_forever.wait()
        except asyncio.CancelledError:
            self.receive_cancelled.set()
            raise
        return {"type": "websocket.disconnect"}


class _CancellationAwareUpstream(_Upstream):
    def __init__(self):
        super().__init__([])
        self.receive_started = asyncio.Event()
        self.receive_cancelled = asyncio.Event()

    async def __anext__(self):
        self.receive_started.set()
        try:
            return await super().__anext__()
        except asyncio.CancelledError:
            self.receive_cancelled.set()
            raise


@pytest.mark.asyncio
async def test_bridge_copies_text_and_binary_in_both_directions():
    gate = asyncio.Event()
    browser = _Browser(
        [{"type": "websocket.receive", "text": "offer", "bytes": None}],
        receive_gate=gate,
    )
    upstream = _Upstream([b"segment"])

    await bridge_websockets(browser, upstream)

    assert upstream.sent == ["offer"]
    assert browser.sent_bytes == [b"segment"]
    assert browser.closed == []


@pytest.mark.asyncio
async def test_bridge_rejects_oversized_browser_message():
    browser = _Browser(
        [{"type": "websocket.receive", "bytes": b"12345", "text": None}],
    )
    upstream = _Upstream([])

    await bridge_websockets(browser, upstream, max_message_bytes=4)

    assert upstream.sent == []
    assert browser.closed == [(1009, "Stream message too large")]


@pytest.mark.asyncio
async def test_bridge_cancellation_reaps_both_pump_tasks():
    browser = _CancellationAwareBrowser()
    upstream = _CancellationAwareUpstream()
    bridge_task = asyncio.create_task(bridge_websockets(browser, upstream))

    await asyncio.wait_for(
        asyncio.gather(
            browser.receive_started.wait(),
            upstream.receive_started.wait(),
        ),
        timeout=1,
    )
    bridge_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(bridge_task, timeout=1)

    assert browser.receive_cancelled.is_set()
    assert upstream.receive_cancelled.is_set()
