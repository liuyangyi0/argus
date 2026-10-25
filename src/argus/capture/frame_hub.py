"""Bounded frame broadcast: independent metadata, shared immutable pixels.

This is an in-process transport, not a high-rate durable queue.  ``latest``
subscribers deliberately replace unconsumed frames; ``ordered_window`` keeps
a bounded FIFO and reports overflow.  Neither can block source capture.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from argus.capture.camera import FrameData

SubscriptionMode = Literal["latest", "ordered_window"]
SourceToken = tuple[str, int]


def copy_frame_metadata(frame: FrameData) -> FrameData:
    """Return task-owned metadata and a view of genuinely read-only pixels."""
    return replace(
        frame,
        frame=frame.frame.view(),
        identity=replace(frame.identity) if frame.identity is not None else None,
    )


def _freeze_frame(frame: FrameData) -> FrameData:
    # setflags(write=False) on an owning ndarray is reversible by a subscriber.
    # An immutable bytes backing store prevents that and detaches capture reuse.
    pixels = np.frombuffer(frame.frame.tobytes(order="C"), dtype=frame.frame.dtype)
    pixels = pixels.reshape(frame.frame.shape)
    return replace(
        frame,
        frame=pixels,
        identity=replace(frame.identity) if frame.identity is not None else None,
    )


@dataclass(frozen=True)
class SubscriptionStatus:
    name: str
    mode: SubscriptionMode
    capacity: int
    queued: int
    dropped_frames: int
    delivered_frames: int
    closed: bool


class FrameSubscription:
    """A consumer-owned FIFO; closing it never closes the source or peers."""

    def __init__(self, name: str, mode: SubscriptionMode, capacity: int) -> None:
        if mode not in ("latest", "ordered_window"):
            raise ValueError("unsupported frame subscription mode")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("subscription capacity must be a positive integer")
        self.name = name
        self.mode = mode
        self.capacity = 1 if mode == "latest" else capacity
        self._condition = threading.Condition()
        self._frames: deque[FrameData] = deque()
        self._closed = False
        self._dropped = 0
        self._delivered = 0

    @property
    def status(self) -> SubscriptionStatus:
        with self._condition:
            return SubscriptionStatus(
                self.name, self.mode, self.capacity, len(self._frames),
                self._dropped, self._delivered, self._closed,
            )

    def _put(self, frame: FrameData) -> None:
        with self._condition:
            if self._closed:
                return
            if len(self._frames) == self.capacity:
                self._frames.popleft()
                self._dropped += 1
            self._frames.append(copy_frame_metadata(frame))
            self._condition.notify()

    def get(self, timeout: float | None = 1.0) -> FrameData | None:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must not be negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._frames and not self._closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if self._closed:
                return None
            self._delivered += 1
            return self._frames.popleft()

    def clear(self) -> None:
        with self._condition:
            self._frames.clear()
            self._condition.notify_all()

    def commit_if_no_drops(self, callback: Callable[[], None]) -> bool:
        """Atomically reject incomplete temporal windows before a short commit.

        SourceRuntime's commit guard must be acquired first, matching the
        source-publish lock order.  This prevents overflow between a status
        check and the externally visible result commit.
        """
        with self._condition:
            if self._closed or self._dropped:
                return False
            callback()
            return True

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._frames.clear()
            self._condition.notify_all()


class FrameHub:
    """Broadcast only frames from the currently active source epoch."""

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._lock = threading.RLock()
        self._subscriptions: dict[str, FrameSubscription] = {}
        self._token: SourceToken | None = None
        self._latest: FrameData | None = None

    @property
    def source_token(self) -> SourceToken | None:
        with self._lock:
            return self._token

    @property
    def latest_frame(self) -> FrameData | None:
        with self._lock:
            return copy_frame_metadata(self._latest) if self._latest is not None else None

    def activate(self, source_session_id: str, source_epoch: int) -> None:
        if not source_session_id or source_epoch < 1:
            raise ValueError("an active source requires a session and positive epoch")
        with self._lock:
            self._token = (source_session_id, source_epoch)
            self._latest = None
            for subscription in self._subscriptions.values():
                subscription.clear()

    def invalidate(self) -> None:
        """Revoke pending results and clear old frames without closing consumers."""
        with self._lock:
            self._token = None
            self._latest = None
            for subscription in self._subscriptions.values():
                subscription.clear()

    def subscribe(
        self, name: str, *, mode: SubscriptionMode = "latest", capacity: int = 32,
    ) -> FrameSubscription:
        if not name:
            raise ValueError("subscription name is required")
        with self._lock:
            existing = self._subscriptions.get(name)
            if existing is not None and not existing.status.closed:
                raise ValueError(f"subscription already registered: {name}")
            subscription = FrameSubscription(name, mode, capacity)
            self._subscriptions[name] = subscription
            return subscription

    def unsubscribe(
        self, name: str, *, subscription: FrameSubscription | None = None,
    ) -> None:
        with self._lock:
            current = self._subscriptions.get(name)
            if subscription is not None and current is not subscription:
                return
            self._subscriptions.pop(name, None)
            if current is not None:
                current.close()

    def publish(self, frame: FrameData) -> bool:
        identity = frame.identity
        if (
            identity is None or frame.camera_id != self.camera_id
            or identity.camera_id != self.camera_id
        ):
            return False
        token = (identity.source_session_id, identity.source_epoch)
        # Copy outside the lock, then recheck the epoch before exposing pixels.
        frozen = _freeze_frame(frame)
        with self._lock:
            if self._token is None or token != self._token:
                return False
            self._latest = frozen
            for subscription in self._subscriptions.values():
                subscription._put(frozen)
            return True
