"""Lightweight in-process event bus for decoupling pipeline components.

Replaces direct method calls between pipeline stages so that new
subscribers (metrics, recording, notifications) can be added without
touching the core pipeline code.

Design: synchronous publish (caller's thread) with optional async
bridge via janus queue for FastAPI/WebSocket consumers.

Usage::

    from argus.core.event_bus import EventBus, FrameAnalyzed

    bus = EventBus()
    bus.subscribe(FrameAnalyzed, my_handler)
    bus.publish(FrameAnalyzed(camera_id="cam-01", ...))
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import structlog

if TYPE_CHECKING:
    from argus.config.schema import AlertSeverity

logger = structlog.get_logger()

T = TypeVar("T", bound="Event")

# Immutable empty mapping used as default for dict-typed event fields.
_EMPTY_MAP: MappingProxyType[str, float] = MappingProxyType({})


# ── Base event ──

@dataclass(frozen=True, slots=True)
class Event:
    """Base class for all domain events."""

    timestamp: float = field(default_factory=time.time)


# ── Domain events ──

@dataclass(frozen=True, slots=True)
class FrameAnalyzed(Event):
    """Emitted after a frame passes through the full detection pipeline."""

    camera_id: str = ""
    frame_number: int = 0
    anomaly_score: float = 0.0
    is_anomalous: bool = False
    stage_latencies: MappingProxyType[str, float] = field(default_factory=lambda: _EMPTY_MAP)
    mog2_skipped: bool = False
    person_detected: bool = False


@dataclass(frozen=True, slots=True)
class AlertRaised(Event):
    """Emitted when AlertGrader produces an alert."""

    alert_id: str = ""
    camera_id: str = ""
    zone_id: str = ""
    severity: AlertSeverity | str = ""
    anomaly_score: float = 0.0
    handling_policy: str = ""
    corroborated: bool = False


@dataclass(frozen=True, slots=True)
class AlertDispatched(Event):
    """Emitted after an alert is sent to all channels."""

    alert_id: str = ""
    db_ok: bool = True
    webhook_ok: bool = True
    ws_ok: bool = True


@dataclass(frozen=True, slots=True)
class CameraConnected(Event):
    """Emitted when a camera establishes a connection."""

    camera_id: str = ""


@dataclass(frozen=True, slots=True)
class CameraDisconnected(Event):
    """Emitted when a camera loses its connection."""

    camera_id: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ModelLoaded(Event):
    """Emitted when an anomaly model is loaded or reloaded."""

    camera_id: str = ""
    model_type: str = ""
    model_path: str = ""
    load_time_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class DriftDetected(Event):
    """Emitted when score-distribution drift is detected."""

    camera_id: str = ""
    ks_statistic: float = 0.0
    p_value: float = 0.0


@dataclass(frozen=True, slots=True)
class PipelineStageCompleted(Event):
    """Emitted after each pipeline stage finishes (for detailed tracing)."""

    camera_id: str = ""
    stage: str = ""
    duration_seconds: float = 0.0
    frame_number: int = 0


@dataclass(frozen=True, slots=True)
class RetrainingTriggered(Event):
    """Emitted when active learning has accumulated enough labeled frames.

    Downstream consumers (scheduler, trainer) should initiate model
    retraining when they receive this event.
    """

    reason: str = ""  # "active_learning", "scheduled", "manual"
    labeled_count: int = 0
    cameras: tuple[str, ...] | list[str] = ()


@dataclass(frozen=True, slots=True)
class UncertainFrameDetected(Event):
    """Emitted when a frame has high prediction uncertainty (entropy).

    Active learning: these frames are candidates for operator labeling
    because the model is least confident about them, providing maximal
    information gain if labeled.
    """

    camera_id: str = ""
    frame_number: int = 0
    anomaly_score: float = 0.0
    entropy: float = 0.0  # Shannon entropy of ensemble or score distribution
    frame_path: str = ""  # saved frame path for labeling queue


# ── Handler type ──

EventHandler = Callable[[Any], None]


# ── Bus implementation ──

class EventBus:
    """Thread-safe publish/subscribe event bus.

    Uses copy-on-write tuples so ``publish()`` never acquires a lock.
    Only ``subscribe``/``unsubscribe`` (rare, startup-only) hold the lock.
    """

    def __init__(self) -> None:
        # Copy-on-write: values are immutable tuples swapped atomically.
        self._handlers: dict[type, tuple[EventHandler, ...]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        """Register *handler* to be called whenever *event_type* is published."""
        with self._lock:
            current = self._handlers.get(event_type, ())
            if handler not in current:
                self._handlers[event_type] = current + (handler,)

    def unsubscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        """Remove a previously registered handler."""
        with self._lock:
            current = self._handlers.get(event_type, ())
            filtered = tuple(h for h in current if h != handler)
            if filtered:
                self._handlers[event_type] = filtered
            elif event_type in self._handlers:
                del self._handlers[event_type]

    def has_subscribers(self, event_type: type) -> bool:
        """Fast check — no lock needed since tuple is immutable."""
        return bool(self._handlers.get(event_type))

    def publish(self, event: Event) -> None:
        """Deliver *event* to all registered handlers for its type.

        Lock-free on the hot path: reads an immutable tuple reference.
        Exceptions in individual handlers are logged but do not
        propagate — one broken subscriber cannot crash the pipeline.
        """
        handlers = self._handlers.get(type(event), ())

        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.error(
                    "event_bus.handler_error",
                    event_type=type(event).__name__,
                    handler=getattr(handler, "__qualname__", str(handler)),
                    exc_info=True,
                )

    def clear(self) -> None:
        """Remove all subscriptions (useful in tests)."""
        with self._lock:
            self._handlers.clear()
