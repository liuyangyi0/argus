"""Formalized degradation state machine for inference pipelines (5.3).

Each camera's inference runner holds a DegradationStateMachine that tracks
the current operational state. Transitions are validated against a fixed
table and logged both via structlog and to the audit trail.

The GlobalDegradationManager (UX v2 §5) tracks system-wide degradation
events with pre-written Chinese copy templates, for the global degradation
bar shown at the top of all dashboard pages.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from argus.dashboard.websocket import ConnectionManager
    from argus.storage.audit import AuditLogger

logger = structlog.get_logger()


class DegradationState(str, Enum):
    """Operational state of a camera inference runner."""

    NOMINAL = "nominal"
    DEGRADED_SEGMENTER = "degraded_segmenter"  # SAM2 failed/timed out
    DEGRADED_DETECTOR = "degraded_detector"  # anomaly model reload failed
    BACKBONE_FAILED = "backbone_failed"  # model could not load at all
    RESTARTING = "restarting"  # attempting model reload


class LockState(str, Enum):
    """Anomaly region lock state for FR-005."""

    UNLOCKED = "unlocked"
    LOCKED = "locked"
    CLEARING = "clearing"  # below hysteresis threshold, timer counting


@dataclass(frozen=True)
class DegradationEvent:
    """Immutable record of a degradation state transition."""

    timestamp: float
    from_state: DegradationState
    to_state: DegradationState
    reason: str
    camera_id: str


# Valid state transitions — anything not listed here is rejected.
_VALID_TRANSITIONS: dict[DegradationState, set[DegradationState]] = {
    DegradationState.NOMINAL: {
        DegradationState.DEGRADED_SEGMENTER,
        DegradationState.DEGRADED_DETECTOR,
        DegradationState.BACKBONE_FAILED,
        DegradationState.RESTARTING,
    },
    DegradationState.DEGRADED_SEGMENTER: {
        DegradationState.NOMINAL,
        DegradationState.RESTARTING,
        DegradationState.DEGRADED_DETECTOR,
    },
    DegradationState.DEGRADED_DETECTOR: {
        DegradationState.NOMINAL,
        DegradationState.RESTARTING,
        DegradationState.BACKBONE_FAILED,
    },
    DegradationState.BACKBONE_FAILED: {
        DegradationState.RESTARTING,
        DegradationState.NOMINAL,
    },
    DegradationState.RESTARTING: {
        DegradationState.NOMINAL,
        DegradationState.DEGRADED_SEGMENTER,
        DegradationState.DEGRADED_DETECTOR,
        DegradationState.BACKBONE_FAILED,
    },
}


class DegradationStateMachine:
    """Manages degradation state transitions with validation and audit logging.

    Args:
        camera_id: Camera identifier for log context.
        audit_logger: Optional AuditLogger for compliance trail.
        max_events: Maximum recent events to retain in memory.
    """

    def __init__(
        self,
        camera_id: str,
        audit_logger: AuditLogger | None = None,
        max_events: int = 100,
    ):
        self._camera_id = camera_id
        self._audit_logger = audit_logger
        self._state = DegradationState.NOMINAL
        self._events: deque[DegradationEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    @property
    def state(self) -> DegradationState:
        with self._lock:
            return self._state

    def transition(self, to: DegradationState, reason: str) -> bool:
        """Attempt a state transition.

        Returns True if the transition was valid and applied, False otherwise.
        Thread-safe: protected by internal lock.
        """
        with self._lock:
            from_state = self._state

            if to == from_state:
                return True  # no-op, already in target state

            valid_targets = _VALID_TRANSITIONS.get(from_state, set())
            if to not in valid_targets:
                logger.error(
                    "degradation.invalid_transition",
                    camera_id=self._camera_id,
                    from_state=from_state.value,
                    to_state=to.value,
                    reason=reason,
                )
                return False

            self._state = to
            event = DegradationEvent(
                timestamp=time.time(),
                from_state=from_state,
                to_state=to,
                reason=reason,
                camera_id=self._camera_id,
            )
            self._events.append(event)

        logger.warning(
            "degradation.transition",
            camera_id=self._camera_id,
            from_state=from_state.value,
            to_state=to.value,
            reason=reason,
        )

        if self._audit_logger:
            try:
                self._audit_logger.log(
                    user="system",
                    action="degradation_transition",
                    target_type="camera",
                    target_id=self._camera_id,
                    detail=f"{from_state.value} -> {to.value}: {reason}",
                )
            except Exception:
                logger.error(
                    "degradation.audit_failed",
                    camera_id=self._camera_id,
                )

        return True

    def get_recent_events(self, n: int = 20) -> list[DegradationEvent]:
        """Return the most recent N degradation events."""
        with self._lock:
            events = list(self._events)
        return events[-n:] if len(events) > n else events

    def reset(self) -> None:
        """Reset to NOMINAL state (e.g. after operator intervention)."""
        if self._state != DegradationState.NOMINAL:
            self.transition(DegradationState.NOMINAL, "manual_reset")


# ── UX v2 §5: Global Degradation Bar ──


class DegradationLevel(str, Enum):
    """Severity level for the global degradation bar."""

    INFO = "info"  # Blue — suggestion, not a failure
    WARNING = "warning"  # Yellow — single subsystem degraded
    MODERATE = "moderate"  # Orange — core capability down
    SEVERE = "severe"  # Red — partial functionality lost


# Canonical ordering: lower number = higher severity
_DEGRADATION_LEVEL_ORDER: dict[DegradationLevel, int] = {
    DegradationLevel.SEVERE: 0,
    DegradationLevel.MODERATE: 1,
    DegradationLevel.WARNING: 2,
    DegradationLevel.INFO: 3,
}


@dataclass
class GlobalDegradationEvent:
    """A system-wide degradation event with pre-written copy.

    These events are displayed in the global degradation bar at the
    top of all dashboard pages (48px height when active).
    """

    event_id: str
    level: DegradationLevel
    category: str  # yolo_down, model_fallback, rtsp_broken, etc.
    camera_id: str | None  # None for system-wide events
    title: str  # Pre-written Chinese copy
    impact: str  # What this means for the operator
    action: str  # What the operator should do
    started_at: float = field(default_factory=time.time)
    resolved_at: float | None = None


# Pre-written degradation copy templates (§4.2)
# Keys match the `category` field of GlobalDegradationEvent.
DEGRADATION_TEMPLATES: dict[str, dict[str, Any]] = {
    "yolo_down": {
        "level": DegradationLevel.WARNING,
        "title": "YOLO 人员过滤暂不可用",
        "impact": "人员在场时可能产生误报",
        "action": "如频繁误报可临时提高告警阈值",
    },
    "model_fallback": {
        "level": DegradationLevel.SEVERE,
        "title": "主检测模型不可用，已切换至 Simplex 兜底",
        "impact": "无法检测细微异常",
        "action": "降级期间告警已标记，联系工程师",
    },
    "circuit_breaker": {
        "level": DegradationLevel.MODERATE,
        "title": "外部告警通道已熔断",
        "impact": "告警仍在检测，但 Webhook/邮件暂停",
        "action": "系统将自动恢复，请关注本地告警",
    },
    "rtsp_broken": {
        "level": DegradationLevel.WARNING,
        "title": "{camera} 视频流已断开",
        "impact": "该摄像头无法检测",
        "action": "正在重连 ({n}/{max})，上次画面 {t}s 前",
    },
    "baseline_drift": {
        "level": DegradationLevel.INFO,
        "title": "{camera} 检测到评分分布漂移",
        "impact": "可能影响检测准确度",
        "action": "建议启动新基线采集",
    },
    "storage_low": {
        "level": DegradationLevel.SEVERE,
        "title": "磁盘空间严重不足 ({free}GB)",
        "impact": "无法保存告警证据",
        "action": "立即清理或扩容",
    },
}


class GlobalDegradationManager:
    """System-wide degradation event tracker (UX v2 §5).

    Manages active degradation events and pushes changes to the
    dashboard via WebSocket. The global degradation bar displays
    up to 3 active events; more collapse to "+N degradations".

    Thread-safe — can be called from pipeline threads, health monitor,
    or alert dispatcher.
    """

    def __init__(
        self,
        ws_manager: ConnectionManager | None = None,
        max_history: int = 1000,
    ):
        self._ws_manager = ws_manager
        self._active: dict[str, GlobalDegradationEvent] = {}
        self._history: deque[GlobalDegradationEvent] = deque(maxlen=max_history)
        self._lock = threading.Lock()

    def report(
        self,
        category: str,
        camera_id: str | None = None,
        **format_kwargs: Any,
    ) -> str:
        """Report a new degradation event using a pre-written template.

        Args:
            category: Template key from DEGRADATION_TEMPLATES.
            camera_id: Camera ID if camera-specific, None for system-wide.
            **format_kwargs: Variables for string formatting (camera, n, max, t, free).

        Returns:
            The event_id of the created event.
        """
        template = DEGRADATION_TEMPLATES.get(category)
        if template is None:
            logger.warning("degradation.unknown_category", category=category)
            # Create a generic event
            template = {
                "level": DegradationLevel.WARNING,
                "title": f"未知降级: {category}",
                "impact": "系统部分功能可能受影响",
                "action": "请联系工程师",
            }

        event_id = f"deg-{uuid.uuid4().hex[:12]}"

        # Format the copy with provided variables
        fmt = {**format_kwargs}
        if camera_id:
            fmt.setdefault("camera", camera_id)

        event = GlobalDegradationEvent(
            event_id=event_id,
            level=template["level"],
            category=category,
            camera_id=camera_id,
            title=template["title"].format(**fmt) if fmt else template["title"],
            impact=template["impact"].format(**fmt) if fmt else template["impact"],
            action=template["action"].format(**fmt) if fmt else template["action"],
        )

        with self._lock:
            # Check for existing event of same category + camera (avoid duplicates)
            existing_key = f"{category}:{camera_id or 'system'}"
            for eid, existing in list(self._active.items()):
                if f"{existing.category}:{existing.camera_id or 'system'}" == existing_key:
                    # Update existing rather than creating duplicate
                    self._active.pop(eid)
                    break
            self._active[event_id] = event
            self._history.append(event)

        logger.warning(
            "degradation.reported",
            event_id=event_id,
            category=category,
            camera_id=camera_id,
            level=event.level.value,
            title=event.title,
        )

        # Push to WebSocket
        if self._ws_manager is not None:
            self._ws_manager.broadcast(
                topic="degradation",
                data={
                    "type": "degradation.new",
                    "event": self._event_to_dict(event),
                },
            )

        return event_id

    def resolve(self, event_id: str) -> bool:
        """Resolve (clear) a degradation event.

        Returns True if the event was found and resolved.
        """
        with self._lock:
            event = self._active.pop(event_id, None)
            if event is None:
                return False
            event.resolved_at = time.time()

        logger.info(
            "degradation.resolved",
            event_id=event_id,
            category=event.category,
            camera_id=event.camera_id,
            duration_s=round(event.resolved_at - event.started_at, 1),
        )

        if self._ws_manager is not None:
            self._ws_manager.broadcast(
                topic="degradation",
                data={
                    "type": "degradation.resolved",
                    "event_id": event_id,
                },
            )

        return True

    def resolve_by_category(self, category: str, camera_id: str | None = None) -> int:
        """Resolve all active events matching category and optional camera_id.

        Resolves all matching events under a single lock acquisition.
        Returns number of events resolved.
        """
        resolved_events: list[GlobalDegradationEvent] = []
        with self._lock:
            to_remove = [
                eid for eid, e in self._active.items()
                if e.category == category
                and (camera_id is None or e.camera_id == camera_id)
            ]
            for eid in to_remove:
                event = self._active.pop(eid, None)
                if event is not None:
                    event.resolved_at = time.time()
                    resolved_events.append(event)

        # Broadcast outside lock
        for event in resolved_events:
            logger.info(
                "degradation.resolved",
                event_id=event.event_id,
                category=event.category,
                camera_id=event.camera_id,
                duration_s=round(event.resolved_at - event.started_at, 1),
            )
            if self._ws_manager is not None:
                self._ws_manager.broadcast(
                    topic="degradation",
                    data={"type": "degradation.resolved", "event_id": event.event_id},
                )
        return len(resolved_events)

    def get_active(self) -> list[dict]:
        """Return all active degradation events as dicts."""
        with self._lock:
            events = sorted(
                self._active.values(),
                key=lambda e: (
                    _DEGRADATION_LEVEL_ORDER.get(e.level, 4),
                    e.started_at,
                ),
            )
            return [self._event_to_dict(e) for e in events]

    def get_history(self, days: int = 7) -> list[dict]:
        """Return degradation event history within the last N days."""
        cutoff = time.time() - (days * 86400)
        with self._lock:
            return [
                self._event_to_dict(e)
                for e in self._history
                if e.started_at >= cutoff
            ]

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def max_active_level(self) -> DegradationLevel | None:
        """Return the highest severity level among active events."""
        with self._lock:
            if not self._active:
                return None
            return min(
                (e.level for e in self._active.values()),
                key=lambda l: _DEGRADATION_LEVEL_ORDER.get(l, 99),
            )

    @staticmethod
    def _event_to_dict(event: GlobalDegradationEvent) -> dict:
        return {
            "event_id": event.event_id,
            "level": event.level.value,
            "category": event.category,
            "camera_id": event.camera_id,
            "title": event.title,
            "impact": event.impact,
            "action": event.action,
            "started_at": event.started_at,
            "resolved_at": event.resolved_at,
        }
