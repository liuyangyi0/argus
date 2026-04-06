"""Formalized degradation state machine for inference pipelines (5.3).

Each camera's inference runner holds a DegradationStateMachine that tracks
the current operational state. Transitions are validated against a fixed
table and logged both via structlog and to the audit trail.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
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

    @property
    def state(self) -> DegradationState:
        return self._state

    def transition(self, to: DegradationState, reason: str) -> bool:
        """Attempt a state transition.

        Returns True if the transition was valid and applied, False otherwise.
        """
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
        events = list(self._events)
        return events[-n:] if len(events) > n else events

    def reset(self) -> None:
        """Reset to NOMINAL state (e.g. after operator intervention)."""
        if self._state != DegradationState.NOMINAL:
            self.transition(DegradationState.NOMINAL, "manual_reset")
