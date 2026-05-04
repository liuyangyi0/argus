"""Tests for the unified runtime ErrorChannel.

Covers:
    1. emit() before publisher injection → buffered in memory
    2. set_publisher() flushes the in-memory buffer atomically
    3. publisher exceptions never propagate up through emit()
    4. Buffer overflow drops the oldest events (FIFO eviction)
    5. emit() payload schema (type / severity / source / code / message /
       context / timestamp) is complete and stable
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from argus.core.error_channel import (
    SEVERITY_CRITICAL,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    ErrorChannel,
)


def _make_collector() -> tuple[list[tuple[str, dict]], Any]:
    """Build a collector + publisher pair that records every broadcast call."""
    received: list[tuple[str, dict]] = []

    def publisher(topic: str, payload: dict) -> None:
        received.append((topic, payload))

    return received, publisher


# ── Test 1: emit before publisher → buffered ──────────────────────────


def test_emit_before_publisher_buffers_in_memory():
    chan = ErrorChannel()
    chan.emit(
        SEVERITY_ERROR, "dispatcher", "db_failed", "DB write failed",
        context={"alert_id": "A1"},
    )
    chan.emit(
        SEVERITY_WARNING, "pipeline", "anomaly_degraded", "降级",
        context={"camera_id": "cam0"},
    )
    # Internal buffer should now hold both events
    assert len(chan._buffer) == 2
    assert chan._buffer[0]["code"] == "db_failed"
    assert chan._buffer[1]["code"] == "anomaly_degraded"


# ── Test 2: set_publisher flushes the buffer ──────────────────────────


def test_set_publisher_flushes_buffered_events():
    chan = ErrorChannel()
    chan.emit(SEVERITY_ERROR, "dispatcher", "db_failed", "msg1")
    chan.emit(SEVERITY_WARNING, "pipeline", "anomaly_degraded", "msg2")

    received, publisher = _make_collector()
    chan.set_publisher(publisher)

    # All buffered events should have been published in order
    assert len(received) == 2
    assert received[0][0] == "system_errors"
    assert received[0][1]["code"] == "db_failed"
    assert received[1][1]["code"] == "anomaly_degraded"
    # Buffer should be drained
    assert chan._buffer == []

    # Subsequent emits go directly to publisher (no buffering)
    chan.emit(SEVERITY_INFO, "release_pipeline", "transition_failed", "msg3")
    assert len(received) == 3
    assert received[2][1]["code"] == "transition_failed"
    assert chan._buffer == []


# ── Test 3: publisher errors must not propagate ──────────────────────


def test_publisher_exception_does_not_propagate():
    chan = ErrorChannel()

    def bad_publisher(topic: str, payload: dict) -> None:
        raise RuntimeError("ws backend offline")

    chan.set_publisher(bad_publisher)

    # Must not raise — emit is a fire-and-forget contract
    chan.emit(SEVERITY_CRITICAL, "database", "migration_failed", "boom")
    chan.emit(SEVERITY_ERROR, "dispatcher", "db_failed", "boom2")


def test_publisher_exception_during_buffer_flush_does_not_propagate():
    chan = ErrorChannel()
    # Buffer some events first
    chan.emit(SEVERITY_ERROR, "dispatcher", "db_failed", "msg1")

    def bad_publisher(topic: str, payload: dict) -> None:
        raise ValueError("bad payload")

    # set_publisher triggers the flush path; must not raise
    chan.set_publisher(bad_publisher)
    # Buffer is still cleared even if publishing failed
    assert chan._buffer == []


# ── Test 4: buffer overflow drops oldest ──────────────────────────────


def test_buffer_overflow_drops_oldest_events():
    chan = ErrorChannel()
    limit = ErrorChannel._BUFFER_LIMIT

    # Fill the buffer past the limit
    for i in range(limit + 50):
        chan.emit(
            SEVERITY_ERROR, "dispatcher", f"code_{i}", f"msg_{i}",
            context={"i": i},
        )

    # Buffer is capped at the limit
    assert len(chan._buffer) == limit
    # The oldest 50 events should have been evicted; we keep the latest ones
    first_code = chan._buffer[0]["code"]
    last_code = chan._buffer[-1]["code"]
    assert first_code == "code_50"
    assert last_code == f"code_{limit + 50 - 1}"


# ── Test 5: payload schema is complete ────────────────────────────────


def test_emit_payload_has_all_required_fields():
    chan = ErrorChannel()
    received, publisher = _make_collector()
    chan.set_publisher(publisher)

    chan.emit(
        SEVERITY_CRITICAL,
        "release_pipeline",
        "rollback_failed",
        "Stage transition rollback failed",
        context={"model_version_id": "mv-42", "from_stage": "shadow"},
    )

    assert len(received) == 1
    topic, payload = received[0]
    assert topic == "system_errors"

    # Required schema fields
    assert payload["type"] == "error_event"
    assert payload["severity"] == SEVERITY_CRITICAL
    assert payload["source"] == "release_pipeline"
    assert payload["code"] == "rollback_failed"
    assert payload["message"] == "Stage transition rollback failed"
    assert payload["context"] == {
        "model_version_id": "mv-42",
        "from_stage": "shadow",
    }
    # Timestamp must be ISO 8601 parseable
    ts = payload["timestamp"]
    assert isinstance(ts, str)
    # datetime.fromisoformat handles "+00:00" suffix on Python 3.11+
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None  # we always emit UTC-aware timestamps


def test_emit_without_context_uses_empty_dict():
    chan = ErrorChannel()
    received, publisher = _make_collector()
    chan.set_publisher(publisher)

    chan.emit(SEVERITY_INFO, "pipeline", "anomaly_degraded", "no ctx")
    payload = received[0][1]
    assert payload["context"] == {}


# ── Bonus: re-injecting publisher replaces the previous one ───────────


def test_set_publisher_replaces_previous_publisher():
    chan = ErrorChannel()
    received_a, publisher_a = _make_collector()
    received_b, publisher_b = _make_collector()

    chan.set_publisher(publisher_a)
    chan.emit(SEVERITY_INFO, "src", "code1", "m1")
    assert len(received_a) == 1

    chan.set_publisher(publisher_b)
    chan.emit(SEVERITY_INFO, "src", "code2", "m2")
    # Only the new publisher should see new events
    assert len(received_a) == 1
    assert len(received_b) == 1
    assert received_b[0][1]["code"] == "code2"


def test_severity_constants_match_expected_strings():
    """Front-end relies on these exact strings; lock them in."""
    assert SEVERITY_INFO == "info"
    assert SEVERITY_WARNING == "warning"
    assert SEVERITY_ERROR == "error"
    assert SEVERITY_CRITICAL == "critical"
