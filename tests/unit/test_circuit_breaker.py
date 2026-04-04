"""Tests for the dispatch circuit breaker (DET-009)."""

import json
import time
from pathlib import Path

from argus.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState


def _make_breaker(tmp_path: Path | None = None, **kwargs) -> CircuitBreaker:
    fallback = tmp_path / "cb_state.json" if tmp_path else Path("__test_cb_state.json")
    config = CircuitBreakerConfig(fallback_file=fallback, **kwargs)
    return CircuitBreaker(config)


def test_initial_state_is_closed():
    cb = _make_breaker()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


def test_closed_allows_requests():
    cb = _make_breaker()
    assert cb.allow_request() is True


def test_opens_after_threshold_failures(tmp_path):
    cb = _make_breaker(tmp_path, failure_threshold=3)

    for _ in range(3):
        cb.record_failure()

    assert cb.state == CircuitState.OPEN
    assert cb.allow_request() is False


def test_stays_closed_below_threshold(tmp_path):
    cb = _make_breaker(tmp_path, failure_threshold=5)

    for _ in range(4):
        cb.record_failure()

    assert cb.state == CircuitState.CLOSED
    assert cb.allow_request() is True


def test_success_resets_failure_count(tmp_path):
    cb = _make_breaker(tmp_path, failure_threshold=5)

    cb.record_failure()
    cb.record_failure()
    cb.record_success()

    assert cb.failure_count == 0
    assert cb.state == CircuitState.CLOSED


def test_half_open_after_timeout(tmp_path):
    cb = _make_breaker(tmp_path, failure_threshold=2, recovery_timeout_seconds=0.1)

    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.allow_request() is False

    time.sleep(0.15)

    assert cb.allow_request() is True
    assert cb.state == CircuitState.HALF_OPEN


def test_half_open_success_closes(tmp_path):
    cb = _make_breaker(tmp_path, failure_threshold=2, recovery_timeout_seconds=0.1)

    cb.record_failure()
    cb.record_failure()
    time.sleep(0.15)
    cb.allow_request()  # transitions to half-open

    cb.record_success()
    assert cb.state == CircuitState.CLOSED


def test_half_open_failure_reopens(tmp_path):
    cb = _make_breaker(
        tmp_path, failure_threshold=2, recovery_timeout_seconds=0.1,
        half_open_max_attempts=1,
    )

    cb.record_failure()
    cb.record_failure()
    time.sleep(0.15)
    cb.allow_request()  # transitions to half-open

    cb.record_failure()
    assert cb.state == CircuitState.OPEN


def test_state_persists_to_file(tmp_path):
    fallback = tmp_path / "cb_state.json"
    config = CircuitBreakerConfig(failure_threshold=2, fallback_file=fallback)

    cb1 = CircuitBreaker(config)
    cb1.record_failure()
    cb1.record_failure()
    assert cb1.state == CircuitState.OPEN

    # New instance should restore OPEN state
    cb2 = CircuitBreaker(config)
    assert cb2.state == CircuitState.OPEN
    assert cb2.failure_count == 2


def test_closed_state_clears_file(tmp_path):
    fallback = tmp_path / "cb_state.json"
    config = CircuitBreakerConfig(failure_threshold=2, fallback_file=fallback)

    cb = CircuitBreaker(config)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()

    # New instance should start closed
    cb2 = CircuitBreaker(config)
    assert cb2.state == CircuitState.CLOSED


def test_get_status(tmp_path):
    cb = _make_breaker(tmp_path, failure_threshold=5, recovery_timeout_seconds=60.0)
    status = cb.get_status()
    assert status["state"] == "closed"
    assert status["failure_count"] == 0
    assert status["failure_threshold"] == 5
    assert status["recovery_timeout"] == 60.0
