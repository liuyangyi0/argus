"""Three-state circuit breaker for alert dispatch (DET-009).

Prevents cascading failures when webhook/email endpoints are down.
When too many consecutive failures occur, the circuit opens and
dispatch is skipped (alerts still saved to DB). After a recovery
timeout, one test request is allowed (half-open). If it succeeds,
the circuit closes; if it fails, it reopens.

State is persisted to a JSON file so OPEN survives restarts.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import structlog

logger = structlog.get_logger()


class CircuitState(str, Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Dispatch disabled after N failures
    HALF_OPEN = "half_open"  # Testing recovery (allow 1 attempt)


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 60.0
    half_open_max_attempts: int = 1
    fallback_file: Path = Path("data/alerts/circuit_breaker_state.json")


class CircuitBreaker:
    """Thread-safe three-state circuit breaker."""

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_attempts = 0
        self._lock = threading.Lock()
        self._load_state()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                if time.monotonic() - self._last_failure_time >= self._config.recovery_timeout_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_attempts = 0
                    logger.info("circuit_breaker.half_open")
                    return True
                return False

            # HALF_OPEN: allow limited attempts
            return self._half_open_attempts < self._config.half_open_max_attempts

    def record_success(self) -> None:
        """Record a successful dispatch. Closes the circuit."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info("circuit_breaker.closed", msg="Recovery successful")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_attempts = 0
            self._persist_state()

    def record_failure(self) -> None:
        """Record a failed dispatch. May open the circuit."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_attempts += 1
                if self._half_open_attempts >= self._config.half_open_max_attempts:
                    self._state = CircuitState.OPEN
                    logger.warning("circuit_breaker.open", msg="Half-open test failed")
            elif self._failure_count >= self._config.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning("circuit_breaker.open", failures=self._failure_count)

            self._persist_state()

    def get_status(self) -> dict:
        """Get circuit breaker status for dashboard display."""
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self._config.failure_threshold,
                "recovery_timeout": self._config.recovery_timeout_seconds,
            }

    def _persist_state(self) -> None:
        """Write state to file so OPEN survives restart."""
        try:
            self._config.fallback_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "state": self._state.value,
                "failure_count": self._failure_count,
            }
            tmp = self._config.fallback_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(self._config.fallback_file)
        except Exception:
            pass  # Best-effort persistence

    def _load_state(self) -> None:
        """Restore state from file on startup."""
        try:
            if self._config.fallback_file.exists():
                data = json.loads(self._config.fallback_file.read_text())
                loaded_state = data.get("state", "closed")
                if loaded_state == "open":
                    self._state = CircuitState.OPEN
                    self._failure_count = data.get("failure_count", 0)
                    self._last_failure_time = time.monotonic()
                    logger.info(
                        "circuit_breaker.restored",
                        state="open",
                        failures=self._failure_count,
                    )
        except Exception:
            pass  # Start fresh if state file is corrupt
