"""Prometheus metrics registry for Argus.

Centralizes all application metrics in one module. The ``/metrics``
endpoint exposes them in Prometheus text format for scraping.

Usage from any module::

    from argus.core.metrics import METRICS

    METRICS.inference_latency.labels(camera_id="cam-01", stage="anomaly").observe(0.042)
    METRICS.frames_processed.labels(camera_id="cam-01").inc()
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Generator

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False


class _NoOpMetric:
    """Drop-in stub when prometheus_client is not installed."""

    def labels(self, **_kw: str) -> _NoOpMetric:
        return self

    def inc(self, _amount: float = 1) -> None:
        pass

    def dec(self, _amount: float = 1) -> None:
        pass

    def set(self, _value: float) -> None:
        pass

    def observe(self, _value: float) -> None:
        pass

    def info(self, _val: dict) -> None:
        pass


_NOOP = _NoOpMetric()


class MetricsRegistry:
    """Lazy-initialized Prometheus metrics for the whole Argus process."""

    def __init__(self) -> None:
        self._registry: CollectorRegistry | None = None  # type: ignore[assignment]
        self._initialized = False
        self._init_lock = threading.Lock()

        # Placeholder attributes — real metrics are created in ``_init()``.
        self.inference_latency: Histogram | _NoOpMetric = _NOOP
        self.frames_processed: Counter | _NoOpMetric = _NOOP
        self.frames_dropped: Counter | _NoOpMetric = _NOOP
        self.frames_skipped: Counter | _NoOpMetric = _NOOP
        self.anomaly_score: Gauge | _NoOpMetric = _NOOP
        self.alerts_total: Counter | _NoOpMetric = _NOOP
        self.model_inference_seconds: Histogram | _NoOpMetric = _NOOP
        self.camera_status: Gauge | _NoOpMetric = _NOOP
        self.go2rtc_streams: Gauge | _NoOpMetric = _NOOP
        self.pipeline_stage_seconds: Histogram | _NoOpMetric = _NOOP
        self.mog2_change_ratio: Gauge | _NoOpMetric = _NOOP
        self.ws_connections: Gauge | _NoOpMetric = _NOOP
        self.db_queue_size: Gauge | _NoOpMetric = _NOOP
        self.webhook_circuit_state: Gauge | _NoOpMetric = _NOOP
        self.app_info: Info | _NoOpMetric = _NOOP

    def _init(self) -> None:
        if self._initialized or not _HAS_PROMETHEUS:
            return
        with self._init_lock:
            if self._initialized:
                return
            self._do_init()
            self._initialized = True

    def _do_init(self) -> None:
        """Create all Prometheus metric objects. Called exactly once under lock."""
        reg = CollectorRegistry()
        self._registry = reg

        ns = "argus"

        self.inference_latency = Histogram(
            f"{ns}_inference_latency_seconds",
            "End-to-end frame processing latency",
            ["camera_id"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=reg,
        )

        self.pipeline_stage_seconds = Histogram(
            f"{ns}_pipeline_stage_seconds",
            "Per-stage processing time",
            ["camera_id", "stage"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
            registry=reg,
        )

        self.frames_processed = Counter(
            f"{ns}_frames_processed_total",
            "Total frames captured",
            ["camera_id"],
            registry=reg,
        )

        self.frames_dropped = Counter(
            f"{ns}_frames_dropped_total",
            "Frames dropped due to backpressure",
            ["camera_id"],
            registry=reg,
        )

        self.frames_skipped = Counter(
            f"{ns}_frames_skipped_total",
            "Frames skipped by pre-filter or person filter",
            ["camera_id", "reason"],
            registry=reg,
        )

        self.anomaly_score = Gauge(
            f"{ns}_anomaly_score",
            "Latest anomaly score per camera/zone",
            ["camera_id", "zone_id"],
            registry=reg,
        )

        self.alerts_total = Counter(
            f"{ns}_alerts_total",
            "Alerts emitted",
            ["camera_id", "severity"],
            registry=reg,
        )

        self.model_inference_seconds = Histogram(
            f"{ns}_model_inference_seconds",
            "Model inference duration",
            ["camera_id", "model_type"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=reg,
        )

        self.camera_status = Gauge(
            f"{ns}_camera_connected",
            "Camera connection status (1=connected, 0=disconnected)",
            ["camera_id"],
            registry=reg,
        )

        self.go2rtc_streams = Gauge(
            f"{ns}_go2rtc_streams_active",
            "Number of active go2rtc streams",
            registry=reg,
        )

        self.mog2_change_ratio = Gauge(
            f"{ns}_mog2_change_ratio",
            "MOG2 background subtraction change ratio",
            ["camera_id"],
            registry=reg,
        )

        self.ws_connections = Gauge(
            f"{ns}_websocket_connections",
            "Active WebSocket connections",
            registry=reg,
        )

        self.db_queue_size = Gauge(
            f"{ns}_db_queue_size",
            "Alert database write queue depth",
            registry=reg,
        )

        self.webhook_circuit_state = Gauge(
            f"{ns}_webhook_circuit_breaker_state",
            "Webhook circuit breaker state (0=closed, 1=open, 2=half-open)",
            registry=reg,
        )

        self.app_info = Info(
            f"{ns}_build",
            "Argus build information",
            registry=reg,
        )

    def ensure_initialized(self) -> None:
        """Initialize metrics on first use (call from app startup)."""
        self._init()

    def generate(self) -> bytes:
        """Return Prometheus text-format metrics output."""
        if not _HAS_PROMETHEUS or self._registry is None:
            return b"# prometheus_client not installed\n"
        return generate_latest(self._registry)

    @contextmanager
    def timer(self, histogram: Histogram | _NoOpMetric, **labels: str) -> Generator[None, None, None]:
        """Context manager to observe duration on a Histogram."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            histogram.labels(**labels).observe(elapsed)


# Module-level singleton — import and use everywhere.
METRICS = MetricsRegistry()
