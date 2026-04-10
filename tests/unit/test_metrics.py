"""Tests for the Prometheus metrics registry."""

import pytest

from argus.core.metrics import METRICS, MetricsRegistry


class TestMetricsRegistry:
    def test_initialize(self):
        reg = MetricsRegistry()
        reg.ensure_initialized()
        assert reg._initialized is True

    def test_double_initialize_is_safe(self):
        reg = MetricsRegistry()
        reg.ensure_initialized()
        reg.ensure_initialized()
        assert reg._initialized is True

    def test_generate_returns_bytes(self):
        reg = MetricsRegistry()
        reg.ensure_initialized()
        output = reg.generate()
        assert isinstance(output, bytes)

    def test_counter_increment(self):
        reg = MetricsRegistry()
        reg.ensure_initialized()
        reg.frames_processed.labels(camera_id="test").inc()
        output = reg.generate().decode()
        assert "argus_frames_processed_total" in output

    def test_histogram_observe(self):
        reg = MetricsRegistry()
        reg.ensure_initialized()
        reg.inference_latency.labels(camera_id="test").observe(0.05)
        output = reg.generate().decode()
        assert "argus_inference_latency_seconds" in output

    def test_gauge_set(self):
        reg = MetricsRegistry()
        reg.ensure_initialized()
        reg.camera_status.labels(camera_id="test").set(1.0)
        output = reg.generate().decode()
        assert "argus_camera_connected" in output

    def test_timer_context_manager(self):
        reg = MetricsRegistry()
        reg.ensure_initialized()
        with reg.timer(reg.pipeline_stage_seconds, camera_id="test", stage="mog2"):
            pass  # instant
        output = reg.generate().decode()
        assert "argus_pipeline_stage_seconds" in output

    def test_noop_fallback(self):
        """When prometheus_client is unavailable, NoOp metrics don't crash."""
        reg = MetricsRegistry()
        # Don't initialize — use NoOp defaults
        reg.frames_processed.labels(camera_id="test").inc()
        reg.anomaly_score.labels(camera_id="test", zone_id="z1").set(0.9)
        output = reg.generate()
        assert b"prometheus_client not installed" in output


class TestModuleSingleton:
    def test_metrics_singleton_exists(self):
        assert METRICS is not None

    def test_metrics_generate_works(self):
        METRICS.ensure_initialized()
        output = METRICS.generate()
        assert isinstance(output, bytes)
