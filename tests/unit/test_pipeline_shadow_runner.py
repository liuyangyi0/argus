"""Tests for DetectionPipeline shadow-runner integration."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from argus.core.pipeline import DetectionPipeline


def _make_stats():
    return SimpleNamespace(
        frames_captured=0,
        frames_skipped_no_change=0,
        frames_skipped_person=0,
        frames_analyzed=0,
        frames_heartbeat=0,
        anomalies_detected=0,
        alerts_emitted=0,
        avg_latency_ms=0.0,
    )


def test_shutdown_flushes_shadow_runner():
    pipeline = DetectionPipeline.__new__(DetectionPipeline)
    pipeline._camera = MagicMock()
    pipeline._shadow_runner = MagicMock()
    pipeline.camera_config = SimpleNamespace(camera_id="cam-01")
    pipeline.stats = _make_stats()

    pipeline.shutdown()

    pipeline._camera.stop.assert_called_once_with()
    pipeline._shadow_runner.flush.assert_called_once_with()


def test_shutdown_continues_when_shadow_flush_fails():
    pipeline = DetectionPipeline.__new__(DetectionPipeline)
    pipeline._camera = MagicMock()
    pipeline._shadow_runner = MagicMock()
    pipeline._shadow_runner.flush.side_effect = RuntimeError("flush failed")
    pipeline.camera_config = SimpleNamespace(camera_id="cam-01")
    pipeline.stats = _make_stats()

    pipeline.shutdown()

    pipeline._camera.stop.assert_called_once_with()
    pipeline._shadow_runner.flush.assert_called_once_with()