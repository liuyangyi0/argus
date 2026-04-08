"""Tests for DetectionPipeline shadow-runner integration."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from argus.core.pipeline import DetectionPipeline, PipelineStats


def test_shutdown_flushes_shadow_runner():
    pipeline = DetectionPipeline.__new__(DetectionPipeline)
    pipeline._camera = MagicMock()
    pipeline._shadow_runner = MagicMock()
    pipeline.camera_config = SimpleNamespace(camera_id="cam-01")
    pipeline.stats = PipelineStats()

    pipeline.shutdown()

    pipeline._camera.stop.assert_called_once_with()
    pipeline._shadow_runner.flush.assert_called_once_with()


def test_shutdown_continues_when_shadow_flush_fails():
    pipeline = DetectionPipeline.__new__(DetectionPipeline)
    pipeline._camera = MagicMock()
    pipeline._shadow_runner = MagicMock()
    pipeline._shadow_runner.flush.side_effect = RuntimeError("flush failed")
    pipeline.camera_config = SimpleNamespace(camera_id="cam-01")
    pipeline.stats = PipelineStats()

    pipeline.shutdown()

    pipeline._camera.stop.assert_called_once_with()
    pipeline._shadow_runner.flush.assert_called_once_with()