"""Tests for DetectionPipeline shadow-runner integration."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from argus.core.pipeline import DetectionPipeline, PipelineStats


def test_shutdown_flushes_shadow_runner():
    pipeline = DetectionPipeline.__new__(DetectionPipeline)
    pipeline._camera = MagicMock()
    shadow_runner = MagicMock()
    pipeline._shadow_runner = shadow_runner
    pipeline.camera_config = SimpleNamespace(camera_id="cam-01")
    pipeline.stats = PipelineStats()

    pipeline.shutdown()

    pipeline._camera.stop.assert_called_once_with()
    shadow_runner.flush.assert_called_once_with()


def test_shutdown_continues_when_shadow_flush_fails():
    pipeline = DetectionPipeline.__new__(DetectionPipeline)
    pipeline._camera = MagicMock()
    shadow_runner = MagicMock()
    shadow_runner.flush.side_effect = RuntimeError("flush failed")
    pipeline._shadow_runner = shadow_runner
    pipeline.camera_config = SimpleNamespace(camera_id="cam-01")
    pipeline.stats = PipelineStats()

    pipeline.shutdown()

    pipeline._camera.stop.assert_called_once_with()
    shadow_runner.flush.assert_called_once_with()


def test_set_shadow_runner_replaces_and_flushes_old_runner():
    pipeline = DetectionPipeline.__new__(DetectionPipeline)
    pipeline.camera_config = SimpleNamespace(camera_id="cam-01")
    old_runner = MagicMock()
    old_runner.shadow_version_id = "shadow-v1"
    new_runner = MagicMock()
    new_runner.shadow_version_id = "shadow-v2"
    pipeline._shadow_runner = old_runner

    state = pipeline.set_shadow_runner(new_runner)

    assert state["shadow_attached"] is True
    assert state["shadow_detached"] is False
    assert state["shadow_previous_version"] == "shadow-v1"
    assert state["shadow_model_version"] == "shadow-v2"
    assert pipeline.get_shadow_runner_version() == "shadow-v2"
    old_runner.flush.assert_called_once_with()


def test_set_shadow_runner_removes_and_flushes_old_runner():
    pipeline = DetectionPipeline.__new__(DetectionPipeline)
    pipeline.camera_config = SimpleNamespace(camera_id="cam-01")
    old_runner = MagicMock()
    old_runner.shadow_version_id = "shadow-v1"
    pipeline._shadow_runner = old_runner

    state = pipeline.set_shadow_runner(None)

    assert state["shadow_attached"] is False
    assert state["shadow_detached"] is True
    assert state["shadow_previous_version"] == "shadow-v1"
    assert state["shadow_model_version"] is None
    assert pipeline.get_shadow_runner_version() is None
    old_runner.flush.assert_called_once_with()
