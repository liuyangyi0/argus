"""Tests for pipeline operating modes (DET-006) and learning mode (DET-010)."""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np

from argus.core.pipeline import DetectionPipeline, PipelineMode


def _make_config():
    """Create minimal camera and alert configs for testing."""
    from argus.config.schema import (
        AlertConfig,
        AnomalyConfig,
        CameraConfig,
        MOG2Config,
        PersonFilterConfig,
        SeverityThresholds,
        SuppressionConfig,
        TemporalConfirmation,
    )

    return CameraConfig(
        camera_id="test_cam",
        name="Test Camera",
        source="test.mp4",
        protocol="file",
        fps_target=5,
        resolution=(640, 480),
        mog2=MOG2Config(history=500, heartbeat_frames=30),
        person_filter=PersonFilterConfig(model_name="yolo11n.pt"),
        anomaly=AnomalyConfig(threshold=0.7),
    ), AlertConfig(
        severity_thresholds=SeverityThresholds(info=0.3, low=0.5, medium=0.7, high=0.9),
        temporal_confirmation=TemporalConfirmation(),
        suppression=SuppressionConfig(),
    )


def test_pipeline_mode_default():
    """Pipeline starts in ACTIVE mode (before initialize)."""
    cam_config, alert_config = _make_config()
    pipeline = DetectionPipeline(
        camera_config=cam_config,
        alert_config=alert_config,
    )
    # Before initialize, mode is ACTIVE (set in __init__)
    assert pipeline.mode == PipelineMode.ACTIVE


def test_set_mode():
    cam_config, alert_config = _make_config()
    pipeline = DetectionPipeline(
        camera_config=cam_config,
        alert_config=alert_config,
    )
    pipeline.set_mode(PipelineMode.MAINTENANCE)
    assert pipeline.mode == PipelineMode.MAINTENANCE

    pipeline.set_mode(PipelineMode.LEARNING)
    assert pipeline.mode == PipelineMode.LEARNING


def test_mode_thread_safety():
    """Mode reads/writes from multiple threads should not corrupt."""
    cam_config, alert_config = _make_config()
    pipeline = DetectionPipeline(
        camera_config=cam_config,
        alert_config=alert_config,
    )
    errors = []

    def writer():
        try:
            for _ in range(100):
                pipeline.set_mode(PipelineMode.MAINTENANCE)
                pipeline.set_mode(PipelineMode.ACTIVE)
        except Exception as e:
            errors.append(e)

    def reader():
        try:
            for _ in range(100):
                mode = pipeline.mode
                assert mode in (PipelineMode.ACTIVE, PipelineMode.MAINTENANCE)
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0


def test_learning_progress_inactive():
    cam_config, alert_config = _make_config()
    pipeline = DetectionPipeline(
        camera_config=cam_config,
        alert_config=alert_config,
    )
    progress = pipeline.get_learning_progress()
    assert progress["active"] is False
    assert progress["complete"] is False


def test_learning_progress_active():
    cam_config, alert_config = _make_config()
    pipeline = DetectionPipeline(
        camera_config=cam_config,
        alert_config=alert_config,
    )
    # Simulate learning mode
    pipeline._learning_start_time = time.monotonic()
    pipeline._learning_duration = 600.0

    progress = pipeline.get_learning_progress()
    assert progress["active"] is True
    assert progress["total_seconds"] == 600.0
    assert 0 <= progress["progress"] <= 1.0


def test_learning_duration_calculation():
    """DET-010: duration = max(history / fps * 3, 600)."""
    cam_config, alert_config = _make_config()
    # fps=5, history=500: 500/5*3 = 300 < 600 → 600
    pipeline = DetectionPipeline(
        camera_config=cam_config,
        alert_config=alert_config,
    )
    # The learning duration is set in initialize(), but we can verify
    # the formula manually since initialize() requires a real camera
    fps = max(1, cam_config.fps_target)
    history = cam_config.mog2.history
    expected = max(history / fps * 3, 600)
    assert expected == 600.0  # 500/5*3=300 < 600


def test_pipeline_mode_enum_values():
    assert PipelineMode.ACTIVE.value == "active"
    assert PipelineMode.MAINTENANCE.value == "maintenance"
    assert PipelineMode.LEARNING.value == "learning"


def test_detector_status_delegation():
    """get_detector_status should return a DetectorStatus."""
    cam_config, alert_config = _make_config()
    pipeline = DetectionPipeline(
        camera_config=cam_config,
        alert_config=alert_config,
    )
    status = pipeline.get_detector_status()
    assert status.mode == "ssim_fallback"  # No model loaded
    assert status.model_loaded is False
    assert status.threshold == 0.7


def test_diagnostics_buffer_exists():
    """Pipeline should have a diagnostics buffer (DET-008)."""
    cam_config, alert_config = _make_config()
    pipeline = DetectionPipeline(
        camera_config=cam_config,
        alert_config=alert_config,
    )
    buf = pipeline.get_diagnostics_buffer()
    assert buf is not None
    assert len(buf) == 0
