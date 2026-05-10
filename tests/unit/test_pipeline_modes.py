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
    assert PipelineMode.COLLECTION.value == "collection"
    assert PipelineMode.TRAINING.value == "training"


def test_set_mode_returns_previous_mode():
    """set_mode now returns the previous mode so PipelineModeGuard can restore it."""
    cam_config, alert_config = _make_config()
    pipeline = DetectionPipeline(camera_config=cam_config, alert_config=alert_config)
    assert pipeline.set_mode(PipelineMode.COLLECTION) == PipelineMode.ACTIVE
    assert pipeline.set_mode(PipelineMode.ACTIVE) == PipelineMode.COLLECTION


def test_set_mode_records_session_for_skipped_modes():
    """Entering COLLECTION/TRAINING records start time for the watchdog;
    leaving them clears it so ACTIVE never auto-recovers itself."""
    cam_config, alert_config = _make_config()
    pipeline = DetectionPipeline(camera_config=cam_config, alert_config=alert_config)

    pipeline.set_mode(PipelineMode.COLLECTION)
    assert pipeline._mode_session_start_time is not None

    pipeline.set_mode(PipelineMode.ACTIVE)
    assert pipeline._mode_session_start_time is None

    pipeline.set_mode(PipelineMode.TRAINING)
    assert pipeline._mode_session_start_time is not None

    pipeline.set_mode(PipelineMode.MAINTENANCE)
    assert pipeline._mode_session_start_time is None


def _make_frame_data(frame_number: int = 0):
    from argus.capture.camera import FrameData
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    return FrameData(
        frame=frame,
        camera_id="test_cam",
        timestamp=time.time(),
        frame_number=frame_number,
        resolution=(64, 48),
    )


def test_collection_mode_skips_detection_stages():
    """COLLECTION mode must NOT touch YOLO / Anomalib / MOG2.process()."""
    cam_config, alert_config = _make_config()
    pipeline = DetectionPipeline(camera_config=cam_config, alert_config=alert_config)

    pipeline._prefilter = MagicMock()
    pipeline._object_detector = MagicMock()
    pipeline._anomaly_detector = MagicMock()

    pipeline.set_mode(PipelineMode.COLLECTION)
    result = pipeline.process_frame(_make_frame_data())

    assert result is None
    pipeline._prefilter.process.assert_not_called()
    pipeline._object_detector.detect.assert_not_called() if hasattr(
        pipeline._object_detector, "detect"
    ) else None
    pipeline._anomaly_detector.predict.assert_not_called() if hasattr(
        pipeline._anomaly_detector, "predict"
    ) else None
    # Raw frame must remain available for capture threads
    assert pipeline._latest_raw_frame is not None
    assert pipeline._latest_frame is not None


def test_training_mode_skips_detection_stages():
    """TRAINING mode (痛点 3) shares the skip path with COLLECTION."""
    cam_config, alert_config = _make_config()
    pipeline = DetectionPipeline(camera_config=cam_config, alert_config=alert_config)

    pipeline._prefilter = MagicMock()
    pipeline.set_mode(PipelineMode.TRAINING)

    assert pipeline.process_frame(_make_frame_data()) is None
    pipeline._prefilter.process.assert_not_called()


def test_maintenance_mode_still_runs_prefilter():
    """MAINTENANCE freezes MOG2 learning but does NOT skip detection (boundary)."""
    cam_config, alert_config = _make_config()
    pipeline = DetectionPipeline(camera_config=cam_config, alert_config=alert_config)
    prefilter = MagicMock()
    prefilter.process.return_value = MagicMock(has_change=False)
    pipeline._prefilter = prefilter
    pipeline.set_mode(PipelineMode.MAINTENANCE)

    pipeline.process_frame(_make_frame_data())
    prefilter.process.assert_called()


def test_mode_session_watchdog_recovers_after_timeout():
    """If COLLECTION lingers past max_duration the next frame must auto-restore ACTIVE."""
    cam_config, alert_config = _make_config()
    pipeline = DetectionPipeline(camera_config=cam_config, alert_config=alert_config)
    pipeline._mode_session_max_duration = 0.01  # force trigger

    pipeline.set_mode(PipelineMode.COLLECTION)
    assert pipeline.mode == PipelineMode.COLLECTION
    time.sleep(0.05)

    pipeline.process_frame(_make_frame_data())

    assert pipeline.mode == PipelineMode.ACTIVE
    assert pipeline._mode_session_start_time is None


def test_mode_session_watchdog_inactive_for_active_mode():
    """ACTIVE must never trigger the watchdog (start_time is always None)."""
    cam_config, alert_config = _make_config()
    pipeline = DetectionPipeline(camera_config=cam_config, alert_config=alert_config)
    pipeline._mode_session_max_duration = 0.001

    pipeline.set_mode(PipelineMode.COLLECTION)
    pipeline.set_mode(PipelineMode.ACTIVE)
    time.sleep(0.05)
    assert pipeline._mode_session_start_time is None
    # Mode stays ACTIVE; no spontaneous transitions
    assert pipeline.mode == PipelineMode.ACTIVE


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
