"""Tests for low-light MOG2 bypass in the detection pipeline."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from argus.config.schema import (
    AlertConfig,
    AnomalyConfig,
    CameraConfig,
    LowLightConfig,
    MOG2Config,
    PersonFilterConfig,
    SeverityThresholds,
    SuppressionConfig,
    TemporalConfirmation,
)
from argus.core.pipeline import DetectionPipeline


def _make_config(
    low_light_enabled: bool = True,
    brightness_threshold: float = 40.0,
    heartbeat_frames: int = 150,
    max_gap_seconds: float = 10.0,
):
    """Create minimal camera and alert configs for low-light testing."""
    cam = CameraConfig(
        camera_id="test_cam",
        name="Test Camera",
        source="test.mp4",
        protocol="file",
        fps_target=5,
        resolution=(640, 480),
        mog2=MOG2Config(history=500, heartbeat_frames=heartbeat_frames),
        person_filter=PersonFilterConfig(model_name="yolo11n.pt"),
        anomaly=AnomalyConfig(threshold=0.5),
        low_light=LowLightConfig(
            enabled=low_light_enabled,
            brightness_threshold=brightness_threshold,
        ),
    )
    alert = AlertConfig(
        severity_thresholds=SeverityThresholds(info=0.3, low=0.5, medium=0.7, high=0.9),
        temporal=TemporalConfirmation(max_gap_seconds=max_gap_seconds),
        suppression=SuppressionConfig(),
    )
    return cam, alert


# ── LowLightConfig defaults ──────────────────────────────────────────


def test_low_light_config_defaults():
    cfg = LowLightConfig()
    assert cfg.enabled is True
    assert cfg.brightness_threshold == 40.0


def test_low_light_config_validation():
    with pytest.raises(Exception):
        LowLightConfig(brightness_threshold=3.0)  # below ge=5.0
    with pytest.raises(Exception):
        LowLightConfig(brightness_threshold=200.0)  # above le=128.0


# ── Pipeline low-light heartbeat calculation ─────────────────────────


def test_low_light_heartbeat_is_half_max_gap():
    """Low-light heartbeat should be max_gap_seconds / 2."""
    cam, alert = _make_config(max_gap_seconds=10.0)
    with patch("argus.core.pipeline.CameraCapture"), \
         patch("argus.core.pipeline.AnomalibDetector"), \
         patch("argus.core.pipeline.YOLOObjectDetector"):
        pipeline = DetectionPipeline(camera_config=cam, alert_config=alert)
    assert pipeline._low_light_heartbeat_seconds == 5.0


def test_low_light_disabled_does_not_set_flag():
    """When low_light.enabled=False, pipeline should not activate low-light."""
    cam, alert = _make_config(low_light_enabled=False)
    with patch("argus.core.pipeline.CameraCapture"), \
         patch("argus.core.pipeline.AnomalibDetector"), \
         patch("argus.core.pipeline.YOLOObjectDetector"):
        pipeline = DetectionPipeline(camera_config=cam, alert_config=alert)
    assert pipeline._low_light_enabled is False


# ── Dark frame detection logic ───────────────────────────────────────


def test_dark_frame_triggers_low_light_bypass():
    """A frame with mean brightness < threshold should bypass MOG2."""
    cam, alert = _make_config(brightness_threshold=40.0)
    with patch("argus.core.pipeline.CameraCapture"), \
         patch("argus.core.pipeline.AnomalibDetector") as mock_det_cls, \
         patch("argus.core.pipeline.YOLOObjectDetector") as mock_yolo_cls:

        # Mock detector returns high anomaly score for dark frame
        mock_det = MagicMock()
        mock_det.predict.return_value = MagicMock(
            anomaly_score=0.9,
            anomaly_map=None,
            is_anomalous=True,
            threshold=0.5,
            detection_failed=False,
        )
        mock_det_cls.return_value = mock_det

        mock_yolo = MagicMock()
        mock_yolo.detect.return_value = MagicMock(
            has_persons=False,
            persons=[],
            objects=[],
            non_person_objects=[],
            masked_frame=None,
            filter_available=True,
        )
        mock_yolo_cls.return_value = mock_yolo

        pipeline = DetectionPipeline(camera_config=cam, alert_config=alert)

        # Create dark frame (mean brightness ~15)
        dark_frame = np.full((480, 640, 3), 15, dtype=np.uint8)
        frame_data = MagicMock()
        frame_data.camera_id = "test_cam"
        frame_data.frame_number = 1
        frame_data.timestamp = time.time()

        # Process the frame — should reach detector (not blocked by MOG2)
        from argus.core.diagnostics import FrameDiagnostics
        diag = FrameDiagnostics(
            frame_number=1, timestamp=time.time(), camera_id="test_cam"
        )
        result = pipeline._process_frame_inner(frame_data, dark_frame, time.monotonic(), diag)

        # Detector should have been called (frame wasn't blocked by MOG2)
        mock_det.predict.assert_called_once()
        # Diagnostics should record low_light
        assert diag.low_light is True


def test_bright_frame_does_not_trigger_low_light():
    """A frame with mean brightness >= threshold should not activate low-light mode."""
    cam, alert = _make_config(brightness_threshold=40.0)
    with patch("argus.core.pipeline.CameraCapture"), \
         patch("argus.core.pipeline.AnomalibDetector"), \
         patch("argus.core.pipeline.YOLOObjectDetector"):
        pipeline = DetectionPipeline(camera_config=cam, alert_config=alert)

        # Verify that a bright frame does NOT set the low-light flag
        # We test the brightness logic directly instead of running the full pipeline
        import cv2
        bright_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        gray = cv2.cvtColor(bright_frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(gray.mean())
        is_low_light = mean_brightness < pipeline._low_light_threshold
        assert is_low_light is False
        assert mean_brightness == 120.0
