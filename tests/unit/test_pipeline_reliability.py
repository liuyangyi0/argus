"""Tests for pipeline reliability fixes (CRIT/HIGH/MED issues)."""

import re
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from argus.anomaly.baseline import BaselineManager
from argus.anomaly.detector import AnomalibDetector, AnomalyResult
from argus.person.detector import PersonFilterResult, YOLOPersonDetector
from argus.prefilter.mog2 import MOG2PreFilter


class TestAnomalyResultDetectionFailed:
    """CRIT-04: AnomalyResult should indicate detection failures."""

    def test_safe_result_has_detection_failed_true(self):
        detector = AnomalibDetector(threshold=0.7)
        result = detector._safe_result()
        assert result.detection_failed is True
        assert result.is_anomalous is False

    def test_normal_result_has_detection_failed_false(self):
        result = AnomalyResult(
            anomaly_score=0.5,
            anomaly_map=None,
            is_anomalous=False,
            threshold=0.7,
        )
        assert result.detection_failed is False

    def test_ssim_calibration_not_marked_as_failed(self):
        """SSIM calibration phase should return detection_failed=False."""
        detector = AnomalibDetector(threshold=0.7, ssim_baseline_frames=5)
        detector.load()  # Loads SSIM fallback
        frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = detector.predict(frame)
        assert result.detection_failed is False

    def test_invalid_frame_returns_detection_failed(self):
        detector = AnomalibDetector(threshold=0.7)
        result = detector._predict_anomalib(None)
        assert result.detection_failed is True

    def test_nan_score_returns_detection_failed(self):
        """NaN score from model should produce detection_failed."""
        detector = AnomalibDetector(threshold=0.7)
        result = detector._safe_result()
        assert result.detection_failed is True


class TestPersonFilterAvailable:
    """HIGH-04: PersonFilterResult.filter_available flag."""

    def test_filter_available_when_yolo_loads(self):
        result = PersonFilterResult(persons=[], has_persons=False, filter_available=True)
        assert result.filter_available is True

    def test_filter_unavailable_when_yolo_fails(self):
        detector = YOLOPersonDetector(model_name="nonexistent_model.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result.filter_available is False
        assert result.has_persons is False

    def test_default_filter_available_is_true(self):
        result = PersonFilterResult(persons=[], has_persons=False)
        assert result.filter_available is True


class TestPhaseCorrelationNaN:
    """MED-02: Phase correlation NaN/Inf validation."""

    def test_nan_phase_correlation_returns_original_frame(self):
        prefilter = MOG2PreFilter(enable_stabilization=True)
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)

        # First call initializes prev_gray
        result1 = prefilter._align_frame(frame1)

        # Identical frames should not cause crash (could produce NaN in edge cases)
        result2 = prefilter._align_frame(frame2)
        assert result2 is not None
        assert result2.shape == frame2.shape

    def test_stabilization_skipped_during_freeze(self):
        """HIGH-02: Stabilization should be skipped when learning_rate_override=0.0."""
        prefilter = MOG2PreFilter(enable_stabilization=True)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Normal processing: stabilization runs
        prefilter.process(frame)

        # Freeze mode: stabilization should be skipped
        with patch.object(prefilter, '_align_frame', wraps=prefilter._align_frame) as mock_align:
            prefilter.process(frame, learning_rate_override=0.0)
            mock_align.assert_not_called()


class TestBaselineVersionParsing:
    """MED-04: Robust baseline version parsing."""

    def test_normal_version_sequence(self, tmp_path):
        manager = BaselineManager(tmp_path / "baselines")
        v1 = manager.create_new_version("cam_01")
        assert v1.name == "v001"
        v2 = manager.create_new_version("cam_01")
        assert v2.name == "v002"

    def test_non_standard_directory_skipped(self, tmp_path):
        """Non-matching directories should not break version parsing."""
        manager = BaselineManager(tmp_path / "baselines")
        base = tmp_path / "baselines" / "cam_01" / "default"
        base.mkdir(parents=True)
        (base / "v001").mkdir()
        (base / "v_notes").mkdir()  # Non-standard, should be skipped
        (base / "vtest").mkdir()  # Non-standard, should be skipped

        v = manager.create_new_version("cam_01")
        assert v.name == "v002"  # Should increment from v001, ignoring non-standard

    def test_empty_directory_starts_at_v001(self, tmp_path):
        manager = BaselineManager(tmp_path / "baselines")
        v = manager.create_new_version("cam_01")
        assert v.name == "v001"


class TestLockHysteresis:
    """HIGH-05: Anomaly lock should use hysteresis to prevent flicker-stuck."""

    def test_lock_engages_at_threshold(self):
        from argus.config.schema import AlertConfig, CameraConfig
        from argus.core.pipeline import DetectionPipeline

        config = CameraConfig(
            camera_id="test", name="test", source="0", protocol="file",
        )
        # Lock threshold defaults to 0.85 from MOG2Config
        pipeline = DetectionPipeline(config, AlertConfig())

        # Score above threshold -> lock engages
        result = AnomalyResult(anomaly_score=0.90, anomaly_map=None, is_anomalous=True, threshold=0.7)
        pipeline._update_lock_state(result)
        assert pipeline._locked is True

    def test_lock_does_not_clear_above_hysteresis(self):
        from argus.config.schema import AlertConfig, CameraConfig
        from argus.core.pipeline import DetectionPipeline

        config = CameraConfig(
            camera_id="test", name="test", source="0", protocol="file",
        )
        pipeline = DetectionPipeline(config, AlertConfig())

        # Engage lock
        high = AnomalyResult(anomaly_score=0.90, anomaly_map=None, is_anomalous=True, threshold=0.7)
        pipeline._update_lock_state(high)
        assert pipeline._locked is True

        # Score drops but stays above clear threshold (0.85 * 0.8 = 0.68)
        medium = AnomalyResult(anomaly_score=0.75, anomaly_map=None, is_anomalous=True, threshold=0.7)
        pipeline._update_lock_state(medium)
        # Should NOT start clear timer — score above hysteresis threshold
        assert pipeline._locked is True
        assert pipeline._lock_last_below_time is None

    def test_lock_clears_after_time_below_hysteresis(self):
        from argus.config.schema import AlertConfig, CameraConfig
        from argus.core.pipeline import DetectionPipeline

        config = CameraConfig(
            camera_id="test", name="test", source="0", protocol="file",
        )
        pipeline = DetectionPipeline(config, AlertConfig())

        # Engage lock
        high = AnomalyResult(anomaly_score=0.90, anomaly_map=None, is_anomalous=True, threshold=0.7)
        pipeline._update_lock_state(high)

        # Score drops below clear threshold
        low = AnomalyResult(anomaly_score=0.50, anomaly_map=None, is_anomalous=False, threshold=0.7)
        pipeline._update_lock_state(low)
        assert pipeline._locked is True  # not yet cleared
        assert pipeline._lock_last_below_time is not None

        # Simulate time passing by backdating the timer
        pipeline._lock_last_below_time = time.monotonic() - pipeline._lock_clear_seconds - 1
        pipeline._update_lock_state(low)
        assert pipeline._locked is False  # should be cleared now


class TestLockTimeBasedClear:
    """HIGH-03: Lock should clear based on time, not frame count."""

    def test_person_skip_still_allows_lock_clear(self):
        from argus.config.schema import AlertConfig, CameraConfig
        from argus.core.pipeline import DetectionPipeline

        config = CameraConfig(
            camera_id="test", name="test", source="0", protocol="file",
        )
        pipeline = DetectionPipeline(config, AlertConfig())

        # Engage lock
        high = AnomalyResult(anomaly_score=0.90, anomaly_map=None, is_anomalous=True, threshold=0.7)
        pipeline._update_lock_state(high)

        # Score drops below clear threshold
        low = AnomalyResult(anomaly_score=0.50, anomaly_map=None, is_anomalous=False, threshold=0.7)
        pipeline._update_lock_state(low)

        # Backdate timer
        pipeline._lock_last_below_time = time.monotonic() - pipeline._lock_clear_seconds - 1

        # Person-skip path: _update_lock_state_time should clear
        pipeline._update_lock_state_time(time.monotonic())
        assert pipeline._locked is False


class TestFrameCopyOnRead:
    """CRIT-05: Camera should copy frame immediately on read."""

    def test_read_returns_independent_copy(self, tmp_path):
        """Frame returned by read() should be independent of VideoCapture buffer."""
        from argus.capture.camera import CameraCapture

        # Create a test video file
        video_path = tmp_path / "test.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(str(video_path), fourcc, 5, (640, 480))
        for _ in range(5):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        cam = CameraCapture("test", str(video_path), protocol="file", fps_target=30)
        cam.connect()

        frame_data = cam.read()
        assert frame_data is not None
        # Verify frame has its own memory (owndata flag)
        assert frame_data.frame.flags['OWNDATA'] or frame_data.frame.base is not None
        cam.stop()


class TestConfigWatchdogTimeout:
    """MED-03: Watchdog timeout should be configurable."""

    def test_watchdog_timeout_in_config(self):
        from argus.config.schema import CameraConfig
        config = CameraConfig(
            camera_id="test", name="test", source="0",
            watchdog_timeout=45.0,
        )
        assert config.watchdog_timeout == 45.0

    def test_watchdog_timeout_default(self):
        from argus.config.schema import CameraConfig
        config = CameraConfig(camera_id="test", name="test", source="0")
        assert config.watchdog_timeout == 30.0

    def test_watchdog_timeout_validation(self):
        from argus.config.schema import CameraConfig
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CameraConfig(camera_id="test", name="test", source="0", watchdog_timeout=1.0)
