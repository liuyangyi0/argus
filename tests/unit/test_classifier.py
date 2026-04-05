"""Tests for open vocabulary classifier (D1)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from argus.anomaly.classifier import OpenVocabClassifier, FOE_VOCAB
from argus.config.schema import AlertSeverity


class TestOpenVocabClassifier:

    def test_classifier_init(self):
        """Classifier initializes with default vocabulary."""
        clf = OpenVocabClassifier()
        assert clf._vocabulary == FOE_VOCAB
        assert clf._loaded is False

    def test_custom_vocabulary(self):
        """Custom vocabulary is stored."""
        custom = ["cat", "dog", "bird"]
        clf = OpenVocabClassifier(vocabulary=custom)
        assert clf._vocabulary == custom

    def test_classify_returns_none_when_not_loaded(self):
        """Without model, classify returns None (graceful degradation)."""
        clf = OpenVocabClassifier(model_name="nonexistent.pt")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = clf.classify(frame)
        # Should not crash, returns None since model can't load
        assert result is None

    def test_classify_empty_crop_returns_none(self):
        """Empty crop should return None."""
        clf = OpenVocabClassifier()
        clf._loaded = True  # Pretend loaded
        clf._model = None  # But no actual model
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = clf.classify(frame, bbox=(50, 50, 0, 0))
        assert result is None

    def test_classify_returns_label_and_confidence(self):
        """With a mock YOLO model, classify returns (label, confidence)."""
        clf = OpenVocabClassifier()
        clf._loaded = True

        # Mock YOLO result structure
        mock_box = MagicMock()
        mock_box.cls = np.array([2])
        mock_box.conf = np.array([0.87])
        mock_box.__len__ = lambda self: 1

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.names = {0: "wrench", 1: "bolt", 2: "nut"}

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        clf._model = mock_model

        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        result = clf.classify(frame)

        assert result is not None
        label, conf = result
        assert label == "nut"
        assert abs(conf - 0.87) < 1e-5

    def test_classify_with_bbox_crops_frame(self):
        """When bbox is provided, only the cropped region is passed to the model."""
        clf = OpenVocabClassifier()
        clf._loaded = True

        mock_box = MagicMock()
        mock_box.cls = np.array([0])
        mock_box.conf = np.array([0.92])
        mock_box.__len__ = lambda self: 1

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.names = {0: "wrench"}

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        clf._model = mock_model

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = clf.classify(frame, bbox=(100, 100, 200, 150))

        assert result is not None
        assert result[0] == "wrench"

        # Verify the model was called with a cropped region (200x150)
        call_args = mock_model.call_args
        crop = call_args[0][0]
        assert crop.shape == (150, 200, 3)

    def test_classify_no_detections_returns_none(self):
        """When model finds no objects in the crop, returns None."""
        clf = OpenVocabClassifier()
        clf._loaded = True

        mock_result = MagicMock()
        mock_result.boxes = []

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        clf._model = mock_model

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = clf.classify(frame)
        assert result is None

    def test_classify_model_exception_returns_none(self):
        """If model throws an exception, classify returns None gracefully."""
        clf = OpenVocabClassifier()
        clf._loaded = True

        mock_model = MagicMock()
        mock_model.side_effect = RuntimeError("CUDA out of memory")
        clf._model = mock_model

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = clf.classify(frame)
        assert result is None


class TestHighRiskEscalation:
    """Test that high-risk labels escalate alert severity (D1 pipeline integration)."""

    def test_high_risk_label_escalates_severity(self):
        """A high-risk classification should bump severity up one level."""
        from argus.core.pipeline import DetectionPipeline

        # Alert at MEDIUM severity
        from argus.alerts.grader import Alert
        alert = Alert(
            alert_id="test-001",
            camera_id="cam_01",
            zone_id="default",
            severity=AlertSeverity.MEDIUM,
            anomaly_score=0.88,
            timestamp=0.0,
            frame_number=1,
        )
        result = DetectionPipeline._adjust_alert_severity(alert, escalate=True)
        assert result.severity == AlertSeverity.HIGH

    def test_high_severity_stays_high(self):
        """Already at HIGH should remain HIGH (no overflow)."""
        from argus.core.pipeline import DetectionPipeline
        from argus.alerts.grader import Alert

        alert = Alert(
            alert_id="test-002",
            camera_id="cam_01",
            zone_id="default",
            severity=AlertSeverity.HIGH,
            anomaly_score=0.96,
            timestamp=0.0,
            frame_number=1,
        )
        result = DetectionPipeline._adjust_alert_severity(alert, escalate=True)
        assert result.severity == AlertSeverity.HIGH


class TestLowRiskSuppression:
    """Test that low-risk labels suppress/downgrade alert severity."""

    def test_low_risk_label_suppresses_severity(self):
        """A low-risk classification should downgrade severity one level."""
        from argus.core.pipeline import DetectionPipeline
        from argus.alerts.grader import Alert

        alert = Alert(
            alert_id="test-003",
            camera_id="cam_01",
            zone_id="default",
            severity=AlertSeverity.MEDIUM,
            anomaly_score=0.88,
            timestamp=0.0,
            frame_number=1,
        )
        result = DetectionPipeline._adjust_alert_severity(alert, escalate=False)
        assert result.severity == AlertSeverity.LOW

    def test_info_severity_stays_info(self):
        """Already at INFO should remain INFO (no underflow)."""
        from argus.core.pipeline import DetectionPipeline
        from argus.alerts.grader import Alert

        alert = Alert(
            alert_id="test-004",
            camera_id="cam_01",
            zone_id="default",
            severity=AlertSeverity.INFO,
            anomaly_score=0.55,
            timestamp=0.0,
            frame_number=1,
        )
        result = DetectionPipeline._adjust_alert_severity(alert, escalate=False)
        assert result.severity == AlertSeverity.INFO


class TestExtractAnomalyBbox:
    """Test heatmap-to-bbox extraction helper."""

    def test_extract_bbox_from_heatmap(self):
        """Should find a bounding box around the bright region in the heatmap."""
        from argus.core.pipeline import DetectionPipeline

        # Create a 256x256 heatmap with a bright spot in center
        heatmap = np.zeros((256, 256), dtype=np.float32)
        heatmap[100:150, 120:180] = 1.0  # bright anomaly region

        frame_shape = (512, 640, 3)  # H, W, C
        bbox = DetectionPipeline._extract_anomaly_bbox(heatmap, frame_shape)

        assert bbox is not None
        x, y, w, h = bbox
        assert x >= 0 and y >= 0
        assert w > 0 and h > 0
        # Should be roughly in center-right area of the frame
        assert x < 400
        assert y < 450

    def test_extract_bbox_none_heatmap(self):
        """None heatmap should return None."""
        from argus.core.pipeline import DetectionPipeline

        bbox = DetectionPipeline._extract_anomaly_bbox(None, (480, 640, 3))
        assert bbox is None

    def test_extract_bbox_flat_heatmap(self):
        """Uniform (flat) heatmap should return None (no peak region)."""
        from argus.core.pipeline import DetectionPipeline

        heatmap = np.ones((256, 256), dtype=np.float32) * 0.5
        bbox = DetectionPipeline._extract_anomaly_bbox(heatmap, (480, 640, 3))
        assert bbox is None
