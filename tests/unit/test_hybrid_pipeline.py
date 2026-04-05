"""Tests for YOLO-004 hybrid detection pipeline and YOLO-005 semantic alert context."""

import time

import numpy as np
import pytest

from argus.alerts.grader import Alert, AlertGrader
from argus.config.schema import (
    AlertConfig,
    AlertSeverity,
    SeverityThresholds,
    SuppressionConfig,
    TemporalConfirmation,
    ZonePriority,
)
from argus.person.detector import ObjectDetection, PersonFilterResult


def make_config(**overrides) -> AlertConfig:
    """Create an AlertConfig with optional overrides for testing."""
    defaults = {
        "severity_thresholds": SeverityThresholds(info=0.5, low=0.7, medium=0.85, high=0.95),
        "temporal": TemporalConfirmation(
            min_consecutive_frames=1,
            max_gap_seconds=10.0,
            min_spatial_overlap=0.0,
        ),
        "suppression": SuppressionConfig(
            same_zone_window_seconds=10.0,
            same_camera_window_seconds=5.0,
        ),
    }
    defaults.update(overrides)
    return AlertConfig(**defaults)


def _emit_alert(grader: AlertGrader, score: float = 0.8, **kwargs) -> Alert | None:
    """Helper to emit a single alert from the grader."""
    defaults = {
        "camera_id": "cam1",
        "zone_id": "z1",
        "zone_priority": ZonePriority.STANDARD,
        "anomaly_score": score,
        "frame_number": 1,
        "frame": np.zeros((100, 100, 3), dtype=np.uint8),
    }
    defaults.update(kwargs)
    return grader.evaluate(**defaults)


class TestAlertDetectionType:
    """YOLO-005: Tests for detection_type field on Alert."""

    def test_default_detection_type_is_anomaly(self):
        """Alert with no YOLO objects should default to detection_type='anomaly'."""
        alert = Alert(
            alert_id="ALT-test-001",
            camera_id="cam1",
            zone_id="z1",
            severity=AlertSeverity.LOW,
            anomaly_score=0.75,
            timestamp=time.time(),
            frame_number=1,
        )
        assert alert.detection_type == "anomaly"
        assert alert.detected_objects == []

    def test_detection_type_hybrid(self):
        """Alert with both YOLO objects and anomaly should be 'hybrid'."""
        objects = [
            {"class_name": "wrench", "confidence": 0.92, "track_id": None, "bbox": [10, 20, 50, 60]}
        ]
        alert = Alert(
            alert_id="ALT-test-002",
            camera_id="cam1",
            zone_id="z1",
            severity=AlertSeverity.MEDIUM,
            anomaly_score=0.88,
            timestamp=time.time(),
            frame_number=2,
            detection_type="hybrid",
            detected_objects=objects,
        )
        assert alert.detection_type == "hybrid"
        assert len(alert.detected_objects) == 1
        assert alert.detected_objects[0]["class_name"] == "wrench"

    def test_detection_type_object_only(self):
        """Alert with YOLO objects but no anomaly should be 'object'."""
        objects = [
            {"class_name": "backpack", "confidence": 0.85, "track_id": 3, "bbox": [100, 200, 300, 400]}
        ]
        alert = Alert(
            alert_id="ALT-test-003",
            camera_id="cam1",
            zone_id="z1",
            severity=AlertSeverity.INFO,
            anomaly_score=0.55,
            timestamp=time.time(),
            frame_number=3,
            detection_type="object",
            detected_objects=objects,
        )
        assert alert.detection_type == "object"
        assert alert.detected_objects[0]["confidence"] == 0.85

    def test_backward_compatibility_no_new_fields(self):
        """Old code creating Alert without new fields should still work."""
        alert = Alert(
            alert_id="ALT-old-001",
            camera_id="cam1",
            zone_id="z1",
            severity=AlertSeverity.HIGH,
            anomaly_score=0.96,
            timestamp=time.time(),
            frame_number=100,
        )
        assert alert.detection_type == "anomaly"
        assert alert.detected_objects == []
        assert alert.snapshot is None
        assert alert.heatmap is None


class TestAlertGraderDetectionContext:
    """YOLO-005: Tests for detection context passed through AlertGrader."""

    def test_grader_passes_detection_type_anomaly(self):
        """Grader should pass detection_type='anomaly' to the emitted alert."""
        grader = AlertGrader(make_config())
        alert = _emit_alert(grader, score=0.8, detection_type="anomaly")
        assert alert is not None
        assert alert.detection_type == "anomaly"
        assert alert.detected_objects == []

    def test_grader_passes_detection_type_hybrid(self):
        """Grader should pass detection_type='hybrid' with detected objects."""
        grader = AlertGrader(make_config())
        objects = [
            {"class_name": "tool", "confidence": 0.92, "track_id": 5, "bbox": [10, 20, 50, 60]}
        ]
        alert = _emit_alert(
            grader, score=0.8,
            detection_type="hybrid",
            detected_objects=objects,
        )
        assert alert is not None
        assert alert.detection_type == "hybrid"
        assert len(alert.detected_objects) == 1
        assert alert.detected_objects[0]["class_name"] == "tool"

    def test_grader_passes_detection_type_object(self):
        """Grader should pass detection_type='object' for YOLO-only detections."""
        grader = AlertGrader(make_config())
        objects = [
            {"class_name": "cat", "confidence": 0.78, "track_id": None, "bbox": [0, 0, 100, 100]},
            {"class_name": "bag", "confidence": 0.71, "track_id": None, "bbox": [200, 200, 300, 300]},
        ]
        alert = _emit_alert(
            grader, score=0.75,
            detection_type="object",
            detected_objects=objects,
        )
        assert alert is not None
        assert alert.detection_type == "object"
        assert len(alert.detected_objects) == 2

    def test_grader_default_detection_type_when_not_specified(self):
        """Grader should default to detection_type='anomaly' when not specified."""
        grader = AlertGrader(make_config())
        alert = _emit_alert(grader, score=0.8)
        assert alert is not None
        assert alert.detection_type == "anomaly"
        assert alert.detected_objects == []


class TestObjectDetectionDataclass:
    """YOLO-004: Tests for the ObjectDetection dataclass."""

    def test_object_detection_fields(self):
        """ObjectDetection should carry class name, bbox, and confidence."""
        obj = ObjectDetection(
            class_name="wrench",
            x1=10, y1=20, x2=50, y2=60,
            confidence=0.92,
            track_id=5,
        )
        assert obj.class_name == "wrench"
        assert obj.confidence == 0.92
        assert obj.track_id == 5
        assert (obj.x1, obj.y1, obj.x2, obj.y2) == (10, 20, 50, 60)

    def test_object_detection_default_track_id(self):
        """ObjectDetection track_id should default to None."""
        obj = ObjectDetection(
            class_name="cat",
            x1=0, y1=0, x2=100, y2=100,
            confidence=0.8,
        )
        assert obj.track_id is None


class TestPersonFilterResultNonPersonObjects:
    """YOLO-004: Tests for non_person_objects in PersonFilterResult."""

    def test_non_person_objects_default_empty(self):
        """PersonFilterResult should default non_person_objects to empty list."""
        result = PersonFilterResult()
        assert result.non_person_objects == []

    def test_non_person_objects_populated(self):
        """PersonFilterResult should carry non-person detections."""
        objects = [
            ObjectDetection(class_name="dog", x1=10, y1=20, x2=50, y2=60, confidence=0.85),
        ]
        result = PersonFilterResult(
            persons=[],
            has_persons=False,
            non_person_objects=objects,
        )
        assert result.non_person_objects is not None
        assert len(result.non_person_objects) == 1
        assert result.non_person_objects[0].class_name == "dog"


class TestDetectedObjectsSerialization:
    """YOLO-005: Tests for detected_objects dict format."""

    def test_detected_objects_dict_format(self):
        """detected_objects dicts should have correct keys and types."""
        obj_dict = {
            "class_name": "hammer",
            "confidence": 0.91,
            "track_id": None,
            "bbox": [10, 20, 50, 60],
        }
        alert = Alert(
            alert_id="ALT-ser-001",
            camera_id="cam1",
            zone_id="z1",
            severity=AlertSeverity.MEDIUM,
            anomaly_score=0.87,
            timestamp=time.time(),
            frame_number=10,
            detection_type="hybrid",
            detected_objects=[obj_dict],
        )
        d = alert.detected_objects[0]
        assert "class_name" in d
        assert "confidence" in d
        assert "track_id" in d
        assert "bbox" in d
        assert isinstance(d["bbox"], list)
        assert len(d["bbox"]) == 4

    def test_multiple_detected_objects(self):
        """Alert should carry multiple detected objects."""
        objects = [
            {"class_name": "wrench", "confidence": 0.92, "track_id": 1, "bbox": [10, 20, 50, 60]},
            {"class_name": "bag", "confidence": 0.75, "track_id": 2, "bbox": [100, 100, 200, 200]},
            {"class_name": "cat", "confidence": 0.80, "track_id": None, "bbox": [300, 300, 400, 400]},
        ]
        alert = Alert(
            alert_id="ALT-ser-002",
            camera_id="cam1",
            zone_id="z1",
            severity=AlertSeverity.HIGH,
            anomaly_score=0.96,
            timestamp=time.time(),
            frame_number=20,
            detection_type="hybrid",
            detected_objects=objects,
        )
        assert len(alert.detected_objects) == 3
        class_names = [o["class_name"] for o in alert.detected_objects]
        assert "wrench" in class_names
        assert "bag" in class_names
        assert "cat" in class_names
