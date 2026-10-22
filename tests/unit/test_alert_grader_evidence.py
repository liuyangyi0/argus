"""Tests for AlertGrader structured GradeResult output."""

from argus.alerts.grader import AlertGrader
from unittest.mock import patch
from argus.config.schema import (
    AlertConfig,
    SeverityThresholds,
    SuppressionConfig,
    TemporalConfirmation,
    ZonePriority,
)


def make_config(*, evidence_threshold: float = 0.5) -> AlertConfig:
    return AlertConfig(
        severity_thresholds=SeverityThresholds(info=0.5, low=0.7, medium=0.85, high=0.95),
        temporal=TemporalConfirmation(
            max_gap_seconds=10.0,
            evidence_lambda=0.95,
            evidence_threshold=evidence_threshold,
        ),
        suppression=SuppressionConfig(
            same_zone_window_seconds=300,
            same_camera_window_seconds=60,
        ),
    )


def test_below_threshold_returns_suppression_reason() -> None:
    grader = AlertGrader(make_config())

    result = grader.evaluate_with_evidence(
        camera_id="cam-01",
        zone_id="z1",
        zone_priority=ZonePriority.STANDARD,
        anomaly_score=0.2,
        frame_number=1,
        detection_type="anomaly",
    )

    assert result.should_alert is False
    assert result.alert is None
    assert result.suppression_reason == "score_below_threshold"
    assert result.evidence.suppression_reason == "score_below_threshold"


def test_candidate_without_enough_duration_returns_evidence() -> None:
    grader = AlertGrader(make_config(evidence_threshold=3.0))

    result = grader.evaluate_with_evidence(
        camera_id="cam-01",
        zone_id="z1",
        zone_priority=ZonePriority.STANDARD,
        anomaly_score=0.8,
        frame_number=2,
        detection_type="scene_change",
    )

    assert result.should_alert is False
    assert result.severity == "low"
    assert result.suppression_reason == "duration_too_short"
    assert result.evidence.detection_type == "scene_change"
    assert result.evidence.score == 0.8


def test_alert_result_attaches_evidence_to_alert() -> None:
    grader = AlertGrader(make_config())

    result = grader.evaluate_with_evidence(
        camera_id="cam-01",
        zone_id="z1",
        zone_priority=ZonePriority.STANDARD,
        anomaly_score=0.9,
        frame_number=3,
        detection_type="new_object",
        detected_objects=[{"class_name": "static_foreign", "confidence": 0.92}],
    )

    assert result.should_alert is True
    assert result.alert is not None
    assert result.alert.detection_type == "new_object"
    assert result.alert.evidence_summary
    assert result.alert.evidence_json["detection_type"] == "new_object"
    assert result.alert.evidence_json["features"]["detected_object_count"] == 1


def test_same_zone_suppression_is_reported() -> None:
    grader = AlertGrader(make_config())

    first = grader.evaluate_with_evidence(
        camera_id="cam-01",
        zone_id="z1",
        zone_priority=ZonePriority.STANDARD,
        anomaly_score=0.9,
        frame_number=1,
        detection_type="anomaly",
    )
    assert first.should_alert is True

    second = grader.evaluate_with_evidence(
        camera_id="cam-01",
        zone_id="z1",
        zone_priority=ZonePriority.STANDARD,
        anomaly_score=0.9,
        frame_number=2,
        detection_type="anomaly",
    )

    assert second.should_alert is False
    assert second.suppression_reason == "same_zone_suppression"


def test_spatial_persistent_change_suppression_is_explained_per_incident() -> None:
    grader = AlertGrader(make_config())

    with patch(
        "argus.alerts.grader.time.monotonic",
        side_effect=[1000.0, 1000.0, 1010.0, 1010.0, 1060.0, 1060.0],
    ):
        first = grader.evaluate_with_evidence(
            camera_id="cam-01",
            zone_id="z1",
            zone_priority=ZonePriority.STANDARD,
            anomaly_score=0.9,
            frame_number=1,
            detection_type="persistent_change",
            detected_objects=[{"class_name": "static_foreign", "bbox": [100, 100, 200, 200]}],
        )
        duplicate = grader.evaluate_with_evidence(
            camera_id="cam-01",
            zone_id="z1",
            zone_priority=ZonePriority.STANDARD,
            anomaly_score=0.9,
            frame_number=2,
            detection_type="persistent_change",
            detected_objects=[{"class_name": "static_foreign", "bbox": [105, 105, 205, 205]}],
        )
        distinct = grader.evaluate_with_evidence(
            camera_id="cam-01",
            zone_id="z1",
            zone_priority=ZonePriority.STANDARD,
            anomaly_score=0.9,
            frame_number=3,
            detection_type="persistent_change",
            detected_objects=[{"class_name": "static_foreign", "bbox": [500, 500, 600, 600]}],
        )

    assert first.should_alert is True
    assert duplicate.should_alert is False
    assert duplicate.suppression_reason == "same_zone_suppression"
    assert distinct.should_alert is True
