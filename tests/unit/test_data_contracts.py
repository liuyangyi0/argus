"""Tests for data contract validation (Phase 4)."""

from dataclasses import dataclass

import numpy as np
import pytest

from argus.contracts.validation import (
    ContractViolation,
    validate_alert,
    validate_anomaly_result,
    validate_feedback,
    validate_inference_record,
)


# ── Mock objects for testing ──

@dataclass
class MockAnomalyResult:
    anomaly_score: float = 0.75
    anomaly_map: object = None


@dataclass
class MockAlert:
    alert_id: str = "ALT-001"
    camera_id: str = "cam_01"
    severity: str = "medium"
    anomaly_score: float = 0.85


@dataclass
class MockFeedback:
    camera_id: str = "cam_01"
    feedback_type: str = "false_positive"
    source: str = "manual"


@dataclass
class MockInferenceRecord:
    camera_id: str = "cam_01"
    anomaly_score: float = 0.5
    timestamp: float = 1711900000.0


class TestValidateAnomalyResult:
    def test_valid_result(self):
        result = MockAnomalyResult(anomaly_score=0.5)
        violations = validate_anomaly_result(result)
        assert len(violations) == 0

    def test_score_out_of_range(self):
        result = MockAnomalyResult(anomaly_score=1.5)
        violations = validate_anomaly_result(result)
        assert len(violations) == 1
        assert violations[0].field == "anomaly_score"
        assert "range" in violations[0].message

    def test_score_negative(self):
        result = MockAnomalyResult(anomaly_score=-0.1)
        violations = validate_anomaly_result(result)
        assert len(violations) == 1

    def test_score_nan(self):
        result = MockAnomalyResult(anomaly_score=float("nan"))
        violations = validate_anomaly_result(result)
        assert len(violations) == 1
        assert "NaN" in violations[0].message

    def test_score_inf(self):
        result = MockAnomalyResult(anomaly_score=float("inf"))
        violations = validate_anomaly_result(result)
        assert len(violations) == 1

    def test_missing_score(self):
        result = MockAnomalyResult()
        result.anomaly_score = None
        violations = validate_anomaly_result(result)
        assert len(violations) == 1
        assert violations[0].field == "anomaly_score"

    def test_valid_anomaly_map(self):
        result = MockAnomalyResult(anomaly_map=np.zeros((256, 256)))
        violations = validate_anomaly_result(result)
        assert len(violations) == 0

    def test_1d_anomaly_map(self):
        result = MockAnomalyResult(anomaly_map=np.zeros(10))
        violations = validate_anomaly_result(result)
        assert len(violations) == 1
        assert "2D" in violations[0].message

    def test_boundary_values(self):
        """Score of exactly 0.0 and 1.0 should be valid."""
        assert len(validate_anomaly_result(MockAnomalyResult(anomaly_score=0.0))) == 0
        assert len(validate_anomaly_result(MockAnomalyResult(anomaly_score=1.0))) == 0


class TestValidateAlert:
    def test_valid_alert(self):
        assert len(validate_alert(MockAlert())) == 0

    def test_missing_camera_id(self):
        alert = MockAlert(camera_id="")
        violations = validate_alert(alert)
        assert any(v.field == "camera_id" for v in violations)

    def test_missing_alert_id(self):
        alert = MockAlert(alert_id="")
        violations = validate_alert(alert)
        assert any(v.field == "alert_id" for v in violations)

    def test_invalid_severity(self):
        alert = MockAlert(severity="critical")  # not a valid enum value
        violations = validate_alert(alert)
        assert any(v.field == "severity" for v in violations)

    def test_zero_score(self):
        alert = MockAlert(anomaly_score=0.0)
        violations = validate_alert(alert)
        assert any(v.field == "anomaly_score" for v in violations)

    def test_no_exceptions_raised(self):
        """Validation should never raise, even with weird input."""
        validate_alert(object())  # no matching attributes


class TestValidateFeedback:
    def test_valid_feedback(self):
        assert len(validate_feedback(MockFeedback())) == 0

    def test_invalid_type(self):
        fb = MockFeedback(feedback_type="invalid")
        violations = validate_feedback(fb)
        assert any(v.field == "feedback_type" for v in violations)

    def test_invalid_source(self):
        fb = MockFeedback(source="unknown_source")
        violations = validate_feedback(fb)
        assert any(v.field == "source" for v in violations)

    def test_missing_camera(self):
        fb = MockFeedback(camera_id="")
        violations = validate_feedback(fb)
        assert any(v.field == "camera_id" for v in violations)

    def test_all_valid_types(self):
        for t in ("confirmed", "false_positive", "uncertain"):
            fb = MockFeedback(feedback_type=t)
            assert len(validate_feedback(fb)) == 0

    def test_all_valid_sources(self):
        for s in ("manual", "drift", "health"):
            fb = MockFeedback(source=s)
            assert len(validate_feedback(fb)) == 0


class TestValidateInferenceRecord:
    def test_valid_record(self):
        assert len(validate_inference_record(MockInferenceRecord())) == 0

    def test_score_out_of_range(self):
        rec = MockInferenceRecord(anomaly_score=2.0)
        violations = validate_inference_record(rec)
        assert any(v.field == "anomaly_score" for v in violations)

    def test_missing_camera(self):
        rec = MockInferenceRecord(camera_id="")
        violations = validate_inference_record(rec)
        assert any(v.field == "camera_id" for v in violations)

    def test_negative_timestamp(self):
        rec = MockInferenceRecord(timestamp=-1.0)
        violations = validate_inference_record(rec)
        assert any(v.field == "timestamp" for v in violations)
