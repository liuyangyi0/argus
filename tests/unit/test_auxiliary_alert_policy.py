"""Pure evidence-policy checks; no model, camera, or frame inference is run."""

from types import SimpleNamespace

import pytest

from argus.alerts.grader import Alert, AlertGrader, HANDLING_POLICIES
from argus.anomaly.classification_policy import trusted_classification
from argus.config.schema import AlertSeverity
from argus.core.pipeline import DetectionPipeline


VOCABULARY = ("wrench", "shadow", "crane")


@pytest.mark.parametrize(
    "result",
    [
        None,
        "shadow",
        {"label": "shadow", "confidence": 0.99},
        (),
        ("shadow",),
        ("shadow", 0.99, "extra"),
        ([], 0.99),
        (None, 0.99),
        (3, 0.99),
        ("", 0.99),
        ("  ", 0.99),
        ("not-in-vocabulary", 0.99),
        ("shadow", None),
        ("shadow", True),
        ("shadow", False),
        ("shadow", "0.99"),
        ("shadow", float("nan")),
        ("shadow", float("inf")),
        ("shadow", float("-inf")),
        ("shadow", -0.1),
        ("shadow", 0.899999),
        ("shadow", 1.000001),
    ],
)
def test_untrusted_classification_cannot_affect_alert(result):
    assert trusted_classification(result, VOCABULARY) is None


@pytest.mark.parametrize(
    "threshold",
    [
        0.0, 0.1, 0.899999, -1.0, 1.000001, True, False, "0.9", None,
        float("nan"), float("inf"), float("-inf"),
    ],
)
def test_invalid_runtime_threshold_cannot_lower_conservative_floor(threshold):
    assert trusted_classification(("shadow", 1.0), VOCABULARY, threshold) is None


@pytest.mark.parametrize("confidence", [0.9, 0.95, 0.999999, 1.0])
@pytest.mark.parametrize("container", [tuple, list])
def test_valid_high_confidence_in_vocabulary_evidence_is_accepted(confidence, container):
    result = container(("wrench", confidence))
    assert trusted_classification(result, VOCABULARY) == ("wrench", confidence)


def test_stricter_threshold_is_respected_without_rounding():
    assert trusted_classification(("shadow", 0.949999), VOCABULARY, 0.95) is None
    assert trusted_classification(("shadow", 0.95), VOCABULARY, 0.95) == ("shadow", 0.95)
    assert trusted_classification(("shadow", 1), VOCABULARY, 1.0) == ("shadow", 1.0)


def _corroborated(
    label, confidence, *, threshold=0.9, detection_type="anomaly", objects=None,
):
    config = SimpleNamespace(
        require_detection_or_classifier=True,
        classifier_min_confidence=threshold,
    )
    return AlertGrader._early_warning_corroborated(
        config, detection_type, objects, label, confidence,
    )


@pytest.mark.parametrize(
    "label,confidence",
    [
        (None, 0.99),
        ("", 0.99),
        (" \t", 0.99),
        ([], 0.99),
        (1, 0.99),
        ("wrench", None),
        ("wrench", True),
        ("wrench", False),
        ("wrench", "0.99"),
        ("wrench", float("nan")),
        ("wrench", float("inf")),
        ("wrench", float("-inf")),
        ("wrench", -0.1),
        ("wrench", 0.899999),
        ("wrench", 1.000001),
    ],
)
def test_early_warning_rejects_malformed_or_uncertain_classifier_evidence(
    label, confidence,
):
    assert _corroborated(label, confidence) is False


@pytest.mark.parametrize("threshold", [0.1, 0.5, 0.899999])
def test_early_warning_preserves_existing_explicit_threshold_configuration(threshold):
    # The conservative floor belongs to trusted_classification.  This older
    # grader entry point retains its configured corroboration threshold.
    assert _corroborated("wrench", threshold, threshold=threshold) is True
    assert _corroborated("wrench", threshold - 0.000001, threshold=threshold) is False


@pytest.mark.parametrize("confidence", [0.9, 0.99, 1.0])
def test_early_warning_accepts_valid_high_confidence_classifier(confidence):
    assert _corroborated("wrench", confidence) is True


def test_unknown_classifier_label_is_filtered_before_early_warning():
    trusted = trusted_classification(("unknown-hazard", 0.99), VOCABULARY)
    assert trusted is None
    label, confidence = trusted if trusted is not None else (None, None)
    assert _corroborated(label, confidence) is False


def test_invalid_classifier_does_not_remove_independent_detector_corroboration():
    assert _corroborated(None, None, objects=[{"class_name": "fast_projectile"}]) is True
    assert _corroborated(None, None, detection_type="hybrid") is True
    assert _corroborated(None, None, objects=[{"class_name": "person"}]) is False


@pytest.mark.parametrize(
    "original,escalate,expected",
    [
        (AlertSeverity.INFO, True, AlertSeverity.LOW),
        (AlertSeverity.LOW, True, AlertSeverity.MEDIUM),
        (AlertSeverity.MEDIUM, True, AlertSeverity.HIGH),
        (AlertSeverity.HIGH, True, AlertSeverity.HIGH),
        (AlertSeverity.HIGH, False, AlertSeverity.MEDIUM),
        (AlertSeverity.MEDIUM, False, AlertSeverity.LOW),
        (AlertSeverity.LOW, False, AlertSeverity.INFO),
        (AlertSeverity.INFO, False, AlertSeverity.INFO),
    ],
)
def test_classifier_severity_adjustment_updates_handling_and_evidence_only(
    original, escalate, expected,
):
    score = 0.937123
    alert = Alert(
        alert_id="aux-policy-only",
        camera_id="offline-camera",
        zone_id="zone-a",
        severity=original,
        anomaly_score=score,
        timestamp=1.0,
        frame_number=7,
        handling_policy=HANDLING_POLICIES[original],
        classification_label="wrench" if escalate else "shadow",
        classification_confidence=0.95,
        evidence_summary=f"severity={original.value}",
        evidence_json={
            "severity": original.value,
            "summary": f"severity={original.value}",
            "score": score,
            "features": {"raw_model_score": score, "retained": "original-evidence"},
        },
    )

    result = DetectionPipeline._adjust_alert_severity(alert, escalate=escalate)

    assert result is alert
    assert alert.severity is expected
    assert alert.handling_policy == HANDLING_POLICIES[expected]
    assert alert.anomaly_score == score
    assert alert.evidence_json["score"] == score
    assert alert.evidence_json["features"] == {
        "raw_model_score": score,
        "retained": "original-evidence",
    }
    assert alert.evidence_json["severity"] == expected.value
    assert f"severity={expected.value}" in alert.evidence_summary
    assert alert.evidence_json["summary"] == alert.evidence_summary
    assert alert.evidence_json["original_summary"] == f"severity={original.value}"
    assert alert.evidence_json["classifier_adjustment"] == {
        "original_severity": original.value,
        "applied_severity": expected.value,
        "label": alert.classification_label,
        "confidence": 0.95,
    }
