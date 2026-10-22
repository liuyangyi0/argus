"""Tests for structured alert evidence payloads."""

from argus.alerts.evidence import DetectionEvidence, SUPPORTED_DETECTION_TYPES


def test_required_detection_types_are_supported() -> None:
    required = {
        "scene_change",
        "new_object",
        "projectile",
        "water_leak",
        "plume",
        "anomaly",
    }
    assert required.issubset(SUPPORTED_DETECTION_TYPES)


def test_detection_evidence_round_trips_and_summarizes() -> None:
    evidence = DetectionEvidence(
        detection_type="new_object",
        score=0.91,
        camera_id="cam-01",
        zone_id="pipe-rack",
        frame_number=42,
        severity="medium",
        features={"changed_fraction": 0.013},
        suppression_reason=None,
    )

    payload = evidence.to_dict()
    assert payload["summary"] == (
        "new_object, severity=medium, score=0.9100, zone=pipe-rack, frame=42"
    )
    assert payload["supported"] is True

    restored = DetectionEvidence.from_dict(payload)
    assert restored.detection_type == "new_object"
    assert restored.features == {"changed_fraction": 0.013}
    assert restored.evidence_summary() == payload["summary"]


def test_detection_evidence_summary_includes_suppression_reason() -> None:
    evidence = DetectionEvidence(
        detection_type="anomaly",
        score=0.62,
        zone_id="z1",
        frame_number=7,
        suppression_reason="duration_too_short",
    )

    assert "suppressed=duration_too_short" in evidence.evidence_summary()
