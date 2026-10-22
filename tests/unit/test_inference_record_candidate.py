"""Tests for lightweight candidate inference record decisions."""

from argus.core.inference_record import (
    PERSISTED_FINAL_DECISIONS,
    FinalDecision,
    InferenceRecord,
)


def test_candidate_record_decision_values_exist() -> None:
    assert FinalDecision.INFO_CANDIDATE.value == "info_candidate"
    assert FinalDecision.DEBUG_NORMAL_SAMPLE.value == "debug_normal_sample"
    assert FinalDecision.SKIPPED_HEARTBEAT.value == "skipped_heartbeat"


def test_default_persistence_set_excludes_normal_and_debug_samples() -> None:
    assert FinalDecision.INFO_CANDIDATE in PERSISTED_FINAL_DECISIONS
    assert FinalDecision.SKIPPED_HEARTBEAT in PERSISTED_FINAL_DECISIONS
    assert FinalDecision.ALERT in PERSISTED_FINAL_DECISIONS
    assert FinalDecision.NORMAL not in PERSISTED_FINAL_DECISIONS
    assert FinalDecision.DEBUG_NORMAL_SAMPLE not in PERSISTED_FINAL_DECISIONS


def test_candidate_decision_round_trips_through_record_json() -> None:
    record = InferenceRecord(
        frame_id="frame-1",
        camera_id="cam-01",
        timestamp_ns=123,
        model_version="v1",
        final_decision=FinalDecision.INFO_CANDIDATE,
        anomaly_score=0.73,
    )

    restored = InferenceRecord.from_dict(record.to_dict())
    assert restored.final_decision == FinalDecision.INFO_CANDIDATE
    assert restored.anomaly_score == 0.73
