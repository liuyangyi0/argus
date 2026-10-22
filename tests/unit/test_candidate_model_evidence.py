import json

import numpy as np
import pytest

from argus.anomaly.candidate_model_evidence import (
    CandidateModelThresholds,
    SpatialTemporalConfirmer,
    build_calibration_id,
    evaluate_candidate_model_evidence,
)


def _evaluate(**overrides):
    anomaly_map = np.zeros((10, 10), dtype=np.float32)
    anomaly_map[2:4, 2:4] = 0.8
    params = {
        "model_anomaly_map": anomaly_map,
        "bbox": (20, 20, 40, 40),
        "frame_shape": (100, 100, 3),
        "model_score": 0.9,
        "output_space": "anomalib_postprocessed",
        "output_comparable": True,
        "calibration_id": "cal-1",
        "expected_calibration_id": "cal-1",
        "thresholds": CandidateModelThresholds(0.7, 0.5, 0.2),
        "context_ratio": 0.5,
        "hot_pixel_threshold": 0.5,
    }
    params.update(overrides)
    return evaluate_candidate_model_evidence(**params)


def test_model_evidence_uses_native_map_coordinates_and_thresholds():
    evidence = _evaluate()

    assert evidence.status == "evaluated"
    assert evidence.would_accept is True
    assert evidence.map_shape == (10, 10)
    assert evidence.roi_p95 == pytest.approx(0.8)
    assert evidence.context_ring_p95 == pytest.approx(0.0)
    assert evidence.roi_context_delta == pytest.approx(0.8)
    assert evidence.roi_hot_fraction == pytest.approx(1.0)
    assert evidence.roi_energy_fraction == pytest.approx(1.0)


def test_uncalibrated_evidence_keeps_metrics_but_not_a_false_reject():
    evidence = _evaluate(
        expected_calibration_id=None,
        thresholds=CandidateModelThresholds(),
    )

    assert evidence.status == "uncalibrated"
    assert evidence.would_accept is None
    assert evidence.roi_p95 == pytest.approx(0.8)


def test_calibration_binding_mismatch_is_fail_closed():
    evidence = _evaluate(expected_calibration_id="different")

    assert evidence.status == "uncalibrated"
    assert evidence.status_reason == "calibration_binding_mismatch"
    assert evidence.would_accept is None


def test_nonfinite_model_map_is_an_inference_error_not_zero():
    anomaly_map = np.zeros((10, 10), dtype=np.float32)
    anomaly_map[2, 2] = np.nan

    evidence = _evaluate(model_anomaly_map=anomaly_map)

    assert evidence.status == "inference_error"
    assert evidence.would_accept is None


def test_incomparable_model_map_never_falls_back_to_legacy_semantics():
    evidence = _evaluate(output_comparable=False)

    assert evidence.status == "unavailable"
    assert evidence.status_reason == "model_output_not_comparable"
    assert evidence.would_accept is None


def test_full_frame_bbox_has_no_context_ring_and_is_unavailable():
    evidence = _evaluate(bbox=(0, 0, 100, 100))

    assert evidence.status == "unavailable"
    assert evidence.status_reason == "context_ring_empty"


def test_evidence_dict_is_strict_json_serializable():
    evidence = _evaluate()

    json.dumps(evidence.to_dict(), allow_nan=False)


def test_calibration_id_is_stable_and_binding_sensitive():
    first = build_calibration_id({"model": "v1", "profile": "p1"})
    reordered = build_calibration_id({"profile": "p1", "model": "v1"})
    changed = build_calibration_id({"model": "v2", "profile": "p1"})

    assert first == reordered
    assert first != changed


def _gate() -> SpatialTemporalConfirmer:
    return SpatialTemporalConfirmer(
        required_passes=2,
        min_gap_seconds=0.4,
        max_gap_seconds=2.0,
        iou_threshold=0.5,
    )


def _observe(
    gate: SpatialTemporalConfirmer,
    *,
    now: float,
    frame: int,
    track: int = 1,
    bbox=(0, 0, 10, 10),
    evaluated: bool = True,
    passed: bool | None = True,
    status: str = "evaluated",
    observed: bool = True,
):
    return gate.observe(
        source_track_id=track,
        bbox=bbox,
        now=now,
        observation_number=frame,
        evaluated=evaluated,
        passed=passed,
        status=status,
        observed_this_frame=observed,
    )


def test_two_consecutive_same_location_passes_confirm_at_min_gap_endpoint():
    gate = _gate()

    first = _observe(gate, now=10.0, frame=1)
    second = _observe(gate, now=10.4, frame=2)

    assert first.would_accept is False
    assert first.streak == 1
    assert second.would_accept is True
    assert second.confirmed is True
    assert second.gap_seconds == pytest.approx(0.4)


def test_too_soon_observation_keeps_vote_until_qualified_consecutive_frame():
    gate = _gate()

    _observe(gate, now=10.0, frame=1)
    too_soon = _observe(gate, now=10.399, frame=2)
    qualified = _observe(gate, now=10.4, frame=3)

    assert too_soon.status_reason == "minimum_gap_pending"
    assert too_soon.streak == 1
    assert qualified.confirmed is True


def test_gap_over_max_restarts_at_one_vote():
    gate = _gate()

    _observe(gate, now=10.0, frame=1)
    result = _observe(gate, now=12.001, frame=2)

    assert result.confirmed is False
    assert result.streak == 1
    assert result.status_reason == "temporal_window_restarted"


def test_skipped_frame_number_prevents_votes_bridging_early_return():
    gate = _gate()

    _observe(gate, now=10.0, frame=10)
    result = _observe(gate, now=10.5, frame=12)

    assert result.confirmed is False
    assert result.streak == 1
    assert result.matched_iou is None


def test_unavailable_or_stale_observation_clears_pending_vote():
    gate = _gate()

    _observe(gate, now=10.0, frame=1)
    skipped = _observe(
        gate,
        now=10.2,
        frame=2,
        evaluated=False,
        passed=None,
        status="rate_limited",
    )
    restarted = _observe(gate, now=10.6, frame=3)

    assert skipped.would_accept is None
    assert skipped.status == "rate_limited"
    assert restarted.streak == 1
    assert restarted.confirmed is False


def test_failed_vote_resets_and_cannot_reseed_on_same_frame():
    gate = _gate()

    _observe(gate, now=10.0, frame=1)
    failed = _observe(gate, now=10.5, frame=2, passed=False)
    restarted = _observe(gate, now=11.0, frame=3)

    assert failed.streak == 0
    assert failed.would_accept is False
    assert restarted.streak == 1


def test_iou_endpoint_and_track_id_change_can_match_same_scene_location():
    gate = _gate()

    _observe(gate, now=10.0, frame=1, track=1, bbox=(0, 0, 10, 10))
    result = _observe(
        gate,
        now=10.5,
        frame=2,
        track=99,
        bbox=(0, 0, 20, 10),
    )

    assert result.matched_iou == pytest.approx(0.5)
    assert result.confirmed is True


def test_confirmed_state_latches_while_consecutive_passes_continue():
    gate = _gate()

    _observe(gate, now=10.0, frame=1)
    _observe(gate, now=10.4, frame=2)
    latched = _observe(gate, now=10.5, frame=3)

    assert latched.confirmed is True
    assert latched.would_accept is True
    assert latched.status_reason == "temporal_confirmation_latched"


def test_confirmed_state_restarts_after_maximum_gap():
    gate = _gate()

    _observe(gate, now=10.0, frame=1)
    _observe(gate, now=10.4, frame=2)
    expired = _observe(gate, now=12.401, frame=3)

    assert expired.confirmed is False
    assert expired.would_accept is False
    assert expired.streak == 1
    assert expired.status_reason == "temporal_window_restarted"


def test_finish_frame_drops_disappeared_tracks():
    gate = _gate()
    _observe(gate, now=10.0, frame=1)

    gate.finish_frame(observed_track_ids=set(), observation_number=2)

    assert gate.snapshot() == []
