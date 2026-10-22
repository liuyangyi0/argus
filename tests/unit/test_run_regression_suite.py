from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.run_regression_suite import (
    RegressionClip,
    _evidence_artifact_sha256_if_present,
    _evaluated_artifact_bindings,
    _evaluate_clip,
    _release_acceptance_policy,
    aggregate_suite,
    build_markdown_report,
    false_positive_events,
    load_manifest,
    main,
    positive_event_summary,
    summarize_clip_result,
)


def _result(frames: list[dict], *, fps: float = 10.0, total_frames: int | None = None) -> dict:
    return {
        "video_path": "clip.mp4",
        "annotation_path": "clip.labels.json",
        "threshold": 0.7,
        "sample_every": 1,
        "fps": fps,
        "total_frames": total_frames if total_frames is not None else len(frames),
        "scored_frames": len(frames),
        "ignored_frames": 0,
        "failed_frames": [],
        "metrics": None,
        "frames": frames,
        "annotations": {
            "positive_ranges": [],
            "ignore_ranges": [],
        },
    }


def _release_ready_summary(
    clip_id: str,
    *,
    positive_events: int = 10,
    normal_seconds: float = 3600.0,
    candidate_enabled: bool = True,
    candidate_mode: str = "enforce",
    kind: str = "static_foreign",
) -> dict:
    observed = max(1, positive_events)
    guard = {
        "state_counts": {"stable": 10},
        "observed_frames": 10,
        "enabled_frames": 10,
        "disabled_frames": 0,
        "usable_frames": 10,
        "blocked_frames": 0,
    }
    return {
        "clip_id": clip_id,
        "kind": kind,
        "camera_id": "cam_a",
        "source_session_id": f"session:{clip_id}",
        "schema_version": "argus.pipeline-video-eval.v1",
        "evaluation_origin": "fresh",
        "video_path": f"{clip_id}.mp4",
        "annotation_path": f"{clip_id}.labels.json",
        "source_video_sha256": hashlib.sha256(
            f"video:{clip_id}".encode()
        ).hexdigest(),
        "annotation_sha256": hashlib.sha256(
            f"annotation:{clip_id}".encode()
        ).hexdigest(),
        "source_artifacts_stable": True,
        "annotation_inventory": {
            "raw_range_counts": {"positive": positive_events, "ignore": 0},
            "parsed_range_counts": {"positive": positive_events, "ignore": 0},
            "clipped_range_counts": {"positive": positive_events, "ignore": 0},
            "invalid_range_count": 0,
            "out_of_bounds_range_count": 0,
            "loader_normalized_range_count": 0,
            "clipped_or_dropped_range_count": 0,
            "all_ranges_within_source": True,
        },
        "tags": ["independent_test"],
        "status": "evaluated",
        "evaluation_mode": "pipeline",
        "fps": 60.0 if kind == "projectile" else 30.0,
        "reported_fps": 60.0 if kind == "projectile" else 30.0,
        "sample_every": 1,
        "effective_eval_fps": 60.0 if kind == "projectile" else 30.0,
        "expected_processed_frames": 10,
        "source_fps_available": True,
        "source_total_frames": 10,
        "source_total_frames_known": True,
        "source_exhausted": True,
        "truncated_by_max_frames": False,
        "source_read_error": False,
        "max_frames": None,
        "total_frames": 10,
        "processed_frames": 10,
        "scored_frames": 10,
        "excluded_frames": 0,
        "failed_frames": 0,
        "normal_seconds": normal_seconds,
        "false_positive_event_count": 0,
        "channel_totals": {},
        "positive_events": {
            "events_total": positive_events,
            "events_hit": positive_events,
            "fully_excluded_events": 0,
            "positive_event_uniqueness_evidence_complete": True,
            "events": [
                {"delay_sec": 1.0, "hit": True}
                for _ in range(positive_events)
            ],
        },
        "expected_detection_types": [
            "fast_motion" if kind == "projectile" else "persistent_change"
        ],
        "scene_reference": {
            "reference_approved": True,
            "release_approved": True,
            "operator_approved": True,
            "operator_authorization": {
                "configured_profile_id": "pose-front-v1",
                "active_profile_id": "pose-front-v1",
                "metadata_field_present": True,
                "operator_approved": True,
                "reason": None,
            },
        },
        "scene_guard": guard,
        "scene_guard_summary_score_eligible": guard,
        "scene_guard_summary_all": guard,
        "model_identity": {
            "release_approved": True,
            "actual": {
                "version_id": "model-v1",
                "artifact_sha256": "a" * 64,
            },
        },
        "production_parity": {"supported": True},
        "release_contract": {"approved": True},
        "candidate_confirmation_contract": {
            "available": True,
            "enabled": candidate_enabled,
            "mode": candidate_mode,
            "model_evidence": {
                "enabled": candidate_enabled,
                "fusion_policy": "model_required",
                "calibration_id": "cal-v1" if candidate_enabled else None,
            },
        },
        "candidate_model_evidence_summary": {
            "observed_candidates": observed,
            "evaluated_candidates": observed,
            "map_extraction_eligible_candidates": observed,
            "map_extracted_candidates": observed,
            "calibrated_evaluation_eligible_candidates": observed,
            "calibrated_evaluated_candidates": observed,
            "accepted_candidates": observed,
            "rejected_candidates": 0,
            "nullable_contract_violations": 0,
            "extraction_contract_violations": 0,
            "calibration_id_counts": {"cal-v1": observed},
            "output_space_counts": {"raw": observed},
        },
        "prospective_candidate_fusion_summary": {
            "observed_candidates": observed,
            "evaluated_candidates": observed,
            "accepted_candidates": observed,
            "pending_candidates": 0,
            "rejected_candidates": 0,
            "unavailable_candidates": 0,
            "confirmed_candidates": observed,
            "mode_counts": {candidate_mode: observed},
            "fusion_policy_counts": {"model_required": observed},
        },
    }


def test_false_positive_events_group_by_gap_and_attribute_channels():
    result = _result([
        {"frame_index": 0, "timestamp_sec": 0.0, "score": 0.9, "decision": "fp"},
        {"frame_index": 10, "timestamp_sec": 30.0, "score": 0.8, "decision": "fp"},
        {"frame_index": 20, "timestamp_sec": 95.0, "score": 0.85, "decision": "fp"},
    ])
    candidate_events = [
        {"frame_index": 0, "candidate_type": "fast_motion"},
        {"frame_index": 10, "candidate_type": "persistent_change"},
        {"frame_index": 20, "candidate_type": "simplex"},
    ]

    events = false_positive_events(result, gap_seconds=60.0, candidate_events=candidate_events)

    assert len(events) == 2
    assert events[0]["start_frame"] == 0
    assert events[0]["end_frame"] == 10
    assert events[0]["channels"] == {"fast_motion": 1, "persistent_change": 1}
    assert events[1]["primary_channel"] == "simplex"


def test_false_positive_events_exclude_non_score_eligible_frames():
    result = _result([
        {
            "frame_index": 0,
            "timestamp_sec": 0.0,
            "score": 0.99,
            "decision": "fp",
            "score_eligible": False,
        },
        {
            "frame_index": 1,
            "timestamp_sec": 0.1,
            "score": 0.8,
            "decision": "fp",
            "score_eligible": True,
        },
    ])

    events = false_positive_events(result)

    assert len(events) == 1
    assert events[0]["start_frame"] == 1


def test_positive_event_summary_computes_event_recall_and_delay():
    result = _result([
        {"frame_index": 10, "timestamp_sec": 1.0, "score": 0.1, "predicted": 0, "decision": "fn"},
        {"frame_index": 15, "timestamp_sec": 1.5, "score": 0.9, "predicted": 1, "decision": "tp"},
        {"frame_index": 30, "timestamp_sec": 3.0, "score": 0.1, "predicted": 0, "decision": "fn"},
    ])
    result["annotations"]["positive_ranges"] = [
        {"start_frame": 10, "end_frame": 20, "label": 1, "name": "book"},
        {"start_frame": 30, "end_frame": 40, "label": 1, "name": "box"},
    ]

    summary = positive_event_summary(result)

    assert summary["events_total"] == 2
    assert summary["events_hit"] == 1
    assert summary["recall"] == pytest.approx(0.5)
    assert summary["delay_p95_sec"] == pytest.approx(0.5)
    assert summary["events"][0]["first_detection_frame"] == 15
    assert summary["events"][1]["hit"] is False


def test_positive_event_summary_requires_expected_emitted_alert_channel():
    result = _result([
        {
            "frame_index": 10,
            "timestamp_sec": 1.0,
            "score": 0.95,
            "predicted": 1,
            "decision": "tp",
            "detection_types": ["water_leak", "plume"],
        },
        {
            "frame_index": 15,
            "timestamp_sec": 1.5,
            "score": 0.96,
            "predicted": 1,
            "decision": "tp",
            "detection_types": ["persistent_change", "water_leak"],
        },
    ])
    result["annotations"]["positive_ranges"] = [
        {"start_frame": 10, "end_frame": 20, "name": "box"},
    ]

    summary = positive_event_summary(
        result,
        expected_detection_types=("persistent_change",),
    )

    event = summary["events"][0]
    assert event["hit"] is True
    assert event["semantic_match"] is True
    assert event["first_detection_frame"] == 15
    assert event["delay_sec"] == pytest.approx(0.5)
    assert event["wrong_channel_alert_count"] == 1
    assert event["alert_detection_type_counts"] == {
        "water_leak": 2,
        "plume": 1,
        "persistent_change": 1,
    }


def test_positive_event_summary_wrong_channel_is_diagnostic_only():
    result = _result([{
        "frame_index": 10,
        "timestamp_sec": 1.0,
        "score": 0.95,
        "predicted": 1,
        "decision": "tp",
        "detection_types": ["water_leak"],
        "candidate_records": [{"candidate_type": "persistent_change"}],
    }])
    result["annotations"]["positive_ranges"] = [
        {"start_frame": 10, "end_frame": 20, "name": "box"},
    ]

    summary = positive_event_summary(
        result,
        candidate_events=[{
            "frame_index": 10,
            "candidate_type": "persistent_change",
        }],
        expected_detection_types=("persistent_change",),
    )

    assert summary["events_hit"] == 0
    assert summary["recall"] == 0.0
    assert summary["delay_p95_sec"] is None
    assert summary["events"][0]["diagnosis"] == "wrong_detection_channel"
    assert summary["events"][0]["semantic_match"] is False


def test_positive_event_summary_subtracts_ignore_ranges_and_uses_effective_start():
    result = _result([
        {
            "frame_index": 32,
            "timestamp_sec": 3.2,
            "score": 0.99,
            "predicted": 1,
            "decision": "excluded",
            "score_eligible": False,
        },
        {
            "frame_index": 37,
            "timestamp_sec": 3.7,
            "score": 0.9,
            "predicted": 1,
            "decision": "tp",
            "score_eligible": True,
        },
    ])
    result["annotations"]["positive_ranges"] = [
        {"start_frame": 10, "end_frame": 20, "name": "fully ignored"},
        {"start_frame": 30, "end_frame": 50, "name": "partially ignored"},
    ]
    result["annotations"]["ignore_ranges"] = [
        {"start_frame": 10, "end_frame": 20},
        {"start_frame": 30, "end_frame": 34},
        {"start_frame": 40, "end_frame": 42},
    ]

    summary = positive_event_summary(result)

    assert summary["events_total"] == 1
    assert summary["events_hit"] == 1
    assert summary["fully_ignored_events"] == 1
    assert summary["fully_excluded_events"] == 1
    assert summary["fully_score_excluded_events"] == 0
    assert summary["annotated_events_total"] == 2
    assert summary["annotation_coverage_ratio"] == pytest.approx(0.5)
    event = summary["events"][0]
    assert event["effective_start_frame"] == 35
    assert event["effective_ranges"] == [
        {"start_frame": 35, "end_frame": 39},
        {"start_frame": 43, "end_frame": 50},
    ]
    assert event["first_detection_frame"] == 37
    assert event["delay_sec"] == pytest.approx(0.2)


def test_positive_event_summary_excludes_warmup_and_rebases_partial_delay():
    frames = [
        {
            "frame_index": frame_index,
            "timestamp_sec": frame_index / 10.0,
            "score": 0.1,
            "predicted": 0,
            "decision": "excluded",
            "score_eligible": False,
            "exclusion_reason": "warmup",
        }
        for frame_index in [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
    ]
    frames.append({
        "frame_index": 17,
        "timestamp_sec": 1.7,
        "score": 0.9,
        "predicted": 1,
        "decision": "tp",
        "score_eligible": True,
        "exclusion_reason": None,
    })
    result = _result(frames, fps=10.0, total_frames=21)
    result["annotations"]["positive_ranges"] = [
        {"start_frame": 0, "end_frame": 4, "name": "warmup-only"},
        {"start_frame": 10, "end_frame": 20, "name": "warmup-overlap"},
    ]

    summary = positive_event_summary(result)

    assert summary["events_total"] == 1
    assert summary["fully_ignored_events"] == 0
    assert summary["fully_excluded_events"] == 1
    assert summary["fully_score_excluded_events"] == 1
    assert summary["annotated_events_total"] == 2
    assert summary["annotation_coverage_ratio"] == pytest.approx(0.5)
    event = summary["events"][0]
    assert event["effective_start_frame"] == 15
    assert event["first_detection_frame"] == 17
    assert event["delay_sec"] == pytest.approx(0.2)


def test_positive_event_summary_merges_duplicate_and_overlapping_annotations():
    result = _result([{
        "frame_index": 15,
        "timestamp_sec": 1.5,
        "score": 0.9,
        "predicted": 1,
        "decision": "tp",
        "detection_type": "persistent_change",
    }])
    result["annotations"]["positive_ranges"] = [
        {"start_frame": 10, "end_frame": 20, "name": "first"},
        {"start_frame": 10, "end_frame": 20, "name": "duplicate"},
        {"start_frame": 18, "end_frame": 25, "name": "overlap"},
    ]

    summary = positive_event_summary(result)

    assert summary["events_total"] == 1
    assert summary["events_hit"] == 1
    assert summary["positive_event_uniqueness_evidence_complete"] is False
    assert summary["duplicate_positive_interval_count"] == 1
    assert summary["overlapping_positive_interval_count"] == 1
    assert summary["events"][0]["start_frame"] == 10
    assert summary["events"][0]["end_frame"] == 25
    assert summary["events"][0]["source_annotation_count"] == 3


def test_positive_event_summary_merges_adjacent_annotation_slices():
    result = _result([{
        "frame_index": 15,
        "timestamp_sec": 1.5,
        "score": 0.9,
        "predicted": 1,
        "decision": "tp",
        "detection_type": "persistent_change",
    }])
    result["annotations"]["positive_ranges"] = [
        {"start_frame": 10, "end_frame": 14, "name": "slice-a"},
        {"start_frame": 15, "end_frame": 20, "name": "slice-b"},
    ]

    summary = positive_event_summary(result)

    assert summary["events_total"] == 1
    assert summary["events_hit"] == 1
    assert summary["adjacent_positive_interval_merge_count"] == 1
    assert summary["events"][0]["start_frame"] == 10
    assert summary["events"][0]["end_frame"] == 20
    assert summary["events"][0]["source_annotation_count"] == 2


def test_positive_event_summary_rejects_event_below_eligible_frame_fraction():
    result = _result([
        {
            "frame_index": frame_index,
            "timestamp_sec": frame_index / 10.0,
            "score": 0.1,
            "predicted": 0,
            "decision": "excluded",
            "score_eligible": False,
        }
        for frame_index in range(7)
    ], total_frames=10)
    result["annotations"]["positive_ranges"] = [
        {"start_frame": 0, "end_frame": 9, "name": "mostly excluded"},
    ]

    summary = positive_event_summary(result)

    assert summary["events_total"] == 0
    assert summary["fully_excluded_events"] == 1
    assert summary["insufficient_eligible_fraction_events"] == 1
    assert summary["annotation_coverage_ratio"] == 0.0


def test_positive_event_summary_explains_suppressed_pipeline_candidate():
    result = _result([{
        "frame_index": 10,
        "timestamp_sec": 1.0,
        "score": 0.96,
        "predicted": 0,
        "decision": "fn",
        "suppression_reason": "duration_too_short",
        "candidate_records": [{
            "candidate_type": "persistent_change",
            "decision": "candidate",
            "suppression_reason": None,
        }],
    }])
    result["annotations"]["positive_ranges"] = [
        {"start_frame": 10, "end_frame": 20, "name": "object"},
    ]

    event = positive_event_summary(result)["events"][0]

    assert event["diagnosis"] == "candidate_suppressed"
    assert event["candidate_type_counts"] == {"persistent_change": 1}
    assert event["candidate_outcome_counts"] == {"persistent_change:candidate": 1}
    assert event["suppression_reason_counts"] == {"duration_too_short": 1}


def test_summarize_clip_result_reports_event_fp_per_hour():
    frames = [
        {"frame_index": 0, "timestamp_sec": 0.0, "score": 0.8, "decision": "fp"},
        {"frame_index": 30, "timestamp_sec": 30.0, "score": 0.7, "decision": "fp"},
        {"frame_index": 120, "timestamp_sec": 120.0, "score": 0.9, "decision": "fp"},
    ]
    result = _result(frames, fps=1.0, total_frames=3600)
    clip = RegressionClip(clip_id="neg_001", kind="negative", camera_id="cam_a")

    summary = summarize_clip_result(result, clip)

    assert summary["false_positive_event_count"] == 2
    assert summary["false_positive_events_per_hour"] == pytest.approx(2.0)
    assert summary["normal_seconds"] == pytest.approx(3600.0)


def test_summarize_clip_result_uses_interval_union_for_normal_seconds():
    result = _result([], fps=10.0, total_frames=100)
    result["annotations"]["positive_ranges"] = [
        {"start_frame": 10, "end_frame": 29},
    ]
    result["annotations"]["ignore_ranges"] = [
        {"start_frame": 20, "end_frame": 39},
    ]

    summary = summarize_clip_result(
        result,
        RegressionClip("overlap", "negative", "cam_a"),
    )

    assert summary["normal_seconds"] == pytest.approx(7.0)


def test_summarize_clip_result_scores_only_eligible_frame_coverage():
    frames = [
        {
            "frame_index": index,
            "timestamp_sec": index,
            "score": 0.9 if index == 0 else 0.1,
            "decision": "excluded" if index < 2 else "tn",
            "score_eligible": index >= 2,
        }
        for index in range(10)
    ]
    result = _result(frames, fps=1.0, total_frames=10)

    summary = summarize_clip_result(
        result,
        RegressionClip("eligible", "negative", "cam_a"),
    )

    assert summary["normal_seconds"] == pytest.approx(8.0)
    assert summary["false_positive_event_count"] == 0
    assert summary["score_distribution"]["count"] == 8
    assert summary["score_distribution"]["max"] == pytest.approx(0.1)


def test_summarize_clip_result_preserves_scene_guard_availability():
    result = _result([], fps=1.0, total_frames=100)
    result["scene_guard_summary"] = {
        "usable_frames": 97,
        "blocked_frames": 3,
        "usable_coverage_ratio": 0.97,
    }
    clip = RegressionClip(clip_id="neg_001", kind="negative", camera_id="cam_a")

    summary = summarize_clip_result(result, clip)

    assert summary["scene_guard"] == result["scene_guard_summary"]


def test_summarize_clip_result_preserves_candidate_evidence_and_fusion_summaries():
    result = _result([], fps=1.0, total_frames=1)
    result["candidate_model_evidence_summary"] = {
        "map_extraction_coverage_ratio": 0.98,
        "calibrated_evaluation_coverage_ratio": 0.96,
    }
    result["prospective_candidate_fusion_summary"] = {
        "observed_candidates": 7,
        "accepted_candidates": 3,
    }
    result["scene_reference"] = {"reference_approved": True}
    result["model_identity"] = {"release_approved": True}
    result["production_parity"] = {"supported": True}
    result["candidate_confirmation_contract"] = {
        "available": True,
        "enabled": True,
    }
    result["release_contract"] = {"approved": True}

    summary = summarize_clip_result(
        result,
        RegressionClip("telemetry", "negative", "cam_a"),
    )

    assert summary["candidate_model_evidence_summary"] == (
        result["candidate_model_evidence_summary"]
    )
    assert summary["prospective_candidate_fusion_summary"] == (
        result["prospective_candidate_fusion_summary"]
    )
    for key in (
        "scene_reference",
        "model_identity",
        "production_parity",
        "candidate_confirmation_contract",
        "release_contract",
    ):
        assert summary[key] == result[key]


def test_summarize_clip_result_accepts_actual_evaluator_contract_payloads(tmp_path):
    from argus.validation.pipeline_video_eval import (
        _candidate_confirmation_contract,
        _production_parity_contract,
        _runtime_model_identity,
    )

    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model")
    pipeline = SimpleNamespace(
        _anomaly_model_path=model_path,
        _model_version_id="model-v1",
        _model_artifact_sha256="a" * 64,
        _offline_evaluation_context={
            "sensor_fusion_enabled": False,
            "synchronized_sensor_corpus": False,
        },
        camera_config=SimpleNamespace(
            camera_id="cam_a",
            static_confirmation={"enabled": False, "mode": "shadow"},
        ),
        get_detector_status=lambda: SimpleNamespace(
            model_path=str(model_path),
            model_loaded=True,
            mode="anomalib",
        ),
    )
    result = _result([], total_frames=0)
    result.update({
        "scene_reference": {"reference_approved": True},
        "model_identity": _runtime_model_identity(
            pipeline,
            requested_path=model_path,
            requested_version_id="model-v1",
        ),
        "production_parity": _production_parity_contract(pipeline),
        "candidate_confirmation_contract": _candidate_confirmation_contract(
            pipeline
        ),
        "release_contract": {"approved": True},
    })

    summary = summarize_clip_result(
        result,
        RegressionClip("actual-payload", "negative", "cam_a"),
    )

    assert summary["model_identity"]["release_approved"] is True
    assert summary["production_parity"]["supported"] is True
    assert summary["candidate_confirmation_contract"] == {
        "available": True,
        "camera_id": "cam_a",
        "enabled": False,
        "mode": "shadow",
    }


def test_aggregate_suite_combines_candidate_evidence_and_fusion_summaries():
    first = summarize_clip_result(
        _result([], fps=1.0, total_frames=1),
        RegressionClip("first", "negative", "cam_a"),
    )
    second = summarize_clip_result(
        _result([], fps=1.0, total_frames=1),
        RegressionClip("second", "negative", "cam_a"),
    )
    first["candidate_model_evidence_summary"] = {
        "observed_candidates": 10,
        "evaluated_candidates": 6,
        "map_extraction_eligible_candidates": 10,
        "map_extracted_candidates": 8,
        "calibrated_evaluation_eligible_candidates": 8,
        "calibrated_evaluated_candidates": 6,
        "accepted_candidates": 3,
        "rejected_candidates": 3,
        "calibration_id_counts": {"cal-a": 10},
        "output_space_counts": {"raw": 10},
    }
    second["candidate_model_evidence_summary"] = {
        "observed_candidates": 5,
        "evaluated_candidates": 4,
        "map_extraction_eligible_candidates": 5,
        "map_extracted_candidates": 5,
        "calibrated_evaluation_eligible_candidates": 5,
        "calibrated_evaluated_candidates": 4,
        "accepted_candidates": 1,
        "rejected_candidates": 3,
        "calibration_id_counts": {"cal-a": 3, "cal-b": 2},
        "output_space_counts": {"raw": 5},
    }
    first["prospective_candidate_fusion_summary"] = {
        "observed_candidates": 5,
        "evaluated_candidates": 4,
        "accepted_candidates": 2,
        "pending_candidates": 1,
        "rejected_candidates": 1,
        "unavailable_candidates": 1,
        "confirmed_candidates": 2,
        "mode_counts": {"enforce": 5},
        "fusion_policy_counts": {"model_required": 5},
        "anomaly_attribution": {
            "observed_frames": 3,
            "decided_frames": 2,
            "would_suppress_frames": 1,
            "would_preserve_frames": 1,
            "unavailable_frames": 1,
            "effective_suppressed_frames": 0,
            "reason_counts": {"attributed": 1},
            "hot_coverage_summary": {
                "count": 2,
                "min": 0.2,
                "max": 0.6,
                "mean": 0.4,
            },
        },
    }
    second["prospective_candidate_fusion_summary"] = {
        "observed_candidates": 2,
        "evaluated_candidates": 2,
        "accepted_candidates": 0,
        "pending_candidates": 0,
        "rejected_candidates": 2,
        "unavailable_candidates": 0,
        "confirmed_candidates": 0,
        "mode_counts": {"enforce": 2},
        "fusion_policy_counts": {"model_required": 2},
        "anomaly_attribution": {
            "observed_frames": 1,
            "decided_frames": 1,
            "would_suppress_frames": 1,
            "would_preserve_frames": 0,
            "unavailable_frames": 0,
            "effective_suppressed_frames": 1,
            "reason_counts": {"attributed": 1},
            "hot_coverage_summary": {
                "count": 1,
                "min": 0.8,
                "max": 0.8,
                "mean": 0.8,
            },
        },
    }

    suite = aggregate_suite([first, second])

    model = suite["candidate_model_evidence_summary"]
    assert model["map_extraction_coverage_ratio"] == pytest.approx(13 / 15)
    assert model["calibrated_evaluation_coverage_ratio"] == pytest.approx(10 / 13)
    assert model["rule_accept_rate"] == pytest.approx(0.4)
    assert model["calibration_id_counts"] == {"cal-a": 13, "cal-b": 2}
    fusion = suite["prospective_candidate_fusion_summary"]
    assert fusion["accepted_candidates"] == 2
    assert fusion["pending_candidates"] == 1
    assert fusion["rejected_candidates"] == 3
    assert fusion["mode_counts"] == {"enforce": 7}
    assert fusion["fusion_policy_counts"] == {"model_required": 7}
    attribution = fusion["anomaly_attribution"]
    assert attribution["suppress_rate"] == pytest.approx(2 / 3)
    assert attribution["hot_coverage_summary"]["mean"] == pytest.approx(1.6 / 3)
    suite["release_acceptance_policy"] = _release_acceptance_policy(
        fp_gap_seconds=60.0,
        warmup_seconds=30.0,
        min_scene_guard_coverage=0.99,
        min_effective_eval_fps=5.0,
        min_projectile_eval_fps=59.0,
    )

    markdown = build_markdown_report({
        "generated_at": "now",
        "suite_id": "suite",
        "git_commit": "abc",
        "status": "evaluated",
        "suite": suite,
        "clips": [first, second],
    })
    assert "## Candidate Model Evidence" in markdown
    assert "## Exact Release Acceptance Policy" in markdown
    assert "`fp_gap_seconds` | 60.0000 | 60.0000 | yes" in markdown
    assert "cal-a=13" in markdown
    assert "## Prospective Candidate Fusion" in markdown
    assert "### Prospective Anomaly Attribution" in markdown


def test_release_acceptance_requires_complete_sufficient_production_corpus():
    first = _release_ready_summary("first", positive_events=20)
    second = _release_ready_summary(
        "second",
        positive_events=20,
        kind="projectile",
    )

    suite = aggregate_suite([first, second], expected_clip_count=2)

    assert suite["normal_hours"] == pytest.approx(2.0)
    assert suite["positive_event_count"] == 40
    assert suite["static_foreign_event_count"] == 20
    assert suite["projectile_event_count"] == 20
    assert suite["positive_annotation_coverage_ratio"] == pytest.approx(1.0)
    assert suite["prospective_candidate_fusion_summary"]["mode_counts"] == {
        "enforce": 40,
    }
    assert suite["acceptance"] == {
        "corpus_completeness": "pass",
        "corpus_sufficiency": "pass",
        "kind_specific_sampling": "pass",
        "independent_holdout": "pass",
        "positive_annotation_coverage": "pass",
        "false_positive_rate": "pass",
        "event_recall": "pass",
        "static_foreign_recall": "pass",
        "delay_p95": "pass",
        "scene_guard_coverage": "pass",
        "trusted_scene_reference": "pass",
        "scene_profile_operator_approval": "pass",
        "runtime_model_identity": "pass",
        "production_parity": "pass",
        "candidate_confirmation": "pass",
    }
    assert suite["source_inventory_evidence_complete"] is True
    assert suite["positive_annotation_coverage_evidence_complete"] is True


def test_release_corpus_requires_projectile_events():
    only_static = _release_ready_summary("static-only", positive_events=20)

    suite = aggregate_suite([only_static])

    assert suite["static_foreign_event_count"] == 20
    assert suite["projectile_event_count"] == 0
    assert suite["positive_corpus_sufficient"] is False
    assert suite["acceptance"]["corpus_sufficiency"] == "fail"


def test_release_corpus_requires_unique_source_sessions():
    first = _release_ready_summary("first")
    second = _release_ready_summary("second")
    second["source_session_id"] = first["source_session_id"]

    suite = aggregate_suite([first, second])

    assert suite["source_session_identity_evidence_complete"] is False
    assert suite["duplicate_source_session_id_count"] == 1
    assert suite["source_inventory_evidence_complete"] is False
    assert suite["acceptance"]["independent_holdout"] == "fail"


def test_release_acceptance_fails_subset_and_insufficient_corpus():
    summary = _release_ready_summary(
        "only",
        positive_events=19,
        normal_seconds=7199.0,
    )

    suite = aggregate_suite([summary], expected_clip_count=2)

    assert suite["expected_clip_count"] == 2
    assert suite["acceptance"]["corpus_completeness"] == "fail"
    assert suite["acceptance"]["corpus_sufficiency"] == "fail"


def test_corpus_completeness_rejects_truncated_or_failed_clip():
    truncated = _release_ready_summary("truncated")
    truncated.update(
        source_total_frames=100,
        total_frames=50,
        truncated_by_max_frames=True,
        source_exhausted=False,
    )
    failed = _release_ready_summary("failed")
    failed["failed_frames"] = 1
    unknown_inventory = _release_ready_summary("unknown")
    unknown_inventory.update(
        source_total_frames=0,
        source_total_frames_known=False,
    )
    underreported_inventory = _release_ready_summary("underreported")
    underreported_inventory.update(
        source_total_frames=99,
        total_frames=100,
    )

    assert aggregate_suite([truncated])["acceptance"]["corpus_completeness"] == (
        "fail"
    )
    assert aggregate_suite([failed])["acceptance"]["corpus_completeness"] == (
        "fail"
    )
    unknown_suite = aggregate_suite([unknown_inventory])
    assert unknown_suite["acceptance"]["corpus_completeness"] == "fail"
    assert unknown_suite["source_inventory_evidence_complete"] is False
    underreported_suite = aggregate_suite([underreported_inventory])
    assert underreported_suite["acceptance"]["corpus_completeness"] == "fail"
    assert underreported_suite["source_inventory_evidence_complete"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        {"sample_every": 0},
        {"sample_every": True},
        {"reported_fps": 29.0},
        {"source_exhausted": 1},
        {"processed_frames": 9},
        {"expected_processed_frames": 9},
        {"source_video_sha256": None},
        {"annotation_sha256": None},
        {"source_artifacts_stable": 1},
        {"schema_version": "legacy"},
    ],
)
def test_source_inventory_rejects_missing_wrong_type_or_inconsistent_fields(
    mutation,
):
    summary = _release_ready_summary("strict-source")
    summary.update(mutation)

    suite = aggregate_suite([summary])

    assert suite["source_inventory_evidence_complete"] is False
    assert suite["acceptance"]["corpus_completeness"] == "fail"


def test_source_inventory_rejects_normalized_or_out_of_bounds_annotations():
    normalized = _release_ready_summary("normalized-annotation")
    normalized["annotation_inventory"]["loader_normalized_range_count"] = 1
    normalized["annotation_inventory"]["all_ranges_within_source"] = False
    out_of_bounds = _release_ready_summary("out-of-bounds-annotation")
    out_of_bounds["annotation_inventory"]["out_of_bounds_range_count"] = 1
    out_of_bounds["annotation_inventory"]["all_ranges_within_source"] = False

    assert aggregate_suite([normalized])[
        "source_inventory_evidence_complete"
    ] is False
    assert aggregate_suite([out_of_bounds])[
        "source_inventory_evidence_complete"
    ] is False


def test_source_inventory_rejects_sparse_replay_below_effective_fps_floor():
    summary = _release_ready_summary("sparse")
    summary.update({
        "sample_every": 10,
        "effective_eval_fps": 3.0,
        "expected_processed_frames": 1,
        "processed_frames": 1,
    })

    suite = aggregate_suite([summary])

    assert suite["min_effective_eval_fps"] == 5.0
    assert suite["effective_eval_fps_min_observed"] == 3.0
    assert suite["source_inventory_evidence_complete"] is False
    assert suite["acceptance"]["corpus_sufficiency"] == "fail"


def test_source_inventory_requires_valid_cached_evaluation_contract():
    cached = _release_ready_summary("cached")
    cached["evaluation_origin"] = "cached"
    cached["cached_evaluation_validation"] = {"valid": True, "reasons": []}
    valid_suite = aggregate_suite([cached])

    cached["cached_evaluation_validation"] = {
        "valid": False,
        "reasons": ["source_video_sha256_mismatch"],
    }
    invalid_suite = aggregate_suite([cached])

    assert valid_suite["source_inventory_evidence_complete"] is True
    assert invalid_suite["source_inventory_evidence_complete"] is False


def test_source_inventory_rejects_duplicate_video_content_and_source_path():
    first = _release_ready_summary("first-source")
    second = _release_ready_summary("second-source")
    second["source_video_sha256"] = first["source_video_sha256"]
    hash_suite = aggregate_suite([first, second])

    second = _release_ready_summary("second-source")
    second["video_path"] = first["video_path"]
    path_suite = aggregate_suite([first, second])

    assert hash_suite["unique_source_artifacts_evidence_complete"] is False
    assert hash_suite["duplicate_source_video_hash_count"] == 1
    assert hash_suite["acceptance"]["corpus_completeness"] == "fail"
    assert path_suite["unique_source_artifacts_evidence_complete"] is False
    assert path_suite["duplicate_source_video_path_count"] == 1
    assert path_suite["acceptance"]["corpus_completeness"] == "fail"


def test_positive_release_clip_requires_expected_channels_and_unique_events():
    missing_channels = _release_ready_summary("missing-channels")
    missing_channels["expected_detection_types"] = []
    non_unique = _release_ready_summary("non-unique")
    non_unique["positive_events"][
        "positive_event_uniqueness_evidence_complete"
    ] = False

    missing_suite = aggregate_suite([missing_channels])
    uniqueness_suite = aggregate_suite([non_unique])

    assert (
        missing_suite["positive_expected_detection_types_evidence_complete"]
        is False
    )
    assert missing_suite["acceptance"]["corpus_completeness"] == "fail"
    assert missing_suite["acceptance"]["positive_annotation_coverage"] == "fail"
    assert uniqueness_suite["positive_event_uniqueness_evidence_complete"] is False
    assert uniqueness_suite["acceptance"]["corpus_completeness"] == "fail"
    assert uniqueness_suite["acceptance"]["positive_annotation_coverage"] == "fail"


def test_suite_requires_one_camera_and_exports_canonical_artifact_mapping():
    first = _release_ready_summary("camera-a")
    second = _release_ready_summary("camera-b")
    second["camera_id"] = "cam_b"

    suite = aggregate_suite(
        [first, second],
        evaluated_artifact_hashes={
            "config_sha256": "1" * 64,
            "scene_profile_sha256": "2" * 64,
            "baseline_sha256": "3" * 64,
            "corpus_sha256": "4" * 64,
        },
    )

    assert suite["single_camera_evidence_complete"] is False
    assert suite["source_inventory_evidence_complete"] is False
    assert suite["evaluated_artifacts"] == {
        "camera_id": None,
        "config_sha256": "1" * 64,
        "scene_profile_sha256": "2" * 64,
        "baseline_sha256": "3" * 64,
        "corpus_sha256": "4" * 64,
    }


def test_canonical_artifact_hash_changes_with_file_and_directory_content(
    tmp_path,
):
    from argus.storage.release_evidence import evidence_artifact_sha256

    config = tmp_path / "config.yaml"
    profile = tmp_path / "profile"
    config.write_text("enabled: true\n", encoding="utf-8")
    profile.mkdir()
    (profile / "profile.json").write_text('{"version":1}', encoding="utf-8")

    config_before = _evidence_artifact_sha256_if_present(config)
    profile_before = _evidence_artifact_sha256_if_present(profile)
    assert config_before == evidence_artifact_sha256(config)
    assert profile_before == evidence_artifact_sha256(profile)
    config.write_text("enabled: false\n", encoding="utf-8")
    (profile / "reference.bin").write_bytes(b"reference")

    assert _evidence_artifact_sha256_if_present(config) != config_before
    assert _evidence_artifact_sha256_if_present(profile) != profile_before
    assert _evidence_artifact_sha256_if_present(tmp_path / "missing") is None


def test_evaluated_artifacts_exact_mapping_and_non_pipeline_nulls():
    summary = _release_ready_summary("artifact-map")
    hashes = {
        "config_sha256": "1" * 64,
        "scene_profile_sha256": "2" * 64,
        "baseline_sha256": "3" * 64,
        "corpus_sha256": "4" * 64,
        "unexpected": "5" * 64,
    }

    pipeline_suite = aggregate_suite(
        [summary],
        evaluated_artifact_hashes=hashes,
    )
    summary["evaluation_mode"] = "detector"
    detector_suite = aggregate_suite(
        [summary],
        evaluated_artifact_hashes=hashes,
    )

    assert pipeline_suite["evaluated_artifacts"] == {
        "camera_id": "cam_a",
        "config_sha256": "1" * 64,
        "scene_profile_sha256": "2" * 64,
        "baseline_sha256": "3" * 64,
        "corpus_sha256": "4" * 64,
    }
    assert detector_suite["evaluated_artifacts"] == {
        "camera_id": "cam_a",
        "config_sha256": None,
        "scene_profile_sha256": None,
        "baseline_sha256": None,
        "corpus_sha256": None,
    }


def test_release_acceptance_policy_requires_exact_canonical_parameters():
    exact = _release_acceptance_policy(
        fp_gap_seconds=60.0,
        warmup_seconds=30.0,
        min_scene_guard_coverage=0.99,
        min_effective_eval_fps=5.0,
        min_projectile_eval_fps=59.0,
    )
    diagnostic_override = _release_acceptance_policy(
        fp_gap_seconds=600.0,
        warmup_seconds=300.0,
        min_scene_guard_coverage=0.5,
        min_effective_eval_fps=1.0,
        min_projectile_eval_fps=30.0,
    )

    assert exact["exact_match"] is True
    assert exact["mismatches"] == []
    assert diagnostic_override["exact_match"] is False
    assert diagnostic_override["mismatches"] == [
        "fp_gap_seconds",
        "warmup_seconds",
        "min_scene_guard_coverage",
        "min_effective_eval_fps",
        "min_projectile_eval_fps",
    ]


def test_artifact_bindings_require_actual_config_profile_baseline_and_corpus(
    tmp_path,
    monkeypatch,
    release_baseline_factory,
):
    corpus = tmp_path / "corpus"
    profile = tmp_path / "profiles" / "cam_a" / "pose-v1"
    output = tmp_path / "reports"
    corpus.mkdir()
    profile.mkdir(parents=True)
    config = tmp_path / "config.yaml"
    baseline = release_baseline_factory(
        root=tmp_path / "baseline",
        camera_id="cam_a",
        scene_profile_id="pose-v1",
    )
    manifest = corpus / "manifest.yaml"
    video = corpus / "case.mp4"
    annotation = corpus / "case.labels.json"
    config.write_text("camera: cam_a\n", encoding="utf-8")
    baseline_manifest = json.loads(
        (baseline / "manifest.json").read_text(encoding="utf-8")
    )
    baseline_reference = baseline / baseline_manifest["reference"]["copied_path"]
    baseline_reference_sha256 = baseline_manifest["reference"]["sha256"]
    (profile / "metadata.json").write_text('{"profile":"pose-v1"}')
    (profile / "reference.jpg").write_bytes(b"reference")
    video.write_bytes(b"video")
    annotation.write_text('{"positive_ranges":[]}', encoding="utf-8")
    manifest.write_text("clips: []\n", encoding="utf-8")
    clip = RegressionClip(
        "case",
        "static_foreign",
        "cam_a",
        source_session_id="evaluation-session",
        video_path=video,
        annotation_path=annotation,
    )
    summary = _release_ready_summary("case")
    summary.update({
        "video_path": str(video),
        "annotation_path": str(annotation),
        "source_video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "annotation_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
        "source_session_id": "evaluation-session",
        "config_path": str(config),
        "config_artifact_sha256": _evidence_artifact_sha256_if_present(config),
    })
    summary["scene_reference"].update({
        "profile_id": "pose-v1",
        "reference_sha256": "b" * 64,
    })
    summary["scene_reference"]["operator_authorization"].update({
        "profile_artifact_path": str(profile),
        "source_baseline_name": baseline_reference.name,
        "source_baseline_sha256": baseline_reference_sha256,
    })

    hashes, binding = _evaluated_artifact_bindings(
        evaluation_mode="pipeline",
        config_path=config,
        requested_scene_profile=profile,
        requested_baseline=baseline,
        requested_corpus=corpus,
        manifest_path=manifest,
        manifest_clips=[clip],
        summaries=[summary],
        output_dir=output,
    )

    assert binding["complete"] is False
    assert binding["baseline_decoder_sufficient_for_model_release"] is False
    assert "baseline_decoder_insufficient_for_model_release" in binding["reasons"]
    assert set(hashes) == {
        "config_sha256",
        "scene_profile_sha256",
        "baseline_sha256",
        "corpus_sha256",
    }
    assert all(len(value or "") == 64 for value in hashes.values())
    assert binding["evaluated_source_inventory"] == [{
        "clip_id": "case",
        "camera_id": "cam_a",
        "source_session_id": "evaluation-session",
        "video_relative_path": "case.mp4",
        "source_video_sha256": summary["source_video_sha256"],
        "annotation_relative_path": "case.labels.json",
        "annotation_sha256": summary["annotation_sha256"],
    }]
    assert binding["evaluated_scene_profile"] == {
        "camera_id": "cam_a",
        "profile_id": "pose-v1",
        "reference_sha256": "b" * 64,
        "source_baseline_sha256": baseline_reference_sha256,
        "model_version_id": "model-v1",
        "model_artifact_sha256": "a" * 64,
    }

    from argus.storage import release_evidence as release_evidence_module

    canonical_verifier = (
        release_evidence_module.validate_baseline_manifest_source
    )

    def strict_v3_projection(root, *, registry_data_hash=None):
        baseline_payload, source_hashes, source_sessions = canonical_verifier(
            root,
            registry_data_hash=registry_data_hash,
        )
        baseline_payload["schema_version"] = "argus.scene-profile-baseline.v3"
        baseline_payload["decoder_evidence"] = {
            **baseline_payload["decoder_evidence"],
            "sufficient_for_model_release": True,
        }
        return baseline_payload, source_hashes, source_sessions

    with monkeypatch.context() as context:
        context.setattr(
            release_evidence_module,
            "validate_baseline_manifest_source",
            strict_v3_projection,
        )
        _, strict_v3_binding = _evaluated_artifact_bindings(
            evaluation_mode="pipeline",
            config_path=config,
            requested_scene_profile=profile,
            requested_baseline=baseline,
            requested_corpus=corpus,
            manifest_path=manifest,
            manifest_clips=[clip],
            summaries=[summary],
            output_dir=output,
        )
    assert strict_v3_binding[
        "baseline_decoder_sufficient_for_model_release"
    ] is True
    assert "baseline_decoder_insufficient_for_model_release" not in (
        strict_v3_binding["reasons"]
    )

    _, diagnostic_binding = _evaluated_artifact_bindings(
        evaluation_mode="pipeline",
        config_path=config,
        requested_scene_profile=profile,
        requested_baseline=baseline,
        requested_corpus=corpus,
        manifest_path=manifest,
        manifest_clips=[clip],
        summaries=[summary],
        output_dir=output,
        fp_gap_seconds=600.0,
    )
    assert diagnostic_binding["release_acceptance_policy"]["exact_match"] is False
    assert "non_canonical_release_acceptance_policy" in diagnostic_binding["reasons"]

    session_overlap_summary = dict(summary)
    session_overlap_summary["source_session_id"] = "session-a"
    session_overlap_hashes, session_overlap_binding = _evaluated_artifact_bindings(
        evaluation_mode="pipeline",
        config_path=config,
        requested_scene_profile=profile,
        requested_baseline=baseline,
        requested_corpus=corpus,
        manifest_path=manifest,
        manifest_clips=[clip],
        summaries=[session_overlap_summary],
        output_dir=output,
    )
    assert session_overlap_hashes["baseline_sha256"] is None
    assert any(
        "training and regression source sessions overlap" in reason
        for reason in session_overlap_binding["reasons"]
    )

    baseline_manifest_path = baseline / "manifest.json"
    baseline_manifest_v2 = json.loads(
        baseline_manifest_path.read_text(encoding="utf-8")
    )
    legacy_manifest = dict(baseline_manifest_v2)
    legacy_manifest["schema_version"] = "argus.scene-profile-baseline.v1"
    baseline_manifest_path.write_text(
        json.dumps(legacy_manifest),
        encoding="utf-8",
    )
    legacy_hashes, legacy_binding = _evaluated_artifact_bindings(
        evaluation_mode="pipeline",
        config_path=config,
        requested_scene_profile=profile,
        requested_baseline=baseline,
        requested_corpus=corpus,
        manifest_path=manifest,
        manifest_clips=[clip],
        summaries=[summary],
        output_dir=output,
    )
    assert legacy_hashes["baseline_sha256"] is None
    assert legacy_binding["complete"] is False
    baseline_manifest_path.write_text(
        json.dumps(baseline_manifest_v2),
        encoding="utf-8",
    )
    split_bypass = json.loads(json.dumps(baseline_manifest_v2))
    split_bypass["samples"][1]["split_role"] = "train"
    baseline_manifest_path.write_text(
        json.dumps(split_bypass),
        encoding="utf-8",
    )
    split_hashes, split_binding = _evaluated_artifact_bindings(
        evaluation_mode="pipeline",
        config_path=config,
        requested_scene_profile=profile,
        requested_baseline=baseline,
        requested_corpus=corpus,
        manifest_path=manifest,
        manifest_clips=[clip],
        summaries=[summary],
        output_dir=output,
    )
    assert split_hashes["baseline_sha256"] is None
    assert split_binding["complete"] is False
    baseline_manifest_path.write_text(
        json.dumps(baseline_manifest_v2),
        encoding="utf-8",
    )

    mismatched_hashes, mismatched_binding = _evaluated_artifact_bindings(
        evaluation_mode="pipeline",
        config_path=config,
        requested_scene_profile=tmp_path / "other-profile",
        requested_baseline=baseline,
        requested_corpus=corpus,
        manifest_path=manifest,
        manifest_clips=[clip],
        summaries=[summary],
        output_dir=output,
    )
    assert mismatched_hashes["scene_profile_sha256"] is None
    assert mismatched_binding["complete"] is False

    outside_annotation = tmp_path / "outside.labels.json"
    outside_annotation.write_text('{"positive_ranges":[]}', encoding="utf-8")
    outside_clip = RegressionClip(
        "case",
        "static_foreign",
        "cam_a",
        source_session_id="outside-session",
        video_path=video,
        annotation_path=outside_annotation,
    )
    corpus_hashes, corpus_binding = _evaluated_artifact_bindings(
        evaluation_mode="pipeline",
        config_path=config,
        requested_scene_profile=profile,
        requested_baseline=baseline,
        requested_corpus=corpus,
        manifest_path=manifest,
        manifest_clips=[outside_clip],
        summaries=[summary],
        output_dir=output,
    )
    assert corpus_hashes["corpus_sha256"] is None
    assert corpus_binding["evaluated_source_inventory"] == []

    overlapping_manifest = json.loads(
        (baseline / "manifest.json").read_text(encoding="utf-8")
    )
    overlapping_manifest["sources"][0]["sha256"] = summary[
        "source_video_sha256"
    ]
    overlapping_manifest["samples"][0]["source_sha256"] = summary[
        "source_video_sha256"
    ]
    (baseline / "manifest.json").write_text(
        json.dumps(overlapping_manifest),
        encoding="utf-8",
    )
    overlap_hashes, overlap_binding = _evaluated_artifact_bindings(
        evaluation_mode="pipeline",
        config_path=config,
        requested_scene_profile=profile,
        requested_baseline=baseline,
        requested_corpus=corpus,
        manifest_path=manifest,
        manifest_clips=[clip],
        summaries=[summary],
        output_dir=output,
    )
    assert overlap_hashes["baseline_sha256"] is None
    assert overlap_binding["complete"] is False


def test_positive_annotation_evidence_rejects_missing_positive_clip_annotation():
    summary = _release_ready_summary("missing-annotation")
    summary["annotation_path"] = None

    suite = aggregate_suite([summary])

    assert suite["positive_annotation_coverage_ratio"] == 1.0
    assert suite["positive_annotation_coverage_evidence_complete"] is False


def test_positive_annotation_coverage_counts_ignored_and_warmup_events():
    summary = _release_ready_summary("coverage", positive_events=99)
    summary["positive_events"]["fully_excluded_events"] = 2

    suite = aggregate_suite([summary])

    assert suite["annotated_positive_event_count"] == 101
    assert suite["positive_annotation_coverage_ratio"] == pytest.approx(99 / 101)
    assert suite["acceptance"]["positive_annotation_coverage"] == "fail"


def test_projectile_events_cannot_satisfy_static_foreign_release_targets():
    static = _release_ready_summary("static", positive_events=10)
    static["positive_events"]["events_hit"] = 9
    projectile = _release_ready_summary("projectile", positive_events=10)
    projectile["kind"] = "projectile"
    projectile["expected_detection_types"] = ["fast_motion"]

    suite = aggregate_suite([static, projectile])

    assert suite["positive_event_count"] == 20
    assert suite["event_recall"] == pytest.approx(0.95)
    assert suite["acceptance"]["event_recall"] == "pass"
    assert suite["static_foreign_event_count"] == 10
    assert suite["static_foreign_event_recall"] == pytest.approx(0.9)
    assert suite["acceptance"]["static_foreign_recall"] == "fail"
    assert suite["acceptance"]["corpus_sufficiency"] == "fail"
    assert suite["acceptance"]["kind_specific_sampling"] == "fail"
    assert suite["acceptance"]["corpus_completeness"] == "fail"


def test_projectile_release_evidence_requires_near_60fps_source_and_sampling():
    projectile = _release_ready_summary("projectile-60", positive_events=10)
    projectile.update({
        "kind": "projectile",
        "expected_detection_types": ["fast_projectile"],
        "fps": 60.0,
        "reported_fps": 60.0,
        "effective_eval_fps": 60.0,
    })

    suite = aggregate_suite([projectile])

    assert suite["min_projectile_eval_fps"] == 59.0
    assert suite["kind_specific_sampling_evidence_complete"] is True
    assert suite["acceptance"]["kind_specific_sampling"] == "pass"
    assert suite["acceptance"]["corpus_completeness"] == "pass"


@pytest.mark.parametrize(
    ("kind", "channels"),
    [
        ("static_foreign", ["persistent_change", "anomaly"]),
        ("static_foreign", ["anomaly"]),
        ("projectile", ["persistent_change"]),
        ("negative", ["persistent_change"]),
        ("unrecognized_positive", ["persistent_change"]),
    ],
)
def test_release_expected_channel_policy_rejects_expansion_and_unknown_kinds(
    kind,
    channels,
):
    summary = _release_ready_summary("channel-policy")
    summary["kind"] = kind
    summary["expected_detection_types"] = channels

    suite = aggregate_suite([summary])

    assert suite["expected_detection_type_policy_evidence_complete"] is False
    assert suite["acceptance"]["corpus_completeness"] == "fail"


def test_independent_holdout_requires_explicit_tag_on_every_clip():
    tagged = _release_ready_summary("tagged")
    tuning = _release_ready_summary("tuning")
    tuning["tags"] = ["calibration", "visual_reviewed"]

    suite = aggregate_suite([tagged, tuning])

    assert suite["independent_holdout_clip_count"] == 1
    assert suite["acceptance"]["independent_holdout"] == "fail"


@pytest.mark.parametrize(
    ("field", "mutation", "gate"),
    [
        (
            "scene_reference",
            {"reference_approved": False, "release_approved": False},
            "trusted_scene_reference",
        ),
        (
            "model_identity",
            {
                "release_approved": False,
                "actual": {
                    "version_id": "model-v1",
                    "artifact_sha256": "a" * 64,
                },
            },
            "runtime_model_identity",
        ),
        (
            "production_parity",
            {"supported": False, "reason": "sensor_corpus_missing"},
            "production_parity",
        ),
    ],
)
def test_release_identity_gates_fail_closed(field, mutation, gate):
    summary = _release_ready_summary("invalid")
    summary[field] = mutation

    suite = aggregate_suite([summary])

    assert suite["acceptance"][gate] == "fail"


@pytest.mark.parametrize(
    ("operator_value", "reason"),
    [
        (False, "scene_profile_operator_approval_not_granted"),
        (None, "scene_profile_operator_approval_evidence_missing"),
    ],
)
def test_release_rejects_missing_or_false_operator_approval(
    operator_value,
    reason,
):
    summary = _release_ready_summary("validation-profile")
    if operator_value is None:
        summary["scene_reference"].pop("operator_approved")
        summary["scene_reference"].pop("operator_authorization")
    else:
        summary["scene_reference"]["operator_approved"] = operator_value
        summary["scene_reference"]["reference_approved"] = False
        summary["scene_reference"]["release_approved"] = False
        summary["scene_reference"]["operator_authorization"] = {
            "operator_approved": False,
            "reason": reason,
        }
        summary["release_contract"] = {
            "approved": False,
            "reasons": [reason],
        }

    suite = aggregate_suite([summary])

    assert suite["operator_approved_scene_profile_clip_count"] == 0
    assert suite["acceptance"]["scene_profile_operator_approval"] == "fail"
    assert suite["acceptance"]["trusted_scene_reference"] == "fail"
    assert suite["scene_profile_operator_approval_reason_counts"] == {reason: 1}
    markdown = build_markdown_report({
        "generated_at": "now",
        "suite_id": "operator-gate",
        "git_commit": "abc",
        "status": "ok",
        "suite": suite,
        "clips": [summary],
    })
    assert "Scene-profile operator approval" in markdown
    assert reason in markdown


def test_runtime_model_identity_rejects_mixed_artifacts():
    first = _release_ready_summary("first")
    second = _release_ready_summary("second")
    second["model_identity"]["actual"]["artifact_sha256"] = "b" * 64

    suite = aggregate_suite([first, second])

    assert len(suite["runtime_model_identity_keys"]) == 2
    assert suite["acceptance"]["runtime_model_identity"] == "fail"


def test_disabled_candidate_confirmation_cannot_bypass_static_foreign_release():
    disabled = _release_ready_summary("disabled", candidate_enabled=False)
    missing_contract = _release_ready_summary("missing")
    missing_contract["candidate_confirmation_contract"] = None

    disabled_suite = aggregate_suite([disabled])
    missing_suite = aggregate_suite([missing_contract])

    assert disabled_suite["candidate_explicitly_not_configured"] is True
    assert disabled_suite["candidate_confirmation_required"] is True
    assert disabled_suite["acceptance"]["candidate_confirmation"] == "fail"
    assert missing_suite["candidate_explicitly_not_configured"] is False
    assert missing_suite["acceptance"]["candidate_confirmation"] == "fail"


def test_disabled_candidate_confirmation_is_not_applicable_without_static_target():
    disabled = _release_ready_summary(
        "negative-only",
        positive_events=0,
        candidate_enabled=False,
    )
    disabled["kind"] = "negative"

    suite = aggregate_suite([disabled])

    assert suite["candidate_confirmation_required"] is False
    assert suite["acceptance"]["candidate_confirmation"] == "not_applicable"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda summary: summary["candidate_confirmation_contract"].update(
            mode="shadow"
        ),
        lambda summary: summary["candidate_model_evidence_summary"].update(
            map_extracted_candidates=98,
            map_extraction_eligible_candidates=100,
        ),
        lambda summary: summary["candidate_model_evidence_summary"].update(
            calibrated_evaluated_candidates=98,
            calibrated_evaluation_eligible_candidates=100,
        ),
        lambda summary: summary["candidate_model_evidence_summary"].update(
            nullable_contract_violations=1
        ),
        lambda summary: summary["prospective_candidate_fusion_summary"].update(
            unavailable_candidates=1
        ),
        lambda summary: summary["prospective_candidate_fusion_summary"].update(
            mode_counts={"shadow": 10}
        ),
        lambda summary: summary["candidate_confirmation_contract"][
            "model_evidence"
        ].update(calibration_id=None),
        lambda summary: summary["candidate_model_evidence_summary"].update(
            calibration_id_counts={"other-calibration": 10}
        ),
    ],
)
def test_candidate_confirmation_gate_rejects_shadow_gaps_and_violations(mutator):
    summary = _release_ready_summary("candidate")
    mutator(summary)

    suite = aggregate_suite([summary])

    assert suite["acceptance"]["candidate_confirmation"] == "fail"


def test_candidate_confirmation_requires_one_calibration_binding_across_corpus():
    first = _release_ready_summary("first")
    second = _release_ready_summary("second")
    second["candidate_confirmation_contract"]["model_evidence"][
        "calibration_id"
    ] = "cal-v2"
    second["candidate_model_evidence_summary"]["calibration_id_counts"] = {
        "cal-v2": 10,
    }

    suite = aggregate_suite([first, second])

    assert suite["candidate_contract_calibration_id"] is None
    assert suite["candidate_calibration_contract_matches"] is False
    assert suite["candidate_model_evidence_summary"]["calibration_id_counts"] == {
        "cal-v1": 10,
        "cal-v2": 10,
    }
    assert suite["acceptance"]["candidate_confirmation"] == "fail"


def test_aggregate_suite_marks_insufficient_without_positive_events():
    result = _result([], fps=1.0, total_frames=1800)
    clip = RegressionClip(clip_id="neg_001", kind="negative", camera_id="cam_a")
    summary = summarize_clip_result(result, clip)

    suite = aggregate_suite([summary])

    assert suite["normal_hours"] == pytest.approx(0.5)
    assert suite["event_recall"] is None
    assert suite["acceptance"]["event_recall"] == "insufficient_data"
    assert suite["acceptance"]["scene_guard_coverage"] == "insufficient_data"


def test_aggregate_suite_weights_scene_guard_coverage_and_enforces_gate():
    first = summarize_clip_result(
        _result([], fps=1.0, total_frames=100),
        RegressionClip("first", "negative", "cam_a"),
    )
    second = summarize_clip_result(
        _result([], fps=1.0, total_frames=900),
        RegressionClip("second", "negative", "cam_a"),
    )
    first["scene_guard"] = {"usable_frames": 90, "blocked_frames": 10}
    first["scene_guard"].update({
        "state_counts": {"stable": 90, "moved": 10},
        "observed_frames": 100,
        "enabled_frames": 100,
        "disabled_frames": 0,
    })
    second["scene_guard"] = {
        "state_counts": {"stable": 900},
        "observed_frames": 900,
        "enabled_frames": 900,
        "disabled_frames": 0,
        "usable_frames": 900,
        "blocked_frames": 0,
    }
    first["scored_frames"] = 100
    second["scored_frames"] = 900
    first["processed_frames"] = 100
    second["processed_frames"] = 900
    first["evaluation_mode"] = "pipeline"
    second["evaluation_mode"] = "pipeline"

    suite = aggregate_suite([first, second], min_scene_guard_coverage=0.995)

    assert suite["scene_guard_usable_frames"] == 990
    assert suite["scene_guard_blocked_frames"] == 10
    assert suite["scene_guard_usable_coverage_ratio"] == pytest.approx(0.99)
    assert suite["scene_guard_evidence_complete"] is True
    assert suite["acceptance"]["scene_guard_coverage"] == "fail"


def test_aggregate_suite_gates_on_score_eligible_guard_and_audits_all_frames():
    summary = summarize_clip_result(
        _result([], fps=1.0, total_frames=10),
        RegressionClip("split-guard", "negative", "cam_a"),
    )
    eligible_guard = {
        "state_counts": {"stable": 8},
        "observed_frames": 8,
        "enabled_frames": 8,
        "disabled_frames": 0,
        "usable_frames": 8,
        "blocked_frames": 0,
    }
    all_guard = {
        "state_counts": {"stable": 7, "moved": 3},
        "observed_frames": 10,
        "enabled_frames": 10,
        "disabled_frames": 0,
        "usable_frames": 7,
        "blocked_frames": 3,
    }
    summary.update({
        "evaluation_mode": "pipeline",
        "processed_frames": 10,
        "scored_frames": 8,
        "excluded_frames": 2,
        # ignored and warmup overlap, so neither independently equals the
        # number excluded from scoring.
        "ignored_frames": 1,
        "warmup_frames": 2,
        "scene_guard": eligible_guard,
        "scene_guard_summary_score_eligible": eligible_guard,
        "scene_guard_summary_all": all_guard,
    })

    suite = aggregate_suite([summary], min_scene_guard_coverage=0.99)

    assert suite["scene_guard_evidence_complete"] is True
    assert suite["scene_guard_usable_coverage_ratio"] == pytest.approx(1.0)
    assert suite["acceptance"]["scene_guard_coverage"] == "fail"
    assert suite["scene_guard_all_evidence_complete"] is True
    assert suite["scene_guard_all_raw_coverage_ratio"] == pytest.approx(0.7)
    assert suite["scene_guard_all_frame_evidence_complete"] is True
    assert suite["scene_guard_all_frame_usable_coverage_ratio"] == pytest.approx(
        0.7
    )
    assert suite["scene_guard_all_usable_frames"] == 7
    assert suite["scene_guard_all_blocked_frames"] == 3


def test_aggregate_suite_rejects_disabled_scene_guard_as_coverage_evidence():
    summary = summarize_clip_result(
        _result([], fps=1.0, total_frames=10),
        RegressionClip("disabled", "negative", "cam_a"),
    )
    summary["scored_frames"] = 10
    summary["processed_frames"] = 10
    summary["scene_guard"] = {
        "state_counts": {"disabled": 10},
        "observed_frames": 10,
        "enabled_frames": 0,
        "disabled_frames": 10,
        "usable_frames": 10,
        "blocked_frames": 0,
    }
    summary["evaluation_mode"] = "pipeline"

    suite = aggregate_suite([summary])

    assert suite["scene_guard_raw_coverage_ratio"] == 1.0
    assert suite["scene_guard_usable_coverage_ratio"] is None
    assert suite["scene_guard_evidence_complete"] is False
    assert suite["acceptance"]["scene_guard_coverage"] == "fail"


def test_aggregate_suite_rejects_mixed_guard_and_missing_evidence():
    protected = summarize_clip_result(
        _result([], fps=1.0, total_frames=10),
        RegressionClip("protected", "negative", "cam_a"),
    )
    protected["scored_frames"] = 10
    protected["processed_frames"] = 10
    protected["scene_guard"] = {
        "state_counts": {"stable": 10},
        "observed_frames": 10,
        "enabled_frames": 10,
        "disabled_frames": 0,
        "usable_frames": 10,
        "blocked_frames": 0,
    }
    protected["evaluation_mode"] = "pipeline"
    legacy = summarize_clip_result(
        _result([], fps=1.0, total_frames=10),
        RegressionClip("legacy", "negative", "cam_a"),
    )

    suite = aggregate_suite([protected, legacy])

    assert suite["scene_guard_evidence_complete"] is False
    assert suite["acceptance"]["scene_guard_coverage"] == "fail"


@pytest.mark.parametrize(
    (
        "failed_frames",
        "usable_frames",
        "blocked_frames",
        "scored_frames",
        "ignored_frames",
    ),
    [
        (1, 10, 0, 10, 0),
        (0, 1, 0, 10, 0),
        (0, 10, 0, 1, 0),
    ],
)
def test_aggregate_suite_rejects_failed_or_inconsistent_guard_observations(
    failed_frames,
    usable_frames,
    blocked_frames,
    scored_frames,
    ignored_frames,
):
    summary = summarize_clip_result(
        _result([], fps=1.0, total_frames=10),
        RegressionClip("invalid", "negative", "cam_a"),
    )
    summary.update({
        "evaluation_mode": "pipeline",
        "scored_frames": scored_frames,
        "processed_frames": 10,
        "ignored_frames": ignored_frames,
        "failed_frames": failed_frames,
        "scene_guard": {
            "state_counts": {"stable": 10},
            "observed_frames": 10,
            "enabled_frames": 10,
            "disabled_frames": 0,
            "usable_frames": usable_frames,
            "blocked_frames": blocked_frames,
        },
    })

    suite = aggregate_suite([summary])

    assert suite["scene_guard_evidence_complete"] is False
    assert suite["acceptance"]["scene_guard_coverage"] == "fail"


@pytest.mark.parametrize("minimum", [-0.01, 1.01])
def test_aggregate_suite_rejects_invalid_scene_guard_coverage(minimum):
    with pytest.raises(ValueError, match="between 0 and 1"):
        aggregate_suite([], min_scene_guard_coverage=minimum)


def test_load_manifest_resolves_paths_and_rejects_duplicate_ids(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
version: 1
default_camera_id: cam_a
clips:
  - id: neg
    kind: negative
    source_session_id: capture-session-001
    video: clips/neg.mp4
    scene_reference: references/neg.png
    scene_reference_frame: 12
    tags: [low_light]
""",
        encoding="utf-8",
    )

    raw, clips = load_manifest(manifest)

    assert raw["version"] == 1
    assert clips[0].clip_id == "neg"
    assert clips[0].camera_id == "cam_a"
    assert clips[0].source_session_id == "capture-session-001"
    assert clips[0].video_path == tmp_path / "clips" / "neg.mp4"
    assert clips[0].scene_reference_path == tmp_path / "references" / "neg.png"
    assert clips[0].scene_reference_frame_index == 12

    manifest.write_text(
        """
clips:
  - id: dup
  - id: dup
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate clip id"):
        load_manifest(manifest)

    manifest.write_text(
        "clips:\n"
        "  - id: invalid-session\n"
        "    source_session_id: ' session-with-whitespace'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source_session_id"):
        load_manifest(manifest)


@pytest.mark.parametrize(
    "expected_yaml",
    [
        "persistent_change",
        "[persistent_change, '']",
        "[persistent_change, 7]",
    ],
)
def test_load_manifest_rejects_non_list_or_empty_expected_detection_types(
    tmp_path,
    expected_yaml,
):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "clips:\n"
        "  - id: positive\n"
        "    kind: static_foreign\n"
        f"    expected_detection_types: {expected_yaml}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="list of non-empty strings"):
        load_manifest(manifest)


def test_main_summarizes_existing_eval_json(tmp_path):
    eval_json = tmp_path / "neg.video_eval.json"
    eval_json.write_text(
        json.dumps(
            _result([
                {"frame_index": 0, "timestamp_sec": 0.0, "score": 0.9, "decision": "fp"},
            ], fps=1.0, total_frames=3600)
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "report"

    code = main([
        "--from-eval-json", str(eval_json),
        "--output-dir", str(out_dir),
    ])

    assert code == 0
    report = json.loads((out_dir / "regression_report.json").read_text(encoding="utf-8"))
    assert report["suite"]["false_positive_event_count"] == 1
    assert (out_dir / "regression_report.md").is_file()


def test_cached_evaluation_requires_schema_existing_sources_and_matching_hashes(
    tmp_path,
):
    video = tmp_path / "case.mp4"
    annotation = tmp_path / "case.labels.json"
    config = tmp_path / "config.yaml"
    evaluation = tmp_path / "case.video_eval.json"
    video.write_bytes(b"video-v1")
    annotation.write_text('{"positive_ranges":[]}', encoding="utf-8")
    config.write_text("cameras: []\n", encoding="utf-8")
    payload = _result([], fps=30.0, total_frames=0) | {
        "schema_version": "argus.pipeline-video-eval.v1",
        "evaluation_origin": "fresh",
        "evaluation_mode": "pipeline",
        "camera_id": "cam_a",
        "video_path": str(video),
        "annotation_path": str(annotation),
        "source_video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "annotation_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
        "config_path": str(config),
        "config_artifact_sha256": _evidence_artifact_sha256_if_present(config),
        "source_artifacts_stable": True,
        "annotation_inventory": {
            "raw_range_counts": {"positive": 0, "ignore": 0},
            "parsed_range_counts": {"positive": 0, "ignore": 0},
            "clipped_range_counts": {"positive": 0, "ignore": 0},
            "invalid_range_count": 0,
            "out_of_bounds_range_count": 0,
            "loader_normalized_range_count": 0,
            "clipped_or_dropped_range_count": 0,
            "all_ranges_within_source": True,
        },
    }
    evaluation.write_text(json.dumps(payload), encoding="utf-8")
    clip = RegressionClip(
        "case",
        "negative",
        "cam_a",
        video_path=video,
        annotation_path=annotation,
        evaluation_json_path=evaluation,
    )

    loaded = _evaluate_clip(
        clip,
        output_dir=tmp_path / "out",
        model_path=None,
        threshold=0.7,
        image_size=256,
        sample_every=1,
        max_frames=None,
        sweep=False,
        config_path=config,
    )
    assert loaded["evaluation_origin"] == "cached"
    assert loaded["cached_evaluation_validation"]["valid"] is True

    annotation.write_text('{"positive_ranges":[{}]}', encoding="utf-8")
    stale = _evaluate_clip(
        clip,
        output_dir=tmp_path / "out",
        model_path=None,
        threshold=0.7,
        image_size=256,
        sample_every=1,
        max_frames=None,
        sweep=False,
        config_path=config,
    )
    assert stale["cached_evaluation_validation"]["valid"] is False
    assert "annotation_sha256_mismatch" in stale[
        "cached_evaluation_validation"
    ]["reasons"]


def test_main_fail_on_acceptance_rejects_insufficient_evidence(tmp_path):
    eval_json = tmp_path / "negative.video_eval.json"
    eval_json.write_text(
        json.dumps(_result([], fps=1.0, total_frames=3600)),
        encoding="utf-8",
    )

    code = main([
        "--from-eval-json",
        str(eval_json),
        "--fail-on-acceptance",
        "--output-dir",
        str(tmp_path / "report"),
    ])

    assert code == 1


def test_main_fail_on_acceptance_allows_explicit_not_applicable(
    monkeypatch,
    tmp_path,
):
    import scripts.run_regression_suite as regression

    monkeypatch.setattr(
        regression,
        "run_suite",
        lambda _args: {
            "status": "ok",
            "suite": {
                "evaluated_artifact_binding_evidence_complete": True,
                "acceptance": {
                    "corpus_completeness": "pass",
                    "candidate_confirmation": "not_applicable",
                },
            },
        },
    )

    assert regression.main([
        "--from-eval-json", str(tmp_path / "unused.json"),
        "--fail-on-acceptance",
        "--output-dir", str(tmp_path / "report"),
    ]) == 0


def test_main_fail_on_acceptance_rejects_incomplete_artifact_binding(
    monkeypatch,
    tmp_path,
):
    import scripts.run_regression_suite as regression

    monkeypatch.setattr(
        regression,
        "run_suite",
        lambda _args: {
            "status": "ok",
            "suite": {
                "evaluated_artifact_binding_evidence_complete": False,
                "acceptance": {
                    "corpus_completeness": "pass",
                    "candidate_confirmation": "not_applicable",
                },
            },
        },
    )

    assert regression.main([
        "--from-eval-json", str(tmp_path / "unused.json"),
        "--fail-on-acceptance",
        "--output-dir", str(tmp_path / "report"),
    ]) == 1


def test_main_allow_missing_reports_missing_inputs_without_inference(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
version: 1
clips:
  - id: missing
    kind: negative
    camera_id: cam_a
    video: clips/missing.mp4
""",
        encoding="utf-8",
    )
    out_dir = tmp_path / "report"

    code = main([
        "--manifest", str(manifest),
        "--allow-missing",
        "--output-dir", str(out_dir),
    ])

    assert code == 0
    report = json.loads((out_dir / "regression_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "missing_inputs"
    assert report["clips"][0]["status"] == "missing"

    fail_code = main([
        "--manifest", str(manifest),
        "--allow-missing",
        "--fail-on-acceptance",
        "--output-dir", str(tmp_path / "release-report"),
    ])
    assert fail_code == 1


def test_main_clip_id_filters_manifest(tmp_path):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
clips:
  - id: first
    kind: negative
    camera_id: cam
    video: first.mp4
  - id: second
    kind: negative
    camera_id: cam
    video: second.mp4
""",
        encoding="utf-8",
    )
    out_dir = tmp_path / "report"

    code = main([
        "--manifest", str(manifest),
        "--clip-id", "second",
        "--dry-run",
        "--output-dir", str(out_dir),
    ])

    assert code == 0
    report = json.loads((out_dir / "regression_report.json").read_text(encoding="utf-8"))
    assert [clip["clip_id"] for clip in report["clips"]] == ["second"]
    assert report["suite"]["expected_clip_count"] == 2
    assert report["suite"]["acceptance"]["corpus_completeness"] == "fail"

    assert main([
        "--manifest", str(manifest),
        "--clip-id", "second",
        "--dry-run",
        "--fail-on-acceptance",
        "--output-dir", str(tmp_path / "release-report"),
    ]) == 1


def test_main_rejects_negative_scene_reference_frame(tmp_path):
    assert main([
        "--manifest", str(tmp_path / "unused.yaml"),
        "--scene-reference-frame", "-1",
    ]) == 2


def test_evaluate_clip_allows_omitted_model_fallback(monkeypatch, tmp_path):
    import argus.anomaly.detector as detector_mod
    import argus.validation.video_eval as video_eval_mod

    load_calls: list[object] = []

    class FakeDetector:
        def __init__(self, *, model_path, threshold, image_size):
            self.model_path = model_path
            self.threshold = threshold
            self.image_size = image_size

        def load(self):
            load_calls.append(self.model_path)
            return False

    def fake_evaluate_video(detector, video_path, **kwargs):
        assert detector.model_path is None
        assert video_path == video
        assert kwargs["annotation_path"].is_file()
        return _result([], total_frames=0)

    monkeypatch.setattr(detector_mod, "AnomalibDetector", FakeDetector)
    monkeypatch.setattr(video_eval_mod, "evaluate_video", fake_evaluate_video)
    video = tmp_path / "neg.mp4"
    video.write_bytes(b"placeholder")
    clip = RegressionClip("neg", "negative", "cam_a", video_path=video)

    result = _evaluate_clip(
        clip,
        output_dir=tmp_path / "out",
        model_path=None,
        threshold=0.7,
        image_size=256,
        sample_every=30,
        max_frames=None,
        sweep=False,
    )

    assert result["total_frames"] == 0
    assert load_calls == [None]


def test_evaluate_clip_rejects_explicit_model_load_failure(monkeypatch, tmp_path):
    import argus.anomaly.detector as detector_mod

    class FakeDetector:
        def __init__(self, *, model_path, threshold, image_size):
            self.model_path = model_path

        def load(self):
            return False

    monkeypatch.setattr(detector_mod, "AnomalibDetector", FakeDetector)
    video = tmp_path / "neg.mp4"
    video.write_bytes(b"placeholder")
    model = tmp_path / "missing.xml"
    clip = RegressionClip("neg", "negative", "cam_a", video_path=video)

    with pytest.raises(RuntimeError, match="failed to load detector"):
        _evaluate_clip(
            clip,
            output_dir=tmp_path / "out",
            model_path=model,
            threshold=0.7,
            image_size=256,
            sample_every=30,
            max_frames=None,
            sweep=False,
        )


def test_evaluate_clip_passes_explicit_scene_reference_to_pipeline(monkeypatch, tmp_path):
    import argus.validation.pipeline_video_eval as pipeline_eval_mod

    video = tmp_path / "case.mp4"
    reference = tmp_path / "approved.png"
    video.write_bytes(b"placeholder")
    reference.write_bytes(b"placeholder")
    calls: list[dict] = []

    def fake_evaluate_pipeline_video(video_path, **kwargs):
        calls.append({"video_path": video_path, **kwargs})
        return _result([], total_frames=0)

    monkeypatch.setattr(
        pipeline_eval_mod,
        "evaluate_pipeline_video",
        fake_evaluate_pipeline_video,
    )
    clip = RegressionClip(
        "case",
        "negative",
        "cam_a",
        video_path=video,
        scene_reference_path=reference,
        scene_reference_frame_index=17,
    )

    _evaluate_clip(
        clip,
        output_dir=tmp_path / "out",
        model_path=None,
        threshold=0.7,
        image_size=256,
        sample_every=1,
        max_frames=None,
        sweep=False,
        evaluation_mode="pipeline",
    )

    assert calls[0]["scene_reference_path"] == reference
    assert calls[0]["scene_reference_frame_index"] == 17
