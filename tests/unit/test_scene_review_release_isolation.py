"""Human pose review is not gate approval, anomaly GT or model release evidence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from argus.integrations.contracts import FrameTimestamp, ReviewAnnotations, ReviewBundle, ReviewClip
from argus.integrations.fiftyone import _frame_fields, _scene_frame_fields
from argus.integrations.fiftyone_scene_images import _image_fields
from argus.storage.release_evidence import build_release_evidence, validate_release_evidence


def _clip() -> ReviewClip:
    return ReviewClip(
        clip_id="review-only", camera_id="camera-one", video_path="unused.mp4",
        video_sha256="a" * 64, fps=10.0, frame_count=1, width=640, height=480,
        timestamps=[FrameTimestamp(
            frame_index=0, pts=100, time_base_num=1, time_base_den=10, timestamp_sec=10.0,
        )],
    )


def _scene_row() -> dict:
    def decision(state: str) -> dict:
        return {
            "status": "evaluated", "state": state, "raw_state": state,
            "reason": "independent diagnostic", "metrics": {"inliers": 30},
            "elapsed_ms": 1.0,
        }

    return {
        "frame_index": 0, "frame_sha256": "b" * 64,
        "review_category": "confirmed_disagreement", "comparable": True,
        "state_disagreement": True, "raw_state_disagreement": True,
        "comparison_error": False, "error_algorithms": [],
        "baseline": decision("moved"), "lightglue": decision("stable"),
    }


@pytest.mark.parametrize("human_state", ["same_pose", "moved", "uncertain", "unreviewable"])
def test_human_scene_review_neither_labels_anomaly_gt_nor_mutates_clip(human_state: str):
    clip = _clip()
    before = clip.model_dump(mode="json")
    frame = {
        **_frame_fields(clip, 0, None, None),
        **_scene_frame_fields(SimpleNamespace(report_sha256="c" * 64), _scene_row()),
        "argus_scene_human_state": human_state,
        "argus_scene_reviewed_report_sha256": "c" * 64,
    }

    assert frame["argus_truth"] is None
    assert frame["argus_review_state"] == "unreviewed"
    assert frame["argus_decision"] == "unreviewed"
    assert frame["argus_scene_guard_usable"] is None
    assert frame["argus_score_eligible"] is None
    assert clip.model_dump(mode="json") == before


def test_stable_lightglue_diagnostic_cannot_clear_an_existing_pipeline_block():
    clip = _clip()
    record = {
        "scene_guard": {"usable": False, "state": "moved", "reason": "production gate"},
        "decision": "blocked", "pipeline_mode": "active", "predicted": True,
        "score": 0.99, "score_eligible": False,
    }
    base_fields = _frame_fields(clip, 0, record, None)
    scene_fields = _scene_frame_fields(SimpleNamespace(report_sha256="c" * 64), _scene_row())
    frame = {**base_fields, **scene_fields}

    assert frame["argus_scene_lightglue_state"] == "stable"
    assert frame["argus_scene_guard_state"] == "moved"
    assert frame["argus_scene_guard_usable"] is False
    assert frame["argus_blocked"] is True
    assert frame["argus_decision"] == "blocked"
    assert frame["argus_score_eligible"] is False
    assert not set(base_fields).intersection(scene_fields)
    assert "argus_scene_human_state" not in scene_fields
    assert "argus_scene_reviewed_report_sha256" not in scene_fields


def test_still_image_observations_cannot_write_human_review_gt_or_gate_fields():
    clip = _clip()
    bundle = ReviewBundle(
        bundle_id="diagnostic-images", suite_id="review-only", manifest_sha256="a" * 64,
        clips=[clip],
    )
    row = _scene_row()
    fields = _image_fields(
        bundle, clip,
        {
            "comparison": row, "frame_index": 0, "frame_number": 1,
            "pts_exact": "100", "time_base_exact": "1/10", "timestamp_sec": 10.0,
            "relative_timestamp_sec": 0.0, "frame_sha256": "b" * 64,
            "image_sha256": "d" * 64, "report_sha256": "c" * 64,
        },
        {
            "report_path": "diagnostic-only.json", "report_status": "partial",
            "profile_id": "diagnostic-profile", "summary": {"release_authority": False},
        },
        {"artifact_id": "e" * 64}, manifest_sha256="f" * 64, generation="1" * 64,
        media_identity="2" * 64, summary_json='{"release_authority":false}',
    )

    forbidden = {
        "argus_truth", "ground_truth", "argus_review_state", "argus_decision",
        "argus_scene_guard_usable", "argus_scene_guard_state", "argus_blocked",
        "argus_score", "argus_score_eligible", "argus_scene_human_state",
        "argus_scene_human_note", "argus_scene_reviewed_report_sha256",
    }
    assert not forbidden.intersection(fields)
    assert fields["argus_scene_diagnostic_only"] is True
    assert fields["argus_scene_gating_effect"] == "none"
    assert fields["argus_scene_release_authority"] is False
    assert fields["argus_scene_ground_truth_authority"] is False
    assert clip.annotations.reviewed is False


@pytest.mark.parametrize("field", [
    "argus_scene_human_state", "argus_scene_human_note", "argus_scene_reviewed_report_sha256",
])
def test_pose_review_fields_are_not_annotation_contract_fields(field: str):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ReviewAnnotations.model_validate({field: "same_pose"})


def _diagnostic_with_human_review() -> dict:
    return {
        "schema_version": "argus.scene-guard-comparison.v1", "status": "complete",
        "diagnostic_only": True, "gating_effect": "none", "release_authority": False,
        "production_pipeline_parity_claimed": False,
        "argus_scene_human_state": "same_pose",
        "argus_scene_reviewed_report_sha256": "c" * 64,
        "argus_scene_comparison_authority": "diagnostic_human_review_only",
        "reviewed": True, "reviewed_by": "pose-reviewer",
        "decoder_evidence": {"sufficient_for_model_release": True},
    }


@pytest.mark.parametrize("schema", [
    "argus.scene-guard-comparison.v1", "argus.scene-review-images.v1",
])
def test_human_reviewed_comparison_cannot_masquerade_as_pipeline_evaluation(schema: str):
    clip = _clip().model_dump(mode="json")
    clip["evaluation"] = _diagnostic_with_human_review()
    clip["evaluation"]["schema_version"] = schema
    with pytest.raises(ValidationError, match="production pipeline video-eval"):
        ReviewClip.model_validate(clip)


@pytest.mark.parametrize("schema", [
    "argus.scene-guard-comparison.v1", "argus.scene-review-images.v1",
])
def test_human_pose_confirmation_is_not_model_release_evidence(schema: str):
    diagnostic = _diagnostic_with_human_review()
    diagnostic["schema_version"] = schema
    subject = {
        "model_version_id": "camera-one-v1", "camera_id": "camera-one",
        "registry_model_hash": "a" * 16, "runtime_artifact_sha256": "b" * 64,
    }
    assert validate_release_evidence(diagnostic, **subject).valid is False
    evidence = build_release_evidence(
        diagnostic,
        model_version_id=subject["model_version_id"], camera_id=subject["camera_id"],
        registry_model_hash=subject["registry_model_hash"], config_sha256="c" * 64,
        scene_profile_sha256="d" * 64, baseline_sha256="e" * 64, corpus_sha256="f" * 64,
        operator_id="pose-reviewer", operator_approved=True,
        approved_at="2026-10-21T00:00:00+00:00", soak_passed=True,
        soak_duration_hours=72.0, soak_report_sha256="1" * 64,
    )
    assert evidence["validation"]["report_status_ok"] is False
    assert evidence["validation"]["pipeline_evaluation"] is False
    assert evidence["validation"]["production_parity"] is False
    assert validate_release_evidence(evidence, **subject).valid is False
