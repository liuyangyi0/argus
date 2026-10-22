"""Diagnostic comparison evidence must never become release authorization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.storage.release_evidence import (
    build_release_evidence,
    build_release_evidence_from_sources,
    validate_release_evidence,
)


def _comparison_report() -> dict:
    return {
        "schema_version": "argus.scene-guard-comparison.v1",
        "status": "complete",
        "diagnostic_only": True,
        "gating_effect": "none",
        "release_authority": False,
        "production_pipeline_parity_claimed": False,
        # Strict decoding says something about the video, not the model release.
        "decoder_evidence": {
            "baseline": {"sufficient_for_model_release": True},
            "lightglue": {"sufficient_for_model_release": True},
        },
    }


def _subject() -> dict[str, str]:
    return {
        "model_version_id": "camera-one-v1",
        "camera_id": "camera-one",
        "registry_model_hash": "a" * 16,
        "runtime_artifact_sha256": "b" * 64,
    }


def test_comparison_report_cannot_be_submitted_as_release_evidence():
    result = validate_release_evidence(_comparison_report(), **_subject())

    assert result.valid is False
    assert result.reasons


def test_release_builder_does_not_turn_complete_comparison_into_passing_validation():
    subject = _subject()
    evidence = build_release_evidence(
        _comparison_report(),
        model_version_id=subject["model_version_id"],
        camera_id=subject["camera_id"],
        registry_model_hash=subject["registry_model_hash"],
        config_sha256="c" * 64,
        scene_profile_sha256="d" * 64,
        baseline_sha256="e" * 64,
        corpus_sha256="f" * 64,
        operator_id="reviewer",
        operator_approved=True,
        approved_at="2026-10-21T00:00:00+00:00",
        soak_passed=True,
        soak_duration_hours=72.0,
        soak_report_sha256="1" * 64,
    )

    assert evidence["validation"]["report_status_ok"] is False
    assert evidence["validation"]["pipeline_evaluation"] is False
    assert evidence["validation"]["production_parity"] is False
    assert evidence["validation"]["runtime_model_identity"] is False
    assert validate_release_evidence(evidence, **subject).valid is False


def test_server_source_builder_rejects_comparison_instead_of_regression(tmp_path: Path):
    fields = (
        "regression_report_path", "config_path", "scene_profile_path", "baseline_path",
        "corpus_path", "soak_report_path", "soak_telemetry_path",
    )
    sources = {}
    for field in fields:
        path = tmp_path / f"{field}.json"
        contents = _comparison_report() if field == "regression_report_path" else {}
        path.write_text(json.dumps(contents), encoding="utf-8")
        sources[field] = str(path)
    subject = _subject()

    with pytest.raises(ValueError, match="regression report suite must be an object"):
        build_release_evidence_from_sources(
            sources,
            model_version_id=subject["model_version_id"],
            camera_id=subject["camera_id"],
            registry_model_hash=subject["registry_model_hash"],
            registry_data_hash="2" * 16,
            runtime_artifact_digest=subject["runtime_artifact_sha256"],
            operator_id="reviewer",
            operator_approved=True,
            approved_at="2026-10-21T00:00:00+00:00",
        )
