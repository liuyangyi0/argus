"""Workflow binding tests; the already-tested canonical V3 parser is mocked."""

from __future__ import annotations

import hashlib
import json
import stat
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from argus.anomaly import scene_training as contract
from argus.storage.release_evidence import evidence_artifact_sha256


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def scene_input(tmp_path, monkeypatch):
    base = tmp_path / "baselines"
    baseline = base / "camera-01" / "default" / "scene-workflow-01"
    images = baseline / "images"
    images.mkdir(parents=True)
    (images / "normal.png").write_bytes(b"immutable normal sample")
    reference = baseline / "reference.jpg"
    reference.write_bytes(b"explicit trusted scene reference")
    scene_policy = {"min_matches": 40, "min_inliers": 18, "min_coverage_ratio": 0.12}
    manifest = {
        "schema_version": "argus.scene-profile-baseline.v3",
        "camera_id": "camera-01",
        "scene_profile_id": "workflow-01",
        "training_ready": True,
        "release_eligible": False,
        "decoder_evidence": {"sufficient_for_model_release": True},
        "parameters": {
            "strict_release_decoder": True,
            "allow_alignable": False,
            "min_training_images": 100,
            "min_calibration_images": 50,
            "min_accepted_time_span_seconds": 300.0,
            "scene_guard": deepcopy(scene_policy),
        },
        "reference": {"sha256": _sha256(reference)},
        "split_contract": {
            "policy": "complete_operator_declared_source_session_grouping",
            "assignment": (
                "lexicographically_greatest_declared_session_id_is_calibration; "
                "all_other_declared_sessions_are_train"
            ),
            "random_frame_split_allowed": False,
            "declared_session_ids": ["session-a", "session-b"],
            "sample_counts": {"train": 100, "calibration": 50, "diagnostic_only": 0},
        },
    }
    manifest_path = baseline / "manifest.json"
    record_path = tmp_path / "scene_workflows" / "camera-01" / "workflow-01" / "workflow.json"
    record_path.parent.mkdir(parents=True)
    descriptor = {
        "schema_version": contract.SCENE_TRAINING_INPUT_SCHEMA,
        "workflow_id": "workflow-01",
        "scene_profile_id": "workflow-01",
        "baseline_sha256": "",
        "manifest_sha256": "",
        "reference_sha256": _sha256(reference),
    }

    def save_binding():
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        descriptor["manifest_sha256"] = _sha256(manifest_path)
        descriptor["baseline_sha256"] = evidence_artifact_sha256(baseline)
        record_path.write_text(json.dumps({
            **descriptor,
            "schema_version": contract.SCENE_WORKFLOW_SCHEMA,
            "camera_id": "camera-01",
            "state": "baseline_ready",
            "scene_guard": deepcopy(scene_policy),
        }), encoding="utf-8")

    save_binding()
    canonical = Mock(side_effect=lambda _root: (
        deepcopy(manifest), {"a" * 64, "b" * 64}, {"session-a", "session-b"},
    ))
    monkeypatch.setattr(contract, "validate_baseline_manifest_source", canonical)
    return {
        "base": base, "baseline": baseline, "images": images,
        "reference": reference, "manifest": manifest, "manifest_path": manifest_path,
        "record_path": record_path, "descriptor": descriptor,
        "save_binding": save_binding, "canonical": canonical,
    }


def _validate(value):
    return contract.validate_scene_training_input(
        baselines_dir=value["base"], camera_id="camera-01", scene_workflow=value["descriptor"],
    )


def test_scene_training_resolves_server_paths_and_reuses_canonical_v3_validation(scene_input):
    verified = _validate(scene_input)

    assert verified.root == scene_input["baseline"]
    assert verified.images_dir == scene_input["images"]
    assert (verified.image_count, verified.training_count, verified.calibration_count) == (150, 100, 50)
    assert verified.lineage == scene_input["descriptor"]
    assert verified.lineage is not scene_input["descriptor"]
    scene_input["canonical"].assert_called_once_with(scene_input["baseline"])
    assert contract.revalidate_scene_training_input(verified) == verified


@pytest.mark.parametrize("damage", [
    "missing", "old-schema", "extra-path", "traversal", "profile", "uppercase-sha", "null",
])
def test_scene_descriptor_never_accepts_legacy_or_client_paths(scene_input, damage):
    descriptor = deepcopy(scene_input["descriptor"])
    if damage == "missing":
        del descriptor["manifest_sha256"]
    elif damage == "old-schema":
        descriptor["schema_version"] = "argus.scene-training-input.v0"
    elif damage == "extra-path":
        descriptor["baseline_path"] = "/arbitrary/source"
    elif damage == "traversal":
        descriptor["workflow_id"] = "../another"
    elif damage == "profile":
        descriptor["scene_profile_id"] = "another-profile"
    elif damage == "uppercase-sha":
        descriptor["baseline_sha256"] = "A" * 64
    else:
        descriptor = None
    with pytest.raises(ValueError):
        contract.validate_scene_training_descriptor(descriptor)


@pytest.mark.parametrize("field", [
    "schema_version", "camera_id", "workflow_id", "scene_profile_id",
    "baseline_sha256", "manifest_sha256", "reference_sha256",
])
def test_scene_job_must_match_persisted_workflow_identity(scene_input, field):
    path = scene_input["record_path"]
    record = json.loads(path.read_text(encoding="utf-8"))
    record[field] = "different-identity"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="record"):
        _validate(scene_input)
    scene_input["canonical"].assert_not_called()


def test_workflow_display_state_cannot_grant_or_revoke_input_identity(scene_input):
    before = _validate(scene_input)
    path = scene_input["record_path"]
    record = json.loads(path.read_text(encoding="utf-8"))
    record["state"] = "production_ready"  # Never treated as an approval capability.
    path.write_text(json.dumps(record), encoding="utf-8")
    assert _validate(scene_input) == before


@pytest.mark.parametrize("artifact", ["image", "manifest", "reference", "extra-file"])
def test_scene_training_refuses_any_changed_baseline_bytes(scene_input, artifact):
    target = {
        "image": scene_input["images"] / "normal.png",
        "manifest": scene_input["manifest_path"],
        "reference": scene_input["reference"],
        "extra-file": scene_input["baseline"] / "untracked.txt",
    }[artifact]
    target.write_bytes(b"changed after queuing")
    with pytest.raises(ValueError, match="SHA256"):
        _validate(scene_input)
    scene_input["canonical"].assert_not_called()


@pytest.mark.parametrize("damage", [
    "v2", "wrong-camera", "wrong-profile", "not-ready", "decoder", "alignable",
    "random-split", "single-session", "calibration-minimum", "calibration-count",
    "diagnostic-samples", "reference", "short-span", "weaker-scene-policy", "missing-scene-policy",
])
def test_scene_workflow_rejects_unsafe_bound_manifest_even_with_fresh_hashes(scene_input, damage):
    manifest = scene_input["manifest"]
    if damage == "v2":
        manifest["schema_version"] = "argus.scene-profile-baseline.v2"
    elif damage == "wrong-camera":
        manifest["camera_id"] = "another-camera"
    elif damage == "wrong-profile":
        manifest["scene_profile_id"] = "another-profile"
    elif damage == "not-ready":
        manifest["training_ready"] = False
    elif damage == "decoder":
        manifest["decoder_evidence"]["sufficient_for_model_release"] = False
    elif damage == "alignable":
        manifest["parameters"]["allow_alignable"] = True
    elif damage == "random-split":
        manifest["split_contract"]["random_frame_split_allowed"] = True
    elif damage == "single-session":
        manifest["split_contract"]["declared_session_ids"] = ["session-a"]
    elif damage == "calibration-minimum":
        manifest["parameters"]["min_calibration_images"] = 20
    elif damage == "calibration-count":
        manifest["split_contract"]["sample_counts"]["calibration"] = 49
    elif damage == "diagnostic-samples":
        manifest["split_contract"]["sample_counts"]["diagnostic_only"] = 1
    elif damage == "reference":
        manifest["reference"]["sha256"] = "c" * 64
    elif damage == "weaker-scene-policy":
        manifest["parameters"]["scene_guard"]["min_matches"] = 24
    elif damage == "missing-scene-policy":
        del manifest["parameters"]["scene_guard"]
    else:
        manifest["parameters"]["min_accepted_time_span_seconds"] = 1.0
    scene_input["save_binding"]()
    with pytest.raises(ValueError):
        _validate(scene_input)


def test_scene_training_propagates_canonical_decoder_or_pts_refusal(scene_input):
    scene_input["canonical"].side_effect = ValueError("strict PTS inventory does not match source")
    with pytest.raises(ValueError, match="strict PTS"):
        _validate(scene_input)


def test_scene_training_refuses_a_record_without_its_frozen_scene_policy(scene_input):
    path = scene_input["record_path"]
    record = json.loads(path.read_text(encoding="utf-8"))
    del record["scene_guard"]
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen scene_guard policy"):
        _validate(scene_input)
    scene_input["canonical"].assert_not_called()


def test_scene_training_refuses_frozen_policy_changes_during_verification(scene_input):
    canonical = scene_input["canonical"].side_effect

    def changed_policy(root):
        result = canonical(root)
        path = scene_input["record_path"]
        record = json.loads(path.read_text(encoding="utf-8"))
        record["scene_guard"]["min_matches"] = 24
        path.write_text(json.dumps(record), encoding="utf-8")
        return result

    scene_input["canonical"].side_effect = changed_policy
    with pytest.raises(ValueError, match="scene_guard policy changed"):
        _validate(scene_input)


def test_revalidation_refuses_baseline_mutation_during_training(scene_input):
    verified = _validate(scene_input)
    (scene_input["images"] / "normal.png").write_bytes(b"changed while model was training")
    with pytest.raises(ValueError, match="SHA256"):
        contract.revalidate_scene_training_input(verified)


def test_scene_workflow_rejects_other_zones_before_resolving_data(scene_input):
    with pytest.raises(ValueError, match="default zone"):
        contract.validate_scene_training_input(
            baselines_dir=scene_input["base"], camera_id="camera-01", zone_id="another-zone",
            scene_workflow=scene_input["descriptor"],
        )


@pytest.fixture
def calibration_artifact(scene_input):
    verified = _validate(scene_input)
    artifact = scene_input["base"].parent / "exports" / "model.pt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"scene runtime")
    calibration = {
        "info": 0.1, "low": 0.11, "medium": 0.12, "high": 0.13,
        "n_samples": 50,
        "target_fprs": {"info": 0.1, "low": 0.01, "medium": 0.001, "high": 0.0001},
        "actual_fprs": {"info": 0.0, "low": 0.0, "medium": 0.0, "high": 0.0},
        "sorted_scores": [0.1] * 50,
    }
    path = artifact.parent / "calibration.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")
    return verified, artifact, path, calibration


def test_scene_calibration_matches_the_complete_frozen_split(calibration_artifact):
    verified, artifact, path, calibration = calibration_artifact
    before = path.read_bytes()
    assert contract.validate_scene_training_calibration(
        model_path=artifact, scene_training_input=verified,
    ) == calibration
    assert path.read_bytes() == before


def test_scene_calibration_accepts_the_real_calibrator_serialization(calibration_artifact):
    import numpy as np

    from argus.alerts.calibration import ConformalCalibrator

    verified, artifact, path, _calibration = calibration_artifact
    # Native scores deliberately differ past the six persisted decimals. A
    # loader must not recompute empirical FPR against rounded scores.
    scores = np.linspace(0.1000001, 0.1000004, verified.calibration_count)
    calibrator = ConformalCalibrator()
    calibrator.save(calibrator.calibrate(scores), path, sorted_scores=np.sort(scores))
    result = contract.validate_scene_training_calibration(
        model_path=artifact, scene_training_input=verified,
    )
    assert result["n_samples"] == len(result["sorted_scores"]) == verified.calibration_count


@pytest.mark.parametrize("damage", [
    "missing-count", "false-count", "bool-count", "missing-scores", "short-scores",
    "long-scores", "bool-score", "nan-score", "infinite-score", "unsorted-scores",
    "missing-threshold", "nan-threshold", "bool-threshold", "unordered-thresholds",
    "missing-alpha", "extra-alpha", "bool-alpha", "zero-alpha", "one-alpha",
    "unordered-alpha", "missing-actual", "nan-actual", "negative-actual",
    "excessive-actual", "bool-actual", "unordered-actual",
])
def test_scene_calibration_refuses_claimed_ok_but_incomplete_or_invalid_evidence(calibration_artifact, damage):
    verified, artifact, path, calibration = calibration_artifact
    if damage == "missing-count":
        del calibration["n_samples"]
    elif damage == "false-count":
        calibration["n_samples"] = 51
    elif damage == "bool-count":
        calibration["n_samples"] = True
    elif damage == "missing-scores":
        del calibration["sorted_scores"]
    elif damage == "short-scores":
        calibration["sorted_scores"].pop()
    elif damage == "long-scores":
        calibration["sorted_scores"].append(0.1)
    elif damage == "bool-score":
        calibration["sorted_scores"][0] = False
    elif damage == "nan-score":
        calibration["sorted_scores"][0] = float("nan")
    elif damage == "infinite-score":
        calibration["sorted_scores"][-1] = float("inf")
    elif damage == "unsorted-scores":
        calibration["sorted_scores"][0] = 0.2
    elif damage == "missing-threshold":
        del calibration["high"]
    elif damage == "nan-threshold":
        calibration["high"] = float("nan")
    elif damage == "bool-threshold":
        calibration["info"] = False
    elif damage == "unordered-thresholds":
        calibration["high"] = calibration["medium"]
    elif damage == "missing-alpha":
        del calibration["target_fprs"]["high"]
    elif damage == "extra-alpha":
        calibration["target_fprs"]["critical"] = 0.00001
    elif damage == "bool-alpha":
        calibration["target_fprs"]["info"] = True
    elif damage == "zero-alpha":
        calibration["target_fprs"]["high"] = 0
    elif damage == "one-alpha":
        calibration["target_fprs"]["info"] = 1
    elif damage == "unordered-alpha":
        calibration["target_fprs"]["low"] = calibration["target_fprs"]["info"]
    elif damage == "missing-actual":
        del calibration["actual_fprs"]["high"]
    elif damage == "nan-actual":
        calibration["actual_fprs"]["info"] = float("nan")
    elif damage == "negative-actual":
        calibration["actual_fprs"]["high"] = -0.1
    elif damage == "excessive-actual":
        calibration["actual_fprs"]["info"] = 1.1
    elif damage == "bool-actual":
        calibration["actual_fprs"]["info"] = False
    else:
        calibration["actual_fprs"]["high"] = 0.1
    path.write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(ValueError, match="scene calibration"):
        contract.validate_scene_training_calibration(model_path=artifact, scene_training_input=verified)


@pytest.mark.parametrize("content", [b"", b"not-json", b"[]", b"{\"sorted_scores\":[0.1]}"])
def test_scene_calibration_refuses_empty_malformed_or_partial_json(calibration_artifact, content):
    verified, artifact, path, _calibration = calibration_artifact
    path.write_bytes(content)
    with pytest.raises(ValueError):
        contract.validate_scene_training_calibration(model_path=artifact, scene_training_input=verified)


def test_scene_calibration_never_falls_back_to_an_ancestor_sidecar(calibration_artifact):
    verified, artifact, path, _calibration = calibration_artifact
    path.replace(path.parent.parent / "calibration.json")
    with pytest.raises(FileNotFoundError):
        contract.validate_scene_training_calibration(model_path=artifact, scene_training_input=verified)


def test_scene_calibration_keeps_the_controlled_json_budget(calibration_artifact, monkeypatch):
    verified, artifact, _path, _calibration = calibration_artifact
    monkeypatch.setattr(contract, "_MAX_METADATA_BYTES", 16)
    with pytest.raises(ValueError, match="budget"):
        contract.validate_scene_training_calibration(model_path=artifact, scene_training_input=verified)


@pytest.mark.parametrize("target", ["model", "sidecar", "directory"])
@pytest.mark.parametrize("kind", ["symlink", "reparse"])
def test_scene_calibration_rejects_links_and_reparse_points(calibration_artifact, monkeypatch, target, kind):
    verified, artifact, path, _calibration = calibration_artifact
    selected = {"model": artifact, "sidecar": path, "directory": artifact.parent}[target]
    original = Path.lstat

    def damaged_metadata(candidate, *args, **kwargs):
        if candidate == selected:
            return Mock(
                st_mode=stat.S_IFLNK if kind == "symlink" else stat.S_IFREG,
                st_file_attributes=0x400 if kind == "reparse" else 0,
            )
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", damaged_metadata)
    with pytest.raises(ValueError, match="links or reparse"):
        contract.validate_scene_training_calibration(model_path=artifact, scene_training_input=verified)


@pytest.mark.parametrize("count", [49, True, 50.0])
def test_scene_calibration_never_accepts_an_uncontrolled_expected_count(calibration_artifact, count):
    verified, artifact, _path, _calibration = calibration_artifact
    with pytest.raises(ValueError, match="complete frozen calibration split"):
        contract.validate_scene_training_calibration(
            model_path=artifact, scene_training_input=replace(verified, calibration_count=count),
        )
