"""Scene job orchestration tests with fake trainer, registry and database."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from argus.anomaly import job_executor as execution
from argus.anomaly.artifacts import new_artifact_run, write_artifact_manifest
from argus.anomaly.scene_training import SCENE_TRAINING_INPUT_SCHEMA, SceneTrainingInput
from argus.anomaly.trainer import TrainingResult, TrainingStatus


@pytest.fixture
def scene_job(tmp_path, monkeypatch):
    lineage = {
        "schema_version": SCENE_TRAINING_INPUT_SCHEMA,
        "workflow_id": "workflow-01",
        "scene_profile_id": "workflow-01",
        "baseline_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "reference_sha256": "c" * 64,
    }
    baseline = tmp_path / "baselines" / "camera-01" / "default" / "scene-workflow-01"
    verified = SceneTrainingInput(
        baselines_dir=tmp_path / "baselines", camera_id="camera-01",
        root=baseline, images_dir=baseline / "images", image_count=150,
        training_count=100, calibration_count=50, lineage=lineage,
    )
    run = new_artifact_run(tmp_path / "models", tmp_path / "exports", "camera-01", "default")
    artifact = run.export_dir / "model.pt"
    artifact.write_bytes(b"fake scene runtime")
    (artifact.parent / "calibration.json").write_text(json.dumps({
        "info": 0.1, "low": 0.11, "medium": 0.12, "high": 0.13,
        "n_samples": 50, "sorted_scores": [0.1] * 50,
        "target_fprs": {"info": 0.1, "low": 0.01, "medium": 0.001, "high": 0.0001},
        "actual_fprs": {"info": 0.0, "low": 0.0, "medium": 0.0, "high": 0.0},
    }), encoding="utf-8")
    write_artifact_manifest(run, artifact, model_type="patchcore", calibration_status="ok")
    trainer = Mock()
    trainer.train.return_value = TrainingResult(
        status=TrainingStatus.COMPLETE, model_path=str(artifact),
        image_count=150, train_count=100, val_count=50,
        duration_seconds=1.0, calibration_status="ok", actual_export_format="torch",
        actual_quantization="fp32",
    )
    registry = Mock()
    registry.register.return_value = "scene-candidate-01"
    db = Mock()
    release = Mock()
    executor = execution.TrainingJobExecutor(
        database=db, trainer=trainer, model_registry=registry,
        baselines_dir=verified.baselines_dir, model_packages_dir=tmp_path / "packages",
        model_release_service=release,
    )
    job = SimpleNamespace(
        job_id="scene-job-01", camera_id="camera-01", zone_id="default",
        job_type="anomaly_head", model_type="patchcore", dataset_selection=None,
        dataset_version="scene-workflow-01",
        hyperparameters=json.dumps({
            "scene_workflow": lineage, "export_format": "torch", "quantization": "fp32",
        }),
    )
    validate = Mock(return_value=verified)
    revalidate = Mock(return_value=verified)
    monkeypatch.setattr(execution, "validate_scene_training_input", validate)
    monkeypatch.setattr(execution, "revalidate_scene_training_input", revalidate)
    return SimpleNamespace(
        job=job, executor=executor, trainer=trainer, registry=registry, db=db,
        release=release, input=verified, validate=validate, revalidate=revalidate,
    )


def test_scene_job_uses_exact_input_requires_calibration_and_only_registers_candidate(scene_job):
    value = scene_job
    value.executor._execute_head(value.job)

    arguments = value.trainer.train.call_args.kwargs
    assert arguments["baseline_dir_override"] == value.input.images_dir
    assert arguments["image_count_override"] == 150
    assert arguments["require_calibration"] is True
    assert arguments["scene_training_input"] is value.input
    assert arguments["skip_baseline_validation"] is False
    value.revalidate.assert_called_once_with(value.input)
    registration = value.registry.register.call_args.kwargs
    assert registration["baseline_dir"] == value.input.root
    assert registration["training_params"]["scene_workflow"] == value.input.lineage
    update = value.db.update_training_job.call_args.kwargs
    assert update["model_version_id"] == "scene-candidate-01"
    metrics = json.loads(update["metrics"])
    assert metrics["scene_workflow"] == value.input.lineage
    assert metrics["calibration_status"] == "ok"
    assert metrics["train_count"] == 100 and metrics["calibration_count"] == 50
    value.registry.activate.assert_not_called()
    value.release.request_transition.assert_not_called()


@pytest.mark.parametrize("status", [None, "skipped", "failed"])
def test_scene_job_cannot_register_an_uncalibrated_trainer_result(scene_job, status):
    scene_job.trainer.train.return_value.calibration_status = status
    with pytest.raises(RuntimeError, match="calibration"):
        scene_job.executor._execute_head(scene_job.job)
    scene_job.registry.register.assert_not_called()
    scene_job.release.request_transition.assert_not_called()


def test_scene_job_refuses_mixed_dataset_selection_before_training(scene_job):
    scene_job.job.dataset_selection = '{"camera_id":"camera-01","items":[]}'
    with pytest.raises(ValueError, match="cannot merge dataset_selection"):
        scene_job.executor._execute_head(scene_job.job)
    scene_job.validate.assert_not_called()
    scene_job.trainer.train.assert_not_called()
    scene_job.registry.register.assert_not_called()


@pytest.mark.parametrize("extra", [
    {"skip_baseline_validation": True}, {"labeling_entry_ids": [101]},
])
def test_scene_job_cannot_skip_validation_or_consume_unrelated_labels(scene_job, extra):
    params = json.loads(scene_job.job.hyperparameters)
    params.update(extra)
    scene_job.job.hyperparameters = json.dumps(params)
    with pytest.raises(ValueError, match="Invalid hyperparameters"):
        scene_job.executor._execute_head(scene_job.job)
    scene_job.validate.assert_not_called()
    scene_job.trainer.train.assert_not_called()
    scene_job.registry.register.assert_not_called()


def test_scene_job_refuses_source_mutation_before_candidate_registration(scene_job):
    scene_job.revalidate.side_effect = ValueError("baseline SHA256 changed during training")
    with pytest.raises(ValueError, match="SHA256"):
        scene_job.executor._execute_head(scene_job.job)
    scene_job.registry.register.assert_not_called()
    scene_job.release.request_transition.assert_not_called()


def test_scene_job_refuses_claimed_ok_calibration_without_a_sealed_sidecar(scene_job):
    artifact = Path(scene_job.trainer.train.return_value.model_path)
    (artifact.parent / "calibration.json").unlink()
    with pytest.raises(RuntimeError, match="finalized calibrated runtime"):
        scene_job.executor._execute_head(scene_job.job)
    scene_job.registry.register.assert_not_called()


@pytest.mark.parametrize("damage", ["one-score", "false-count", "nan", "unsorted", "thresholds", "alpha"])
def test_scene_job_refuses_false_calibration_evidence_before_registration(scene_job, damage):
    artifact = Path(scene_job.trainer.train.return_value.model_path)
    path = artifact.parent / "calibration.json"
    calibration = json.loads(path.read_text(encoding="utf-8"))
    if damage == "one-score":
        calibration["sorted_scores"] = [0.1]
    elif damage == "false-count":
        calibration["n_samples"] = 100
    elif damage == "nan":
        calibration["sorted_scores"][0] = float("nan")
    elif damage == "unsorted":
        calibration["sorted_scores"][0] = 0.2
    elif damage == "thresholds":
        calibration["high"] = calibration["medium"]
    else:
        calibration["target_fprs"]["high"] = 0.0
    path.write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(ValueError, match="scene calibration"):
        scene_job.executor._execute_head(scene_job.job)
    scene_job.registry.register.assert_not_called()
    scene_job.db.update_training_job.assert_not_called()
    scene_job.release.request_transition.assert_not_called()


def test_present_but_null_scene_descriptor_cannot_fall_back_to_legacy(scene_job):
    scene_job.job.hyperparameters = json.dumps({"scene_workflow": None})
    with pytest.raises(ValueError, match="exact scene training identity fields"):
        scene_job.executor._execute_head(scene_job.job)
    scene_job.trainer.train.assert_not_called()
    scene_job.registry.register.assert_not_called()


def test_scene_hyperparameters_never_accept_a_client_baseline_path(scene_job):
    params = json.loads(scene_job.job.hyperparameters)
    params["scene_workflow"]["baseline_path"] = str(Path("arbitrary") / "baseline")
    errors = execution.validate_hyperparameters(params)
    assert any("exact scene training identity fields" in error for error in errors)


def test_scene_dataset_version_without_descriptor_cannot_fall_back_to_current_baseline(scene_job):
    scene_job.job.hyperparameters = json.dumps({"image_size": 256})
    with pytest.raises(ValueError, match="legacy fallback is forbidden"):
        scene_job.executor._execute_head(scene_job.job)
    scene_job.validate.assert_not_called()
    scene_job.trainer.train.assert_not_called()
    scene_job.registry.register.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("dataset_version", None), ("dataset_version", "v001"),
    ("dataset_version", "scene-another-workflow"),
    ("zone_id", None), ("zone_id", "other-zone"), ("job_type", "ssl_backbone"),
])
def test_scene_job_metadata_must_match_its_descriptor_before_any_training(scene_job, field, value):
    setattr(scene_job.job, field, value)
    with pytest.raises(ValueError, match="must match its descriptor"):
        scene_job.executor._execute_head(scene_job.job)
    scene_job.validate.assert_not_called()
    scene_job.trainer.train.assert_not_called()
    scene_job.registry.register.assert_not_called()


def test_backbone_job_cannot_consume_a_scene_version_even_without_descriptor(scene_job):
    backbone = Mock()
    scene_job.executor._backbone_trainer = backbone
    scene_job.job.job_type = "ssl_backbone"
    scene_job.job.hyperparameters = json.dumps({"image_size": 256})
    with pytest.raises(ValueError, match="only supported for anomaly_head"):
        scene_job.executor._execute_backbone(scene_job.job)
    backbone.train.assert_not_called()
    scene_job.trainer.train.assert_not_called()


def test_late_scene_completion_cannot_overwrite_terminal_state_or_consume_labels(scene_job):
    scene_job.db.update_training_job.return_value = False
    consume = Mock()
    scene_job.executor._mark_labeling_entries_trained = consume
    with pytest.raises(RuntimeError, match="state changed before model completion"):
        scene_job.executor._execute_head(scene_job.job)
    assert scene_job.db.update_training_job.call_args.kwargs["expected_status"] == "running"
    consume.assert_not_called()
