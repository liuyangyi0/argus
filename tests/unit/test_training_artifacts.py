"""Regression guards for immutable runs and precise deployment registration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from argus.anomaly.artifacts import (
    artifact_metadata,
    exported_artifact,
    new_artifact_run,
    training_directory,
    write_artifact_manifest,
)
from argus.anomaly.trainer import ModelTrainer, TrainingStatus


def test_new_runs_do_not_reuse_previous_training_or_export_dirs(tmp_path):
    first = new_artifact_run(tmp_path / "models", tmp_path / "exports", "cam", "zone")
    (first.export_dir / "model.pt").write_bytes(b"first model")
    second = new_artifact_run(tmp_path / "models", tmp_path / "exports", "cam", "zone")

    assert first.run_id != second.run_id
    assert first.training_dir != second.training_dir
    assert first.export_dir != second.export_dir
    assert (first.export_dir / "model.pt").read_bytes() == b"first model"
    assert list(second.export_dir.iterdir()) == []


@pytest.mark.parametrize("component", ["../other", "..", "camera/zone", "camera\\zone"])
def test_run_rejects_path_components_that_escape_camera_directory(tmp_path, component):
    with pytest.raises(ValueError, match="camera_id"):
        new_artifact_run(tmp_path / "models", tmp_path / "exports", component, "zone")


def test_logical_group_camera_id_has_a_portable_encoded_directory(tmp_path):
    run = new_artifact_run(tmp_path / "models", tmp_path / "exports", "group:line-one", "zone")
    artifact = run.export_dir / "model.pt"
    artifact.write_bytes(b"runtime")
    write_artifact_manifest(run, artifact, model_type="patchcore", calibration_status="skipped")

    assert run.export_dir.parent.parent.name == "group%3Aline-one"
    assert artifact_metadata(artifact)["camera_id"] == "group:line-one"
    assert training_directory(artifact) == run.training_dir


def test_find_best_model_accepts_an_exact_registered_file(tmp_path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"runtime")
    assert ModelTrainer._find_best_model_file(artifact) == artifact


def test_group_training_counts_resolved_group_baseline_not_synthetic_camera_path(mocked_trainer, tmp_path):
    trainer, registry = mocked_trainer
    group_baseline = tmp_path / "baselines" / "_groups" / "line" / "zone" / "v001"
    group_baseline.mkdir(parents=True)
    for index in range(60):
        (group_baseline / f"baseline_{index:05d}.png").write_bytes(b"image")
    manager = trainer._baseline_manager
    manager.get_group_baseline_dir.return_value = group_baseline
    manager.count_images.return_value = 0

    result = trainer.train(camera_id="group:line", zone_id="zone", group_id="line", export_format="torch")

    assert result.status == TrainingStatus.COMPLETE
    assert result.image_count == 60
    manager.count_images.assert_not_called()
    assert registry.register.call_args.kwargs["baseline_dir"] == group_baseline


def test_export_resolution_uses_actual_format_after_failed_export(tmp_path):
    (tmp_path / "model.xml").write_bytes(b"partial failed OpenVINO export")
    torch_dir = tmp_path / "weights" / "torch"
    torch_dir.mkdir(parents=True)
    artifact = torch_dir / "model.pt"
    artifact.write_bytes(b"successful Torch export")

    assert exported_artifact(tmp_path, "torch") == artifact.resolve()


def test_export_resolution_rejects_missing_weights_and_ambiguous_versions(tmp_path):
    (tmp_path / "model.xml").write_bytes(b"xml")
    with pytest.raises(ValueError, match="weights missing"):
        exported_artifact(tmp_path, "openvino")
    for version in ("first", "second"):
        directory = tmp_path / version
        directory.mkdir()
        (directory / "model.pt").write_bytes(version.encode())
    with pytest.raises(ValueError, match="found 2"):
        exported_artifact(tmp_path, "torch")


def test_manifest_binds_runtime_sidecars_and_connects_exact_file_to_checkpoint(tmp_path):
    run = new_artifact_run(tmp_path / "models", tmp_path / "exports", "cam", "zone")
    runtime_dir = run.export_dir / "weights" / "torch"
    runtime_dir.mkdir(parents=True)
    artifact = runtime_dir / "model.pt"
    artifact.write_bytes(b"model")
    calibration = runtime_dir / "calibration.json"
    calibration.write_text('{"sorted_scores": [0.1, 0.2]}', encoding="utf-8")

    manifest = write_artifact_manifest(
        run, artifact, model_type="patchcore", calibration_status="ok",
    )

    assert manifest["files"]["weights/torch/model.pt"] == hashlib.sha256(b"model").hexdigest()
    assert manifest["files"]["weights/torch/calibration.json"] == hashlib.sha256(
        calibration.read_bytes(),
    ).hexdigest()
    assert artifact_metadata(artifact)["artifact_path"] == str(artifact.resolve())
    assert training_directory(artifact) == run.training_dir
    assert artifact_metadata(run.training_dir)["run_id"] == run.run_id
    with pytest.raises(FileExistsError):
        write_artifact_manifest(run, artifact, model_type="patchcore", calibration_status="ok")


def test_manifest_cannot_redirect_an_exact_file_to_a_different_model(tmp_path):
    run = new_artifact_run(tmp_path / "models", tmp_path / "exports", "cam", "zone")
    artifact = run.export_dir / "model.pt"
    artifact.write_bytes(b"selected")
    write_artifact_manifest(run, artifact, model_type="patchcore", calibration_status="skipped")
    other = run.export_dir / "other.pt"
    other.write_bytes(b"different model")

    with pytest.raises(ValueError, match="selected runtime file"):
        artifact_metadata(other)


def test_manifest_rejects_artifact_outside_its_declared_export_run(tmp_path):
    run = new_artifact_run(tmp_path / "models", tmp_path / "exports", "cam", "zone")
    artifact = run.export_dir / "model.pt"
    artifact.write_bytes(b"selected")
    metadata = write_artifact_manifest(
        run, artifact, model_type="patchcore", calibration_status="skipped",
    )
    metadata["artifact_path"] = str(tmp_path / "unrelated.pt")
    (run.export_dir / "artifact.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes"):
        artifact_metadata(artifact)


@pytest.fixture
def mocked_trainer(tmp_path, monkeypatch):
    baseline = tmp_path / "baselines" / "cam" / "zone"
    baseline.mkdir(parents=True)
    manager = MagicMock()
    manager._lifecycle = None
    manager.get_baseline_dir.return_value = baseline
    manager.count_images.return_value = 100
    registry = MagicMock()
    registry.register.side_effect = ["candidate-one", "candidate-two"]
    trainer = ModelTrainer(
        manager, tmp_path / "models", tmp_path / "exports", model_registry=registry,
    )
    monkeypatch.setattr(trainer, "_validate_baseline_quality", lambda *a, **k: {
        "passed": True, "corruption_rate": 0.0, "near_duplicate_rate": 0.0,
        "brightness_std": 20.0,
    })

    def split(_baseline, output_dir, **_kwargs):
        train_dir = output_dir / "_split" / "train"
        val_dir = output_dir / "_split" / "val"
        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)
        return train_dir, val_dir, 40, 60

    def train(**kwargs):
        (kwargs["output_dir"] / "model.ckpt").write_bytes(b"checkpoint")
        return MagicMock(), MagicMock()

    def export(**kwargs):
        artifact = Path(kwargs["export_path"]) / "weights" / "torch" / "model.pt"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"runtime model")
        return {"actual_format": "torch", "actual_quantization": "fp32", "artifact_path": str(artifact)}

    monkeypatch.setattr(trainer, "_split_train_val", split)
    monkeypatch.setattr(trainer, "_train_anomalib", train)
    monkeypatch.setattr(trainer, "_export_model", export)
    monkeypatch.setattr(trainer, "_validate_output", lambda *a, **k: (
        {"checkpoint_valid": True, "smoke_test_passed": True}, MagicMock(),
    ))
    monkeypatch.setattr(trainer, "_validate_on_val_set", lambda *a, **k: {
        "scores": list(np.linspace(0.01, 0.3, 60)), "mean": 0.15, "std": 0.1, "p95": 0.28,
    })
    monkeypatch.setattr(trainer, "_recommend_threshold", lambda *_a: 0.5)
    monkeypatch.setattr(trainer, "_generate_quality_report", lambda *a: SimpleNamespace(
        grade="A", suggestions=[],
    ))
    monkeypatch.setattr(trainer, "_restore_gpu_after_training", lambda: None)
    return trainer, registry


def test_training_registers_exact_export_after_calibration_and_manifest(mocked_trainer):
    trainer, registry = mocked_trainer

    def register(**kwargs):
        path = Path(kwargs["model_path"])
        assert path.is_file() and path.suffix == ".pt"
        assert (path.parent / "calibration.json").is_file()
        manifest = artifact_metadata(path)
        assert manifest["calibration_status"] == "ok"
        assert kwargs["training_params"]["artifact_manifest"] == manifest
        return "candidate-one"

    registry.register.side_effect = register
    result = trainer.train("cam", "zone", export_format="torch", quantization="fp32")

    assert result.status == TrainingStatus.COMPLETE
    assert result.model_version_id == "candidate-one"
    assert Path(result.model_path).is_file()
    assert Path(result.training_dir).name == result.run_id
    assert Path(result.export_dir).name == result.run_id


def test_repeat_training_preserves_old_export_calibration_and_checkpoint(mocked_trainer):
    trainer, _registry = mocked_trainer
    first = trainer.train("cam", "zone", export_format="torch", quantization="fp32")
    old_files = {
        path: path.read_bytes()
        for root in (Path(first.training_dir), Path(first.export_dir))
        for path in root.rglob("*") if path.is_file()
    }

    second = trainer.train("cam", "zone", export_format="torch", quantization="fp32")

    assert first.run_id != second.run_id
    assert first.model_path != second.model_path
    assert first.model_version_id != second.model_version_id
    assert all(path.read_bytes() == content for path, content in old_files.items())


def test_failed_calibration_never_registers_partial_candidate(mocked_trainer, monkeypatch):
    trainer, registry = mocked_trainer
    monkeypatch.setattr(
        "argus.alerts.calibration.ConformalCalibrator.save",
        MagicMock(side_effect=OSError("disk full")),
    )

    result = trainer.train("cam", "zone", export_format="torch", quantization="fp32")

    assert result.status == TrainingStatus.FAILED
    registry.register.assert_not_called()


def test_failed_registry_write_does_not_report_training_complete(mocked_trainer):
    trainer, registry = mocked_trainer
    registry.register.side_effect = RuntimeError("database unavailable")

    result = trainer.train("cam", "zone", export_format="torch", quantization="fp32")

    assert result.status == TrainingStatus.FAILED
    assert "database unavailable" in result.error


def test_reexport_from_exact_artifact_creates_a_new_candidate(mocked_trainer, monkeypatch):
    trainer, registry = mocked_trainer
    source = trainer.train("cam", "zone", export_format="torch", quantization="fp32")
    source_bytes = Path(source.model_path).read_bytes()
    engine = MagicMock()

    def export(**kwargs):
        artifact = Path(kwargs["export_root"]) / "model.pt"
        artifact.write_bytes(b"new runtime model")
        return artifact

    engine.export.side_effect = export
    monkeypatch.setattr("anomalib.engine.Engine", MagicMock(return_value=engine))
    monkeypatch.setattr(trainer, "_instantiate_model", lambda *_a: MagicMock())

    derived = trainer.reexport_model(Path(source.model_path), export_format="torch", quantization="fp32")

    assert derived["status"] == "ok"
    assert derived["model_version_id"] == "candidate-two"
    assert derived["model_path"] != source.model_path
    assert Path(source.model_path).read_bytes() == source_bytes
    assert registry.register.call_args.kwargs["training_params"]["derived_operation"] == "reexport"
    assert Path(engine.export.call_args.kwargs["ckpt_path"]).parent != Path(source.training_dir)
    # New exports must not inherit score thresholds from a differently
    # normalized model; absent enough baseline samples this is explicit.
    assert artifact_metadata(Path(derived["model_path"]))["calibration_status"] == "skipped"


def test_recalibration_copies_runtime_bundle_and_never_edits_source(mocked_trainer, monkeypatch):
    trainer, registry = mocked_trainer
    source = trainer.train("cam", "zone", export_format="torch", quantization="fp32")
    source_calibration = Path(source.model_path).parent / "calibration.json"
    source_bytes = source_calibration.read_bytes()
    unrelated = Path(source.model_path).parent / "unrelated.pt"
    unrelated.write_bytes(b"must not be copied")
    baseline = trainer._baseline_manager.get_baseline_dir()
    for index in range(50):
        (baseline / f"frame_{index}.png").write_bytes(b"mock image")
    detector = MagicMock()
    detector.load.return_value = True
    detector.predict.return_value = SimpleNamespace(anomaly_score=0.2, detection_failed=False)
    detector._minmax_broken = False
    detector_factory = MagicMock(return_value=detector)
    monkeypatch.setattr("argus.anomaly.detector.AnomalibDetector", detector_factory)
    monkeypatch.setattr("argus.anomaly.trainer.cv2.imread", lambda *_a: np.zeros((4, 4, 3)))

    derived = trainer.recalibrate_model(Path(source.model_path), baseline, "cam", "zone")

    assert derived["status"] == "ok"
    assert derived["model_version_id"] == "candidate-two"
    assert detector_factory.call_args.kwargs["enable_calibration"] is False
    assert source_calibration.read_bytes() == source_bytes
    assert Path(derived["model_path"]).read_bytes() == Path(source.model_path).read_bytes()
    assert not (Path(derived["export_path"]) / "unrelated.pt").exists()
    assert json.loads((Path(derived["export_path"]) / "calibration.json").read_text())["n_samples"] == 50
    assert registry.register.call_args.kwargs["training_params"]["derived_operation"] == "recalibrate"


def test_registry_binding_does_not_replace_an_already_bound_registry(mocked_trainer):
    trainer, registry = mocked_trainer
    trainer.bind_model_registry(registry)
    with pytest.raises(ValueError, match="already bound"):
        trainer.bind_model_registry(MagicMock())


def test_skipped_run_does_not_load_or_fingerprint_ancestor_calibration(tmp_path):
    from argus.anomaly.detector import AnomalibDetector
    from argus.preprocessing.scene_profile import runtime_artifact_sha256

    run = new_artifact_run(tmp_path / "models", tmp_path / "exports", "cam", "zone")
    artifact = run.export_dir / "weights" / "torch" / "model.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"runtime")
    old_calibration = run.export_dir.parent / "calibration.json"
    old_calibration.write_text('{"sorted_scores": [0.1, 0.2]}', encoding="utf-8")
    write_artifact_manifest(run, artifact, model_type="patchcore", calibration_status="skipped")
    detector = AnomalibDetector(model_path=artifact)
    detector._load_calibration()
    initial_hash = runtime_artifact_sha256(artifact)

    old_calibration.write_text('{"sorted_scores": [0.8, 0.9]}', encoding="utf-8")
    detector._load_calibration()

    assert not detector.is_calibrated
    assert runtime_artifact_sha256(artifact) == initial_hash
    assert runtime_artifact_sha256(run.export_dir) == initial_hash
    assert runtime_artifact_sha256(run.training_dir) == initial_hash
    assert runtime_artifact_sha256(artifact.parent) == initial_hash

    local_calibration = artifact.parent / "calibration.json"
    local_calibration.write_text('{"sorted_scores": [0.3, 0.4]}', encoding="utf-8")
    detector._load_calibration()
    assert detector.is_calibrated
    local_hash = runtime_artifact_sha256(artifact)
    assert local_hash != initial_hash
    local_calibration.write_text('{"sorted_scores": [0.5, 0.6]}', encoding="utf-8")
    assert runtime_artifact_sha256(artifact) != local_hash


def test_runtime_identity_binds_the_adopted_manifest_boundary(tmp_path):
    from argus.preprocessing.scene_profile import runtime_artifact_sha256

    run = new_artifact_run(tmp_path / "models", tmp_path / "exports", "cam", "zone")
    artifact = run.export_dir / "weights" / "torch" / "model.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"runtime")
    metadata = write_artifact_manifest(run, artifact, model_type="patchcore", calibration_status="skipped")
    first = runtime_artifact_sha256(artifact)
    marker = artifact.parent / "artifact.json"
    metadata["boundary_revision"] = 2
    marker.write_text(json.dumps(metadata), encoding="utf-8")
    assert runtime_artifact_sha256(artifact) != first
    changed = runtime_artifact_sha256(artifact)
    marker.unlink()
    assert runtime_artifact_sha256(artifact) != changed


def test_legacy_calibration_lookup_remains_compatible_without_a_manifest(tmp_path):
    from argus.anomaly.detector import AnomalibDetector
    from argus.preprocessing.scene_profile import runtime_artifact_sha256

    artifact = tmp_path / "legacy" / "weights" / "model.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"runtime")
    calibration = artifact.parent.parent / "calibration.json"
    calibration.write_text('{"sorted_scores": [0.1, 0.2]}', encoding="utf-8")
    first = runtime_artifact_sha256(artifact)
    detector = AnomalibDetector(model_path=artifact)
    detector._load_calibration()
    assert detector.is_calibrated
    calibration.write_text('{"sorted_scores": [0.7, 0.8]}', encoding="utf-8")
    assert runtime_artifact_sha256(artifact) != first


def test_legacy_checkpoint_search_does_not_select_a_new_run(tmp_path):
    legacy = tmp_path / "models" / "cam" / "zone"
    legacy.mkdir(parents=True)
    old_checkpoint = legacy / "model.ckpt"
    old_checkpoint.write_bytes(b"legacy checkpoint")
    newer = legacy / "run-20261021-new" / "weights" / "model.ckpt"
    newer.parent.mkdir(parents=True)
    newer.write_bytes(b"unrelated newer model")

    assert ModelTrainer._find_checkpoint(legacy, legacy_only=True) == old_checkpoint
    (legacy / "ambiguous.ckpt").write_bytes(b"another old checkpoint")
    with pytest.raises(ValueError, match="精确checkpoint"):
        ModelTrainer._find_checkpoint(legacy, legacy_only=True)
    assert ModelTrainer._find_checkpoint(old_checkpoint, legacy_only=True) == old_checkpoint


def test_nonempty_export_with_failed_smoke_is_not_registered(mocked_trainer, monkeypatch):
    trainer, registry = mocked_trainer
    monkeypatch.setattr(trainer, "_validate_output", lambda *a, **k: (
        {"checkpoint_valid": True, "smoke_test_passed": False}, None,
    ))
    result = trainer.train("cam", "zone", export_format="torch", quantization="fp32")
    assert result.status == TrainingStatus.FAILED
    registry.register.assert_not_called()


@pytest.mark.parametrize("scores", [[], [float("nan")], [float("inf")]])
def test_export_requires_finite_validation_scores_before_registration(
    mocked_trainer, monkeypatch, scores,
):
    trainer, registry = mocked_trainer
    monkeypatch.setattr(trainer, "_validate_on_val_set", lambda *a, **k: {"scores": scores})
    result = trainer.train("cam", "zone", export_format="torch", quantization="fp32")
    assert result.status == TrainingStatus.FAILED
    registry.register.assert_not_called()


def test_detection_failed_prediction_cannot_pass_output_smoke(mocked_trainer, tmp_path, monkeypatch):
    trainer, _registry = mocked_trainer
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"nonempty runtime")
    validation = tmp_path / "validation"
    validation.mkdir()
    (validation / "frame.png").write_bytes(b"mock frame")
    detector = MagicMock()
    detector.load.return_value = True
    detector.predict.return_value = SimpleNamespace(anomaly_score=0.0, detection_failed=True)
    monkeypatch.setattr("argus.anomaly.detector.AnomalibDetector", MagicMock(return_value=detector))
    monkeypatch.setattr("argus.anomaly.trainer.cv2.imread", lambda *_a: np.zeros((4, 4, 3)))

    validation_result, loaded = ModelTrainer._validate_output(
        trainer, tmp_path, tmp_path, validation, artifact_path=artifact,
    )

    assert validation_result["checkpoint_valid"] is True
    assert validation_result["smoke_test_passed"] is False
    assert loaded is None


def test_mandatory_calibration_refuses_insufficient_scores_before_registration(mocked_trainer, monkeypatch):
    trainer, registry = mocked_trainer
    monkeypatch.setattr(trainer, "_validate_on_val_set", lambda *a, **k: {
        "scores": list(np.linspace(0.01, 0.3, 49)), "mean": 0.15, "std": 0.1, "p95": 0.28,
    })

    result = trainer.train(
        "cam", "zone", export_format="torch", quantization="fp32", require_calibration=True,
    )

    assert result.status == TrainingStatus.FAILED
    assert result.calibration_status == "skipped"
    assert "校准" in result.error
    registry.register.assert_not_called()


def test_mandatory_calibration_refuses_missing_detector_before_registration(mocked_trainer, monkeypatch):
    trainer, registry = mocked_trainer
    monkeypatch.setattr(trainer, "_validate_output", lambda *a, **k: (
        {"checkpoint_valid": True, "smoke_test_passed": True}, None,
    ))

    result = trainer.train(
        "cam", "zone", export_format="torch", quantization="fp32", require_calibration=True,
    )

    assert result.status == TrainingStatus.FAILED
    assert result.calibration_status == "skipped"
    registry.register.assert_not_called()


def test_scene_training_rechecks_baseline_after_fit_before_registration(mocked_trainer, monkeypatch):
    from argus.anomaly.scene_training import SceneTrainingInput

    trainer, registry = mocked_trainer
    baseline = trainer._baseline_manager.get_baseline_dir.return_value
    verified = SceneTrainingInput(
        baselines_dir=baseline.parents[1], camera_id="cam", root=baseline.parent,
        images_dir=baseline, image_count=160, training_count=100, calibration_count=60,
        lineage={"workflow_id": "workflow-01"},
    )
    revalidate = MagicMock(side_effect=[verified, ValueError("immutable baseline changed")])
    monkeypatch.setattr("argus.anomaly.trainer.revalidate_scene_training_input", revalidate)

    result = trainer.train(
        "cam", "default", export_format="torch", quantization="fp32",
        require_calibration=True, scene_training_input=verified,
        baseline_dir_override=baseline, image_count_override=160,
    )

    assert result.status == TrainingStatus.FAILED
    assert "训练期间发生变化" in result.error
    assert revalidate.call_count == 2
    registry.register.assert_not_called()


def test_scene_training_registration_retains_exact_workflow_lineage(mocked_trainer, monkeypatch):
    from argus.anomaly.scene_training import SceneTrainingInput

    trainer, registry = mocked_trainer
    baseline = trainer._baseline_manager.get_baseline_dir.return_value
    lineage = {
        "schema_version": "argus.scene-training-input.v1", "workflow_id": "workflow-01",
        "scene_profile_id": "workflow-01", "baseline_sha256": "a" * 64,
        "manifest_sha256": "b" * 64, "reference_sha256": "c" * 64,
    }
    verified = SceneTrainingInput(
        baselines_dir=baseline.parents[1], camera_id="cam", root=baseline.parent,
        images_dir=baseline, image_count=160, training_count=100, calibration_count=60,
        lineage=lineage,
    )
    revalidate = MagicMock(return_value=verified)
    monkeypatch.setattr("argus.anomaly.trainer.revalidate_scene_training_input", revalidate)

    result = trainer.train(
        "cam", "default", export_format="torch", quantization="fp32",
        require_calibration=True, scene_training_input=verified,
        baseline_dir_override=baseline, image_count_override=160,
    )

    assert result.status == TrainingStatus.COMPLETE
    assert result.calibration_status == "ok"
    assert revalidate.call_count == 2
    assert registry.register.call_args.kwargs["training_params"]["scene_workflow"] == lineage
    registry.activate.assert_not_called()


def _scene_calibration_validation_case(mocked_trainer, monkeypatch, calibration_count, *, unreadable=None):
    from argus.anomaly.scene_training import SceneTrainingInput

    trainer, registry = mocked_trainer
    baseline = trainer._baseline_manager.get_baseline_dir.return_value
    verified = SceneTrainingInput(
        baselines_dir=baseline.parents[1], camera_id="cam", root=baseline.parent,
        images_dir=baseline, image_count=100 + calibration_count,
        training_count=100, calibration_count=calibration_count,
        lineage={"workflow_id": "workflow-01"},
    )
    monkeypatch.setattr("argus.anomaly.trainer.revalidate_scene_training_input", lambda _value: verified)

    def split(_baseline, output_dir, **_kwargs):
        train_dir = output_dir / "_split" / "train" / "normal"
        val_dir = output_dir / "_split" / "val" / "normal"
        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)
        for index in range(calibration_count):
            (val_dir / f"calibration_{index:05d}.png").write_bytes(b"mock frozen image")
        return train_dir, val_dir.parent, 100, calibration_count

    detector = MagicMock()
    detector.predict.side_effect = [
        SimpleNamespace(anomaly_score=0.01 + index * 0.001, detection_failed=False)
        for index in range(calibration_count)
    ]

    def read_image(path):
        if unreadable is not None and Path(path).name == f"calibration_{unreadable:05d}.png":
            return None
        return np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(trainer, "_split_train_val", split)
    monkeypatch.setattr(trainer, "_validate_output", lambda *a, **k: (
        {"checkpoint_valid": True, "smoke_test_passed": True}, detector,
    ))
    monkeypatch.setattr("argus.anomaly.trainer.cv2.imread", read_image)
    # Exercise the actual all-images collector, not the fixture's score stub.
    monkeypatch.setattr(
        trainer, "_validate_on_val_set", ModelTrainer._validate_on_val_set.__get__(trainer),
    )
    return trainer, registry, verified, detector


def test_scene_calibration_cannot_hide_one_unreadable_image_behind_fifty_remaining_scores(
    mocked_trainer, monkeypatch,
):
    trainer, registry, verified, detector = _scene_calibration_validation_case(
        mocked_trainer, monkeypatch, 51, unreadable=10,
    )
    result = trainer.train(
        "cam", "default", export_format="torch", quantization="fp32",
        require_calibration=True, scene_training_input=verified,
        baseline_dir_override=verified.images_dir, image_count_override=verified.image_count,
    )

    assert detector.predict.call_count == 50
    assert result.status == TrainingStatus.FAILED and result.calibration_status == "failed"
    assert "完整覆盖冻结校准集" in result.error
    registry.register.assert_not_called()


def test_scene_calibration_evaluates_every_frozen_image_above_legacy_export_budget(
    mocked_trainer, monkeypatch,
):
    trainer, registry, verified, detector = _scene_calibration_validation_case(
        mocked_trainer, monkeypatch, 120,
    )
    result = trainer.train(
        "cam", "default", export_format="torch", quantization="fp32",
        calibration_images=100, require_calibration=True, scene_training_input=verified,
        baseline_dir_override=verified.images_dir, image_count_override=verified.image_count,
    )

    assert detector.predict.call_count == 120
    assert result.status == TrainingStatus.COMPLETE and result.calibration_status == "ok"
    assert result.val_count == 120
    registration = registry.register.call_args.kwargs
    calibration_path = Path(registration["model_path"]).parent / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    assert calibration["n_samples"] == 120 and len(calibration["sorted_scores"]) == 120
