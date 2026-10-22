"""Tests for TrainingJobExecutor packaging failure handling.

Regression: if ``ModelPackager.package()`` raises after a CANDIDATE has been
registered, the executor must:

  * retire the just-registered CANDIDATE through ModelReleaseService so we do
    not leave a ghost version pointing at a half-assembled package,
  * retain the release-operation/version-event/audit trail without invoking
    any runtime callback for that inactive candidate,
  * persist the packaging error on the job's ``metrics`` field,
  * re-raise so the outer ``_execute_job`` handler marks the job FAILED
    (instead of silently COMPLETE).

Before the fix this branch only ``logger.error``-ed the exception, silently
completed the job, and left the CANDIDATE in the registry.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from argus.anomaly.job_executor import TrainingJobExecutor
from argus.anomaly.trainer import TrainingResult, TrainingStatus
from argus.storage.database import Database
from argus.storage.model_release_service import ModelReleaseService
from argus.storage.model_registry import (
    ModelRegistry,
    ModelStageMutationDisabledError,
)
from argus.storage.models import (
    AuditLog,
    LabelingQueueRecord,
    ModelReleaseOperation,
    ModelStage,
    ReleaseOperationStatus,
    TrainingJobRecord,
    TrainingJobStatus,
    TrainingJobType,
)
from argus.storage.release_pipeline import ReleasePipeline


@pytest.fixture
def db(tmp_path):
    """Create a fresh SQLite database on disk for each test."""
    db_path = (tmp_path / "argus.db").as_posix()
    database = Database(database_url=f"sqlite:///{db_path}")
    database.initialize()
    return database


@pytest.fixture
def registry(db):
    """Real ModelRegistry wired to the same database session factory."""
    return ModelRegistry(session_factory=db._session_factory)


@pytest.fixture
def model_artifacts(tmp_path):
    """Create a fake trained model directory + baseline directory."""
    model_dir = tmp_path / "model_out"
    model_dir.mkdir()
    (model_dir / "model.xml").write_text("<fake/>")
    (model_dir / "model.bin").write_bytes(b"fake weights")

    baseline_dir = tmp_path / "baselines" / "cam_01" / "default"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "img.png").write_bytes(b"px")

    return model_dir, baseline_dir


@pytest.fixture
def queued_job(db):
    """Persist a queued anomaly_head training job and return it."""
    job = TrainingJobRecord(
        job_id="job-packaging-test",
        job_type=TrainingJobType.ANOMALY_HEAD.value,
        camera_id="cam_01",
        zone_id="default",
        model_type="patchcore",
        trigger_type="manual",
        status=TrainingJobStatus.QUEUED.value,
        hyperparameters=json.dumps({"image_size": 256, "skip_baseline_validation": True}),
    )
    db.save_training_job(job)
    return job


def _make_trainer(model_dir: Path, exports_dir: Path) -> MagicMock:
    """MagicMock trainer that succeeds with a TrainingResult pointing at model_dir."""
    trainer = MagicMock()
    trainer.exports_dir = exports_dir
    trainer.train.return_value = TrainingResult(
        status=TrainingStatus.COMPLETE,
        model_path=str(model_dir),
        duration_seconds=1.0,
        model_version_id=None,  # force registry.register() branch
    )
    return trainer


@pytest.mark.parametrize("model_type, uses_custom", [("patchcore", False), ("dinomaly2", True)])
def test_only_supported_heads_claim_active_backbone(
    db, queued_job, model_artifacts, tmp_path, monkeypatch, model_type, uses_custom,
):
    model_dir, _baseline = model_artifacts
    queued_job.model_type = model_type
    active = SimpleNamespace(checkpoint_path="selected-encoder.pt", backbone_version_id="backbone-selected")
    get_active = MagicMock(return_value=active)
    monkeypatch.setattr(db, "get_active_backbone", get_active)
    trainer = _make_trainer(model_dir, tmp_path / "exports")
    executor = TrainingJobExecutor(
        database=db, trainer=trainer, baselines_dir=tmp_path / "baselines",
        model_packages_dir=tmp_path / "packages",
    )

    db.update_training_job(queued_job.job_id, status=TrainingJobStatus.RUNNING.value)
    executor._execute_head(queued_job)

    assert trainer.train.call_args.kwargs["backbone_checkpoint"] == (
        "selected-encoder.pt" if uses_custom else None
    )
    assert get_active.call_count == int(uses_custom)


def _save_labeled_entries(db: Database, tmp_path: Path, count: int = 5) -> list[int]:
    """Create labeled queue rows that are still consumable by training."""
    entry_ids: list[int] = []
    for i in range(count):
        frame = tmp_path / f"frame_{i}.jpg"
        frame.write_bytes(b"jpg")
        record = db.save_labeling_entry(
            LabelingQueueRecord(
                camera_id="cam_01",
                zone_id="zone_a",
                frame_number=i,
                frame_path=str(frame),
                anomaly_score=0.4,
                entropy=0.8,
            )
        )
        labeled = db.label_entry(record.id, label="anomaly", labeled_by="tester")
        entry_ids.append(labeled.id)
    return entry_ids


def test_dataset_selection_drives_multi_version_training(
    db, registry, model_artifacts, tmp_path,
):
    """痛点 2: when job.dataset_selection is set, executor must merge the
    selected versions and pass baseline_dir_override / image_count_override
    / baseline_versions_label to trainer.train."""
    from argus.anomaly.dataset_selection import (
        DatasetSelection,
        DatasetSelectionItem,
    )

    model_dir, _ = model_artifacts
    baselines_root = tmp_path / "baselines"
    # Two real baseline versions
    for ver in ("v001", "v003"):
        d = baselines_root / "cam_01" / "default" / ver
        d.mkdir(parents=True)
        for i in range(3):
            (d / f"baseline_{i:05d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    selection = DatasetSelection(
        items=[
            DatasetSelectionItem("cam_01", "default", "v001"),
            DatasetSelectionItem("cam_01", "default", "v003"),
        ],
        total_frames=6,
    )

    job = TrainingJobRecord(
        job_id="job-multi-ds",
        job_type=TrainingJobType.ANOMALY_HEAD.value,
        camera_id="cam_01",
        zone_id="default",
        model_type="patchcore",
        trigger_type="manual",
        status=TrainingJobStatus.QUEUED.value,
        hyperparameters=json.dumps({"skip_baseline_validation": True}),
        dataset_selection=selection.to_json(),
    )
    db.save_training_job(job)

    trainer = _make_trainer(model_dir, tmp_path / "exports")
    # BaselineManager wired to our fake baselines_root
    from argus.anomaly.baseline import BaselineManager
    trainer._baseline_manager = BaselineManager(baselines_dir=str(baselines_root))

    executor = TrainingJobExecutor(
        database=db,
        trainer=trainer,
        baselines_dir=baselines_root,
        model_packages_dir=tmp_path / "packages",
    )
    executor.execute(job.job_id)

    # trainer.train must have been called once with override args populated
    call = trainer.train.call_args
    kwargs = call.kwargs
    assert kwargs["baseline_dir_override"] is not None
    assert kwargs["baseline_dir_override"].is_dir() is False, (
        "merger tmp dir should be cleaned up after the with-block"
    )
    assert kwargs["image_count_override"] == 6
    assert kwargs["baseline_versions_label"] == "v001+v003 / 6 frames"


def test_non_default_zone_used_for_training_and_registration(
    db, model_artifacts, tmp_path,
):
    """A queued anomaly_head job must train and hash the requested zone."""
    model_dir, _ = model_artifacts
    baselines_root = tmp_path / "baselines"
    zone_baseline = baselines_root / "cam_01" / "zone_a"
    zone_baseline.mkdir(parents=True)
    (zone_baseline / "img.png").write_bytes(b"px")

    job = TrainingJobRecord(
        job_id="job-zone-a",
        job_type=TrainingJobType.ANOMALY_HEAD.value,
        camera_id="cam_01",
        zone_id="zone_a",
        model_type="patchcore",
        trigger_type="manual",
        status=TrainingJobStatus.QUEUED.value,
        hyperparameters=json.dumps({"skip_baseline_validation": True}),
    )
    db.save_training_job(job)

    registry = MagicMock()
    registry.register.return_value = "cam_01-patchcore-zone-a"
    trainer = _make_trainer(model_dir, tmp_path / "exports")
    executor = TrainingJobExecutor(
        database=db,
        trainer=trainer,
        model_registry=registry,
        baselines_dir=baselines_root,
        model_packages_dir=tmp_path / "packages",
    )

    executor.execute(job.job_id)

    assert trainer.train.call_args.kwargs["zone_id"] == "zone_a"
    assert registry.register.call_args.kwargs["baseline_dir"] == zone_baseline


def test_completed_job_consumes_labeling_entries_with_model_version(
    db, model_artifacts, tmp_path,
):
    """Active-learning labels are consumed only by the completed model ID."""
    model_dir, _ = model_artifacts
    entry_ids = _save_labeled_entries(db, tmp_path)
    job = TrainingJobRecord(
        job_id="job-labels-complete",
        job_type=TrainingJobType.ANOMALY_HEAD.value,
        camera_id="cam_01",
        zone_id="zone_a",
        model_type="patchcore",
        trigger_type="drift_suggested",
        status=TrainingJobStatus.QUEUED.value,
        hyperparameters=json.dumps({
            "skip_baseline_validation": True,
            "labeling_entry_ids": entry_ids,
        }),
    )
    db.save_training_job(job)

    trainer = MagicMock()
    trainer.exports_dir = tmp_path / "exports"
    trainer.train.return_value = TrainingResult(
        status=TrainingStatus.COMPLETE,
        model_path=str(model_dir),
        duration_seconds=1.0,
        model_version_id="model-version-123",
    )
    executor = TrainingJobExecutor(
        database=db,
        trainer=trainer,
        baselines_dir=tmp_path / "baselines",
        model_packages_dir=tmp_path / "packages",
    )

    executor.execute(job.job_id)

    consumed = db.get_labeled_entries(trained_into="model-version-123")
    assert {entry.id for entry in consumed} == set(entry_ids)


def test_failed_job_leaves_labeling_entries_reusable(
    db, model_artifacts, tmp_path,
):
    """Failed training must not consume labels attached to the job."""
    entry_ids = _save_labeled_entries(db, tmp_path)
    job = TrainingJobRecord(
        job_id="job-labels-failed",
        job_type=TrainingJobType.ANOMALY_HEAD.value,
        camera_id="cam_01",
        zone_id="zone_a",
        model_type="patchcore",
        trigger_type="drift_suggested",
        status=TrainingJobStatus.QUEUED.value,
        hyperparameters=json.dumps({
            "skip_baseline_validation": True,
            "labeling_entry_ids": entry_ids,
        }),
    )
    db.save_training_job(job)

    trainer = MagicMock()
    trainer.exports_dir = tmp_path / "exports"
    trainer.train.return_value = TrainingResult(
        status=TrainingStatus.FAILED,
        error="training exploded",
        duration_seconds=1.0,
    )
    executor = TrainingJobExecutor(
        database=db,
        trainer=trainer,
        baselines_dir=tmp_path / "baselines",
        model_packages_dir=tmp_path / "packages",
    )

    executor.execute(job.job_id)

    updated = db.get_training_job(job.job_id)
    assert updated.status == TrainingJobStatus.FAILED.value
    reusable = db.get_labeled_entries(camera_id="cam_01", trained_into="")
    assert {entry.id for entry in reusable} == set(entry_ids)


def test_dataset_selection_camera_mismatch_fails_job(
    db, registry, model_artifacts, tmp_path,
):
    """A dataset_selection that targets a different camera must abort the
    job with a clear error rather than training the wrong baseline."""
    from argus.anomaly.dataset_selection import (
        DatasetSelection,
        DatasetSelectionItem,
    )

    model_dir, _ = model_artifacts

    bad_selection = DatasetSelection(
        items=[DatasetSelectionItem("cam_99", "default", "v001")],
    )
    job = TrainingJobRecord(
        job_id="job-cam-mismatch",
        job_type=TrainingJobType.ANOMALY_HEAD.value,
        camera_id="cam_01",
        zone_id="default",
        model_type="patchcore",
        trigger_type="manual",
        status=TrainingJobStatus.QUEUED.value,
        hyperparameters=json.dumps({"skip_baseline_validation": True}),
        dataset_selection=bad_selection.to_json(),
    )
    db.save_training_job(job)

    trainer = _make_trainer(model_dir, tmp_path / "exports")
    executor = TrainingJobExecutor(
        database=db,
        trainer=trainer,
        baselines_dir=tmp_path / "baselines",
        model_packages_dir=tmp_path / "packages",
    )
    executor.execute(job.job_id)

    updated = db.get_training_job(job.job_id)
    assert updated.status == TrainingJobStatus.FAILED.value
    assert "不一致" in (updated.error or "")
    trainer.train.assert_not_called()


def test_packaging_failure_retires_candidate_through_release_service(
    db, registry, model_artifacts, queued_job, tmp_path,
):
    """Packaging cleanup is audited and never invokes the runtime callback."""
    model_dir, baselines_root = model_artifacts
    # baselines_root is .../baselines/cam_01/default — walk up to the root
    baselines_dir = baselines_root.parent.parent

    packager = MagicMock()
    packager.package.side_effect = RuntimeError("disk full during manifest write")
    runtime_prepare = MagicMock(
        side_effect=AssertionError("failed candidate must not touch runtime")
    )
    release_service = ModelReleaseService(
        db.get_session,
        ReleasePipeline(db.get_session, min_shadow_days=0, min_canary_days=0),
        runtime_prepare=runtime_prepare,
    )

    executor = TrainingJobExecutor(
        database=db,
        trainer=_make_trainer(model_dir, tmp_path / "exports"),
        model_packager=packager,
        model_registry=registry,
        model_release_service=release_service,
        baselines_dir=baselines_dir,
        model_packages_dir=tmp_path / "packages",
    )

    # Execute — the packaging exception re-raised from _execute_head is
    # caught by the outer handler in _execute_job, which marks the job
    # FAILED and logs but does NOT propagate. So execute() returns
    # normally here; we assert on the persisted side-effects below.
    executor.execute(queued_job.job_id)

    # 1. Job must be FAILED (not COMPLETE)
    updated = db.get_training_job(queued_job.job_id)
    assert updated.status == TrainingJobStatus.FAILED.value
    assert updated.error and "disk full" in updated.error

    # 2. metrics must include the packaging_error
    assert updated.metrics is not None
    metrics = json.loads(updated.metrics)
    assert "packaging_error" in metrics
    assert "disk full" in metrics["packaging_error"]
    assert metrics["candidate_retirement_error"] is None
    assert metrics["candidate_retirement"]["status"] == (
        ReleaseOperationStatus.APPLIED.value
    )
    assert metrics["candidate_retirement"]["runtime_state"]["runtime_impact"] == (
        "none"
    )

    # 3. The CANDIDATE that was registered must now be RETIRED
    all_models = registry.list_models(camera_id="cam_01")
    assert len(all_models) == 1, "exactly one model should have been registered"
    rec = all_models[0]
    assert rec.stage == ModelStage.RETIRED.value
    assert rec.is_active is False

    # 4. A version event must capture the retirement with the reason.
    events = registry.get_version_events(model_version_id=rec.model_version_id)
    retire_events = [e for e in events if e.to_stage == ModelStage.RETIRED.value]
    assert retire_events, "expected a retirement event on the candidate"
    assert "packaging_failed" in (retire_events[0].reason or "")

    # 5. The release command and compliance audit are durable; the callback
    # that could manipulate a camera runtime was never called.
    runtime_prepare.assert_not_called()
    with db.get_session() as session:
        operation = session.query(ModelReleaseOperation).one()
        assert operation.status == ReleaseOperationStatus.APPLIED.value
        assert operation.requested_by == "training_job_executor"
        assert operation.from_stage == ModelStage.CANDIDATE.value
        assert operation.target_stage == ModelStage.RETIRED.value
        audits = session.query(AuditLog).filter_by(
            action="model_stage_transition",
            target_id=rec.model_version_id,
        ).all()
        assert len(audits) == 1


def test_packaging_success_leaves_candidate_as_is(
    db, registry, model_artifacts, queued_job, tmp_path,
):
    """Baseline: when packaging succeeds, CANDIDATE stays CANDIDATE and job completes."""
    model_dir, baselines_root = model_artifacts
    baselines_dir = baselines_root.parent.parent

    pkg_out = tmp_path / "pkg_out"
    pkg_out.mkdir()
    packager = MagicMock()
    packager.package.return_value = pkg_out  # success path returns a path
    runtime_prepare = MagicMock()
    release_service = ModelReleaseService(
        db.get_session,
        ReleasePipeline(db.get_session, min_shadow_days=0, min_canary_days=0),
        runtime_prepare=runtime_prepare,
    )

    executor = TrainingJobExecutor(
        database=db,
        trainer=_make_trainer(model_dir, tmp_path / "exports"),
        model_packager=packager,
        model_registry=registry,
        model_release_service=release_service,
        baselines_dir=baselines_dir,
        model_packages_dir=tmp_path / "packages",
    )

    executor.execute(queued_job.job_id)

    updated = db.get_training_job(queued_job.job_id)
    assert updated.status == TrainingJobStatus.COMPLETE.value
    assert updated.artifacts_path == str(pkg_out)

    all_models = registry.list_models(camera_id="cam_01")
    assert len(all_models) == 1
    assert all_models[0].stage == ModelStage.CANDIDATE.value
    runtime_prepare.assert_not_called()
    with db.get_session() as session:
        assert session.query(ModelReleaseOperation).count() == 0


def test_packager_with_registry_requires_release_service(
    db, registry, model_artifacts, tmp_path,
):
    model_dir, _baseline_dir = model_artifacts

    with pytest.raises(ValueError, match="model_release_service is required"):
        TrainingJobExecutor(
            database=db,
            trainer=_make_trainer(model_dir, tmp_path / "exports"),
            model_packager=MagicMock(),
            model_registry=registry,
            model_packages_dir=tmp_path / "packages",
        )


def test_registry_retire_cannot_bypass_release_service(registry, model_artifacts):
    model_dir, baseline_dir = model_artifacts
    version_id = registry.register(
        model_path=model_dir,
        baseline_dir=baseline_dir,
        camera_id="cam_01",
        model_type="patchcore",
    )
    with pytest.raises(ModelStageMutationDisabledError, match="ModelReleaseService"):
        registry.retire(version_id, reason="packaging_failed: boom")

    unchanged = registry.get_by_version_id(version_id)
    assert unchanged.stage == ModelStage.CANDIDATE.value
    assert unchanged.is_active is False
    events = registry.get_version_events(model_version_id=version_id)
    assert len(events) == 1
    assert events[0].to_stage == ModelStage.CANDIDATE.value


def test_job_packages_the_selected_run_and_registers_its_exact_cuda_file(db, queued_job, tmp_path):
    from argus.anomaly.artifacts import new_artifact_run, write_artifact_manifest

    run = new_artifact_run(tmp_path / "models", tmp_path / "exports", "cam_01", "default")
    runtime_dir = run.export_dir / "weights" / "torch"
    runtime_dir.mkdir(parents=True)
    artifact = runtime_dir / "model.pt"
    artifact.write_bytes(b"runtime")
    (run.training_dir / "best.ckpt").write_bytes(b"checkpoint")
    calibration = run.training_dir / "calibration.json"
    calibration.write_text('{"sorted_scores": [0.1]}', encoding="utf-8")
    (runtime_dir / "calibration.json").write_bytes(calibration.read_bytes())
    write_artifact_manifest(run, artifact, model_type="patchcore", calibration_status="ok")
    trainer = MagicMock()
    trainer.train.return_value = TrainingResult(
        status=TrainingStatus.COMPLETE, model_path=str(artifact),
        training_dir=str(run.training_dir), export_dir=str(run.export_dir), run_id=run.run_id,
        actual_export_format="torch", actual_quantization="fp32", calibration_status="ok",
    )
    registry = MagicMock()
    registry.register.return_value = "candidate-selected-run"
    packager = MagicMock()
    packager.package.return_value = tmp_path / "completed-package"
    executor = TrainingJobExecutor(
        database=db, trainer=trainer, model_registry=registry, model_packager=packager,
        model_release_service=MagicMock(), model_packages_dir=tmp_path / "packages",
    )

    executor.execute(queued_job.job_id)

    assert db.get_training_job(queued_job.job_id).status == TrainingJobStatus.COMPLETE.value
    assert registry.register.call_args.kwargs["model_path"] == artifact
    assert registry.register.call_args.kwargs["training_params"]["export_format"] == "torch"
    assert registry.register.call_args.kwargs["training_params"]["quantization"] == "fp32"
    assert packager.package.call_args.kwargs["model_dir"] == run.training_dir
    assert packager.package.call_args.kwargs["export_dir"] == run.export_dir
    assert packager.package.call_args.kwargs["calibration_path"] == calibration


def test_queued_cuda_training_defaults_to_torch_fp32(db, queued_job, tmp_path, monkeypatch):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"runtime")
    trainer = MagicMock()
    trainer.train.return_value = TrainingResult(status=TrainingStatus.COMPLETE, model_path=str(artifact))
    camera_manager = MagicMock()
    camera_manager.get_camera_config.return_value = SimpleNamespace(
        anomaly=SimpleNamespace(inference_backend="torch-cuda"),
    )
    executor = TrainingJobExecutor(
        database=db, trainer=trainer, camera_manager=camera_manager,
        model_packages_dir=tmp_path / "packages",
    )
    monkeypatch.setattr(executor, "_training_mode_guard", lambda **_kwargs: nullcontext())

    executor.execute(queued_job.job_id)

    assert db.get_training_job(queued_job.job_id).status == TrainingJobStatus.COMPLETE.value
    assert trainer.train.call_args.kwargs["export_format"] == "torch"
    assert trainer.train.call_args.kwargs["quantization"] == "fp32"


@pytest.mark.parametrize("params", [{"export_format": "openvino"}, {"quantization": "fp16"}])
def test_queued_cuda_training_rejects_explicit_incompatible_export(
    db, queued_job, tmp_path, params,
):
    db.update_training_job(queued_job.job_id, hyperparameters=json.dumps(params))
    trainer = MagicMock()
    camera_manager = MagicMock()
    camera_manager.get_camera_config.return_value = SimpleNamespace(
        anomaly=SimpleNamespace(inference_backend="torch-cuda"),
    )
    executor = TrainingJobExecutor(
        database=db, trainer=trainer, camera_manager=camera_manager,
        model_packages_dir=tmp_path / "packages",
    )

    executor.execute(queued_job.job_id)

    updated = db.get_training_job(queued_job.job_id)
    assert updated.status == TrainingJobStatus.FAILED.value
    assert "torch-cuda" in updated.error
    trainer.train.assert_not_called()


def test_job_executor_binds_real_trainer_registry_before_dataset_lifecycle(db, tmp_path):
    from argus.anomaly.trainer import ModelTrainer

    trainer = ModelTrainer(MagicMock(), tmp_path / "models", tmp_path / "exports")
    registry = MagicMock()
    TrainingJobExecutor(
        database=db, trainer=trainer, model_registry=registry,
        model_packages_dir=tmp_path / "packages",
    )
    assert trainer._model_registry is registry
