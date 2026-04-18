"""Tests for TrainingJobExecutor packaging failure handling.

Regression: if ``ModelPackager.package()`` raises after a CANDIDATE has been
registered, the executor must:

  * retire the just-registered CANDIDATE so we do not leave a ghost version
    pointing at a half-assembled (or missing) package,
  * persist the packaging error on the job's ``metrics`` field,
  * re-raise so the outer ``_execute_job`` handler marks the job FAILED
    (instead of silently COMPLETE).

Before the fix this branch only ``logger.error``-ed the exception, silently
completed the job, and left the CANDIDATE in the registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from argus.anomaly.job_executor import TrainingJobExecutor
from argus.anomaly.trainer import TrainingResult, TrainingStatus
from argus.storage.database import Database
from argus.storage.model_registry import ModelRegistry
from argus.storage.models import (
    ModelStage,
    TrainingJobRecord,
    TrainingJobStatus,
    TrainingJobType,
)


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


def test_packaging_failure_retires_candidate_and_marks_job_failed(
    db, registry, model_artifacts, queued_job, tmp_path,
):
    """Packaging exception must retire CANDIDATE + re-raise + mark job FAILED."""
    model_dir, baselines_root = model_artifacts
    # baselines_root is .../baselines/cam_01/default — walk up to the root
    baselines_dir = baselines_root.parent.parent

    packager = MagicMock()
    packager.package.side_effect = RuntimeError("disk full during manifest write")

    executor = TrainingJobExecutor(
        database=db,
        trainer=_make_trainer(model_dir, tmp_path / "exports"),
        model_packager=packager,
        model_registry=registry,
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

    executor = TrainingJobExecutor(
        database=db,
        trainer=_make_trainer(model_dir, tmp_path / "exports"),
        model_packager=packager,
        model_registry=registry,
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


def test_registry_retire_records_reason_and_event(registry, model_artifacts):
    """Unit-level check that registry.retire() records a version event with reason."""
    model_dir, baseline_dir = model_artifacts

    vid = registry.register(
        model_path=model_dir,
        baseline_dir=baseline_dir,
        camera_id="cam_01",
        model_type="patchcore",
    )
    rec = registry.retire(vid, reason="packaging_failed: boom")
    assert rec is not None
    assert rec.stage == ModelStage.RETIRED.value
    assert rec.is_active is False

    events = registry.get_version_events(model_version_id=vid)
    retire_events = [e for e in events if e.to_stage == ModelStage.RETIRED.value]
    assert len(retire_events) == 1
    evt = retire_events[0]
    assert evt.from_stage == ModelStage.CANDIDATE.value
    assert evt.reason == "packaging_failed: boom"
    assert evt.triggered_by == "system"


def test_registry_retire_idempotent(registry, model_artifacts):
    """Retiring an already-retired model is a no-op (no duplicate event, no error)."""
    model_dir, baseline_dir = model_artifacts
    vid = registry.register(
        model_path=model_dir,
        baseline_dir=baseline_dir,
        camera_id="cam_01",
        model_type="patchcore",
    )

    registry.retire(vid, reason="first")
    # Second call should be idempotent and NOT add another event.
    registry.retire(vid, reason="second — should be ignored")

    events = registry.get_version_events(model_version_id=vid)
    retire_events = [e for e in events if e.to_stage == ModelStage.RETIRED.value]
    assert len(retire_events) == 1
    assert retire_events[0].reason == "first"


def test_registry_retire_missing_returns_none(registry):
    """Retiring a non-existent version returns None (safe for executor's guard)."""
    assert registry.retire("no-such-model-id", reason="x") is None
