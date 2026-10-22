"""Crash-window and two-phase semantics for ModelReleaseService."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from argus.storage.model_release_service import (
    ModelReleaseService,
    RuntimeReleaseCommit,
    RuntimeReleaseIdentity,
    RuntimeReleasePreparation,
)
from argus.storage.models import (
    Base,
    ModelRecord,
    ModelReleaseOperation,
    ReleaseOperationStatus,
)
from argus.storage.release_pipeline import ReleasePipeline, StageTransitionError


@pytest.fixture()
def session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _model(session_factory, version: str = "model-v1") -> None:
    with session_factory() as session:
        session.add(ModelRecord(
            model_version_id=version,
            camera_id="cam-01",
            model_type="patchcore",
            model_hash="a" * 64,
            data_hash="b" * 64,
            stage="candidate",
            component_type="full",
        ))
        session.commit()


@dataclass
class _RuntimeTransaction:
    commit_success: bool = True
    commit_error: str | None = None
    before_commit: object = None
    commit_calls: int = 0
    abort_calls: int = 0

    def commit(self) -> RuntimeReleaseCommit:
        self.commit_calls += 1
        if callable(self.before_commit):
            self.before_commit()
        return RuntimeReleaseCommit(
            self.commit_success,
            state={"applied": self.commit_success},
            error=self.commit_error,
        )

    def abort(self) -> RuntimeReleaseCommit:
        self.abort_calls += 1
        return RuntimeReleaseCommit(True, state={"restored": True})

    def finalize(self) -> RuntimeReleaseCommit:
        return RuntimeReleaseCommit(True, state={"released": True})


def _prepared(plan, transaction: _RuntimeTransaction) -> RuntimeReleasePreparation:
    return RuntimeReleasePreparation(
        success=True,
        identity=RuntimeReleaseIdentity(
            camera_id=plan.camera_id,
            model_version_id=plan.model_version_id,
            target_stage=plan.target_stage,
            runtime_artifact_sha256=plan.runtime_artifact_sha256,
        ),
        transaction=transaction,
        state={"standby_loaded": True},
    )


def test_success_keeps_applied_row_old_until_runtime_commit(session_factory):
    _model(session_factory)
    published: list[tuple[str, dict]] = []
    transaction = _RuntimeTransaction()

    def assert_old_applied_state() -> None:
        with session_factory() as session:
            row = session.query(ModelRecord).one()
            operation = session.query(ModelReleaseOperation).one()
            assert row.stage == "candidate"
            assert row.release_revision == 0
            assert operation.status == ReleaseOperationStatus.PENDING.value

    transaction.before_commit = assert_old_applied_state
    service = ModelReleaseService(
        session_factory,
        ReleasePipeline(session_factory, min_shadow_days=0, min_canary_days=0),
        runtime_prepare=lambda plan: _prepared(plan, transaction),
        event_publisher=lambda topic, payload: published.append((topic, payload)),
    )

    operation, record = service.request_transition(
        model_version_id="model-v1",
        target_stage="shadow",
        requested_by="engineer",
        idempotency_key="release-once",
        expected_revision=0,
    )

    assert transaction.commit_calls == 1
    assert transaction.abort_calls == 0
    assert operation.status == ReleaseOperationStatus.APPLIED.value
    assert operation.applied_revision == 1
    assert record.stage == "shadow"
    assert record.release_revision == 1
    assert len(published) == 1


def test_runtime_prepare_failure_leaves_model_and_revision_unchanged(session_factory):
    _model(session_factory)
    service = ModelReleaseService(
        session_factory,
        ReleasePipeline(session_factory, min_shadow_days=0, min_canary_days=0),
        runtime_prepare=lambda _plan: RuntimeReleasePreparation(
            success=False,
            error="standby load failed",
        ),
    )

    operation, record = service.request_transition(
        model_version_id="model-v1",
        target_stage="shadow",
        requested_by="engineer",
    )

    assert operation.status == ReleaseOperationStatus.FAILED.value
    assert "standby load failed" in (operation.error or "")
    assert record.stage == "candidate"
    assert record.is_active is False
    assert record.release_revision == 0


def test_inactive_candidate_retirement_is_audited_without_runtime_prepare(
    session_factory,
):
    _model(session_factory)
    with session_factory() as session:
        model = session.query(ModelRecord).filter_by(model_version_id="model-v1").one()
        model.model_path = "/missing/incomplete-candidate/model.pt"
        session.commit()
    runtime_calls = 0

    def forbidden_runtime_prepare(_plan):
        nonlocal runtime_calls
        runtime_calls += 1
        raise AssertionError("inactive candidate retirement must not touch runtime")

    service = ModelReleaseService(
        session_factory,
        ReleasePipeline(session_factory, min_shadow_days=0, min_canary_days=0),
        runtime_prepare=forbidden_runtime_prepare,
    )

    operation, record = service.request_transition(
        model_version_id="model-v1",
        target_stage="retired",
        requested_by="training_job_executor",
        reason="packaging_failed: disk full",
        idempotency_key="packaging-failed:job-1:model-v1",
    )

    assert runtime_calls == 0
    assert operation.status == ReleaseOperationStatus.APPLIED.value
    assert operation.runtime_state_json is not None
    assert operation.to_dict()["runtime_state"]["runtime_impact"] == "none"
    assert record.stage == "retired"
    assert record.is_active is False
    assert record.release_revision == 1


def test_runtime_commit_failure_aborts_standby_without_db_transition(session_factory):
    _model(session_factory)
    transaction = _RuntimeTransaction(
        commit_success=False,
        commit_error="atomic swap rejected",
    )
    service = ModelReleaseService(
        session_factory,
        ReleasePipeline(session_factory, min_shadow_days=0, min_canary_days=0),
        runtime_prepare=lambda plan: _prepared(plan, transaction),
    )

    operation, record = service.request_transition(
        model_version_id="model-v1",
        target_stage="shadow",
        requested_by="engineer",
    )

    assert transaction.abort_calls == 1
    assert operation.status == ReleaseOperationStatus.FAILED.value
    assert record.stage == "candidate"
    assert record.release_revision == 0


def test_db_commit_failure_rolls_runtime_back_and_preserves_applied_row(
    session_factory,
    monkeypatch,
):
    _model(session_factory)
    pipeline = ReleasePipeline(session_factory, min_shadow_days=0, min_canary_days=0)
    transaction = _RuntimeTransaction()

    def fail_commit(*_args, **_kwargs):
        raise StageTransitionError("injected DB commit failure")

    monkeypatch.setattr(pipeline, "commit_transition", fail_commit)
    service = ModelReleaseService(
        session_factory,
        pipeline,
        runtime_prepare=lambda plan: _prepared(plan, transaction),
    )

    with pytest.raises(StageTransitionError, match="injected DB commit failure"):
        service.request_transition(
            model_version_id="model-v1",
            target_stage="shadow",
            requested_by="engineer",
        )

    assert transaction.abort_calls == 1
    with session_factory() as session:
        row = session.query(ModelRecord).one()
        operation = session.query(ModelReleaseOperation).one()
        assert row.stage == "candidate"
        assert row.release_revision == 0
        assert operation.status == ReleaseOperationStatus.FAILED.value


def test_process_exit_during_prepare_leaves_pending_desired_and_old_applied_state(
    session_factory,
):
    _model(session_factory)

    def crash(_plan):
        raise SystemExit(73)

    service = ModelReleaseService(
        session_factory,
        ReleasePipeline(session_factory, min_shadow_days=0, min_canary_days=0),
        runtime_prepare=crash,
    )

    with pytest.raises(SystemExit, match="73"):
        service.request_transition(
            model_version_id="model-v1",
            target_stage="shadow",
            requested_by="engineer",
        )

    with session_factory() as session:
        row = session.query(ModelRecord).one()
        operation = session.query(ModelReleaseOperation).one()
        assert row.stage == "candidate"
        assert row.release_revision == 0
        assert operation.status == ReleaseOperationStatus.PENDING.value


def test_process_exit_during_runtime_swap_aborts_and_leaves_pending_old_state(
    session_factory,
):
    _model(session_factory)

    class CrashingTransaction(_RuntimeTransaction):
        def commit(self) -> RuntimeReleaseCommit:
            self.commit_calls += 1
            raise SystemExit(74)

    transaction = CrashingTransaction()
    service = ModelReleaseService(
        session_factory,
        ReleasePipeline(session_factory, min_shadow_days=0, min_canary_days=0),
        runtime_prepare=lambda plan: _prepared(plan, transaction),
    )

    with pytest.raises(SystemExit, match="74"):
        service.request_transition(
            model_version_id="model-v1",
            target_stage="shadow",
            requested_by="engineer",
        )

    assert transaction.abort_calls == 1
    with session_factory() as session:
        row = session.query(ModelRecord).one()
        operation = session.query(ModelReleaseOperation).one()
        assert row.stage == "candidate"
        assert row.release_revision == 0
        assert operation.status == ReleaseOperationStatus.PENDING.value


def test_idempotent_retry_resumes_interrupted_pending_operation(session_factory):
    _model(session_factory)
    attempts = 0
    transaction = _RuntimeTransaction()

    def prepare(plan):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise SystemExit(75)
        return _prepared(plan, transaction)

    service = ModelReleaseService(
        session_factory,
        ReleasePipeline(session_factory, min_shadow_days=0, min_canary_days=0),
        runtime_prepare=prepare,
    )
    with pytest.raises(SystemExit, match="75"):
        service.request_transition(
            model_version_id="model-v1",
            target_stage="shadow",
            requested_by="engineer",
            idempotency_key="resume-me",
        )

    operation, record = service.request_transition(
        model_version_id="model-v1",
        target_stage="shadow",
        requested_by="engineer",
        idempotency_key="resume-me",
    )

    assert attempts == 2
    assert operation.status == ReleaseOperationStatus.APPLIED.value
    assert record.stage == "shadow"
    with session_factory() as session:
        assert session.query(ModelReleaseOperation).count() == 1


def test_idempotency_returns_original_operation_without_second_runtime(session_factory):
    _model(session_factory)
    calls = 0
    transaction = _RuntimeTransaction()

    def prepare(plan):
        nonlocal calls
        calls += 1
        return _prepared(plan, transaction)

    service = ModelReleaseService(
        session_factory,
        ReleasePipeline(session_factory, min_shadow_days=0, min_canary_days=0),
        runtime_prepare=prepare,
    )
    first, _ = service.request_transition(
        model_version_id="model-v1",
        target_stage="shadow",
        requested_by="engineer",
        idempotency_key="same-command",
    )
    second, record = service.request_transition(
        model_version_id="model-v1",
        target_stage="shadow",
        requested_by="engineer",
        idempotency_key="same-command",
    )

    assert second.operation_id == first.operation_id
    assert record.stage == "shadow"
    assert calls == 1


def test_stale_revision_is_durable_failed_desired_without_runtime(session_factory):
    _model(session_factory)
    calls = 0

    def prepare(_plan):
        nonlocal calls
        calls += 1
        raise AssertionError("runtime must not be called")

    service = ModelReleaseService(
        session_factory,
        ReleasePipeline(session_factory, min_shadow_days=0, min_canary_days=0),
        runtime_prepare=prepare,
    )

    with pytest.raises(StageTransitionError, match="Stale release revision"):
        service.request_transition(
            model_version_id="model-v1",
            target_stage="shadow",
            requested_by="engineer",
            expected_revision=9,
        )

    assert calls == 0
    with session_factory() as session:
        row = session.query(ModelRecord).one()
        operation = session.query(ModelReleaseOperation).one()
        assert row.stage == "candidate"
        assert row.release_revision == 0
        assert operation.status == ReleaseOperationStatus.FAILED.value


def test_pipeline_cas_rejects_second_prepared_plan(session_factory):
    _model(session_factory)
    pipeline = ReleasePipeline(session_factory, min_shadow_days=0, min_canary_days=0)
    first = pipeline.prepare_transition(
        model_version_id="model-v1",
        target_stage="shadow",
        expected_revision=0,
    )
    second = pipeline.prepare_transition(
        model_version_id="model-v1",
        target_stage="shadow",
        expected_revision=0,
    )

    pipeline.commit_transition(first, triggered_by="engineer-a")
    with pytest.raises(StageTransitionError, match="Stale release plan"):
        pipeline.commit_transition(second, triggered_by="engineer-b")

    with session_factory() as session:
        row = session.query(ModelRecord).one()
        assert row.stage == "shadow"
        assert row.release_revision == 1


def test_lost_commit_response_does_not_restore_runtime_after_db_commit(
    session_factory, monkeypatch,
):
    _model(session_factory)
    pipeline = ReleasePipeline(session_factory, min_shadow_days=0, min_canary_days=0)
    commit = pipeline.commit_transition
    transaction = _RuntimeTransaction()

    def commit_then_lose_response(*args, **kwargs):
        commit(*args, **kwargs)
        raise RuntimeError("database response lost after successful commit")

    monkeypatch.setattr(pipeline, "commit_transition", commit_then_lose_response)
    service = ModelReleaseService(
        session_factory, pipeline,
        runtime_prepare=lambda plan: _prepared(plan, transaction),
    )
    operation, model = service.request_transition(
        model_version_id="model-v1", target_stage="shadow", requested_by="operator",
    )
    assert operation.status == "applied"
    assert model.stage == "shadow"
    assert transaction.abort_calls == 0


def test_invalid_runtime_commit_contract_is_compensated(session_factory):
    _model(session_factory)

    class InvalidTransaction(_RuntimeTransaction):
        def commit(self):
            self.commit_calls += 1
            return None

    transaction = InvalidTransaction()
    service = ModelReleaseService(
        session_factory,
        ReleasePipeline(session_factory, min_shadow_days=0, min_canary_days=0),
        runtime_prepare=lambda plan: _prepared(plan, transaction),
    )
    operation, model = service.request_transition(
        model_version_id="model-v1", target_stage="shadow", requested_by="operator",
    )
    assert operation.status == "failed"
    assert "invalid contract" in operation.error
    assert model.stage == "candidate"
    assert transaction.abort_calls == 1
