"""Controlled rollback uses production evidence and the existing two-phase worker.

Only fake artifact bytes and runtime transactions are used; no engine or media
is loaded. Cryptographic/decoder evidence contracts retain their own real tests.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from argus.storage.model_release_service import (
    ModelReleaseService, RuntimeReleaseCommit, RuntimeReleaseIdentity, RuntimeReleasePreparation,
)
from argus.storage.models import Base, ModelRecord, ModelReleaseOperation, ModelVersionEvent, AuditLog
from argus.storage.release_evidence import (
    ReleaseEvidenceValidation,
    build_release_evidence_from_sources as real_build_release_evidence_from_sources,
)
from argus.storage.release_pipeline import ReleasePipeline, StageTransitionError


def _valid(evidence=None, **_kwargs):
    text = json.dumps(evidence or {"original": "verified"}, sort_keys=True, separators=(",", ":"))
    return ReleaseEvidenceValidation(True, (), text, hashlib.sha256(text.encode()).hexdigest())


class Runtime:
    def __init__(self):
        self.version = "current"
        self.calls = 0
        self.aborts = 0
        self.prepare_error = False
        self.swap_error = False
        self.abort_error = False
        self.before_commit = None

    def prepare(self, plan):
        self.calls += 1
        if self.prepare_error:
            return RuntimeReleasePreparation(False, error="injected load failure")
        previous = self.version

        def commit():
            if self.before_commit:
                self.before_commit()
            self.version = plan.model_version_id
            return RuntimeReleaseCommit(not self.swap_error, error="injected swap failure" if self.swap_error else None)

        def abort():
            self.aborts += 1
            if self.abort_error:
                raise RuntimeError("injected abort failure")
            self.version = previous
            return RuntimeReleaseCommit(True, state={"restored": True})

        return RuntimeReleasePreparation(
            True,
            RuntimeReleaseIdentity(plan.camera_id, plan.model_version_id, plan.target_stage, plan.runtime_artifact_sha256),
            SimpleNamespace(commit=commit, abort=abort, finalize=lambda: RuntimeReleaseCommit(True)),
        )


@pytest.fixture
def rollback(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'rollback.db'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"fake weights: no model engine is used")
    original = _valid()
    sources = {"regression_report_path": "server-owned-original-report"}
    with sessions() as session:
        for version, stage, active, revision in (
            ("target", "retired", False, 4), ("current", "production", True, 3),
        ):
            session.add(ModelRecord(
                model_version_id=version, camera_id="cam-1", model_type="patchcore",
                model_hash="a" * 64, data_hash="b" * 64, model_path=str(artifact),
                stage=stage, is_active=active, release_revision=revision,
                release_evidence_json=original.canonical_json,
                release_evidence_sha256=original.sha256,
            ))
        session.add(ModelReleaseOperation(
            operation_id="historical-production", model_version_id="target", camera_id="cam-1",
            from_stage="canary", target_stage="production", status="applied",
            expected_revision=2, applied_revision=3, requested_by="previous-reviewer",
            requested_at=datetime.now(timezone.utc),
            desired_state_json=json.dumps({"release_evidence_sources": sources}),
        ))
        session.add(ModelVersionEvent(
            camera_id="cam-1", from_version="target", to_version="target",
            from_stage="canary", to_stage="production", triggered_by="previous-reviewer",
        ))
        session.commit()
    builder = Mock(return_value={"rebuilt": "strict v3 + signed soak fixture"})
    monkeypatch.setattr("argus.storage.release_evidence.build_release_evidence_from_sources", builder)
    old_validation = Mock(return_value=original)
    monkeypatch.setattr("argus.storage.model_registry.validate_persisted_release_evidence", old_validation)
    monkeypatch.setattr("argus.storage.release_pipeline.validate_release_evidence", _valid)
    runtime = Runtime()
    pipeline = ReleasePipeline(sessions)
    service = ModelReleaseService(sessions, pipeline, runtime_prepare=runtime.prepare)
    yield SimpleNamespace(
        sessions=sessions, service=service, runtime=runtime, pipeline=pipeline,
        artifact=artifact, builder=builder, old_validation=old_validation, sources=sources,
    )
    engine.dispose()


def submit(state, **overrides):
    args = dict(
        model_version_id="target", source_model_version_id="current", source_release_revision=3,
        expected_revision=4, requested_by="reviewer", reason="restore validated previous version",
        idempotency_key="rollback-once",
    )
    args.update(overrides)
    return state.service.submit_rollback(**args)[0]


def assert_unchanged(state):
    with state.sessions() as session:
        target = session.query(ModelRecord).filter_by(model_version_id="target").one()
        current = session.query(ModelRecord).filter_by(model_version_id="current").one()
        assert (target.stage, target.is_active, target.release_revision) == ("retired", False, 4)
        assert (current.stage, current.is_active) == ("production", True)
    assert state.runtime.version == "current"


def test_rollback_is_durable_then_atomically_applied_and_retry_is_idempotent(rollback):
    operation = submit(rollback)
    assert operation.status == "pending"
    assert rollback.builder.call_count == rollback.runtime.calls == 0
    assert_unchanged(rollback)
    rollback.runtime.before_commit = lambda: assert_unchanged(rollback)
    applied, target = rollback.service.execute_operation(operation.operation_id)
    assert applied.status == "applied"
    assert (target.stage, target.is_active, target.release_revision) == ("production", True, 5)
    assert rollback.runtime.version == "target"
    assert rollback.builder.call_args.args[0] == rollback.sources
    assert rollback.builder.call_args.kwargs["operator_id"] == "reviewer"
    with rollback.sessions() as session:
        current = session.query(ModelRecord).filter_by(model_version_id="current").one()
        assert (current.stage, current.is_active, current.release_revision) == ("retired", False, 4)
        assert session.query(ModelRecord).filter_by(stage="production", is_active=True).count() == 1
        assert session.query(AuditLog).filter_by(action="model_rollback").count() == 1
    assert submit(rollback).operation_id == applied.operation_id
    assert rollback.service.execute_operation(applied.operation_id)[0].status == "applied"
    assert rollback.runtime.calls == 1


@pytest.mark.parametrize("change", ["no_history", "candidate", "active", "canary", "no_sources"])
def test_rollback_rejects_unproven_or_occupied_history(rollback, change):
    with rollback.sessions() as session:
        target = session.query(ModelRecord).filter_by(model_version_id="target").one()
        if change == "no_history":
            session.query(ModelReleaseOperation).delete()
        elif change == "candidate":
            target.stage = "candidate"
        elif change == "active":
            target.is_active = True
        elif change == "canary":
            session.query(ModelRecord).filter_by(model_version_id="current").one().stage = "canary"
        else:
            session.query(ModelReleaseOperation).one().desired_state_json = "{}"
        session.commit()
    with pytest.raises(ValueError):
        submit(rollback)
    assert rollback.runtime.calls == 0


@pytest.mark.parametrize("overrides", [
    {"source_model_version_id": "other-camera-model"}, {"source_release_revision": 9},
    {"expected_revision": 9}, {"reason": " "},
])
def test_rollback_requires_exact_source_and_target_revisions(rollback, overrides):
    with pytest.raises(ValueError):
        submit(rollback, **overrides)
    assert_unchanged(rollback)


@pytest.mark.parametrize("phase", ["load", "swap", "commit", "evidence", "old_evidence"])
def test_failed_rollback_keeps_current_production_and_runtime(rollback, monkeypatch, phase):
    if phase == "load":
        rollback.runtime.prepare_error = True
    elif phase == "swap":
        rollback.runtime.swap_error = True
    elif phase == "commit":
        monkeypatch.setattr(rollback.pipeline, "commit_transition", Mock(side_effect=RuntimeError("commit failed")))
    elif phase == "evidence":
        rollback.builder.side_effect = ValueError("baseline v3 or trusted soak rejected")
    else:
        rollback.old_validation.return_value = ReleaseEvidenceValidation(False, ("old v2 rejected",), None, None)
    operation = submit(rollback)
    failed, _ = rollback.service.execute_operation(operation.operation_id)
    assert failed.status == "failed"
    assert_unchanged(rollback)
    if phase in {"swap", "commit"}:
        assert rollback.runtime.aborts == 1
    assert submit(rollback).operation_id == failed.operation_id


@pytest.mark.parametrize("moment", ["before_prepare", "after_prepare"])
def test_late_rollback_cannot_replace_newer_production_revision(rollback, moment):
    operation = submit(rollback)

    def advance():
        with rollback.sessions() as session:
            session.query(ModelRecord).filter_by(model_version_id="current").update({"release_revision": 7})
            session.commit()

    if moment == "before_prepare":
        advance()
    else:
        rollback.runtime.before_commit = advance
    failed, _ = rollback.service.execute_operation(operation.operation_id)
    assert failed.status == "failed"
    assert_unchanged(rollback)
    assert rollback.runtime.calls == (0 if moment == "before_prepare" else 1)


def test_concurrent_pending_rollbacks_only_apply_once(rollback):
    first = submit(rollback)
    second = submit(rollback, idempotency_key="other-request")
    assert rollback.service.execute_operation(first.operation_id)[0].status == "applied"
    assert rollback.service.execute_operation(second.operation_id)[0].status == "failed"
    assert rollback.runtime.calls == 1


def test_rollback_requires_reauthorization_after_worker_restart(rollback):
    operation = submit(rollback)
    assert rollback.service.reconcile_interrupted_operations() == 1
    assert rollback.service.get_operation(operation.operation_id).status == "failed"
    assert_unchanged(rollback)
    assert rollback.builder.call_count == 0


def test_rollback_readiness_is_read_only_and_rebuilds_original_sources(rollback):
    readiness = rollback.service.rollback_readiness("target")
    assert readiness["source_model_version_id"] == "current"
    assert readiness["source_release_revision"] == 3
    assert readiness["expected_revision"] == 4
    assert rollback.builder.call_args.kwargs["operator_id"] == "previous-reviewer"
    with rollback.sessions() as session:
        assert session.query(ModelReleaseOperation).count() == 1
    assert_unchanged(rollback)


def test_normal_promote_cannot_bypass_rollback_policy(rollback):
    with pytest.raises(StageTransitionError, match="Invalid transition"):
        rollback.pipeline.prepare_transition(model_version_id="target", target_stage="production")


def test_idempotency_rejects_changed_requester_reason_or_source(rollback):
    submit(rollback)
    for changed in ({"requested_by": "another"}, {"reason": "changed"}, {"source_release_revision": 4}):
        with pytest.raises(ValueError, match="another rollback command"):
            submit(rollback, **changed)


def test_changed_target_evidence_after_submit_cannot_be_applied(rollback):
    operation = submit(rollback)
    with rollback.sessions() as session:
        session.query(ModelRecord).filter_by(model_version_id="target").update({"release_evidence_sha256": "c" * 64})
        session.commit()
    assert rollback.service.execute_operation(operation.operation_id)[0].status == "failed"
    assert_unchanged(rollback)
    assert rollback.runtime.calls == 0


def test_changed_historical_sources_after_submit_reject_before_runtime(rollback):
    operation = submit(rollback)
    with rollback.sessions() as session:
        history = session.query(ModelReleaseOperation).filter_by(operation_id="historical-production").one()
        history.desired_state_json = json.dumps({"release_evidence_sources": {"regression_report_path": "changed"}})
        session.commit()
    assert rollback.service.execute_operation(operation.operation_id)[0].status == "failed"
    assert rollback.runtime.calls == 0
    assert_unchanged(rollback)


def test_compensation_failure_is_visible_and_never_marks_rollback_applied(rollback):
    rollback.runtime.swap_error = True
    rollback.runtime.abort_error = True
    operation = submit(rollback)
    failed, _ = rollback.service.execute_operation(operation.operation_id)
    assert failed.status == "failed" and failed.applied_revision is None
    state = json.loads(failed.runtime_state_json)
    assert state["abort"]["success"] is False
    assert "injected abort failure" in state["abort"]["error"]
    with rollback.sessions() as session:
        assert session.query(ModelRecord).filter_by(model_version_id="current").one().is_active
        assert session.query(ModelRecord).filter_by(model_version_id="target").one().stage == "retired"
    assert rollback.service.execute_operation(operation.operation_id)[0].status == "failed"
    assert rollback.runtime.calls == 1


def test_profile_changed_after_readiness_is_rejected_by_real_evidence_builder(
    rollback, tmp_path, release_baseline_factory, release_evidence_sources_factory,
):
    from argus.storage.release_evidence import registry_training_data_hash

    baseline = release_baseline_factory(root=tmp_path / "baseline")
    sources = release_evidence_sources_factory(
        root=tmp_path / "sources", model_path=rollback.artifact,
        model_version_id="target", registry_model_hash="a" * 64, baseline_path=baseline,
    )
    with rollback.sessions() as session:
        session.query(ModelRecord).update({"camera_id": "cam_01", "data_hash": registry_training_data_hash(baseline)})
        session.query(ModelVersionEvent).update({"camera_id": "cam_01"})
        history = session.query(ModelReleaseOperation).one()
        history.camera_id = "cam_01"
        history.desired_state_json = json.dumps({"release_evidence_sources": sources})
        session.commit()
    assert rollback.service.rollback_readiness("target")["expected_revision"] == 4
    operation = submit(rollback)
    # Change only authoritative profile selection after the prior readiness
    # snapshot. No media is decoded and no runtime model engine is loaded.
    (Path(sources["scene_profile_path"]).parent / "active.json").write_text(json.dumps({
        "schema_version": 1, "camera_id": "cam_01", "profile_id": "different-profile",
        "updated_at": "2026-07-14T00:00:00+00:00",
    }), encoding="utf-8")
    rollback.builder.side_effect = real_build_release_evidence_from_sources
    failed, _ = rollback.service.execute_operation(operation.operation_id)
    assert failed.status == "failed"
    assert "not the camera's active profile" in failed.error
    assert rollback.runtime.calls == 0
    assert_unchanged(rollback)


def test_repeated_rollbacks_keep_each_durable_approval_and_persist_matching_evidence(rollback):
    with rollback.sessions() as session:
        session.add(ModelReleaseOperation(
            operation_id="original-current-production", model_version_id="current", camera_id="cam-1",
            from_stage="canary", target_stage="production", status="applied",
            expected_revision=1, applied_revision=2, requested_by="original-current-reviewer",
            requested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            desired_state_json=json.dumps({"release_evidence_sources": rollback.sources}),
        ))
        session.add(ModelVersionEvent(
            camera_id="cam-1", to_version="current", from_stage="canary",
            to_stage="production", triggered_by="original-current-reviewer",
        ))
        session.commit()
    rollback.builder.side_effect = lambda _sources, **kwargs: {
        "operator": kwargs["operator_id"], "approved_at": kwargs["approved_at"],
    }

    def validate_stored(record):
        # No evidence field or digest is ignored during the repeated cycle.
        parsed = json.loads(record.release_evidence_json)
        validation = _valid(parsed)
        assert record.release_evidence_json == validation.canonical_json
        assert record.release_evidence_sha256 == validation.sha256
        return validation

    rollback.old_validation.side_effect = validate_stored
    first = submit(rollback, requested_by="first-reviewer")
    applied, target = rollback.service.execute_operation(first.operation_id)
    assert applied.status == "applied"
    first_evidence = json.loads(target.release_evidence_json)
    assert first_evidence["operator"] == "first-reviewer"

    second, _ = rollback.service.submit_rollback(
        model_version_id="current", source_model_version_id="target", source_release_revision=5,
        expected_revision=4, requested_by="second-reviewer", reason="restore B", idempotency_key="restore-B",
    )
    assert rollback.service.execute_operation(second.operation_id)[0].status == "applied"
    context = rollback.service.rollback_readiness("target")
    assert context["history_operation_id"] == first.operation_id
    assert rollback.builder.call_args.kwargs["operator_id"] == "first-reviewer"
    assert rollback.builder.call_args.kwargs["approved_at"] == first_evidence["approved_at"]
    last = submit(rollback, source_release_revision=5, expected_revision=6,
                  requested_by="third-reviewer", idempotency_key="restore-A-again")
    # Explicitly distinct durable timestamp prevents a same-tick fixture from
    # hiding a replay of the initial production approval.
    with rollback.sessions() as session:
        session.query(ModelReleaseOperation).filter_by(operation_id=last.operation_id).update({
            "requested_at": datetime(2027, 1, 1, tzinfo=timezone.utc),
        })
        session.commit()
    applied, target = rollback.service.execute_operation(last.operation_id)
    assert applied.status == "applied"
    assert json.loads(target.release_evidence_json) == {
        "operator": "third-reviewer", "approved_at": "2027-01-01T00:00:00+00:00",
    }
    assert target.release_evidence_sha256 != hashlib.sha256(json.dumps(first_evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert rollback.runtime.calls == 3
