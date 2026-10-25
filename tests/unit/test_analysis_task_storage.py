"""SQLite/mock-only coverage for additive analysis identities and projection."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event, func, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from argus.core.analysis_task_contracts import LEGACY_TEMPLATE_ID, legacy_task_id
from argus.storage.analysis_tasks import (
    AnalysisTaskIntegrityError,
    AnalysisTaskProjectionConflict,
    AnalysisTaskService,
)
from argus.storage.database import (
    _ANALYSIS_TASK_INDEX_MIGRATIONS,
    _ANALYSIS_TASK_TRIGGER_MIGRATIONS,
    Database,
)
from argus.storage.models import (
    AnalysisTaskRecord,
    AnalysisTaskRevisionRecord,
    Base,
    ModelArtifactRecord,
    ModelRecord,
    TaskDeploymentRecord,
)


@pytest.fixture
def db(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'tasks.db'}")
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def service(db):
    return AnalysisTaskService(db)


def _count(db, model):
    with db.get_session() as session:
        return session.scalar(select(func.count()).select_from(model))


def _model(db, *, camera_id="cam-1", version="version-1", **overrides):
    values = {
        "camera_id": camera_id,
        "model_version_id": version,
        "model_type": "patchcore",
        "model_hash": "a" * 16,
        "data_hash": "b" * 64,
        "model_path": "not-opened-by-these-mock-tests.pt",
        "stage": "production",
        "is_active": True,
        "release_revision": 3,
        "release_evidence_json": json.dumps({
            "artifacts": {"runtime_artifact_sha256": "c" * 64},
        }),
    }
    values.update(overrides)
    with db.get_session() as session:
        session.add(ModelRecord(**values))
        session.commit()


def _mock_approved(monkeypatch):
    # Never load a model or claim this fixture is real production evidence.
    monkeypatch.setattr(
        "argus.storage.model_registry.validate_persisted_release_evidence",
        lambda _record: SimpleNamespace(valid=True),
    )


def test_stable_identity_preserves_camera_identity():
    assert legacy_task_id("cam-1") == legacy_task_id("cam-1")
    assert legacy_task_id("Cam-1") != legacy_task_id("cam-1")
    assert legacy_task_id("摄像头一") != legacy_task_id("摄像头二")
    assert len(legacy_task_id("a" * 50)) <= 64
    assert len(legacy_task_id("a" * 200)) <= 64  # Existing source IDs had no length cap.


@pytest.mark.parametrize("camera_id", [None, "", 42])
def test_invalid_camera_identity_rejected(camera_id):
    with pytest.raises(ValueError):
        legacy_task_id(camera_id)


def test_bootstrap_is_idempotent_and_never_overwrites_config(service, db):
    original = {"sample_fps": 5, "scene_guard": {"enabled": True}}
    first = service.bootstrap_legacy("cam-1", original)
    original["sample_fps"] = 99
    again = service.bootstrap_legacy("cam-1", {"sample_fps": 25})
    assert again == first
    assert again["config"]["sample_fps"] == 5
    assert again["template_id"] == LEGACY_TEMPLATE_ID
    assert again["authority"] == "legacy"
    assert again["current_revision"] == 1
    assert again["deployment"]["runtime_status"] == "unknown"
    assert again["deployment"]["applied_artifact_id"] is None
    assert _count(db, AnalysisTaskRecord) == 1
    assert _count(db, AnalysisTaskRevisionRecord) == 1
    assert _count(db, TaskDeploymentRecord) == 1


def test_get_and_list_do_not_bootstrap_or_write(service, db):
    assert service.get_task("missing") is None
    assert service.resolve_legacy("cam-1") is None
    assert service.sync_legacy_model_projection("cam-1") is None
    assert service.list_tasks() == []
    assert _count(db, AnalysisTaskRecord) == 0
    service.bootstrap_legacy("cam-2", {})
    task = service.bootstrap_legacy("cam-1", {})
    assert service.get_task(task["task_id"]) == task
    assert [item["camera_id"] for item in service.list_tasks()] == ["cam-1", "cam-2"]
    assert service.list_tasks("cam-1") == [task]
    assert service.list_tasks("") == []


@pytest.mark.parametrize("config", [[], {"value": float("nan")}, {"value": object()}])
def test_invalid_config_does_not_leave_partial_identity(service, db, config):
    with pytest.raises((ValueError, TypeError)):
        service.bootstrap_legacy("cam-1", config)
    assert _count(db, AnalysisTaskRecord) == 0
    assert _count(db, AnalysisTaskRevisionRecord) == 0
    assert _count(db, TaskDeploymentRecord) == 0


def test_concurrent_bootstrap_uses_database_lock_and_one_initial_revision(db):
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            lambda value: AnalysisTaskService(db).bootstrap_legacy("cam-1", {"value": value}),
            range(8),
        ))
    assert all(result == results[0] for result in results)
    assert _count(db, AnalysisTaskRecord) == 1
    assert _count(db, AnalysisTaskRevisionRecord) == 1
    assert _count(db, TaskDeploymentRecord) == 1


def test_bootstrap_commit_failure_rolls_back_and_retry_is_safe(service, db, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(Session, "commit", lambda _self: (_ for _ in ()).throw(RuntimeError("disk")))
        with pytest.raises(RuntimeError, match="disk"):
            service.bootstrap_legacy("cam-1", {"value": 1})
    assert _count(db, AnalysisTaskRecord) == 0
    assert _count(db, AnalysisTaskRevisionRecord) == 0
    assert _count(db, TaskDeploymentRecord) == 0
    result = service.bootstrap_legacy("cam-1", {"value": 2})
    assert result["config"] == {"value": 2}


def test_bootstrap_unknown_commit_outcome_retry_preserves_first_config(service, db, monkeypatch):
    commit = Session.commit

    def commit_then_fail(session):
        commit(session)
        raise RuntimeError("acknowledgement lost")

    with monkeypatch.context() as patch:
        patch.setattr(Session, "commit", commit_then_fail)
        with pytest.raises(RuntimeError, match="acknowledgement lost"):
            service.bootstrap_legacy("cam-1", {"value": 1})
    result = service.bootstrap_legacy("cam-1", {"value": 2})
    assert result["config"] == {"value": 1}
    assert _count(db, AnalysisTaskRevisionRecord) == 1


def test_unbound_and_invalid_production_do_not_authorize_artifacts(service, db):
    service.bootstrap_legacy("cam-1", {})
    unbound = service.sync_legacy_model_projection("cam-1")
    assert unbound["deployment"]["projection_status"] == "unbound"
    _model(db, release_evidence_json=None)
    blocked = service.sync_legacy_model_projection("cam-1")
    assert blocked["deployment"]["projection_status"] == "blocked"
    assert blocked["deployment"]["registry_artifact"] is None
    assert blocked["deployment"]["desired_artifact_id"] is None
    assert _count(db, ModelArtifactRecord) == 0


@pytest.mark.parametrize("stage", ["candidate", "shadow", "canary", "retired"])
def test_nonproduction_active_flag_does_not_authorize_projection(service, db, stage):
    service.bootstrap_legacy("cam-1", {})
    _model(db, stage=stage)
    result = service.sync_legacy_model_projection("cam-1")
    assert result["deployment"]["projection_status"] == "unbound"
    assert _count(db, ModelArtifactRecord) == 0


def test_projection_separates_artifact_and_deployment_and_never_claims_applied(
    service, db, monkeypatch,
):
    _mock_approved(monkeypatch)
    service.bootstrap_legacy("cam-1", {})
    _model(db)
    with db.get_session() as session:
        original = session.scalar(select(ModelRecord)).to_dict()
    result = service.sync_legacy_model_projection("cam-1")
    projection = result["deployment"]
    artifact = projection["registry_artifact"]
    assert projection["authority"] == "legacy"
    assert projection["projection_status"] == "authorized"
    assert projection["registry_release_revision"] == 3
    assert projection["registry_model_version_id"] == "version-1"
    assert projection["registry_stage"] == "production"
    assert projection["desired_artifact_id"] == artifact["artifact_id"]
    assert projection["applied_artifact_id"] is None
    assert projection["runtime_status"] == "unknown"
    assert artifact["registry_model_hash"] == "a" * 16
    assert artifact["artifact_sha256"] == "c" * 64
    assert artifact["scope_kind"] == "camera"
    assert artifact["scope_id"] == "cam-1"
    assert "stage" not in artifact and "is_active" not in artifact
    assert "model_path" not in artifact and "metadata_json" not in artifact
    assert service.sync_legacy_model_projection("cam-1")["deployment"]["registry_artifact"] == artifact
    assert _count(db, ModelArtifactRecord) == 1
    with db.get_session() as session:
        assert session.scalar(select(ModelRecord)).to_dict() == original


def test_retired_registry_projection_is_cleared_but_artifact_history_is_retained(
    service, db, monkeypatch,
):
    _mock_approved(monkeypatch)
    service.bootstrap_legacy("cam-1", {})
    _model(db)
    service.sync_legacy_model_projection("cam-1")
    with db.get_session() as session:
        model = session.scalar(select(ModelRecord))
        model.is_active = False
        model.stage = "retired"
        session.commit()
    result = service.sync_legacy_model_projection("cam-1")
    assert result["deployment"]["projection_status"] == "unbound"
    assert result["deployment"]["registry_artifact_id"] is None
    assert result["deployment"]["registry_model_version_id"] is None
    assert _count(db, ModelArtifactRecord) == 1


def test_projection_rejects_registry_change_during_validation(service, db, monkeypatch):
    service.bootstrap_legacy("cam-1", {})
    _model(db)

    def approve_then_change(_record):
        with db.get_session() as session:
            session.scalar(select(ModelRecord)).release_revision += 1
            session.commit()
        return SimpleNamespace(valid=True)

    monkeypatch.setattr(
        "argus.storage.model_registry.validate_persisted_release_evidence", approve_then_change,
    )
    with pytest.raises(AnalysisTaskProjectionConflict):
        service.sync_legacy_model_projection("cam-1")
    assert _count(db, ModelArtifactRecord) == 0
    assert service.resolve_legacy("cam-1")["deployment"]["projection_status"] == "not_synced"


def test_projection_commit_failure_has_no_partial_artifact_and_can_retry(
    service, db, monkeypatch,
):
    _mock_approved(monkeypatch)
    service.bootstrap_legacy("cam-1", {})
    _model(db)
    with monkeypatch.context() as patch:
        patch.setattr(Session, "commit", lambda _self: (_ for _ in ()).throw(RuntimeError("disk")))
        with pytest.raises(RuntimeError, match="disk"):
            service.sync_legacy_model_projection("cam-1")
    assert _count(db, ModelArtifactRecord) == 0
    assert service.resolve_legacy("cam-1")["deployment"]["projection_status"] == "not_synced"
    assert service.sync_legacy_model_projection("cam-1")["deployment"]["projection_status"] == "authorized"
    assert _count(db, ModelArtifactRecord) == 1


def test_projection_unknown_commit_outcome_retry_reuses_artifact(service, db, monkeypatch):
    _mock_approved(monkeypatch)
    service.bootstrap_legacy("cam-1", {})
    _model(db)
    commit = Session.commit

    def commit_then_fail(session):
        commit(session)
        raise RuntimeError("acknowledgement lost")

    with monkeypatch.context() as patch:
        patch.setattr(Session, "commit", commit_then_fail)
        with pytest.raises(RuntimeError, match="acknowledgement lost"):
            service.sync_legacy_model_projection("cam-1")
    before = service.resolve_legacy("cam-1")["deployment"]["registry_artifact_id"]
    after = service.sync_legacy_model_projection("cam-1")["deployment"]["registry_artifact_id"]
    assert before == after
    assert _count(db, ModelArtifactRecord) == 1


def test_concurrent_projections_reuse_the_same_immutable_artifact(service, db, monkeypatch):
    _mock_approved(monkeypatch)
    service.bootstrap_legacy("cam-1", {})
    _model(db)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            lambda _value: AnalysisTaskService(db).sync_legacy_model_projection("cam-1"),
            range(8),
        ))
    identities = {result["deployment"]["registry_artifact_id"] for result in results}
    assert len(identities) == 1
    assert all(result["deployment"]["applied_artifact_id"] is None for result in results)
    assert _count(db, ModelArtifactRecord) == 1


@pytest.mark.parametrize("table", ["analysis_task_revisions", "model_artifacts"])
@pytest.mark.parametrize("operation", ["UPDATE", "DELETE"])
def test_revisions_and_artifacts_are_immutable_in_sql(
    service, db, monkeypatch, table, operation,
):
    _mock_approved(monkeypatch)
    service.bootstrap_legacy("cam-1", {})
    _model(db)
    service.sync_legacy_model_projection("cam-1")
    statement = (
        f"DELETE FROM {table}" if operation == "DELETE" else
        f"UPDATE {table} SET {'revision = 2' if table == 'analysis_task_revisions' else 'scope_id = 1'}"
    )
    with pytest.raises(IntegrityError, match="immutable analysis identity"):
        with db._engine.begin() as conn:
            conn.execute(text(statement))


def test_schema_and_guard_migrations_are_idempotent_and_preserve_legacy(db):
    _model(db)
    db._auto_migrate()
    db._auto_migrate_indexes()
    db._auto_migrate_analysis_task_guards()
    db._auto_migrate_indexes()
    db._auto_migrate_analysis_task_guards()
    indexes = {
        item["name"]
        for table in inspect(db._engine).get_table_names()
        for item in inspect(db._engine).get_indexes(table)
    }
    assert {name for name, _ in _ANALYSIS_TASK_INDEX_MIGRATIONS} <= indexes
    with db._engine.connect() as conn:
        triggers = set(conn.execute(text("SELECT name FROM sqlite_master WHERE type='trigger'")).scalars())
    assert {name for name, _ in _ANALYSIS_TASK_TRIGGER_MIGRATIONS} <= triggers
    with db.get_session() as session:
        model = session.scalar(select(ModelRecord))
        assert model.model_version_id == "version-1"
        assert model.stage == "production" and model.release_revision == 3


def test_old_database_gains_new_tables_without_rewriting_legacy_records(tmp_path):
    url = f"sqlite:///{tmp_path / 'old.db'}"
    engine = create_engine(url)
    new_tables = {
        "analysis_tasks", "analysis_task_revisions", "model_artifacts", "task_deployments",
    }
    Base.metadata.create_all(engine, tables=[
        table for table in Base.metadata.sorted_tables if table.name not in new_tables
    ])
    with Session(engine) as session:
        session.add(ModelRecord(
            camera_id="old-camera", model_version_id="old-version", model_type="patchcore",
            model_hash="old-hash", data_hash="old-data", stage="candidate", is_active=False,
        ))
        session.commit()
        before = session.scalar(select(ModelRecord)).to_dict()
    engine.dispose()
    db = Database(url)
    try:
        db.initialize()
        assert new_tables <= set(inspect(db._engine).get_table_names())
        assert AnalysisTaskService(db).list_tasks() == []
        with db.get_session() as session:
            assert session.scalar(select(ModelRecord)).to_dict() == before
        AnalysisTaskService(db).bootstrap_legacy("old-camera", {"original": True})
        db.initialize()
        assert AnalysisTaskService(db).bootstrap_legacy("old-camera", {"original": False})[
            "config"
        ] == {"original": True}
    finally:
        db.close()


def test_duplicate_legacy_camera_identity_is_rejected(service, db):
    service.bootstrap_legacy("cam-1", {})
    with pytest.raises(IntegrityError):
        with db.get_session() as session:
            session.add(AnalysisTaskRecord(
                task_id="different", camera_id="cam-1", template_id=LEGACY_TEMPLATE_ID,
                authority="legacy", name="duplicate", current_revision=1,
            ))
            session.commit()


def test_corrupted_task_identity_does_not_silently_bootstrap_again(service, db):
    task = service.bootstrap_legacy("cam-1", {})
    with db.get_session() as session:
        session.get(AnalysisTaskRecord, task["task_id"]).authority = "other"
        session.commit()
    with pytest.raises(AnalysisTaskIntegrityError):
        service.bootstrap_legacy("cam-1", {"new": True})
    assert _count(db, AnalysisTaskRevisionRecord) == 1


def test_model_artifact_can_be_referenced_by_multiple_deployment_rows(
    service, db, monkeypatch,
):
    _mock_approved(monkeypatch)
    service.bootstrap_legacy("cam-1", {})
    _model(db)
    artifact_id = service.sync_legacy_model_projection("cam-1")["deployment"]["registry_artifact_id"]
    # Schema can represent reuse. P0-P2 deliberately supplies no public API
    # that creates this generic task or authorizes its deployment/run.
    with db.get_session() as session:
        session.add(TaskDeploymentRecord(
            task_id="future-same-camera-task", authority="disabled",
            desired_artifact_id=artifact_id,
            projection_status="not_synced", runtime_status="unknown",
        ))
        session.commit()
    assert _count(db, ModelArtifactRecord) == 1
    assert _count(db, TaskDeploymentRecord) == 2


def test_task_config_corruption_is_rejected_on_read(service, db):
    task = service.bootstrap_legacy("cam-1", {})
    with db.get_session() as session:
        session.add(AnalysisTaskRevisionRecord(
            task_id=task["task_id"], revision=2, config_json="{}", config_sha256="bad",
        ))
        session.get(AnalysisTaskRecord, task["task_id"]).current_revision = 2
        session.commit()
    with pytest.raises(AnalysisTaskIntegrityError):
        service.get_task(task["task_id"])


@pytest.mark.parametrize(
    "change",
    ["version", "release_revision", "model_hash", "inactive", "retired", "missing", "missing_revision"],
)
def test_registry_changes_mark_projection_stale_without_database_writes(
    service, db, monkeypatch, change,
):
    _mock_approved(monkeypatch)
    service.bootstrap_legacy("cam-1", {})
    _model(db)
    initial = service.sync_legacy_model_projection("cam-1")
    task_id = initial["task_id"]
    with db.get_session() as session:
        model = session.scalar(select(ModelRecord))
        if change == "version":
            model.model_version_id = "replacement-version"
        elif change == "release_revision":
            model.release_revision += 1
        elif change == "model_hash":
            model.model_hash = "f" * 16
        elif change == "inactive":
            model.is_active = False
        elif change == "retired":
            model.stage = "retired"
        elif change == "missing":
            session.delete(model)
        else:
            session.get(TaskDeploymentRecord, task_id).registry_release_revision = None
        session.commit()
    statements = []

    def observe(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement.strip().split()[0].upper())

    event.listen(db._engine, "before_cursor_execute", observe)
    try:
        by_id = service.get_task(task_id)
        by_camera = service.resolve_legacy("cam-1")
        listed = service.list_tasks("cam-1")
    finally:
        event.remove(db._engine, "before_cursor_execute", observe)
    assert by_id["deployment"]["projection_status"] == "stale"
    assert by_camera["deployment"]["projection_status"] == "stale"
    assert listed[0]["deployment"]["projection_status"] == "stale"
    assert by_id["deployment"]["applied_artifact_id"] is None
    assert by_id["deployment"]["runtime_status"] == "unknown"
    assert set(statements) == {"SELECT"}
    with db.get_session() as session:
        persisted = session.get(TaskDeploymentRecord, task_id)
        assert persisted.projection_status == "authorized"  # GET did not rewrite history.
        assert persisted.registry_model_version_id == "version-1"
        assert persisted.observed_at.isoformat() == initial["deployment"]["observed_at"].replace(
            "+00:00", "",
        )


def test_unchanged_registry_projection_remains_authorized_without_database_writes(
    service, db, monkeypatch,
):
    _mock_approved(monkeypatch)
    service.bootstrap_legacy("cam-1", {})
    _model(db)
    initial = service.sync_legacy_model_projection("cam-1")
    statements = []

    def observe(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement.strip().split()[0].upper())

    event.listen(db._engine, "before_cursor_execute", observe)
    try:
        current = service.get_task(initial["task_id"])
    finally:
        event.remove(db._engine, "before_cursor_execute", observe)
    assert current["deployment"]["projection_status"] == "authorized"
    assert current["deployment"]["registry_release_revision"] == 3
    assert current["deployment"]["registry_model_version_id"] == "version-1"
    assert current["deployment"]["registry_artifact"]["registry_model_hash"] == "a" * 16
    assert current["deployment"]["applied_artifact_id"] is None
    assert current["deployment"]["runtime_status"] == "unknown"
    assert set(statements) == {"SELECT"}
