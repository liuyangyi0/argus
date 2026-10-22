"""Deleting a registry row cannot delete a different release's artifact."""

import json

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from argus.storage.model_registry import ModelDeletionConflictError, ModelRegistry
from argus.storage.models import AuditLog, Base, ModelRecord, ModelReleaseOperation


@pytest.fixture
def registry():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return ModelRegistry(sessionmaker(bind=engine))


def add_model(registry, version, path, *, stage="candidate", active=False, revision=0):
    with registry._session_factory() as session:
        session.add(ModelRecord(
            model_version_id=version,
            camera_id="cam1",
            model_type="patchcore",
            model_hash="a" * 64,
            data_hash="b" * 64,
            model_path=str(path),
            stage=stage,
            is_active=active,
            release_revision=revision,
        ))
        session.commit()


def test_deleting_legacy_parent_keeps_nested_production_artifact(registry, tmp_path):
    parent = tmp_path / "legacy"
    parent.mkdir()
    artifact = parent / "model.pt"
    artifact.write_bytes(b"active-model")
    sidecar = parent / "calibration.json"
    sidecar.write_text('{"threshold":0.5}', encoding="utf-8")
    add_model(registry, "old", parent, stage="retired")
    add_model(registry, "live", artifact, stage="production", active=True)

    result = registry.delete_model("old", requested_by="operator")

    assert result == {
        "deleted": "old",
        "artifacts_retained": True,
        "garbage_collection": "deferred",
    }
    assert artifact.read_bytes() == b"active-model"
    assert sidecar.read_text(encoding="utf-8") == '{"threshold":0.5}'
    assert registry.get_by_version_id("old") is None
    assert registry.get_by_version_id("live").is_active is True
    with registry._session_factory() as session:
        audit = session.query(AuditLog).filter_by(target_id="old").one()
        assert audit.user == "operator"
        assert audit.action == "model_artifacts_deferred"
        assert json.loads(audit.detail)["model_path"] == str(parent)


@pytest.mark.parametrize("stage,active", [
    ("shadow", False),
    ("canary", False),
    ("production", False),
    ("production", True),
    ("candidate", True),
    ("retired", True),
])
def test_registry_itself_rejects_in_use_models(registry, tmp_path, stage, active):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"do-not-delete")
    add_model(registry, "protected", artifact, stage=stage, active=active)

    with pytest.raises(ModelDeletionConflictError):
        registry.delete_model("protected")

    assert registry.get_by_version_id("protected") is not None
    assert artifact.read_bytes() == b"do-not-delete"


def test_pending_release_prevents_deleting_candidate(registry, tmp_path):
    add_model(registry, "candidate", tmp_path / "model.pt")
    with registry._session_factory() as session:
        session.add(ModelReleaseOperation(
            operation_id="operation-1",
            model_version_id="candidate",
            camera_id="cam1",
            from_stage="candidate",
            target_stage="shadow",
            expected_revision=0,
            requested_by="operator",
            status="pending",
        ))
        session.commit()

    with pytest.raises(ModelDeletionConflictError, match="pending"):
        registry.delete_model("candidate")
    assert registry.get_by_version_id("candidate") is not None


def test_stale_revision_cannot_delete_model(registry, tmp_path):
    add_model(registry, "candidate", tmp_path / "model.pt", revision=5)
    with pytest.raises(ModelDeletionConflictError, match="revision"):
        registry.delete_model("candidate", expected_revision=4)
    assert registry.get_by_version_id("candidate").release_revision == 5


def test_pending_arriving_after_preflight_is_rechecked_by_delete(registry, tmp_path):
    add_model(registry, "candidate", tmp_path / "model.pt")
    engine = registry._session_factory.kw["bind"]

    def insert_pending_before_delete(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("DELETE FROM MODELS"):
            cursor.execute(
                "INSERT INTO model_release_operations "
                "(operation_id, model_version_id, camera_id, from_stage, "
                "target_stage, expected_revision, status, requested_by) "
                "VALUES ('late-operation', 'candidate', 'cam1', 'candidate', "
                "'shadow', 0, 'pending', 'operator')"
            )

    event.listen(engine, "before_cursor_execute", insert_pending_before_delete)
    try:
        with pytest.raises(ModelDeletionConflictError, match="changed"):
            registry.delete_model("candidate")
    finally:
        event.remove(engine, "before_cursor_execute", insert_pending_before_delete)
    assert registry.get_by_version_id("candidate") is not None


def test_failed_audit_commit_rolls_back_registry_deletion(registry, tmp_path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"retained")
    add_model(registry, "candidate", artifact)

    def reject_tombstone(session):
        if any(
            isinstance(item, AuditLog) and item.action == "model_artifacts_deferred"
            for item in session.new
        ):
            raise RuntimeError("injected database failure")

    session_class = registry._session_factory.class_
    event.listen(session_class, "before_commit", reject_tombstone)
    try:
        with pytest.raises(RuntimeError, match="injected database failure"):
            registry.delete_model("candidate")
    finally:
        event.remove(session_class, "before_commit", reject_tombstone)

    assert registry.get_by_version_id("candidate") is not None
    assert artifact.read_bytes() == b"retained"
    with registry._session_factory() as session:
        assert session.query(AuditLog).count() == 0


def test_missing_model_is_not_reported_as_deleted(registry):
    with pytest.raises(ValueError, match="not found"):
        registry.delete_model("missing")


def test_exact_artifact_registration_binds_calibration(registry, tmp_path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"same-model-weights")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "normal.png").write_bytes(b"baseline")
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"threshold":0.5}', encoding="utf-8")
    first = registry.register(artifact, baseline, "cam1", "patchcore")

    calibration.write_text('{"threshold":0.8}', encoding="utf-8")
    second = registry.register(artifact, baseline, "cam1", "patchcore")

    assert registry.get_by_version_id(first).model_hash != (
        registry.get_by_version_id(second).model_hash
    )
