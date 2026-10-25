"""Managed task storage contracts: SQLite and opaque artifact bytes only."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import create_engine, func, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from argus.core.analysis_task_contracts import get_default_config
from argus.storage.analysis_tasks import (
    AnalysisTaskConflictError,
    AnalysisTaskIntegrityError,
    AnalysisTaskService,
    AnalysisTaskValidationError,
)
from argus.storage.database import Database
from argus.storage.models import (
    AnalysisTaskOperationRecord,
    AnalysisTaskRecord,
    AnalysisTaskRevisionRecord,
    TaskDatasetRecord,
    TaskExecutionBundleRecord,
    TaskModelRecord,
)


@pytest.fixture
def db(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'managed.db'}")
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def service(db, tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    return AnalysisTaskService(db, models_root=root)


def _create(service, *, key="create", camera="cam-a", template="object_detection.v1", config=None):
    return service.create_task(camera, template, "Inspection", config or {},
                               requested_by="operator", idempotency_key=key)


def _descriptor():
    return {
        "name": "Opaque mock registration", "adapter_id": "onnx_detection.v1",
        "output_schema": "yolov8_detection.v1", "labels": ["person", "helmet"],
        "input_size": [640, 640],
    }


def _register(service, *, key="model", name="mock.onnx", scope=None):
    # These are intentionally not executable ONNX bytes. No SDK or model loader
    # is imported by any test in this module.
    path = service._models_root / name
    path.write_bytes(b"opaque mock bytes: fingerprint only")
    return service.register_task_model(name, _descriptor(), scope_camera_id=scope,
                                       requested_by="operator", idempotency_key=key)


def _freeze(service, task, model=None, *, key="freeze"):
    return service.freeze_bundle(task["task_id"], expected_revision=task["current_revision"],
                                 model_ids={"detector": model["model_id"]} if model else {},
                                 requested_by="operator", idempotency_key=key)


def _count(db, model):
    with db.get_session() as session:
        return session.scalar(select(func.count()).select_from(model))


def test_managed_task_defaults_are_detached_and_remain_shadow_only(service):
    config = {"rules": {"labels": ["helmet"]}}
    task = _create(service, config=config)
    config["rules"]["labels"].append("person")
    assert task["authority"] == "managed"
    assert task["current_revision"] == 1
    assert task["archived"] is False
    assert task["created_by"] == "operator"
    assert task["config"]["rules"]["labels"] == ["helmet"]
    assert task["deployment"]["applied_artifact_id"] is None
    assert task["deployment"]["runtime_status"] == "unknown"
    assert task["config"]["resources"] == get_default_config("object_detection.v1")["resources"]


def test_create_receipt_is_atomic_under_concurrent_same_key(service, db):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: _create(service), range(8)))
    assert all(result == results[0] for result in results)
    assert _count(db, AnalysisTaskRecord) == 1
    assert _count(db, AnalysisTaskRevisionRecord) == 1
    assert _count(db, AnalysisTaskOperationRecord) == 1


def test_idempotency_key_is_bound_to_command_and_authenticated_actor(service):
    task = _create(service)
    with pytest.raises(AnalysisTaskConflictError):
        _create(service, camera="cam-b")
    with pytest.raises(AnalysisTaskConflictError):
        service.create_task("cam-a", "object_detection.v1", "Inspection", {},
                            requested_by="different-human", idempotency_key="create")
    with pytest.raises(AnalysisTaskConflictError):
        service.update_task(task["task_id"], expected_revision=1, name="renamed",
                            requested_by="operator", idempotency_key="create")


def test_config_cas_allows_one_concurrent_writer(service, db):
    task = _create(service)

    def write(index):
        try:
            return service.update_task(task["task_id"], expected_revision=1,
                                       name=f"Name {index}", requested_by="operator",
                                       idempotency_key=f"update-{index}")
        except AnalysisTaskConflictError:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(write, range(8)))
    assert len([result for result in results if result is not None]) == 1
    assert service.get_task(task["task_id"])["current_revision"] == 2
    assert _count(db, AnalysisTaskRevisionRecord) == 2


def test_archive_is_revisioned_idempotent_and_visible_to_runtime(service):
    task = _create(service)
    archived = service.archive_task(task["task_id"], expected_revision=1,
                                    requested_by="operator", idempotency_key="archive")
    assert archived["archived"] is True and archived["current_revision"] == 2
    assert service.get_task(task["task_id"])["archived"] is True
    assert service.list_tasks("cam-a")[0]["archived"] is True
    assert service.archive_task(task["task_id"], expected_revision=1,
                                requested_by="operator", idempotency_key="archive") == archived
    with pytest.raises(AnalysisTaskConflictError, match="archived"):
        service.update_task(task["task_id"], expected_revision=2, name="restart",
                            requested_by="operator", idempotency_key="after-archive")


@pytest.mark.parametrize("operation", ["create", "update", "archive", "model", "bundle"])
@pytest.mark.parametrize("after_commit", [False, True])
def test_failed_or_unknown_commit_can_retry_without_partial_state(
    service, db, monkeypatch, operation, after_commit,
):
    task = _create(service) if operation not in {"create", "model"} else None
    model = _register(service) if operation == "bundle" else None
    if operation == "model":
        (service._models_root / "mock.onnx").write_bytes(b"opaque artifact")
    calls = {
        "create": lambda: _create(service, key="retry"),
        "update": lambda: service.update_task(task["task_id"], expected_revision=1, name="new",
                                              requested_by="operator", idempotency_key="retry"),
        "archive": lambda: service.archive_task(task["task_id"], expected_revision=1,
                                                requested_by="operator", idempotency_key="retry"),
        "model": lambda: service.register_task_model("mock.onnx", _descriptor(),
                                                     requested_by="operator", idempotency_key="retry"),
        "bundle": lambda: _freeze(service, task, model, key="retry"),
    }
    before = _count(db, AnalysisTaskOperationRecord)
    commit = Session.commit

    def fail(session):
        if after_commit:
            commit(session)
        raise RuntimeError("acknowledgement lost" if after_commit else "disk write failed")

    with monkeypatch.context() as patch:
        patch.setattr(Session, "commit", fail)
        with pytest.raises(RuntimeError):
            calls[operation]()
    assert _count(db, AnalysisTaskOperationRecord) == before + int(after_commit)
    result = calls[operation]()
    assert calls[operation]() == result
    assert _count(db, AnalysisTaskOperationRecord) == before + 1
    if operation in {"create", "update", "archive"}:
        assert _count(db, AnalysisTaskRevisionRecord) == (1 if operation == "create" else 2)


def test_model_registration_retry_returns_receipt_after_file_disappears(service):
    model = _register(service)
    (service._models_root / "mock.onnx").unlink()
    assert service.register_task_model("mock.onnx", _descriptor(), requested_by="operator",
                                       idempotency_key="model") == model
    task = _create(service)
    with pytest.raises(AnalysisTaskValidationError):
        _freeze(service, task, model)


@pytest.mark.parametrize("action", ["update_task", "archive_task"])
def test_task_command_receipt_is_read_only_and_replays_only_exact_actor_and_payload(
    service, db, monkeypatch, action,
):
    task = _create(service)
    arguments = {"name": "  Renamed  "} if action == "update_task" else {}
    command = getattr(service, action)
    result = command(task["task_id"], expected_revision=1, requested_by=" operator ",
                     idempotency_key="committed-command", **arguments)
    before = _count(db, AnalysisTaskOperationRecord)
    monkeypatch.setattr(service, "_write_lock", lambda _session: pytest.fail("receipt lookup wrote"))
    lookup = {"action": action, "task_id": task["task_id"], "expected_revision": 1,
              "requested_by": "operator", "idempotency_key": "committed-command", **arguments}
    assert service.get_task_command_receipt(**lookup) == result
    assert service.get_task_command_receipt(**{**lookup, "idempotency_key": "not-yet-committed"}) is None
    for changed in (
        {"requested_by": "another-human"}, {"expected_revision": 2}, {"task_id": "another-task"},
        *([{"name": "Different"}, {"config": {}}] if action == "update_task" else []),
    ):
        with pytest.raises(AnalysisTaskConflictError, match="different request"):
            service.get_task_command_receipt(**{**lookup, **changed})
    assert _count(db, AnalysisTaskOperationRecord) == before
    assert service.get_task(task["task_id"]) == result


@pytest.mark.parametrize("change", [
    {"action": "create_task"}, {"action": "freeze_bundle"},
    {"name": "unexpected"}, {"config": {}}, {"expected_revision": True},
    {"requested_by": ""}, {"idempotency_key": ""},
])
def test_task_command_receipt_rejects_unsafe_or_unrelated_requests(service, db, change):
    arguments = {"action": "archive_task", "task_id": "missing-task", "expected_revision": 1,
                 "requested_by": "operator", "idempotency_key": "receipt-check"}
    with pytest.raises(AnalysisTaskValidationError):
        service.get_task_command_receipt(**{**arguments, **change})
    assert _count(db, AnalysisTaskOperationRecord) == 0


def test_changed_model_bytes_cannot_be_frozen_or_resolved(service):
    task = _create(service)
    model = _register(service)
    bundle = _freeze(service, task, model)
    (service._models_root / "mock.onnx").write_bytes(b"new bytes")
    with pytest.raises(AnalysisTaskConflictError, match="changed"):
        _freeze(service, task, model, key="freeze-new")
    with pytest.raises(AnalysisTaskConflictError, match="changed"):
        service.resolve_bundle(task["task_id"], bundle["bundle_id"])
    assert service.register_task_model("mock.onnx", _descriptor(), requested_by="operator",
                                       idempotency_key="model") == model


@pytest.mark.parametrize("path", [
    "../escape.onnx", "nested/../escape.onnx", "/absolute.onnx", "C:/absolute.onnx",
    "C:relative.onnx", "nested\\model.onnx", "//server/model.onnx", "nested//model.onnx",
    "./model.onnx", "model.onnx:stream", "model.pt", "empty.onnx", "missing.onnx",
])
def test_model_paths_are_root_scoped_bounded_and_onnx_only(service, db, path):
    (service._models_root / "empty.onnx").touch()
    with pytest.raises(AnalysisTaskValidationError):
        service.register_task_model(path, _descriptor(), requested_by="operator",
                                    idempotency_key="invalid")
    assert _count(db, TaskModelRecord) == 0
    assert _count(db, AnalysisTaskOperationRecord) == 0


def test_model_file_or_parent_symlink_cannot_escape_root(service, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    artifact = outside / "model.onnx"
    artifact.write_bytes(b"opaque")
    try:
        (service._models_root / "linked.onnx").symlink_to(artifact)
        (service._models_root / "linked-dir").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Windows does not grant symbolic-link creation to this test process")
    for relative in ("linked.onnx", "linked-dir/model.onnx"):
        with pytest.raises(AnalysisTaskValidationError, match="links"):
            service.register_task_model(relative, _descriptor(), requested_by="operator",
                                        idempotency_key=relative)


def test_model_scope_adapter_and_label_rules_are_checked_before_bundle(service):
    task = _create(service, config={"rules": {"labels": ["helmet"]}})
    global_model = _register(service, key="global", name="global.onnx")
    local_model = _register(service, key="local", name="local.onnx", scope="cam-a")
    other_model = _register(service, key="other", name="other.onnx", scope="cam-b")
    assert {item["model_id"] for item in service.list_compatible_models(task["task_id"])} == {
        global_model["model_id"], local_model["model_id"],
    }
    with pytest.raises(AnalysisTaskValidationError, match="scope"):
        _freeze(service, task, other_model)
    with pytest.raises(AnalysisTaskValidationError, match="roles"):
        _freeze(service, task)
    pose = _create(service, key="pose", template="pose_estimation.v1")
    assert service.list_compatible_models(pose["task_id"]) == []
    task = service.update_task(task["task_id"], expected_revision=1,
                               config={"rules": {"labels": ["unregistered-class"]}},
                               requested_by="operator", idempotency_key="labels")
    with pytest.raises(AnalysisTaskValidationError, match="label"):
        _freeze(service, task, global_model)


def test_frozen_bundle_rechecks_revision_and_keeps_paths_server_only(service):
    task = _create(service)
    model = _register(service)
    bundle = _freeze(service, task, model)
    assert bundle["stage"] == "shadow"
    assert bundle["production_readiness"]["decision"] == "NO-GO"
    assert "artifact_path" not in bundle["manifest"]["models"]["detector"]
    resolved = service.resolve_bundle(task["task_id"], bundle["bundle_id"])
    assert resolved["models"]["detector"]["artifact_path"] == str(
        (service._models_root / "mock.onnx").resolve(),
    )
    assert service.get_bundle(bundle["bundle_id"]) == bundle
    service.update_task(task["task_id"], expected_revision=1, name="Renamed",
                        requested_by="operator", idempotency_key="rename")
    with pytest.raises(AnalysisTaskConflictError, match="revision"):
        service.resolve_bundle(task["task_id"], bundle["bundle_id"])
    # Retrying a previously committed freeze returns its historical result,
    # while starting it still requires the current task revision.
    assert _freeze(service, task, model) == bundle


def test_model_less_fast_motion_bundle_stays_shadow_and_stale_archive_is_rejected(service):
    task = _create(service, template="fast_motion.v1")
    bundle = _freeze(service, task)
    assert bundle["manifest"]["config"]["sampling"]["mode"] == "ordered_window"
    assert service.resolve_bundle(task["task_id"], bundle["bundle_id"])["models"] == {}
    service.archive_task(task["task_id"], expected_revision=1,
                         requested_by="operator", idempotency_key="archive")
    with pytest.raises(AnalysisTaskConflictError, match="archived"):
        service.resolve_bundle(task["task_id"], bundle["bundle_id"])


def _role_model(service, name, labels, scope=None):
    path = service._models_root / f"{name}.onnx"
    path.write_bytes(f"opaque role bytes: {name}".encode())
    return service.register_task_model(path.name, {**_descriptor(), "labels": labels},
                                       scope_camera_id=scope, requested_by="operator", idempotency_key=name)


def test_dual_ppe_compatibility_reports_each_role_without_treating_head_labels_as_person_requirement(service):
    task = _create(service, template="ppe_compliance_dual.v1", config={"rules": {"labels": ["person", "helmet"]}})
    person = _role_model(service, "person", ["person"])
    head = _role_model(service, "head", ["helmet", "bare_head"])
    incomplete = _role_model(service, "helmet-only", ["helmet"])
    other = _role_model(service, "other-camera", ["person"], "cam-b")
    models = {model["model_id"]: model for model in service.list_model_compatibility(task["task_id"])}
    assert models[person["model_id"]]["compatible_roles"] == ["person_detector"]
    assert models[head["model_id"]]["compatible_roles"] == ["head_detector"]
    assert models[incomplete["model_id"]]["role_compatibility"]["head_detector"] == {
        "compatible": False, "reasons": ["REQUIRED_LABELS_MISSING"],
    }
    assert "CAMERA_SCOPE_MISMATCH" in models[other["model_id"]]["role_compatibility"]["person_detector"]["reasons"]
    assert {model["model_id"] for model in service.list_compatible_models(task["task_id"])} == {
        person["model_id"], head["model_id"],
    }


def test_dual_ppe_freezes_and_resolves_all_model_roles_as_one_immutable_bundle(service):
    task = _create(service, template="ppe_compliance_dual.v1", config={"rules": {"labels": ["helmet"]}})
    person = _role_model(service, "person", ["person"])
    head = _role_model(service, "head", ["helmet", "bare_head"])
    arguments = {"expected_revision": 1, "requested_by": "operator"}
    with pytest.raises(AnalysisTaskValidationError, match="roles"):
        service.freeze_bundle(task["task_id"], **arguments, model_ids={"person_detector": person["model_id"]},
                               idempotency_key="missing-role")
    with pytest.raises(AnalysisTaskValidationError, match="label"):
        service.freeze_bundle(task["task_id"], **arguments,
                               model_ids={"person_detector": head["model_id"], "head_detector": person["model_id"]},
                               idempotency_key="swapped-roles")
    bundle = service.freeze_bundle(task["task_id"], **arguments,
                                    model_ids={"person_detector": person["model_id"], "head_detector": head["model_id"]},
                                    idempotency_key="both-roles")
    assert bundle["production_readiness"]["decision"] == "NO-GO"
    resolved = service.resolve_bundle(task["task_id"], bundle["bundle_id"])
    assert set(resolved["models"]) == {"person_detector", "head_detector"}
    assert resolved["models"]["person_detector"]["artifact_sha256"] == person["artifact_sha256"]
    assert resolved["models"]["head_detector"]["artifact_sha256"] == head["artifact_sha256"]
    (service._models_root / "head.onnx").write_bytes(b"changed-head-model")
    with pytest.raises(AnalysisTaskConflictError, match="changed"):
        service.resolve_bundle(task["task_id"], bundle["bundle_id"])


def test_dual_ppe_rejects_unknown_task_output_labels_and_retains_original_ppe_requirements(service):
    with pytest.raises(AnalysisTaskValidationError, match="dual PPE"):
        _create(service, template="ppe_compliance_dual.v1", config={"rules": {"labels": ["vehicle"]}})
    single = _create(service, key="single", template="ppe_compliance.v1")
    detector = _role_model(service, "missing-bare-head", ["person", "helmet"])
    assert service.list_compatible_models(single["task_id"]) == []
    with pytest.raises(AnalysisTaskValidationError, match="label"):
        _freeze(service, single, detector)


def test_fast_motion_label_rules_cannot_silently_target_an_unsupported_class(service):
    with pytest.raises(AnalysisTaskValidationError, match="fast_projectile"):
        _create(service, template="fast_motion.v1", config={"rules": {"labels": ["person"]}})
    task = _create(service, template="fast_motion.v1",
                   config={"rules": {"labels": ["fast_projectile"]}})
    assert task["config"]["rules"]["labels"] == ["fast_projectile"]
    # The managed catalog restriction never rewrites existing anomaly config.
    legacy = service.bootstrap_legacy("legacy-camera", {"rules": {"labels": ["legacy-label"]}})
    assert legacy["config"]["rules"]["labels"] == ["legacy-label"]


def test_bundle_fingerprinting_does_not_hold_write_lock_and_rechecks_task_cas(service, db, monkeypatch):
    task = _create(service)
    model = _register(service)
    fingerprint = service._trusted_artifact

    def fingerprint_then_concurrent_update(relative):
        result = fingerprint(relative)
        service.update_task(task["task_id"], expected_revision=1, name="changed during fingerprint",
                            requested_by="operator", idempotency_key="concurrent-update")
        return result

    monkeypatch.setattr(service, "_trusted_artifact", fingerprint_then_concurrent_update)
    with pytest.raises(AnalysisTaskConflictError, match="revision"):
        _freeze(service, task, model)
    assert _count(db, TaskExecutionBundleRecord) == 0


def test_legacy_task_configuration_and_publisher_authority_cannot_be_adopted(service):
    task = service.bootstrap_legacy("cam-a", {"guard_enabled": True})
    for action in (
        lambda: service.update_task(task["task_id"], expected_revision=1, config={},
                                    requested_by="operator", idempotency_key="legacy-update"),
        lambda: service.archive_task(task["task_id"], expected_revision=1,
                                     requested_by="operator", idempotency_key="legacy-archive"),
        lambda: _freeze(service, task),
    ):
        with pytest.raises(AnalysisTaskConflictError, match="legacy"):
            action()
    assert service.resolve_legacy("cam-a")["config"] == {"guard_enabled": True}


@pytest.mark.parametrize("table", ["task_models", "task_execution_bundles", "analysis_task_operations", "task_datasets"])
@pytest.mark.parametrize("operation", ["DELETE", "UPDATE"])
def test_shadow_registrations_bundles_receipts_and_datasets_are_immutable_sql(
    service, db, table, operation,
):
    task = _create(service)
    model = _register(service)
    _freeze(service, task, model)
    with db.get_session() as session:
        session.add(TaskDatasetRecord(dataset_id="frozen", task_id=task["task_id"], task_revision=1,
                                      manifest_json="{}", manifest_sha256=hashlib.sha256(b"{}").hexdigest(),
                                      created_by="operator"))
        session.commit()
    with pytest.raises(IntegrityError, match="immutable"):
        with db._engine.begin() as conn:
            conn.execute(text(f"DELETE FROM {table}" if operation == "DELETE" else
                              f"UPDATE {table} SET created_by='rewritten'" if table != "analysis_task_operations"
                              else f"UPDATE {table} SET requested_by='rewritten'"))


@pytest.mark.parametrize("manifest", ["[]", "null", "{\"stage\":\"production\"}"])
def test_corrupt_bundle_manifest_fails_with_integrity_error(service, db, manifest):
    task = _create(service)
    with db.get_session() as session:
        session.add(TaskExecutionBundleRecord(
            bundle_id="corrupt", task_id=task["task_id"], task_revision=1,
            template_id="object_detection.v1", manifest_json=manifest,
            manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(), created_by="operator",
        ))
        session.commit()
    with pytest.raises(AnalysisTaskIntegrityError):
        service.get_bundle("corrupt")


def test_old_dataset_migration_preserves_unsigned_data_without_blessing_it(tmp_path):
    url = f"sqlite:///{tmp_path / 'old-datasets.db'}"
    engine = create_engine(url)
    old_manifest = json.dumps({"legacy": True})
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE task_datasets (dataset_id VARCHAR(64) PRIMARY KEY, "
                          "task_id VARCHAR(64) NOT NULL, task_revision INTEGER NOT NULL, "
                          "manifest_json TEXT NOT NULL, created_by VARCHAR(100) NOT NULL, "
                          "created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL)"))
        conn.execute(text("INSERT INTO task_datasets (dataset_id,task_id,task_revision,manifest_json,created_by) "
                          "VALUES ('old','task-old',1,:manifest,'old-human')"), {"manifest": old_manifest})
    engine.dispose()
    database = Database(url)
    try:
        database.initialize()
        database.initialize()
        assert "manifest_sha256" in {item["name"] for item in inspect(database._engine).get_columns("task_datasets")}
        with database.get_session() as session:
            row = session.get(TaskDatasetRecord, "old")
            assert row.manifest_json == old_manifest
            assert row.manifest_sha256 is None
            assert row.created_by == "old-human"
    finally:
        database.close()


@pytest.mark.parametrize("config", [
    {"unexpected": True}, {"resources": {"max_inflight": 2}},
    {"sampling": {"fps_target": float("nan")}}, {"rules": {"labels": ["x", "x"]}},
    {"scene": {"policy": "none", "profile_id": "unapproved"}},
])
def test_closed_configuration_rejects_unknown_unsafe_and_nonfinite_values(service, config):
    with pytest.raises(AnalysisTaskValidationError):
        _create(service, config=config)


def test_registered_pose_descriptor_matches_the_only_supported_coco17_schema(service):
    (service._models_root / "pose.onnx").write_bytes(b"opaque pose artifact")
    descriptor = {**_descriptor(), "adapter_id": "onnx_pose.v1", "output_schema": "yolov8_pose.v1"}
    with pytest.raises(AnalysisTaskValidationError, match="COCO-17"):
        service.register_task_model("pose.onnx", descriptor, requested_by="operator",
                                    idempotency_key="bad-pose")
    descriptor["labels"] = ["person"]
    model = service.register_task_model("pose.onnx", descriptor, requested_by="operator",
                                        idempotency_key="good-pose")
    assert model["labels"] == ["person"]


def test_ppe_bundles_require_explicit_bare_head_evidence_schema(service):
    task = _create(service, template="ppe_compliance.v1")
    model = _register(service)
    assert service.list_compatible_models(task["task_id"]) == []
    with pytest.raises(AnalysisTaskValidationError, match="bare_head"):
        _freeze(service, task, model)
    descriptor = _descriptor()
    descriptor["labels"].append("bare_head")
    model = service.register_task_model("mock.onnx", descriptor, requested_by="operator",
                                        idempotency_key="ppe-model")
    assert service.list_compatible_models(task["task_id"])[0]["model_id"] == model["model_id"]
    assert _freeze(service, task, model, key="ppe-bundle")["stage"] == "shadow"


def test_result_commit_is_serialized_with_task_update_and_stale_revision_cannot_commit(service):
    task = _create(service)
    entered, release, writer_started = threading.Event(), threading.Event(), threading.Event()
    results = []

    def append_result():
        entered.set()
        assert release.wait(3)
        results.append("committed")

    def update():
        writer_started.set()
        return service.update_task(task["task_id"], expected_revision=1, name="new",
                                    requested_by="operator", idempotency_key="update")

    with ThreadPoolExecutor(max_workers=2) as pool:
        commit = pool.submit(service.commit_if_current, task["task_id"], 1,
                             task["config_sha256"], append_result)
        assert entered.wait(3)
        write = pool.submit(update)
        assert writer_started.wait(3)
        assert not write.done()
        release.set()
        assert commit.result(timeout=3) is True
        assert write.result(timeout=3)["current_revision"] == 2
    assert results == ["committed"]
    assert not service.commit_if_current(task["task_id"], 1, task["config_sha256"],
                                          lambda: results.append("stale"))
    assert results == ["committed"]


def test_result_commit_rejects_hash_archive_legacy_and_releases_lock_after_callback_failure(service):
    task = _create(service)
    assert not service.commit_if_current(task["task_id"], 1, "0" * 64,
                                          lambda: pytest.fail("mismatched config committed"))
    with pytest.raises(RuntimeError, match="append failed"):
        service.commit_if_current(task["task_id"], 1, task["config_sha256"],
                                   lambda: (_ for _ in ()).throw(RuntimeError("append failed")))
    service.archive_task(task["task_id"], expected_revision=1,
                         requested_by="operator", idempotency_key="archive")
    assert not service.commit_if_current(task["task_id"], 2, task["config_sha256"],
                                          lambda: pytest.fail("archived task committed"))
    legacy = service.bootstrap_legacy("legacy-camera", {})
    assert not service.commit_if_current(legacy["task_id"], 1, legacy["config_sha256"],
                                          lambda: pytest.fail("legacy publisher was adopted"))


def test_result_commit_never_waits_for_a_database_writer_and_restores_connection_timeout(
    service, db, monkeypatch,
):
    task = _create(service)
    committed = []
    # A distinct DBAPI connection represents another process running a normal
    # long workflow transaction. Keep the result fence on one known connection
    # so timeout restoration cannot accidentally be checked on a different one.
    locker = sqlite3.connect(str(db._engine.url.database), timeout=0)
    try:
        with db._engine.connect() as fence_connection:
            fence_connection.exec_driver_sql("PRAGMA busy_timeout=1379")
            fence_connection.commit()
            monkeypatch.setattr(db, "get_session", lambda: Session(bind=fence_connection))
            locker.execute("BEGIN IMMEDIATE")
            started = time.perf_counter()
            result = service.commit_if_current(task["task_id"], 1, task["config_sha256"],
                                                lambda: committed.append("result"))
            elapsed = time.perf_counter() - started
            assert result is False and committed == []
            assert elapsed < 0.5, f"result commit waited {elapsed:.3f}s under the source fence"
            assert fence_connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one() == 1379
            fence_connection.rollback()
            locker.rollback()
            assert service.commit_if_current(task["task_id"], 1, task["config_sha256"],
                                               lambda: committed.append("result"))
            assert committed == ["result"]
            assert fence_connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one() == 1379
            fence_connection.rollback()
            with pytest.raises(RuntimeError, match="append failed"):
                service.commit_if_current(task["task_id"], 1, task["config_sha256"],
                                           lambda: (_ for _ in ()).throw(RuntimeError("append failed")))
            assert fence_connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one() == 1379
            fence_connection.rollback()
    finally:
        locker.close()
