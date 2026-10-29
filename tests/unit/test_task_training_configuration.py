"""Startup trust/configuration and cached capabilities never execute model work."""

import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from argus.config.schema import ArgusConfig, TaskTrainingConfig, TrustedTaskTrainingBaseModelConfig
from argus.core.task_training import TaskTrainingService
from argus.dashboard.task_lifecycle_contracts import TaskTrainingCapabilitiesResponse
from argus.storage.models import TaskTrainingExecutionRecord
from argus.storage.task_workflows import TaskWorkflowBlocked
from test_task_training import environment as environment, training_environment


def _base(**changes):
    return {"model_id": "approved-yolov8n", "name": "Operator-approved YOLOv8n",
            "relative_path": "training-bases/yolov8n.pt", "artifact_sha256": "a" * 64, **changes}


def _service(tmp_path, **kwargs):
    workflows = SimpleNamespace(_db=Mock())
    return TaskTrainingService(workflows, tmp_path / "models" / "training", tmp_path / "models", **kwargs)


def test_task_training_defaults_deny_execution_and_trust():
    config = ArgusConfig().task_training
    assert config == TaskTrainingConfig()
    assert config.enabled is False and config.maintenance_approved is False
    assert config.trusted_base_models == []
    assert config.resource_mode == "exclusive_maintenance" and config.device == "cuda:0"
    assert config.minimum_free_memory_mb == 4096 and config.poll_interval_seconds == 1.0


def test_task_training_enabled_without_maintenance_is_valid_waiting_configuration():
    config = TaskTrainingConfig(enabled=True, maintenance_approved=False)
    assert config.enabled is True and config.maintenance_approved is False
    assert ArgusConfig.model_validate({"task_training": config.model_dump()}).task_training == config


@pytest.mark.parametrize("changes", [
    {"enabled": "true"}, {"enabled": 1}, {"maintenance_approved": "false"},
    {"device": "cpu"}, {"device": "cuda:1"}, {"resource_mode": "shared"},
    {"minimum_free_memory_mb": True}, {"minimum_free_memory_mb": "4096"},
    {"minimum_free_memory_mb": 4096.0}, {"minimum_free_memory_mb": 0},
    {"minimum_free_memory_mb": 1048577}, {"poll_interval_seconds": True},
    {"poll_interval_seconds": "1"}, {"poll_interval_seconds": 0},
    {"poll_interval_seconds": 61}, {"poll_interval_seconds": float("nan")},
    {"poll_interval_seconds": float("inf")}, {"production_allowed": True},
])
def test_task_training_configuration_is_strict_and_bounded(changes):
    with pytest.raises(ValidationError):
        TaskTrainingConfig.model_validate(changes)


@pytest.mark.parametrize("path", [
    "../base.pt", "bases/../base.pt", "./base.pt", "/base.pt", "C:/base.pt",
    "C:\\base.pt", "\\\\server\\base.pt", "bases\\base.pt", "bases//base.pt",
    "base.xml", "base.PT", "https://example.test/base.pt", "base.pt:stream", "base.pt\x00",
    "bases/./base.pt", "base .pt", "bases/CON.pt", "bases/NUL/base.pt", "bases./base.pt",
])
def test_trusted_training_base_paths_are_canonical_relative_pt_identifiers(path):
    with pytest.raises(ValidationError):
        TrustedTaskTrainingBaseModelConfig.model_validate(_base(relative_path=path))


@pytest.mark.parametrize("changes", [
    {"model_id": "../unsafe"}, {"model_id": ""}, {"name": " "},
    {"name": " display label"}, {"name": "label\nsecret"},
    {"artifact_sha256": "A" * 64}, {"artifact_sha256": "a" * 63},
    {"artifact_sha256": "g" * 64}, {"artifact_sha256": 123},
    {"absolute_path": "C:/private/base.pt"},
])
def test_trusted_training_base_metadata_is_strict(changes):
    with pytest.raises(ValidationError):
        TrustedTaskTrainingBaseModelConfig.model_validate(_base(**changes))


@pytest.mark.parametrize("second", [
    _base(relative_path="other.pt"),
    _base(model_id="second"),
    _base(model_id="second", relative_path="TRAINING-BASES/yolov8n.pt"),
])
def test_trusted_training_base_ids_and_paths_are_unique(second):
    with pytest.raises(ValidationError):
        TaskTrainingConfig(trusted_base_models=[_base(), second])


def test_capabilities_copy_startup_trust_without_touching_models_or_gpu(tmp_path, monkeypatch):
    config = TaskTrainingConfig(enabled=True, trusted_base_models=[_base()])
    service = _service(tmp_path, configuration=config)
    forbidden = Mock(side_effect=AssertionError("capabilities may not read model files"))
    monkeypatch.setattr("argus.core.task_training.artifact_sha256", forbidden)
    monkeypatch.setattr("pathlib.Path.open", forbidden)
    config.trusted_base_models.clear()
    capabilities = service.capabilities()
    TaskTrainingCapabilitiesResponse.model_validate(capabilities)
    assert capabilities["configured_enabled"] is True
    assert capabilities["runtime"] == {"state": "disabled", "enabled": False, "error_code": None}
    assert capabilities["resource"]["state"] == "disabled"
    assert capabilities["trusted_base_models"] == [{**_base(), "configured_trust": True}]
    assert capabilities["validation_only"] is True and capabilities["production_allowed"] is False
    assert not (tmp_path / "models").exists()
    forbidden.assert_not_called()


def test_legacy_trusted_mapping_constructor_remains_compatible(tmp_path):
    original = {"base.pt": "a" * 64}
    service = _service(tmp_path, trusted_base_models=original)
    original.clear()
    assert service.trusted_base_models == {"base.pt": "a" * 64}
    assert service.trusted_base_models.copy() == {"base.pt": "a" * 64}
    options = service.capabilities()["trusted_base_models"]
    assert options[0]["relative_path"] == "base.pt" and options[0]["artifact_sha256"] == "a" * 64
    service.trusted_base_models.clear()
    assert service.capabilities()["trusted_base_models"] == []


def test_trust_cannot_be_ambiguously_combined_with_legacy_mapping(tmp_path):
    with pytest.raises(ValueError, match="one startup"):
        _service(tmp_path, trusted_base_models={"base.pt": "a" * 64},
                 configuration=TaskTrainingConfig(trusted_base_models=[_base()]))


@pytest.mark.parametrize("change", ["add_path", "replace_digest"])
def test_startup_trust_cannot_be_extended_by_mutating_public_mapping(environment, tmp_path, change):
    service, arguments = training_environment(environment, tmp_path)
    relative = "unconfigured.pt" if change == "add_path" else arguments["base_model_path"]
    payload = b"synthetic replacement weights; never deserialized"
    (service.models_root / relative).write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    with pytest.raises(TaskWorkflowBlocked, match="TRUST"):
        service.trusted_base_models[relative] = digest
        service.submit(**{**arguments, "base_model_path": relative, "base_model_sha256": digest})
    assert service.list() == []


def test_submit_rechecks_revoked_trust_at_queue_commit(environment, tmp_path, monkeypatch):
    service, arguments = training_environment(environment, tmp_path)
    begin = service.workflows._begin

    def revoke_before_commit(session):
        begin(session)
        service.trusted_base_models.clear()

    monkeypatch.setattr(service.workflows, "_begin", revoke_before_commit)
    with pytest.raises(TaskWorkflowBlocked, match="TRUST"):
        service.submit(**arguments)
    assert service.list() == []


def test_claim_rechecks_revoked_trust_after_preparation(environment, tmp_path, monkeypatch):
    service, arguments = training_environment(environment, tmp_path)
    job = service.submit(**arguments)
    prepare = service.prepare

    def prepare_then_revoke(job_id):
        plan = prepare(job_id)
        service.trusted_base_models.clear()
        return plan

    monkeypatch.setattr(service, "prepare", prepare_then_revoke)
    with pytest.raises(TaskWorkflowBlocked, match="TRUST"):
        service.claim(job["job_id"], expected_revision=job["revision"])
    assert service.get(job["job_id"])["status"] == "queued"


def test_completion_rechecks_revoked_trust_at_result_commit(environment, tmp_path):
    service, arguments = training_environment(environment, tmp_path)
    job = service.submit(**arguments)
    claimed, plan = service.claim(job["job_id"], expected_revision=job["revision"])
    service.trusted_base_models.clear()
    result = {"job_id": job["job_id"], "execution_id": plan["execution_id"],
              "request_sha256": plan["request_sha256"],
              "package_sha256": arguments["expected_package_sha256"]}
    with pytest.raises(TaskWorkflowBlocked, match="TRUST"):
        service.transition(job["job_id"], expected_revision=claimed["revision"],
                           from_states={"running"}, status="complete", result=result)
    assert service.get(job["job_id"])["status"] == "running"


def test_exact_submission_receipt_remains_readable_after_trust_revocation(environment, tmp_path):
    service, arguments = training_environment(environment, tmp_path)
    job = service.submit(**arguments)
    service.trusted_base_models.clear()
    assert service.submit(**arguments) == job
    with pytest.raises(TaskWorkflowBlocked, match="TRUST"):
        service.claim(job["job_id"], expected_revision=job["revision"])


def test_execution_authorization_rechecks_claim_before_any_worker_start(environment, tmp_path):
    service, arguments = training_environment(environment, tmp_path)
    job = service.submit(**arguments)
    claimed, plan = service.claim(job["job_id"], expected_revision=job["revision"])
    service.cancel(job["job_id"], expected_revision=claimed["revision"], actor="human")
    started = Mock()
    with pytest.raises((TaskWorkflowBlocked, RuntimeError), match="changed|queued|current|cancel"):
        with service.execution_authorization(job["job_id"], expected_revision=claimed["revision"],
                                            execution_id=plan["execution_id"], request_sha256=plan["request_sha256"]):
            started()
    started.assert_not_called()


@pytest.mark.parametrize("field,value", [("task_id", "another-task"),
                                        ("workflow_id", "another-workflow"),
                                        ("created_by", "another-human")])
def test_training_receipt_rejects_row_identity_drift(environment, tmp_path, field, value):
    service, arguments = training_environment(environment, tmp_path)
    job = service.submit(**arguments)
    with service._db.get_session() as session:
        row = session.get(TaskTrainingExecutionRecord, job["job_id"])
        setattr(row, field, value)
        session.commit()
    with pytest.raises(TaskWorkflowBlocked, match="IDENTITY"):
        service.get(job["job_id"])
    with pytest.raises(TaskWorkflowBlocked, match="IDENTITY"):
        service.submit(**arguments)


def test_capabilities_only_copy_safe_cached_runtime_resource_fields(tmp_path):
    service = _service(tmp_path)
    capabilities = service.capabilities(
        runtime_snapshot={"state": "waiting_resources", "enabled": True,
                          "error_code": "TRAINING_MAINTENANCE_REQUIRED", "exception": "C:/private"},
        resource_snapshot={"state": "waiting_maintenance", "reason_code": "TRAINING_MAINTENANCE_REQUIRED",
                           "free_memory_mb": 2048, "device": "cuda:0", "memory_limit_enforced": True,
                           "private_path": "C:/private"})
    TaskTrainingCapabilitiesResponse.model_validate(capabilities)
    assert capabilities["runtime"]["state"] == "waiting_resources"
    assert capabilities["resource"]["state"] == "waiting_maintenance"
    assert capabilities["resource"]["free_memory_mb"] == 2048
    assert capabilities["resource"]["memory_limit_enforced"] is False
    assert "private" not in str(capabilities)


def test_invalid_cached_snapshots_fail_closed_without_diagnostic_text(tmp_path):
    capabilities = _service(tmp_path).capabilities(
        runtime_snapshot={"state": [], "enabled": "yes", "error_code": "C:/private/model.pt"},
        resource_snapshot={"state": {}, "reason_code": "secret exception", "free_memory_mb": True})
    assert capabilities["runtime"] == {"state": "error", "enabled": False, "error_code": None}
    assert capabilities["resource"]["state"] == "blocked"
    assert capabilities["resource"]["reason_code"] is None
    assert capabilities["resource"]["free_memory_mb"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [
    {"enabled": True}, {"maintenance_approved": True}, {"minimum_free_memory_mb": 8192},
    {"poll_interval_seconds": 2.0}, {"trusted_base_models": [_base()]},
])
async def test_generic_config_reload_cannot_replace_startup_training_authority(monkeypatch, changes):
    from argus.dashboard.routes import config as routes

    current, replacement = ArgusConfig(), ArgusConfig(task_training=TaskTrainingConfig(**changes))
    manager = Mock()
    state = SimpleNamespace(config=current, config_path="no-file-read.yaml", camera_manager=manager)
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    monkeypatch.setattr(routes, "_deny_without_permission", lambda *args: None)
    monkeypatch.setattr("argus.config.loader.load_config", lambda _: replacement)
    response = await routes._reload_config(request, None)
    assert response.status_code == 409
    assert state.config is current
    manager.apply_runtime_camera_config.assert_not_called()


@pytest.mark.asyncio
async def test_generic_config_reload_preserves_unchanged_startup_training_identity(monkeypatch):
    from argus.dashboard.routes import config as routes

    current, replacement = ArgusConfig(), ArgusConfig()
    manager = Mock()
    state = SimpleNamespace(config=current, config_path="no-file-read.yaml", camera_manager=manager)
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    monkeypatch.setattr(routes, "_deny_without_permission", lambda *args: None)
    monkeypatch.setattr("argus.config.loader.load_config", lambda _: replacement)
    response = await routes._reload_config(request, None)
    assert response.status_code == 200
    assert state.config is replacement
    assert replacement.task_training is current.task_training


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["task_training.enabled", "task_training.maintenance_approved"])
async def test_generic_module_toggle_cannot_mutate_training_authority(monkeypatch, key):
    from argus.dashboard.routes import config as routes

    configuration = ArgusConfig()
    state = SimpleNamespace(config=configuration)
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    monkeypatch.setattr(routes, "_deny_without_permission", lambda *args: None)
    response = await routes._apply_module_toggle(request, routes.ModuleToggleRequest(key=key, value=True))
    assert response.status_code == 400
    assert configuration.task_training == TaskTrainingConfig()
