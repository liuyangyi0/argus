"""Managed commands and workflow authority, with no model or media execution."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from argus.core.task_execution import TaskExecutionError
from argus.dashboard.routes import task_platform as routes
from argus.dashboard.routes import task_workflows as workflow_routes
from argus.dashboard.task_platform_contracts import (
    TaskCreateRequest, TaskModelRegisterRequest, TaskReviewRequest, TaskRevisionRequest,
    TaskStartRequest, TaskUpdateRequest,
)
from argus.storage.analysis_tasks import AnalysisTaskService
from argus.storage.database import Database
from argus.storage.task_workflows import TaskWorkflowService


@pytest.fixture
def context(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'platform-api.db'}")
    db.initialize()
    service = AnalysisTaskService(db, models_root=tmp_path / "models")
    state = SimpleNamespace(
        database=db, analysis_task_service=service, camera_manager=None,
        task_execution=None, task_workflows=TaskWorkflowService(db, tmp_path / "datasets"),
        config=SimpleNamespace(cameras=[SimpleNamespace(camera_id="cam")]),
    )
    yield state
    state.task_workflows.close()
    db.close()


def request(state, role="admin", auth_method="session"):
    return Request({"type": "http", "method": "POST", "path": "/api/analysis-tasks", "headers": [],
                    "state": {"user": {"role": role, "username": "actual-user"}, "auth_method": auth_method},
                    "app": SimpleNamespace(state=state)})


def create(state, **changes):
    body = TaskCreateRequest(camera_id="cam", template_id="fast_motion.v1", name="Motion", **changes)
    response = routes.create_task(request(state), body, "create-task-0001")
    assert response.status_code == 201, response.body
    return json.loads(response.body)["data"]


def test_crud_idempotency_revision_and_legacy_config_remain_separate(context):
    task = create(context)
    repeated = create(context)
    assert repeated["task_id"] == task["task_id"]
    assert task["authority"] == "managed" and task["config"] is not None
    response = routes.update_task(request(context), task["task_id"],
                                 TaskUpdateRequest(expected_revision=1, name="Second name"), "update-task-0001")
    assert response.status_code == 200
    assert json.loads(response.body)["data"]["current_revision"] == 2
    stale = routes.update_task(request(context), task["task_id"],
                              TaskUpdateRequest(expected_revision=1, name="Late name"), "update-task-0002")
    assert stale.status_code == 409
    legacy = context.analysis_task_service.bootstrap_legacy("cam", {"source": "rtsp://secret"})
    denied = routes.update_task(request(context), legacy["task_id"],
                               TaskUpdateRequest(expected_revision=1, config={}), "legacy-change-0001")
    assert denied.status_code in {400, 409}
    assert context.analysis_task_service.resolve_legacy("cam")["current_revision"] == 1


def test_readiness_and_runtime_start_never_publish(context):
    task = create(context)
    response = routes.release_readiness(request(context), task["task_id"])
    assert response.status_code == 200
    assert json.loads(response.body)["data"]["production_ready"] is False
    bundle_id = "bundle-" + "a" * 32
    assert routes.start_task(request(context), task["task_id"], TaskStartRequest(bundle_id=bundle_id)).status_code == 409
    executor = Mock()
    executor.start.return_value = {"task_id": task["task_id"], "state": "running", "mode": "shadow"}
    context.task_execution = executor
    assert routes.start_task(request(context), task["task_id"], TaskStartRequest(bundle_id=bundle_id)).status_code == 409
    executor.start.assert_not_called()
    assert routes.stop_task(request(context), task["task_id"]).status_code == 409
    executor.stop.assert_not_called()
    assert context.analysis_task_service.get_task(task["task_id"])["deployment"]["applied_artifact_id"] is None


def test_busy_runtime_blocks_update_and_archive_without_storage_mutation(context):
    task = create(context)
    context.task_execution = Mock()
    context.task_execution.mutate_if_stopped.side_effect = TaskExecutionError("BUSY", "cleanup pending")
    assert routes.update_task(request(context), task["task_id"],
                              TaskUpdateRequest(expected_revision=1, name="unsafe"), "busy-update-0001").status_code == 409
    assert routes.archive_task(request(context), task["task_id"],
                               TaskRevisionRequest(expected_revision=1), "busy-archive-0001").status_code == 409
    assert context.analysis_task_service.get_task(task["task_id"])["current_revision"] == 1


@pytest.mark.parametrize("action", ["update", "archive"])
def test_committed_mutation_retries_reconcile_while_runtime_is_busy(context, action):
    task = create(context)
    body = TaskUpdateRequest(expected_revision=1, name="Updated") if action == "update" else TaskRevisionRequest(expected_revision=1)
    route = routes.update_task if action == "update" else routes.archive_task
    key = f"committed-{action}-0001"
    first = route(request(context), task["task_id"], body, key)
    assert first.status_code == 200
    context.task_execution = Mock()
    context.task_execution.mutate_if_stopped.side_effect = TaskExecutionError("BUSY", "runtime started after commit")
    repeated = route(request(context), task["task_id"], body, key)
    assert repeated.status_code == 200
    assert json.loads(repeated.body)["data"]["current_revision"] == 2
    context.task_execution.mutate_if_stopped.assert_not_called()


def test_mutation_reconciles_commit_between_receipt_read_and_busy_admission(context):
    task = create(context)
    body = TaskUpdateRequest(expected_revision=1, name="Concurrent update")
    key = "parallel-update-0001"
    context.task_execution = Mock()

    def parallel_commit(_task_id, _callback):
        context.analysis_task_service.update_task(task["task_id"], **body.model_dump(), requested_by="actual-user", idempotency_key=key)
        raise TaskExecutionError("BUSY", "parallel command finished and started runtime")

    context.task_execution.mutate_if_stopped.side_effect = parallel_commit
    response = routes.update_task(request(context), task["task_id"], body, key)
    assert response.status_code == 200
    assert context.analysis_task_service.get_task(task["task_id"])["current_revision"] == 2


def test_mutation_retries_do_not_reuse_a_receipt_for_changed_payload(context):
    task = create(context)
    key = "changed-update-0001"
    assert routes.update_task(request(context), task["task_id"], TaskUpdateRequest(expected_revision=1, name="Original"), key).status_code == 200
    context.task_execution = Mock()
    response = routes.update_task(request(context), task["task_id"], TaskUpdateRequest(expected_revision=1, name="Different"), key)
    assert response.status_code == 409
    context.task_execution.mutate_if_stopped.assert_not_called()


def test_mutation_uses_the_runtime_serialization_gate(context):
    task = create(context)
    context.task_execution = Mock()
    context.task_execution.mutate_if_stopped.side_effect = lambda _task_id, callback: callback()
    response = routes.update_task(request(context), task["task_id"], TaskUpdateRequest(expected_revision=1, name="Serialized"), "serialized-update-01")
    assert response.status_code == 200
    context.task_execution.mutate_if_stopped.assert_called_once()


def test_workflow_queries_reject_unknown_tasks_and_filter_recipe_capabilities(context):
    assert workflow_routes.workflows(request(context), "missing").status_code == 404
    assert workflow_routes.recipes(request(context), "missing").status_code == 404
    task = create(context)
    response = workflow_routes.recipes(request(context), task["task_id"])
    assert response.status_code == 200
    assert json.loads(response.body)["data"]["documents"] == []
    pose = context.analysis_task_service.create_task("cam", "pose_estimation.v1", "Pose", {}, requested_by="actual-user", idempotency_key="pose-create-0001")
    compatible = workflow_routes.recipes(request(context), pose["task_id"])
    assert [recipe["recipe_id"] for recipe in json.loads(compatible.body)["data"]["documents"]] == ["external_pose.v1"]


def test_denied_identity_and_unknown_camera_do_not_create_tasks(context):
    body = TaskCreateRequest(camera_id="cam", template_id="fast_motion.v1", name="Denied")
    assert routes.create_task(request(context, role="viewer"), body, "denied-create-0001").status_code == 403
    assert routes.create_task(request(context), body.model_copy(update={"camera_id": "missing"}), "unknown-camera-01").status_code == 404
    assert context.analysis_task_service.list_tasks() == []
    response = routes.create_task(request(context, role="engineer"), body, "engineer-create-0001")
    assert response.status_code == 201


def test_human_approval_cannot_come_from_shared_api_identity(context, monkeypatch):
    monkeypatch.setattr(routes, "require_human_permission", lambda _request, _permission: False)
    body = TaskModelRegisterRequest(relative_path="private.onnx", descriptor={})
    assert routes.register_task_model(request(context, auth_method="api_key"), body, "model-key-0001").status_code == 403
    assert workflow_routes.review(request(context, auth_method="api_key"), "task", "workflow", TaskReviewRequest(
        expected_revision=1, sample_id="sample", expected_image_sha256="b" * 64,
        decision="approved", annotation={},
    )).status_code == 403


def test_workflow_task_scope_is_checked_before_mutation_and_image_read(context):
    task = create(context)
    workflow = context.task_workflows.create(task["task_id"], actor="actual-user", idempotency_key="workflow-0001")
    response = workflow_routes.stop_collection(request(context), "other-task", workflow["workflow_id"],
                                               TaskRevisionRequest(expected_revision=workflow["revision"]))
    assert response.status_code == 404
    assert workflow_routes.sample_image(request(context), "other-task", workflow["workflow_id"], "sample").status_code == 404
    assert context.task_workflows.get(workflow["workflow_id"])["revision"] == workflow["revision"]


def test_public_documents_redact_source_credentials_and_resolved_model_paths():
    payload = routes.safe_document({"artifact_path": "C:/private/model.onnx", "models": [{
        "model_path": "private", "sha256": "a" * 64,
    }], "identity": {"source_declared": "rtsp://user:secret@camera", "source_runtime": "secret",
                      "source_session_id": "session", "pts": None}})
    assert "private" not in json.dumps(payload) and "secret" not in json.dumps(payload)
    assert payload["identity"] == {"source_session_id": "session", "pts": None}


def test_http_contract_rejects_extra_actor_and_requires_idempotency_key(context, monkeypatch):
    app = FastAPI()
    for key, value in vars(context).items():
        setattr(app.state, key, value)
    app.include_router(routes.router, prefix="/api/analysis-tasks")
    monkeypatch.setattr(routes, "_actor", lambda *_args, **_kwargs: "actual-user")
    with TestClient(app) as client:
        payload = {"camera_id": "cam", "template_id": "fast_motion.v1", "name": "Test"}
        assert client.post("/api/analysis-tasks", json=payload).status_code == 422
        assert client.post("/api/analysis-tasks", json={**payload, "requested_by": "forged"},
                           headers={"Idempotency-Key": "http-key-0001"}).status_code == 422
        response = client.post("/api/analysis-tasks", json=payload,
                               headers={"Idempotency-Key": "http-key-0001"})
        assert response.status_code == 201
        record = context.analysis_task_service.get_task(response.json()["data"]["task_id"])
        assert record["created_by"] == "actual-user"


def test_committed_result_with_response_failure_can_reconcile_original_command(context, monkeypatch):
    original = routes._public_task
    monkeypatch.setattr(routes, "_public_task", lambda *_args: {"invalid": True})
    body = TaskCreateRequest(camera_id="cam", template_id="fast_motion.v1", name="Committed")
    response = routes.create_task(request(context), body, "lost-response-0001")
    assert response.status_code == 500
    assert len(context.analysis_task_service.list_tasks()) == 1
    monkeypatch.setattr(routes, "_public_task", original)
    retry = routes.create_task(request(context), body, "lost-response-0001")
    assert retry.status_code == 201
    assert len(context.analysis_task_service.list_tasks()) == 1


def test_start_contract_accepts_the_actual_frozen_bundle_identity(context):
    task = create(context)
    frozen = context.analysis_task_service.freeze_bundle(
        task["task_id"], expected_revision=1, model_ids={},
        requested_by="actual-user", idempotency_key="freeze-identity-0001",
    )
    assert TaskStartRequest(bundle_id=frozen["bundle_id"]).bundle_id == frozen["bundle_id"]


def test_pending_durable_intent_blocks_configuration_without_a_materialized_worker(context):
    from argus.storage.task_deployments import TaskDeploymentService

    task = create(context)
    frozen = context.analysis_task_service.freeze_bundle(task["task_id"], expected_revision=1, model_ids={},
        requested_by="actual-user", idempotency_key="freeze-pending-0001")
    deployments = TaskDeploymentService(context.database, context.analysis_task_service)
    deployments.request(task["task_id"], action="deploy", bundle_id=frozen["bundle_id"],
        expected_deployment_revision=0, requested_by="actual-user", idempotency_key="deploy-pending-0001")
    body = TaskUpdateRequest(expected_revision=1, name="Do not race accepted start")
    assert routes.update_task(request(context), task["task_id"], body, "update-pending-0001").status_code == 409
    assert routes.archive_task(request(context), task["task_id"], TaskRevisionRequest(expected_revision=1),
                               "archive-pending-0001").status_code == 409
    deployments.request(task["task_id"], action="stop", expected_deployment_revision=1,
        requested_by="actual-user", idempotency_key="stop-pending-0001")
    assert routes.update_task(request(context), task["task_id"], body, "update-pending-0001").status_code == 200
