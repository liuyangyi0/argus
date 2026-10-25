"""Deployment and source HTTP contracts with mock computation only."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from argus.dashboard.routes import cameras, task_deployments as routes
from argus.dashboard.task_platform_contracts import TaskConfigurationRestoreRequest, TaskDeploymentRequest


def request(state, role="admin"):
    return Request({"type": "http", "method": "POST", "path": "/", "headers": [],
                    "state": {"user": {"role": role, "username": "operator"}, "auth_method": "session"},
                    "app": SimpleNamespace(state=state)})


def test_durable_intent_returns_accepted_even_when_wakeup_fails():
    service, coordinator = Mock(), Mock()
    state = SimpleNamespace(task_deployments=service, task_deployment_coordinator=coordinator)
    service.request.return_value = {"task_id": "task", "operation_id": "op-1", "status": "pending"}
    coordinator.submit.side_effect = RuntimeError("wake-up unavailable")
    body = TaskDeploymentRequest(action="deploy", bundle_id="bundle-" + "a" * 32,
                                 expected_deployment_revision=4)
    response = routes.request_deployment(request(state), "task", body, "durable-key-0001")
    assert response.status_code == 202
    assert json.loads(response.body)["data"]["document"]["operation_id"] == "op-1"
    service.request.assert_called_once_with("task", **body.model_dump(), requested_by="operator",
                                            idempotency_key="durable-key-0001")


def test_deployment_read_is_observational_and_operation_is_task_scoped():
    service, coordinator = Mock(), Mock()
    state = SimpleNamespace(task_deployments=service, task_deployment_coordinator=coordinator)
    service.get.return_value = {"task_id": "task", "desired_bundle_id": "bundle", "state": "waiting_source"}
    assert routes.deployment(request(state, "viewer"), "task").status_code == 200
    coordinator.reconcile.assert_not_called()
    coordinator.recover.assert_not_called()
    service.get_operation.return_value = {"task_id": "other", "operation_id": "op"}
    assert routes.operation(request(state, "viewer"), "task", "op").status_code == 404


def test_viewer_cannot_submit_or_reconcile_and_no_runtime_fallback_exists():
    service, coordinator = Mock(), Mock()
    state = SimpleNamespace(task_deployments=service, task_deployment_coordinator=coordinator)
    body = TaskDeploymentRequest(action="stop", expected_deployment_revision=0)
    assert routes.request_deployment(request(state, "viewer"), "task", body, "denied-key-0001").status_code == 403
    assert routes.reconcile(request(state, "viewer"), "task").status_code == 403
    service.request.assert_not_called()
    unavailable = SimpleNamespace(task_deployments=service, task_deployment_coordinator=None)
    assert routes.request_deployment(request(unavailable), "task", body, "missing-runtime-01").status_code == 503
    service.request.assert_not_called()


def test_deployment_http_requires_key_and_refuses_client_actor(monkeypatch):
    app = FastAPI()
    app.state.task_deployments = Mock()
    app.state.task_deployment_coordinator = Mock()
    app.include_router(routes.router, prefix="/api/analysis-tasks")
    monkeypatch.setattr(routes, "_actor", lambda *_args: "operator")
    body = {"action": "stop", "expected_deployment_revision": 0}
    with TestClient(app) as client:
        assert client.post("/api/analysis-tasks/task/deployments", json=body).status_code == 422
        assert client.post("/api/analysis-tasks/task/deployments", json={**body, "requested_by": "forged"},
                           headers={"Idempotency-Key": "forged-key-0001"}).status_code == 422


@pytest.mark.asyncio
async def test_source_start_does_not_start_legacy_or_publish(monkeypatch):
    manager = Mock()
    manager.get_camera_config.return_value = SimpleNamespace(camera_id="cam")
    manager.start_source.return_value = True
    manager.get_source_runtime_status.return_value = {
        "state": "running", "running": True, "connected": True, "generation": 2,
        "error": None, "source_declared": "must-not-disclose", "cleanup_pending": False,
    }
    state = SimpleNamespace(camera_manager=manager)
    monkeypatch.setattr(cameras, "_ensure_go2rtc_stream", lambda *_args: None)
    response = await cameras.camera_source_command(request(state), "cam", "start")
    assert response.status_code == 200
    data = json.loads(response.body)["data"]
    assert data["state"] == "running" and "source_declared" not in data
    manager.start_camera.assert_not_called()
    manager.start_source.assert_called_once_with("cam")
    manager.set_analysis_task_enabled.assert_not_called()


@pytest.mark.asyncio
async def test_source_stop_cleanup_failure_does_not_report_stopped():
    manager = Mock()
    manager.get_camera_config.return_value = SimpleNamespace(camera_id="cam")
    manager.stop_source.return_value = False
    manager.get_source_runtime_status.return_value = {
        "state": "cleanup_pending", "running": False, "connected": False,
        "cleanup_pending": True, "error": "rtsp://private-credential",
    }
    response = await cameras.camera_source_command(request(SimpleNamespace(camera_manager=manager)), "cam", "stop")
    assert response.status_code == 409
    assert b"private-credential" not in response.body
    assert json.loads(response.body)["data"]["reason_code"] == "CLEANUP_PENDING"


def test_source_status_requires_read_permission_and_does_not_start():
    manager = Mock()
    manager.get_source_runtime_status.return_value = {"state": "stopped", "running": False, "connected": False}
    assert cameras.camera_source_status(request(SimpleNamespace(camera_manager=manager), "viewer"), "cam").status_code == 200
    manager.start_source.assert_not_called()


def test_deployment_lost_commit_ack_is_500_and_same_key_can_reconcile():
    service, coordinator = Mock(), Mock()
    committed = {"task_id": "task", "operation_id": "only-operation", "status": "pending"}
    service.request.side_effect = [RuntimeError("committed but ACK lost"), committed]
    state = SimpleNamespace(task_deployments=service, task_deployment_coordinator=coordinator)
    body = TaskDeploymentRequest(action="stop", expected_deployment_revision=0)
    assert routes.request_deployment(request(state), "task", body, "lost-deploy-ack-01").status_code == 500
    response = routes.request_deployment(request(state), "task", body, "lost-deploy-ack-01")
    assert response.status_code == 202
    assert service.request.call_args_list[0] == service.request.call_args_list[1]
    assert json.loads(response.body)["data"]["document"]["operation_id"] == "only-operation"


def test_configuration_restore_unknown_commit_retains_receipt_retry_ahead_of_runtime_gate(monkeypatch):
    # This test isolates commit acknowledgement recovery, not account lookup.
    # Human-session identity refresh is independently enforced by _actor.
    monkeypatch.setattr(routes, "_actor", lambda *_args, **_kwargs: "operator")
    service, executor = Mock(), Mock()
    service.get_bundle_configuration_restore_receipt.return_value = None
    receipt = {"task": {"task_id": "task", "current_revision": 3},
               "bundle": {"bundle_id": "bundle-" + "b" * 32}, "restore": {"new_task_revision": 3}}

    def committed(*_args, **_kwargs):
        service.get_bundle_configuration_restore_receipt.return_value = receipt
        raise RuntimeError("commit acknowledgement lost")

    executor.mutate_if_stopped.side_effect = lambda _task_id, callback: callback()
    service.restore_bundle_configuration.side_effect = committed
    state = SimpleNamespace(analysis_task_service=service, task_execution=executor)
    body = TaskConfigurationRestoreRequest(bundle_id="bundle-" + "a" * 32, expected_revision=2)
    assert routes.restore_configuration(request(state), "task", body, "restore-ack-0001").status_code == 500
    executor.mutate_if_stopped.reset_mock()
    executor.mutate_if_stopped.side_effect = RuntimeError("a later execution has started")
    repeated = routes.restore_configuration(request(state), "task", body, "restore-ack-0001")
    assert repeated.status_code == 201
    assert json.loads(repeated.body)["data"]["document"]["restore"]["new_task_revision"] == 3
    executor.mutate_if_stopped.assert_not_called()
    service.restore_bundle_configuration.assert_called_once()
