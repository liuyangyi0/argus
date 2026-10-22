"""Legacy training routes cannot forge review identity or scene workflow inputs."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from starlette.requests import Request

from argus.dashboard.routes import training_jobs as routes


def _request(database, body, *, role="engineer", username="authenticated-reviewer"):
    payload = json.dumps(body).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    return Request({
        "type": "http", "method": "POST", "path": "/api/training-jobs/job1/confirm",
        "headers": [(b"content-type", b"application/json")],
        "state": {"user": {"username": username, "role": role}},
        "app": SimpleNamespace(state=SimpleNamespace(
            database=database, config=SimpleNamespace(cameras=[SimpleNamespace(camera_id="cam1")]),
        )),
    }, receive=receive)


@pytest.mark.asyncio
async def test_training_confirmation_uses_authenticated_reviewer_not_body_claim(monkeypatch):
    database = Mock()
    database.get_training_job.return_value = SimpleNamespace(
        status="pending_confirmation", job_type="anomaly_head", camera_id="cam1",
    )
    database.update_training_job.return_value = True
    audit = Mock()
    monkeypatch.setattr(routes, "_write_audit", audit)
    response = await routes.confirm_training_job(
        _request(database, {"confirmed_by": "forged-admin", "approved_by": "another-user"}), "job1",
    )

    assert response.status_code == 200
    assert json.loads(response.body)["data"]["new_status"] == "queued"
    saved = database.update_training_job.call_args
    assert saved.args == ("job1",)
    assert saved.kwargs["confirmed_by"] == "authenticated-reviewer"
    assert saved.kwargs["confirmed_at"].tzinfo is not None
    assert saved.kwargs["expected_status"] == "pending_confirmation"
    assert saved.kwargs["status"] == "queued"
    assert audit.call_args.args[1] == "authenticated-reviewer"


@pytest.mark.asyncio
async def test_confirmation_conflict_does_not_create_a_second_approval_audit(monkeypatch):
    database = Mock()
    database.get_training_job.return_value = SimpleNamespace(
        status="pending_confirmation", job_type="anomaly_head", camera_id="cam1",
    )
    database.update_training_job.return_value = False
    audit = Mock()
    monkeypatch.setattr(routes, "_write_audit", audit)
    response = await routes.confirm_training_job(_request(database, {"confirmed_by": "forged"}), "job1")
    assert response.status_code == 409
    audit.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["viewer", "operator"])
async def test_training_confirmation_still_requires_training_permission(monkeypatch, role):
    database, audit = Mock(), Mock()
    monkeypatch.setattr(routes, "_write_audit", audit)
    response = await routes.confirm_training_job(_request(database, {"confirmed_by": "admin"}, role=role), "job1")
    assert response.status_code == 403
    database.get_training_job.assert_not_called()
    database.update_training_job.assert_not_called()
    audit.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("lineage", [None, {}, {"workflow_id": "forged", "baseline_sha256": "a" * 64}])
async def test_legacy_training_creation_cannot_supply_scene_lineage(monkeypatch, lineage):
    database = Mock()
    validate = Mock(return_value=[])
    monkeypatch.setattr(routes, "validate_hyperparameters", validate)
    response = await routes.create_training_job(_request(database, {
        "job_type": "anomaly_head", "camera_id": "cam1",
        "hyperparameters": {"scene_workflow": lineage},
    }))
    assert response.status_code == 400
    assert "受控场景工作流" in json.loads(response.body)["msg"]
    database.save_training_job.assert_not_called()
    database.get_active_backbone.assert_not_called()
    validate.assert_not_called()


@pytest.mark.asyncio
async def test_training_rejection_uses_authenticated_reviewer_for_status_and_audit(monkeypatch):
    database, audit = Mock(), Mock()
    database.get_training_job.return_value = SimpleNamespace(
        status="pending_confirmation", job_type="anomaly_head", camera_id="cam1",
    )
    database.update_training_job.return_value = True
    monkeypatch.setattr(routes, "_write_audit", audit)
    response = await routes.reject_training_job(_request(database, {
        "rejected_by": "forged-admin", "reason": "wrong scene",
    }), "job1")
    assert response.status_code == 200
    saved = database.update_training_job.call_args.kwargs
    assert saved["expected_status"] == "pending_confirmation" and saved["status"] == "rejected"
    assert saved["error"] == "Rejected by authenticated-reviewer: wrong scene"
    assert audit.call_args.args[1] == "authenticated-reviewer"
    assert audit.call_args.args[4] == "wrong scene"


@pytest.mark.asyncio
async def test_legacy_training_creation_ignores_client_claimed_requester():
    database = Mock()
    database.get_active_backbone.return_value = None
    response = await routes.create_training_job(_request(database, {
        "job_type": "anomaly_head", "camera_id": "cam1", "triggered_by": "forged-admin",
    }))
    assert response.status_code == 201
    job = database.save_training_job.call_args.args[0]
    assert job.triggered_by == "authenticated-reviewer"
    assert job.confirmation_required is True and job.status == "pending_confirmation"


@pytest.mark.asyncio
@pytest.mark.parametrize("username", [None, "", "unknown", "   ", " Unknown ", 123])
@pytest.mark.parametrize("action", ["create_training_job", "confirm_training_job", "reject_training_job"])
async def test_mutations_fail_closed_without_an_auditable_session_identity(monkeypatch, username, action):
    database, audit = Mock(), Mock()
    monkeypatch.setattr(routes, "_write_audit", audit)
    request = _request(database, {
        "job_type": "anomaly_head", "camera_id": "cam1",
        "confirmed_by": "forged-admin", "rejected_by": "forged-admin", "triggered_by": "forged-admin",
    }, username=username)
    args = (request,) if action == "create_training_job" else (request, "job1")
    response = await getattr(routes, action)(*args)
    assert response.status_code == 403
    database.get_training_job.assert_not_called()
    database.update_training_job.assert_not_called()
    database.save_training_job.assert_not_called()
    audit.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["create_training_job", "reject_training_job"])
@pytest.mark.parametrize("body", [[], ["forged-admin"], None, "forged-admin", 42])
async def test_non_object_mutation_payload_is_rejected_without_writes(action, body):
    database = Mock()
    database.get_training_job.return_value = SimpleNamespace(status="pending_confirmation")
    request = _request(database, body)
    args = (request,) if action == "create_training_job" else (request, "job1")
    response = await getattr(routes, action)(*args)
    assert response.status_code == 400
    database.update_training_job.assert_not_called()
    database.save_training_job.assert_not_called()


@pytest.mark.asyncio
async def test_losing_training_rejection_does_not_write_an_approval_audit(monkeypatch):
    database, audit = Mock(), Mock()
    database.get_training_job.return_value = SimpleNamespace(status="pending_confirmation")
    database.update_training_job.return_value = False
    monkeypatch.setattr(routes, "_write_audit", audit)
    response = await routes.reject_training_job(_request(database, {"rejected_by": "forged"}), "job1")
    assert response.status_code == 409
    audit.assert_not_called()
