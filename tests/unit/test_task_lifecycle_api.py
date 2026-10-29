"""Task lifecycle HTTP contracts use mocked services, never compute or media."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from argus.config.schema import TaskTrainingConfig
from argus.dashboard.routes.task_lifecycle import router
from argus.storage.database import Database
from argus.storage.task_events import TaskEventConflict, TaskEventEvidenceUnavailable


class _Identity:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            scope.setdefault("state", {}).update({
                "user": {"username": "reviewer", "role": headers.get(b"x-test-role", b"admin").decode()},
                "auth_method": headers.get(b"x-test-auth", b"session").decode(),
            })
        await self.app(scope, receive, send)


@pytest.fixture
def env(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'lifecycle.db'}")
    db.initialize()
    db.create_user("reviewer", "test-only-unusable-hash", "admin", "Reviewer")
    app = FastAPI()
    app.add_middleware(_Identity)
    app.include_router(router, prefix="/api/analysis-tasks")
    app.state.database = db
    app.state.analysis_task_service = Mock()
    app.state.analysis_task_service.get_task.side_effect = lambda task: {"task_id": task} if task == "task-a" else None
    app.state.task_events = Mock()
    app.state.task_releases = Mock()
    app.state.task_training = Mock()
    app.state.task_workflows = Mock()
    app.state.task_workflows.get.return_value = {"task_id": "task-a"}
    app.state.task_training_config = TaskTrainingConfig()
    app.state.config = SimpleNamespace(task_training=app.state.task_training_config)
    app.state.task_training.workflows = app.state.task_workflows
    app.state.task_training.configuration_matches.side_effect = lambda cfg: (
        isinstance(cfg, TaskTrainingConfig) and cfg == app.state.task_training_config)
    state = SimpleNamespace(app=app, client=TestClient(app), events=app.state.task_events,
                            releases=app.state.task_releases, training=app.state.task_training)
    yield state
    state.client.close()
    db.close()


def test_events_are_task_scoped_and_missing_event_is_not_empty_success(env):
    env.events.list.return_value = [{"event_id": "event-a", "mode": "validation"}]
    response = env.client.get("/api/analysis-tasks/task-a/events")
    assert response.status_code == 200
    assert response.json()["data"]["documents"][0]["mode"] == "validation"
    assert env.client.get("/api/analysis-tasks/missing/events").status_code == 404
    env.events.get.return_value = None
    assert env.client.get("/api/analysis-tasks/task-a/events/other-task-event").status_code == 404


@pytest.mark.parametrize("headers", [{"x-test-role": "viewer"}, {"x-test-auth": "api_token"}])
def test_event_review_requires_personal_permission(env, headers):
    response = env.client.post("/api/analysis-tasks/task-a/events/event-a/review",
                               json={"expected_revision": 1, "status": "acknowledged"}, headers=headers)
    assert response.status_code == 403
    env.events.review.assert_not_called()


def test_review_conflict_is_409_and_omitted_notes_remain_none(env):
    env.events.review.side_effect = TaskEventConflict("stale")
    response = env.client.post("/api/analysis-tasks/task-a/events/event-a/review",
                               json={"expected_revision": 1, "status": "resolved"})
    assert response.status_code == 409
    assert env.events.review.call_args.kwargs["notes"] is None


def test_event_evidence_is_private_and_failures_are_explicit(env):
    env.events.load_evidence.return_value = (b"synthetic-not-a-source-video", "image/jpeg")
    response = env.client.get("/api/analysis-tasks/task-a/events/event-a/evidence")
    assert response.status_code == 200 and response.headers["cache-control"] == "private, no-store"
    env.events.load_evidence.side_effect = TaskEventEvidenceUnavailable("private path must not escape")
    failed = env.client.get("/api/analysis-tasks/task-a/events/event-a/evidence")
    assert failed.status_code == 503
    assert "private path" not in failed.text


def test_replay_cannot_advertise_a_video_for_a_keyframe(env):
    env.events.replay_metadata.return_value = {"kind": "keyframe_only", "video_available": False, "ready": True}
    response = env.client.get("/api/analysis-tasks/task-a/events/event-a/replay")
    assert response.json()["data"]["document"]["video_available"] is False


@pytest.mark.parametrize("suffix", ["replay", "evidence"])
def test_event_history_does_not_expand_viewer_replay_permission(env, suffix):
    env.events.replay_metadata.return_value = {"kind": "keyframe_only", "video_available": False}
    env.events.load_evidence.return_value = (b"test-only-image", "image/jpeg")
    path = f"/api/analysis-tasks/task-a/events/event-a/{suffix}"
    assert env.client.get(path, headers={"x-test-role": "viewer"}).status_code == 403
    assert env.client.get(path, headers={"x-test-role": "operator"}).status_code == 200


def test_readiness_is_read_only_and_keeps_production_no_go(env):
    env.releases.readiness.return_value = {"decision": "NO-GO", "production_ready": False}
    response = env.client.get("/api/analysis-tasks/task-a/releases/readiness")
    assert response.status_code == 200
    assert response.json()["data"]["document"]["production_ready"] is False
    env.releases.request.assert_not_called()


def test_release_submission_is_202_not_runtime_applied(env):
    env.releases.request.return_value = {"state": "requested", "production_allowed": False}
    response = env.client.post("/api/analysis-tasks/task-a/releases", json={
        "bundle_id": "bundle-" + "a" * 32, "expected_task_revision": 1,
        "expected_deployment_revision": 0, "evidence_sources": [],
    }, headers={"Idempotency-Key": "release-request-0001"})
    assert response.status_code == 202
    assert response.json()["data"]["document"]["production_allowed"] is False
    assert env.releases.request.call_args.kwargs["actor"] == "reviewer"


def test_unknown_release_commit_is_500_and_same_key_is_retryable(env):
    env.releases.request.side_effect = [RuntimeError("simulated commit acknowledgement unknown"),
                                       {"state": "requested", "release_id": "release-committed"}]
    body = {"bundle_id": "bundle-" + "a" * 32, "expected_task_revision": 1,
            "expected_deployment_revision": 0, "evidence_sources": []}
    headers = {"Idempotency-Key": "release-unknown-0001"}
    path = "/api/analysis-tasks/task-a/releases"
    assert env.client.post(path, json=body, headers=headers).status_code == 500
    assert env.client.post(path, json=body, headers=headers).status_code == 202
    calls = env.releases.request.call_args_list
    assert calls[0] == calls[1]


def test_cross_task_release_review_never_reaches_mutation(env):
    env.releases.get.return_value = {"task_id": "task-b"}
    response = env.client.post("/api/analysis-tasks/task-a/releases/release-b/review", json={
        "expected_revision": 1, "expected_subject_sha256": "a" * 64, "decision": "approve",
    }, headers={"Idempotency-Key": "release-review-0001"})
    assert response.status_code == 404
    env.releases.review.assert_not_called()


@pytest.mark.parametrize("identity", ["api_token", "disabled", "anonymous", "inactive", "demoted", "deleted"])
def test_all_human_lifecycle_writes_recheck_session_identity_before_mutation(env, identity):
    if identity == "inactive":
        assert env.app.state.database.update_user("reviewer", active=False)
    elif identity == "demoted":
        assert env.app.state.database.update_user("reviewer", role="viewer")
    elif identity == "deleted":
        assert env.app.state.database.delete_user("reviewer")
    headers = {"x-test-auth": identity} if identity in {"api_token", "disabled", "anonymous"} else {}
    for suffix, body in (
        ("events/event-a/review", {"expected_revision": 1, "status": "resolved"}),
        ("releases", {"bundle_id": "bundle-" + "a" * 32, "expected_task_revision": 1,
                      "expected_deployment_revision": 0, "evidence_sources": []}),
        ("releases/release-a/review", {"expected_revision": 1, "expected_subject_sha256": "a" * 64,
                                      "decision": "approve"}),
        ("workflows/wf/training-executions", {"expected_revision": 1, "expected_package_sha256": "a" * 64,
                                            "base_model_path": "trusted/base.pt", "base_model_sha256": "b" * 64}),
        ("training-executions/job-a/cancel", {"expected_revision": 1}),
    ):
        response = env.client.post("/api/analysis-tasks/task-a/" + suffix, json=body,
                                   headers={**headers, "Idempotency-Key": "human-boundary-test-0001"})
        assert response.status_code == 403
    env.events.review.assert_not_called()
    env.releases.request.assert_not_called()
    env.releases.review.assert_not_called()
    env.training.submit.assert_not_called()
    env.training.cancel.assert_not_called()


@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_revision_does_not_coerce_client_values(env, value):
    response = env.client.post("/api/analysis-tasks/task-a/events/event-a/review",
                               json={"expected_revision": value, "status": "resolved"})
    assert response.status_code == 422
    env.events.review.assert_not_called()


def test_training_is_task_scoped_personal_and_submission_only(env):
    env.training.submit.return_value = {"job_id": "train-a", "status": "queued", "production_allowed": False}
    payload = {"expected_revision": 3, "expected_package_sha256": "a" * 64,
               "base_model_path": "trusted/base.pt", "base_model_sha256": "b" * 64}
    path = "/api/analysis-tasks/task-a/workflows/workflow-a/training-executions"
    headers = {"Idempotency-Key": "training-submit-0001"}
    assert env.client.post(path, json=payload, headers={**headers, "x-test-auth": "api_token"}).status_code == 403
    response = env.client.post(path, json=payload, headers=headers)
    assert response.status_code == 202 and response.json()["data"]["document"]["status"] == "queued"
    env.app.state.task_workflows.get.return_value = {"task_id": "other"}
    assert env.client.post(path, json=payload, headers=headers).status_code == 404
    assert env.training.submit.call_count == 1


def _capability_service(env, tmp_path):
    from argus.core.task_training import TaskTrainingService

    configuration = TaskTrainingConfig(enabled=True, trusted_base_models=[{
            "model_id": "trusted-base", "name": "Preprovisioned base", "relative_path": "bases/base.pt",
            "artifact_sha256": "a" * 64,
        }])
    service = TaskTrainingService(env.app.state.task_workflows, tmp_path / "models" / "training",
                                  tmp_path / "models", configuration=configuration)
    env.app.state.task_training_config = configuration
    env.app.state.config.task_training = configuration
    env.training.configuration_matches.side_effect = service.configuration_matches
    env.training.capabilities.side_effect = service.capabilities
    return service


def test_training_capabilities_are_typed_read_only_and_missing_runtime_is_disabled(env, tmp_path):
    _capability_service(env, tmp_path)
    response = env.client.get("/api/analysis-tasks/task-a/training-capabilities")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["schema"] == "argus.task-training-capabilities.v1"
    assert data["configured_enabled"] is True
    assert data["runtime"] == {"state": "disabled", "enabled": False, "error_code": None}
    assert data["trusted_base_models"] == [{"model_id": "trusted-base", "name": "Preprovisioned base",
        "relative_path": "bases/base.pt", "artifact_sha256": "a" * 64, "configured_trust": True}]
    assert data["validation_only"] is True and data["production_allowed"] is False
    assert data["resource"]["state"] == "disabled"
    env.training.submit.assert_not_called()
    env.training.prepare.assert_not_called()
    env.training.claim.assert_not_called()
    assert "models_root" not in response.text and str(tmp_path) not in response.text


@pytest.mark.parametrize("role", ["viewer", "operator"])
def test_training_capabilities_require_manage_training_permission(env, tmp_path, role):
    _capability_service(env, tmp_path)
    response = env.client.get("/api/analysis-tasks/task-a/training-capabilities", headers={"x-test-role": role})
    assert response.status_code == 403
    env.training.capabilities.assert_not_called()


def test_training_capabilities_check_task_before_cached_runtime_and_resource_snapshots(env, tmp_path):
    _capability_service(env, tmp_path)
    runtime, admission = Mock(), Mock()
    runtime.coordinator = SimpleNamespace(service=env.training)
    admission.config = env.app.state.task_training_config
    env.app.state.task_training_runtime = runtime
    env.app.state.task_training_admission = admission
    assert env.client.get("/api/analysis-tasks/missing/training-capabilities").status_code == 404
    runtime.snapshot.assert_not_called()
    admission.snapshot.assert_not_called()
    env.training.capabilities.assert_not_called()


def test_training_capabilities_only_use_cached_snapshots_and_strip_private_details(env, tmp_path):
    _capability_service(env, tmp_path)
    runtime = Mock()
    runtime.coordinator = SimpleNamespace(service=env.training)
    runtime.snapshot.return_value = {"state": "waiting_resources", "enabled": True,
        "error_code": "TRAINING_MAINTENANCE_REQUIRED", "exception": "C:/private/runtime"}
    admission = Mock()
    admission.config = env.app.state.task_training_config
    admission.snapshot.return_value = {"state": "waiting_maintenance", "reason_code": "TRAINING_MAINTENANCE_REQUIRED",
        "free_memory_mb": 1024, "device": "cuda:0", "memory_limit_enforced": False, "private": "secret"}
    env.app.state.task_training_runtime = runtime
    env.app.state.task_training_admission = admission
    response = env.client.get("/api/analysis-tasks/task-a/training-capabilities")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["runtime"] == {"state": "waiting_resources", "enabled": True,
                               "error_code": "TRAINING_MAINTENANCE_REQUIRED"}
    assert data["resource"]["state"] == "waiting_maintenance"
    assert data["resource"]["free_memory_mb"] == 1024
    assert "private" not in response.text and "secret" not in response.text
    assert runtime.method_calls == [("snapshot", (), {})]
    assert admission.method_calls == [("snapshot", (), {})]


def test_training_capabilities_snapshot_failure_is_safe_and_explicit(env, tmp_path):
    _capability_service(env, tmp_path)
    env.app.state.task_training_runtime = SimpleNamespace(coordinator=SimpleNamespace(service=env.training),
        snapshot=Mock(side_effect=RuntimeError("C:/private")))
    env.app.state.task_training_admission = SimpleNamespace(config=env.app.state.task_training_config,
        snapshot=Mock(side_effect=RuntimeError("secret")))
    response = env.client.get("/api/analysis-tasks/task-a/training-capabilities")
    assert response.status_code == 200
    assert response.json()["data"]["runtime"] == {"state": "error", "enabled": False,
        "error_code": "TRAINING_RUNTIME_SNAPSHOT_UNAVAILABLE"}
    assert response.json()["data"]["resource"]["reason_code"] == "TRAINING_RESOURCE_SNAPSHOT_UNAVAILABLE"
    assert "private" not in response.text and "secret" not in response.text


def test_training_capabilities_refuse_runtime_bound_to_another_service(env, tmp_path):
    _capability_service(env, tmp_path)
    env.app.state.task_training_runtime = SimpleNamespace(
        coordinator=SimpleNamespace(service=object()),
        snapshot=Mock(return_value={"state": "running", "enabled": True, "error_code": None}))
    response = env.client.get("/api/analysis-tasks/task-a/training-capabilities")
    assert response.status_code == 503
    env.app.state.task_training_runtime.snapshot.assert_not_called()
    env.training.capabilities.assert_not_called()


def test_training_submission_refuses_workflow_service_instance_drift(env):
    env.training.workflows = object()
    env.training.submit.return_value = {"job_id": "train-a", "status": "queued", "production_allowed": False}
    response = env.client.post("/api/analysis-tasks/task-a/workflows/workflow-a/training-executions",
        json={"expected_revision": 1, "expected_package_sha256": "a" * 64,
              "base_model_path": "bases/base.pt", "base_model_sha256": "b" * 64},
        headers={"Idempotency-Key": "training-instance-drift"})
    assert response.status_code == 503
    env.training.submit.assert_not_called()


def test_training_submission_rechecks_personal_identity_after_scope_lookup(env):
    def scope_then_disable(task_id):
        env.app.state.database.update_user("reviewer", active=False)
        return {"task_id": task_id}
    env.app.state.analysis_task_service.get_task.side_effect = scope_then_disable
    env.training.submit.return_value = {"job_id": "train-a", "status": "queued", "production_allowed": False}
    response = env.client.post("/api/analysis-tasks/task-a/workflows/workflow-a/training-executions",
        json={"expected_revision": 1, "expected_package_sha256": "a" * 64,
              "base_model_path": "bases/base.pt", "base_model_sha256": "b" * 64},
        headers={"Idempotency-Key": "training-account-revoked"})
    assert response.status_code == 403
    env.training.submit.assert_not_called()


@pytest.mark.parametrize("operation", ["submit", "cancel"])
def test_training_writes_recheck_identity_after_bound_workflow_or_job_lookup(env, operation):
    def read_scope_then_disable(_identifier):
        env.app.state.database.update_user("reviewer", active=False)
        return {"task_id": "task-a"}
    if operation == "submit":
        env.app.state.task_workflows.get.side_effect = read_scope_then_disable
        path = "/api/analysis-tasks/task-a/workflows/workflow-a/training-executions"
        payload = {"expected_revision": 1, "expected_package_sha256": "a" * 64,
                   "base_model_path": "bases/base.pt", "base_model_sha256": "b" * 64}
    else:
        env.training.get.side_effect = read_scope_then_disable
        path = "/api/analysis-tasks/task-a/training-executions/job-a/cancel"
        payload = {"expected_revision": 1}
    getattr(env.training, operation).return_value = {"job_id": "train-a", "status": "queued"}
    response = env.client.post(path, json=payload, headers={"Idempotency-Key": "training-bound-scope-revoked"})
    assert response.status_code == 403
    env.training.submit.assert_not_called()
    env.training.cancel.assert_not_called()


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_training_capabilities_have_no_http_trust_mutation_route(env, method):
    response = getattr(env.client, method)("/api/analysis-tasks/task-a/training-capabilities")
    assert response.status_code == 405
    env.training.capabilities.assert_not_called()
    env.training.submit.assert_not_called()


def test_unknown_training_submit_is_500_and_retains_request_identity(env):
    env.training.submit.side_effect = [RuntimeError("unconfirmed training commit"),
                                      {"job_id": "same-committed-job", "status": "queued"}]
    path = "/api/analysis-tasks/task-a/workflows/wf/training-executions"
    body = {"expected_revision": 3, "expected_package_sha256": "a" * 64,
            "base_model_path": "trusted/base.pt", "base_model_sha256": "b" * 64}
    headers = {"Idempotency-Key": "training-unknown-0001"}
    assert env.client.post(path, json=body, headers=headers).status_code == 500
    assert env.client.post(path, json=body, headers=headers).status_code == 202
    calls = env.training.submit.call_args_list
    assert calls[0] == calls[1]


def test_candidate_unknown_registration_commit_is_500_and_same_body_reconciles(env):
    from argus.dashboard.routes.task_workflows import router as workflows_router

    env.app.include_router(workflows_router, prefix="/api/analysis-tasks")
    env.app.state.task_workflows.import_candidate.side_effect = [RuntimeError("synthetic registration acknowledgement lost"),
                                                               {"state": "candidate_registered", "revision": 5}]
    path = "/api/analysis-tasks/task-a/workflows/wf/candidate"
    body = {"expected_revision": 3, "expected_package_sha256": "a" * 64,
            "candidate": {"relative_path": "candidate.onnx", "artifact_sha256": "b" * 64,
                          "descriptor": {"name": "frozen candidate"}}}
    failed = env.client.post(path, json=body)
    assert failed.status_code == 500
    assert "synthetic registration" not in failed.text
    assert env.client.post(path, json=body).status_code == 200
    calls = env.app.state.task_workflows.import_candidate.call_args_list
    assert calls[0].args == calls[1].args == ("wf",)
    assert {key: value for key, value in calls[0].kwargs.items() if key != "authorize"} == {
        key: value for key, value in calls[1].kwargs.items() if key != "authorize"
    } == {"actor": "reviewer", **body}
    assert all(callable(call.kwargs["authorize"]) and call.kwargs["authorize"]() is True for call in calls)


@pytest.mark.parametrize("kind", ["conflict", "blocked"])
def test_candidate_known_domain_conflict_is_not_misreported_as_unknown_commit(env, kind):
    from argus.dashboard.routes.task_workflows import router as workflows_router
    from argus.storage.task_workflows import TaskWorkflowBlocked, TaskWorkflowConflict

    env.app.include_router(workflows_router, prefix="/api/analysis-tasks")
    error = TaskWorkflowConflict("revision changed") if kind == "conflict" else TaskWorkflowBlocked("APPROVAL_REQUIRED")
    env.app.state.task_workflows.import_candidate.side_effect = error
    response = env.client.post("/api/analysis-tasks/task-a/workflows/wf/candidate", json={
        "expected_revision": 3, "expected_package_sha256": "a" * 64, "candidate": {},
    })
    assert response.status_code == 409


@pytest.mark.parametrize("change", ["inactive", "demoted", "deleted", "instance_changed"])
def test_candidate_rechecks_personal_identity_after_workflow_scope_read(env, change):
    from argus.dashboard.routes.task_workflows import router as workflows_router

    env.app.include_router(workflows_router, prefix="/api/analysis-tasks")
    original = env.app.state.task_workflows

    def scope(_workflow_id):
        if change == "inactive":
            env.app.state.database.update_user("reviewer", active=False)
        elif change == "demoted":
            env.app.state.database.update_user("reviewer", role="viewer")
        elif change == "deleted":
            env.app.state.database.delete_user("reviewer")
        else:
            env.app.state.task_workflows = Mock()
        return {"task_id": "task-a"}

    original.get.side_effect = scope
    response = env.client.post("/api/analysis-tasks/task-a/workflows/wf/candidate", json={
        "expected_revision": 3, "expected_package_sha256": "a" * 64, "candidate": {},
    })
    assert response.status_code == (503 if change == "instance_changed" else 403)
    original.import_candidate.assert_not_called()


def test_annotation_return_is_suggestion_not_human_review(env):
    env.app.state.task_workflows.import_annotation_exchange.return_value = {"state": "reviewing", "production_allowed": False}
    response = env.client.post("/api/analysis-tasks/task-a/workflows/wf/annotation-import",
                               json={"expected_revision": 2, "payload": {"schema": "v2"}},
                               headers={"x-test-auth": "api_token"})
    assert response.status_code == 200
    env.app.state.task_workflows.review.assert_not_called()


def test_real_app_assembles_lifecycle_without_starting_training(tmp_path):
    from argus.config.schema import ArgusConfig, StorageConfig
    from argus.dashboard.app import create_app

    db = Database(f"sqlite:///{tmp_path / 'assembled.db'}")
    db.initialize()
    config = ArgusConfig(storage=StorageConfig(
        models_dir=tmp_path / "models", baselines_dir=tmp_path / "baselines",
    ))
    app = create_app(database=db, config=config, alerts_dir=str(tmp_path / "alerts"))
    try:
        assert app.state.task_events.snapshot()["state"] == "not_started"
        assert app.state.task_releases is not None
        assert app.state.task_training.trusted_base_models == {}
        # There is no implicit scheduler, worker construction or model import.
        with TestClient(app) as client:
            assert app.state.task_events.snapshot()["state"] == "running"
            assert client.get("/api/analysis-tasks/missing/events").status_code == 404
            assert client.get("/api/analysis-tasks/missing/training-executions").status_code == 404
        assert app.state.task_events.snapshot()["state"] == "stopped"
    finally:
        db.close()


@pytest.mark.parametrize("outcome", [False, "raises"])
def test_event_worker_start_failure_does_not_abort_dashboard_or_start_training(tmp_path, monkeypatch, outcome):
    from argus.config.schema import ArgusConfig, StorageConfig
    from argus.dashboard.app import create_app

    db = Database(f"sqlite:///{tmp_path / 'worker-start-failure.db'}")
    db.initialize()
    config = ArgusConfig(storage=StorageConfig(
        models_dir=tmp_path / "models", baselines_dir=tmp_path / "baselines",
    ))
    app = create_app(database=db, config=config, alerts_dir=str(tmp_path / "alerts"))
    start = Mock(side_effect=RuntimeError("synthetic worker start refusal")) if outcome == "raises" else Mock(return_value=False)
    monkeypatch.setattr(app.state.task_events, "start", start)
    try:
        with TestClient(app) as client:
            assert client.get("/api/analysis-tasks/missing/events").status_code == 404
            assert client.get("/api/analysis-tasks/missing/training-executions").status_code == 404
            assert app.state.task_events.snapshot()["state"] == "not_started"
            assert app.state.task_training.trusted_base_models == {}
        start.assert_called_once()
        assert app.state.task_events.snapshot()["state"] == "stopped"
    finally:
        db.close()
