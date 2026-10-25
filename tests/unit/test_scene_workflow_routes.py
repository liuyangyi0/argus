"""Scene workflow HTTP contracts keep RBAC, actor binding and release separation."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import TypeAdapter, ValidationError
from starlette.requests import Request

from argus.dashboard.routes import scene_workflow as routes
from argus.config.schema import SceneGuardConfig
from argus.storage.scene_workflow import SceneWorkflowConflict


def _request(tmp_path, *, role="engineer", username="authenticated-operator", camera=True):
    database = Mock()
    database.get_user.return_value = SimpleNamespace(active=True, role=role)
    config = SimpleNamespace(
        storage=SimpleNamespace(baselines_dir=tmp_path / "baselines"),
        cameras=[SimpleNamespace(
            camera_id="cam1", scene_guard=SceneGuardConfig(profiles_dir=str(tmp_path / "profiles")),
        )] if camera else [],
    )
    return Request({
        "type": "http", "method": "POST", "path": "/api/scene-workflows/cam1",
        "headers": [], "state": {
            "user": {"username": username, "role": role}, "auth_method": "session",
        },
        "app": SimpleNamespace(state=SimpleNamespace(
            config=config, database=database, task_manager=Mock(), ws_manager=Mock(),
        )),
    })


@pytest.fixture
def contract_service(monkeypatch):
    service = Mock()
    service.create.return_value = {"workflow_id": "scene1", "state": "draft"}
    service.start_baseline.return_value = "baseline-task"
    service.submit_training.return_value = "training-job"
    service.approve_profile.return_value = "scene1"
    service.activate_profile.return_value = "scene1"
    service.snapshot.return_value = {"camera_id": "cam1", "workflows": [], "profiles": []}
    factory = Mock(return_value=service)
    monkeypatch.setattr(routes, "SceneWorkflowService", factory)
    monkeypatch.setattr(routes, "get_registry", Mock(return_value=object()))
    return service, factory


def _create_body():
    return routes.SceneCreateRequest(
        workflow_id="scene1", reference_path="default/v001/reference.png",
        reference_sha256="a" * 64, normal_reference_confirmed=True,
    )


def _baseline_body():
    return routes.StrictBaselineRequest(sources=[
        {"video": "normal1.mp4", "annotations": "normal1.json"},
        {"video": "normal2.mp4", "annotations": "normal2.json"},
    ])


def _mutations():
    return [
        (routes.create_scene_workflow, lambda: (_create_body(),)),
        (routes.create_scene_baseline, lambda: ("scene1", _baseline_body())),
        (routes.create_scene_training, lambda: ("scene1", routes.SceneTrainingRequest())),
        (routes.approve_scene_profile, lambda: ("scene1", routes.ProfileApprovalRequest(operator_approved=True))),
        (routes.select_scene_profile, lambda: ("scene1", routes.ProfileSelectionRequest(
            acknowledge_runtime_unchanged=True, expected_active_profile_id=None, expected_active_profile_revision=None,
        ))),
    ]


@pytest.mark.parametrize("role", ["viewer", "operator"])
def test_read_only_roles_cannot_mutate_scenes_training_or_approval(tmp_path, contract_service, role):
    service, factory = contract_service
    for endpoint, args in _mutations():
        request = _request(tmp_path, role=role)
        response = endpoint(request, "cam1", *args())
        assert response.status_code == 403
        assert json.loads(response.body)["code"] != 0
        request.app.state.ws_manager.broadcast.assert_not_called()
    factory.assert_not_called()
    assert service.mock_calls == []


def test_read_only_snapshot_does_not_grant_scene_mutation_permission(tmp_path, contract_service):
    service, _factory = contract_service
    request = _request(tmp_path, role="viewer")
    response = routes.scene_workflow_snapshot(request, "cam1")
    assert response.status_code == 200
    service.snapshot.assert_called_once_with("cam1")
    request.app.state.ws_manager.broadcast.assert_not_called()


def test_service_cache_changes_when_camera_policy_is_reloaded(tmp_path, contract_service):
    _service, factory = contract_service
    request = _request(tmp_path)
    routes.scene_workflow_snapshot(request, "cam1")
    routes.scene_workflow_snapshot(request, "cam1")
    assert factory.call_count == 1
    camera = request.app.state.config.cameras[0]
    old_policy = camera.scene_guard
    camera.scene_guard = old_policy.model_copy(update={"min_matches": old_policy.min_matches + 1})
    routes.scene_workflow_snapshot(request, "cam1")
    assert factory.call_count == 2
    assert factory.call_args_list[0].kwargs["scene_guard_policy"].min_matches == 40
    assert factory.call_args_list[1].kwargs["scene_guard_policy"].min_matches == 41


def test_actor_is_server_authenticated_identity_and_not_a_request_field(tmp_path, contract_service):
    service, _factory = contract_service
    request = _request(tmp_path)
    response = routes.create_scene_workflow(request, "cam1", _create_body())
    assert response.status_code == 201
    service.create.assert_called_once_with(
        "cam1", actor="authenticated-operator", workflow_id="scene1",
        reference_path="default/v001/reference.png", reference_sha256="a" * 64,
        normal_reference_confirmed=True,
    )
    request.app.state.ws_manager.broadcast.assert_called_once_with(
        "models", {"type": "scene_workflow.changed", "camera_id": "cam1", "workflow_id": "scene1"},
        event_type="scene_workflow.changed", resource_id="cam1:scene1",
    )
    for actor_field in ("actor", "approved_by", "created_by"):
        with pytest.raises(ValidationError):
            routes.SceneCreateRequest.model_validate({**_create_body().model_dump(), actor_field: "forged-admin"})


@pytest.mark.parametrize("model,field,other", [
    (routes.SceneCreateRequest, "normal_reference_confirmed", {
        "workflow_id": "scene1", "reference_path": "default/v1/reference.png", "reference_sha256": "a" * 64,
    }),
    (routes.ProfileApprovalRequest, "operator_approved", {}),
    (routes.ProfileSelectionRequest, "acknowledge_runtime_unchanged", {
        "expected_active_profile_id": None, "expected_active_profile_revision": None,
    }),
])
@pytest.mark.parametrize("value", [False, 0, 1, 1.0, "true", "True", "1", None, [], {}])
def test_confirmation_requires_literal_json_true_not_truthy_or_coerced_values(model, field, other, value):
    with pytest.raises(ValidationError):
        model.model_validate({**other, field: value})
    assert model.model_validate({**other, field: True}).model_dump()[field] is True


@pytest.mark.parametrize("value", ["../camera", "camera/child", "camera\\child", "C:camera", "", ".hidden", "a" * 49])
def test_camera_and_workflow_identifiers_cannot_be_filesystem_paths(value):
    with pytest.raises(ValidationError):
        TypeAdapter(routes.Identifier).validate_python(value)


@pytest.mark.parametrize("model,valid,override", [
    (routes.StrictBaselineRequest, {"sources": [
        {"video": "a.mp4", "annotations": "a.json"}, {"video": "b.mp4", "annotations": "b.json"},
    ]}, {"strict_release_decoder": False}),
    (routes.StrictBaselineRequest, {"sources": [
        {"video": "a.mp4", "annotations": "a.json"}, {"video": "b.mp4", "annotations": "b.json"},
    ]}, {"allow_alignable": True}),
    (routes.SceneTrainingRequest, {}, {"baseline_path": "/arbitrary/server/path"}),
    (routes.SceneTrainingRequest, {}, {"confirmation_required": False}),
    (routes.ProfileApprovalRequest, {"operator_approved": True}, {"actor": "forged-admin"}),
    (routes.ProfileSelectionRequest, {
        "acknowledge_runtime_unchanged": True, "expected_active_profile_id": None,
        "expected_active_profile_revision": None,
    }, {"stage": "production"}),
])
def test_request_bodies_cannot_override_strict_gates_authority_or_server_paths(model, valid, override):
    with pytest.raises(ValidationError):
        model.model_validate({**valid, **override})


def test_unknown_camera_and_unavailable_backend_do_not_construct_a_service(tmp_path, contract_service, monkeypatch):
    _service, factory = contract_service
    request = _request(tmp_path, camera=False)
    assert routes.create_scene_workflow(request, "cam1", _create_body()).status_code == 404
    factory.assert_not_called()
    request = _request(tmp_path)
    request.app.state.database = None
    assert routes.create_scene_workflow(request, "cam1", _create_body()).status_code == 503
    factory.assert_not_called()
    request = _request(tmp_path)
    monkeypatch.setattr(routes, "get_registry", Mock(return_value=None))
    assert routes.create_scene_workflow(request, "cam1", _create_body()).status_code == 503
    factory.assert_not_called()


def test_baseline_job_keeps_strict_sources_and_task_metadata_server_resolved(tmp_path, contract_service):
    service, _factory = contract_service
    request = _request(tmp_path)
    response = routes.create_scene_baseline(request, "cam1", "scene1", _baseline_body())
    assert response.status_code == 202
    assert json.loads(response.body)["data"] == {"task_id": "baseline-task"}
    service.start_baseline.assert_called_once_with(
        "cam1", "scene1", actor="authenticated-operator", task_manager=request.app.state.task_manager,
        **_baseline_body().model_dump(),
    )


def test_missing_task_service_cannot_leave_a_baseline_submission(tmp_path, contract_service):
    service, _factory = contract_service
    request = _request(tmp_path)
    request.app.state.task_manager = None
    response = routes.create_scene_baseline(request, "cam1", "scene1", _baseline_body())
    assert response.status_code == 503
    service.start_baseline.assert_not_called()
    request.app.state.ws_manager.broadcast.assert_not_called()


def test_training_creation_reports_pending_confirmation_not_a_release(tmp_path, contract_service):
    service, _factory = contract_service
    request = _request(tmp_path)
    response = routes.create_scene_training(request, "cam1", "scene1", routes.SceneTrainingRequest())
    assert response.status_code == 201
    assert json.loads(response.body)["data"] == {"job_id": "training-job", "status": "pending_confirmation"}
    service.submit_training.assert_called_once_with(
        "cam1", "scene1", actor="authenticated-operator", model_type="patchcore", image_size=256,
    )
    service.approve_profile.assert_not_called()
    service.activate_profile.assert_not_called()


def test_approval_and_selection_are_separate_explicit_operations(tmp_path, contract_service):
    service, _factory = contract_service
    request = _request(tmp_path)
    response = routes.approve_scene_profile(
        request, "cam1", "scene1", routes.ProfileApprovalRequest(operator_approved=True),
    )
    assert response.status_code == 200
    service.approve_profile.assert_called_once_with("cam1", "scene1", actor="authenticated-operator")
    service.activate_profile.assert_not_called()
    response = routes.select_scene_profile(
        request, "cam1", "scene1", routes.ProfileSelectionRequest(
            acknowledge_runtime_unchanged=True, expected_active_profile_id=None, expected_active_profile_revision=None,
        ),
    )
    assert response.status_code == 200
    service.activate_profile.assert_called_once_with(
        "cam1", "scene1", actor="authenticated-operator",
        expected_active_profile_id=None, expected_active_profile_revision=None,
    )
    assert not hasattr(request.app.state, "model_release_service")


@pytest.mark.parametrize("auth_method", [None, "disabled", "api_token", "anonymous"])
def test_profile_approval_requires_a_personal_account_session(tmp_path, contract_service, auth_method):
    service, factory = contract_service
    request = _request(tmp_path)
    request.state.auth_method = auth_method
    response = routes.approve_scene_profile(
        request, "cam1", "scene1", routes.ProfileApprovalRequest(operator_approved=True),
    )
    assert response.status_code == 403
    factory.assert_not_called()
    service.approve_profile.assert_not_called()
    request.app.state.ws_manager.broadcast.assert_not_called()


@pytest.mark.parametrize("record", [None, SimpleNamespace(active=False, role="admin"),
                                    SimpleNamespace(active=True, role="viewer")])
def test_profile_approval_rechecks_current_account_and_role(tmp_path, contract_service, record):
    service, factory = contract_service
    request = _request(tmp_path, role="admin")
    request.app.state.database.get_user.return_value = record
    response = routes.approve_scene_profile(
        request, "cam1", "scene1", routes.ProfileApprovalRequest(operator_approved=True),
    )
    assert response.status_code == 403
    factory.assert_not_called()
    service.approve_profile.assert_not_called()


@pytest.mark.parametrize("username", ["api", "system"])
def test_profile_approval_does_not_confuse_named_accounts_with_machine_credentials(
    tmp_path, contract_service, username,
):
    service, _factory = contract_service
    request = _request(tmp_path, username=username)
    response = routes.approve_scene_profile(
        request, "cam1", "scene1", routes.ProfileApprovalRequest(operator_approved=True),
    )
    assert response.status_code == 200
    service.approve_profile.assert_called_once_with("cam1", "scene1", actor=username)


@pytest.mark.parametrize("failure,status", [
    (SceneWorkflowConflict("busy"), 409), (FileExistsError("immutable profile exists"), 409),
    (FileNotFoundError("missing source"), 404), (ValueError("invalid input"), 400),
])
def test_failed_mutation_does_not_publish_success_or_hide_its_error(tmp_path, contract_service, failure, status):
    service, _factory = contract_service
    service.create.side_effect = failure
    request = _request(tmp_path)
    response = routes.create_scene_workflow(request, "cam1", _create_body())
    assert response.status_code == status
    assert json.loads(response.body)["code"] != 0
    request.app.state.ws_manager.broadcast.assert_not_called()


def test_event_failure_does_not_turn_a_durable_scene_mutation_into_a_retry(tmp_path, contract_service):
    _service, _factory = contract_service
    request = _request(tmp_path)
    request.app.state.ws_manager.broadcast.side_effect = RuntimeError("transport unavailable")
    response = routes.create_scene_training(request, "cam1", "scene1", routes.SceneTrainingRequest())
    assert response.status_code == 201
    assert json.loads(response.body)["data"]["job_id"] == "training-job"


@pytest.mark.parametrize("missing", ["expected_active_profile_id", "expected_active_profile_revision"])
def test_selection_requires_explicit_observed_pointer_identity(missing):
    body = {
        "acknowledge_runtime_unchanged": True,
        "expected_active_profile_id": None, "expected_active_profile_revision": None,
    }
    del body[missing]
    with pytest.raises(ValidationError):
        routes.ProfileSelectionRequest.model_validate(body)


def test_selection_forwards_observed_pointer_revision_not_just_profile_id(tmp_path, contract_service):
    service, _factory = contract_service
    response = routes.select_scene_profile(
        _request(tmp_path), "cam1", "scene1", routes.ProfileSelectionRequest(
            acknowledge_runtime_unchanged=True,
            expected_active_profile_id="legacy-profile", expected_active_profile_revision="f" * 64,
        ),
    )
    assert response.status_code == 200
    service.activate_profile.assert_called_once_with(
        "cam1", "scene1", actor="authenticated-operator",
        expected_active_profile_id="legacy-profile", expected_active_profile_revision="f" * 64,
    )


@pytest.mark.parametrize("actor", ["", "   ", "unknown", " Unknown "])
def test_scene_mutations_require_a_named_authenticated_actor(tmp_path, contract_service, actor):
    service, factory = contract_service
    for endpoint, args in _mutations():
        request = _request(tmp_path)
        request.state.user["username"] = actor
        assert endpoint(request, "cam1", *args()).status_code == 403
    factory.assert_not_called()
    service.create.assert_not_called()
    service.approve_profile.assert_not_called()
    service.activate_profile.assert_not_called()
