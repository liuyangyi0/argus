"""Ordinary, mocked API checks: no camera, decoding, training or inference."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from argus.dashboard.routes import analysis_tasks as routes


def _request(*, service=None, manager=None, role="viewer"):
    return Request({
        "type": "http", "method": "GET", "path": "/api/analysis-tasks",
        "headers": [], "state": {"user": {"role": role, "username": "reader"}},
        "app": SimpleNamespace(state=SimpleNamespace(
            database=None, analysis_task_service=service, camera_manager=manager,
        )),
    })


def _task():
    return {
        "task_id": "task-1", "camera_id": "cam1", "template_id": "legacy_anomaly.v1",
        "authority": "legacy", "current_revision": 1, "config_sha256": "a" * 64,
        "created_at": "2026-10-23T00:00:00", "config": {
            "source": "rtsp://private:password@camera/private", "model_path": "C:/private/model.pt",
        },
        "deployment": {
            "authority": "legacy", "desired_artifact_id": "artifact-1",
            "applied_artifact_id": None, "runtime_status": "unknown",
            "registry_release_revision": 2, "projection_status": "authorized",
            "model_path": "C:/private/model.pt", "active": True,
        },
    }


def _service():
    service = Mock()
    service.list_tasks.return_value = [_task()]
    service.get_task.return_value = _task()
    return service


def test_list_is_read_only_and_does_not_expose_config_or_model_paths():
    service = _service()
    response = routes.analysis_tasks_list(_request(service=service), camera_id="cam1")
    assert response.status_code == 200
    body = json.loads(response.body)
    assert set(body) == {"code", "msg", "data"}
    task = body["data"]["tasks"][0]
    assert task["config"] is None
    assert b"private" not in response.body
    assert b"password" not in response.body
    assert task["deployment"]["desired_artifact_id"] == "artifact-1"
    assert task["deployment"]["applied_artifact_id"] is None
    assert task["runtime"]["state"] == "unknown"
    service.list_tasks.assert_called_once_with(camera_id="cam1")
    service.bootstrap_legacy.assert_not_called()
    service.sync_legacy_model_projection.assert_not_called()


def test_runtime_and_source_are_independent_and_do_not_infer_model_application():
    manager = Mock()
    manager.get_analysis_task_runtime.return_value = {
        "state": "stopped", "running": False, "generation": 7,
        "reason_code": "MANUAL_STOP", "model_version": "v003",
        "artifact_sha256": "b" * 64, "model_identity_verified": True,
        "model_path": "C:/private/model.pt",
    }
    manager.get_source_runtime_status.return_value = {
        "state": "running", "running": True, "connected": True, "generation": 4,
        "source": "rtsp://private:password@camera/stream",
    }
    response = routes.analysis_task_detail(_request(service=_service(), manager=manager), "task-1")
    task = json.loads(response.body)["data"]
    assert task["runtime"]["state"] == "stopped"
    assert task["source"]["state"] == "running"
    assert task["runtime"]["model_identity_verified"] is True
    assert task["deployment"]["applied_artifact_id"] is None
    assert b"private" not in response.body
    manager.get_analysis_task_runtime.assert_called_once_with("task-1")
    manager.get_source_runtime_status.assert_called_once_with("cam1")


def test_runtime_snapshot_failure_is_unknown_not_false_healthy():
    manager = Mock()
    manager.get_analysis_task_runtime.side_effect = RuntimeError("C:/private/model.pt")
    manager.get_source_runtime_status.return_value = {"state": "running", "running": True}
    response = routes.analysis_task_detail(_request(service=_service(), manager=manager), "task-1")
    assert response.status_code == 200
    task = json.loads(response.body)["data"]
    assert task["runtime"]["state"] == "unknown"
    assert task["runtime"]["running"] is None
    assert task["source"]["state"] == "running"
    assert b"private" not in response.body


@pytest.mark.parametrize("role", ["invalid", None])
def test_denied_read_does_not_touch_storage_or_runtime(role):
    service = _service()
    manager = Mock()
    request = _request(service=service, manager=manager, role=role)
    for response in (
        routes.analysis_tasks_list(request), routes.analysis_task_detail(request, "task-1"),
        routes.analysis_task_templates(request),
    ):
        assert response.status_code == 403
        assert json.loads(response.body)["code"] == 40300
    assert service.mock_calls == []
    assert manager.mock_calls == []


def test_missing_task_and_unavailable_storage_are_not_empty_successes():
    service = _service()
    service.get_task.return_value = None
    response = routes.analysis_task_detail(_request(service=service), "absent")
    assert response.status_code == 404
    assert json.loads(response.body)["code"] == 40400
    assert routes.analysis_tasks_list(_request()).status_code == 503
    assert routes.analysis_task_detail(_request(), "task-1").status_code == 503


def test_storage_error_is_enveloped_without_leaking_exception():
    service = _service()
    service.list_tasks.side_effect = RuntimeError("rtsp://private:password@camera/stream")
    response = routes.analysis_tasks_list(_request(service=service))
    assert response.status_code == 500
    assert json.loads(response.body)["code"] == 50000
    assert b"password" not in response.body


def test_managed_templates_are_shadow_only_and_legacy_remains_not_creatable():
    response = routes.analysis_task_templates(_request())
    templates = json.loads(response.body)["data"]["templates"]
    assert {item["template_id"] for item in templates} == {
        "legacy_anomaly.v1", "object_detection.v1", "pose_estimation.v1",
        "ppe_compliance.v1", "ppe_compliance_dual.v1", "fast_motion.v1",
    }
    legacy = next(item for item in templates if item["template_id"] == "legacy_anomaly.v1")
    assert legacy["supports_creation"] is False
    assert legacy["deployment_authority"] == "legacy"
    assert all(item["availability"] == "shadow" and item["supports_creation"]
               for item in templates if item is not legacy)
    dual = next(item for item in templates if item["template_id"] == "ppe_compliance_dual.v1")
    assert dual["model_roles"] == {"person_detector": "onnx_detection.v1", "head_detector": "onnx_detection.v1"}
    assert dual["model_role_requirements"] == {
        "person_detector": {"required_labels": ["person"]},
        "head_detector": {"required_labels": ["helmet", "bare_head"]},
    }
    assert all(route.methods == {"GET"} for route in routes.router.routes)


def test_openapi_describes_concrete_envelope_and_unknown_detail_stays_404():
    service = _service()
    service.get_task.return_value = None
    app = FastAPI()
    app.state.analysis_task_service = service
    app.state.camera_manager = None

    class ReaderIdentity:
        def __init__(self, inner):
            self.inner = inner

        async def __call__(self, scope, receive, send):
            scope.setdefault("state", {})["user"] = {"role": "viewer"}
            await self.inner(scope, receive, send)

    app.include_router(routes.router, prefix="/api/analysis-tasks")
    app.include_router(routes.templates_router, prefix="/api/analysis-task-templates")
    with TestClient(ReaderIdentity(app)) as client:
        response = client.get("/api/analysis-tasks/absent")
        assert response.status_code == 404
        assert response.json()["code"] == 40400
    schema = app.openapi()
    assert "AnalysisTaskResponse" in schema["components"]["schemas"]
    assert "AnalysisTaskSourceResponse" in schema["components"]["schemas"]


def test_real_storage_payload_matches_public_dto_without_config_leak(tmp_path):
    from argus.storage.analysis_tasks import AnalysisTaskService
    from argus.storage.database import Database

    db = Database(f"sqlite:///{tmp_path / 'tasks-api.db'}")
    db.initialize()
    try:
        service = AnalysisTaskService(db)
        created = service.bootstrap_legacy("cam1", {"source": "rtsp://private:password@camera"})
        response = routes.analysis_task_detail(_request(service=service), created["task_id"])
        assert response.status_code == 200
        data = json.loads(response.body)["data"]
        assert data["deployment"]["projection_status"] == "not_synced"
        assert data["deployment"]["applied_artifact_id"] is None
        assert data["config_sha256"] == created["config_sha256"]
        assert b"password" not in response.body
    finally:
        db.close()


@pytest.mark.parametrize(("state", "reason"), [
    ("waiting_source", "WAITING_SOURCE"),
    ("sampling_insufficient", "SAMPLING_INSUFFICIENT"),
    ("cleanup_pending", "CLEANUP_PENDING"),
    ("runtime_dead", "RUNTIME_DEAD"),
    ("cleanup_failed", "CLEANUP_FAILED"),
    ("draining", "DRAINING"),
])
def test_runtime_degradation_states_are_preserved_without_raw_exception(state, reason):
    manager = Mock()
    manager.get_analysis_task_runtime.return_value = {
        "state": state, "reason": "rtsp://private:password@camera", "reason_code": "C:/private/file",
    }
    manager.get_source_runtime_status.return_value = {
        "state": "cleanup_pending", "error": "C:/private/model.pt",
    }
    response = routes.analysis_task_detail(_request(service=_service(), manager=manager), "task-1")
    assert response.status_code == 200
    data = json.loads(response.body)["data"]
    assert data["runtime"]["state"] == state
    assert data["runtime"]["reason_code"] == reason
    assert data["source"]["state"] == "cleanup_pending"
    assert data["source"]["reason_code"] == "CLEANUP_PENDING"
    assert b"private" not in response.body


def test_stale_registry_projection_is_not_mapped_to_authorized_or_error():
    service = _service()
    service.get_task.return_value["deployment"]["projection_status"] = "stale"
    response = routes.analysis_task_detail(_request(service=service), "task-1")
    assert response.status_code == 200
    assert json.loads(response.body)["data"]["deployment"]["projection_status"] == "stale"


def _startup_fixture(tmp_path):
    from argus.config.schema import ArgusConfig, CameraConfig
    from argus.storage.database import Database

    database = Database(f"sqlite:///{tmp_path / 'startup-api.db'}")
    database.initialize()
    config = ArgusConfig(cameras=[
        CameraConfig(camera_id="cam1", name="First", source="mock://first"),
        CameraConfig(camera_id="cam2", name="Second", source="mock://second"),
    ])
    config.dashboard.go2rtc_enabled = False
    config.auth.enabled = False
    return database, config


def _app_request(app):
    return Request({
        "type": "http", "method": "GET", "path": "/api/analysis-tasks",
        "headers": [], "state": {"user": {"role": "viewer", "username": "reader"}},
        "app": app,
    })


def test_startup_projection_conflict_does_not_prevent_dashboard_or_other_cameras(tmp_path, monkeypatch):
    from argus.dashboard import app as app_module
    from argus.storage.analysis_tasks import AnalysisTaskProjectionConflict, AnalysisTaskService

    db, config = _startup_fixture(tmp_path)
    calls = []
    logger = Mock()
    monkeypatch.setattr(app_module, "logger", logger)

    def project(_service, camera_id):
        calls.append(camera_id)
        if camera_id == "cam1":
            raise AnalysisTaskProjectionConflict("rtsp://private:password@camera")

    monkeypatch.setattr(AnalysisTaskService, "sync_legacy_model_projection", project)
    try:
        # No lifespan is entered: no camera/source, background worker or model
        # is started. This tests only application assembly plus SQLite metadata.
        app = app_module.create_app(database=db, config=config)
        assert app.state.analysis_task_bootstrap_errors == set()
        assert calls == ["cam1", "cam2"]
        response = routes.analysis_tasks_list(_app_request(app))
        assert response.status_code == 200
        assert len(json.loads(response.body)["data"]["tasks"]) == 2
        assert all(
            row["deployment"]["projection_status"] == "not_synced"
            and row["deployment"]["applied_artifact_id"] is None
            for row in json.loads(response.body)["data"]["tasks"]
        )
        assert "password" not in repr(logger.mock_calls)
        assert any(route.path == "/api/analysis-tasks" for route in app.routes)
    finally:
        db.close()


def test_startup_identity_failure_is_isolated_and_not_reported_as_empty_success(tmp_path, monkeypatch):
    from argus.dashboard import app as app_module
    from argus.storage.analysis_tasks import AnalysisTaskIntegrityError, AnalysisTaskService

    db, config = _startup_fixture(tmp_path)
    original_bootstrap = AnalysisTaskService.bootstrap_legacy
    bootstrap_calls = []
    projection = Mock()
    logger = Mock()
    monkeypatch.setattr(app_module, "logger", logger)

    def bootstrap(service, camera_id, camera_config):
        bootstrap_calls.append(camera_id)
        if camera_id == "cam1":
            raise AnalysisTaskIntegrityError("C:/private/credentials.json")
        return original_bootstrap(service, camera_id, camera_config)

    monkeypatch.setattr(AnalysisTaskService, "bootstrap_legacy", bootstrap)
    monkeypatch.setattr(AnalysisTaskService, "sync_legacy_model_projection", projection)
    try:
        app = app_module.create_app(database=db, config=config)
        assert app.state.analysis_task_bootstrap_errors == {"cam1"}
        assert bootstrap_calls == ["cam1", "cam2"]
        projection.assert_called_once_with("cam2")
        request = _app_request(app)
        assert routes.analysis_tasks_list(request, camera_id="cam1").status_code == 503
        assert routes.analysis_tasks_list(request, camera_id="cam2").status_code == 200
        assert routes.analysis_tasks_list(request).status_code == 503
        assert "private" not in repr(logger.mock_calls)

        # Simulate an explicit, later successful manager bootstrap. Read APIs
        # recover using the row, without editing the old startup error set or
        # calling the bootstrapping method themselves.
        original_bootstrap(app.state.analysis_task_service, "cam1", {"recovered": True})
        assert routes.analysis_tasks_list(request, camera_id="cam1").status_code == 200
        assert routes.analysis_tasks_list(request).status_code == 200
        assert app.state.analysis_task_bootstrap_errors == {"cam1"}
        assert bootstrap_calls == ["cam1", "cam2"]
        projection.assert_called_once_with("cam2")
    finally:
        db.close()
