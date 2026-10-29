"""Dashboard training wiring must remain inert outside explicit lifespan."""

from unittest.mock import Mock
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from argus.config.schema import ArgusConfig, TaskTrainingConfig
from argus.storage.database import Database


def test_dashboard_training_configuration_is_wired_but_default_runtime_inert(tmp_path):
    from argus.dashboard.app import create_app
    config = ArgusConfig()
    config.dashboard.go2rtc_enabled = False
    config.storage.models_dir = tmp_path / "models"
    config.storage.baselines_dir = tmp_path / "baselines"
    db = Database(f"sqlite:///{tmp_path / 'wiring.db'}")
    db.initialize()
    try:
        app = create_app(database=db, config=config)
        assert app.state.task_training is not None
        assert app.state.task_training_runtime is not None
        assert app.state.task_training_runtime.snapshot()["enabled"] is False
        assert app.state.task_training_runtime.snapshot()["started"] is False
        assert app.state.task_training_admission.snapshot()["state"] == "disabled"
    finally:
        db.close()


def test_enabled_runtime_is_started_only_by_lifespan_and_closed_on_exit(tmp_path, monkeypatch):
    from argus.dashboard.app import create_app
    created = []
    class FakeRuntime:
        def __init__(self, service, **kwargs):
            self.service, self.kwargs = service, kwargs
            self.enabled = kwargs["enabled"]
            self.start = Mock(return_value=True)
            self.close = Mock(return_value=True)
            created.append(self)
    monkeypatch.setattr("argus.runtime.task_training_runtime.TaskTrainingRuntime", FakeRuntime)
    monkeypatch.setattr("argus.dashboard.routes.models.release_worker_for_app", lambda _state: None)
    config = ArgusConfig(task_training=TaskTrainingConfig(enabled=True))
    config.dashboard.go2rtc_enabled = False
    config.storage.models_dir = tmp_path / "models"
    config.storage.baselines_dir = tmp_path / "baselines"
    db = Database(f"sqlite:///{tmp_path / 'lifespan.db'}")
    db.initialize()
    try:
        app = create_app(database=db, config=config)
        runtime = created[0]
        runtime.start.assert_not_called()
        with TestClient(app):
            runtime.start.assert_called_once_with()
            runtime.close.assert_not_called()
        runtime.close.assert_called_once_with(timeout=5)
        assert app.state.task_training_admission.snapshot()["state"] == "waiting_maintenance"
    finally:
        db.close()


@pytest.mark.parametrize("state", ["running", "cancelling", "reconciling"])
def test_restart_quarantines_online_compute_for_unresolved_training_ownership(tmp_path, state):
    from argus.runtime.task_training_resources import training_compute_exclusion_required
    from argus.storage.models import TaskTrainingExecutionRecord
    db = Database(f"sqlite:///{tmp_path / 'orphan-policy.db'}")
    db.initialize()
    try:
        assert training_compute_exclusion_required(ArgusConfig(), db) is False
        with db.get_session() as session:
            session.add(TaskTrainingExecutionRecord(job_id="synthetic-orphan", task_id="synthetic-task",
                workflow_id="synthetic-workflow", status=state, revision=1, request_json="{}",
                request_sha256="0" * 64, created_by="synthetic-human"))
            session.commit()
        assert training_compute_exclusion_required(ArgusConfig(), db) is True
    finally:
        db.close()


def test_legacy_compute_http_is_blocked_but_read_and_task_queue_are_not():
    from fastapi import FastAPI
    from argus.dashboard.training_maintenance import TrainingMaintenanceMiddleware
    app = FastAPI()
    calls = []
    @app.post("/api/models/batch-inference")
    def legacy():
        calls.append("legacy")
        return {}
    @app.post("/api/analysis-tasks/task/workflows/workflow/training-executions")
    def queue():
        calls.append("queue")
        return {}
    app.add_middleware(TrainingMaintenanceMiddleware, enabled=True)
    with TestClient(app) as client:
        denied = client.post("/api/models/batch-inference")
        assert denied.status_code == 409 and denied.json()["code"] == 40900
        assert client.post("/api/analysis-tasks/task/workflows/workflow/training-executions").status_code == 200
    assert calls == ["queue"]


def test_cli_signals_dashboard_then_confirms_training_cleanup_before_database_close():
    from argus.__main__ import _stop_dashboard_training
    sequence = []
    runtime = Mock(close=Mock(side_effect=lambda **_kwargs: sequence.append("training") or True))
    thread = Mock(join=Mock(side_effect=lambda **_kwargs: sequence.append("lifespan")),
                  is_alive=Mock(return_value=False))
    server = SimpleNamespace(should_exit=False)
    app = SimpleNamespace(state=SimpleNamespace(task_training_runtime=runtime))
    assert _stop_dashboard_training(app, server, thread) is True
    assert server.should_exit is True
    assert sequence == ["lifespan", "training"]
    runtime.close.assert_called_once_with(timeout=5)


def test_cli_does_not_claim_unknown_training_cleanup_succeeded():
    from argus.__main__ import _stop_dashboard_training
    runtime = Mock(close=Mock(return_value=False))
    app = SimpleNamespace(state=SimpleNamespace(task_training_runtime=runtime))
    assert _stop_dashboard_training(app, None, None) is False


def test_other_argus_compute_owner_blocks_legacy_http_without_hiding_reads():
    from fastapi import FastAPI
    from argus.dashboard.training_maintenance import TrainingMaintenanceMiddleware
    app = FastAPI()
    @app.get("/api/models/json")
    def read():
        return {"visible": True}
    @app.post("/api/models/batch-inference")
    def execute():
        pytest.fail("compute must be excluded before the handler loads a model")
    claim = Mock(return_value=False)
    app.add_middleware(TrainingMaintenanceMiddleware, compute_admission=claim)
    with TestClient(app) as client:
        assert client.get("/api/models/json").json() == {"visible": True}
        claim.assert_not_called()
        assert client.post("/api/models/batch-inference").status_code == 409
        claim.assert_called_once_with()


def test_disabled_scheduler_does_not_release_host_lock_for_unresolved_orphan(tmp_path):
    from argus.capture.manager import CameraManager
    from argus.config.schema import AlertConfig
    from argus.dashboard.app import create_app
    from argus.runtime.compute_ownership import ComputeOwnership
    from argus.storage.models import TaskTrainingExecutionRecord
    db = Database(f"sqlite:///{tmp_path / 'disabled-orphan.db'}")
    db.initialize()
    manager = CameraManager([], AlertConfig(), database=db)
    path = tmp_path / "compute.lock"
    manager.enable_compute_ownership(ComputeOwnership(path))
    peer, peer_owner = ComputeOwnership(path), object()
    config = ArgusConfig()
    config.dashboard.go2rtc_enabled = False
    config.storage.models_dir = tmp_path / "models"
    config.storage.baselines_dir = tmp_path / "baselines"
    try:
        with db.get_session() as session:
            session.add(TaskTrainingExecutionRecord(job_id="orphan", task_id="task", workflow_id="flow",
                status="reconciling", revision=1, request_json="{}", request_sha256="0" * 64,
                created_by="synthetic-human"))
            session.commit()
        app = create_app(database=db, camera_manager=manager, config=config)
        with TestClient(app):
            assert app.state.task_training_runtime.snapshot()["state"] == "disabled"
            assert peer.acquire(peer_owner) is False
        assert manager.close_compute_ownership() is False
        assert peer.acquire(peer_owner) is False
        # Synthetic positive owner reconciliation, not an arbitrary unlock.
        with db.get_session() as session:
            session.get(TaskTrainingExecutionRecord, "orphan").status = "failed"
            session.commit()
        assert manager.close_compute_ownership() is True
        assert peer.acquire(peer_owner) is True
        assert peer.release(peer_owner)
    finally:
        with db.get_session() as session:
            session.query(TaskTrainingExecutionRecord).delete()
            session.commit()
        manager.close_compute_ownership()
        peer.release(peer_owner)
        manager._inference_executor.shutdown(wait=False)
        manager._encode_executor.shutdown(wait=False)
        db.close()


def test_reassembled_app_cannot_bypass_existing_manager_maintenance(tmp_path, monkeypatch):
    from argus.capture.manager import CameraManager
    from argus.config.schema import AlertConfig
    from argus.dashboard.app import create_app
    from argus.runtime.compute_ownership import ComputeOwnership
    manager = CameraManager([], AlertConfig())
    manager.enable_compute_ownership(ComputeOwnership(tmp_path / "sticky.lock"))
    assert manager.enter_training_maintenance()
    configuration = ArgusConfig()
    configuration.dashboard.go2rtc_enabled = False
    configuration.storage.models_dir = tmp_path / "models"
    configuration.storage.baselines_dir = tmp_path / "baselines"
    monkeypatch.setattr("argus.dashboard.routes.models.release_worker_for_app", lambda _state: None)
    app = create_app(camera_manager=manager, config=configuration)
    @app.post("/api/models/adversarial-compute")
    def unsafe_compute():
        pytest.fail("reassembling a manager cannot bypass its existing maintenance latch")
    try:
        with TestClient(app) as client:
            assert client.post("/api/models/adversarial-compute").status_code == 409
    finally:
        manager.close_compute_ownership()
        manager._inference_executor.shutdown(wait=False)
        manager._encode_executor.shutdown(wait=False)


def test_ad_hoc_compute_admission_fences_late_maintenance_until_restart(tmp_path):
    from argus.capture.manager import CameraManager
    from argus.config.schema import AlertConfig
    from argus.runtime.compute_ownership import ComputeOwnership
    manager = CameraManager([], AlertConfig())
    ownership = ComputeOwnership(tmp_path / "legacy-compute.lock")
    manager.enable_compute_ownership(ownership)
    try:
        assert manager.authorize_legacy_compute() is True
        # The HTTP response can precede an independently owned background
        # executor or a cached detector's final CUDA cleanup.
        assert manager.enter_training_maintenance() is False
        assert manager.close_compute_ownership() is False
    finally:
        assert ownership.release(manager)  # Synthetic fixture never ran compute.
        manager._inference_executor.shutdown(wait=False)
        manager._encode_executor.shutdown(wait=False)


def test_manager_retains_host_owner_for_another_apps_training_slot(tmp_path):
    from argus.capture.manager import CameraManager
    from argus.config.schema import AlertConfig
    from argus.runtime.compute_ownership import ComputeOwnership
    from argus.runtime.task_training_resources import TaskTrainingResourceAdmission
    manager = CameraManager([], AlertConfig())
    manager.enable_compute_ownership(ComputeOwnership(tmp_path / "shared-manager.lock"))
    configuration = TaskTrainingConfig(enabled=True, maintenance_approved=True)
    admission = TaskTrainingResourceAdmission(configuration, manager, probe=lambda: {
        "device": "cuda:0", "free_memory_mb": 9000, "active_compute": False,
    })
    lease = None
    try:
        lease = admission({})
        assert lease is not None
        assert manager.close_compute_ownership() is False
        lease.release()
        assert manager.close_compute_ownership() is True
    finally:
        if lease is not None:
            lease.release()
        manager.close_compute_ownership()
        manager._inference_executor.shutdown(wait=False)
        manager._encode_executor.shutdown(wait=False)


def test_middleware_rechecks_maintenance_instead_of_trusting_startup_snapshot():
    from fastapi import FastAPI
    from argus.dashboard.training_maintenance import TrainingMaintenanceMiddleware
    app = FastAPI()
    mode = {"maintenance": False}
    claim = Mock(return_value=True)
    @app.post("/api/models/compute")
    def legacy():
        return {"synthetic": True}
    app.add_middleware(TrainingMaintenanceMiddleware, enabled=False,
        compute_admission=claim, maintenance_check=lambda: mode["maintenance"])
    with TestClient(app) as client:
        mode["maintenance"] = True
        assert client.post("/api/models/compute").status_code == 409
        claim.assert_not_called()


def test_worker_factory_uses_exact_manager_secondary_guard_path(tmp_path, monkeypatch):
    from argus.capture.manager import CameraManager
    from argus.config.schema import AlertConfig
    from argus.dashboard.app import create_app
    from argus.runtime.compute_ownership import ComputeOwnership
    from argus.runtime.task_training import SpawnTrainingWorker
    created = []
    class FakeRuntime:
        def __init__(self, service, **kwargs):
            self.enabled = kwargs["enabled"]
            self.kwargs = kwargs
            self.start = Mock(return_value=True)
            self.close = Mock(return_value=True)
            created.append(self)
    monkeypatch.setattr("argus.runtime.task_training_runtime.TaskTrainingRuntime", FakeRuntime)
    monkeypatch.setattr("argus.dashboard.routes.models.release_worker_for_app", lambda _state: None)
    db = Database(f"sqlite:///{tmp_path / 'factory.db'}")
    db.initialize()
    manager = CameraManager([], AlertConfig(), database=db)
    ownership = ComputeOwnership(tmp_path / "custom-parent.lock")
    manager.enable_compute_ownership(ownership)
    configuration = ArgusConfig(task_training=TaskTrainingConfig(enabled=True))
    configuration.dashboard.go2rtc_enabled = False
    configuration.storage.models_dir = tmp_path / "models"
    configuration.storage.baselines_dir = tmp_path / "baselines"
    try:
        app = create_app(database=db, camera_manager=manager, config=configuration)
        factory = created[0].kwargs["worker_factory"]
        assert factory.func is SpawnTrainingWorker
        assert factory.keywords["worker_ownership_path"] == ownership.worker_path
        with TestClient(app):
            created[0].start.assert_called_once_with()
        created[0].close.assert_called_once_with(timeout=5)
    finally:
        manager.close_compute_ownership()
        manager._inference_executor.shutdown(wait=False)
        manager._encode_executor.shutdown(wait=False)
        db.close()


@pytest.mark.parametrize("path", [
    "/api/models/model/reexport", "/api/models/model/recalibrate",
    "/api/models/backbone/upgrade", "/api/baseline/compare",
])
def test_unassembled_model_execution_cannot_bypass_host_ownership(path):
    from fastapi import FastAPI
    from argus.dashboard.training_maintenance import TrainingMaintenanceMiddleware
    app = FastAPI()
    calls = []
    @app.post(path)
    def unowned_execution():
        calls.append("executed")
        return {}
    app.add_middleware(TrainingMaintenanceMiddleware, require_compute_authority=True)
    with TestClient(app) as client:
        assert client.post(path).status_code == 409
    assert calls == []
