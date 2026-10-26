"""Durable shadow deployments; SQLite and synthetic workers, never inference."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time
import subprocess
import sys
import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session

from argus.capture.task_execution_manager import TaskExecutionManager
from argus.core.task_execution import TaskExecutionError
from argus.runtime.task_deployments import TaskDeploymentCoordinator
from argus.storage.analysis_tasks import AnalysisTaskConflictError, AnalysisTaskService
from argus.storage.database import Database
from argus.storage.models import TaskDeploymentOperationRecord
from argus.storage.task_deployments import TaskDeploymentService
from test_task_execution_manager import FakeCameraManager, FakeWorker


def until(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


@pytest.fixture
def deployment(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'deploy.db'}")
    db.initialize()
    models = tmp_path / "models"
    models.mkdir()
    tasks = AnalysisTaskService(db, models_root=models)
    task = tasks.create_task("cam", "fast_motion.v1", "Motion", {}, requested_by="human",
                             idempotency_key="create-task")
    bundles = [tasks.freeze_bundle(task["task_id"], expected_revision=1, model_ids={}, requested_by="human",
                                  idempotency_key=f"freeze-{number}") for number in range(2)]
    service = TaskDeploymentService(db, tasks)
    camera = FakeCameraManager()
    executor = TaskExecutionManager(tasks, camera, models_root=models, worker_factory=FakeWorker)
    coordinator = TaskDeploymentCoordinator(service, executor, poll_interval=0.05)
    yield db, tasks, task, bundles, service, camera, executor, coordinator
    coordinator.close(wait=False)
    camera.allow_unregister = True
    for owned in [*executor._executions.values(), *executor._prepared.values()]:
        owned.worker.cleanup_ok = True
        owned.worker.release.set()
    assert executor.close()
    assert coordinator.join(5)
    db.close()


def request(platform, *, action="deploy", index=0, revision=0, key="deploy-key", actor="human"):
    _, _, task, bundles, service, _, _, _ = platform
    return service.request(task["task_id"], action=action,
                           bundle_id=None if action == "stop" else bundles[index]["bundle_id"],
                           expected_deployment_revision=revision, requested_by=actor, idempotency_key=key)


def apply(platform, **kwargs):
    coordinator = platform[-1]
    if not coordinator._owner.held:
        assert coordinator._owner.acquire()
    operation = request(platform, **kwargs)
    result = coordinator.process(operation["operation_id"])
    status = platform[6].status(operation["task_id"]) or {}
    assert result["status"] == "applied", (result["failure_code"], status.get("state"),
        status.get("deployment_operation_id"), status.get("deployment_revision"), status.get("error_code"))
    return result


def test_durable_deployment_has_separate_intent_applied_and_revision(deployment):
    operation = request(deployment)
    service, executor = deployment[4], deployment[6]
    row = service.get(operation["task_id"])
    assert row["desired_bundle_id"] == operation["bundle_id"]
    assert row["applied_bundle_id"] is None and row["deployment_revision"] == 1
    assert executor.status(operation["task_id"]) is None
    result = apply(deployment)
    row = service.get(operation["task_id"])
    assert row["applied_bundle_id"] == operation["bundle_id"]
    assert row["applied_deployment_revision"] == 1
    assert result["production_allowed"] is False and row["stage"] == "shadow"


def test_exact_retry_preserves_operation_but_changed_actor_and_payload_are_rejected(deployment):
    operation = apply(deployment)
    assert request(deployment) == operation
    for arguments in ({"actor": "different-human"}, {"index": 1}, {"revision": 1}):
        with pytest.raises(AnalysisTaskConflictError):
            request(deployment, **arguments)


@pytest.mark.parametrize("after_commit", [False, True])
def test_first_command_commit_failure_and_lost_ack_retry_are_atomic(deployment, monkeypatch, after_commit):
    commit = Session.commit

    def fail(session):
        if after_commit:
            commit(session)
        raise RuntimeError("commit acknowledgement unknown")

    with monkeypatch.context() as patch:
        patch.setattr(Session, "commit", fail)
        with pytest.raises(RuntimeError):
            request(deployment)
    result = request(deployment)
    assert result["deployment_revision"] == 1
    with deployment[0].get_session() as session:
        assert len(list(session.scalars(select(TaskDeploymentOperationRecord)))) == 1


def test_concurrent_cas_accepts_one_intent(deployment):
    def write(number):
        try:
            return request(deployment, key=f"parallel-{number}")
        except AnalysisTaskConflictError:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(write, range(8)))
    assert sum(result is not None for result in results) == 1


def test_stop_supersedes_pending_intent_and_late_receipt_cannot_overwrite_it(deployment):
    service = deployment[4]
    first = request(deployment)
    stop = request(deployment, action="stop", revision=1, key="stop-key")
    assert service.get_operation(first["operation_id"])["status"] == "superseded"
    service.observe(first["operation_id"], {"task_id": first["task_id"], "state": "running",
        "bundle_id": first["bundle_id"], "deployment_operation_id": first["operation_id"], "deployment_revision": 1})
    row = service.get(first["task_id"])
    assert row["desired_state"] == "stopped" and row["operation_id"] == stop["operation_id"]
    assert row["applied_bundle_id"] is None


def test_upgrade_and_same_configuration_rollback_have_distinct_durable_receipts(deployment):
    first = apply(deployment)
    second = apply(deployment, index=1, revision=1, key="upgrade-key")
    row = deployment[4].get(first["task_id"])
    assert row["previous_bundle_id"] == first["bundle_id"]
    assert row["applied_bundle_id"] == second["bundle_id"]
    rolled = apply(deployment, action="rollback", index=0, revision=2, key="rollback-key")
    assert rolled["action"] == "rollback" and rolled["deployment_revision"] == 3
    assert deployment[6].status(first["task_id"])["bundle_id"] == first["bundle_id"]


def test_cross_configuration_rollback_cannot_bypass_current_task_authority(deployment):
    first = apply(deployment)
    apply(deployment, action="stop", revision=1, key="stop-key")
    deployment[1].update_task(first["task_id"], expected_revision=1, config={"rules": {"confidence_threshold": 0.8}},
                              requested_by="human", idempotency_key="new-config")
    with pytest.raises(AnalysisTaskConflictError):
        request(deployment, action="rollback", revision=2, key="rollback-key")
    assert deployment[4].get(first["task_id"])["deployment_revision"] == 2


def test_candidate_loading_failure_preserves_old_running_worker_and_applied_bundle(deployment):
    first = apply(deployment)
    executor = deployment[6]
    previous = executor._executions[first["task_id"]]

    class Reject(FakeWorker):
        def start(self):
            raise TaskExecutionError("LOAD_TIMEOUT", "synthetic loading failure")

    executor._worker_factory = Reject
    operation = request(deployment, index=1, revision=1, key="failed-upgrade")
    result = deployment[-1].process(operation["operation_id"])
    assert result["status"] == "failed" and result["failure_code"] == "LOAD_TIMEOUT"
    assert executor._executions[first["task_id"]] is previous and previous.worker.running
    assert deployment[4].get(first["task_id"])["applied_bundle_id"] == first["bundle_id"]
    assert executor._prepared == {}


def test_resource_shortage_does_not_stop_old_worker_to_make_room(deployment):
    first = apply(deployment)
    executor = deployment[6]
    previous = executor._executions[first["task_id"]]
    executor._max_workers = 1
    operation = request(deployment, index=1, revision=1, key="no-spare-worker")
    result = deployment[-1].process(operation["operation_id"])
    assert result["failure_code"] == "RESOURCE_EXHAUSTED"
    assert previous.worker.running and executor._executions[first["task_id"]] is previous


def test_failed_activation_compensates_to_old_authorized_bundle(deployment, monkeypatch):
    first = apply(deployment)
    camera = deployment[5]
    register = camera.register_analysis_task_runtime
    calls = []

    def reject_once(runtime):
        calls.append(runtime)
        if len(calls) == 1:
            raise RuntimeError("synthetic candidate subscription failure")
        return register(runtime)

    monkeypatch.setattr(camera, "register_analysis_task_runtime", reject_once)
    operation = request(deployment, index=1, revision=1, key="activation-fail")
    assert deployment[-1].process(operation["operation_id"])["status"] == "failed"
    current = deployment[6].status(first["task_id"])
    assert current["bundle_id"] == first["bundle_id"] and current["state"] in {"running", "waiting_source"}
    assert current["deployment_operation_id"] == first["operation_id"]


def test_unknown_apply_commit_is_reconciled_by_exact_runtime_operation(deployment, monkeypatch):
    service, coordinator = deployment[4], deployment[-1]
    assert coordinator._owner.acquire()
    operation = request(deployment)
    observe = service.observe
    monkeypatch.setattr(service, "observe", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("disk ack unknown")))
    assert coordinator.process(operation["operation_id"]) is None
    assert service.get_operation(operation["operation_id"])["status"] == "running"
    monkeypatch.setattr(service, "observe", observe)
    row = coordinator.reconcile(operation["task_id"])
    assert row["operation"]["status"] == "applied"
    assert row["applied_bundle_id"] == operation["bundle_id"]


def test_source_absent_is_waiting_and_later_retried_without_opening_source(deployment, monkeypatch):
    service, camera, coordinator = deployment[4], deployment[5], deployment[-1]
    source = camera.source
    monkeypatch.setattr(camera, "get_camera_source", lambda camera_id: None)
    operation = request(deployment)
    assert coordinator.start()
    coordinator.submit(operation["operation_id"])
    until(lambda: service.get_operation(operation["operation_id"])["status"] == "waiting_source")
    assert deployment[6]._executions == {} and deployment[6]._prepared == {}
    monkeypatch.setattr(camera, "get_camera_source", lambda camera_id: source)
    until(lambda: service.get_operation(operation["operation_id"])["status"] == "applied")
    assert service.get_operation(operation["operation_id"])["requested_by"] == "human"


def test_restart_recovers_last_applied_bundle_from_database_not_files(deployment):
    first = apply(deployment)
    old = deployment[-1]
    old.close(wait=False)
    assert deployment[6].close()
    assert old.join()
    executor = TaskExecutionManager(deployment[1], deployment[5], models_root=deployment[1]._models_root,
                                    worker_factory=FakeWorker)
    restarted = TaskDeploymentCoordinator(deployment[4], executor, poll_interval=0.05)
    try:
        assert restarted.start()
        until(lambda: deployment[4].get(first["task_id"])["deployment_revision"] == 2)
        until(lambda: deployment[4].get(first["task_id"])["operation"]["status"] == "applied")
        current = executor.status(first["task_id"])
        assert current["bundle_id"] == first["bundle_id"]
        assert deployment[4].get(first["task_id"])["operation"]["requested_by"] == "human"
    finally:
        restarted.close(wait=False)
        assert executor.close()
        assert restarted.join()


def test_second_owner_is_rejected_and_unknown_cleanup_keeps_lock(deployment, monkeypatch):
    coordinator, executor = deployment[-1], deployment[6]
    assert coordinator.start()
    other = TaskDeploymentCoordinator(deployment[4], executor)
    try:
        assert other.start() is False
        monkeypatch.setattr(executor, "cleanup_confirmed", lambda task_id=None: False)
        assert coordinator.close(wait=False) is False
        assert other.start() is False
        monkeypatch.setattr(executor, "cleanup_confirmed", lambda task_id=None: True)
        assert coordinator.join()
        assert other.start()
    finally:
        other.close(wait=True)


def test_stale_hook_cannot_notify_old_applied_command_and_pending_hook_retries(deployment):
    first = apply(deployment)
    coordinator = deployment[-1]
    calls = []
    coordinator._hooked.clear()
    coordinator._on_observed = lambda op: calls.append(op["operation_id"]) or {"state": "cleanup_pending"}
    coordinator.reconcile(first["task_id"])
    coordinator.reconcile(first["task_id"])
    assert calls == [first["operation_id"], first["operation_id"]]
    second = request(deployment, action="stop", revision=1, key="stop-key")
    coordinator._after_receipt(first)
    assert calls == [first["operation_id"], first["operation_id"]]
    assert deployment[4].get(first["task_id"])["operation_id"] == second["operation_id"]


def test_deployment_indexes_are_recreated_and_guard_duplicate_commands(deployment):
    db = deployment[0]
    expected = {"uq_task_deployment_operation_key", "uq_task_deployment_operation_revision"}
    with db._engine.begin() as connection:
        for name in expected:
            connection.execute(text(f"DROP INDEX {name}"))
    db._auto_migrate_indexes()
    assert expected <= {row["name"] for row in inspect(db._engine).get_indexes("task_deployment_operations")}


def test_stop_during_candidate_loading_cancels_both_and_never_resurrects_old(deployment):
    first = apply(deployment)
    entered, release = threading.Event(), threading.Event()

    class Loading(FakeWorker):
        def start(self):
            entered.set()
            assert release.wait(3)
            self.running = True

        def stop(self):
            release.set()
            return super().stop()

    deployment[6]._worker_factory = Loading
    operation = request(deployment, index=1, revision=1, key="slow-upgrade")
    with ThreadPoolExecutor(max_workers=2) as pool:
        upgrade = pool.submit(deployment[-1].process, operation["operation_id"])
        assert entered.wait(2)
        stop = request(deployment, action="stop", revision=2, key="cancel-upgrade")
        stopped = pool.submit(deployment[-1].process, stop["operation_id"])
        assert stopped.result(timeout=4)["status"] == "applied"
        assert upgrade.result(timeout=4)["status"] == "superseded"
    assert deployment[6].status(first["task_id"])["state"] == "stopped"
    assert deployment[6].cleanup_confirmed(first["task_id"])


def test_source_stop_cancels_prepared_load_and_preserves_durable_running_intent(deployment, monkeypatch):
    first = apply(deployment)
    entered, release = threading.Event(), threading.Event()

    class Loading(FakeWorker):
        def start(self):
            entered.set()
            assert release.wait(3)
            self.running = True

        def stop(self):
            release.set()
            return super().stop()

    executor, camera = deployment[6], deployment[5]
    executor._worker_factory = Loading
    operation = request(deployment, index=1, revision=1, key="source-stop-load")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(deployment[-1].process, operation["operation_id"])
        assert entered.wait(2)
        # Mirrors CameraManager's stopping fence; no source start is called.
        monkeypatch.setattr(camera, "get_camera_source", lambda camera_id: None)
        assert executor.stop_source("cam")
        assert future.result(timeout=4)["status"] == "waiting_source"
    assert executor.cleanup_confirmed(first["task_id"]) and executor._prepared == {}
    row = deployment[4].get(first["task_id"])
    assert row["desired_state"] == "running" and row["applied_bundle_id"] == first["bundle_id"]


def test_completed_source_stop_recovers_only_after_source_returns(deployment, monkeypatch):
    first = apply(deployment)
    executor, camera, coordinator = deployment[6], deployment[5], deployment[-1]
    source = camera.source
    monkeypatch.setattr(camera, "get_camera_source", lambda camera_id: None)
    assert executor.stop_source("cam")
    recovered = coordinator.recover()
    assert len(recovered) == 1 and recovered[0]["bundle_id"] == first["bundle_id"]
    assert coordinator.process(recovered[0]["operation_id"])["status"] == "waiting_source"
    assert executor.cleanup_confirmed(first["task_id"])
    monkeypatch.setattr(camera, "get_camera_source", lambda camera_id: source)
    assert coordinator.process(recovered[0]["operation_id"])["status"] == "applied"
    assert executor.status(first["task_id"])["deployment_revision"] == 2


def _probe_owner_in_another_process(database_url, cwd):
    # The ordinary CI profile installs dependencies, not an editable Argus
    # package. A child interpreter must use this checkout explicitly instead
    # of relying on pytest's in-process sys.path or a developer installation.
    source_root = Path(__file__).resolve().parents[2] / "src"
    code = (
        "import sys; from pathlib import Path; from types import SimpleNamespace; "
        "sys.path.insert(0, sys.argv[2]); "
        "import argus.runtime.task_deployments as deployments; "
        "assert Path(deployments.__file__).resolve() == "
        "Path(sys.argv[2], 'argus/runtime/task_deployments.py').resolve(); "
        "owner=deployments._RuntimeOwner(SimpleNamespace(_database_url=sys.argv[1])); "
        "print(owner.acquire(), flush=True)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, database_url, str(source_root)],
        cwd=cwd, capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, (
        f"runtime-owner subprocess failed ({result.returncode})\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert result.stdout.strip() in {"True", "False"}, result.stdout
    return result.stdout.strip() == "True"


def test_owner_os_lock_rejects_another_process_without_pid_or_time_lease(deployment, tmp_path):
    coordinator = deployment[-1]
    assert coordinator._owner.acquire()
    assert not _probe_owner_in_another_process(deployment[0]._database_url, tmp_path)
    coordinator._owner.release()
    assert _probe_owner_in_another_process(deployment[0]._database_url, tmp_path)


def test_owner_os_lock_is_released_when_holding_process_exits(deployment, tmp_path):
    # The child deliberately exits without release(). The OS lock must then
    # be available without deleting the lock file or waiting for a PID lease.
    owner = deployment[-1]._owner
    assert _probe_owner_in_another_process(deployment[0]._database_url, tmp_path)
    assert owner.path.is_file()
    assert owner.acquire()
    assert not _probe_owner_in_another_process(deployment[0]._database_url, tmp_path)


def test_intent_commit_rechecks_task_revision_after_bundle_validation(deployment, monkeypatch):
    tasks, task = deployment[1], deployment[2]
    resolve = tasks.resolve_bundle

    def concurrent_edit(task_id, bundle_id):
        manifest = resolve(task_id, bundle_id)
        tasks.update_task(task_id, expected_revision=1, config={"rules": {"confidence_threshold": 0.7}},
                          requested_by="human", idempotency_key="edit-during-deploy")
        return manifest

    monkeypatch.setattr(tasks, "resolve_bundle", concurrent_edit)
    with pytest.raises(AnalysisTaskConflictError, match="authority changed"):
        request(deployment)
    assert deployment[4].get(task["task_id"])["deployment_revision"] == 0


def test_required_scene_waits_for_existing_gate_without_bypassing_approval(deployment):
    tasks, service, camera, coordinator = deployment[1], deployment[4], deployment[5], deployment[-1]
    task = tasks.create_task("cam", "fast_motion.v1", "Bound scene", {
        "scene": {"policy": "require_profile", "profile_id": "approved-scene"}},
        requested_by="human", idempotency_key="scene-task")
    bundle = tasks.freeze_bundle(task["task_id"], expected_revision=1, model_ids={},
                                  requested_by="human", idempotency_key="scene-bundle")
    operation = service.request(task["task_id"], action="deploy", bundle_id=bundle["bundle_id"],
                                expected_deployment_revision=0, requested_by="human", idempotency_key="scene-deploy")
    assert coordinator._owner.acquire()
    assert coordinator.process(operation["operation_id"])["status"] == "waiting_scene"
    assert deployment[6]._prepared == {}
    camera.scene["usable"] = True
    assert coordinator.process(operation["operation_id"])["status"] == "applied"


def stopped_with_new_configuration(platform):
    first = apply(platform)
    apply(platform, action="stop", revision=1, key="stop-key")
    task = platform[1].update_task(first["task_id"], expected_revision=1,
        config={"rules": {"confidence_threshold": 0.9}}, requested_by="human", idempotency_key="edit-config")
    return first, task


def restore(platform, *, key="restore-configuration", actor="human"):
    return platform[1].restore_bundle_configuration(platform[2]["task_id"], platform[3][0]["bundle_id"],
        expected_revision=2, requested_by=actor, idempotency_key=key)


def test_historical_configuration_restore_creates_new_revision_bundle_and_provenance(deployment):
    first, edited = stopped_with_new_configuration(deployment)
    result = restore(deployment)
    assert result["task"]["current_revision"] == 3
    assert result["task"]["config"] == deployment[2]["config"]
    assert edited["config"] != result["task"]["config"]
    assert result["bundle"]["bundle_id"] != first["bundle_id"]
    assert result["bundle"]["task_revision"] == 3
    assert result["restore"]["restored_from_bundle_id"] == first["bundle_id"]
    assert result["restore"]["restored_from_task_revision"] == 1
    assert deployment[4].get(first["task_id"])["desired_state"] == "stopped"
    assert deployment[4].get(first["task_id"])["applied_bundle_id"] == first["bundle_id"]
    with pytest.raises(AnalysisTaskConflictError):
        deployment[1].resolve_bundle(first["task_id"], first["bundle_id"])
    assert deployment[1].resolve_bundle(first["task_id"], result["bundle"]["bundle_id"])["task_revision"] == 3
    operation = deployment[4].request(first["task_id"], action="deploy", bundle_id=result["bundle"]["bundle_id"],
        expected_deployment_revision=2, requested_by="human", idempotency_key="deploy-restored-config")
    assert deployment[-1].process(operation["operation_id"])["status"] == "applied"
    # Original receipt remains reconcilable even after the new runtime starts.
    assert restore(deployment) == result
    assert deployment[1].get_bundle_configuration_restore_receipt(first["task_id"], first["bundle_id"],
        expected_revision=2, requested_by="human", idempotency_key="restore-configuration") == result


def test_bundle_apply_metadata_uses_complete_task_scoped_history_without_changing_manifest(deployment):
    first = apply(deployment)
    db, tasks, task, bundles, service, *_ = deployment
    other = tasks.create_task("cam", "fast_motion.v1", "Other task", {}, requested_by="human",
                              idempotency_key="history-other-task")
    with db.get_session() as session:
        # Push the real apply outside the bounded recent-operation timeline.
        for revision in range(2, 103):
            session.add(TaskDeploymentOperationRecord(
                operation_id=f"history-{revision}", task_id=task["task_id"],
                idempotency_key=f"history-key-{revision}", request_sha256="0" * 64,
                action="deploy", bundle_id=bundles[1]["bundle_id"],
                deployment_revision=revision, status="failed", requested_by="human"))
        # Neither a stop receipt nor another task's receipt authorizes a
        # bundle, even if historical data contains a malformed bundle pointer.
        session.add(TaskDeploymentOperationRecord(
            operation_id="history-stop", task_id=task["task_id"], idempotency_key="history-stop-key",
            request_sha256="1" * 64, action="stop", bundle_id=bundles[1]["bundle_id"],
            deployment_revision=103, status="applied", requested_by="human"))
        session.add(TaskDeploymentOperationRecord(
            operation_id="history-other", task_id=other["task_id"], idempotency_key="history-other-key",
            request_sha256="2" * 64, action="deploy", bundle_id=bundles[1]["bundle_id"],
            deployment_revision=1, status="applied", requested_by="human"))
        session.commit()
    recent = service.list_operations(task["task_id"])
    assert len(recent) == 100
    assert first["operation_id"] not in {item["operation_id"] for item in recent}
    listed = {item["bundle_id"]: item for item in tasks.list_bundles(task["task_id"])}
    assert listed[bundles[0]["bundle_id"]]["historically_applied"] is True
    assert listed[bundles[1]["bundle_id"]]["historically_applied"] is False
    for original in bundles:
        item = listed[original["bundle_id"]]
        assert item["manifest"] == original["manifest"]
        assert item["manifest_sha256"] == original["manifest_sha256"]


@pytest.mark.parametrize("after_commit", [False, True])
def test_configuration_restore_commit_failure_is_atomic_and_same_key_retryable(deployment, monkeypatch, after_commit):
    stopped_with_new_configuration(deployment)
    commit = Session.commit

    def fail(session):
        if after_commit:
            commit(session)
        raise RuntimeError("restoration commit acknowledgement lost")

    with monkeypatch.context() as patch:
        patch.setattr(Session, "commit", fail)
        with pytest.raises(RuntimeError):
            restore(deployment)
    assert deployment[1].get_task(deployment[2]["task_id"])["current_revision"] == (3 if after_commit else 2)
    result = restore(deployment)
    assert result["task"]["current_revision"] == 3 and restore(deployment) == result
    with pytest.raises(AnalysisTaskConflictError):
        restore(deployment, actor="different-human")


def test_configuration_restore_rejects_unapplied_and_cross_task_bundles(deployment):
    first, _ = stopped_with_new_configuration(deployment)
    tasks = deployment[1]
    with pytest.raises(AnalysisTaskConflictError, match="applied"):
        tasks.restore_bundle_configuration(first["task_id"], deployment[3][1]["bundle_id"], expected_revision=2,
            requested_by="human", idempotency_key="unapplied-target")
    assert tasks.get_task(first["task_id"])["current_revision"] == 2


def test_configuration_restore_rejects_running_desired_state_without_mutating_task(deployment):
    first = apply(deployment)
    with pytest.raises(AnalysisTaskConflictError, match="durable stop"):
        deployment[1].restore_bundle_configuration(first["task_id"], first["bundle_id"], expected_revision=1,
            requested_by="human", idempotency_key="restore-while-running")
    assert deployment[1].get_task(first["task_id"])["current_revision"] == 1


def test_configuration_restore_rechecks_historical_artifact_bytes(deployment):
    db, tasks, _, _, service, _, executor, coordinator = deployment
    task = tasks.create_task("cam", "object_detection.v1", "Objects", {}, requested_by="human",
        idempotency_key="object-task")
    path = tasks._models_root / "opaque.onnx"
    path.write_bytes(b"opaque registration bytes, never executed")
    model = tasks.register_task_model("opaque.onnx", {"name": "Fake detector", "adapter_id": "onnx_detection.v1",
        "output_schema": "yolov8_detection.v1", "labels": ["person"], "input_size": [640, 640]},
        requested_by="human", idempotency_key="opaque-model")
    bundle = tasks.freeze_bundle(task["task_id"], expected_revision=1, model_ids={"detector": model["model_id"]},
        requested_by="human", idempotency_key="object-bundle")
    assert coordinator._owner.acquire()
    operation = service.request(task["task_id"], action="deploy", bundle_id=bundle["bundle_id"],
        expected_deployment_revision=0, requested_by="human", idempotency_key="object-deploy")
    assert coordinator.process(operation["operation_id"])["status"] == "applied"
    stopped = service.request(task["task_id"], action="stop", expected_deployment_revision=1,
        requested_by="human", idempotency_key="object-stop")
    assert coordinator.process(stopped["operation_id"])["status"] == "applied"
    tasks.update_task(task["task_id"], expected_revision=1, config={"rules": {"confidence_threshold": 0.9}},
        requested_by="human", idempotency_key="object-edit")
    path.write_bytes(b"changed registration bytes")
    with pytest.raises(AnalysisTaskConflictError, match="artifact bytes changed"):
        tasks.restore_bundle_configuration(task["task_id"], bundle["bundle_id"], expected_revision=2,
            requested_by="human", idempotency_key="object-restore")
    assert tasks.get_task(task["task_id"])["current_revision"] == 2
    assert service.get(task["task_id"])["applied_bundle_id"] == bundle["bundle_id"]
    assert executor.cleanup_confirmed(task["task_id"])


def test_applied_runtime_failure_updates_observation_and_releases_dead_lease(deployment):
    first = apply(deployment)
    executor, coordinator, service = deployment[6], deployment[-1], deployment[4]
    execution = executor._executions[first["task_id"]]
    execution.worker.stop()
    execution.state, execution.error_code = "failed", "WORKER_FAILED"
    coordinator.reconcile(first["task_id"])
    assert service.get(first["task_id"])["state"] == "failed"
    assert executor.cleanup_confirmed(first["task_id"])
    recovered = coordinator.recover()
    assert len(recovered) == 1 and recovered[0]["bundle_id"] == first["bundle_id"]
    assert coordinator.process(recovered[0]["operation_id"])["status"] == "applied"


def test_unknown_owner_is_kept_alive_in_registry_until_confirmed_release(tmp_path):
    from argus.runtime.task_deployments import _RuntimeOwner

    db = SimpleNamespace(_database_url=f"sqlite:///{tmp_path / 'unknown-owner.db'}")
    owner = _RuntimeOwner(db)
    assert owner.acquire()
    reference = weakref.ref(owner)
    del owner
    gc.collect()
    assert reference() is not None
    contender = _RuntimeOwner(db)
    assert not contender.acquire()
    reference().release()
    assert contender.acquire()
    contender.release()


def test_owner_release_failure_is_visible_and_keeps_exclusion(deployment, monkeypatch):
    coordinator = deployment[-1]
    assert coordinator._owner.acquire()
    release = coordinator._owner.release
    other = TaskDeploymentCoordinator(deployment[4], deployment[6])
    try:
        monkeypatch.setattr(coordinator._owner, "release", lambda: (_ for _ in ()).throw(OSError("unknown unlock")))
        assert coordinator.close(wait=False) is False
        assert coordinator.status()["error_code"] == "RUNTIME_OWNER_RELEASE_FAILED"
        assert other.start() is False
        monkeypatch.setattr(coordinator._owner, "release", release)
        assert coordinator.join()
        assert other.start()
    finally:
        other.close(wait=True)
