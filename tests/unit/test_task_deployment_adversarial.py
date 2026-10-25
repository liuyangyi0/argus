"""Adversarial deployment schedules with synthetic workers only."""

from concurrent.futures import ThreadPoolExecutor
import threading

from test_task_deployments import deployment, request as command  # noqa: F401
from test_task_deployments import apply, until
from test_task_execution_manager import synthetic_frame
from argus.core.task_execution import TaskExecutionError


def wait_for_committed_result(executor, task_id):
    # results() takes a nonblocking DB authority fence. Polling it while the
    # sole synthetic frame is committing can itself make that frame get
    # deliberately dropped on DB contention. Wait for in-memory completion,
    # then assert the public read before injecting the race under test.
    def committed():
        with executor._lock:
            return bool(executor._executions[task_id].results)

    until(committed)
    assert len(executor.results(task_id)) == 1


def test_stop_applied_before_claimed_deploy_enters_executor_cannot_resurrect(request, monkeypatch):
    platform = request.getfixturevalue("deployment")
    service, executor, coordinator = platform[4], platform[6], platform[-1]
    assert coordinator._owner.acquire()
    first = command(platform)
    entered, release = threading.Event(), threading.Event()
    deploy = executor.deploy

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return deploy(*args, **kwargs)

    monkeypatch.setattr(executor, "deploy", delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        old = pool.submit(coordinator.process, first["operation_id"])
        try:
            assert entered.wait(1)
            assert service.get_operation(first["operation_id"])["status"] == "running"
            stop = command(platform, action="stop", revision=1, key="stop-before-allocation")
            assert coordinator.process(stop["operation_id"])["status"] == "applied"
            assert service.get(first["task_id"])["desired_state"] == "stopped"
        finally:
            release.set()
        assert old.result(timeout=4)["status"] == "superseded"
    observed = executor.status(first["task_id"])
    assert observed is None or observed["state"] == "stopped", "superseded deploy resurrected after stop"
    assert executor.cleanup_confirmed(first["task_id"])


def test_claimed_old_stop_cannot_cancel_a_newer_applied_deployment(request, monkeypatch):
    platform = request.getfixturevalue("deployment")
    first = apply(platform)
    executor, coordinator = platform[6], platform[-1]
    old_stop = command(platform, action="stop", revision=1, key="old-stop-claim")
    entered, release = threading.Event(), threading.Event()
    stop = executor.stop

    def delayed(*args, **kwargs):
        if kwargs.get("operation_id") == old_stop["operation_id"]:
            entered.set()
            assert release.wait(3)
        return stop(*args, **kwargs)

    monkeypatch.setattr(executor, "stop", delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        old = pool.submit(coordinator.process, old_stop["operation_id"])
        try:
            assert entered.wait(1)
            apply(platform, action="stop", revision=2, key="new-stop-claim")
            newest = apply(platform, index=1, revision=3, key="newest-deployment")
        finally:
            release.set()
        assert old.result(timeout=4)["status"] == "superseded"
    observed = executor.status(first["task_id"])
    assert observed["deployment_operation_id"] == newest["operation_id"]
    assert observed["bundle_id"] == newest["bundle_id"]
    assert observed["state"] in {"running", "waiting_source"}


def test_applied_worker_failure_reconciles_live_state_without_rewriting_applied_history(request, monkeypatch):
    platform = request.getfixturevalue("deployment")
    first = apply(platform)
    service, camera, executor, coordinator = platform[4], platform[5], platform[6], platform[-1]
    execution = executor._executions[first["task_id"]]

    def fail(_frame):
        raise TaskExecutionError("WORKER_FAILED", "synthetic inference worker failed")

    monkeypatch.setattr(execution.worker, "infer", fail)
    camera.source.hub.publish(synthetic_frame())
    until(lambda: execution.runtime.status.state == "failed")
    row = coordinator.reconcile(first["task_id"])
    assert row["observed_state"] in {"failed", "stopped", "cleanup_pending"}
    assert service.get_operation(first["operation_id"])["status"] == "applied"
    assert executor.cleanup_confirmed(first["task_id"])
    recovery = coordinator.recover()
    assert len(recovery) == 1 and recovery[0]["bundle_id"] == first["bundle_id"]
    assert recovery[0]["operation_id"] != first["operation_id"]


def test_same_bundle_redeployment_discards_inflight_and_retained_old_results(request, monkeypatch):
    platform = request.getfixturevalue("deployment")
    first = apply(platform)
    executor, camera, coordinator = platform[6], platform[5], platform[-1]
    execution = executor._executions[first["task_id"]]
    camera.source.hub.publish(synthetic_frame())
    wait_for_committed_result(executor, first["task_id"])
    entered, release = threading.Event(), threading.Event()
    infer = execution.worker.infer

    def delayed(frame):
        entered.set()
        assert release.wait(3)
        return infer(frame)

    monkeypatch.setattr(execution.worker, "infer", delayed)
    camera.source.hub.publish(synthetic_frame(2, 10.5))
    try:
        assert entered.wait(1)
        second = apply(platform, revision=1, key="same-bundle-new-deployment")
        assert executor._executions[first["task_id"]] is execution
        assert executor.results(first["task_id"]) == []
        assert execution.runtime.status.deployment_revision == second["deployment_revision"]
    finally:
        release.set()
    until(lambda: execution.runtime.status.discarded_results >= 1)
    assert executor.results(first["task_id"]) == []
    camera.source.hub.publish(synthetic_frame(3, 11.0))
    wait_for_committed_result(executor, first["task_id"])
    assert "deployment_changed" in execution.worker.resets
    assert coordinator.reconcile(first["task_id"])["operation_id"] == second["operation_id"]


def test_durable_same_bundle_adopts_compatibility_runtime_without_reusing_task_revision_as_fence(request):
    platform = request.getfixturevalue("deployment")
    task_id, bundle_id = platform[2]["task_id"], platform[3][0]["bundle_id"]
    executor = platform[6]
    executor.start(task_id, bundle_id)
    previous = executor._executions[task_id]
    assert previous.runtime.status.deployment_revision == 0
    applied = apply(platform)
    assert executor._executions[task_id] is previous
    assert previous.runtime.status.deployment_revision == applied["deployment_revision"] == 1


def test_result_read_rechecks_exact_execution_after_runtime_is_replaced(request, monkeypatch):
    platform = request.getfixturevalue("deployment")
    first = apply(platform)
    executor, camera = platform[6], platform[5]
    camera.source.hub.publish(synthetic_frame())
    wait_for_committed_result(executor, first["task_id"])
    commit = platform[1].commit_if_current
    switched = False

    def replace_before_read(*args):
        nonlocal switched
        if not switched:
            switched = True
            apply(platform, index=1, revision=1, key="replace-during-result-read")
        return commit(*args)

    monkeypatch.setattr(platform[1], "commit_if_current", replace_before_read)
    assert executor.results(first["task_id"]) == []
    assert switched
