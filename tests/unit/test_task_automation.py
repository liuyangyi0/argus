from unittest.mock import ANY, Mock
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from argus.runtime.task_automation import TaskAutomationService


def setup(enabled=True):
    tasks, executor, workflows, manager = Mock(), Mock(), Mock(), Mock()
    tasks.get_task.return_value = {"config": {"collection": {"enabled": enabled}}}
    status = {"task_id": "task", "state": "running", "worker": {"execution_id": "execution-1", "state": "running"}}
    executor.start.return_value = status
    executor.status.return_value = status
    flow = {"workflow_id": "flow", "revision": 1, "state": "collectable", "collection_runtime": None}
    workflows._flow = flow
    workflows.create.side_effect = lambda *args, **kwargs: deepcopy(flow)
    workflows.get.side_effect = lambda *args, **kwargs: deepcopy(flow)

    def start_collection(*args, **kwargs):
        flow.update(revision=flow["revision"] + 1, state="collecting", collection_runtime={
            "collection_id": kwargs["collection_id"], "state": "collecting",
            "persistence_pending": False, "cleanup_confirmed": False,
        }, operation={"kind": "collection", "collection_id": kwargs["collection_id"]})
        return deepcopy(flow)

    def stop_collection(*args, **kwargs):
        collection_id = flow["collection_runtime"]["collection_id"]
        flow.update(revision=flow["revision"] + 1, state="review", collection_runtime={
            "collection_id": collection_id, "state": "stopped", "persistence_pending": False, "cleanup_confirmed": True,
        }, operation=None)
        return deepcopy(flow)

    workflows.start_collection.side_effect = start_collection
    workflows.stop_collection.side_effect = stop_collection
    return TaskAutomationService(tasks, executor, workflows, manager), executor, workflows


def test_enabled_collection_is_started_once_and_task_stop_drains_its_own_collector():
    service, executor, workflows = setup()
    assert service.start("task", "bundle", actor="engineer")["automatic_collection"]["state"] == "collecting"
    service.start("task", "bundle", actor="engineer")
    workflows.create.assert_called_once()
    workflows.start_collection.assert_called_once()
    executor.stop.return_value = {"task_id": "task", "state": "stopped"}
    stopped = service.stop("task", actor="engineer")
    assert stopped["automatic_collection"]["state"] == "stopped"
    workflows.stop_collection.assert_called_once_with("flow", expected_revision=2, actor="engineer", expected_collection_id=ANY)


def test_collection_failure_is_visible_and_does_not_misrepresent_execution_or_duplicate_workflow():
    service, executor, workflows = setup()
    workflows.start_collection.side_effect = RuntimeError("source unavailable")
    result = service.start("task", "bundle", actor="engineer")
    assert result["state"] == "running"
    assert result["automatic_collection"]["state"] == "blocked"
    assert result["automatic_collection"]["workflow_id"] == "flow"
    service.start("task", "bundle", actor="engineer")
    workflows.create.assert_called_once()
    assert workflows.start_collection.call_count == 2
    assert service.start("task", "bundle", actor="engineer")["automatic_collection"]["state"] == "blocked"
    assert executor.stop.call_count == 0


def test_disabled_collection_and_concurrent_stopping_never_start_an_orphan_collector():
    service, executor, workflows = setup(enabled=False)
    service.start("task", "bundle", actor="engineer")
    workflows.create.assert_not_called()
    service, executor, workflows = setup()
    executor.status.return_value = {"state": "stopped", "worker": {"execution_id": "execution-1"}}
    assert service.start("task", "bundle", actor="engineer")["automatic_collection"]["state"] == "blocked"
    workflows.start_collection.assert_not_called()


def test_failed_first_start_retries_real_collector_with_same_workflow():
    service, _executor, workflows = setup()
    start = workflows.start_collection.side_effect
    attempts = []

    def fail_then_start(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("source unavailable before collection claim")
        return start(*args, **kwargs)

    workflows.start_collection.side_effect = fail_then_start
    first = service.start("task", "bundle", actor="engineer")
    assert first["automatic_collection"]["state"] == "blocked"
    assert workflows._flow["state"] == "collectable"
    second = service.start("task", "bundle", actor="engineer")
    assert second["automatic_collection"]["state"] == "collecting"
    assert second["automatic_collection"]["workflow_id"] == first["automatic_collection"]["workflow_id"]
    assert workflows.start_collection.call_count == 2
    workflows.create.assert_called_once()


def test_missing_owned_runtime_does_not_turn_collecting_metadata_into_success():
    service, _executor, workflows = setup()

    def lost_claim(*args, **kwargs):
        workflows._flow.update(state="collecting", revision=2, collection_runtime=None,
                               operation={"kind": "collection", "collection_id": kwargs["collection_id"]})
        raise RuntimeError("unknown database commit outcome")

    workflows.start_collection.side_effect = lost_claim
    workflows.stop_collection.side_effect = RuntimeError("owner cleanup not confirmed")
    first = service.start("task", "bundle", actor="engineer")
    second = service.start("task", "bundle", actor="engineer")
    assert first["automatic_collection"]["state"] == "cleanup_pending"
    assert second["automatic_collection"]["state"] == "cleanup_pending"
    workflows.stop_collection.assert_called_once()
    workflows.create.assert_called_once()


def test_synchronous_worker_failure_is_blocked_and_retried_in_same_workflow():
    service, _executor, workflows = setup()
    start = workflows.start_collection.side_effect
    attempts = []

    def failed_worker(*args, **kwargs):
        attempts.append(True)
        if len(attempts) > 1:
            return start(*args, **kwargs)
        workflows._flow.update(state="collection_failed", revision=2, collection_runtime={
            "collection_id": kwargs["collection_id"], "state": "failed", "error_code": "COLLECTION_START_FAILED",
            "persistence_pending": False, "cleanup_confirmed": True,
        })
        return deepcopy(workflows._flow)

    workflows.start_collection.side_effect = failed_worker
    first = service.start("task", "bundle", actor="engineer")
    assert first["automatic_collection"]["state"] == "blocked"
    assert first["automatic_collection"]["reason_code"] == "COLLECTION_START_FAILED"
    assert service.start("task", "bundle", actor="engineer")["automatic_collection"]["state"] == "collecting"
    workflows.create.assert_called_once()


def test_pending_manifest_write_cannot_be_reported_as_collecting_or_stopped():
    service, executor, workflows = setup()
    service.start("task", "bundle", actor="engineer")
    workflows._flow["collection_runtime"].update(state="failed", persistence_pending=True, cleanup_confirmed=True)
    workflows.stop_collection.side_effect = RuntimeError("receipt commit still unavailable")
    retried = service.start("task", "bundle", actor="engineer")
    assert retried["automatic_collection"]["state"] == "cleanup_pending"
    executor.stop.return_value = {"task_id": "task", "state": "stopped"}
    stopped = service.stop("task", actor="engineer")
    assert stopped["automatic_collection"]["state"] == "cleanup_pending"
    assert stopped["automatic_collection"]["pending_workflow_ids"] == ["flow"]


def test_unknown_create_commit_preserves_original_idempotency_identity_on_retry():
    service, _executor, workflows = setup()
    attempts = []

    def create(*args, **kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RuntimeError("create committed, acknowledgement lost")
        return deepcopy(workflows._flow)

    workflows.create.side_effect = create
    assert service.start("task", "bundle", actor="engineer")["automatic_collection"]["state"] == "blocked"
    assert service.start("task", "bundle", actor="second-engineer")["automatic_collection"]["state"] == "collecting"
    assert attempts[0] == attempts[1]
    assert attempts[1]["actor"] == "engineer"


def test_late_model_start_after_stop_does_not_start_collection_or_return_old_running_state():
    service, executor, workflows = setup()
    started, release = threading.Event(), threading.Event()
    old = deepcopy(executor.start.return_value)

    def delayed_start(*args, **kwargs):
        started.set()
        assert release.wait(2)
        return old

    executor.start.side_effect = delayed_start

    def stop(*args, **kwargs):
        executor.status.return_value = {"task_id": "task", "state": "stopped"}
        return executor.status.return_value

    executor.stop.side_effect = stop
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(service.start, "task", "bundle", actor="engineer")
        assert started.wait(2)
        assert service.stop("task", actor="engineer")["state"] == "stopped"
        release.set()
        late = future.result(timeout=2)
    assert late["state"] == "stopped"
    assert late["automatic_collection"]["reason_code"] == "COMMAND_SUPERSEDED"
    workflows.create.assert_not_called()


def test_stop_retries_cleanup_instead_of_trusting_noncollecting_database_state():
    service, executor, workflows = setup()
    service.start("task", "bundle", actor="engineer")
    workflows._flow.update(state="collection_failed")
    workflows._flow["collection_runtime"].update(state="failed", cleanup_confirmed=False)
    executor.stop.return_value = {"task_id": "task", "state": "stopped"}
    stopped = service.stop("task", actor="engineer")
    assert stopped["automatic_collection"]["state"] == "stopped"
    assert stopped["automatic_collection"]["pending_workflow_ids"] == []
    workflows.stop_collection.assert_called_once_with("flow", expected_revision=2, actor="engineer", expected_collection_id=ANY)


def test_new_execution_drains_prior_automatic_collector_before_starting_its_own():
    service, executor, workflows = setup()
    service.start("task", "bundle", actor="engineer")
    flows = {"flow": deepcopy(workflows._flow), "new-flow": {
        "workflow_id": "new-flow", "revision": 1, "state": "collectable", "collection_runtime": None,
    }}
    events = []
    workflows.get.side_effect = lambda workflow_id: deepcopy(flows[workflow_id])
    workflows.create.side_effect = lambda *args, **kwargs: deepcopy(flows["new-flow"])

    def stop_collection(workflow_id, **kwargs):
        events.append(("stop", workflow_id))
        flows[workflow_id].update(state="review", revision=3, collection_runtime={
            "collection_id": kwargs["expected_collection_id"], "state": "stopped",
            "persistence_pending": False, "cleanup_confirmed": True,
        }, operation=None)

    def start_collection(workflow_id, **kwargs):
        events.append(("start", workflow_id))
        flows[workflow_id].update(state="collecting", revision=2, collection_runtime={
            "collection_id": kwargs["collection_id"], "state": "collecting",
            "persistence_pending": False, "cleanup_confirmed": False,
        }, operation={"kind": "collection", "collection_id": kwargs["collection_id"]})

    workflows.stop_collection.side_effect = stop_collection
    workflows.start_collection.side_effect = start_collection
    newer = {"task_id": "task", "state": "running", "worker": {"execution_id": "execution-2", "state": "running"}}
    executor.start.return_value = executor.status.return_value = newer
    result = service.start("task", "new-bundle", actor="engineer")
    assert result["automatic_collection"]["state"] == "collecting"
    assert events == [("stop", "flow"), ("start", "new-flow")]
    assert flows["flow"]["collection_runtime"]["cleanup_confirmed"]


def test_prior_execution_cleanup_failure_blocks_new_automatic_collection_without_duplicate_workflow():
    service, executor, workflows = setup()
    service.start("task", "bundle", actor="engineer")
    newer = {"task_id": "task", "state": "running", "worker": {"execution_id": "execution-2", "state": "running"}}
    executor.start.return_value = executor.status.return_value = newer
    workflows.stop_collection.side_effect = RuntimeError("old collector still owned")
    result = service.start("task", "new-bundle", actor="engineer")
    assert result["state"] == "running"
    assert result["automatic_collection"] == {
        "state": "cleanup_pending", "reason_code": "PRIOR_EXECUTION_COLLECTION_CLEANUP_PENDING",
        "pending_workflow_ids": ["flow"],
    }
    workflows.create.assert_called_once()
    workflows.start_collection.assert_called_once()


def test_late_stop_does_not_stop_collection_belonging_to_newer_execution():
    service, executor, workflows = setup()
    service.start("task", "bundle", actor="engineer")
    flows = {"flow": deepcopy(workflows._flow), "new-flow": {
        "workflow_id": "new-flow", "revision": 1, "state": "collectable", "collection_runtime": None,
    }}
    workflows.get.side_effect = lambda workflow_id: deepcopy(flows[workflow_id])
    workflows.create.side_effect = lambda *args, **kwargs: deepcopy(flows["new-flow"])

    def start_collection(workflow_id, **kwargs):
        flows[workflow_id].update(state="collecting", revision=2, collection_runtime={
            "collection_id": kwargs["collection_id"], "state": "collecting",
            "persistence_pending": False, "cleanup_confirmed": False,
        }, operation={"kind": "collection", "collection_id": kwargs["collection_id"]})
        return deepcopy(flows[workflow_id])

    def stop_collection(workflow_id, **kwargs):
        flows[workflow_id].update(state="review", revision=3, collection_runtime={
            "collection_id": kwargs["expected_collection_id"], "state": "stopped",
            "persistence_pending": False, "cleanup_confirmed": True,
        }, operation=None)
        return deepcopy(flows[workflow_id])

    workflows.start_collection.side_effect = start_collection
    workflows.stop_collection.side_effect = stop_collection
    entered, release = threading.Event(), threading.Event()

    def delayed_stop(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return {"task_id": "task", "state": "stopped"}

    executor.stop.side_effect = delayed_stop
    with ThreadPoolExecutor(max_workers=2) as pool:
        stopped = pool.submit(service.stop, "task", actor="engineer")
        assert entered.wait(2)
        newer = {"task_id": "task", "state": "running", "worker": {"execution_id": "execution-2", "state": "running"}}
        executor.start.return_value = newer
        executor.status.return_value = newer
        assert service.start("task", "new-bundle", actor="engineer")["automatic_collection"]["state"] == "collecting"
        release.set()
        late = stopped.result(timeout=2)
    assert late["state"] == "running"
    assert late["worker"]["execution_id"] == "execution-2"
    assert late["automatic_collection"]["reason_code"] == "COMMAND_SUPERSEDED"
    workflows.stop_collection.assert_called_once_with("flow", expected_revision=2, actor="engineer", expected_collection_id=ANY)
    assert flows["new-flow"]["collection_runtime"]["state"] == "collecting"


def test_ready_worker_waiting_for_first_frame_starts_collector_without_user_retry():
    service, executor, workflows = setup()
    executor.start.return_value["state"] = "waiting_source"
    result = service.start("task", "bundle", actor="engineer")
    assert result["state"] == "waiting_source"
    assert result["automatic_collection"]["state"] == "collecting"
    workflows.start_collection.assert_called_once()


def test_worker_loading_is_not_mistaken_for_waiting_for_first_frame():
    service, executor, workflows = setup()
    executor.start.return_value["state"] = "waiting_source"
    executor.start.return_value["worker"]["state"] = "starting"
    assert service.start("task", "bundle", actor="engineer")["automatic_collection"]["state"] == "blocked"
    workflows.create.assert_not_called()


def test_lost_collection_start_ack_after_execution_stop_drains_owned_collector():
    service, executor, workflows = setup()
    start = workflows.start_collection.side_effect

    def start_then_lose_ack(*args, **kwargs):
        start(*args, **kwargs)
        executor.status.return_value = {"task_id": "task", "state": "stopped"}
        raise RuntimeError("collection started but acknowledgement lost")

    workflows.start_collection.side_effect = start_then_lose_ack
    result = service.start("task", "bundle", actor="engineer")
    assert result["state"] == "stopped"
    assert result["automatic_collection"]["state"] == "blocked"
    assert result["automatic_collection"]["reason_code"] == "EXECUTION_CHANGED"
    workflows.stop_collection.assert_called_once_with("flow", expected_revision=2, actor="engineer", expected_collection_id=ANY)


def test_model_cleanup_failure_still_drains_collector_and_preserves_failure():
    service, executor, workflows = setup()
    service.start("task", "bundle", actor="engineer")
    executor.stop.side_effect = RuntimeError("worker cleanup not confirmed")
    with pytest.raises(RuntimeError, match="worker cleanup not confirmed"):
        service.stop("task", actor="engineer")
    workflows.stop_collection.assert_called_once_with("flow", expected_revision=2, actor="engineer", expected_collection_id=ANY)
    assert workflows._flow["collection_runtime"]["cleanup_confirmed"]


def test_old_execution_cannot_stop_or_claim_a_manually_reused_workflow_collector():
    service, executor, workflows = setup()
    service.start("task", "bundle", actor="engineer")
    manual_id = "manual-collection"
    workflows._flow.update(state="collecting", revision=8,
        operation={"kind": "collection", "collection_id": manual_id}, collection_runtime={
            "collection_id": manual_id, "state": "collecting", "cleanup_confirmed": False,
            "persistence_pending": False,
        })
    retry = service.start("task", "bundle", actor="engineer")
    assert retry["automatic_collection"]["reason_code"] == "COLLECTION_OWNERSHIP_CHANGED"
    executor.stop.return_value = {"task_id": "task", "state": "stopped"}
    assert service.stop("task", actor="engineer")["automatic_collection"]["state"] == "stopped"
    workflows.stop_collection.assert_not_called()
    workflows.start_collection.assert_called_once()
    assert workflows._flow["collection_runtime"]["state"] == "collecting"


def test_failed_retry_before_metadata_commit_keeps_prior_collection_identity_recoverable():
    service, _executor, workflows = setup()
    start = workflows.start_collection.side_effect
    attempts = []

    def fail_in_two_stages(*args, **kwargs):
        attempts.append(kwargs["collection_id"])
        if len(attempts) == 1:
            workflows._flow.update(state="collection_failed", revision=2, collection_runtime={
                "collection_id": kwargs["collection_id"], "state": "failed", "error_code": "COLLECTION_START_FAILED",
                "cleanup_confirmed": True, "persistence_pending": False,
            })
        elif len(attempts) == 2:
            raise RuntimeError("metadata transaction rolled back before commit")
        else:
            return start(*args, **kwargs)

    workflows.start_collection.side_effect = fail_in_two_stages
    assert service.start("task", "bundle", actor="engineer")["automatic_collection"]["state"] == "blocked"
    assert service.start("task", "bundle", actor="engineer")["automatic_collection"]["state"] == "blocked"
    assert service.start("task", "bundle", actor="engineer")["automatic_collection"]["state"] == "collecting"
    assert len(attempts) == len(set(attempts)) == 3
    workflows.create.assert_called_once()


def test_durable_deployment_hook_only_reconciles_collection_and_never_starts_a_worker():
    service, executor, workflows = setup()
    executor.status.return_value["deployment_operation_id"] = "deploy-current"
    operation = {"task_id": "task", "action": "deploy", "operation_id": "deploy-current", "requested_by": "engineer"}
    assert service.reconcile_deployment(operation)["automatic_collection"]["state"] == "collecting"
    service.reconcile_deployment(operation)
    executor.start.assert_not_called()
    executor.stop.assert_not_called()
    workflows.start_collection.assert_called_once()


def test_late_durable_hook_cannot_resurrect_a_stopped_or_replaced_execution():
    service, executor, workflows = setup()
    operation = {"task_id": "task", "action": "deploy", "operation_id": "deploy-old", "requested_by": "engineer"}
    executor.status.return_value["deployment_operation_id"] = "deploy-new"
    assert service.reconcile_deployment(operation)["automatic_collection"]["reason_code"] == "COMMAND_SUPERSEDED"
    executor.status.return_value = {"task_id": "task", "state": "stopped"}
    assert service.reconcile_deployment(operation)["automatic_collection"]["reason_code"] == "COMMAND_SUPERSEDED"
    workflows.create.assert_not_called()
    executor.start.assert_not_called()
    executor.stop.assert_not_called()


def test_late_durable_stop_hook_cannot_stop_a_new_execution_collection():
    service, executor, workflows = setup()
    executor.status.return_value["deployment_operation_id"] = "deploy-new"
    current = {"task_id": "task", "action": "deploy", "operation_id": "deploy-new", "requested_by": "engineer"}
    service.reconcile_deployment(current)
    late_stop = {"task_id": "task", "action": "stop", "operation_id": "stop-old", "requested_by": "engineer"}
    assert service.reconcile_deployment(late_stop)["automatic_collection"]["state"] == "stopped"
    workflows.stop_collection.assert_not_called()
    executor.start.assert_not_called()
    executor.stop.assert_not_called()
