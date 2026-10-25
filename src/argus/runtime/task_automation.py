"""Coordinate explicit task collection policy with a shadow execution session."""

import hashlib
import threading
import uuid


class TaskAutomationService:
    def __init__(self, task_service, executor, workflows, camera_manager):
        self._tasks, self._executor, self._workflows, self._manager = task_service, executor, workflows, camera_manager
        self._lock = threading.RLock()
        self._automatic: dict[tuple[str, str], str] = {}
        self._creation_requests: dict[tuple[str, str], tuple[str, str]] = {}
        self._owned_collections: dict[tuple[str, str], set[str]] = {}
        self._generations: dict[str, int] = {}

    def _new_command(self, task_id: str) -> int:
        generation = self._generations.get(task_id, 0) + 1
        self._generations[task_id] = generation
        return generation

    def _superseded(self, task_id: str, fallback: dict) -> dict:
        current = self._executor.status(task_id)
        return {**(current if isinstance(current, dict) else {**fallback, "state": "blocked"}),
                "automatic_collection": {"state": "blocked", "reason_code": "COMMAND_SUPERSEDED"}}

    @staticmethod
    def _execution_ready(status: dict | None, execution_id: str | None = None) -> bool:
        if not isinstance(status, dict) or status.get("state") not in {"running", "waiting_source"}:
            return False
        worker = status.get("worker")
        return (isinstance(worker, dict) and worker.get("state") == "running"
                and isinstance(worker.get("execution_id"), str) and bool(worker["execution_id"])
                and (execution_id is None or worker["execution_id"] == execution_id))

    @staticmethod
    def _collection_identity(workflow: dict) -> str | None:
        operation = workflow.get("operation")
        # A durable collecting claim supersedes a prior local terminal worker,
        # including an unknown-commit/manual-retry interval before allocation.
        if workflow["state"] == "collecting" and isinstance(operation, dict):
            return operation.get("collection_id")
        receipts = workflow.get("manifest", {}).get("collections", [])
        if receipts:
            return receipts[-1].get("collection_id")
        runtime = workflow.get("collection_runtime")
        return runtime.get("collection_id") if isinstance(runtime, dict) else None

    @staticmethod
    def _collection_status(workflow: dict, expected_collection_ids: set[str] | None = None) -> dict:
        status = {"workflow_id": workflow["workflow_id"], "workflow_revision": workflow["revision"]}
        identity = TaskAutomationService._collection_identity(workflow)
        if expected_collection_ids is not None and identity not in expected_collection_ids:
            return {**status, "state": "blocked", "reason_code": "COLLECTION_OWNERSHIP_CHANGED"}
        runtime = workflow.get("collection_runtime")
        if not isinstance(runtime, dict) or runtime.get("collection_id") != identity:
            return {**status, "state": "cleanup_pending" if workflow["state"] == "collecting" else "blocked",
                    "reason_code": "COLLECTION_RUNTIME_UNAVAILABLE"}
        if runtime.get("persistence_pending") is True:
            return {**status, "state": "cleanup_pending", "reason_code": "COLLECTION_PERSISTENCE_PENDING"}
        state = runtime.get("state")
        if state == "collecting":
            if workflow["state"] == "collecting":
                return {**status, "state": "collecting"}
            return {**status, "state": "cleanup_pending", "reason_code": "COLLECTION_STATE_DIVERGED"}
        if runtime.get("cleanup_confirmed") is not True:
            return {**status, "state": "cleanup_pending", "reason_code": "COLLECTION_CLEANUP_NOT_CONFIRMED"}
        if state in {"completed", "stopped"} and workflow["state"] not in {
            "collectable", "collecting", "collection_failed",
        }:
            return {**status, "state": "completed"}
        return {**status, "state": "blocked", "reason_code": runtime.get("error_code") or "COLLECTION_NOT_READY"}

    def _ensure_collection(self, workflow: dict, *, key: tuple[str, str], actor: str) -> dict:
        runtime = workflow.get("collection_runtime")
        owned_ids = self._owned_collections.setdefault(key, set())
        observed_id = self._collection_identity(workflow)
        if observed_id is not None and observed_id not in owned_ids:
            raise RuntimeError("COLLECTION_OWNERSHIP_CHANGED")
        if workflow["state"] == "collecting":
            if observed_id is None:
                raise RuntimeError("COLLECTION_OWNERSHIP_UNRECORDED")
            if (isinstance(runtime, dict) and runtime.get("collection_id") == observed_id and runtime.get("state") == "collecting"
                    and runtime.get("persistence_pending") is not True):
                return workflow
            # The metadata claim alone does not prove that a collector exists.
            # The workflow service owns recovery and must confirm cleanup.
            self._workflows.stop_collection(
                workflow["workflow_id"], expected_revision=workflow["revision"], actor=actor,
                expected_collection_id=observed_id,
            )
            workflow = self._workflows.get(workflow["workflow_id"])
        if workflow["state"] in {"collectable", "collection_failed"}:
            if not self._drain_collection(workflow["workflow_id"], collection_ids=owned_ids, actor=actor):
                raise RuntimeError("COLLECTION_CLEANUP_NOT_CONFIRMED")
            workflow = self._workflows.get(workflow["workflow_id"])
            if (self._collection_identity(workflow) is not None
                    and self._collection_identity(workflow) not in owned_ids):
                raise RuntimeError("COLLECTION_OWNERSHIP_CHANGED")
            collection_id = uuid.uuid4().hex
            # Preserve ownership before the metadata transaction, so an
            # acknowledgement loss cannot leave an unidentifiable collector.
            owned_ids.add(collection_id)
            self._workflows.start_collection(
                workflow["workflow_id"], expected_revision=workflow["revision"],
                manager=self._manager, actor=actor, params={}, collection_id=collection_id,
            )
            # Read the actual owned worker, including a synchronous start
            # failure or a receipt that committed before acknowledgement loss.
            workflow = self._workflows.get(workflow["workflow_id"])
        return workflow

    def _drain_collection(self, workflow_id: str, *, collection_ids: set[str] | None, actor: str) -> bool:
        """Drain this workflow's owned collector, not a newer execution's one."""
        def pending(workflow):
            runtime = workflow.get("collection_runtime")
            return workflow["state"] == "collecting" or isinstance(runtime, dict) and (
                runtime.get("state") == "collecting" or runtime.get("persistence_pending") is True
                or runtime.get("cleanup_confirmed") is not True
            )

        workflow = self._workflows.get(workflow_id)
        collection_id = self._collection_identity(workflow)
        if not collection_ids or collection_id not in collection_ids:
            # Workflow reuse does not transfer collector ownership. The
            # service only replaces a local worker after confirmed cleanup.
            return True
        if pending(workflow):
            self._workflows.stop_collection(
                workflow_id, expected_revision=workflow["revision"], actor=actor,
                expected_collection_id=collection_id,
            )
            workflow = self._workflows.get(workflow_id)
            if self._collection_identity(workflow) != collection_id:
                return True
        return not pending(workflow)

    def start(self, task_id: str, bundle_id: str, *, actor: str) -> dict:
        # Worker loading must remain interruptible by stop(), so no coordinator
        # lock is held while the executor loads a model or confirms cleanup.
        with self._lock:
            generation = self._new_command(task_id)
        status = self._executor.start(task_id, bundle_id)
        return self._reconcile_collection(task_id, status, actor=actor, generation=generation)

    def _reconcile_collection(self, task_id: str, status: dict, *, actor: str, generation: int) -> dict:
        task = self._tasks.get_task(task_id)
        with self._lock:
            if self._generations[task_id] != generation:
                return self._superseded(task_id, status)
        if not task or not task["config"]["collection"]["enabled"]:
            with self._lock:
                return status if self._generations[task_id] == generation else self._superseded(task_id, status)
        if not self._execution_ready(status):
            return {**status, "automatic_collection": {"state": "blocked", "reason_code": "EXECUTION_NOT_READY"}}
        execution_id = status["worker"]["execution_id"]
        key = (task_id, execution_id)
        with self._lock:
            if self._generations[task_id] != generation:
                return self._superseded(task_id, status)
            try:
                current = self._executor.status(task_id)
                if not self._execution_ready(current, execution_id):
                    return {**(current or status), "automatic_collection": {
                        "state": "blocked", "reason_code": "EXECUTION_CHANGED"}}
                # A worker may stop independently of the coordinator (source
                # or model failure). A subsequent start must first retire only
                # its older execution collectors; otherwise both collections
                # can survive and consume frames under different ownership.
                stale_ids = {workflow_id: self._owned_collections.get((owner_task, owner_execution))
                             for (owner_task, owner_execution), workflow_id in self._automatic.items()
                             if owner_task == task_id and owner_execution != execution_id}
                pending_ids = []
                for stale_id in sorted(stale_ids):
                    try:
                        if not self._drain_collection(stale_id, collection_ids=stale_ids[stale_id], actor=actor):
                            pending_ids.append(stale_id)
                    except Exception:
                        pending_ids.append(stale_id)
                if pending_ids:
                    return {**current, "automatic_collection": {
                        "state": "cleanup_pending", "reason_code": "PRIOR_EXECUTION_COLLECTION_CLEANUP_PENDING",
                        "pending_workflow_ids": pending_ids,
                    }}
                workflow_id = self._automatic.get(key)
                if workflow_id is None:
                    digest = hashlib.sha256(f"task-collection:{task_id}:{execution_id}".encode()).hexdigest()
                    original_actor, digest = self._creation_requests.setdefault(key, (actor, digest))
                    workflow = self._workflows.create(task_id, actor=original_actor, idempotency_key=digest)
                    workflow_id = workflow["workflow_id"]
                    self._automatic[key] = workflow_id
                    # Recheck after the metadata transaction: a concurrent task
                    # stop must not leave an automatically started collector.
                    current = self._executor.status(task_id)
                    if not self._execution_ready(current, execution_id):
                        raise RuntimeError("EXECUTION_CHANGED")
                else:
                    workflow = self._workflows.get(workflow_id)
                workflow = self._ensure_collection(workflow, key=key, actor=actor)
                current = self._executor.status(task_id)
                if not self._execution_ready(current, execution_id):
                    automatic = self._collection_status(workflow, self._owned_collections.get(key))
                    if workflow["state"] == "collecting":
                        try:
                            if not self._drain_collection(
                                workflow_id, collection_ids=self._owned_collections.get(key), actor=actor,
                            ):
                                return {**(current or status), "automatic_collection": {
                                    **automatic, "state": "cleanup_pending", "reason_code": "EXECUTION_CHANGED"}}
                        except Exception:
                            return {**(current or status), "automatic_collection": {
                                **automatic, "state": "cleanup_pending", "reason_code": "EXECUTION_CHANGED"}}
                    return {**(current or status), "automatic_collection": {
                        **automatic, "state": "blocked", "reason_code": "EXECUTION_CHANGED"}}
                return {**current, "automatic_collection": self._collection_status(workflow, self._owned_collections.get(key))}
            except Exception:
                # Execution and collection are independent facts. Retain the
                # workflow identity for repair rather than claim a rolled-back
                # task start or create duplicate collections on retry.
                automatic = {"workflow_id": self._automatic.get(key), "state": "blocked",
                             "reason_code": "AUTOMATIC_COLLECTION_REQUIRES_RECONCILIATION"}
                if automatic["workflow_id"] is not None:
                    try:
                        automatic = self._collection_status(
                            self._workflows.get(automatic["workflow_id"]), self._owned_collections.get(key),
                        )
                    except Exception:
                        pass
                current = self._executor.status(task_id)
                if not self._execution_ready(current, execution_id) and automatic["state"] == "collecting":
                    try:
                        clean = self._drain_collection(automatic["workflow_id"],
                                                       collection_ids=self._owned_collections.get(key), actor=actor)
                        automatic.update(state="blocked" if clean else "cleanup_pending", reason_code="EXECUTION_CHANGED")
                    except Exception:
                        automatic.update(state="cleanup_pending", reason_code="EXECUTION_CHANGED")
                return {**(current if isinstance(current, dict) else status), "automatic_collection": automatic}

    def reconcile_deployment(self, operation: dict) -> dict:
        """Reconcile collection ownership without changing runtime intent.

        A late deployment hook must never call executor.start/stop: it could
        resurrect a stopped task or kill its replacement. Only the exact
        current execution may acquire collection; old owned collections may
        still be safely drained after a later deployment has superseded it.
        """
        task_id, actor = operation["task_id"], operation["requested_by"]
        with self._lock:
            current = self._executor.status(task_id)
            if operation["action"] != "stop":
                if (not isinstance(current, dict)
                        or current.get("deployment_operation_id") != operation["operation_id"]):
                    return {"automatic_collection": {"state": "blocked", "reason_code": "COMMAND_SUPERSEDED"}}
                generation = self._new_command(task_id)
            else:
                # Never stop a newer execution's collector from an old stop
                # receipt. Cleanup is fenced by exact collection ownership.
                active_execution = (current["worker"]["execution_id"]
                                    if self._execution_ready(current) else None)
                self._new_command(task_id)
                ids = {workflow_id: set(self._owned_collections.get((owner_task, execution), ()))
                       for (owner_task, execution), workflow_id in self._automatic.items()
                       if owner_task == task_id and execution != active_execution}
                failures = []
                for workflow_id, collection_ids in ids.items():
                    try:
                        if not self._drain_collection(workflow_id, collection_ids=collection_ids, actor=actor):
                            failures.append(workflow_id)
                    except Exception:
                        failures.append(workflow_id)
                return {"automatic_collection": {
                    "state": "cleanup_pending" if failures else "stopped", "pending_workflow_ids": failures,
                }}
        return self._reconcile_collection(task_id, current, actor=actor, generation=generation)

    def stop(self, task_id: str, *, actor: str) -> dict:
        with self._lock:
            generation = self._new_command(task_id)
            # A later start may install another execution while stop() waits
            # for its worker; it must not inherit this older cleanup request.
            ids = {workflow_id: set(self._owned_collections.get((owner_task, execution), ()))
                   for (owner_task, execution), workflow_id in self._automatic.items() if owner_task == task_id}
        stop_error = None
        try:
            status = self._executor.stop(task_id)
        except Exception as exc:
            # Collector ownership is independent from a failed model-worker
            # cleanup, so still drain it before preserving the executor error.
            stop_error = exc
            status = {"task_id": task_id, "state": "cleanup_pending"}
        failures = []
        with self._lock:
            for workflow_id in ids:
                try:
                    if not self._drain_collection(workflow_id, collection_ids=ids[workflow_id], actor=actor):
                        failures.append(workflow_id)
                except Exception:
                    failures.append(workflow_id)
            if self._generations[task_id] != generation:
                result = self._superseded(task_id, status)
                if failures:
                    result["automatic_collection"].update(state="cleanup_pending", pending_workflow_ids=failures)
                return result
        if stop_error is not None:
            raise stop_error
        return {**status, "automatic_collection": {
            "state": "cleanup_pending" if failures else "stopped", "pending_workflow_ids": failures,
        }}
