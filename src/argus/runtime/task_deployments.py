"""Exclusive runtime reconciler for durable shadow-task deployments."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
import os
from pathlib import Path
import threading
import time
from uuid import uuid4

import structlog

logger = structlog.get_logger()
_OWNERS_LOCK = threading.Lock()
_OWNERS: dict[str, object] = {}


class _RuntimeOwner:
    """OS advisory lock, held until all owned work is confirmed cleaned.

    The file is not a stale PID lease. Process death releases the OS lock;
    deleting a lock file or expiring a wall-clock lease is never required.
    """

    def __init__(self, db):
        url = db._database_url
        if not url.startswith("sqlite:///"):
            raise RuntimeError("task runtime ownership requires a local SQLite database")
        database = url.removeprefix("sqlite:///")
        self.path = None if database == ":memory:" else Path(database).resolve().with_suffix(".task-runtime.lock")
        self.key = str(self.path) if self.path else f"in-memory:{id(db)}"
        self.stream = None
        self.held = False

    def acquire(self):
        with _OWNERS_LOCK:
            if self.key in _OWNERS:
                return False
            # Keep a strong reference as well as a key. If an application drops
            # its coordinator after unknown cleanup, GC must not close the OS
            # lock and let another process inherit a possibly live worker.
            _OWNERS[self.key] = self
        try:
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.stream = self.path.open("a+b")
                if self.stream.seek(0, 2) == 0:
                    self.stream.write(b"0")
                    self.stream.flush()
                self.stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.held = True
            return True
        except (OSError, ValueError):
            if self.stream:
                self.stream.close()
                self.stream = None
            with _OWNERS_LOCK:
                _OWNERS.pop(self.key, None)
            return False

    def release(self):
        if not self.held:
            return
        # If unlock/close raises, retain ownership in the process registry.
        # Another reconciler must not inherit an unknown runtime owner.
        if self.stream:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None
        self.held = False
        with _OWNERS_LOCK:
            _OWNERS.pop(self.key, None)


class TaskDeploymentCoordinator:
    def __init__(self, service, executor, *, poll_interval=1.0, on_observed=None):
        self.service, self.executor = service, executor
        self._poll_interval = max(0.05, float(poll_interval))
        self._owner = _RuntimeOwner(service._db)
        self._boot_id = uuid4().hex
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.RLock()
        self._thread = None
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="task-deploy")
        self._stop_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="task-deploy-stop")
        self._jobs = {}
        self._recovered = {}
        self._accepting = False
        self._error_code = None
        self._on_observed = on_observed
        self._hooked = {}
        self._hook_inflight = set()
        self._hook_errors = {}

    def start(self):
        with self._lock:
            if self._accepting:
                return True
            if self._stop.is_set() or not self._owner.acquire():
                self._error_code = "RUNTIME_OWNER_UNAVAILABLE"
                return False
            self._accepting = True
            self._thread = threading.Thread(target=self._loop, name="task-deployment-reconciler", daemon=True)
            try:
                self._thread.start()
            except Exception:
                self._accepting = False
                self._owner.release()
                raise
            return True

    def status(self):
        with self._lock:
            return {"available": self._accepting and self._owner.held,
                    "owner_held": self._owner.held, "active_operations": len(self._jobs),
                    "error_code": self._error_code, "automation_errors": dict(self._hook_errors),
                    "production_allowed": False}

    def _after_receipt(self, operation):
        if operation["status"] != "applied":
            return operation
        with self._lock:
            if (self._hooked.get(operation["task_id"]) == operation["operation_id"]
                    or operation["operation_id"] in self._hook_inflight):
                return operation
            self._hook_inflight.add(operation["operation_id"])
        try:
            current = self.service.get(operation["task_id"])
            runtime = self.executor.status(operation["task_id"])
            if current["operation_id"] != operation["operation_id"]:
                return operation
            if operation["action"] == "stop":
                if runtime is not None and runtime.get("state") != "stopped":
                    return operation
            elif (runtime is None or runtime.get("deployment_operation_id") != operation["operation_id"]
                  or runtime.get("deployment_revision") != operation["deployment_revision"]
                  or runtime.get("state") not in {"running", "waiting_source", "blocked", "sampling_insufficient"}):
                return operation
            # This notification may reconcile collection only. It must not
            # start/stop execution; its recipient rechecks these exact tokens
            # before accepting side effects against a newer command.
            if self._on_observed is not None:
                outcome = self._on_observed(operation)
                if isinstance(outcome, dict):
                    automatic = outcome.get("automatic_collection", outcome)
                    if isinstance(automatic, dict) and automatic.get("state") in {"blocked", "cleanup_pending"}:
                        raise RuntimeError("automatic collection still requires reconciliation")
            with self._lock:
                self._hooked[operation["task_id"]] = operation["operation_id"]
                self._hook_errors.pop(operation["task_id"], None)
        except Exception:
            with self._lock:
                self._hook_errors[operation["task_id"]] = "AUTOMATION_RECONCILIATION_FAILED"
        finally:
            with self._lock:
                self._hook_inflight.discard(operation["operation_id"])
        return operation

    def submit(self, operation_id):
        operation = self.service.get_operation(operation_id)
        with self._lock:
            if not self._accepting or not self._owner.held:
                raise RuntimeError("RUNTIME_OWNER_UNAVAILABLE")
        self._wake.set()
        return operation

    def process(self, operation_id):
        """Run one accepted intent; never hold a global lock during loading."""
        if not self._owner.held or self._stop.is_set():
            return None
        operation = self.service.claim(operation_id)
        if operation is None:
            return self.service.get_operation(operation_id)
        try:
            if self.service.get(operation["task_id"])["operation_id"] != operation_id:
                return self.service.get_operation(operation_id)
            if operation["action"] == "stop":
                runtime = self.executor.stop(operation["task_id"], operation_id=operation_id,
                    deployment_revision=operation["deployment_revision"])
            else:
                runtime = self.executor.deploy(
                    operation["task_id"], operation["bundle_id"],
                    operation_id=operation_id, deployment_revision=operation["deployment_revision"],
                )
            return self._after_receipt(self.service.observe(operation_id, runtime))
        except Exception as exc:
            code = getattr(exc, "code", "DEPLOYMENT_FAILED")
            if code in {"CANCELLED", "SOURCE_CHANGED"} and not self.executor.source_available(operation["task_id"]):
                code = "SOURCE_UNAVAILABLE"
            try:
                runtime = self.executor.status(operation["task_id"])
            except Exception:
                runtime = None
            try:
                return self._after_receipt(self.service.observe(operation_id, runtime, failure_code=code,
                    uncertain=code in {"CLEANUP_PENDING", "RESTORE_FAILED"}))
            except Exception:
                # Applying may have succeeded while the DB acknowledgement was
                # lost. The next pass observes this exact operation identity.
                logger.warning("task_deployment.receipt_unknown", operation_id=operation_id)
                return None

    def reconcile(self, task_id):
        with self._lock:
            if not self._owner.held:
                raise RuntimeError("RUNTIME_OWNER_UNAVAILABLE")
            deployment = self.service.get(task_id)
            operation = deployment["operation"]
            if operation is None or operation["operation_id"] in self._jobs:
                return deployment
        if operation["status"] == "applied":
            self._after_receipt(operation)
            runtime = self.executor.status(task_id)
            self.service.observe_runtime(task_id, operation["operation_id"], runtime)
            if (deployment["desired_state"] == "running" and runtime is not None
                    and runtime.get("state") in {"failed", "stopped", "cleanup_pending"}):
                try:
                    self.executor.stop(task_id, operation_id=operation["operation_id"],
                        deployment_revision=operation["deployment_revision"])
                except Exception:
                    pass  # Retained ownership must remain visible, never discarded.
        if operation["status"] in {"running", "reconcile_required"}:
            runtime = self.executor.status(task_id)
            if runtime is not None and runtime.get("deployment_operation_id") == operation["operation_id"]:
                self._after_receipt(self.service.observe(operation["operation_id"], runtime))
            elif self.executor.cleanup_confirmed(task_id):
                self.service.requeue_owned_recovery(operation["operation_id"])
                self._wake.set()
        return self.service.get(task_id)

    def recover(self):
        """Recover DB-authorized bundles only, never open a source or scan files."""
        if not self._owner.held or self._stop.is_set():
            return []
        recovered = []
        for deployment in self.service.list_recoverable():
            task_id = deployment["task_id"]
            current = self.service.get(task_id)
            operation = current["operation"]
            if operation and operation["status"] in {"pending", "running", "waiting_source", "waiting_scene", "reconcile_required"}:
                self.reconcile(task_id)
                continue
            if operation and operation["status"] == "applied":
                self.reconcile(task_id)
            runtime = self.executor.status(task_id)
            if (operation and self._recovered.get(task_id) == operation["operation_id"] and operation["status"] == "failed"
                    or runtime is not None and runtime.get("state") not in {"stopped", "failed"}
                    or not self.executor.cleanup_confirmed(task_id)):
                continue
            # A failed attempted upgrade is not permission to activate it on
            # restart. Restore only the last successfully applied bundle.
            bundle = current["applied_bundle_id"]
            if not bundle:
                continue
            try:
                command = self.service.request(task_id, action="deploy", bundle_id=bundle,
                    expected_deployment_revision=current["deployment_revision"],
                    requested_by=self.service.recovery_actor(task_id, bundle),
                    idempotency_key=f"recover:{self._boot_id}:{task_id}:{current['deployment_revision']}")
                self._recovered[task_id] = command["operation_id"]
                recovered.append(command)
            except Exception:
                # Archived/config-changed tasks remain blocked; recovery cannot
                # fabricate a matching configuration or bypass scope checks.
                logger.warning("task_deployment.recovery_blocked", task_id=task_id)
        return recovered

    def _loop(self):
        while not self._stop.is_set():
            try:
                with self._lock:
                    finished = [key for key, future in self._jobs.items() if future.done()]
                    for key in finished:
                        self._jobs.pop(key)
                self.recover()
                with self._lock:
                    hook_retries = list(self._hook_errors)
                for task_id in hook_retries:
                    self.reconcile(task_id)
                pending = self.service.list_pending()
                for operation in pending:
                    if self._stop.is_set():
                        break
                    operation_id = operation["operation_id"]
                    if operation["status"] in {"running", "reconcile_required"}:
                        self.reconcile(operation["task_id"])
                        continue
                    with self._lock:
                        if operation_id in self._jobs:
                            continue
                        stops = sum(self.service.get_operation(key)["action"] == "stop" for key in self._jobs)
                        limit = 2 if operation["action"] == "stop" else 4
                        active = stops if operation["action"] == "stop" else len(self._jobs) - stops
                        if active >= limit:
                            continue
                        pool = self._stop_pool if operation["action"] == "stop" else self._pool
                        self._jobs[operation_id] = pool.submit(self.process, operation_id)
                self._error_code = None
            except Exception:
                self._error_code = "RECONCILIATION_FAILED"
                logger.warning("task_deployment.reconciliation_failed")
            self._wake.wait(self._poll_interval)
            self._wake.clear()

    def close(self, *, wait=False):
        with self._lock:
            self._accepting = False
            self._stop.set()
            self._wake.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._stop_pool.shutdown(wait=False, cancel_futures=True)
        return self.join(timeout=5.0 if wait else 0.0)

    def join(self, timeout=5.0):
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lock:
            jobs = list(self._jobs.values())
        if jobs:
            _, unfinished = wait(jobs, timeout=max(0.0, deadline - time.monotonic()))
            if unfinished:
                return False
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                return False
        if not self.executor.cleanup_confirmed():
            return False
        try:
            self._owner.release()
        except Exception:
            self._error_code = "RUNTIME_OWNER_RELEASE_FAILED"
            return False
        return True
