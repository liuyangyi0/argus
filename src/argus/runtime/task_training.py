"""Explicit, resource-admitted spawn training. Default construction is inert."""

from __future__ import annotations

import multiprocessing
import os
import copy
from pathlib import Path
import threading
import time
from dataclasses import dataclass
from uuid import uuid4

from argus.core.task_training import artifact_sha256
from argus.storage.task_workflows import TaskWorkflowBlocked, TaskWorkflowConflict, _process_alive

_OWNERS_LOCK = threading.Lock()
_OWNERS = {}
_QUEUED_STATES = {"queued", "waiting_resources"}
_EXECUTING_STATES = {"running", "cancelling", "reconciling"}
_TERMINAL_STATES = {"complete", "failed", "cancelled"}
_KNOWN_STATES = _QUEUED_STATES | _EXECUTING_STATES | _TERMINAL_STATES


def _safe_error_code(value, fallback="TRAINING_RESULT_INVALID"):
    if (isinstance(value, str) and value.startswith("TRAINING_") and len(value) <= 96
            and all("A" <= char <= "Z" or "0" <= char <= "9" or char == "_" for char in value)):
        return value
    return fallback


def _worker_message(value):
    if (not isinstance(value, dict) or not isinstance(value.get("state"), str)
            or value["state"] not in {"complete", "failed", "cancelled"}):
        return {"state": "failed", "error_code": "TRAINING_RESULT_INVALID"}
    if value["state"] == "failed":
        return {"state": "failed", "error_code": _safe_error_code(value.get("error_code"))}
    return value


class _TrainingOwner:
    def __init__(self, database):
        url = database._database_url
        if not url.startswith("sqlite:///"):
            raise RuntimeError("task training ownership requires SQLite")
        name = url.removeprefix("sqlite:///")
        self.path = None if name == ":memory:" else Path(name).resolve().with_suffix(".task-training.lock")
        self.key = os.path.normcase(str(self.path)) if self.path else f"memory:{id(database)}"
        self.stream = None
        self.held = False
        self.cleanup_pending = False
        self.claimant = None

    def acquire(self, coordinator):
        with _OWNERS_LOCK:
            if self.key in _OWNERS:
                return False
            # Retain the whole coordinator, including worker/resource leases,
            # until cleanup is known. GC is not evidence of worker termination.
            _OWNERS[self.key] = coordinator
            self.claimant = coordinator
        try:
            if self.path is not None:
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
        except Exception:
            if not self.release():
                self.cleanup_pending = True
            return False

    def release(self):
        if self.stream is not None:
            try:
                # The handle owns the OS lock. Do not unlock first: a failed
                # close must not expose the database to a second coordinator.
                self.stream.close()
            except Exception:
                if getattr(self.stream, "closed", False) is not True:
                    self.cleanup_pending = True
                    return False
            self.stream = None
        self.held = False
        self.cleanup_pending = False
        with _OWNERS_LOCK:
            if _OWNERS.get(self.key) is self.claimant:
                _OWNERS.pop(self.key, None)
        self.claimant = None
        return True


def _training_child_authorized(connection, plan, worker_ownership_path):
    """No engine import before secondary ownership and exact authorization."""
    from argus.runtime.compute_ownership import ComputeOwnership

    try:
        path = worker_ownership_path if worker_ownership_path is not None else ComputeOwnership().worker_path
        ownership, owner = ComputeOwnership(path, guard_training_workers=False), object()
        if ownership.acquire(owner) is not True:
            return False
        # Deliberately never release this claim, even after execute returns:
        # CUDA contexts/caches can outlive model computation. The registry
        # retains the handle until the child OS process actually exits.
        execution_id = plan["execution_id"]
        connection.send({"state": "ready", "execution_id": execution_id})
        if not connection.poll(5):
            return False
        return connection.recv() == {"state": "authorize", "execution_id": execution_id}
    except Exception:
        return False


def _training_child(connection, plan, worker_ownership_path=None):
    """Only this target may call the real engine. Parent EOF kills this child."""
    if not _training_child_authorized(connection, plan, worker_ownership_path):
        try:
            connection.send({"state": "failed", "error_code": "TRAINING_WORKER_AUTHORIZATION_FAILED"})
        except (EOFError, OSError, ValueError):
            pass
        connection.close()
        return
    finished = threading.Event()
    def watch_parent():
        try:
            while connection.recv() != "cancel":
                pass
        except (EOFError, OSError, ValueError):
            pass
        if not finished.is_set():
            os._exit(2)

    threading.Thread(target=watch_parent, name="training-parent-watch", daemon=True).start()
    try:
        from argus.core.task_training import execute_training_plan
        result = execute_training_plan(plan)
        connection.send({"state": "complete", "result": result})
    except Exception:
        # SDK paths/URLs and raw exception text never become API documents.
        connection.send({"state": "failed", "error_code": "TRAINING_WORKER_FAILED"})
    finally:
        finished.set()
        connection.close()


class SpawnTrainingWorker:
    def __init__(self, *, context=None, worker_ownership_path=None):
        from argus.runtime.compute_ownership import ComputeOwnership

        self.context = context or multiprocessing.get_context("spawn")
        self.worker_ownership_path = (Path(worker_ownership_path) if worker_ownership_path is not None
                                      else ComputeOwnership().worker_path)
        self.process = None
        self.connection = None
        self.started = None
        self.timeout = None
        self.message = None
        self._process_started = False
        self._start_attempted = False
        self._child_connection = None
        self._cleaned = False
        self._ready_identity = None
        self._authorized = False

    @property
    def pid(self):
        return self.process.pid if self.process else None

    @staticmethod
    def _identity(plan):
        return (plan.get("job_id"), plan.get("execution_id"), plan.get("request_sha256"))

    def prepare(self, plan):
        """Spawn and prove secondary ownership outside the writer guard."""
        identity = self._identity(plan)
        if self._ready_identity == identity and not self._authorized:
            return
        if self.process is not None:
            raise RuntimeError("training worker cannot be reused")
        parent, child = self.context.Pipe(duplex=True)
        self.connection = parent
        self._child_connection = child
        self.timeout = plan["request"]["submission"]["params"]["timeout_seconds"]
        self.started = time.monotonic()
        try:
            self.process = self.context.Process(target=_training_child,
                args=(child, plan, str(self.worker_ownership_path)), daemon=True)
            self._start_attempted = True
            self.process.start()
            self._process_started = True
        finally:
            child.close()
            self._child_connection = None
        if not parent.poll(5):
            raise RuntimeError("training worker readiness was not confirmed")
        if parent.recv() != {"state": "ready", "execution_id": plan["execution_id"]}:
            raise RuntimeError("training worker readiness identity changed")
        self._ready_identity = identity

    def start(self, plan):
        if self._authorized:
            raise RuntimeError("training worker cannot be reused")
        if self._ready_identity is None:
            self.prepare(plan)
        if self._ready_identity != self._identity(plan):
            raise RuntimeError("training worker execution identity changed")
        self.connection.send({"state": "authorize", "execution_id": plan["execution_id"]})
        self._authorized = True
        self.started = time.monotonic()

    def poll(self):
        if self.message is not None:
            return self.message
        try:
            if self.connection.poll():
                self.message = _worker_message(self.connection.recv())
            elif self.process.is_alive() is False:
                self.message = {"state": "failed", "error_code": "TRAINING_WORKER_EXITED"}
            elif time.monotonic() - self.started > self.timeout:
                self.message = {"state": "failed", "error_code": "TRAINING_WORKER_TIMEOUT"}
        except (EOFError, OSError, ValueError):
            self.message = {"state": "failed", "error_code": "TRAINING_WORKER_CHANNEL_LOST"}
        if self.message is not None:
            self.message = _worker_message(self.message)
        return self.message

    def _close_connections(self):
        complete = True
        for name in ("connection", "_child_connection"):
            connection = getattr(self, name)
            if connection is not None:
                try:
                    connection.close()
                    setattr(self, name, None)
                except Exception:
                    complete = False
        return complete

    def stop(self):
        if self._cleaned:
            return True
        try:
            if self.process is not None and self._start_attempted:
                # start() can create the process and THEN raise. Only the OS
                # process state, never the success flag, proves cleanup.
                if self.process.pid is None and not self._process_started:
                    # Closing our pipe ends encourages a delayed child to see
                    # parent EOF and stop itself. It does NOT prove OS exit:
                    # retain the lease/owner and the unknown cleanup state.
                    self._close_connections()
                    return False
                alive = self.process.is_alive()
                if alive is not True and alive is not False:
                    self._close_connections()
                    return False
                if alive is True:
                    self.process.terminate()
                self.process.join(timeout=5)
                alive = self.process.is_alive()
                if alive is True:
                    self.process.kill()
                    self.process.join(timeout=5)
                    alive = self.process.is_alive()
                if alive is not False:
                    self._close_connections()
                    return False
            if not self._close_connections():
                return False
        except Exception:
            self._close_connections()
            return False
        self._cleaned = True
        return True


@dataclass
class _OwnedTraining:
    worker: object
    lease: object
    execution_id: str
    request_sha256: str
    pending_message: dict | None = None
    cleanup_pending: bool = False
    cleanup_error_code: str = "TRAINING_WORKER_CLEANUP_UNCONFIRMED"


@dataclass
class _PendingLease:
    lease: object
    execution_id: str | None
    error_code: str
    released: bool = False
    preserve_queue: bool = False


class TaskTrainingCoordinator:
    """Call tick explicitly from a configured maintenance/compute scheduler.

    resource_admission(job) must return a retained lease with release(), or
    None. No callback means no GPU permission; jobs wait rather than guessing
    that a separate process has spare GPU capacity.
    """

    def __init__(self, service, *, resource_admission=None, worker_factory=SpawnTrainingWorker):
        self.service = service
        self.resource_admission = resource_admission
        self.worker_factory = worker_factory
        self._owned = {}
        self._pending_leases = {}
        self._lock = threading.Lock()
        self._closed = False
        self._stop_requested = threading.Event()
        self._owner = _TrainingOwner(service._db)
        self._snapshot_lock = threading.Lock()
        self._snapshot = {"owner_held": False, "closed": False, "active_job_id": None,
                          "reconciling_job_ids": [], "waiting_resources": 0,
                          "unknown_job_ids": [], "cleanup_pending": False, "error_code": None}

    def start(self):
        """Acquire the OS/database owner without ticking or inspecting devices."""
        with self._lock:
            if self._closed or self._stop_requested.is_set():
                return False
            acquired = self._owner.held or self._owner.acquire(self)
            with self._snapshot_lock:
                self._snapshot["owner_held"] = self._owner.held
                self._snapshot["cleanup_pending"] = self._owner.cleanup_pending
                self._snapshot["error_code"] = (None if acquired else "TRAINING_RUNTIME_OWNER_CLEANUP_UNCONFIRMED"
                    if self._owner.cleanup_pending else "TRAINING_RUNTIME_OWNER_UNAVAILABLE")
            return acquired

    def snapshot(self):
        """Cached status only: this must never acquire resources or query a PID."""
        with self._snapshot_lock:
            return copy.deepcopy(self._snapshot)

    def request_stop(self):
        """Non-blocking lifecycle signal; a tick must not admit another job."""
        self._stop_requested.set()

    def _publish(self, jobs):
        reconciling = [job["job_id"] for job in jobs if job["status"] == "reconciling"]
        unknown = [job["job_id"] for job in jobs if job["status"] not in _KNOWN_STATES]
        cleanup = (self._owner.cleanup_pending or bool(self._pending_leases)
                   or any(item.cleanup_pending for item in self._owned.values()))
        cleanup_error = ("TRAINING_RUNTIME_OWNER_CLEANUP_UNCONFIRMED" if self._owner.cleanup_pending else
                         next((item.cleanup_error_code for item in self._owned.values() if item.cleanup_pending),
                              "TRAINING_EXECUTION_RECONCILIATION_REQUIRED"))
        active = next(iter(self._owned), None)
        waiting = [job for job in jobs if job["status"] == "waiting_resources"]
        error = ("TRAINING_JOB_STATE_UNKNOWN" if unknown else
                 cleanup_error if cleanup else
                 "TRAINING_OWNER_RECONCILIATION_REQUIRED" if reconciling else
                 (waiting[0]["result"] or {}).get("error_code") if waiting else None)
        with self._snapshot_lock:
            self._snapshot = {"owner_held": self._owner.held, "closed": self._closed,
                "active_job_id": active, "reconciling_job_ids": reconciling,
                "unknown_job_ids": unknown, "cleanup_pending": cleanup,
                "waiting_resources": len(waiting), "error_code": error}

    def _admission_reason(self):
        # An optional read-only admission snapshot explains denial. Arbitrary
        # exception text, paths and device metadata never enter job documents.
        reason = "TRAINING_RESOURCE_APPROVAL_REQUIRED"
        snapshot = getattr(self.resource_admission, "snapshot", None)
        if callable(snapshot):
            try:
                value = snapshot()
                candidate = (value.get("error_code") or value.get("reason_code")) if isinstance(value, dict) else None
                reason = _safe_error_code(candidate, reason)
            except Exception:
                pass
        return reason

    @staticmethod
    def _cancel_requested(job):
        result = job["result"] or {}
        return job["status"] == "cancelling" or result.get("cancellation_requested") is True or bool(result.get("cancelled_by"))

    def _recover_orphan(self, job):
        """No kill/retry by PID. Only an exact negative OS observation is exit."""
        result = job["result"] or {}
        if job["status"] in {"running", "cancelling"}:
            self.service.transition(job["job_id"], expected_revision=job["revision"],
                from_states={job["status"]}, status="reconciling", result={**result,
                    "cancellation_requested": self._cancel_requested(job),
                    "error_code": "TRAINING_OWNER_RECONCILIATION_REQUIRED"})
            return
        pid = result.get("worker_pid")
        if type(pid) is not int or pid < 1:
            return
        try:
            exited = _process_alive(pid) is False
        except Exception:
            exited = False
        if exited:
            self.service.transition(job["job_id"], expected_revision=job["revision"],
                from_states={"reconciling"},
                status="cancelled" if self._cancel_requested(job) else "failed",
                result={**result, "error_code": "TRAINING_OWNER_EXITED", "worker_exit_confirmed": True})

    def _finish_not_started(self, job_id, pending):
        """Reconcile local sequencing knowledge; no worker.start was invoked."""
        current = self.service.get(job_id)
        result = current["result"] or {}
        ours = pending.execution_id is not None and result.get("execution_id") == pending.execution_id
        if not pending.preserve_queue and (current["status"] in {"queued", "waiting_resources"} or ours and current["status"] in {"running", "cancelling", "reconciling"}):
            self.service.transition(job_id, expected_revision=current["revision"],
                from_states={current["status"]}, status="cancelled" if self._cancel_requested(current) else "failed",
                result={**result, "error_code": pending.error_code, "worker_start_confirmed_absent": True})
        if not pending.released:
            pending.lease.release()
            pending.released = True
        self._pending_leases.pop(job_id, None)

    def _finish(self, job, owned, message):
        message = _worker_message(message)
        owned.pending_message = message
        owned.cleanup_pending = True
        owned.cleanup_error_code = "TRAINING_WORKER_CLEANUP_UNCONFIRMED"
        if owned.worker.stop() is not True:
            current = self.service.get(job["job_id"])
            if (current["status"] == "running"
                    and (current["result"] or {}).get("execution_id") == owned.execution_id):
                self.service.transition(current["job_id"], expected_revision=current["revision"],
                    from_states={"running"}, status="reconciling", result={**(current["result"] or {}),
                        "error_code": "TRAINING_WORKER_CLEANUP_UNCONFIRMED"})
            return
        owned.cleanup_error_code = "TRAINING_EXECUTION_RECONCILIATION_REQUIRED"
        current = self.service.get(job["job_id"])
        if (current["status"] in {"running", "cancelling", "reconciling"}
                and (current["result"] or {}).get("execution_id") != owned.execution_id):
            # A delayed result/cleanup from an older execution has no authority
            # over a replacement execution, even when job_id still matches.
            owned.lease.release()
            self._owned.pop(current["job_id"], None)
            return
        if current["status"] in {"complete", "failed", "cancelled"}:
            # The prior commit may have succeeded before an exception, or
            # resource release failed afterwards. Do not replay training or
            # attempt another terminal CAS; retry just the outstanding cleanup.
            owned.lease.release()
            self._owned.pop(current["job_id"], None)
            return
        if self._cancel_requested(current):
            status, result = "cancelled", current["result"] or {}
        else:
            status, result = "failed", {"error_code": message.get("error_code", "TRAINING_RESULT_INVALID")}
            candidate = message.get("result")
            if message.get("state") == "complete" and isinstance(candidate, dict):
                if (set(candidate) == {"schema", "job_id", "execution_id", "request_sha256",
                                      "package_sha256", "provider", "candidate", "production_allowed", "stage"}
                        and candidate.get("schema") == "argus.task-training-result.v1"
                        and candidate.get("job_id") == current["job_id"]
                        and candidate.get("execution_id") == owned.execution_id
                        and candidate.get("request_sha256") == owned.request_sha256
                        and candidate.get("provider") == "torch-cuda"
                        and candidate.get("production_allowed") is False
                        and candidate.get("stage") == "candidate"):
                    try:
                        plan = self.service.prepare(job["job_id"])
                        if candidate["package_sha256"] != plan["request"]["package"]["package_sha256"]:
                            raise ValueError("package mismatch")
                        from argus.capture.task_collector import controlled_path
                        path = controlled_path(self.service.models_root, candidate["candidate"]["relative_path"], must_exist=True)
                        output = self.service.output_root / current["job_id"] / owned.execution_id
                        if not path.resolve().is_relative_to(output.resolve()) or path.suffix != ".onnx":
                            raise ValueError("candidate escaped this training execution")
                        if artifact_sha256(path) != candidate["candidate"]["artifact_sha256"]:
                            raise ValueError("candidate changed")
                        from argus.core.analysis_task_contracts import TaskModelDescriptor
                        if set(candidate["candidate"]) != {"relative_path", "artifact_sha256", "descriptor"}:
                            raise ValueError("invalid candidate result")
                        descriptor = TaskModelDescriptor.model_validate(candidate["candidate"]["descriptor"]).model_dump(mode="json")
                        if (descriptor != candidate["candidate"]["descriptor"]
                                or descriptor["labels"] != plan["request"]["package"]["dataset"]["label_schema"]["labels"]
                                or descriptor["adapter_id"] != "onnx_detection.v1"
                                or descriptor["input_size"] != [plan["request"]["submission"]["params"]["imgsz"]] * 2):
                            raise ValueError("candidate descriptor changed")
                        status, result = "complete", candidate
                    except Exception:
                        result = {"error_code": "TRAINING_RESULT_IDENTITY_OR_APPROVAL_CHANGED"}
        # Cleanup ownership remains until both the database commit and resource
        # release are confirmed. An unknown commit is reconciled, not retrained.
        try:
            self.service.transition(current["job_id"], expected_revision=current["revision"],
                from_states={"running", "cancelling", "reconciling"}, status=status, result=result)
        except (TaskWorkflowBlocked, TaskWorkflowConflict):
            if status != "complete":
                raise
            current = self.service.get(current["job_id"])
            if (current["status"] in {"running", "cancelling", "reconciling"}
                    and (current["result"] or {}).get("execution_id") == owned.execution_id):
                # A fresh cancellation wins; changed approval may never become
                # a successful result merely because preflight already passed.
                cancelled = self._cancel_requested(current)
                self.service.transition(current["job_id"], expected_revision=current["revision"],
                    from_states={current["status"]}, status="cancelled" if cancelled else "failed",
                    result=current["result"] if cancelled else {
                        "error_code": "TRAINING_RESULT_IDENTITY_OR_APPROVAL_CHANGED"})
        owned.lease.release()
        self._owned.pop(current["job_id"], None)

    def tick(self):
        with self._lock:
            if self._closed or self._stop_requested.is_set():
                return
            if not self._owner.held and not self._owner.acquire(self):
                with self._snapshot_lock:
                    self._snapshot["error_code"] = "TRAINING_RUNTIME_OWNER_UNAVAILABLE"
                return
            try:
                for job_id, pending in list(self._pending_leases.items()):
                    self._finish_not_started(job_id, pending)
                # Recovery is a complete first pass, independent of queue order.
                # A newer orphan must block an older queued job just as surely.
                for job in self.service.list():
                    owned = self._owned.get(job["job_id"])
                    if owned is not None:
                        message = ({"state": "cancelled"} if job["status"] in {"cancelling", "cancelled", "complete", "failed"}
                                   else owned.pending_message or owned.worker.poll())
                        if message is not None:
                            self._finish(job, owned, message)
                    elif job["status"] in {"running", "cancelling", "reconciling"}:
                        self._recover_orphan(job)
                jobs = self.service.list()
                if (self._owned or self._pending_leases
                        or any(job["status"] in _EXECUTING_STATES or job["status"] not in _KNOWN_STATES for job in jobs)):
                    return
                # One slot, FIFO. No second resource callback or start in this
                # tick even if a failed admission/claim was locally reconciled.
                job = next((job for job in jobs if job["status"] in {"queued", "waiting_resources"}), None)
                if job is None:
                    return
                if self._stop_requested.is_set():
                    return
                admission_error = None
                try:
                    lease = self.resource_admission(job) if self.resource_admission else None
                except Exception:
                    lease = None
                    admission_error = "TRAINING_RESOURCE_ADMISSION_UNAVAILABLE"
                if lease is None:
                    reason = admission_error or self._admission_reason()
                    if job["status"] != "waiting_resources" or (job["result"] or {}).get("error_code") != reason:
                        self.service.transition(job["job_id"], expected_revision=job["revision"],
                            from_states={job["status"]}, status="waiting_resources", result={"error_code": reason})
                    return
                if self._stop_requested.is_set():
                    # Shutdown raced the external admission call. No claim or
                    # worker start occurred, so keep this durable queue entry.
                    pending = _PendingLease(lease, None, "TRAINING_COORDINATOR_CLOSED", preserve_queue=True)
                    self._pending_leases[job["job_id"]] = pending
                    self._finish_not_started(job["job_id"], pending)
                    return
                worker = None
                execution_id = uuid4().hex
                try:
                    worker = self.worker_factory()
                    claimed, plan = self.service.claim(job["job_id"], expected_revision=job["revision"], execution_id=execution_id)
                    owned = _OwnedTraining(worker, lease, plan["execution_id"], plan["request_sha256"])
                    self._owned[job["job_id"]] = owned
                    if self._stop_requested.is_set():
                        self._finish(self.service.get(job["job_id"]), owned,
                            {"state": "failed", "error_code": "TRAINING_COORDINATOR_CLOSED"})
                        return
                    prepare = getattr(worker, "prepare", None)
                    if callable(prepare):
                        prepare(plan)
                    if self._stop_requested.is_set():
                        raise TaskWorkflowBlocked("TRAINING_COORDINATOR_CLOSED")
                    # The final authorize/start boundary is serialized with
                    # cancellation and trust revocation, not merely a stale
                    # read after claim. Spawn readiness is prepared outside
                    # this short database authority transaction.
                    with self.service.execution_authorization(job["job_id"],
                            expected_revision=claimed["revision"], execution_id=plan["execution_id"],
                            request_sha256=plan["request_sha256"]):
                        # Entering authority can wait behind another writer.
                        # Shutdown stays non-blocking and must be rechecked at
                        # the actual authorize boundary, not just before wait.
                        if self._stop_requested.is_set():
                            raise TaskWorkflowBlocked("TRAINING_COORDINATOR_CLOSED")
                        worker.start(plan)
                    self.service.transition(job["job_id"], expected_revision=claimed["revision"],
                        from_states={"running"}, status="running", result={"execution_id": plan["execution_id"], "worker_pid": worker.pid})
                except Exception:
                    if job["job_id"] not in self._owned:
                        pending = _PendingLease(lease, execution_id,
                            "TRAINING_WORKER_FACTORY_FAILED" if worker is None else "TRAINING_CLAIM_WORKER_NOT_STARTED")
                        self._pending_leases[job["job_id"]] = pending
                        self._finish_not_started(job["job_id"], pending)
                        return
                    self._finish(self.service.get(job["job_id"]), owned,
                                 {"state": "failed", "error_code": "TRAINING_COORDINATOR_CLOSED"
                                  if self._stop_requested.is_set() else "TRAINING_START_FAILED"})
            finally:
                # A failed commit/read never releases local ownership. Reporting
                # uses only this cache, not a second queue or a live device probe.
                try:
                    self._publish(self.service.list())
                except Exception:
                    with self._snapshot_lock:
                        self._snapshot.update(owner_held=self._owner.held,
                            error_code="TRAINING_RUNTIME_STATE_UNAVAILABLE")

    def reconcile_exited(self, job_id, *, expected_revision):
        """No PID lease expiry: only positive OS exit permits failed cleanup."""
        with self._lock:
            if not self._owner.held and not self._owner.acquire(self):
                raise TaskWorkflowConflict("training runtime owner is unavailable")
            job = self.service.get(job_id)
            if type(expected_revision) is not int or job["revision"] != expected_revision:
                raise TaskWorkflowConflict("training execution revision changed")
            pid = (job["result"] or {}).get("worker_pid")
            try:
                exited = type(pid) is int and pid > 0 and _process_alive(pid) is False
            except Exception:
                exited = False
            if job["status"] != "reconciling" or job_id in self._owned or not exited:
                raise TaskWorkflowConflict("training worker exit has not been confirmed")
            result = self.service.transition(job_id, expected_revision=expected_revision,
                from_states={"reconciling"}, status="cancelled" if self._cancel_requested(job) else "failed",
                result={**job["result"], "error_code": "TRAINING_OWNER_EXITED", "worker_exit_confirmed": True})
            self._publish(self.service.list())
            return result

    def close(self):
        self.request_stop()
        with self._lock:
            self._closed = True
            complete = True
            for job_id, pending in list(self._pending_leases.items()):
                try:
                    self._finish_not_started(job_id, pending)
                except Exception:
                    complete = False
            for job_id, owned in list(self._owned.items()):
                try:
                    owned.cleanup_pending = True
                    owned.cleanup_error_code = "TRAINING_WORKER_CLEANUP_UNCONFIRMED"
                    if owned.worker.stop() is not True:
                        complete = False
                        continue
                    owned.cleanup_error_code = "TRAINING_EXECUTION_RECONCILIATION_REQUIRED"
                    current = self.service.get(job_id)
                    if (current["status"] in {"running", "cancelling", "reconciling"}
                            and (current["result"] or {}).get("execution_id") == owned.execution_id):
                        self.service.transition(job_id, expected_revision=current["revision"],
                            from_states={current["status"]},
                            status="cancelled" if self._cancel_requested(current) else "failed",
                            result={**(current["result"] or {}), "error_code": "TRAINING_COORDINATOR_CLOSED"})
                    owned.lease.release()
                    self._owned.pop(job_id, None)
                except Exception:
                    complete = False
            # Persisted orphan identity is unresolved compute too. Releasing
            # this owner would falsely certify that shutdown cleaned it up.
            # An explicit reconciliation (or a running tick) must establish exit.
            try:
                jobs = self.service.list()
                if any(job["status"] in _EXECUTING_STATES or job["status"] not in _KNOWN_STATES for job in jobs):
                    complete = False
            except Exception:
                jobs = None
                complete = False
            if complete and not self._owned and not self._pending_leases:
                try:
                    if self._owner.release() is not True:
                        complete = False
                except Exception:
                    complete = False
            try:
                self._publish(jobs if jobs is not None else self.service.list())
            except Exception:
                with self._snapshot_lock:
                    self._snapshot.update(owner_held=self._owner.held, closed=True,
                        error_code="TRAINING_RUNTIME_STATE_UNAVAILABLE")
            return complete
