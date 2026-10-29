"""Application control for fixed, isolated, shadow-only analysis tasks."""

from __future__ import annotations

import json
import threading
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from argus.core.analysis_task_runtime import TaskFrameBlocked, TaskRuntime
from argus.core.task_execution import (
    IsolatedTaskWorker,
    TaskExecutionError,
    validate_execution_manifest,
)


@dataclass
class _Execution:
    manifest: dict
    source: object
    worker: object
    runtime: TaskRuntime | None = None
    state: str = "starting"
    error_code: str | None = None
    reason: str | None = None
    lease_held: bool = True
    registered: bool = False
    results: deque = field(default_factory=lambda: deque(maxlen=20))
    last_sample_time: float | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    applied_task_revision: int | None = None
    deployment_operation_id: str | None = None
    deployment_revision: int | None = None
    event_error: str | None = None
    compute_reservation: object | None = None


@dataclass
class _TaskControl:
    lock: object = field(default_factory=threading.RLock)
    users: int = 0


class _SceneBlocked(TaskExecutionError, TaskFrameBlocked):
    """A transient closed scene gate, not a poisoned model process."""


class TaskExecutionManager:
    """No registry production writes, alert dispatch, training, or frame files."""

    def __init__(self, task_service, camera_manager, *, models_root: str | Path,
                 max_workers: int = 2, memory_budget_mb: int = 4096,
                 worker_factory=IsolatedTaskWorker, event_sink=None):
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("max_workers must be positive")
        if type(memory_budget_mb) is not int or memory_budget_mb < 128:
            raise ValueError("memory_budget_mb must be at least 128")
        self._service = task_service
        self._manager = camera_manager
        self._models_root = Path(models_root)
        self._max_workers = max_workers
        self._memory_budget_mb = memory_budget_mb
        self._worker_factory = worker_factory
        self._event_sink = event_sink
        self._lock = threading.RLock()
        self._task_controls: dict[str, _TaskControl] = {}
        self._executions: dict[str, _Execution] = {}
        self._prepared: dict[str, _Execution] = {}
        self._pending_starts: dict[str, list[threading.Event]] = {}
        self._deployment_fences: dict[str, tuple[int, str]] = {}
        self._closed = False

    @contextmanager
    def _control(self, task_id: str):
        # Count owners and waiters before releasing the registry lock. Never
        # wait for a task lock under _lock: result callbacks only use _lock.
        with self._lock:
            control = self._task_controls.setdefault(task_id, _TaskControl())
            control.users += 1
        try:
            with control.lock:
                yield
        finally:
            with self._lock:
                control.users -= 1
                if not control.users:
                    del self._task_controls[task_id]

    def _check_authority(self, execution: _Execution) -> None:
        manifest = execution.manifest
        if execution.cancelled.is_set():
            raise TaskExecutionError("CANCELLED", "task execution ownership was cancelled")
        task = self._service.get_task(manifest["task_id"])
        if (not task or task.get("authority") != "managed"
                or task.get("template_id") != manifest["template_id"]
                or task.get("archived") or task.get("current_revision") != manifest["task_revision"]
                or task.get("config_sha256") != manifest["task_config_sha256"]
                or task.get("camera_id") != manifest["camera_id"]):
            raise TaskExecutionError("STALE_TASK", "task was archived or its authoritative revision changed")
        if self._manager.get_camera_source(manifest["camera_id"]) is not execution.source:
            raise TaskExecutionError("SOURCE_CHANGED", "task source was stopped or replaced")
        self._check_scene(execution)

    def _check_scene(self, execution: _Execution) -> None:
        manifest = execution.manifest
        scene = manifest["config"]["scene"]
        if scene["policy"] == "require_profile":
            getter = getattr(self._manager, "get_managed_scene_guard_status", None)
            status = getter(manifest["camera_id"], source=execution.source) if callable(getter) else None
            if (not isinstance(status, dict) or status.get("enabled") is not True
                    or status.get("usable") is not True or status.get("reference_available") is not True
                    or status.get("consumer_enabled") is not True
                    or status.get("profile_id") != scene["profile_id"]):
                raise _SceneBlocked("SCENE_BLOCKED", "required approved scene profile is not currently usable")

    def start(self, task_id: str, bundle_id: str) -> dict:
        # Register the command before waiting for control ownership or doing
        # registry I/O. A stop issued during bundle resolution must fence this
        # start even though no worker/execution has been allocated yet.
        cancelled = threading.Event()
        with self._lock:
            if self._closed:
                raise TaskExecutionError("CLOSED", "execution manager is shutting down")
            self._pending_starts.setdefault(task_id, []).append(cancelled)
        try:
            return self._start_task(task_id, bundle_id, cancelled)
        finally:
            with self._lock:
                pending = self._pending_starts[task_id]
                pending.remove(cancelled)
                if not pending:
                    del self._pending_starts[task_id]

    def _start_task(self, task_id: str, bundle_id: str, cancelled: threading.Event,
                    prepared: _Execution | None = None, *, deployment_operation_id: str | None = None,
                    deployment_revision: int | None = None) -> dict:
        # Only this task's commands serialize; one blocked model load must not
        # prevent a peer from starting, stopping, or confirming cleanup.
        with self._control(task_id):
            with self._lock:
                if self._closed:
                    raise TaskExecutionError("CLOSED", "execution manager is shutting down")
                if cancelled.is_set():
                    raise TaskExecutionError("CANCELLED", "task start was cancelled before worker allocation")
                previous = self._executions.get(task_id)
            if previous is not None:
                if previous.runtime is not None and not previous.runtime.is_cleanly_stopped:
                    if previous.manifest["bundle_id"] == bundle_id and previous.runtime.status.state not in {
                        "failed", "cleanup_pending", "sampling_insufficient",
                    }:
                        self._check_authority(previous)
                        return self.status(task_id)
                    raise TaskExecutionError("BUSY", "stop the existing task before selecting another execution")
                if not self._stop_execution(previous):
                    raise TaskExecutionError("CLEANUP_PENDING", "old task ownership has not been released")
            manifest = validate_execution_manifest(self._service.resolve_bundle(task_id, bundle_id))
            if manifest["task_id"] != task_id or manifest["bundle_id"] != bundle_id:
                raise TaskExecutionError("INVALID_BUNDLE", "resolver returned a different task or bundle")
            source = self._manager.get_camera_source(manifest["camera_id"])
            if source is None or source.source_token is None:
                raise TaskExecutionError("SOURCE_UNAVAILABLE", "start the camera source before its shadow task")
            with self._lock:
                if self._closed:
                    raise TaskExecutionError("CLOSED", "execution manager is shutting down")
                if cancelled.is_set():
                    raise TaskExecutionError("CANCELLED", "task start was cancelled during bundle resolution")
                if prepared is not None:
                    if (self._prepared.get(task_id) is not prepared or prepared.manifest != manifest
                            or prepared.source is not source or prepared.cancelled.is_set()):
                        raise TaskExecutionError("STALE_PREPARATION", "prepared worker identity changed")
                    execution, worker = prepared, prepared.worker
                    del self._prepared[task_id]
                else:
                    reservation = self._reserve(manifest)
                    try:
                        worker = self._worker_factory(manifest, self._models_root)
                    except Exception:
                        if reservation is not None:
                            reservation.release()
                        raise
                    execution = _Execution(manifest=manifest, source=source, worker=worker,
                                           deployment_operation_id=deployment_operation_id,
                                           deployment_revision=deployment_revision, compute_reservation=reservation)
                self._executions[task_id] = execution
            try:
                self._check_authority(execution)
                worker.start()
                self._check_authority(execution)
                config = manifest["config"]

                def processor(frame):
                    # Both DB reads are deliberately outside runtime/source
                    # commit locks; direct service-layer edits also invalidate.
                    try:
                        self._check_authority(execution)
                        target = config["sampling"]["fps_target"]
                        if config["sampling"]["mode"] == "latest":
                            previous_time = execution.last_sample_time
                            if previous_time is not None and 0 <= frame.timestamp - previous_time < 1 / target:
                                return None
                        execution.last_sample_time = frame.timestamp
                        result = worker.infer(frame)
                        self._check_authority(execution)
                        return result
                    except Exception as exc:
                        with self._lock:
                            execution.error_code = getattr(exc, "code", "WORKER_FAILED")
                            execution.reason = str(exc)
                        raise

                def commit(result, frame):
                    if result is None:
                        return
                    def append():
                        with self._lock:
                            if (self._executions.get(task_id) is execution
                                    and execution.state == "running" and not execution.cancelled.is_set()):
                                execution.results.append(result)
                                execution.error_code = None
                                execution.reason = None
                                if self._event_sink is not None and manifest["template_id"] == "object_detection.v1":
                                    try:
                                        accepted = self._event_sink.offer(
                                            manifest, result, frame,
                                            deployment_revision=execution.deployment_revision,
                                            operation_id=execution.deployment_operation_id,
                                        )
                                        execution.event_error = None if accepted else "EVENT_QUEUE_UNAVAILABLE"
                                    except Exception:
                                        # Evidence transport is isolated from model/source
                                        # ownership. The sink must only enqueue here: this
                                        # callback holds the source/task commit fences.
                                        execution.event_error = "EVENT_QUEUE_FAILED"

                    self._commit_scene(execution, append)

                def prepare_result(frame):
                    result = processor(frame)
                    if result is None:
                        return None
                    # Serialization and DB admission are outside source locks.
                    encoded = json.dumps(result, allow_nan=False)
                    if len(encoded.encode()) > 1024 * 1024:
                        raise TaskExecutionError("INVALID_OUTPUT", "shadow result exceeds its memory budget")
                    return json.loads(encoded)

                def commit_transaction(callback):
                    return self._service.commit_if_current(
                        task_id, manifest["task_revision"], manifest["task_config_sha256"], callback,
                    )

                def cleanup():
                    clean = worker.stop()
                    with self._lock:
                        if clean:
                            execution.state = "stopped"
                        else:
                            execution.state = "cleanup_pending"
                            execution.error_code = "CLEANUP_PENDING"
                    return clean

                def reset(reason):
                    execution.last_sample_time = None
                    worker.reset(reason)

                runtime = TaskRuntime(
                    task_id, source, prepare_result, on_result=commit, cleanup=cleanup,
                    mode=config["sampling"]["mode"], capacity=config["sampling"]["capacity"],
                    deployment_revision=(execution.deployment_revision
                                         if execution.deployment_revision is not None
                                         else 0), stateful=True, reset=reset,
                    commit_transaction=commit_transaction,
                )
                execution.runtime = runtime
                self._manager.register_analysis_task_runtime(runtime)
                execution.registered = True
                with self._lock:
                    execution.state = "running"
                if not runtime.start():
                    raise TaskExecutionError("WORKER_FAILED", "analysis task did not start")
                with self._lock:
                    execution.applied_task_revision = manifest["task_revision"]
                return self.status(task_id)
            except Exception as exc:
                execution.error_code = getattr(exc, "code", "WORKER_FAILED")
                execution.reason = str(exc)
                clean = self._stop_execution(execution)
                if clean:
                    execution.state = "failed"
                if isinstance(exc, TaskExecutionError):
                    raise
                raise TaskExecutionError("WORKER_FAILED", str(exc)) from exc

    def _reserve(self, manifest):
        """Caller holds _lock; prepared and active leases both count."""
        owned = [*self._executions.values(), *self._prepared.values()]
        if sum(item.lease_held for item in owned) >= self._max_workers:
            raise TaskExecutionError("RESOURCE_EXHAUSTED", "concurrent shadow worker limit reached")
        reserved = sum(item.manifest["config"]["resources"]["memory_limit_mb"]
                       for item in owned if item.lease_held)
        if reserved + manifest["config"]["resources"]["memory_limit_mb"] > self._memory_budget_mb:
            raise TaskExecutionError("RESOURCE_EXHAUSTED", "declared shadow memory reservation exhausted")
        reserve = getattr(self._manager, "acquire_task_compute_reservation", None)
        if callable(reserve):
            try:
                return reserve()
            except Exception as exc:
                raise TaskExecutionError("RESOURCE_EXHAUSTED", "training maintenance excludes model loading") from exc
        return None

    def deploy(self, task_id: str, bundle_id: str, *, operation_id: str, deployment_revision: int) -> dict:
        """Prepare beside the old worker; replace only after readiness checks.

        The extra worker/bytes are real reservations. Resource exhaustion or
        candidate load failure cannot be made to fit by stopping the old task.
        A post-quiescence failure attempts an authorized old-bundle restart;
        uncertain cleanup keeps its lease and requires explicit reconciliation.
        """
        if (not isinstance(operation_id, str) or not operation_id
                or type(deployment_revision) is not int or deployment_revision < 1):
            raise TaskExecutionError("INVALID_DEPLOYMENT", "invalid durable deployment identity")
        cancelled = threading.Event()
        with self._lock:
            if self._closed:
                raise TaskExecutionError("CLOSED", "execution manager is shutting down")
            current_fence = self._deployment_fences.get(task_id, (0, ""))
            if (deployment_revision < current_fence[0]
                    or deployment_revision == current_fence[0] and operation_id != current_fence[1]):
                raise TaskExecutionError("SUPERSEDED", "a later deployment command already fenced this task")
            self._deployment_fences[task_id] = (deployment_revision, operation_id)
            self._pending_starts.setdefault(task_id, []).append(cancelled)
        try:
            with self._control(task_id):
                with self._lock:
                    previous = self._executions.get(task_id)
                    orphan = self._prepared.get(task_id)
                    if self._closed or cancelled.is_set():
                        raise TaskExecutionError("CANCELLED", "deployment was cancelled")
                if orphan is not None:
                    raise TaskExecutionError("CLEANUP_PENDING", "prepared worker ownership is retained")
                if (previous is not None and previous.manifest["bundle_id"] == bundle_id
                        and previous.lease_held and previous.state == "running"
                        and previous.runtime is not None and previous.runtime.status.state in {"running", "waiting_source"}):
                    self._check_authority(previous)
                    if (previous.deployment_operation_id != operation_id
                            or previous.deployment_revision != deployment_revision):
                        # Reusing immutable weights is not permission to carry
                        # in-flight/retained results into a new deployment.
                        # Never take runtime._lock while holding _lock: its
                        # result callback acquires those in the reverse order.
                        previous.runtime.set_deployment_revision(deployment_revision)
                        with self._lock:
                            previous.results.clear()
                    with self._lock:
                        previous.deployment_operation_id = operation_id
                        previous.deployment_revision = deployment_revision
                    return self.status(task_id)
                manifest = validate_execution_manifest(self._service.resolve_bundle(task_id, bundle_id))
                if manifest["task_id"] != task_id or manifest["bundle_id"] != bundle_id:
                    raise TaskExecutionError("INVALID_BUNDLE", "resolver returned a different deployment bundle")
                source = self._manager.get_camera_source(manifest["camera_id"])
                if source is None or source.source_token is None:
                    raise TaskExecutionError("SOURCE_UNAVAILABLE", "source is not running")
                with self._lock:
                    if self._closed or cancelled.is_set():
                        raise TaskExecutionError("CANCELLED", "deployment was cancelled")
                    reservation = self._reserve(manifest)
                    try:
                        worker = self._worker_factory(manifest, self._models_root)
                    except Exception:
                        if reservation is not None:
                            reservation.release()
                        raise
                    candidate = _Execution(manifest, source, worker,
                                           cancelled=cancelled, deployment_operation_id=operation_id,
                                           deployment_revision=deployment_revision, compute_reservation=reservation)
                    self._prepared[task_id] = candidate
                switched = False
                try:
                    self._check_authority(candidate)
                    candidate.worker.start()
                    self._check_authority(candidate)
                    if previous is not None:
                        previous.cancelled.set()
                        cancel_previous = getattr(previous.worker, "cancel", None)
                        try:
                            (cancel_previous or previous.worker.stop)()
                        except Exception:
                            pass  # Confirm cleanup below; never infer it from interrupt success.
                        if not self._stop_execution(previous):
                            raise TaskExecutionError("CLEANUP_PENDING", "old execution cleanup was not confirmed")
                    switched = True
                    return self._start_task(task_id, bundle_id, cancelled, prepared=candidate)
                except Exception as exc:
                    clean = self._stop_execution(candidate)
                    with self._lock:
                        if clean and self._prepared.get(task_id) is candidate:
                            del self._prepared[task_id]
                    # A stop/close superseding this operation must never be
                    # compensated by resurrecting its old consumer.
                    if switched and clean and previous is not None and not cancelled.is_set() and not self._closed:
                        try:
                            restored = self._start_task(task_id, previous.manifest["bundle_id"], threading.Event(),
                                deployment_operation_id=previous.deployment_operation_id,
                                deployment_revision=previous.deployment_revision)
                            if restored is None:
                                raise RuntimeError("missing restoration receipt")
                        except Exception as restore_error:
                            raise TaskExecutionError("RESTORE_FAILED", "old bundle recovery requires reconciliation") from restore_error
                    if not clean:
                        raise TaskExecutionError("CLEANUP_PENDING", "candidate ownership is retained") from exc
                    if isinstance(exc, TaskExecutionError):
                        raise
                    raise TaskExecutionError("WORKER_FAILED", "candidate deployment failed") from exc
        finally:
            with self._lock:
                pending = self._pending_starts[task_id]
                pending.remove(cancelled)
                if not pending:
                    del self._pending_starts[task_id]

    def _stop_execution(self, execution: _Execution) -> bool:
        try:
            return self._stop_owned_execution(execution)
        except Exception as exc:
            execution.state = "cleanup_pending"
            execution.error_code = "CLEANUP_PENDING"
            execution.reason = str(exc)
            return False

    def _stop_owned_execution(self, execution: _Execution) -> bool:
        runtime = execution.runtime
        timeout = execution.manifest["config"]["resources"]["stop_timeout_seconds"]
        if runtime is not None and not runtime.stop(timeout):
            execution.state = "cleanup_pending"
            return False
        # A runtime constructed but never started (e.g. registration rejected)
        # has no acquired task cleanup callback. Always confirm worker cleanup
        # independently before releasing the resource lease.
        if not execution.worker.stop():
            execution.state = "cleanup_pending"
            return False
        if execution.registered:
            if not self._manager.unregister_analysis_task_runtime(
                execution.manifest["task_id"], runtime=runtime,
            ):
                execution.state = "cleanup_pending"
                return False
            execution.registered = False
        if execution.compute_reservation is not None:
            execution.compute_reservation.release()
            execution.compute_reservation = None
        with self._lock:
            execution.lease_held = False
            execution.state = "stopped"
        return True

    def stop(self, task_id: str, *, operation_id: str | None = None,
             deployment_revision: int | None = None) -> dict:
        # Interrupt a blocked load/infer before waiting for serialized control
        # operations. Resource release remains confirmed by _stop_execution.
        stale, execution, prepared = False, None, None
        if (operation_id is None) != (deployment_revision is None):
            raise TaskExecutionError("INVALID_DEPLOYMENT", "stop requires a complete durable identity")
        with self._lock:
            if operation_id is not None:
                if not operation_id or type(deployment_revision) is not int or deployment_revision < 1:
                    raise TaskExecutionError("INVALID_DEPLOYMENT", "invalid stop deployment identity")
                watermark = self._deployment_fences.get(task_id, (0, ""))
                stale = (deployment_revision < watermark[0]
                         or deployment_revision == watermark[0] and operation_id != watermark[1])
                if not stale:
                    # Keep this watermark even with no worker/pending command:
                    # a claimed old deploy may not have reached this manager.
                    self._deployment_fences[task_id] = (deployment_revision, operation_id)
            if not stale:
                # Fence admission and capture/cancel the exact ownership in
                # one critical section. An older stop cannot cancel a command
                # accepted after its watermark check.
                for pending in self._pending_starts.get(task_id, ()):
                    pending.set()
                execution = self._executions.get(task_id)
                prepared = self._prepared.get(task_id)
                if execution is not None:
                    execution.cancelled.set()
                if prepared is not None:
                    prepared.cancelled.set()
        if stale:
            return self.status(task_id) or {"task_id": task_id, "state": "stopped", "production_allowed": False}
        for owned in (execution, prepared):
            if owned is None:
                continue
            cancel = getattr(owned.worker, "cancel", None)
            try:
                (cancel or owned.worker.stop)()
            except Exception:
                pass  # The confirming cleanup below retains ownership.
        with self._control(task_id):
            with self._lock:
                current = self._executions.get(task_id)
                remaining = self._prepared.get(task_id)
                superseded = (operation_id is not None and self._deployment_fences.get(task_id)
                              != (deployment_revision, operation_id))
            if superseded:
                return self.status(task_id) or {"task_id": task_id, "state": "stopped", "production_allowed": False}
            if remaining is not None and remaining is prepared:
                if not self._stop_execution(remaining):
                    raise TaskExecutionError("CLEANUP_PENDING", "prepared worker cleanup is incomplete")
                with self._lock:
                    if self._prepared.get(task_id) is remaining:
                        del self._prepared[task_id]
            # A later start may win control ownership before this stop. Never
            # cancel its fresh worker using an older command's ownership.
            if current is not execution:
                return self.status(task_id) or {
                    "task_id": task_id, "state": "stopped", "production_allowed": False,
                }
            if execution is None:
                return {"task_id": task_id, "state": "stopped", "production_allowed": False}
            if not self._stop_execution(execution):
                raise TaskExecutionError("CLEANUP_PENDING", "task cleanup is incomplete; ownership retained")
            return self.status(task_id)

    def mutate_if_stopped(self, task_id: str, callback):
        """Serialize configuration writes with start/stop and cleanup ownership.

        Status is observational and cannot authorize a mutation: a worker may
        start between a status read and a database write. This guard holds the
        same control lock as start until the caller's short mutation completes.
        Failed cleanup retains its lease and remains unmodifiable too.
        """
        with self._control(task_id):
            with self._lock:
                execution = self._executions.get(task_id)
                if execution is not None and execution.lease_held:
                    raise TaskExecutionError("BUSY", "stop the task and confirm cleanup before mutation")
            return callback()

    def set_event_sink(self, sink) -> None:
        """Application wiring only; setting a sink never authorizes production."""
        with self._lock:
            self._event_sink = sink

    def event_context(self, task_id: str) -> dict | None:
        """Observe current execution identity without I/O under manager locks."""
        with self._lock:
            execution = self._executions.get(task_id)
            if execution is None or execution.cancelled.is_set() or execution.state != "running":
                return None
        # A retained old source can still expose a plausible token after the
        # camera manager replaces it. Token equality is not source ownership.
        if self._manager.get_camera_source(execution.manifest["camera_id"]) is not execution.source:
            return None
        token = execution.source.source_token
        try:
            self._check_scene(execution)
        except TaskExecutionError:
            return None
        with self._lock:
            if (self._executions.get(task_id) is not execution or execution.cancelled.is_set()
                    or execution.state != "running" or token is None):
                return None
            return {"task_id": task_id, "bundle_id": execution.manifest["bundle_id"],
                    "manifest_sha256": execution.manifest["manifest_sha256"],
                    "execution_id": getattr(execution.worker, "execution_id", None),
                    "deployment_revision": execution.deployment_revision,
                    "deployment_operation_id": execution.deployment_operation_id,
                    "source_session_id": token[0], "source_epoch": token[1]}

    def status(self, task_id: str) -> dict | None:
        with self._lock:
            execution = self._executions.get(task_id)
            if execution is None:
                return None
            state, reason = execution.state, execution.reason
            reason_code = execution.error_code
            last_result = execution.results[-1] if execution.results else None
        token = execution.source.source_token
        last_state = last_result.get("data", {}).get("state") if (
            last_result and (last_result["source"]["source_session_id"], last_result["source"]["source_epoch"])
            == token
        ) else None
        runtime = execution.runtime.snapshot() if execution.runtime is not None else None
        if runtime is not None and state == "running":
            state = runtime["state"]
            reason = runtime["reason"]
        if state not in {"stopped", "failed", "cleanup_pending"} and last_state == "sampling_insufficient":
            state = "sampling_insufficient"
        if state not in {"stopped", "failed", "cleanup_pending"}:
            try:
                self._check_authority(execution)
            except TaskExecutionError as exc:
                state, reason = "blocked", str(exc)
                reason_code = exc.code
        return {"task_id": task_id, "bundle_id": execution.manifest["bundle_id"],
                "manifest_sha256": execution.manifest["manifest_sha256"],
                "task_revision": execution.manifest["task_revision"], "state": state,
                "applied_task_revision": execution.applied_task_revision,
                "reason": reason, "reason_code": reason_code, "error_code": execution.error_code,
                "runtime": runtime, "worker": execution.worker.snapshot(),
                "deployment_operation_id": execution.deployment_operation_id,
                "deployment_revision": execution.deployment_revision,
                "event_error": execution.event_error,
                "production_allowed": False, "alerts_emitted": False}

    def results(self, task_id: str, limit: int = 20) -> list[dict]:
        if not 1 <= limit <= 20:
            raise ValueError("result limit must be between 1 and 20")
        with self._lock:
            execution = self._executions.get(task_id)
        if execution is None:
            return []
        # An authoritative revision change immediately removes old results
        # from the current-task surface, including a narrow check/commit race.
        if execution.cancelled.is_set():
            return []
        self._check_authority(execution)
        current_token = execution.source.source_token
        if current_token is None:
            return []
        results = []

        def read_current():
            def read():
                with self._lock:
                    if (self._executions.get(task_id) is not execution
                            or execution.cancelled.is_set()):
                        return
                    results.extend([item for item in execution.results
                                    if (item["source"]["source_session_id"], item["source"]["source_epoch"])
                                    == current_token][-limit:])

            self._commit_scene(execution, read)

        self._service.commit_if_current(
            task_id, execution.manifest["task_revision"], execution.manifest["task_config_sha256"],
            lambda: execution.source.commit_if_current(current_token, read_current),
        )
        return deepcopy(results)

    def _commit_scene(self, execution: _Execution, callback) -> None:
        scene = execution.manifest["config"]["scene"]
        if scene["policy"] == "require_profile":
            commit = getattr(self._manager, "commit_managed_scene_if_current", None)
            if not callable(commit) or not commit(
                execution.manifest["camera_id"], source=execution.source,
                profile_id=scene["profile_id"], callback=callback,
            ):
                raise _SceneBlocked("SCENE_BLOCKED", "scene evidence changed before the shadow result commit")
        else:
            callback()

    def latest_results(self, task_id: str, limit: int = 20) -> list[dict]:
        return self.results(task_id, limit)

    def close(self) -> bool:
        with self._lock:
            self._closed = True
            for starts in self._pending_starts.values():
                for pending_start in starts:
                    pending_start.set()
            pending = [*self._executions.values(), *self._prepared.values()]
            task_ids = set(self._executions) | set(self._prepared) | set(self._pending_starts)
            for execution in pending:
                execution.cancelled.set()
        for execution in pending:
            cancel = getattr(execution.worker, "cancel", None)
            try:
                (cancel or execution.worker.stop)()
            except Exception:
                pass
        results = []
        for task_id in sorted(task_ids):
            # Cancel every worker before waiting for any one task. Start's
            # allocation check shares _lock with _closed, so no later worker
            # can appear outside this snapshot. Never hold two task locks.
            with self._control(task_id):
                with self._lock:
                    execution = self._executions.get(task_id)
                    prepared = self._prepared.get(task_id)
                if prepared is not None:
                    clean = self._stop_execution(prepared)
                    results.append(clean)
                    if clean:
                        with self._lock:
                            self._prepared.pop(task_id, None)
                if execution is not None:
                    results.append(self._stop_execution(execution))
        return all(results)

    def cleanup_confirmed(self, task_id: str | None = None) -> bool:
        with self._lock:
            return not any(item.lease_held for key, item in [*self._executions.items(), *self._prepared.items()]
                           if task_id is None or key == task_id)

    def source_available(self, task_id: str) -> bool:
        task = self._service.get_task(task_id)
        if task is None:
            return False
        source = self._manager.get_camera_source(task["camera_id"])
        return source is not None and source.source_token is not None

    def stop_source(self, camera_id: str) -> bool:
        """Cancel every accepted command for one source before awaiting cleanup.

        The camera manager marks the source stopping before this callback, so
        no later start can acquire that source. Durable desired state is not
        changed here; the reconciler waits for a later explicit source start.
        """
        with self._lock:
            task_ids = {key for key, item in [*self._executions.items(), *self._prepared.items()]
                        if item.manifest["camera_id"] == camera_id}
            pending_ids = list(self._pending_starts)
        confirmed = True
        for task_id in pending_ids:
            try:
                task = self._service.get_task(task_id)
                if task is not None and task["camera_id"] == camera_id:
                    task_ids.add(task_id)
            except Exception:
                confirmed = False
        with self._lock:
            owned = []
            for task_id in task_ids:
                for cancellation in self._pending_starts.get(task_id, ()):
                    cancellation.set()
                for mapping in (self._executions, self._prepared):
                    execution = mapping.get(task_id)
                    if execution is not None:
                        execution.cancelled.set()
                        owned.append(execution)
        for execution in owned:
            try:
                cancel = getattr(execution.worker, "cancel", None)
                (cancel or execution.worker.stop)()
            except Exception:
                pass
        for task_id in sorted(task_ids):
            try:
                self.stop(task_id)
            except Exception:
                confirmed = False
        return confirmed
