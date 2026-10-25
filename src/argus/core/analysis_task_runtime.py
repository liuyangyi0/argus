"""Independent task lifecycles over a shared camera source.

This foundation only accepts processors registered by application code.  It is
not a plugin loader, model publisher, or safety-gate bypass.  Processors return
results; externally visible effects belong in ``on_result`` after generation
validation.  Legacy DetectionPipeline keeps its existing guarded runner until
its effects have been separated and can obey this contract.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from typing import Any

from argus.capture.camera import FrameData
from argus.capture.frame_hub import FrameSubscription, SourceToken, SubscriptionMode
from argus.capture.source_runtime import CameraSourceRuntime


class TaskFrameBlocked(RuntimeError):
    """Discard a frame while waiting for fresh safety evidence, retaining resources."""


@dataclass(frozen=True)
class TaskRuntimeStatus:
    task_id: str
    camera_id: str
    state: str
    generation: int
    deployment_revision: int
    cleanup_pending: bool
    reason: str | None
    processed_frames: int
    discarded_results: int
    dropped_frames: int
    stateful: bool


class TaskRuntime:
    """One task's worker, queue, generation fence, and retryable cleanup."""

    def __init__(
        self,
        task_id: str,
        source: CameraSourceRuntime,
        processor: Callable[[FrameData], Any],
        *,
        on_result: Callable[[Any, FrameData], None] | None = None,
        cleanup: Callable[[], bool | None] | None = None,
        mode: SubscriptionMode = "latest",
        capacity: int = 32,
        deployment_revision: int = 0,
        stateful: bool = False,
        reset: Callable[[str], None] | None = None,
        commit_transaction: Callable[[Callable[[], None]], bool | None] | None = None,
    ) -> None:
        if not task_id:
            raise ValueError("task_id is required")
        if deployment_revision < 0:
            raise ValueError("deployment revision must not be negative")
        if stateful and not callable(reset):
            raise ValueError("stateful processors require an explicit reset callback")
        self.task_id = task_id
        self.source = source
        self._processor = processor
        self._on_result = on_result
        self._cleanup = cleanup
        self._stateful = stateful
        self._reset = reset
        self._commit_transaction = commit_transaction
        self._mode = mode
        self._capacity = capacity
        self._deployment_revision = deployment_revision
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._subscription: FrameSubscription | None = None
        self._generation = 0
        self._state = "stopped"
        self._reason: str | None = None
        self._cleanup_pending = False
        self._processed = 0
        self._discarded = 0
        self._last_dropped = 0
        self._processor_context: tuple[SourceToken, int] | None = None
        self._sampling_blocked = False
        self._registration_closed = False

    @property
    def status(self) -> TaskRuntimeStatus:
        with self._lock:
            return TaskRuntimeStatus(
                self.task_id, self.source.camera_id, self._state,
                self._generation, self._deployment_revision, self._cleanup_pending,
                self._reason, self._processed, self._discarded,
                self._subscription.status.dropped_frames
                if self._subscription else self._last_dropped,
                self._stateful,
            )

    def snapshot(self) -> dict:
        return asdict(self.status)

    def start(self) -> bool:
        with self._lifecycle_lock, self._lock:
            if self._cleanup_pending or self._registration_closed:
                return False
            if self._thread is not None and self._thread.is_alive():
                return True
            if self._thread is not None:
                # A failed worker still owns processor resources until cleanup.
                self._cleanup_pending = True
                self._state = "cleanup_pending"
                self._reason = "previous task worker requires cleanup"
                return False
            self._generation += 1
            generation = self._generation
            self._stop_event.clear()
            self._reason = "waiting for a current source frame"
            self._state = "waiting_source"
            self._processor_context = None
            self._sampling_blocked = False
            try:
                self._subscription = self.source.subscribe(
                    f"task:{self.task_id}", mode=self._mode, capacity=self._capacity,
                )
                self._thread = threading.Thread(
                    target=self._run, args=(generation,),
                    name=f"analysis-task-{self.task_id}", daemon=True,
                )
                self._thread.start()
                return True
            except Exception as exc:
                self._state = "failed"
                self._reason = f"task start failed: {exc}"
                self._cleanup_pending = True
                return False

    def set_deployment_revision(self, revision: int) -> None:
        """Fence old results only; this does not load or authorize any model."""
        with self._lock:
            if revision <= self._deployment_revision:
                raise ValueError("deployment revision must increase")
            self._deployment_revision = revision

    @staticmethod
    def _frame_token(frame: FrameData) -> SourceToken | None:
        identity = frame.identity
        if identity is None or identity.source_session_id is None:
            return None
        return identity.source_session_id, identity.source_epoch

    def _run(self, generation: int) -> None:
        subscription = self._subscription
        if subscription is None:
            return
        while not self._stop_event.is_set():
            # A broken ordered window requires explicit stop/start.  Waiting
            # here also avoids draining a closed/invalid queue in a busy loop.
            with self._lock:
                sampling_blocked = self._sampling_blocked
            if sampling_blocked:
                self._stop_event.wait(0.1)
                continue
            frame = subscription.get(0.1)
            with self._lock:
                if generation != self._generation or self._stop_event.is_set():
                    return
                if self._block_incomplete_window(subscription):
                    if frame is not None:
                        self._discarded += 1
                    continue
                if frame is None:
                    self._state = "waiting_source"
                    self._reason = "waiting for a current source frame"
                    continue
                token = self._frame_token(frame)
                if token is None or token != self.source.source_token:
                    self._discarded += 1
                    continue
                deployment_revision = self._deployment_revision
                context = (token, deployment_revision)
                previous_context = self._processor_context
                self._state = "running"
                self._reason = None
            try:
                if context != previous_context:
                    if self._reset is not None:
                        if previous_context is None:
                            reason = "task_started"
                        elif token != previous_context[0]:
                            reason = "source_changed"
                        else:
                            reason = "deployment_changed"
                        if self._reset(reason) is False:
                            raise RuntimeError("processor state reset was not confirmed")
                    with self._lock:
                        if not self._is_current(generation, token, deployment_revision, frame):
                            self._discarded += 1
                            continue
                        self._processor_context = context
                        if self._block_incomplete_window(subscription):
                            self._discarded += 1
                            continue
                result = self._processor(frame)
                # A database authority fence belongs outside the source lock.
                # The transaction may wait; the source must keep publishing.
                def commit_current() -> None:
                    with self._lock:
                        if not self._is_current(generation, token, deployment_revision, frame):
                            self._discarded += 1
                            return
                        if self._block_incomplete_window(subscription):
                            self._discarded += 1
                            return

                        def commit() -> None:
                            if self._on_result is not None:
                                self._on_result(result, frame)

                        committed = False

                        def guarded_commit() -> None:
                            nonlocal committed
                            if self._mode == "ordered_window":
                                committed = subscription.commit_if_no_drops(commit)
                            else:
                                commit()
                                committed = True

                        source_current = self.source.commit_if_current(token, guarded_commit)
                        if not source_current or not committed:
                            self._discarded += 1
                            self._block_incomplete_window(subscription)
                            return
                        self._processed += 1

                if self._commit_transaction is not None:
                    if self._commit_transaction(commit_current) is False:
                        with self._lock:
                            self._discarded += 1
                else:
                    commit_current()
            except Exception as exc:
                with self._lock:
                    if generation != self._generation or self._stop_event.is_set():
                        self._discarded += 1
                        return
                    if not self._is_current(generation, token, deployment_revision, frame):
                        self._discarded += 1
                        continue
                    if isinstance(exc, TaskFrameBlocked):
                        self._discarded += 1
                        self._state = "blocked"
                        self._reason = str(exc)
                        self._processor_context = None
                        continue
                    self._state = "failed"
                    self._reason = f"task processor failed: {exc}"
                    self._cleanup_pending = True
                return

    def _is_current(
        self, generation: int, token: SourceToken,
        deployment_revision: int, frame: FrameData,
    ) -> bool:
        return (
            generation == self._generation
            and not self._stop_event.is_set()
            and token == self.source.source_token
            and token == self._frame_token(frame)
            and deployment_revision == self._deployment_revision
        )

    def _block_incomplete_window(self, subscription: FrameSubscription) -> bool:
        if self._mode == "ordered_window" and subscription.status.dropped_frames:
            self._sampling_blocked = True
            self._state = "sampling_insufficient"
            self._reason = "ordered frame window overflowed; stop/start required"
        return self._sampling_blocked

    @property
    def registration_closed(self) -> bool:
        with self._lock:
            return self._registration_closed

    @property
    def is_cleanly_stopped(self) -> bool:
        with self._lock:
            return self._state == "stopped" and not self._cleanup_pending and self._thread is None

    def detach_if_clean(self) -> bool:
        """Permanently fence an old runtime before removing its registration.

        The lifecycle lock makes this atomic with direct start() calls too;
        stale references cannot restart an orphan worker after unregistering.
        """
        with self._lifecycle_lock, self._lock:
            if self._state != "stopped" or self._cleanup_pending or self._thread is not None:
                return False
            self._registration_closed = True
            return True

    def stop(self, timeout: float = 5.0) -> bool:
        with self._lifecycle_lock:
            with self._lock:
                if self._thread is None and not self._cleanup_pending:
                    self._state = "stopped"
                    return True
                self._generation += 1
                self._stop_event.set()
                self._cleanup_pending = True
                self._state = "cleanup_pending"
                self._reason = "waiting for task worker cleanup"
                thread = self._thread
                subscription = self._subscription
                if subscription is not None:
                    self._last_dropped = subscription.status.dropped_frames
                    subscription.close()
            if thread is not None and thread.is_alive():
                if thread is not threading.current_thread():
                    thread.join(max(0.0, timeout))
                if thread.is_alive():
                    with self._lock:
                        self._reason = "task worker still running; restart blocked"
                    return False
            try:
                if self._cleanup is not None and self._cleanup() is False:
                    raise RuntimeError("processor cleanup was not confirmed")
            except Exception as exc:
                with self._lock:
                    self._reason = f"task cleanup failed: {exc}"
                return False
            with self._lock:
                if self._subscription is not None:
                    self.source.hub.unsubscribe(
                        f"task:{self.task_id}", subscription=self._subscription,
                    )
                self._subscription = None
                self._thread = None
                self._cleanup_pending = False
                self._state = "stopped"
                self._reason = None
                return True


class TaskSupervisor:
    """Application-registered task collection; no dynamic code or model loading."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tasks: dict[str, TaskRuntime] = {}

    def register(self, runtime: TaskRuntime) -> TaskRuntime:
        # Task callbacks may hold runtime._lock and read this collection.  Take
        # lifecycle/state ownership first; never wait for a task under _lock.
        # Holding lifecycle through insertion also prevents a concurrent
        # detach from slipping between validation and registration.
        # Registration can also be called under CameraManager's source-CAS
        # lock.  Do not wait for either task lock: callbacks may read manager.
        with ExitStack() as locks:
            if not runtime._lifecycle_lock.acquire(blocking=False):
                raise RuntimeError("analysis task lifecycle busy; retry registration")
            locks.callback(runtime._lifecycle_lock.release)
            if not runtime._lock.acquire(blocking=False):
                raise RuntimeError("analysis task state busy; retry registration")
            locks.callback(runtime._lock.release)
            if runtime.registration_closed:
                raise ValueError("cannot register a detached task runtime; create a new runtime")
            with self._lock:
                if runtime.task_id in self._tasks:
                    raise ValueError(f"analysis task already registered: {runtime.task_id}")
                self._tasks[runtime.task_id] = runtime
                return runtime

    def get(self, task_id: str) -> TaskRuntime:
        with self._lock:
            return self._tasks[task_id]

    def list(self) -> list[TaskRuntime]:
        with self._lock:
            return list(self._tasks.values())

    def status(self) -> list[dict]:
        return [task.snapshot() for task in self.list()]

    def start(self, task_id: str) -> bool:
        # detach_if_clean permanently fences stale references.  If unregister
        # wins after get(), this start returns False; if start wins, unregister
        # cannot detach a live runtime.  No collection lock is needed here.
        return self.get(task_id).start()

    def stop(self, task_id: str, timeout: float = 5.0) -> bool:
        return self.get(task_id).stop(timeout)

    def stop_source(self, camera_id: str, timeout: float = 5.0) -> bool:
        results = [
            task.stop(timeout) for task in self.list()
            if task.source.camera_id == camera_id
        ]
        return all(results)

    def stop_all(self, timeout: float = 5.0) -> bool:
        return all([task.stop(timeout) for task in self.list()])

    def unregister_clean_source(
        self, camera_id: str, *, source: CameraSourceRuntime | None = None,
    ) -> bool:
        """Drop only fully cleaned tasks, optionally scoped to an old source.

        Cleanup failures remain registered and retain their task identities.
        Successful removals fence old TaskRuntime references permanently, so a
        camera restart can safely bind the same task IDs to fresh runtimes.
        """
        with self._lock, ExitStack() as locks:
            candidates = [
                (task_id, runtime) for task_id, runtime in self._tasks.items()
                if runtime.source.camera_id == camera_id
                and (source is None or runtime.source is source)
            ]
            # Freeze direct TaskRuntime.start/stop calls across the whole
            # candidate set: an unsuccessful cleanup leaves every identity.
            for _, runtime in candidates:
                # Never wait while holding the supervisor lock: an in-flight
                # task callback may itself read the supervisor's status.
                if not runtime._lifecycle_lock.acquire(blocking=False):
                    return False
                locks.callback(runtime._lifecycle_lock.release)
                if not runtime._lock.acquire(blocking=False):
                    return False
                locks.callback(runtime._lock.release)
            if any(not runtime.is_cleanly_stopped for _, runtime in candidates):
                return False
            for task_id, runtime in candidates:
                runtime.detach_if_clean()
                del self._tasks[task_id]
            return True

    def unregister_clean_task(self, task_id: str, *, runtime: TaskRuntime | None = None) -> bool:
        """Detach one fully cleaned expected runtime without disturbing peers."""
        with self._lock, ExitStack() as locks:
            current = self._tasks.get(task_id)
            if current is None:
                return True
            if runtime is not None and current is not runtime:
                return False
            if not current._lifecycle_lock.acquire(blocking=False):
                return False
            locks.callback(current._lifecycle_lock.release)
            if not current._lock.acquire(blocking=False):
                return False
            locks.callback(current._lock.release)
            if not current.is_cleanly_stopped or not current.detach_if_clean():
                return False
            del self._tasks[task_id]
            return True
