"""Lifespan-owned task training scheduler; construction and reads are inert."""

from __future__ import annotations

import math
import threading
import time

from argus.runtime.task_training import SpawnTrainingWorker, TaskTrainingCoordinator


class TaskTrainingRuntime:
    """Run one serialized coordinator only after explicit startup permission.

    ``enabled`` controls scheduling, not resource admission. An enabled runtime
    without an admission callback can only leave durable jobs waiting.
    """

    def __init__(self, service, *, resource_admission=None,
                 worker_factory=SpawnTrainingWorker, interval_seconds=1, enabled=False):
        if (isinstance(interval_seconds, bool) or not isinstance(interval_seconds, (int, float))
                or not math.isfinite(interval_seconds) or interval_seconds <= 0):
            raise ValueError("training scheduler interval must be positive and finite")
        self.enabled = bool(enabled)
        self.interval_seconds = float(interval_seconds)
        self.coordinator = TaskTrainingCoordinator(service,
            resource_admission=resource_admission, worker_factory=worker_factory)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._started = False
        self._running = False
        self._closed = False
        self._cleanup_complete = True
        self._error_code = None

    def start(self):
        """Explicit lifespan boundary: acquire ownership, then start one thread."""
        with self._lock:
            if self._closed or not self.enabled:
                return False
            if self._started:
                return self._thread is not None and self._thread.is_alive()
            if not self.coordinator.start():
                snapshot = self.coordinator.snapshot()
                self._error_code = snapshot["error_code"] or "TRAINING_RUNTIME_OWNER_UNAVAILABLE"
                self._cleanup_complete = not snapshot["owner_held"] and not snapshot.get("cleanup_pending")
                return False
            self._started = True
            self._cleanup_complete = False
            self._error_code = None
            try:
                self._thread = threading.Thread(target=self._run,
                    name="task-training-runtime", daemon=True)
                self._running = True
                self._thread.start()
            except Exception:
                self._error_code = "TRAINING_RUNTIME_START_FAILED"
                self._closed = True
                self._running = False
                self._stop.set()
                # Thread creation/start may itself have an unknown outcome.
                # Keep the owner and let bounded close/join retry cleanup;
                # never perform potentially blocking DB cleanup under _lock.
                return False
            return True

    def _run(self):
        try:
            while not self._stop.is_set():
                try:
                    self.coordinator.tick()
                except Exception:
                    # Unknown claims/commits stay owned inside the coordinator;
                    # the next tick reads them back rather than replaying work.
                    with self._lock:
                        self._error_code = "TRAINING_RUNTIME_TICK_FAILED"
                else:
                    with self._lock:
                        self._error_code = None
                self._stop.wait(self.interval_seconds)
        finally:
            self._cleanup()

    def _cleanup(self):
        try:
            complete = self.coordinator.close()
        except Exception:
            complete = False
        with self._lock:
            self._cleanup_complete = complete
            self._running = False
            self._error_code = None if complete else "TRAINING_RUNTIME_CLEANUP_UNCONFIRMED"

    def snapshot(self):
        """Read cached lifecycle/coordinator state, never DB/SDK/device probes."""
        with self._lock:
            running = self._running
            value = {"enabled": self.enabled, "started": self._started,
                     "running": running, "stopping": self._stop.is_set(),
                     "cleanup_complete": self._cleanup_complete,
                     "error_code": self._error_code}
        coordinator = self.coordinator.snapshot()
        value["owner_held"] = coordinator["owner_held"]
        value["coordinator"] = coordinator
        error = value["error_code"] or coordinator["error_code"]
        if not value["enabled"]:
            state = "disabled"
        elif value["stopping"]:
            state = "stopping" if running else "stopped" if value["cleanup_complete"] else "cleanup_unconfirmed"
        elif error == "TRAINING_RUNTIME_OWNER_UNAVAILABLE":
            state = "owner_unavailable"
        elif coordinator.get("cleanup_pending"):
            state = "cleanup_unconfirmed"
        elif coordinator.get("unknown_job_ids"):
            state = "error"
        elif not value["started"]:
            state = "not_started"
        elif coordinator["reconciling_job_ids"]:
            state = "reconciling"
        elif value["error_code"] or error == "TRAINING_RUNTIME_STATE_UNAVAILABLE":
            state = "error"
        elif coordinator["waiting_resources"]:
            state = "waiting_resources"
        else:
            state = "running" if running else "stopped"
        return {**value, "state": state, "error_code": error}

    @staticmethod
    def _timeout(timeout):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("training shutdown timeout must be finite and nonnegative")
        return float(timeout)

    def close(self, timeout=5):
        """Stop scheduling and join boundedly; false retains all unsafe claims."""
        timeout = self._timeout(timeout)
        with self._lock:
            self._closed = True
            self._stop.set()
        self.coordinator.request_stop()
        return self.join(timeout)

    def join(self, timeout=5):
        """Wait for shutdown and retry cleanup, including after a failed close."""
        timeout = self._timeout(timeout)
        deadline = time.monotonic() + timeout
        with self._lock:
            thread = self._thread
            complete = self._cleanup_complete
            if not self._stop.is_set():
                return False
            if thread is None and not self._started and complete:
                # Disabled/never-started runtimes own no resources or owner.
                return complete
            if (thread is None or not thread.is_alive()) and not complete:
                # Cleanup itself may wait for a stubborn worker. Keep every
                # retry on a single background thread so the caller's timeout
                # bounds shutdown latency without dropping resource ownership.
                try:
                    thread = threading.Thread(target=self._cleanup,
                        name="task-training-cleanup", daemon=True)
                    self._thread = thread
                    self._running = True
                    thread.start()
                except Exception:
                    self._running = False
                    self._error_code = "TRAINING_RUNTIME_CLEANUP_UNCONFIRMED"
                    return False
        if thread is not None and thread.ident is not None:
            if thread is threading.current_thread():
                return False
            thread.join(max(0, deadline - time.monotonic()))
            if thread.is_alive():
                return False
        with self._lock:
            return self._cleanup_complete
