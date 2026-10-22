"""Background task manager for long-running operations.

Manages baseline capture, model training, and other time-consuming tasks
using Python threads. Progress is reported via callbacks and polled by
HTMX from the dashboard.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import structlog

logger = structlog.get_logger()


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    PAUSED = "paused"
    ABORTED = "aborted"


def _set_event() -> threading.Event:
    e = threading.Event()
    e.set()  # not paused by default
    return e


class TaskAbortedError(Exception):
    """Raised when a task detects the abort event."""


class TaskSubmissionRejected(RuntimeError):
    """Preflight rejection: no task was reserved and no worker was started.

    Callers may retry these failures. Errors after reservation deliberately do
    not use this type because a worker may already have produced side effects.
    """


@dataclass
class TaskInfo:
    """State of a background task."""

    task_id: str
    task_type: str  # e.g., "baseline_capture", "model_training"
    camera_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0  # 0-100
    message: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.monotonic)
    completed_at: float | None = None
    result: Any = None
    pause_event: threading.Event = field(default_factory=lambda: _set_event())
    abort_event: threading.Event = field(default_factory=threading.Event)


class TaskManager:
    """Manages background tasks with progress tracking.

    Usage:
        manager = TaskManager()
        task_id = manager.submit("baseline_capture", my_func, camera_id="cam_01", count=100)
        info = manager.get_task(task_id)  # poll for progress
    """

    def __init__(self, max_concurrent: int = 2, on_change: Callable[[str, dict], None] | None = None):
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = threading.Lock()
        self._max_concurrent = max_concurrent
        self._on_change = on_change

    def submit(
        self,
        task_type: str,
        func: Callable,
        camera_id: str = "",
        **kwargs,
    ) -> str:
        """Submit a task for background execution.

        The func must accept a `progress_callback(progress: int, message: str)`
        as its first positional argument, followed by any **kwargs.

        Returns the task_id.
        """
        # Reserve both the global slot and camera/type ownership atomically.
        # A registered-but-not-yet-started worker and a paused worker still own
        # their budget; counting RUNNING alone allows parallel submitters to
        # oversubscribe before either new thread gets scheduled.
        with self._lock:
            active = [
                task for task in tuple(self._tasks.values())
                if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.PAUSED)
            ]
            if len(active) >= self._max_concurrent:
                raise TaskSubmissionRejected(f"已达到最大并发任务数 ({self._max_concurrent})")
            if any(task.task_type == task_type and task.camera_id == camera_id for task in active):
                raise TaskSubmissionRejected(f"摄像头 {camera_id} 已有相同类型的任务在运行")
            task_id = f"{task_type}-{uuid.uuid4().hex[:8]}"
            info = TaskInfo(task_id=task_id, task_type=task_type, camera_id=camera_id)
            self._tasks[task_id] = info

        def _progress_callback(progress: int, message: str):
            with self._lock:
                info.progress = min(max(progress, 0), 100)
                info.message = message
            self._notify_task_change(info)

        def _run():
            with self._lock:
                if info.status != TaskStatus.PENDING:
                    # A failed starter may have scheduled a late thread. The
                    # reservation was already failed/released, so it must not
                    # resurrect itself and run outside the concurrency budget.
                    return
                info.status = TaskStatus.RUNNING
                info.message = "正在启动..."
            self._notify_task_change(info)
            try:
                result = func(_progress_callback, **kwargs)
                with self._lock:
                    info.status = TaskStatus.COMPLETE
                    info.progress = 100
                    info.message = "完成"
                    info.result = result
                    info.completed_at = time.monotonic()
                self._notify_task_change(info)
                logger.info("task.complete", task_id=task_id, task_type=task_type)
            except TaskAbortedError:
                with self._lock:
                    info.status = TaskStatus.ABORTED
                    info.message = "已中止"
                    info.completed_at = time.monotonic()
                self._notify_task_change(info)
                logger.info("task.aborted", task_id=task_id, task_type=task_type)
            except BaseException as e:
                # A background callable may raise SystemExit/KeyboardInterrupt
                # (or have a broken __str__). It must still release its slot
                # with an explicit failure instead of remaining RUNNING forever.
                try:
                    error = str(e) or type(e).__name__
                except BaseException:
                    error = type(e).__name__
                with self._lock:
                    info.status = TaskStatus.FAILED
                    info.error = error
                    info.message = f"失败: {error}"
                    info.completed_at = time.monotonic()
                self._notify_task_change(info)
                logger.error("task.failed", task_id=task_id, error=error)

        try:
            # Starting outside the lock also supports a worker that immediately
            # reports progress or finishes before Thread.start() returns.
            thread = threading.Thread(target=_run, name=f"argus-task-{task_id}", daemon=True)
            thread.start()
        except BaseException as exc:
            failed_before_run = False
            with self._lock:
                if info.status == TaskStatus.PENDING:
                    info.status = TaskStatus.FAILED
                    info.error = f"Task worker could not start ({type(exc).__name__})"
                    info.message = "后台任务启动失败"
                    info.completed_at = time.monotonic()
                    failed_before_run = True
            if failed_before_run:
                self._notify_task_change(info)
            # If a custom starter actually launched the worker before failing,
            # do not overwrite its RUNNING/terminal outcome or free a live slot.
            if isinstance(exc, TaskSubmissionRejected):
                # Preserve the public distinction even if a custom starter
                # accidentally reuses the preflight error type after reservation.
                raise RuntimeError("Task startup failed after reservation; inspect task state before retry") from exc
            raise

        logger.info("task.submitted", task_id=task_id, task_type=task_type, camera_id=camera_id)
        return task_id

    def _notify_task_change(self, info: TaskInfo) -> None:
        """Notify WebSocket subscribers of task progress change.

        Throttled: only fires on status change, completion, or every 5% progress.
        """
        if not self._on_change:
            return
        last_progress = getattr(info, "_last_notified_progress", -10)
        is_terminal = info.status in (TaskStatus.COMPLETE, TaskStatus.FAILED, TaskStatus.ABORTED)
        if not is_terminal and abs(info.progress - last_progress) < 5:
            return
        info._last_notified_progress = info.progress
        try:
            self._on_change("tasks", {
                "task_id": info.task_id,
                "task_type": info.task_type,
                "camera_id": info.camera_id,
                "status": info.status.value,
                "progress": info.progress,
                "message": info.message,
                "error": info.error,
            })
        except BaseException as exc:
            # Notification hooks are observational: even a broken adapter must
            # not stop the worker before its terminal outcome is recorded.
            try:
                error = str(exc) or type(exc).__name__
            except BaseException:
                error = type(exc).__name__
            logger.debug("task.notify_failed", error=error)

    def pause_task(self, task_id: str) -> bool:
        """Pause a running task. The task must check pause_event in its loop."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != TaskStatus.RUNNING:
                return False
            task.pause_event.clear()
            task.status = TaskStatus.PAUSED
            task.message = "已暂停"
        self._notify_task_change(task)
        logger.info("task.paused", task_id=task_id)
        return True

    def resume_task(self, task_id: str) -> bool:
        """Resume a paused task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != TaskStatus.PAUSED:
                return False
            task.status = TaskStatus.RUNNING
            task.message = "已恢复"
            task.pause_event.set()
        self._notify_task_change(task)
        logger.info("task.resumed", task_id=task_id)
        return True

    def abort_task(self, task_id: str) -> bool:
        """Signal a running or paused task to abort."""
        task = self._tasks.get(task_id)
        if task and task.status in (TaskStatus.RUNNING, TaskStatus.PAUSED):
            task.abort_event.set()
            task.pause_event.set()  # unblock if paused so it can detect abort
            logger.info("task.abort_requested", task_id=task_id)
            return True
        return False

    def get_task(self, task_id: str) -> TaskInfo | None:
        """Get task info by ID."""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[TaskInfo]:
        """Get all tasks, most recent first."""
        with self._lock:
            snapshot = tuple(self._tasks.values())
        return sorted(snapshot, key=lambda t: t.created_at, reverse=True)

    def get_active_tasks(self) -> list[TaskInfo]:
        """Get running and recently completed tasks."""
        now = time.monotonic()
        with self._lock:
            snapshot = tuple(self._tasks.values())
        return [
            t for t in snapshot
            if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.PAUSED)
            or (t.completed_at and now - t.completed_at < 300)  # show completed for 5 min
        ]

    def dismiss(self, task_id: str) -> bool:
        """Remove a completed/failed task from the list."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status in (TaskStatus.COMPLETE, TaskStatus.FAILED, TaskStatus.ABORTED):
                del self._tasks[task_id]
                return True
            return False

    def cleanup_old(self, max_age_seconds: float = 3600) -> int:
        """Remove tasks completed more than max_age_seconds ago."""
        now = time.monotonic()
        with self._lock:
            to_remove = [
                tid for tid, t in self._tasks.items()
                if t.completed_at and (now - t.completed_at) > max_age_seconds
            ]
            for tid in to_remove:
                del self._tasks[tid]
        return len(to_remove)
