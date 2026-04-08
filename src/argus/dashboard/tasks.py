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
        # Check concurrent limit
        running = sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING)
        if running >= self._max_concurrent:
            raise RuntimeError(f"已达到最大并发任务数 ({self._max_concurrent})")

        # Check for duplicate camera tasks of the same type
        for t in self._tasks.values():
            if (
                t.task_type == task_type
                and t.camera_id == camera_id
                and t.status in (TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.PAUSED)
            ):
                raise RuntimeError(f"摄像头 {camera_id} 已有相同类型的任务在运行")

        task_id = f"{task_type}-{uuid.uuid4().hex[:8]}"
        info = TaskInfo(task_id=task_id, task_type=task_type, camera_id=camera_id)

        with self._lock:
            self._tasks[task_id] = info

        def _progress_callback(progress: int, message: str):
            with self._lock:
                info.progress = min(max(progress, 0), 100)
                info.message = message
            self._notify_task_change(info)

        def _run():
            with self._lock:
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
            except Exception as e:
                with self._lock:
                    info.status = TaskStatus.FAILED
                    info.error = str(e)
                    info.message = f"失败: {e}"
                    info.completed_at = time.monotonic()
                self._notify_task_change(info)
                logger.error("task.failed", task_id=task_id, error=str(e))

        thread = threading.Thread(target=_run, name=f"argus-task-{task_id}", daemon=True)
        thread.start()

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
        except Exception as e:
            logger.debug("task.notify_failed", error=str(e))

    def pause_task(self, task_id: str) -> bool:
        """Pause a running task. The task must check pause_event in its loop."""
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.RUNNING:
            with self._lock:
                task.pause_event.clear()
                task.status = TaskStatus.PAUSED
                task.message = "已暂停"
            self._notify_task_change(task)
            logger.info("task.paused", task_id=task_id)
            return True
        return False

    def resume_task(self, task_id: str) -> bool:
        """Resume a paused task."""
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.PAUSED:
            with self._lock:
                task.status = TaskStatus.RUNNING
                task.message = "已恢复"
                task.pause_event.set()
            self._notify_task_change(task)
            logger.info("task.resumed", task_id=task_id)
            return True
        return False

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
        return sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)

    def get_active_tasks(self) -> list[TaskInfo]:
        """Get running and recently completed tasks."""
        now = time.monotonic()
        return [
            t for t in self._tasks.values()
            if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.PAUSED)
            or (t.completed_at and now - t.completed_at < 300)  # show completed for 5 min
        ]

    def dismiss(self, task_id: str) -> bool:
        """Remove a completed/failed task from the list."""
        task = self._tasks.get(task_id)
        if task and task.status in (TaskStatus.COMPLETE, TaskStatus.FAILED, TaskStatus.ABORTED):
            del self._tasks[task_id]
            return True
        return False

    def cleanup_old(self, max_age_seconds: float = 3600) -> int:
        """Remove tasks completed more than max_age_seconds ago."""
        now = time.monotonic()
        to_remove = [
            tid for tid, t in self._tasks.items()
            if t.completed_at and (now - t.completed_at) > max_age_seconds
        ]
        for tid in to_remove:
            del self._tasks[tid]
        return len(to_remove)
