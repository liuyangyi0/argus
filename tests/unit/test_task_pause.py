"""Tests for TaskManager pause/resume/abort (Phase 1)."""

import threading
import time
import pytest
from argus.dashboard.tasks import TaskManager, TaskStatus, TaskAbortedError


def _slow_task(progress_callback, *, pause_event, abort_event, steps=10):
    """A task that checks pause/abort events each step."""
    for i in range(steps):
        pause_event.wait()  # block if paused
        if abort_event.is_set():
            raise TaskAbortedError("aborted")
        progress_callback(int(100 * (i + 1) / steps), f"Step {i+1}/{steps}")
        time.sleep(0.05)
    return {"completed_steps": steps}


class TestTaskManagerPauseResume:
    def test_pause_and_resume(self):
        mgr = TaskManager(max_concurrent=5)
        pause_ev = threading.Event()
        pause_ev.set()
        abort_ev = threading.Event()

        task_id = mgr.submit(
            "test_task", _slow_task,
            pause_event=pause_ev, abort_event=abort_ev, steps=20,
        )
        time.sleep(0.1)  # let it start

        assert mgr.pause_task(task_id) is True
        info = mgr.get_task(task_id)
        # Sync events
        info.pause_event = pause_ev
        info.abort_event = abort_ev
        # Actually use mgr.pause_task which clears the event
        # But we need the TaskInfo to have the same event objects
        # Let's re-check: pause_task uses task.pause_event.clear()
        # But the default pause_event on TaskInfo is a DIFFERENT object than pause_ev
        # We need to set them first
        # This test should actually set the events on the TaskInfo before pausing

        # Reset and redo properly:
        mgr2 = TaskManager(max_concurrent=5)
        pause_ev2 = threading.Event()
        pause_ev2.set()
        abort_ev2 = threading.Event()

        task_id2 = mgr2.submit(
            "test_task", _slow_task,
            pause_event=pause_ev2, abort_event=abort_ev2, steps=50,
        )
        # Sync events on TaskInfo
        info2 = mgr2.get_task(task_id2)
        info2.pause_event = pause_ev2
        info2.abort_event = abort_ev2

        time.sleep(0.1)
        assert info2.status == TaskStatus.RUNNING

        assert mgr2.pause_task(task_id2) is True
        assert info2.status == TaskStatus.PAUSED
        progress_at_pause = info2.progress

        time.sleep(0.2)
        # Progress should not advance while paused
        assert info2.progress == progress_at_pause

        assert mgr2.resume_task(task_id2) is True
        assert info2.status == TaskStatus.RUNNING

        # Wait for completion
        for _ in range(100):
            if info2.status == TaskStatus.COMPLETE:
                break
            time.sleep(0.1)
        assert info2.status == TaskStatus.COMPLETE

    def test_abort(self):
        mgr = TaskManager(max_concurrent=5)
        pause_ev = threading.Event()
        pause_ev.set()
        abort_ev = threading.Event()

        task_id = mgr.submit(
            "test_task", _slow_task,
            pause_event=pause_ev, abort_event=abort_ev, steps=100,
        )
        info = mgr.get_task(task_id)
        info.pause_event = pause_ev
        info.abort_event = abort_ev

        time.sleep(0.1)
        assert mgr.abort_task(task_id) is True

        for _ in range(50):
            if info.status == TaskStatus.ABORTED:
                break
            time.sleep(0.1)
        assert info.status == TaskStatus.ABORTED

    def test_abort_while_paused(self):
        mgr = TaskManager(max_concurrent=5)
        pause_ev = threading.Event()
        pause_ev.set()
        abort_ev = threading.Event()

        task_id = mgr.submit(
            "test_task", _slow_task,
            pause_event=pause_ev, abort_event=abort_ev, steps=100,
        )
        info = mgr.get_task(task_id)
        info.pause_event = pause_ev
        info.abort_event = abort_ev

        time.sleep(0.1)
        mgr.pause_task(task_id)
        time.sleep(0.1)
        assert info.status == TaskStatus.PAUSED

        # Abort while paused — should unblock the pause and abort
        mgr.abort_task(task_id)

        for _ in range(50):
            if info.status == TaskStatus.ABORTED:
                break
            time.sleep(0.1)
        assert info.status == TaskStatus.ABORTED

    def test_pause_nonexistent_task(self):
        mgr = TaskManager()
        assert mgr.pause_task("nonexistent") is False
        assert mgr.resume_task("nonexistent") is False
        assert mgr.abort_task("nonexistent") is False

    def test_dismiss_aborted_task(self):
        mgr = TaskManager(max_concurrent=5)
        pause_ev = threading.Event()
        pause_ev.set()
        abort_ev = threading.Event()

        task_id = mgr.submit(
            "test_task", _slow_task,
            pause_event=pause_ev, abort_event=abort_ev, steps=100,
        )
        info = mgr.get_task(task_id)
        info.pause_event = pause_ev
        info.abort_event = abort_ev

        time.sleep(0.1)
        mgr.abort_task(task_id)
        for _ in range(50):
            if info.status == TaskStatus.ABORTED:
                break
            time.sleep(0.1)

        assert mgr.dismiss(task_id) is True
        assert mgr.get_task(task_id) is None

    def test_paused_in_active_tasks(self):
        mgr = TaskManager(max_concurrent=5)
        pause_ev = threading.Event()
        pause_ev.set()
        abort_ev = threading.Event()

        task_id = mgr.submit(
            "test_task", _slow_task,
            pause_event=pause_ev, abort_event=abort_ev, steps=200,
        )
        info = mgr.get_task(task_id)
        info.pause_event = pause_ev
        info.abort_event = abort_ev

        time.sleep(0.1)
        mgr.pause_task(task_id)

        active = mgr.get_active_tasks()
        assert any(t.task_id == task_id for t in active)

        # Cleanup
        mgr.abort_task(task_id)
