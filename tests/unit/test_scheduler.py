"""Tests for the task scheduler."""

from argus.core.scheduler import TaskScheduler


class TestTaskScheduler:
    def test_start_without_apscheduler(self):
        """Should handle missing apscheduler gracefully."""
        scheduler = TaskScheduler()
        # If apscheduler is installed, this works; if not, it's a no-op
        scheduler.start()
        scheduler.stop()

    def test_add_task_without_start(self):
        """Adding task without starting should be a no-op."""
        scheduler = TaskScheduler()
        # No scheduler backend, so this should not raise
        scheduler.add_interval_task("test", lambda: None, seconds=10)

    def test_remove_nonexistent_task(self):
        """Removing a non-existent task should not raise."""
        scheduler = TaskScheduler()
        scheduler.remove_task("nonexistent")
