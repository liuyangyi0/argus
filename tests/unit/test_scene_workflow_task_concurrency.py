"""Task reservation races with held/failing worker-thread fakes."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from argus.dashboard import tasks


def _thread_facade(monkeypatch, factory):
    # Replace only the task module's facade, not the process-wide threading
    # module used by the two real submitter threads in the race tests.
    monkeypatch.setattr(tasks, "threading", SimpleNamespace(
        Thread=factory, Lock=threading.Lock, Event=threading.Event,
    ))


class _HeldWorker:
    def __init__(self, *, target, name, daemon):
        self.target, self.name, self.daemon = target, name, daemon

    def start(self):
        # Preserve PENDING, forcing it to count as a live reservation.
        return None


class _ContendedEntryLock:
    """Both submitters reach reservation before either can inspect/register."""

    def __init__(self):
        self._entries = threading.Barrier(2)
        self._lock = threading.Lock()
        self._entry_count_lock = threading.Lock()
        self._entry_count = 0

    def __enter__(self):
        with self._entry_count_lock:
            self._entry_count += 1
            contend = self._entry_count <= 2
        if contend:
            self._entries.wait(timeout=5)
        self._lock.acquire()
        return self

    def __exit__(self, *_args):
        self._lock.release()


@pytest.mark.parametrize("same_camera,limit", [(False, 1), (True, 2)])
def test_two_simultaneous_scene_submissions_cannot_overbook_or_duplicate(
    monkeypatch, same_camera, limit,
):
    _thread_facade(monkeypatch, _HeldWorker)
    manager = tasks.TaskManager(max_concurrent=limit)
    manager._lock = _ContendedEntryLock()
    started = threading.Barrier(3)
    results, failures = [], []
    result_lock = threading.Lock()
    callback = Mock()

    def submit(index):
        started.wait(timeout=5)
        try:
            task_id = manager.submit(
                "scene_baseline", callback,
                camera_id="same-camera" if same_camera else f"camera-{index}",
                workflow_id=f"workflow-{index}",
            )
            with result_lock:
                results.append(task_id)
        except BaseException as exc:
            with result_lock:
                failures.append(exc)

    submitters = [threading.Thread(target=submit, args=(index,)) for index in range(2)]
    for thread in submitters:
        thread.start()
    started.wait(timeout=5)
    for thread in submitters:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert len(results) == 1 and len(failures) == 1
    assert isinstance(failures[0], tasks.TaskSubmissionRejected)
    assert len(manager.get_all_tasks()) == 1
    assert manager.get_task(results[0]).status == tasks.TaskStatus.PENDING
    callback.assert_not_called()


@pytest.mark.parametrize("reserved_status", [
    tasks.TaskStatus.PENDING, tasks.TaskStatus.RUNNING, tasks.TaskStatus.PAUSED,
])
def test_all_active_reservations_consume_global_budget(monkeypatch, reserved_status):
    _thread_facade(monkeypatch, _HeldWorker)
    manager = tasks.TaskManager(max_concurrent=1)
    task_id = manager.submit("scene_baseline", Mock(), camera_id="camera-a")
    manager.get_task(task_id).status = reserved_status
    with pytest.raises(tasks.TaskSubmissionRejected, match="最大并发"):
        manager.submit("scene_baseline", Mock(), camera_id="camera-b")
    assert len(manager.get_all_tasks()) == 1


@pytest.mark.parametrize("phase", ["construct", "start"])
def test_worker_start_failure_is_terminal_not_a_permanent_pending_slot(monkeypatch, phase):
    failure = RuntimeError("worker unavailable")

    class FailingWorker:
        def __init__(self, **_kwargs):
            if phase == "construct":
                raise failure

        def start(self):
            raise failure

    _thread_facade(monkeypatch, FailingWorker)
    changed = Mock()
    manager = tasks.TaskManager(max_concurrent=1, on_change=changed)
    with pytest.raises(RuntimeError) as caught:
        manager.submit("scene_baseline", Mock(), camera_id="camera-a")
    assert caught.value is failure
    assert not isinstance(caught.value, tasks.TaskSubmissionRejected)
    failed = manager.get_all_tasks()[0]
    assert failed.status == tasks.TaskStatus.FAILED
    assert failed.completed_at is not None
    assert "RuntimeError" in failed.error
    assert changed.call_args.args[1]["status"] == "failed"

    _thread_facade(monkeypatch, _HeldWorker)
    replacement = manager.submit("scene_baseline", Mock(), camera_id="camera-a")
    assert manager.get_task(replacement).status == tasks.TaskStatus.PENDING


def test_a_late_thread_cannot_resurrect_a_failed_start_reservation(monkeypatch):
    late = []

    class LateWorker:
        def __init__(self, *, target, **_kwargs):
            late.append(target)

        def start(self):
            raise RuntimeError("starter outcome unknown")

    _thread_facade(monkeypatch, LateWorker)
    manager = tasks.TaskManager(max_concurrent=1)
    callback = Mock()
    with pytest.raises(RuntimeError, match="starter outcome"):
        manager.submit("scene_baseline", callback, camera_id="camera-a")
    failed = manager.get_all_tasks()[0]
    _thread_facade(monkeypatch, _HeldWorker)
    replacement = manager.submit("scene_baseline", Mock(), camera_id="camera-a")
    late[0]()

    callback.assert_not_called()
    assert failed.status == tasks.TaskStatus.FAILED
    assert manager.get_task(replacement).status == tasks.TaskStatus.PENDING


def test_immediate_worker_preserves_camera_metadata_without_forwarding_it(monkeypatch):
    class ImmediateWorker:
        def __init__(self, *, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

    _thread_facade(monkeypatch, ImmediateWorker)
    manager = tasks.TaskManager(max_concurrent=1)
    callback = Mock(return_value={"built": True})
    task_id = manager.submit(
        "scene_baseline", callback, camera_id="camera-a", workflow_id="workflow-01",
    )
    callback.assert_called_once()
    assert len(callback.call_args.args) == 1
    assert callable(callback.call_args.args[0])
    assert callback.call_args.kwargs == {"workflow_id": "workflow-01"}
    info = manager.get_task(task_id)
    assert info.camera_id == "camera-a" and info.status == tasks.TaskStatus.COMPLETE


def test_starter_error_after_completed_worker_does_not_overwrite_its_known_outcome(monkeypatch):
    class CompletedThenRaised:
        def __init__(self, *, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()
            raise RuntimeError("starter response lost after worker completion")

    _thread_facade(monkeypatch, CompletedThenRaised)
    manager = tasks.TaskManager(max_concurrent=1)
    callback = Mock(return_value="completed output")
    with pytest.raises(RuntimeError, match="starter response lost"):
        manager.submit("scene_baseline", callback, camera_id="camera-a")
    info = manager.get_all_tasks()[0]
    assert info.status == tasks.TaskStatus.COMPLETE
    assert info.result == "completed output" and info.error == ""
    callback.assert_called_once()


def test_after_reservation_errors_cannot_masquerade_as_a_preflight_rejection(monkeypatch):
    class CompletedThenRejected:
        def __init__(self, *, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()
            raise tasks.TaskSubmissionRejected("misclassified custom starter error")

    _thread_facade(monkeypatch, CompletedThenRejected)
    manager = tasks.TaskManager(max_concurrent=1)
    callback = Mock(return_value="published baseline")
    with pytest.raises(RuntimeError, match="after reservation") as caught:
        manager.submit("scene_baseline", callback, camera_id="camera-a")
    assert not isinstance(caught.value, tasks.TaskSubmissionRejected)
    assert isinstance(caught.value.__cause__, tasks.TaskSubmissionRejected)
    info = manager.get_all_tasks()[0]
    assert info.status == tasks.TaskStatus.COMPLETE and info.result == "published baseline"
    callback.assert_called_once()


@pytest.mark.parametrize("operation,initial_status", [
    ("pause_task", tasks.TaskStatus.RUNNING), ("resume_task", tasks.TaskStatus.PAUSED),
])
def test_pause_resume_cannot_resurrect_a_worker_that_finished_before_lock_acquisition(
    monkeypatch, operation, initial_status,
):
    _thread_facade(monkeypatch, _HeldWorker)
    changed = Mock()
    manager = tasks.TaskManager(max_concurrent=1, on_change=changed)
    task_id = manager.submit("scene_baseline", Mock(), camera_id="camera-a")
    info = manager.get_task(task_id)
    info.status = initial_status

    class WorkerFinishedAtLockEntry:
        def __enter__(self):
            info.status = tasks.TaskStatus.COMPLETE
            info.result = "frozen output"
            return self

        def __exit__(self, *_args):
            return False

    manager._lock = WorkerFinishedAtLockEntry()
    assert getattr(manager, operation)(task_id) is False
    assert info.status == tasks.TaskStatus.COMPLETE
    assert info.result == "frozen output"
    changed.assert_not_called()


class _UnprintableTaskError(Exception):
    def __str__(self):
        raise RuntimeError("cannot format error")


@pytest.mark.parametrize("failure", [SystemExit("worker exit"), KeyboardInterrupt(), _UnprintableTaskError()])
def test_background_control_exceptions_release_the_slot_with_explicit_failure(monkeypatch, failure):
    class ImmediateWorker:
        def __init__(self, *, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

    _thread_facade(monkeypatch, ImmediateWorker)
    manager = tasks.TaskManager(max_concurrent=1)

    def worker(_progress):
        raise failure

    task_id = manager.submit("scene_baseline", worker, camera_id="camera-a")
    info = manager.get_task(task_id)
    assert info.status == tasks.TaskStatus.FAILED
    assert info.completed_at is not None and info.error
    _thread_facade(monkeypatch, _HeldWorker)
    retry = manager.submit("scene_baseline", Mock(), camera_id="camera-a")
    assert manager.get_task(retry).status == tasks.TaskStatus.PENDING


@pytest.mark.parametrize("failure", [SystemExit("observer exit"), KeyboardInterrupt(), _UnprintableTaskError()])
def test_notification_control_exceptions_cannot_prevent_the_worker_terminal_outcome(monkeypatch, failure):
    class ImmediateWorker:
        def __init__(self, *, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

    _thread_facade(monkeypatch, ImmediateWorker)
    changed = Mock(side_effect=failure)
    manager = tasks.TaskManager(max_concurrent=1, on_change=changed)

    def worker(progress):
        progress(10, "baseline decoding")
        return "immutable baseline"

    task_id = manager.submit("scene_baseline", worker, camera_id="camera-a")
    info = manager.get_task(task_id)
    assert info.status == tasks.TaskStatus.COMPLETE and info.result == "immutable baseline"
    assert info.completed_at is not None
    assert changed.call_count == 3  # RUNNING, progress, COMPLETE all remain observational.


def test_task_collection_snapshots_and_terminal_deletions_hold_the_shared_lock(monkeypatch):
    _thread_facade(monkeypatch, _HeldWorker)
    manager = tasks.TaskManager(max_concurrent=2)
    old = manager.submit("scene_baseline", Mock(), camera_id="camera-a")
    recent = manager.submit("scene_baseline", Mock(), camera_id="camera-b")
    manager.get_task(old).status = tasks.TaskStatus.COMPLETE
    manager.get_task(old).completed_at = 1.0
    manager.get_task(recent).status = tasks.TaskStatus.COMPLETE
    manager.get_task(recent).completed_at = 990.0
    monkeypatch.setattr(tasks, "time", SimpleNamespace(monotonic=lambda: 1000.0))

    class GuardedCollection(dict):
        def values(self):
            assert manager._lock.locked(), "task collection iteration requires its shared lock"
            return super().values()

        def items(self):
            assert manager._lock.locked(), "cleanup discovery and deletion must share ownership"
            return super().items()

        def __delitem__(self, key):
            assert manager._lock.locked(), "dismiss and cleanup must not race another remover"
            return super().__delitem__(key)

    manager._tasks = GuardedCollection(manager._tasks)
    assert {item.task_id for item in manager.get_all_tasks()} == {old, recent}
    assert [item.task_id for item in manager.get_active_tasks()] == [recent]
    assert manager.cleanup_old(max_age_seconds=100) == 1
    assert manager.get_task(old) is None and manager.get_task(recent) is not None
    assert manager.dismiss(recent) is True
    assert manager.dismiss(recent) is False
    assert manager.cleanup_old(max_age_seconds=100) == 0


def test_simultaneous_terminal_dismissals_have_one_successful_owner(monkeypatch):
    _thread_facade(monkeypatch, _HeldWorker)
    manager = tasks.TaskManager(max_concurrent=1)
    task_id = manager.submit("scene_baseline", Mock(), camera_id="camera-a")
    manager.get_task(task_id).status = tasks.TaskStatus.COMPLETE
    entry = threading.Barrier(3)
    results, failures = [], []
    result_lock = threading.Lock()

    def dismiss():
        entry.wait(timeout=5)
        try:
            result = manager.dismiss(task_id)
            with result_lock:
                results.append(result)
        except BaseException as exc:
            with result_lock:
                failures.append(exc)

    workers = [threading.Thread(target=dismiss) for _ in range(2)]
    for worker in workers:
        worker.start()
    entry.wait(timeout=5)
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive()
    assert failures == [] and sorted(results) == [False, True]
    assert manager.get_task(task_id) is None
