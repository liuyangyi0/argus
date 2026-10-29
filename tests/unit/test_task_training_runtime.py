"""Lifespan scheduler contracts with fake workers and fake OS observations."""

import threading
import time
from types import SimpleNamespace

import pytest

from argus.runtime.task_training_runtime import TaskTrainingRuntime
from test_task_training import FakeWorker, Lease, environment as _environment_fixture, training_environment

environment = _environment_fixture


def wait_until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    pulse = threading.Event()
    while time.monotonic() < deadline:
        if predicate():
            return
        pulse.wait(0.01)
    assert predicate()


class NotifyingWorker(FakeWorker):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.starts = 0

    def start(self, plan):
        super().start(plan)
        self.starts += 1
        self.started.set()


def test_construction_disabled_start_and_snapshots_are_inert(environment, tmp_path, monkeypatch):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    calls = []
    runtime = TaskTrainingRuntime(service,
        resource_admission=lambda _job: calls.append("admission"),
        worker_factory=lambda: calls.append("worker"))
    monkeypatch.setattr(service, "list", lambda: pytest.fail("snapshot may not read the queue"))
    monkeypatch.setattr("argus.runtime.task_training._process_alive", lambda _pid: pytest.fail("snapshot may not probe"))
    try:
        assert runtime.snapshot()["state"] == "disabled"
        assert runtime.snapshot()["enabled"] is False
        assert not runtime.coordinator._owner.path.exists()
        assert runtime.start() is False
        assert calls == []
        assert service.get(job["job_id"])["status"] == "queued"
        assert runtime.snapshot()["owner_held"] is False
    finally:
        assert runtime.close()


def test_explicit_enabled_start_resumes_queue_once_and_close_stops_before_release(environment, tmp_path):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    worker, lease = NotifyingWorker(), Lease()
    runtime = TaskTrainingRuntime(service, enabled=True, interval_seconds=0.01,
        resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    try:
        assert runtime.snapshot()["state"] == "not_started"
        assert runtime.start()
        assert worker.started.wait(2)
        wait_until(lambda: service.get(job["job_id"])["status"] == "running")
        assert runtime.start()  # Idempotent, not another thread or execution.
        assert worker.starts == 1 and lease.released == 0
        assert runtime.close(timeout=2)
        assert worker.stop_calls >= 1 and lease.released == 1
        assert runtime.snapshot()["state"] == "stopped"
        assert runtime.snapshot()["owner_held"] is False
        assert service.get(job["job_id"])["status"] == "failed"
        assert runtime.start() is False
    finally:
        assert runtime.close(timeout=2)


def test_enabled_without_admission_remains_waiting_and_snapshots_do_not_recheck_resources(environment, tmp_path, monkeypatch):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    runtime = TaskTrainingRuntime(service, enabled=True, interval_seconds=0.01,
        worker_factory=lambda: pytest.fail("no GPU resource lease"))
    try:
        assert runtime.start()
        wait_until(lambda: runtime.snapshot()["state"] == "waiting_resources")
        snapshot = runtime.snapshot()
        assert snapshot["error_code"] == "TRAINING_RESOURCE_APPROVAL_REQUIRED"
        assert snapshot["enabled"] is True and snapshot["owner_held"] is True
        assert service.get(job["job_id"])["status"] == "waiting_resources"
        assert runtime.close(timeout=2)
        monkeypatch.setattr(service, "list", lambda: pytest.fail("GET cannot drive recovery"))
        assert runtime.snapshot()["state"] == "stopped"
    finally:
        assert runtime.close(timeout=2)


def test_second_runtime_cannot_acquire_database_owner_or_mutate_running_job(environment, tmp_path):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    worker, lease = NotifyingWorker(), Lease()
    first = TaskTrainingRuntime(service, enabled=True, interval_seconds=0.01,
        resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    second = TaskTrainingRuntime(service, enabled=True, interval_seconds=0.01,
        resource_admission=lambda _job: pytest.fail("second owner must not admit"), worker_factory=FakeWorker)
    try:
        assert first.start() and worker.started.wait(2)
        wait_until(lambda: service.get(job["job_id"])["result"].get("worker_pid") is not None)
        before = service.get(job["job_id"])
        assert second.start() is False
        assert second.snapshot()["state"] == "owner_unavailable"
        assert second.snapshot()["error_code"] == "TRAINING_RUNTIME_OWNER_UNAVAILABLE"
        assert service.get(job["job_id"]) == before
    finally:
        assert second.close(timeout=2)
        assert first.close(timeout=2)


def test_snapshot_does_not_wait_on_blocked_admission_and_shutdown_is_bounded(environment, tmp_path):
    service, args = training_environment(environment, tmp_path)
    service.submit(**args)
    entered, proceed = threading.Event(), threading.Event()
    def admission(_job):
        entered.set()
        assert proceed.wait(3)
        return None
    runtime = TaskTrainingRuntime(service, enabled=True, resource_admission=admission, interval_seconds=0.01)
    try:
        assert runtime.start() and entered.wait(2)
        started = time.monotonic()
        assert runtime.snapshot()["owner_held"]
        assert time.monotonic() - started < 0.25
        assert runtime.close(timeout=0.01) is False
        assert runtime.snapshot()["owner_held"]
        proceed.set()
        assert runtime.join(timeout=2)
    finally:
        proceed.set()
        assert runtime.close(timeout=2)


def test_shutdown_during_resource_admission_releases_lease_without_consuming_queue(environment, tmp_path):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    entered, proceed = threading.Event(), threading.Event()
    lease = Lease()
    def admission(_job):
        entered.set()
        assert proceed.wait(3)
        return lease
    runtime = TaskTrainingRuntime(service, enabled=True, resource_admission=admission,
        worker_factory=lambda: pytest.fail("shutdown must not construct a worker"), interval_seconds=0.01)
    try:
        assert runtime.start() and entered.wait(2)
        assert runtime.close(timeout=0.01) is False
        proceed.set()
        assert runtime.join(timeout=2)
        assert lease.released == 1
        assert service.get(job["job_id"])["status"] == "queued"
    finally:
        proceed.set()
        assert runtime.close(timeout=2)


def test_close_keeps_unknown_cleanup_owned_then_retries_after_scheduler_is_closed(environment, tmp_path):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    worker, lease = NotifyingWorker(), Lease()
    runtime = TaskTrainingRuntime(service, enabled=True, interval_seconds=0.01,
        resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    try:
        assert runtime.start() and worker.started.wait(2)
        worker.cleanup = False
        assert runtime.close(timeout=2) is False
        assert runtime.snapshot()["state"] == "cleanup_unconfirmed"
        assert runtime.snapshot()["owner_held"] is True and lease.released == 0
        assert job["job_id"] in runtime.coordinator._owned
        worker.cleanup = True
        assert runtime.close(timeout=2)
        assert lease.released == 1 and not runtime.coordinator._owner.held
    finally:
        worker.cleanup = True
        assert runtime.close(timeout=2)


def test_worker_stop_cannot_make_close_exceed_join_timeout(environment, tmp_path):
    service, args = training_environment(environment, tmp_path)
    service.submit(**args)
    stop_entered, stopped = threading.Event(), threading.Event()
    class SlowStopWorker(NotifyingWorker):
        def stop(self):
            stop_entered.set()
            return stopped.wait(3)
    worker, lease = SlowStopWorker(), Lease()
    runtime = TaskTrainingRuntime(service, enabled=True, interval_seconds=0.01,
        resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    try:
        assert runtime.start() and worker.started.wait(2)
        before = time.monotonic()
        assert runtime.close(timeout=0.02) is False
        assert time.monotonic() - before < 0.3
        assert stop_entered.wait(1)
        assert lease.released == 0
        stopped.set()
        assert runtime.join(timeout=2)
        assert lease.released == 1
    finally:
        stopped.set()
        assert runtime.close(timeout=2)


def test_orphan_runtime_retains_owner_and_does_not_resume_until_exit_is_positive(environment, tmp_path, monkeypatch):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    claimed, plan = service.claim(job["job_id"], expected_revision=job["revision"])
    service.transition(job["job_id"], expected_revision=claimed["revision"], from_states={"running"},
        status="cancelling", result={"execution_id": plan["execution_id"], "worker_pid": 123456789})
    monkeypatch.setattr("argus.runtime.task_training._process_alive", lambda _pid: None)
    runtime = TaskTrainingRuntime(service, enabled=True, interval_seconds=0.01,
        resource_admission=lambda _job: pytest.fail("unknown orphan must block resource admission"))
    try:
        assert runtime.start()
        wait_until(lambda: runtime.snapshot()["state"] == "reconciling")
        assert runtime.close(timeout=2) is False
        assert runtime.coordinator._owner.held
        current = service.get(job["job_id"])
        assert current["status"] == "reconciling"
        monkeypatch.setattr("argus.runtime.task_training._process_alive", lambda _pid: False)
        final = runtime.coordinator.reconcile_exited(job["job_id"], expected_revision=current["revision"])
        assert final["status"] == "cancelled"
        assert runtime.close(timeout=2)
    finally:
        monkeypatch.setattr("argus.runtime.task_training._process_alive", lambda _pid: False)
        current = service.get(job["job_id"])
        if current["status"] == "reconciling":
            runtime.coordinator.reconcile_exited(job["job_id"], expected_revision=current["revision"])
        assert runtime.close(timeout=2)


@pytest.mark.parametrize("interval", [0, -1, True, float("inf"), float("nan")])
def test_runtime_rejects_unbounded_or_invalid_scheduler_intervals(environment, tmp_path, interval):
    service, _args = training_environment(environment, tmp_path)
    with pytest.raises(ValueError):
        TaskTrainingRuntime(service, interval_seconds=interval)


def test_adversarial_thread_construction_failure_retains_owner_and_reports_retryable_cleanup(environment, tmp_path, monkeypatch):
    service, _args = training_environment(environment, tmp_path)
    runtime = TaskTrainingRuntime(service, enabled=True)
    thread_factory = threading.Thread
    monkeypatch.setattr("argus.runtime.task_training_runtime.threading.Thread",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("private thread failure")))
    try:
        assert runtime.start() is False
        assert runtime.snapshot()["error_code"] == "TRAINING_RUNTIME_START_FAILED"
        assert runtime.coordinator._owner.held
    finally:
        monkeypatch.setattr("argus.runtime.task_training_runtime.threading.Thread", thread_factory)
        assert runtime.close(timeout=2)


def test_adversarial_unknown_owner_acquire_cleanup_cannot_be_certified_by_never_started_runtime(environment, tmp_path):
    service, _args = training_environment(environment, tmp_path)
    runtime = TaskTrainingRuntime(service, enabled=True)
    complete = False
    runtime.coordinator = SimpleNamespace(start=lambda: False, request_stop=lambda: None,
        snapshot=lambda: {"owner_held": False, "cleanup_pending": True, "unknown_job_ids": [],
            "error_code": "TRAINING_RUNTIME_OWNER_CLEANUP_UNCONFIRMED", "reconciling_job_ids": [], "waiting_resources": 0},
        close=lambda: complete)
    try:
        assert runtime.start() is False
        assert runtime.close(timeout=2) is False
    finally:
        complete = True
        assert runtime.close(timeout=2)


@pytest.mark.parametrize("cleanup_confirmed", [False, True])
def test_adversarial_shutdown_during_ready_prepare_never_authorizes_prepared_child(environment, tmp_path, cleanup_confirmed):
    service, args = training_environment(environment, tmp_path)
    job = service.submit(**args)
    preparing, ready = threading.Event(), threading.Event()
    class PreparingWorker(NotifyingWorker):
        def prepare(self, _plan):
            preparing.set()
            assert ready.wait(3)
    worker, lease = PreparingWorker(), Lease()
    worker.cleanup = cleanup_confirmed
    runtime = TaskTrainingRuntime(service, enabled=True, interval_seconds=0.01,
        resource_admission=lambda _job: lease, worker_factory=lambda: worker)
    try:
        assert runtime.start() and preparing.wait(2)
        assert runtime.close(timeout=0.01) is False
        ready.set()
        assert runtime.join(timeout=2) is cleanup_confirmed
        assert worker.starts == 0
        assert worker.stop_calls >= 1
        if not cleanup_confirmed:
            assert lease.released == 0 and runtime.coordinator._owner.held
            assert runtime.snapshot()["state"] == "cleanup_unconfirmed"
            worker.cleanup = True
            assert runtime.close(timeout=2)
        assert lease.released == 1
        current = service.get(job["job_id"])
        assert current["status"] == "failed"
        assert current["result"]["error_code"] == "TRAINING_COORDINATOR_CLOSED"
    finally:
        ready.set()
        worker.cleanup = True
        assert runtime.close(timeout=2)
