"""Resource admission/compute exclusion with synthetic observations only."""

from types import SimpleNamespace
import threading
from unittest.mock import Mock

import pytest

from argus.capture.manager import CameraManager
from argus.config.schema import AlertConfig, TaskTrainingConfig
from argus.runtime.task_training_resources import NvidiaTrainingProbe, TaskTrainingResourceAdmission


def config(**values):
    return TaskTrainingConfig(enabled=True, maintenance_approved=True, **values)


def close_manager(manager):
    manager._inference_executor.shutdown(wait=False)
    manager._encode_executor.shutdown(wait=False)


def test_disabled_or_unapproved_never_probes_devices_or_stops_cameras():
    for configuration in (TaskTrainingConfig(), TaskTrainingConfig(enabled=True)):
        manager, probe = Mock(), Mock()
        admission = TaskTrainingResourceAdmission(configuration, manager, probe=probe)
        assert admission({}) is None
        probe.assert_not_called()
        manager.enter_training_maintenance.assert_not_called()
        manager.stop_all.assert_not_called()


@pytest.mark.parametrize(("observation", "reason"), [
    ({"device": "cuda:0", "free_memory_mb": 4000, "active_compute": False}, "TRAINING_INSUFFICIENT_GPU_MEMORY"),
    ({"device": "cuda:0", "free_memory_mb": 9000, "active_compute": True}, "TRAINING_EXTERNAL_COMPUTE_ACTIVE"),
    ({"device": "cuda:1", "free_memory_mb": 9000, "active_compute": False}, "TRAINING_RESOURCE_OBSERVATION_UNAVAILABLE"),
    ({"device": "cuda:0", "free_memory_mb": True, "active_compute": False}, "TRAINING_RESOURCE_OBSERVATION_UNAVAILABLE"),
    ({}, "TRAINING_RESOURCE_OBSERVATION_UNAVAILABLE"),
])
def test_bad_or_insufficient_observation_retains_queue(observation, reason):
    manager = Mock(enter_training_maintenance=Mock(return_value=True))
    admission = TaskTrainingResourceAdmission(config(), manager, probe=lambda: observation)
    assert admission({}) is None
    assert admission.snapshot()["reason_code"] == reason


def test_single_retained_lease_and_readonly_snapshot():
    manager = Mock(enter_training_maintenance=Mock(return_value=True))
    probe = Mock(return_value={"device": "cuda:0", "free_memory_mb": 9000, "active_compute": False})
    admission = TaskTrainingResourceAdmission(config(), manager, probe=probe)
    assert admission.snapshot()["memory_limit_enforced"] is False
    probe.assert_not_called()
    lease = admission({})
    assert lease is not None and admission({}) is None
    assert probe.call_count == 1
    lease.release()
    lease.release()
    second = admission({})
    assert second is not None and second is not lease
    second.release()


def test_distinct_admission_instances_share_one_manager_training_slot():
    manager = Mock(enter_training_maintenance=Mock(return_value=True))
    probe = Mock(return_value={"device": "cuda:0", "free_memory_mb": 9000, "active_compute": False})
    first = TaskTrainingResourceAdmission(config(), manager, probe=probe)
    second = TaskTrainingResourceAdmission(config(), manager, probe=probe)
    lease = first({})
    successor = None
    try:
        assert lease is not None
        assert second({}) is None
        assert probe.call_count == 1
        assert second.snapshot()["reason_code"] == "TRAINING_RESOURCE_IN_USE"
        lease.release()
        successor = second({})
        assert successor is not None
        assert first.snapshot()["state"] == "leased"
        lease.release()  # A delayed old release cannot clear the successor.
        assert first({}) is None
    finally:
        if successor is not None:
            successor.release()
        if lease is not None:
            lease.release()


def test_shared_training_slot_is_reserved_before_slow_probe_and_released_on_known_failure():
    from argus.runtime.task_training_resources import training_resource_lease_held
    manager = Mock(enter_training_maintenance=Mock(return_value=True))
    entered, release = threading.Event(), threading.Event()
    def slow_probe():
        entered.set()
        assert release.wait(2)
        raise RuntimeError("synthetic observation failed before any worker was created")
    first = TaskTrainingResourceAdmission(config(), manager, probe=slow_probe)
    second_probe = Mock(return_value={"device": "cuda:0", "free_memory_mb": 9000, "active_compute": False})
    second = TaskTrainingResourceAdmission(config(), manager, probe=second_probe)
    results = []
    thread = threading.Thread(target=lambda: results.append(first({})))
    lease = None
    thread.start()
    try:
        assert entered.wait(1)
        assert training_resource_lease_held(manager) is True
        assert second({}) is None
        second_probe.assert_not_called()
        release.set()
        thread.join(2)
        assert not thread.is_alive() and results == [None]
        assert training_resource_lease_held(manager) is False
        lease = second({})
        assert lease is not None
        assert training_resource_lease_held(manager) is True
    finally:
        release.set()
        thread.join(2)
        if lease is not None:
            lease.release()
    assert training_resource_lease_held(manager) is False


@pytest.mark.parametrize("state", ["starting", "unknown", "RUNNING", "", "legacy-unrecognized"])
def test_unknown_durable_execution_state_requires_compute_quarantine(tmp_path, state):
    from argus.config.schema import ArgusConfig
    from argus.runtime.task_training_resources import training_compute_exclusion_required
    from argus.storage.database import Database
    from argus.storage.models import TaskTrainingExecutionRecord
    db = Database(f"sqlite:///{tmp_path / 'unknown-state.db'}")
    db.initialize()
    try:
        with db.get_session() as session:
            session.add(TaskTrainingExecutionRecord(job_id="unknown-owner", task_id="task", workflow_id="flow",
                status=state, revision=1, request_json="{}", request_sha256="0" * 64,
                created_by="synthetic-human"))
            session.commit()
        assert training_compute_exclusion_required(ArgusConfig(), db) is True
    finally:
        db.close()


def test_cached_snapshot_does_not_wait_for_resource_probe():
    entered, release, read = threading.Event(), threading.Event(), threading.Event()
    manager = Mock(enter_training_maintenance=Mock(return_value=True))
    def probe():
        entered.set()
        assert release.wait(2)
        return {"device": "cuda:0", "free_memory_mb": 9000, "active_compute": False}
    admission = TaskTrainingResourceAdmission(config(), manager, probe=probe)
    held = []
    worker = threading.Thread(target=lambda: held.append(admission({})))
    reader = threading.Thread(target=lambda: (admission.snapshot(), read.set()))
    worker.start()
    try:
        assert entered.wait(1)
        reader.start()
        assert read.wait(0.5)
    finally:
        release.set()
        worker.join(2)
        reader.join(2)
        if held and held[0] is not None:
            held[0].release()


def test_probe_failure_and_old_training_activity_are_fail_closed():
    manager = Mock(enter_training_maintenance=Mock(return_value=True))
    probe = Mock(side_effect=RuntimeError("private credentials must not be exposed"))
    admission = TaskTrainingResourceAdmission(config(), manager, probe=probe)
    assert admission({}) is None
    assert "private" not in str(admission.snapshot())
    database = Mock(list_training_jobs=Mock(return_value=[{"status": "running"}]))
    admission = TaskTrainingResourceAdmission(config(), manager, database=database, probe=probe)
    assert admission({}) is None
    assert admission.snapshot()["reason_code"] == "TRAINING_LEGACY_JOB_ACTIVE"


def test_maintenance_entry_excludes_model_start_but_not_source_ownership():
    manager = CameraManager([], AlertConfig())
    source = object()
    manager._sources["camera"] = source
    assert manager.enter_training_maintenance() is True
    assert manager._sources["camera"] is source
    with pytest.raises(RuntimeError):
        manager.acquire_task_compute_reservation()
    close_manager(manager)


def test_prepared_compute_reservation_fences_training_before_worker_load():
    manager = CameraManager([], AlertConfig())
    reservation = manager.acquire_task_compute_reservation()
    assert manager.enter_training_maintenance() is False
    reservation.release()
    reservation.release()
    assert manager.enter_training_maintenance() is True
    close_manager(manager)


def test_assembled_managers_exclude_other_databases_before_model_load(tmp_path):
    from argus.runtime.compute_ownership import ComputeOwnership
    first, second = CameraManager([], AlertConfig()), CameraManager([], AlertConfig())
    path = tmp_path / "host-compute.lock"
    first.enable_compute_ownership(ComputeOwnership(path))
    second.enable_compute_ownership(ComputeOwnership(path))
    try:
        assert first.enter_training_maintenance() is True
        assert second.claim_compute_ownership() is False
        with pytest.raises(RuntimeError, match="another Argus instance"):
            second.acquire_task_compute_reservation()
        assert first.close_compute_ownership() is True
        reservation = second.acquire_task_compute_reservation()
        assert second.close_compute_ownership() is False
        reservation.release()
        second._shared_yolo = object()
        assert second.close_compute_ownership() is False
        second._shared_yolo = None
        assert second.close_compute_ownership() is True
    finally:
        first.close_compute_ownership()
        second.close_compute_ownership()
        close_manager(first)
        close_manager(second)


def test_legacy_background_executor_host_lock_remains_until_process_exit(tmp_path):
    from argus.runtime.compute_ownership import ComputeOwnership
    manager = CameraManager([], AlertConfig())
    ownership = ComputeOwnership(tmp_path / "legacy-host.lock")
    peer, other = ComputeOwnership(ownership.path), object()
    manager.enable_compute_ownership(ownership, retain_until_exit=True)
    try:
        assert manager.claim_compute_ownership() is True
        manager.enable_compute_ownership()  # Reassembly cannot weaken retention.
        assert manager.close_compute_ownership() is False
        assert peer.acquire(other) is False
    finally:
        # There are no actual background executors in this synthetic fixture.
        assert ownership.release(manager)
        close_manager(manager)


@pytest.mark.parametrize("field", ["_pipelines", "_runners", "_threads", "_retiring_cameras"])
def test_retained_legacy_ownership_prevents_maintenance(field):
    manager = CameraManager([], AlertConfig())
    getattr(manager, field)["camera"] = object()
    assert manager.enter_training_maintenance() is False
    close_manager(manager)


def test_nvidia_probe_is_bounded_and_rejects_ambiguous_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("argus.runtime.task_training_resources.shutil.which", lambda _: "mock-nvidia-smi")
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout="0, GPU-test, 12000, 9000\n" if len(calls) == 1 else "")
    monkeypatch.setattr("argus.runtime.task_training_resources.subprocess.run", run)
    assert NvidiaTrainingProbe()() == {"device": "cuda:0", "free_memory_mb": 9000, "active_compute": False}
    assert len(calls) == 2 and all(call[1]["timeout"] == 5 for call in calls)
    monkeypatch.setattr("argus.runtime.task_training_resources.subprocess.run", lambda *_args, **_kw:
                        SimpleNamespace(stdout="0, GPU-a, 12000, 9000\n1, GPU-b, 12000, 9000\n"))
    with pytest.raises(RuntimeError):
        NvidiaTrainingProbe()()


@pytest.mark.parametrize("mapping", ["", "1", "0,1", "GPU-other", "-1"])
def test_nvidia_probe_rejects_hidden_or_remapped_cuda_before_query(monkeypatch, mapping):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mapping)
    query = Mock(side_effect=AssertionError("a remapped or hidden device cannot be admitted"))
    monkeypatch.setattr("argus.runtime.task_training_resources.subprocess.run", query)
    with pytest.raises(RuntimeError, match="device mapping unavailable"):
        NvidiaTrainingProbe()()
    query.assert_not_called()
