"""Control-plane/runtime integration with synthetic frames and fake SDK workers."""

import threading
import time
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from argus.capture.frame_hub import FrameHub
from argus.capture.task_execution_manager import TaskExecutionManager
from argus.core.task_execution import TaskExecutionError, frame_provenance
from test_task_execution import execution_manifest, synthetic_frame


def wait_for(predicate):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


class SyntheticSource:
    camera_id = "cam"

    def __init__(self):
        self.hub, self.lock = FrameHub("cam"), threading.RLock()
        self.hub.activate("session", 1)

    @property
    def source_token(self):
        return self.hub.source_token

    def subscribe(self, name, **kwargs):
        return self.hub.subscribe(name, **kwargs)

    def commit_if_current(self, token, callback):
        with self.lock:
            if token != self.source_token:
                return False
            callback()
            return True

    def reconnect(self):
        with self.lock:
            self.hub.activate("session", self.source_token[1] + 1)


class FakeTaskService:
    def __init__(self):
        self.manifests = {f"task-{number}": execution_manifest(task_id=f"task-{number}") for number in range(1, 4)}
        self.tasks = {key: {"authority": "managed", "template_id": value["template_id"],
                            "camera_id": "cam", "current_revision": 1,
                            "config_sha256": value["task_config_sha256"], "archived": False}
                      for key, value in self.manifests.items()}
        self.lock = threading.RLock()
        self.before_commit = None

    def resolve_bundle(self, task_id, bundle_id):
        return deepcopy(self.manifests[task_id])

    def get_task(self, task_id):
        with self.lock:
            return deepcopy(self.tasks.get(task_id))

    def commit_if_current(self, task_id, revision, config_sha256, callback):
        with self.lock:
            if self.before_commit:
                self.before_commit()
            task = self.tasks[task_id]
            if task["archived"] or revision != task["current_revision"] or config_sha256 != task["config_sha256"]:
                return False
            callback()
            return True


class FakeCameraManager:
    def __init__(self):
        self.source = SyntheticSource()
        self.tasks = {}
        self.allow_unregister = True
        self.scene = {"enabled": True, "usable": False, "reference_available": True,
                      "consumer_enabled": True, "profile_id": "approved-scene", "source_token": ("session", 1)}

    def get_camera_source(self, camera_id):
        return self.source

    def get_managed_scene_guard_status(self, camera_id, *, source):
        status = deepcopy(self.scene)
        if source is not self.source or status["source_token"] != self.source.source_token:
            status["usable"] = False
        return status

    def commit_managed_scene_if_current(self, camera_id, *, source, profile_id, callback):
        status = self.get_managed_scene_guard_status(camera_id, source=source)
        if not status["usable"] or not status["consumer_enabled"] or status["profile_id"] != profile_id:
            return False
        callback()
        return True

    def register_analysis_task_runtime(self, runtime):
        if runtime.task_id in self.tasks:
            raise RuntimeError("duplicate task")
        self.tasks[runtime.task_id] = runtime

    def unregister_analysis_task_runtime(self, task_id, *, runtime):
        if not self.allow_unregister:
            return False
        if self.tasks.get(task_id) is runtime:
            assert runtime.detach_if_clean()
            del self.tasks[task_id]
        return True


class FakeWorker:
    def __init__(self, manifest, models_root):
        self.manifest = manifest
        self.running, self.cleanup_ok = False, True
        self.entered, self.release = threading.Event(), threading.Event()
        self.hang = False
        self.resets = []

    def start(self):
        self.running = True

    def infer(self, frame):
        self.entered.set()
        if self.hang:
            assert self.release.wait(2)
        return {"source": frame_provenance(frame), "bundle_id": self.manifest["bundle_id"],
                "task_revision": self.manifest["task_revision"], "data": {"state": "observed"}}

    def reset(self, reason):
        self.resets.append(reason)

    def stop(self):
        if self.cleanup_ok:
            self.running = False
            self.release.set()
        return self.cleanup_ok

    def snapshot(self):
        return {"state": "running" if self.running else "stopped"}


@pytest.fixture
def platform(tmp_path):
    service, camera = FakeTaskService(), FakeCameraManager()
    manager = TaskExecutionManager(service, camera, models_root=tmp_path, worker_factory=FakeWorker)
    yield manager, service, camera
    for execution in manager._executions.values():
        execution.worker.cleanup_ok = True
        execution.worker.release.set()
    camera.allow_unregister = True
    assert manager.close()


@pytest.fixture
def compute_exclusion_gate(platform):
    from argus.capture.manager import CameraManager
    from argus.config.schema import AlertConfig
    gate = CameraManager([], AlertConfig())
    platform[2].acquire_task_compute_reservation = gate.acquire_task_compute_reservation
    yield gate
    # Fixture callers confirm cleanup before releasing this local mock gate.
    gate._inference_executor.shutdown(wait=False)
    gate._encode_executor.shutdown(wait=False)


def test_worker_construction_reservation_prevents_training_entry(platform, compute_exclusion_gate):
    manager, _, _ = platform
    entered, release = threading.Event(), threading.Event()
    def factory(*args):
        entered.set()
        assert release.wait(2)
        return FakeWorker(*args)
    manager._worker_factory = factory
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(manager.start, "task-1", "bundle-1")
        try:
            assert entered.wait(1)
            assert compute_exclusion_gate.enter_training_maintenance() is False
        finally:
            release.set()
        assert future.result(timeout=2)["state"] in {"running", "waiting_source"}
    manager.stop("task-1")
    assert compute_exclusion_gate.enter_training_maintenance() is True


def test_failed_worker_cleanup_retains_training_exclusion_until_retry(platform, compute_exclusion_gate):
    manager, _, _ = platform
    manager.start("task-1", "bundle-1")
    worker = manager._executions["task-1"].worker
    worker.cleanup_ok = False
    with pytest.raises(TaskExecutionError) as error:
        manager.stop("task-1")
    assert error.value.code == "CLEANUP_PENDING"
    assert compute_exclusion_gate.enter_training_maintenance() is False
    worker.cleanup_ok = True
    assert manager.stop("task-1")["state"] == "stopped"
    assert compute_exclusion_gate.enter_training_maintenance() is True


def test_worker_constructor_failure_releases_unloaded_compute_reservation(platform, compute_exclusion_gate):
    manager, _, _ = platform
    manager._worker_factory = Mock(side_effect=RuntimeError("synthetic constructor failure"))
    with pytest.raises(RuntimeError, match="synthetic constructor failure"):
        manager.start("task-1", "bundle-1")
    assert compute_exclusion_gate.enter_training_maintenance() is True


def test_managed_tasks_share_source_and_independent_results_with_no_alerts(platform):
    manager, _, camera = platform
    manager.start("task-1", "bundle-1")
    manager.start("task-2", "bundle-1")
    assert len(camera.tasks) == 2
    camera.source.hub.publish(synthetic_frame())
    wait_for(lambda: len(manager.results("task-1")) == len(manager.results("task-2")) == 1)
    assert manager.status("task-1")["applied_task_revision"] == 1
    assert manager.status("task-1")["production_allowed"] is False
    assert manager.status("task-1")["alerts_emitted"] is False
    assert manager.stop("task-1")["state"] == "stopped"
    camera.source.hub.publish(synthetic_frame(2, 10.02))
    wait_for(lambda: len(manager.results("task-2")) == 2)
    assert manager.results("task-1") == []


def test_concurrent_start_cannot_over_admit_workers(platform):
    manager, _, _ = platform
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(manager.start, f"task-{number}", "bundle-1") for number in range(1, 4)]
        accepted = rejected = 0
        for future in futures:
            try:
                future.result(timeout=2)
                accepted += 1
            except TaskExecutionError as error:
                assert error.code == "RESOURCE_EXHAUSTED"
                rejected += 1
    assert (accepted, rejected) == (2, 1)


def test_memory_reservation_and_failed_unregistration_keep_admission_lease(platform):
    manager, _, camera = platform
    manager._memory_budget_mb = 2048
    manager.start("task-1", "bundle-1")
    camera.allow_unregister = False
    with pytest.raises(TaskExecutionError) as failure:
        manager.stop("task-1")
    assert failure.value.code == "CLEANUP_PENDING"
    with pytest.raises(TaskExecutionError) as failure:
        manager.start("task-2", "bundle-1")
    assert failure.value.code == "RESOURCE_EXHAUSTED"
    camera.allow_unregister = True
    manager.stop("task-1")
    assert manager.start("task-2", "bundle-1")


@pytest.mark.parametrize("mutation", ["revision", "archive", "source"])
def test_late_result_cannot_escape_changed_authority_or_source(platform, mutation):
    manager, service, camera = platform
    manager.start("task-1", "bundle-1")
    execution = manager._executions["task-1"]
    execution.worker.hang = True
    camera.source.hub.publish(synthetic_frame())
    assert execution.worker.entered.wait(1)
    if mutation == "revision":
        service.tasks["task-1"]["current_revision"] = 2
    elif mutation == "archive":
        service.tasks["task-1"]["archived"] = True
    else:
        camera.source.reconnect()
    execution.worker.release.set()
    wait_for(lambda: execution.runtime.status.state == "failed" if mutation != "source"
             else execution.runtime.status.discarded_results == 1)
    assert list(execution.results) == []


def test_revision_change_in_final_commit_gap_is_fenced(platform):
    manager, service, camera = platform
    manager.start("task-1", "bundle-1")
    execution = manager._executions["task-1"]
    service.before_commit = lambda: service.tasks["task-1"].update(current_revision=2)
    camera.source.hub.publish(synthetic_frame())
    wait_for(lambda: execution.runtime.status.discarded_results == 1)
    assert list(execution.results) == []


def test_reconnect_resets_state_and_hides_previous_source_results(platform):
    manager, _, camera = platform
    manager.start("task-1", "bundle-1")
    worker = manager._executions["task-1"].worker
    camera.source.hub.publish(synthetic_frame())
    wait_for(lambda: len(manager.results("task-1")) == 1)
    camera.source.reconnect()
    assert manager.results("task-1") == []
    camera.source.hub.publish(synthetic_frame(2, 10.02, epoch=2))
    wait_for(lambda: worker.resets == ["task_started", "source_changed"])
    wait_for(lambda: len(manager.results("task-1")) == 1)
    assert manager.results("task-1")[0]["source"]["source_epoch"] == 2


def test_ordered_sampling_overflow_discards_inflight_result(platform):
    manager, _, camera = platform
    manager.start("task-1", "bundle-1")
    execution = manager._executions["task-1"]
    execution.worker.hang = True
    camera.source.hub.publish(synthetic_frame())
    assert execution.worker.entered.wait(1)
    for number in range(2, 70):
        camera.source.hub.publish(synthetic_frame(number, 10 + number / 60))
    execution.worker.release.set()
    wait_for(lambda: execution.runtime.status.state == "sampling_insufficient")
    assert manager.results("task-1") == []


def test_close_cleans_all_workers_after_one_failure_and_prevents_new_start(platform):
    manager, _, _ = platform
    manager.start("task-1", "bundle-1")
    manager.start("task-2", "bundle-1")
    manager._executions["task-1"].worker.cleanup_ok = False
    assert manager.close() is False
    assert manager._executions["task-1"].lease_held
    assert not manager._executions["task-2"].lease_held
    with pytest.raises(TaskExecutionError) as failure:
        manager.start("task-3", "bundle-1")
    assert failure.value.code == "CLOSED"


def test_registration_failure_does_not_claim_applied_revision(platform, monkeypatch):
    manager, _, camera = platform

    def reject(runtime):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(camera, "register_analysis_task_runtime", reject)
    with pytest.raises(TaskExecutionError):
        manager.start("task-1", "bundle-1")
    assert manager.status("task-1")["applied_task_revision"] is None
    assert not manager._executions["task-1"].lease_held


def test_stop_fences_start_during_bundle_resolution_before_worker_exists(platform):
    manager, service, _ = platform
    entered, release = threading.Event(), threading.Event()
    resolve = service.resolve_bundle

    def slow_resolve(task_id, bundle_id):
        entered.set()
        assert release.wait(2)
        return resolve(task_id, bundle_id)

    service.resolve_bundle = slow_resolve
    with ThreadPoolExecutor(max_workers=2) as executor:
        start = executor.submit(manager.start, "task-1", "bundle-1")
        assert entered.wait(1)
        stop = executor.submit(manager.stop, "task-1")
        wait_for(lambda: manager._pending_starts["task-1"][0].is_set())
        assert "task-1" not in manager._executions
        release.set()
        with pytest.raises(TaskExecutionError) as failure:
            start.result(timeout=1)
        assert failure.value.code == "CANCELLED"
        assert stop.result(timeout=1)["state"] == "stopped"
    assert not manager._pending_starts and "task-1" not in manager._executions
    assert manager.start("task-1", "bundle-1")["applied_task_revision"] == 1


def test_unrelated_stop_does_not_cancel_a_resolving_start(platform):
    manager, service, _ = platform
    resolve = service.resolve_bundle

    def resolve_after_peer_stop(task_id, bundle_id):
        assert manager.stop("task-2")["state"] == "stopped"
        return resolve(task_id, bundle_id)

    service.resolve_bundle = resolve_after_peer_stop
    assert manager.start("task-1", "bundle-1")["applied_task_revision"] == 1


def test_old_stop_cannot_clean_a_later_start_that_wins_control_ownership(platform):
    manager, _, _ = platform
    entered, release = threading.Event(), threading.Event()
    lock = threading.RLock()
    main_thread = threading.current_thread()

    class DelayedStopOwnership:
        def __enter__(self):
            if threading.current_thread() is not main_thread:
                entered.set()
                assert release.wait(2)
            lock.acquire()

        def __exit__(self, *args):
            lock.release()

    manager._control = lambda task_id: DelayedStopOwnership()
    with ThreadPoolExecutor(max_workers=1) as executor:
        stop = executor.submit(manager.stop, "task-1")
        assert entered.wait(1)
        assert manager.start("task-1", "bundle-1")["applied_task_revision"] == 1
        worker = manager._executions["task-1"].worker
        release.set()
        assert stop.result(timeout=1)["state"] == "waiting_source"
    assert worker.running and not manager._executions["task-1"].cancelled.is_set()


def test_mutation_requires_confirmed_cleanup_and_retained_lease_blocks_it(platform):
    manager, _, _ = platform
    changed = Mock(return_value="changed")
    manager.start("task-1", "bundle-1")
    with pytest.raises(TaskExecutionError) as failure:
        manager.mutate_if_stopped("task-1", changed)
    assert failure.value.code == "BUSY"
    worker = manager._executions["task-1"].worker
    worker.cleanup_ok = False
    with pytest.raises(TaskExecutionError):
        manager.stop("task-1")
    with pytest.raises(TaskExecutionError, match="confirm cleanup"):
        manager.mutate_if_stopped("task-1", changed)
    changed.assert_not_called()
    worker.cleanup_ok = True
    manager.stop("task-1")
    assert manager.mutate_if_stopped("task-1", changed) == "changed"
    changed.assert_called_once_with()


def test_configuration_mutation_and_start_share_the_same_control_fence(platform):
    manager, service, _ = platform
    entered, release = threading.Event(), threading.Event()

    def mutate():
        entered.set()
        assert release.wait(2)
        service.tasks["task-1"]["current_revision"] = 2
        return "changed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        mutation = executor.submit(manager.mutate_if_stopped, "task-1", mutate)
        assert entered.wait(1)
        start = executor.submit(manager.start, "task-1", "bundle-1")
        wait_for(lambda: bool(manager._pending_starts.get("task-1")))
        assert "task-1" not in manager._executions
        release.set()
        assert mutation.result(timeout=1) == "changed"
        with pytest.raises(TaskExecutionError) as failure:
            start.result(timeout=1)
        assert failure.value.code == "STALE_TASK"
    assert not manager._executions["task-1"].worker.running
    assert not manager._executions["task-1"].lease_held


def test_peer_load_does_not_block_start_stop_or_resource_release(platform):
    manager, _, _ = platform
    entered, release = threading.Event(), threading.Event()

    class SlowLoadWorker(FakeWorker):
        def start(self):
            super().start()
            if self.manifest["task_id"] == "task-1":
                entered.set()
                assert release.wait(3)

        def stop(self):
            clean = super().stop()
            if clean and self.manifest["task_id"] == "task-1":
                release.set()
            return clean

    manager._worker_factory = SlowLoadWorker
    with ThreadPoolExecutor(max_workers=2) as executor:
        loading = executor.submit(manager.start, "task-1", "bundle-1")
        try:
            assert entered.wait(1)
            peer_start = executor.submit(manager.start, "task-2", "bundle-1")
            assert peer_start.result(timeout=1)["applied_task_revision"] == 1
            peer_stop = executor.submit(manager.stop, "task-2")
            assert peer_stop.result(timeout=1)["state"] == "stopped"
            assert not release.is_set() and not loading.done()
            assert manager._executions["task-1"].lease_held
            assert not manager._executions["task-2"].lease_held
            # The cleaned peer's reservation is usable while A still loads.
            assert manager.start("task-3", "bundle-1")["applied_task_revision"] == 1
        finally:
            release.set()
        assert loading.result(timeout=1)["applied_task_revision"] == 1
    assert not manager._task_controls


def test_parallel_loading_reservations_cannot_over_admit_memory(platform):
    manager, _, _ = platform
    manager._max_workers = 3
    manager._memory_budget_mb = 2048
    entered, release = threading.Event(), threading.Event()

    class SlowLoadWorker(FakeWorker):
        def start(self):
            super().start()
            entered.set()
            assert release.wait(3)

        def stop(self):
            clean = super().stop()
            if clean:
                release.set()
            return clean

    manager._worker_factory = SlowLoadWorker
    with ThreadPoolExecutor(max_workers=3) as executor:
        loading = executor.submit(manager.start, "task-1", "bundle-1")
        try:
            assert entered.wait(1)
            peers = [executor.submit(manager.start, task_id, "bundle-1")
                     for task_id in ("task-2", "task-3")]
            for peer in peers:
                with pytest.raises(TaskExecutionError) as failure:
                    peer.result(timeout=1)
                assert failure.value.code == "RESOURCE_EXHAUSTED"
            assert sum(item.lease_held for item in manager._executions.values()) == 1
            assert not release.is_set()
        finally:
            release.set()
        assert loading.result(timeout=1)["applied_task_revision"] == 1
    assert not manager._task_controls


def test_close_cancels_all_parallel_loads_and_queued_starts_before_waiting(platform):
    manager, _, camera = platform
    entered = {task_id: threading.Event() for task_id in ("task-1", "task-2")}
    release = {task_id: threading.Event() for task_id in entered}
    workers = []

    class SlowLoadWorker(FakeWorker):
        def __init__(self, manifest, models_root):
            super().__init__(manifest, models_root)
            workers.append(self)

        def start(self):
            super().start()
            task_id = self.manifest["task_id"]
            entered[task_id].set()
            assert release[task_id].wait(3)

        def stop(self):
            clean = super().stop()
            if clean:
                release[self.manifest["task_id"]].set()
            return clean

    manager._worker_factory = SlowLoadWorker
    with ThreadPoolExecutor(max_workers=5) as executor:
        starts = [executor.submit(manager.start, task_id, "bundle-1") for task_id in entered]
        try:
            assert all(event.wait(1) for event in entered.values())
            # Queued duplicate commands have not allocated another worker.
            starts.extend(executor.submit(manager.start, task_id, "bundle-1") for task_id in entered)
            wait_for(lambda: all(len(manager._pending_starts.get(task_id, [])) == 2 for task_id in entered))
            closing = executor.submit(manager.close)
            assert closing.result(timeout=1)
            for start in starts:
                with pytest.raises(TaskExecutionError) as failure:
                    start.result(timeout=1)
                assert failure.value.code in {"CLOSED", "CANCELLED"}
        finally:
            for event in release.values():
                event.set()
    assert len(workers) == 2 and all(not worker.running for worker in workers)
    assert all(not item.lease_held for item in manager._executions.values())
    assert not manager._pending_starts and not manager._task_controls and not camera.tasks
    with pytest.raises(TaskExecutionError) as failure:
        manager.start("task-3", "bundle-1")
    assert failure.value.code == "CLOSED"


def require_scene(service, task_id="task-1"):
    manifest = service.manifests[task_id]
    manifest["config"]["scene"] = {"policy": "require_profile", "profile_id": "approved-scene"}
    def canonical(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest["task_config_sha256"] = hashlib.sha256(canonical(manifest["config"]).encode()).hexdigest()
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest).encode()).hexdigest()
    service.tasks[task_id]["config_sha256"] = manifest["task_config_sha256"]


def test_require_profile_missing_current_evidence_fails_before_loading_worker(platform):
    manager, service, camera = platform
    require_scene(service)
    with pytest.raises(TaskExecutionError) as failure:
        manager.start("task-1", "bundle-1")
    assert failure.value.code == "SCENE_BLOCKED"
    assert not manager._executions["task-1"].worker.running
    camera.scene["usable"] = True
    assert manager.start("task-1", "bundle-1")


def test_status_exposes_transient_scene_reason_without_poisoning_worker(platform):
    manager, service, camera = platform
    require_scene(service)
    camera.scene["usable"] = True
    manager.start("task-1", "bundle-1")
    camera.scene["usable"] = False
    blocked = manager.status("task-1")
    assert blocked["state"] == "blocked"
    assert blocked.get("reason_code") == "SCENE_BLOCKED"
    assert blocked["error_code"] is None
    assert blocked["production_allowed"] is False
    camera.scene["usable"] = True
    assert manager.status("task-1").get("reason_code") is None


def test_require_profile_consumer_pause_discards_late_result_and_waits_for_fresh_evidence(platform):
    manager, service, camera = platform
    require_scene(service)
    camera.scene["usable"] = True
    manager.start("task-1", "bundle-1")
    execution = manager._executions["task-1"]
    execution.worker.hang = True
    camera.source.hub.publish(synthetic_frame())
    assert execution.worker.entered.wait(1)
    camera.scene["consumer_enabled"] = False
    execution.worker.release.set()
    wait_for(lambda: execution.runtime.status.state == "blocked")
    assert list(execution.results) == [] and not execution.runtime.status.cleanup_pending
    camera.scene["consumer_enabled"] = True
    camera.source.hub.publish(synthetic_frame(2, 10.02))
    wait_for(lambda: len(manager.results("task-1")) == 1)
    assert execution.worker.resets == ["task_started", "task_started"]
    assert manager.status("task-1")["error_code"] is None


def test_require_profile_reconnect_cannot_reuse_previous_epoch_guard_cache(platform):
    manager, service, camera = platform
    require_scene(service)
    camera.scene["usable"] = True
    manager.start("task-1", "bundle-1")
    execution = manager._executions["task-1"]
    camera.source.reconnect()
    camera.source.hub.publish(synthetic_frame(epoch=2))
    wait_for(lambda: execution.runtime.status.state == "blocked")
    assert not execution.worker.entered.is_set()
    camera.scene["source_token"] = ("session", 2)
    camera.source.hub.publish(synthetic_frame(2, 10.02, epoch=2))
    wait_for(lambda: len(manager.results("task-1")) == 1)


def test_final_scene_check_and_database_fence_have_safe_lock_order(platform):
    manager, service, camera = platform
    require_scene(service)
    camera.scene["usable"] = True
    manager.start("task-1", "bundle-1")
    execution = manager._executions["task-1"]

    def database_fence():
        assert not camera.source.lock._is_owned()
        assert not execution.runtime._lock._is_owned()
        camera.scene["usable"] = False

    service.before_commit = database_fence
    camera.source.hub.publish(synthetic_frame())
    wait_for(lambda: execution.runtime.status.state == "blocked")
    assert list(execution.results) == []


def scene_pipeline():
    from argus.config.schema import SceneGuardConfig
    from argus.core.pipeline import DetectionPipeline
    from argus.preprocessing.scene_profile import RUNTIME_ARTIFACT_HASH_SCHEME

    pipeline = DetectionPipeline.__new__(DetectionPipeline)
    pipeline.camera_config = SimpleNamespace(camera_id="cam", scene_guard=SceneGuardConfig())
    pipeline._scene_guard_lock = threading.RLock()
    pipeline._scene_processing_lock = threading.RLock()
    pipeline._scene_guard_last_result = None
    pipeline._scene_guard_last_check_frame = 0
    pipeline._scene_guard_last_check_monotonic = 0.0
    pipeline._scene_guard_last_usable = False
    pipeline._scene_guard_policy_recovery_required = False
    pipeline._scene_guard_policy_recovery_count = 0
    pipeline._scene_guard_last_reconnect_count = 0
    pipeline._scene_guard_last_stream_key = None
    pipeline._scene_guard_runtime_error = None
    pipeline._scene_guard_reference_frame = synthetic_frame().frame
    pipeline._scene_guard_reference_sha256 = "a" * 64
    pipeline._scene_guard_profile_id = "approved-scene"
    pipeline._scene_guard_reference_source = "profile"
    pipeline._model_artifact_sha256 = "b" * 64
    pipeline._model_version_id = "approved-model"
    pipeline._scene_guard_bound_model_version = "approved-model"
    pipeline._scene_guard_bound_model_artifact_sha256 = "b" * 64
    pipeline._managed_scene_reference_metadata = {
        "operator_approved": True, "validation_only": False, "model_version_id": "approved-model",
        "approved_by": "human-operator", "approved_at": "2026-10-23T10:00:00+08:00",
        "model_artifact_sha256": "b" * 64, "model_artifact_hash_scheme": RUNTIME_ARTIFACT_HASH_SCHEME,
    }
    pipeline._frame_counter = 1
    pipeline._camera = SimpleNamespace(state=SimpleNamespace(reconnect_count=0))
    pipeline._managed_scene_evidence = None
    pipeline._scene_guard = Mock()
    state = SimpleNamespace(value="stable")
    pipeline._scene_guard.evaluate.return_value = SimpleNamespace(
        state=state, raw_state=state, usable=True, reason="mock geometry passed",
        metrics=SimpleNamespace(to_dict=lambda: {}), homography=None,
    )
    pipeline._reset_scene_sensitive_state = Mock()
    pipeline._publish_scene_guard_availability = Mock()
    return pipeline


def test_pipeline_records_source_token_only_after_actual_mock_guard_processing():
    from argus.core.diagnostics import FrameDiagnostics

    pipeline = scene_pipeline()
    pipeline.get_latest_frame_identity = Mock(side_effect=AssertionError("preview identity is not guard evidence"))
    assert not pipeline.get_managed_scene_guard_observation()["usable"]
    frame = synthetic_frame()
    _, usable = pipeline._apply_scene_guard(frame.frame, frame, FrameDiagnostics(1, 10, "cam"))
    assert usable and pipeline._scene_guard.evaluate.call_count == 1
    observation = pipeline.get_managed_scene_guard_observation()
    assert observation["usable"] and observation["source_token"] == ("session", 1)
    frame.identity.source_session_id = None
    pipeline._scene_guard_last_result = None
    pipeline._scene_guard_last_stream_key = None
    pipeline._frame_counter += 1
    pipeline._apply_scene_guard(frame.frame, frame, FrameDiagnostics(2, 11, "cam"))
    assert pipeline._scene_guard_last_usable is True  # Legacy decision is unchanged.
    assert not pipeline.get_managed_scene_guard_observation()["usable"]


def test_managed_observation_expires_and_does_not_change_legacy_usable_decision():
    from argus.core.diagnostics import FrameDiagnostics

    pipeline = scene_pipeline()
    frame = synthetic_frame()
    pipeline._apply_scene_guard(frame.frame, frame, FrameDiagnostics(1, 10, "cam"))
    pipeline._managed_scene_evidence["observed_monotonic"] -= pipeline.camera_config.scene_guard.check_interval_seconds + 1
    assert pipeline._scene_guard_last_usable is True
    assert pipeline.get_managed_scene_guard_observation()["reason_code"] == "SCENE_EVIDENCE_EXPIRED"


@pytest.mark.parametrize("state,raw", [("alignable", "alignable"), ("stable", "alignable")])
def test_managed_raw_frame_tasks_reject_alignment_only_geometry_without_changing_legacy(state, raw):
    from argus.core.diagnostics import FrameDiagnostics

    pipeline = scene_pipeline()
    pipeline.camera_config.scene_guard.allow_alignable = True
    frame = synthetic_frame()
    pipeline._apply_scene_guard(frame.frame, frame, FrameDiagnostics(1, 10, "cam"))
    pipeline._scene_guard_last_result.state = SimpleNamespace(value=state)
    pipeline._scene_guard_last_result.raw_state = SimpleNamespace(value=raw)
    assert pipeline._scene_guard_last_usable is True
    assert pipeline.get_managed_scene_guard_observation()["usable"] is False


def test_managed_observation_never_waits_for_geometry_lock():
    pipeline = scene_pipeline()
    entered, release = threading.Event(), threading.Event()

    def geometry():
        with pipeline._scene_guard_lock:
            entered.set()
            release.wait(2)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(geometry)
        assert entered.wait(1)
        observation = pipeline.get_managed_scene_guard_observation()
        assert observation["usable"] is False and observation["reason_code"] == "SCENE_EVIDENCE_BUSY"
        release.set()
        future.result(timeout=1)


def test_manager_rejects_paused_consumer_and_prior_epoch_without_touching_legacy_decision():
    from argus.capture.manager import CameraManager

    pipeline = scene_pipeline()
    pipeline.get_managed_scene_guard_observation = Mock(return_value={
        "enabled": True, "usable": True, "reference_available": True,
        "profile_id": "approved-scene", "source_token": ("session", 1),
    })
    manager = CameraManager.__new__(CameraManager)
    source = SyntheticSource()
    manager._lock, manager._sources = threading.RLock(), {"cam": source}
    manager._pipelines, manager._threads = {"cam": pipeline}, {"cam": SimpleNamespace(is_alive=lambda: True)}
    manager._legacy_task_enabled = {"cam": True}
    assert manager.get_managed_scene_guard_status("cam", source=source)["usable"]
    appended = []
    assert manager.commit_managed_scene_if_current(
        "cam", source=source, profile_id="approved-scene", callback=lambda: appended.append(1),
    )
    manager._legacy_task_enabled["cam"] = False
    assert not manager.get_managed_scene_guard_status("cam", source=source)["usable"]
    assert not manager.commit_managed_scene_if_current(
        "cam", source=source, profile_id="approved-scene", callback=lambda: appended.append(2),
    )
    assert appended == [1]
    manager._legacy_task_enabled["cam"] = True
    source.reconnect()
    assert not manager.get_managed_scene_guard_status("cam", source=source)["usable"]
