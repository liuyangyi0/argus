"""Mock-only cross-module lifecycle/lock-order regression tests; no media I/O."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from argus.capture.manager import CameraManager
from argus.config.schema import SceneGuardConfig
from argus.core.analysis_task_runtime import TaskRuntime, TaskSupervisor
from argus.core.diagnostics import FrameDiagnostics
from argus.core.pipeline import DetectionPipeline
from argus.preprocessing.scene_profile import RUNTIME_ARTIFACT_HASH_SCHEME


def _manager():
    manager = CameraManager.__new__(CameraManager)
    manager._lock = threading.RLock()
    manager._cameras = []
    manager._pipelines = {}
    manager._runners = {}
    manager._threads = {}
    manager._broadcast_camera_and_wall = Mock()
    manager._ensure_lifecycle_state()
    return manager


def test_manager_does_not_hold_its_lock_while_reading_independent_task_snapshot():
    manager = _manager()

    def snapshot():
        # Task snapshot obtains TaskRuntime._lock. Its result callback already
        # holds that lock and may legitimately make a read-only manager query.
        # Taking task._lock under manager._lock creates the reverse lock order.
        assert not manager._lock._is_owned()
        return {"task_id": "aux-1", "state": "stopped"}

    runtime = TaskRuntime("aux-1", SimpleNamespace(camera_id="cam1"), lambda frame: None)
    runtime.snapshot = snapshot
    manager._task_supervisor.register(runtime)
    assert manager.get_analysis_task_runtime("aux-1")["state"] == "stopped"


def test_clean_camera_stop_allows_task_identity_to_bind_replacement_source():
    manager = _manager()
    old_source = SimpleNamespace(camera_id="cam1", stop=Mock(return_value=True))
    manager._sources["cam1"] = old_source
    old_task = TaskRuntime("aux-1", old_source, lambda frame: None)
    manager.register_analysis_task_runtime(old_task)

    manager.stop_camera("cam1")

    new_source = SimpleNamespace(camera_id="cam1", stop=Mock(return_value=True))
    manager._sources["cam1"] = new_source
    new_task = TaskRuntime("aux-1", new_source, lambda frame: None)
    manager.register_analysis_task_runtime(new_task)
    assert manager._task_supervisor.get("aux-1") is new_task
    assert old_task.status.state == "stopped"
    assert old_source.stop.call_count == 1


def test_failed_task_cleanup_keeps_old_identity_reserved():
    supervisor = TaskSupervisor()
    source = SimpleNamespace(camera_id="cam1")
    task = TaskRuntime("aux-1", source, lambda frame: None, cleanup=lambda: False)
    # Simulate acquired worker resources without starting any worker/model.
    task._cleanup_pending = True
    supervisor.register(task)
    assert supervisor.stop_source("cam1", timeout=0) is False
    assert supervisor.get("aux-1") is task
    assert task.status.cleanup_pending is True


def test_supervisor_releases_collection_lock_before_starting_task():
    supervisor = TaskSupervisor()
    task = TaskRuntime("aux-1", SimpleNamespace(camera_id="cam1"), lambda frame: None)
    supervisor.register(task)

    def start():
        # start() can wait for task._lock while a result callback already owns
        # it and reads supervisor.get/status. Collection ownership must end
        # before invoking the lifecycle operation.
        assert not supervisor._lock._is_owned()
        return True

    task.start = start
    assert supervisor.start("aux-1") is True


def _scene_pipeline():
    """Run scene orchestration with a fake geometric decision, never a model."""
    pipeline = DetectionPipeline.__new__(DetectionPipeline)
    pipeline.camera_config = SimpleNamespace(camera_id="cam1", scene_guard=SceneGuardConfig())
    pipeline._scene_guard_lock = threading.RLock()
    pipeline._scene_guard_last_result = None
    pipeline._scene_guard_last_check_frame = 0
    pipeline._scene_guard_last_check_monotonic = 0.0
    pipeline._scene_guard_last_usable = False
    pipeline._scene_guard_last_reconnect_count = 0
    pipeline._scene_guard_last_stream_key = None
    pipeline._scene_guard_policy_recovery_required = False
    pipeline._scene_guard_policy_recovery_count = 0
    pipeline._scene_guard_reference_frame = np.zeros((16, 16, 3), dtype=np.uint8)
    pipeline._scene_guard_reference_sha256 = "a" * 64
    pipeline._scene_guard_reference_source = "profile:approved-scene"
    pipeline._scene_guard_reference_error = None
    pipeline._scene_guard_runtime_error = None
    pipeline._scene_guard_profile_id = "approved-scene"
    pipeline._model_version_id = "approved-model"
    pipeline._model_artifact_sha256 = "b" * 64
    pipeline._scene_guard_bound_model_version = pipeline._model_version_id
    pipeline._scene_guard_bound_model_artifact_sha256 = pipeline._model_artifact_sha256
    pipeline._managed_scene_reference_metadata = {
        "operator_approved": True, "validation_only": False,
        "approved_by": "human-operator", "approved_at": "2026-10-23T10:00:00+08:00",
        "model_version_id": pipeline._model_version_id,
        "model_artifact_sha256": pipeline._model_artifact_sha256,
        "model_artifact_hash_scheme": RUNTIME_ARTIFACT_HASH_SCHEME,
    }
    pipeline._managed_scene_evidence = None
    pipeline._frame_counter = 1
    pipeline._camera = SimpleNamespace(state=SimpleNamespace(reconnect_count=0))
    pipeline._scene_guard = Mock()
    stable = SimpleNamespace(value="stable")
    pipeline._scene_guard.evaluate.return_value = SimpleNamespace(
        state=stable, raw_state=stable, usable=True, reason="mock geometric match",
        metrics=SimpleNamespace(to_dict=lambda: {}), homography=None,
    )
    pipeline._reset_scene_sensitive_state = Mock()
    pipeline._publish_scene_guard_availability = Mock()
    return pipeline


@pytest.mark.parametrize("metadata", [
    {}, {"operator_approved": False}, {"operator_approved": 1},
    {"validation_only": True}, {"validation_only": 0},
    {"approved_by": ""}, {"approved_by": "unknown"}, {"approved_by": "fake\nidentity"},
    {"approved_at": "2026-10-23T10:00:00"}, {"approved_at": "invalid"},
    {"model_version_id": "another-model"}, {"model_artifact_sha256": "c" * 64},
    {"model_artifact_hash_scheme": "unknown"},
])
def test_managed_scene_never_inherits_permissive_legacy_reference_authority(metadata):
    pipeline = _scene_pipeline()
    if metadata:
        pipeline._managed_scene_reference_metadata.update(metadata)
    else:
        pipeline._managed_scene_reference_metadata = {}
    # A validation/offline legacy policy is allowed to remain permissive; only
    # the new managed require_profile evidence must be tightened.
    pipeline.camera_config.scene_guard.require_operator_approved_profile = False
    pipeline.camera_config.scene_guard.require_profile_model_binding = False
    frame = SimpleNamespace(
        frame=pipeline._scene_guard_reference_frame, camera_id="cam1",
        identity=SimpleNamespace(source_session_id="session", source_epoch=1),
    )
    _, legacy_usable = pipeline._apply_scene_guard(frame.frame, frame, FrameDiagnostics(1, 10, "cam1"))
    assert legacy_usable and pipeline._scene_guard_last_usable
    assert pipeline.get_managed_scene_guard_observation()["reason_code"] == "SCENE_PROFILE_UNAPPROVED"
    committed = []
    assert not pipeline.commit_managed_scene_guard_observation(
        ("session", 1), "approved-scene", lambda: committed.append("unsafe"),
    )
    assert committed == [] and pipeline._scene_guard_last_usable


def test_managed_scene_model_version_change_invalidates_same_artifact_evidence():
    pipeline = _scene_pipeline()
    frame = SimpleNamespace(
        frame=pipeline._scene_guard_reference_frame, camera_id="cam1",
        identity=SimpleNamespace(source_session_id="session", source_epoch=1),
    )
    pipeline._apply_scene_guard(frame.frame, frame, FrameDiagnostics(1, 10, "cam1"))
    assert pipeline.get_managed_scene_guard_observation()["usable"]
    pipeline._model_version_id = "another-model-with-same-bytes"
    assert not pipeline.get_managed_scene_guard_observation()["usable"]
    assert pipeline._scene_guard_last_usable


@pytest.mark.parametrize("invalid", [
    {"observed_monotonic": None}, {"observed_monotonic": float("nan")},
    {"observed_monotonic": True}, {"source_token": None},
    {"source_token": ("session", True)}, {"source_token": ("session", 0)},
    {"source_token": ("", 1)}, {"reference_sha256": "changed"},
])
def test_malformed_managed_scene_evidence_is_blocked_without_touching_legacy(invalid):
    pipeline = _scene_pipeline()
    frame = SimpleNamespace(
        frame=pipeline._scene_guard_reference_frame, camera_id="cam1",
        identity=SimpleNamespace(source_session_id="session", source_epoch=1),
    )
    pipeline._apply_scene_guard(frame.frame, frame, FrameDiagnostics(1, 10, "cam1"))
    assert pipeline.get_managed_scene_guard_observation()["usable"]
    pipeline._managed_scene_evidence.update(invalid)
    assert not pipeline.get_managed_scene_guard_observation()["usable"]
    assert pipeline._scene_guard_last_usable


@pytest.mark.parametrize("marker", ["_stopping_cameras", "_retiring_cameras", "_retiring_sources"])
def test_stop_intent_blocks_new_task_registration_and_scene_commit_before_resources_detach(marker):
    manager = _manager()
    source = SimpleNamespace(camera_id="cam1", source_token=("session", 1))
    manager._sources["cam1"] = source
    pipeline = _scene_pipeline()
    pipeline.get_managed_scene_guard_observation = Mock(return_value={
        "usable": True, "source_token": source.source_token, "profile_id": "approved-scene",
    })
    pipeline.commit_managed_scene_guard_observation = Mock(return_value=True)
    manager._pipelines["cam1"] = pipeline
    manager._threads["cam1"] = SimpleNamespace(is_alive=lambda: True)
    if marker == "_stopping_cameras":
        manager._stopping_cameras.add("cam1")
    else:
        getattr(manager, marker)["cam1"] = object()
    runtime = TaskRuntime("managed", source, lambda _frame: None)
    with pytest.raises(ValueError, match="current source"):
        manager.register_analysis_task_runtime(runtime)
    assert not manager.get_managed_scene_guard_status("cam1", source=source)["usable"]
    assert not manager.commit_managed_scene_if_current(
        "cam1", source=source, profile_id="approved-scene", callback=lambda: pytest.fail("stop won"),
    )
    pipeline.get_managed_scene_guard_observation.assert_not_called()
    pipeline.commit_managed_scene_guard_observation.assert_not_called()


def test_scene_observation_rechecks_consumer_thread_after_reading_guard_evidence():
    manager = _manager()
    source = SimpleNamespace(camera_id="cam1", source_token=("session", 1))
    manager._sources["cam1"] = source
    thread = SimpleNamespace(is_alive=lambda: True)
    manager._threads["cam1"] = thread

    def observe():
        manager._threads["cam1"] = SimpleNamespace(is_alive=lambda: False)
        return {"usable": True, "profile_id": "approved-scene", "source_token": source.source_token}

    manager._pipelines["cam1"] = SimpleNamespace(get_managed_scene_guard_observation=observe)
    assert not manager.get_managed_scene_guard_status("cam1", source=source)["usable"]
