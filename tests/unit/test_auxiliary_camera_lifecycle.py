"""Camera ownership races with fake pipelines/runners only; no model/media I/O."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from argus.capture.manager import CameraManager
from argus.capture.camera import CameraState
from argus.config.schema import AlertConfig, CameraConfig, ClassifierConfig, SegmenterConfig


class _FakePipeline:
    def __init__(self, control, **kwargs):
        self.control = control
        self.camera_config = kwargs["camera_config"]
        self.on_alert = kwargs["on_alert"]
        self.classifier_initially_enabled = bool(kwargs["classifier_config"].enabled)
        self.desired = None
        self.reconciliations = []
        self.shutdown_calls = 0
        self.shutdown_failures = 0
        self.stats = SimpleNamespace(frames_captured=0)
        self._camera = SimpleNamespace(state=SimpleNamespace(connected=True))

    def initialize(self):
        return self.control.initialize(self)

    def reconcile_auxiliary_desired(self, desired):
        self.reconciliations.append(dict(desired))
        self.control.reconcile(self, desired)
        self.desired = dict(desired)
        return True

    def shutdown(self):
        self.shutdown_calls += 1
        self.control.shutdown(self)
        if self.shutdown_failures:
            self.shutdown_failures -= 1
            raise RuntimeError("fake cleanup failure")


class _FakeRunner:
    def __init__(self, pipeline, **kwargs):
        self.pipeline = pipeline

    def initialize(self):
        return self.pipeline.initialize()


@pytest.fixture
def fake_manager(monkeypatch):
    control = SimpleNamespace(
        initialize=lambda pipeline: True,
        reconcile=lambda pipeline, desired: None,
        shutdown=lambda pipeline: None,
    )
    instances = []

    def pipeline_factory(**kwargs):
        pipeline = _FakePipeline(control, **kwargs)
        instances.append(pipeline)
        return pipeline

    monkeypatch.setattr("argus.capture.manager.DetectionPipeline", pipeline_factory)
    monkeypatch.setattr("argus.capture.manager.CameraInferenceRunner", _FakeRunner)
    manager = CameraManager(
        cameras=[], alert_config=AlertConfig(),
        classifier_config=ClassifierConfig(enabled=True),
        segmenter_config=SegmenterConfig(enabled=False),
    )
    manager._cameras = [CameraConfig(camera_id="camera-a", name="Fake", source="offline")]
    monkeypatch.setattr(manager, "_camera_loop", lambda *args: None)
    monkeypatch.setattr(manager, "_broadcast_camera_and_wall", MagicMock())
    monkeypatch.setattr(manager, "_resolve_model_path", lambda config: None)
    monkeypatch.setattr(manager, "_model_version_id", lambda *args: "fake-version")
    monkeypatch.setattr(manager, "_create_shadow_runner", lambda config: None)
    monkeypatch.setattr(manager, "_get_legacy_shared_yolo", lambda config: None)

    class SyntheticCapture:
        def __init__(self, config):
            self.camera_id, self.source, self.protocol = config.camera_id, config.source, config.protocol
            self.state = CameraState()
            self.stopped = threading.Event()

        def connect(self):
            self.state.connected = True
            return True

        def read(self):
            self.stopped.wait(0.01)
            return None

        def request_stop(self):
            self.stopped.set()

        def stop(self):
            self.stopped.set()
            self.state.connected = False

    monkeypatch.setattr(manager, "_create_source_capture", lambda config, *_args: SyntheticCapture(config))
    yield manager, control, instances
    control.shutdown = lambda pipeline: None
    for pipeline in instances:
        pipeline.shutdown_failures = 0
    manager.stop_all()


def _start_in_background(manager):
    results = []
    failures = []

    def start():
        try:
            results.append(manager.start_camera("camera-a"))
        except Exception as exc:
            failures.append(exc)

    caller = threading.Thread(target=start)
    caller.start()
    return caller, results, failures


def test_stop_invalidates_unpublished_start_and_cleans_candidate(fake_manager):
    manager, control, instances = fake_manager
    entered, release = threading.Event(), threading.Event()

    def initialize(pipeline):
        entered.set()
        assert release.wait(2)
        return True

    control.initialize = initialize
    caller, results, failures = _start_in_background(manager)
    try:
        assert entered.wait(1)
        manager.stop_camera("camera-a")
        assert manager.start_camera("camera-a") is False
    finally:
        release.set()
        caller.join(2)
    assert not caller.is_alive()
    assert not failures
    assert results == [False]
    assert instances[0].shutdown_calls == 1
    assert "camera-a" not in manager._pipelines
    assert "camera-a" not in manager._starting_cameras


def test_module_disable_includes_pending_start_and_reconciles_before_publish(fake_manager):
    manager, control, instances = fake_manager
    entered, release = threading.Event(), threading.Event()

    def initialize(pipeline):
        entered.set()
        assert release.wait(2)
        return True

    control.initialize = initialize
    caller, results, failures = _start_in_background(manager)
    try:
        assert entered.wait(1)
        outcome = manager.reload_module_for_cameras(["camera-a"], "classifier.enabled", False)
        assert outcome == {"seen": 1, "succeeded": 0, "failed": 1, "pending": 1}
    finally:
        release.set()
        caller.join(2)
    assert not failures
    assert results == [True]
    assert instances[0].classifier_initially_enabled is True
    assert instances[0].desired["classifier.enabled"] is False
    assert manager._classifier_config.enabled is False


def test_module_change_during_reconcile_retries_revision_before_publish(fake_manager):
    manager, control, instances = fake_manager
    entered, release = threading.Event(), threading.Event()

    def reconcile(pipeline, desired):
        if len(pipeline.reconciliations) == 1:
            entered.set()
            assert release.wait(2)

    control.reconcile = reconcile
    caller, results, failures = _start_in_background(manager)
    try:
        assert entered.wait(1)
        manager.reload_module_for_cameras(["camera-a"], "classifier.enabled", False)
    finally:
        release.set()
        caller.join(2)
    assert not failures
    assert results == [True]
    assert len(instances[0].reconciliations) == 2
    assert instances[0].desired["classifier.enabled"] is False


def test_failed_initialize_cleans_unregistered_candidate(fake_manager):
    manager, control, instances = fake_manager
    control.initialize = lambda pipeline: False
    assert manager.start_camera("camera-a") is False
    assert instances[0].shutdown_calls == 1
    assert not manager._retiring_cameras
    assert not manager._pipelines


def test_successful_candidate_cleanup_clears_previous_error(fake_manager):
    manager, control, instances = fake_manager
    manager._camera_lifecycle_errors["camera-a"] = "previous cleanup failure"
    control.initialize = lambda pipeline: False

    assert manager.start_camera("camera-a") is False

    assert instances[0].shutdown_calls == 1
    assert "camera-a" not in manager._retiring_cameras
    assert "camera-a" not in manager._camera_lifecycle_errors


def test_failed_initialize_cleanup_failure_remains_retryable(fake_manager):
    manager, control, instances = fake_manager

    def initialize(pipeline):
        pipeline.shutdown_failures = 1
        return False

    control.initialize = initialize
    assert manager.start_camera("camera-a") is False
    assert manager._retiring_cameras["camera-a"][0] is instances[0]
    assert manager.start_camera("camera-a") is False
    manager.stop_camera("camera-a")
    assert not manager._retiring_cameras


def test_unpublished_candidate_callback_cannot_emit(fake_manager):
    manager, control, _ = fake_manager
    alerts = []
    manager._on_alert = alerts.append

    def initialize(pipeline):
        pipeline.on_alert(SimpleNamespace(camera_id="camera-a"))
        return True

    control.initialize = initialize
    assert manager.start_camera("camera-a") is True
    assert alerts == []
    assert manager.alert_count == 0


def test_initialize_exception_also_cleans_candidate(fake_manager):
    manager, control, instances = fake_manager

    def initialize(pipeline):
        raise RuntimeError("fake initialization failure")

    control.initialize = initialize
    with pytest.raises(RuntimeError, match="fake initialization failure"):
        manager.start_camera("camera-a")
    assert instances[0].shutdown_calls == 1
    assert not manager._starting_cameras


def test_cleanup_failure_retains_owner_and_stop_can_retry(fake_manager):
    manager, _, instances = fake_manager
    assert manager.start_camera("camera-a") is True
    first = instances[0]
    first.shutdown_failures = 1
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        manager.stop_camera("camera-a")
    assert manager._retiring_cameras["camera-a"][0] is first
    assert manager.start_camera("camera-a") is False
    manager.stop_camera("camera-a")
    assert first.shutdown_calls == 2
    assert "camera-a" not in manager._retiring_cameras
    assert manager.start_camera("camera-a") is True
    assert len(instances) == 2


def test_start_cannot_overlap_in_progress_stop(fake_manager):
    manager, control, _ = fake_manager
    assert manager.start_camera("camera-a") is True
    entered, release = threading.Event(), threading.Event()
    failures = []

    def shutdown(pipeline):
        entered.set()
        assert release.wait(2)

    def stop():
        try:
            manager.stop_camera("camera-a")
        except Exception as exc:
            failures.append(exc)

    control.shutdown = shutdown
    caller = threading.Thread(target=stop)
    caller.start()
    try:
        assert entered.wait(1)
        assert manager.start_camera("camera-a") is False
    finally:
        release.set()
        caller.join(2)
        control.shutdown = lambda pipeline: None
    assert not failures
    assert manager.start_camera("camera-a") is True


def test_unfinished_thread_keeps_retiring_ownership(fake_manager):
    manager, _, _ = fake_manager
    assert manager.start_camera("camera-a") is True
    thread = MagicMock()
    thread.is_alive.return_value = True
    manager._threads["camera-a"] = thread
    with pytest.raises(RuntimeError, match="still running"):
        manager.stop_camera("camera-a")
    assert manager.start_camera("camera-a") is False
    assert "camera-a" in manager._retiring_cameras
    thread.is_alive.return_value = False
    manager.stop_camera("camera-a")
    assert "camera-a" not in manager._retiring_cameras


def test_stop_all_continues_after_one_camera_cleanup_failure(fake_manager):
    manager, _, instances = fake_manager
    manager._cameras.append(CameraConfig(camera_id="camera-b", name="Fake B", source="offline"))
    assert manager.start_camera("camera-a") is True
    assert manager.start_camera("camera-b") is True
    instances[0].shutdown_failures = 1
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        manager.stop_all()
    assert instances[0].shutdown_calls == 1
    assert instances[1].shutdown_calls == 1
    assert "camera-a" in manager._retiring_cameras
    assert "camera-b" not in manager._retiring_cameras
    assert manager._inference_executor._shutdown is True
    manager.stop_all()
    assert not manager._retiring_cameras


def test_restart_cannot_override_a_later_stop(fake_manager, monkeypatch):
    manager, _, instances = fake_manager
    assert manager.start_camera("camera-a") is True
    original_stop = manager.stop_camera

    def stop_then_newer_stop(camera_id):
        original_stop(camera_id)
        original_stop(camera_id)

    monkeypatch.setattr(manager, "stop_camera", stop_then_newer_stop)
    assert manager.restart_camera("camera-a") is False
    assert len(instances) == 1
    assert "camera-a" not in manager._pipelines


def test_old_generation_alert_callback_is_rejected_after_stop(fake_manager):
    manager, _, instances = fake_manager
    alerts = []
    manager._on_alert = alerts.append
    assert manager.start_camera("camera-a") is True
    old_callback = instances[0].on_alert
    manager.stop_camera("camera-a")
    old_callback(SimpleNamespace(camera_id="camera-a"))
    assert alerts == []
    assert manager.alert_count == 0
