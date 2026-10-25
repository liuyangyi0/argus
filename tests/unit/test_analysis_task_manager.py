"""Managed source integration using synthetic pixels and fake detectors only.

The real CameraManager lifecycle, shared source reader, frame hub and task
runtime execute here. All capture adapters, pipelines and model runners are
replaced before construction: no camera, video, encoder or model is opened.
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from argus.capture.camera import CameraState, FrameData
from argus.capture.manager import CameraManager
from argus.config.schema import AlertConfig, CameraConfig
from argus.core.analysis_task_contracts import legacy_task_id
from argus.core.analysis_task_runtime import TaskRuntime
from argus.core.frame_identity import build_frame_identity
from argus.core.pipeline import DetectionPipeline
from argus.core.runtime import ModelIdentity
from argus.storage.analysis_tasks import AnalysisTaskIntegrityError, AnalysisTaskService
from argus.storage.database import Database


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("mock-only manager state transition timed out")
        time.sleep(0.005)


def frame(number=1, camera_id="cam"):
    return FrameData(
        np.full((3, 4, 3), number, dtype=np.uint8),
        camera_id, time.monotonic(), number, (4, 3),
        identity=build_frame_identity(
            camera_id=camera_id, source_declared="synthetic", source_runtime="synthetic",
            stream_id=None, frame_index=number, source_frame_index=number + 100,
            pts=number * 17, time_base_num=1, time_base_den=90000,
            timestamp_provenance="test_pts", frame_index_provenance="test_frame_index",
        ),
    )


class FakeCapture:
    reconnect_delay = 0.01

    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.source = "synthetic"
        self.protocol = "rtsp"
        self.state = CameraState()
        self.frames = queue.Queue()
        self.connect_count = 0
        self.stop_count = 0
        self.fail_stop = False
        self.reader_threads = set()
        self.identity_context = {}

    def configure_frame_identity(self, **kwargs):
        self.identity_context.update(kwargs)

    def connect(self):
        self.connect_count += 1
        self.state.connected = True
        return True

    def read(self):
        self.reader_threads.add(threading.current_thread().name)
        try:
            return self.frames.get(timeout=0.02)
        except queue.Empty:
            return None

    def request_stop(self):
        pass  # read is bounded; native handle remains owned until reader exits.

    def stop(self):
        self.stop_count += 1
        if self.fail_stop:
            raise RuntimeError("synthetic capture cleanup failed")
        self.state.connected = False


@pytest.fixture
def harness(monkeypatch, tmp_path):
    captures, pipelines, alerts, recordings = [], [], [], []
    controls = {"initialize": True, "reconfirm": True, "shutdown": lambda _pipeline: None}

    def forbidden_io(*_args, **_kwargs):
        raise AssertionError("real model/capture I/O is forbidden in these tests")

    monkeypatch.setattr("argus.capture.camera.cv2.VideoCapture", forbidden_io)
    monkeypatch.setattr("argus.person.detector.get_shared_yolo", forbidden_io)

    class FakePipeline:
        def __init__(self, **kwargs):
            self.camera_config = kwargs["camera_config"]
            self.on_alert = kwargs["on_alert"]
            self.constructor_args = kwargs
            self._camera = self.create_capture_adapter()
            self._camera.configure_frame_identity(
                source_declared=kwargs["frame_identity_declared_source"],
                source_runtime=kwargs["frame_identity_runtime_source"],
                stream_id=kwargs["frame_identity_stream_id"],
            )
            self.latest = None
            self.reconfirm_count = 0
            self.shutdown_count = 0
            self.stats = SimpleNamespace(
                frames_captured=0, frames_analyzed=0, anomalies_detected=0,
                alerts_emitted=0, avg_latency_ms=0,
            )
            pipelines.append(self)

        def create_capture_adapter(self):
            capture = FakeCapture(self.camera_config.camera_id)
            capture.source = self.camera_config.source
            return capture

        def bind_capture_runtime(self, callback):
            self._camera = callback(self._camera)
            self._camera._timeout = 0.02
            self._capture_runtime_bound = True

        def initialize(self):
            return self._camera.connect() and controls["initialize"]

        def process(self):
            value = self._camera.read_latest()
            if value is not None:
                self.latest = value
                self.stats.frames_captured += 1
                self.stats.frames_analyzed += 1

        def shutdown(self):
            self.shutdown_count += 1
            controls["shutdown"](self)
            self._camera.stop()

        def reconcile_auxiliary_desired(self, _desired):
            return True

        def reconfirm_task_session(self):
            self.reconfirm_count += 1
            if not controls["reconfirm"]:
                raise RuntimeError("synthetic scene reconfirmation failure")

        def get_latest_frame_identity(self):
            return self.latest.identity if self.latest else None

        def get_latest_frame(self):
            return self.latest.frame if self.latest else None

        def get_raw_frame(self):
            return self.get_latest_frame()

        def get_latest_frame_with_identity(self):
            return self.get_latest_frame(), self.get_latest_frame_identity()

        def get_latest_anomaly_map(self):
            return None

        def get_scene_guard_status(self):
            return {"enabled": True, "usable": False, "state": "unconfirmed"}

    class FakeRunner:
        def __init__(self, *, pipeline, **kwargs):
            self.pipeline = pipeline
            self.constructor_args = kwargs
            self.model_identity = ModelIdentity(None, None)

        def initialize(self):
            return self.pipeline.initialize()

        def run_once(self):
            self.pipeline.process()
            return None

    monkeypatch.setattr("argus.capture.manager.DetectionPipeline", FakePipeline)
    monkeypatch.setattr("argus.capture.manager.CameraInferenceRunner", FakeRunner)
    database = Database(f"sqlite:///{tmp_path / 'manager.db'}")
    database.initialize()
    # Empty initial config avoids the legacy eager YOLO preload. Dynamic
    # registration below then exercises the real bootstrap path.
    manager = CameraManager([], AlertConfig(), on_alert=alerts.append, database=database)
    manager._continuous_recording_mgr = SimpleNamespace(
        push_frame=lambda camera_id, pixels, timestamp: recordings.append(
            (camera_id, int(pixels[0, 0, 0]), timestamp),
        ),
    )
    monkeypatch.setattr(manager, "_resolve_model_path", lambda _camera: None)
    monkeypatch.setattr(manager, "_model_version_id", lambda _camera, _path: None)
    monkeypatch.setattr(manager, "_create_shadow_runner", lambda _camera: None)
    monkeypatch.setattr(manager, "_get_legacy_shared_yolo", lambda _camera: None)

    def make_source_capture(runtime_config, _declared, _stream_id):
        capture = FakeCapture(runtime_config.camera_id)
        capture.source, capture.protocol = runtime_config.source, runtime_config.protocol
        captures.append(capture)
        return capture

    monkeypatch.setattr(manager, "_create_source_capture", make_source_capture)
    monkeypatch.setattr(manager, "_broadcast_camera_and_wall", lambda _camera: None)
    config = CameraConfig(camera_id="cam", name="Synthetic", source="synthetic", protocol="rtsp")
    manager.add_camera_config(config)
    yield SimpleNamespace(
        manager=manager, captures=captures, pipelines=pipelines,
        alerts=alerts, recordings=recordings, controls=controls, db=database, config=config,
    )
    for capture in captures:
        capture.fail_stop = False
    manager._stop_event.set()
    for camera_id in set(manager._sources) | set(manager._pipelines) | set(manager._retiring_sources):
        manager.stop_camera(camera_id)
    manager._task_supervisor.stop_all()
    manager._inference_executor.shutdown(wait=True)
    manager._encode_executor.shutdown(wait=True)
    database.close()


def test_actual_manager_start_has_one_reader_shared_by_legacy_and_peer(harness):
    manager = harness.manager
    assert manager.start_camera("cam")
    source = manager._sources["cam"]
    peer_values = []
    task = TaskRuntime("peer", source, lambda value: value.frame_number,
                       on_result=lambda value, _frame: peer_values.append(value))
    manager.register_analysis_task_runtime(task)
    assert task.start()
    harness.captures[0].frames.put(frame())
    wait_until(lambda: peer_values == [1] and harness.pipelines[0].stats.frames_captured == 1)
    wait_until(lambda: len(harness.recordings) == 1)
    assert manager.start_camera("cam")
    assert len(harness.captures) == 1
    assert harness.captures[0].connect_count == 1
    assert harness.captures[0].reader_threads == {"source-cam"}
    assert harness.pipelines[0].constructor_args["auto_discover_model"] is False
    assert manager._runners["cam"].constructor_args["authorized_model_resolver"] is not None
    assert harness.recordings[0][1] == 1


def test_stopping_legacy_task_keeps_source_peer_preview_and_recording(harness):
    manager = harness.manager
    assert manager.start_camera("cam")
    source = manager._sources["cam"]
    peer_values = []
    task = TaskRuntime("peer", source, lambda value: value.frame_number,
                       on_result=lambda value, _frame: peer_values.append(value))
    manager.register_analysis_task_runtime(task)
    assert task.start()
    harness.captures[0].frames.put(frame())
    wait_until(lambda: harness.pipelines[0].stats.frames_captured == 1 and peer_values == [1])
    assert manager.set_analysis_task_enabled(legacy_task_id("cam"), False)
    captured = harness.pipelines[0].stats.frames_captured
    harness.captures[0].frames.put(frame(2))
    wait_until(lambda: peer_values == [1, 2] and len(harness.recordings) == 2)
    assert source.state.running and harness.captures[0].stop_count == 0
    assert harness.pipelines[0].stats.frames_captured == captured
    assert int(manager.get_latest_frame("cam")[0, 0, 0]) == 2
    assert manager.get_latest_frame_identity("cam").source_frame_index == 102
    assert manager.set_analysis_task_enabled(legacy_task_id("cam"), True)
    assert harness.pipelines[0].reconfirm_count == 1


def test_scene_reconfirmation_failure_does_not_resume_task(harness):
    manager = harness.manager
    assert manager.start_camera("cam")
    assert manager.set_analysis_task_enabled(legacy_task_id("cam"), False)
    harness.controls["reconfirm"] = False
    with pytest.raises(RuntimeError, match="scene reconfirmation"):
        manager.set_analysis_task_enabled(legacy_task_id("cam"), True)
    assert manager._legacy_task_enabled["cam"] is False
    assert manager._sources["cam"].state.running


def test_late_scene_reconfirmation_cannot_enable_restarted_camera(harness, monkeypatch):
    manager = harness.manager
    assert manager.start_camera("cam")
    assert manager.set_analysis_task_enabled(legacy_task_id("cam"), False)
    old_pipeline = harness.pipelines[0]
    entered, release = threading.Event(), threading.Event()
    outcomes = []

    def blocked_reconfirmation():
        entered.set()
        assert release.wait(3)

    monkeypatch.setattr(old_pipeline, "reconfirm_task_session", blocked_reconfirmation)
    worker = threading.Thread(
        target=lambda: outcomes.append(manager.set_analysis_task_enabled(legacy_task_id("cam"), True)),
        daemon=True,
    )
    worker.start()
    try:
        assert entered.wait(1)
        assert manager.restart_camera("cam")
        assert manager._pipelines["cam"] is not old_pipeline
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert outcomes == [False]
    assert manager._legacy_task_enabled["cam"] is False


def test_stop_camera_cleans_shared_source_and_all_tasks(harness):
    manager = harness.manager
    assert manager.start_camera("cam")
    source = manager._sources["cam"]
    task = TaskRuntime("peer", source, lambda value: value.frame_number)
    manager.register_analysis_task_runtime(task)
    assert task.start()
    manager.stop_camera("cam")
    assert not source.state.running and not source.state.cleanup_pending
    assert task.status.state == "stopped"
    assert harness.captures[0].stop_count == 1
    assert "cam" not in manager._sources
    assert "cam" not in manager._retiring_sources
    assert "cam" not in manager._pipelines


def test_source_cleanup_failure_keeps_reservation_and_blocks_restart(harness):
    manager = harness.manager
    assert manager.start_camera("cam")
    source = manager._sources["cam"]
    harness.captures[0].fail_stop = True
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        manager.stop_camera("cam")
    assert manager._retiring_sources["cam"] is source
    assert not manager.start_camera("cam")
    assert len(harness.captures) == 1
    harness.captures[0].fail_stop = False
    manager.stop_camera("cam")
    assert manager.start_camera("cam")
    assert len(harness.captures) == 2


def test_failed_runner_initialization_preserves_source_without_publishing_legacy(harness):
    harness.controls["initialize"] = False
    assert not harness.manager.start_camera("cam")
    assert len(harness.captures) == 1
    assert harness.captures[0].stop_count == 0
    assert harness.manager._sources["cam"].state.running
    assert harness.manager._retiring_sources == {}
    assert harness.manager._pipelines == {}
    observed = harness.manager.get_analysis_task_runtime(legacy_task_id("cam"))
    assert observed["state"] == "failed" and observed["reason_code"] == "LEGACY_INITIALIZATION_FAILED"
    harness.controls["initialize"] = True
    assert harness.manager.start_camera("cam")
    assert len(harness.captures) == 1


def test_source_only_start_never_constructs_pipeline_or_resolves_models(harness, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("source startup must not construct model consumers"))
    monkeypatch.setattr("argus.capture.manager.DetectionPipeline", forbidden)
    monkeypatch.setattr(harness.manager, "_resolve_model_path", forbidden)
    assert harness.manager.start_source("cam")
    assert harness.pipelines == []
    source = harness.manager.get_camera_source("cam")
    peer_values = []
    peer = TaskRuntime("peer", source, lambda value: value.frame_number,
                       on_result=lambda value, _frame: peer_values.append(value))
    harness.manager.register_analysis_task_runtime(peer)
    assert peer.start()
    harness.captures[0].frames.put(frame(4))
    wait_until(lambda: peer_values == [4] and len(harness.recordings) == 1)
    assert harness.manager.start_source("cam")
    assert len(harness.captures) == 1
    assert int(harness.manager.get_latest_frame("cam")[0, 0, 0]) == 4
    assert int(harness.manager.get_raw_frame("cam")[0, 0, 0]) == 4
    assert harness.manager.get_latest_frame_identity("cam").source_frame_index == 104
    snapshot = harness.manager.record_snapshot_access("cam", "snapshot-4")
    preview = harness.manager.record_preview_stream_access("cam")
    diagnostics = harness.manager.get_frame_identity_status("cam")
    assert snapshot.source_frame_index == preview.source_frame_index == 104
    assert diagnostics["latest"]["source_frame_index"] == 104
    assert diagnostics["last_snapshot"]["snapshot_id"] == "snapshot-4"
    assert diagnostics["last_preview_stream_frame_index"] == 4
    assert diagnostics["last_pipeline_ts"] is None
    status = harness.manager.get_camera_status_payload("cam")
    assert status["connected"] and status["running"] and status["lifecycle"]["state"] == "source_only"
    legacy = harness.manager.get_analysis_task_runtime(legacy_task_id("cam"))
    assert not legacy["running"] and legacy["desired_state"] == "stopped"
    forbidden.assert_not_called()


def test_source_only_can_enable_legacy_without_another_reader(harness):
    assert harness.manager.start_source("cam")
    source = harness.manager.get_camera_source("cam")
    assert harness.manager.set_analysis_task_enabled(legacy_task_id("cam"), True)
    assert harness.manager.get_camera_source("cam") is source
    assert len(harness.captures) == 1
    assert harness.pipelines[0].reconfirm_count == 1
    assert harness.manager._legacy_task_enabled["cam"]


def test_failed_legacy_start_does_not_interrupt_peer_preview_or_recording(harness):
    manager = harness.manager
    assert manager.start_source("cam")
    source = manager.get_camera_source("cam")
    values = []
    peer = TaskRuntime("peer", source, lambda value: value.frame_number,
                       on_result=lambda value, _frame: values.append(value))
    manager.register_analysis_task_runtime(peer)
    assert peer.start()
    harness.controls["initialize"] = False
    assert not manager.set_analysis_task_enabled(legacy_task_id("cam"), True)
    harness.captures[0].frames.put(frame(6))
    wait_until(lambda: values == [6] and len(harness.recordings) == 1)
    assert manager.get_camera_source("cam") is source
    assert int(manager.get_latest_frame("cam")[0, 0, 0]) == 6
    assert harness.captures[0].stop_count == 0
    assert manager._pipelines == {}


def test_legacy_constructor_failure_keeps_independent_source_and_reports_failure(harness, monkeypatch):
    assert harness.manager.start_source("cam")
    source = harness.manager.get_camera_source("cam")
    forbidden = Mock(side_effect=RuntimeError("synthetic model constructor failure"))
    monkeypatch.setattr("argus.capture.manager.DetectionPipeline", forbidden)
    with pytest.raises(RuntimeError, match="synthetic model constructor failure"):
        harness.manager.set_analysis_task_enabled(legacy_task_id("cam"), True)
    assert harness.manager.get_camera_source("cam") is source
    assert harness.captures[0].stop_count == 0
    # Disabled consumer remains stopped; its independent initialization error
    # is still available on the camera lifecycle without claiming source loss.
    payload = harness.manager.get_camera_status_payload("cam")
    assert payload["connected"] and payload["lifecycle"]["state"] == "source_only"
    assert payload["lifecycle"]["legacy_error"] == "LEGACY_INITIALIZATION_FAILED"
    legacy = harness.manager.get_analysis_task_runtime(legacy_task_id("cam"))
    assert legacy["state"] == "failed" and legacy["reason_code"] == "LEGACY_INITIALIZATION_FAILED"
    assert harness.manager.set_analysis_task_enabled(legacy_task_id("cam"), False)
    assert harness.manager.get_analysis_task_runtime(legacy_task_id("cam"))["state"] == "stopped"


def test_slow_failed_legacy_cleanup_does_not_fence_peer_or_recording(harness):
    manager = harness.manager
    assert manager.start_source("cam")
    source = manager.get_camera_source("cam")
    values = []
    peer = TaskRuntime("peer", source, lambda value: value.frame_number,
                       on_result=lambda value, _frame: values.append(value))
    manager.register_analysis_task_runtime(peer)
    assert peer.start()
    entered, release = threading.Event(), threading.Event()
    outcomes = []

    def shutdown(_pipeline):
        entered.set()
        assert release.wait(3)

    harness.controls["initialize"] = False
    harness.controls["shutdown"] = shutdown
    start = threading.Thread(target=lambda: outcomes.append(manager.start_camera("cam")), daemon=True)
    start.start()
    try:
        assert entered.wait(1)
        assert manager.get_camera_source("cam") is source
        harness.captures[0].frames.put(frame(8))
        wait_until(lambda: values == [8] and len(harness.recordings) == 1)
        other = TaskRuntime("other-peer", source, lambda value: value.frame_number)
        manager.register_analysis_task_runtime(other)
        assert other.start()
        assert int(manager.get_latest_frame("cam")[0, 0, 0]) == 8
    finally:
        release.set()
        start.join(2)
        harness.controls["shutdown"] = lambda _pipeline: None
    assert outcomes == [False] and manager.get_camera_source("cam") is source


def test_changing_declared_source_requires_explicit_stop(harness):
    assert harness.manager.start_source("cam")
    source = harness.manager.get_camera_source("cam")
    harness.manager._cameras[0] = harness.config.model_copy(update={"source": "other-synthetic"})
    assert not harness.manager.start_source("cam")
    assert not harness.manager.start_camera("cam")
    assert harness.manager.get_camera_source("cam") is source
    assert len(harness.captures) == 1


def test_source_reconnect_uses_frozen_capture_config_after_in_place_edit(harness):
    manager = harness.manager
    assert manager.start_source("cam")
    source = manager.get_camera_source("cam")
    harness.config.source = "unapproved-replacement"
    source.request_reconnect()
    wait_until(lambda: len(harness.captures) == 2 and source.state.connected)
    assert harness.captures[1].source == "synthetic"
    assert harness.captures[1].identity_context["source_declared"] == "synthetic"
    assert not manager.start_source("cam")


def test_stop_source_cancels_pending_connect_and_cannot_publish_late(harness, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    capture = FakeCapture("cam")

    def connect():
        entered.set()
        assert release.wait(2)
        capture.state.connected = True
        return True

    monkeypatch.setattr(capture, "connect", connect)
    monkeypatch.setattr(capture, "request_stop", release.set)
    monkeypatch.setattr(harness.manager, "_create_source_capture", lambda *_args: capture)
    outcomes = []
    start = threading.Thread(target=lambda: outcomes.append(harness.manager.start_source("cam")), daemon=True)
    start.start()
    try:
        assert entered.wait(1)
        harness.manager.stop_source("cam")
    finally:
        release.set()
        start.join(2)
    assert not start.is_alive() and outcomes == [False]
    assert harness.manager.get_camera_source("cam") is None
    assert not harness.manager._starting_sources and not harness.manager._retiring_sources
    assert not capture.state.connected


def test_source_only_failed_cleanup_reserves_reader_until_retry(harness):
    manager = harness.manager
    assert manager.start_source("cam")
    source = manager.get_camera_source("cam")
    harness.captures[0].fail_stop = True
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        manager.stop_source("cam")
    assert manager._retiring_sources["cam"] is source
    assert not manager.start_source("cam")
    assert len(harness.captures) == 1
    harness.captures[0].fail_stop = False
    manager.stop_source("cam")
    assert manager.start_source("cam") and len(harness.captures) == 2


def test_source_stop_cancels_prepared_tasks_after_fencing_new_source_access(harness):
    manager = harness.manager
    assert manager.start_source("cam")
    source = manager.get_camera_source("cam")
    calls = []

    def cancel(camera_id):
        assert camera_id == "cam" and manager.get_camera_source(camera_id) is None
        assert source.state.running  # Not closed before prepared loads are fenced.
        calls.append(camera_id)
        return True

    manager.set_source_task_canceller(cancel)
    manager.stop_source("cam")
    assert calls == ["cam"] and not source.state.running


@pytest.mark.parametrize("failure", [False, RuntimeError("synthetic private detail")])
def test_prepared_task_cleanup_failure_retains_source_ownership_for_retry(harness, failure):
    manager = harness.manager
    assert manager.start_source("cam")
    source = manager.get_camera_source("cam")

    def cancel(_camera_id):
        if isinstance(failure, Exception):
            raise failure
        return failure

    manager.set_source_task_canceller(cancel)
    with pytest.raises(RuntimeError, match="cleanup incomplete") as raised:
        manager.stop_source("cam")
    assert "synthetic private detail" not in str(raised.value)
    assert manager._retiring_sources["cam"] is source
    assert not manager.start_source("cam")
    manager.set_source_task_canceller(lambda _camera_id: True)
    manager.stop_source("cam")
    assert not manager._retiring_sources


def test_dynamic_camera_bootstrap_does_not_replace_existing_config(harness):
    service = AnalysisTaskService(harness.db)
    initial = service.resolve_legacy("cam")
    assert initial is not None
    assert "source" not in initial["config"]
    changed = harness.config.model_copy(update={"name": "Changed in YAML"})
    harness.manager._bootstrap_task_identity(changed)
    assert service.resolve_legacy("cam")["config"] == initial["config"]
    second = CameraConfig(camera_id="second", name="Second", source="synthetic")
    harness.manager.add_camera_config(second)
    assert service.resolve_legacy("second")["camera_id"] == "second"
    assert harness.captures == []  # Adding identities never starts capture.


def test_reconnect_rejects_alert_from_old_source_epoch(harness):
    manager = harness.manager
    assert manager.start_camera("cam")
    harness.captures[0].frames.put(frame())
    wait_until(lambda: harness.pipelines[0].stats.frames_captured == 1)
    source = manager._sources["cam"]
    identity = harness.pipelines[0].latest.identity
    source.request_reconnect()
    wait_until(lambda: len(harness.captures) == 2 and source.state.connected)
    assert source.source_token != (identity.source_session_id, identity.source_epoch)
    harness.pipelines[0].on_alert(SimpleNamespace(camera_id="cam", zone_id="default"))
    assert harness.alerts == []


def test_reconnect_preserves_declared_runtime_source_and_stream_context(harness, monkeypatch):
    manager = harness.manager
    manager._source_resolver = lambda config: config.model_copy(update={"source": "resolved-synthetic"})
    monkeypatch.setattr(manager, "_frame_identity_stream_id", lambda _declared, _runtime: "same-stream")
    assert manager.start_camera("cam")
    expected = {
        "source_declared": "synthetic", "source_runtime": "resolved-synthetic",
        "stream_id": "same-stream",
    }
    assert harness.captures[0].identity_context == expected
    source = manager._sources["cam"]
    source.request_reconnect()
    wait_until(lambda: len(harness.captures) == 2 and source.state.connected)
    assert harness.captures[1].identity_context == expected


def test_reconnect_during_alert_corroboration_rejects_late_commit(harness):
    manager = harness.manager
    assert manager.start_camera("cam")
    harness.captures[0].frames.put(frame())
    wait_until(lambda: harness.pipelines[0].stats.frames_captured == 1)
    source = manager._sources["cam"]

    def reconnect_during_check(**_kwargs):
        source.request_reconnect()
        return SimpleNamespace(corroborated=True, partner_camera="peer")

    manager._correlator = SimpleNamespace(check=reconnect_during_check)
    harness.pipelines[0].on_alert(SimpleNamespace(camera_id="cam", zone_id="default"))
    assert harness.alerts == []
    assert manager.alert_count == 0


def test_stopped_task_rejects_late_external_alert_callback(harness):
    manager = harness.manager
    assert manager.start_camera("cam")
    harness.captures[0].frames.put(frame())
    wait_until(lambda: harness.pipelines[0].stats.frames_captured == 1)
    assert manager.set_analysis_task_enabled(legacy_task_id("cam"), False)
    harness.pipelines[0].on_alert(SimpleNamespace(camera_id="cam", zone_id="default"))
    assert harness.alerts == []
    assert manager.alert_count == 0


def test_alert_dispatcher_runs_outside_shared_source_lock(harness):
    manager = harness.manager
    assert manager.start_camera("cam")
    harness.captures[0].frames.put(frame())
    wait_until(lambda: harness.pipelines[0].stats.frames_captured == 1)
    source = manager._sources["cam"]
    acquired = threading.Event()
    probes = []

    def dispatch(alert):
        def probe_lock():
            with source._lock:
                acquired.set()

        probe = threading.Thread(target=probe_lock, daemon=True)
        probes.append(probe)
        probe.start()
        assert acquired.wait(1), "dispatcher must not retain the source acquisition lock"
        harness.alerts.append(alert)

    manager._on_alert = dispatch
    alert = SimpleNamespace(camera_id="cam", zone_id="default")
    try:
        harness.pipelines[0].on_alert(alert)
    finally:
        for probe in probes:
            probe.join(1)
    assert harness.alerts == [alert]
    assert manager.alert_count == 1


def test_successful_camera_restart_allows_peer_registration_on_new_source(harness):
    manager = harness.manager
    assert manager.start_camera("cam")
    first = TaskRuntime("peer", manager._sources["cam"], lambda value: value.frame_number)
    manager.register_analysis_task_runtime(first)
    assert first.start()
    assert manager.restart_camera("cam")
    second = TaskRuntime("peer", manager._sources["cam"], lambda value: value.frame_number)
    assert second.source is not first.source
    assert manager.register_analysis_task_runtime(second) is second
    assert second.start()


def test_real_pipeline_resume_requires_original_scene_reconfirmation_without_model_io():
    pipeline = DetectionPipeline.__new__(DetectionPipeline)
    pipeline._scene_processing_lock = threading.RLock()
    pipeline._scene_guard_lock = threading.RLock()
    pipeline._auxiliary_shutdown = False
    original_guard = SimpleNamespace(require_reconfirmation=Mock(), profile_id="original-profile")
    pipeline._scene_guard = original_guard
    pipeline._scene_guard_last_usable = True
    pipeline._scene_guard_last_check_frame = 99
    pipeline._scene_guard_policy_recovery_count = 10
    pipeline._reset_scene_sensitive_state = Mock()
    pipeline.reconfirm_task_session()
    assert pipeline._scene_guard is original_guard
    assert original_guard.profile_id == "original-profile"
    original_guard.require_reconfirmation.assert_called_once_with()
    pipeline._reset_scene_sensitive_state.assert_called_once_with()
    assert pipeline._scene_guard_last_usable is False
    assert pipeline._scene_guard_last_result is None
    assert pipeline._scene_guard_policy_recovery_required is True
    assert pipeline._scene_guard_policy_recovery_count == 0
    pipeline._auxiliary_shutdown = True
    with pytest.raises(RuntimeError, match="reinitialization"):
        pipeline.reconfirm_task_session()


def test_real_pipeline_capture_handoff_rejects_live_or_duplicate_ownership():
    pipeline = DetectionPipeline.__new__(DetectionPipeline)
    capture = FakeCapture("cam")
    pipeline._camera = capture
    pipeline.stats = SimpleNamespace(frames_captured=0)
    view = object()
    wrapper = Mock(return_value=view)
    capture.state.connected = True
    with pytest.raises(RuntimeError, match="before initialization"):
        pipeline.bind_capture_runtime(wrapper)
    wrapper.assert_not_called()
    capture.state.connected = False
    pipeline.bind_capture_runtime(wrapper)
    wrapper.assert_called_once_with(capture)
    assert pipeline._camera is view
    with pytest.raises(RuntimeError, match="already bound"):
        pipeline.bind_capture_runtime(wrapper)


@pytest.fixture
def identity_only_manager(monkeypatch, tmp_path):
    """Real empty DB and manager; any capture/model path is forbidden."""
    forbidden = Mock(side_effect=AssertionError("identity bootstrap must not start real I/O"))
    monkeypatch.setattr("argus.capture.camera.cv2.VideoCapture", forbidden)
    monkeypatch.setattr("argus.capture.manager.DetectionPipeline", forbidden)
    monkeypatch.setattr("argus.person.detector.get_shared_yolo", forbidden)
    database = Database(f"sqlite:///{tmp_path / 'identity-only.db'}")
    database.initialize()
    manager = CameraManager([], AlertConfig(), database=database)
    yield manager, database, forbidden
    manager.stop_all()
    database.close()


def test_non_authoritative_projection_failure_does_not_block_camera_identity_bootstrap(
    identity_only_manager, monkeypatch,
):
    manager, database, forbidden = identity_only_manager
    projection = Mock(side_effect=RuntimeError("synthetic projection failure with private details"))
    logger = Mock()
    monkeypatch.setattr(AnalysisTaskService, "sync_legacy_model_projection", projection)
    monkeypatch.setattr("argus.capture.manager.logger", logger)
    config = CameraConfig(camera_id="cam", name="Identity only", source="synthetic")
    manager.add_camera_config(config)
    task = AnalysisTaskService(database).resolve_legacy("cam")
    assert task is not None
    assert task["deployment"]["projection_status"] == "not_synced"
    assert task["deployment"]["applied_artifact_id"] is None
    assert manager._cameras == [config]
    assert manager._sources == {} and manager._pipelines == {}
    projection.assert_called_once_with("cam")
    logger.warning.assert_called_once_with(
        "manager.analysis_task_projection_unavailable", camera_id="cam",
    )
    forbidden.assert_not_called()


def test_identity_integrity_failure_still_blocks_bootstrap_and_skips_projection(
    identity_only_manager, monkeypatch,
):
    manager, database, forbidden = identity_only_manager
    bootstrap = Mock(side_effect=AnalysisTaskIntegrityError("synthetic identity corruption"))
    projection = Mock()
    monkeypatch.setattr(AnalysisTaskService, "bootstrap_legacy", bootstrap)
    monkeypatch.setattr(AnalysisTaskService, "sync_legacy_model_projection", projection)
    config = CameraConfig(camera_id="cam", name="Identity only", source="synthetic")
    with pytest.raises(AnalysisTaskIntegrityError, match="identity corruption"):
        manager.add_camera_config(config)
    assert manager._cameras == []
    assert manager._sources == {} and manager._pipelines == {}
    assert AnalysisTaskService(database).resolve_legacy("cam") is None
    bootstrap.assert_called_once()
    projection.assert_not_called()
    forbidden.assert_not_called()


def test_configured_manager_construction_never_preloads_recognition(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("manager construction must not initialize recognition"))
    monkeypatch.setattr("argus.person.detector.get_shared_yolo", forbidden)
    config = CameraConfig(camera_id="lazy-camera", name="Lazy camera", source="not-opened")
    manager = CameraManager([config], AlertConfig())
    try:
        assert manager._shared_yolo is None and not manager._shared_yolo_attempted
        assert not manager._sources and not manager._pipelines
        forbidden.assert_not_called()
    finally:
        manager.stop_all()


@pytest.mark.parametrize("protocol", ["rtsp", "file", "usb", "gige"])
def test_source_factory_preserves_protocol_and_identity_without_connect(monkeypatch, protocol):
    import sys
    from argus.capture.source_runtime import create_source_capture

    constructor = Mock()
    monkeypatch.setattr("argus.capture.camera.CameraCapture", constructor)
    monkeypatch.setitem(sys.modules, "argus.capture.gige_capture", SimpleNamespace(GigECapture=constructor))
    config = CameraConfig(camera_id="source-factory", name="Source factory", source="synthetic", protocol=protocol)
    result = create_source_capture(config, source_declared="declared-synthetic", stream_id="same-stream")
    assert result is constructor.return_value
    kwargs = constructor.call_args.kwargs
    assert kwargs["source_declared"] == "declared-synthetic"
    assert kwargs["source_runtime"] == config.source and kwargs["stream_id"] == "same-stream"
    assert kwargs["fps_target"] == config.fps_target and kwargs["resolution"] == config.resolution
    if protocol == "usb":
        assert kwargs["usb_backend"] == config.usb.preferred_backend
        assert kwargs["usb_pixel_format"] == config.usb.pixel_format
    elif protocol == "gige":
        assert kwargs["exposure"] == config.gige.exposure and kwargs["gain"] == config.gige.gain
        assert "protocol" not in kwargs
    else:
        assert kwargs["protocol"] == protocol
    result.connect.assert_not_called()
