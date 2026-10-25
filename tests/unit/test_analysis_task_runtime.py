"""Mock-only source/task lifecycle and failure-isolation coverage."""

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from argus.capture.camera import CameraCapture, CameraState, FrameData
from argus.capture.source_runtime import CameraSourceRuntime, SourceCaptureView
from argus.core.analysis_task_runtime import TaskRuntime, TaskSupervisor
from argus.core.frame_identity import build_frame_identity


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("mock state transition timed out")
        time.sleep(0.005)


def frame(number=1):
    return FrameData(
        np.zeros((2, 2, 3), dtype=np.uint8), "cam", time.monotonic(), number, (2, 2),
        identity=build_frame_identity(
            camera_id="cam", source_declared="mock", source_runtime="mock",
            stream_id=None, frame_index=number, source_frame_index=number + 10,
            pts=number * 17, time_base_num=1, time_base_den=90000,
            frame_index_provenance="decoder_index", timestamp_provenance="decoder_pts",
        ),
    )


class MockCapture:
    camera_id = "cam"
    source = "mock"
    protocol = "mock"
    reconnect_delay = 0.01

    def __init__(self):
        self.state = CameraState()
        self.frames = queue.Queue()
        self.connects = 0
        self.stops = 0
        self.fail_stop = False

    def connect(self):
        self.connects += 1
        self.state.connected = True
        return True

    def read(self):
        try:
            return self.frames.get(timeout=0.01)
        except queue.Empty:
            return None

    def stop(self):
        self.stops += 1
        if self.fail_stop:
            raise RuntimeError("mock cleanup failure")
        self.state.connected = False


@pytest.fixture
def source():
    captures = []

    def factory():
        cap = MockCapture()
        captures.append(cap)
        return cap

    runtime = CameraSourceRuntime("cam", factory)
    yield runtime, captures
    for cap in captures:
        cap.fail_stop = False
    assert runtime.stop()


def test_source_is_lazy_and_concurrent_start_opens_once(source):
    runtime, captures = source
    assert captures == []
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert all(executor.map(lambda _: runtime.start(), range(8)))
    assert len(captures) == 1 and captures[0].connects == 1
    assert runtime.snapshot()["running"]


def test_stop_fences_late_connect_without_waiting_forever(monkeypatch):
    capture = MockCapture()
    entered, release = threading.Event(), threading.Event()

    def connect():
        entered.set()
        assert release.wait(2)
        capture.state.connected = True
        return True

    monkeypatch.setattr(capture, "connect", connect)
    runtime = CameraSourceRuntime("cam", lambda: capture)
    results = []
    start = threading.Thread(target=lambda: results.append(runtime.start()), daemon=True)
    start.start()
    try:
        assert entered.wait(1)
        before = time.monotonic()
        assert not runtime.stop(timeout=0.01)
        assert time.monotonic() - before < 0.5
        assert runtime.state.cleanup_pending and runtime.source_token is None
        release.set()
        start.join(2)
        assert not start.is_alive() and results == [False]
        assert runtime.source_token is None and not runtime.state.running
        assert runtime.stop()
        assert not runtime.state.cleanup_pending and not capture.state.connected
    finally:
        release.set()
        start.join(2)
        runtime.stop()


def test_cancelled_late_capture_factory_does_not_open_device():
    capture = MockCapture()
    entered, release = threading.Event(), threading.Event()

    def factory():
        entered.set()
        assert release.wait(2)
        return capture

    runtime = CameraSourceRuntime("cam", factory)
    results = []
    start = threading.Thread(target=lambda: results.append(runtime.start()), daemon=True)
    start.start()
    try:
        assert entered.wait(1)
        assert not runtime.stop(timeout=0.01)
        release.set()
        start.join(2)
        assert not start.is_alive() and results == [False]
        assert capture.connects == 0 and not capture.state.connected
        assert runtime.source_token is None
    finally:
        release.set()
        start.join(2)
        runtime.stop()


def test_legacy_view_stop_only_closes_its_subscription(source):
    runtime, captures = source
    legacy = SourceCaptureView(runtime, "legacy", timeout=0.5)
    peer = runtime.subscribe("peer")
    assert not legacy.connect()
    assert runtime.start() and legacy.connect()
    captures[0].frames.put(frame())
    legacy_frame = legacy.read_latest()
    assert legacy_frame is not None
    peer_frame = peer.get(0.5)
    assert peer_frame is not None
    legacy_frame.frame[:] = 55
    assert np.all(peer_frame.frame == 0)
    legacy.stop()
    assert runtime.state.running and captures[0].stops == 0
    captures[0].frames.put(frame(2))
    assert peer.get(0.5).frame_number == 2
    assert legacy.read_latest() is None


def test_source_reconnect_changes_epoch_and_preserves_media_identity(source):
    runtime, captures = source
    subscription = runtime.subscribe("consumer")
    assert runtime.start()
    captures[0].frames.put(frame())
    before = subscription.get(0.5)
    runtime.request_reconnect()
    assert runtime.source_token is None
    wait_until(lambda: len(captures) == 2 and runtime.state.connected)
    captures[1].frames.put(frame(2))
    after = subscription.get(0.5)
    assert after.identity.source_epoch > before.identity.source_epoch
    assert after.identity.source_session_id == before.identity.source_session_id
    assert after.identity.source_frame_index == 12
    assert after.identity.pts == 34
    assert after.identity.time_base_den == 90000
    assert after.identity.timestamp_provenance == "decoder_pts"


def test_source_cleanup_failure_blocks_reopen_until_retry(source):
    runtime, captures = source
    assert runtime.start()
    captures[0].fail_stop = True
    assert not runtime.stop(0.1)
    assert runtime.state.cleanup_pending
    assert not runtime.start()
    assert len(captures) == 1
    captures[0].fail_stop = False
    assert runtime.stop() and runtime.start()
    assert len(captures) == 2


def test_source_late_read_does_not_publish_or_allow_restart():
    entered, release = threading.Event(), threading.Event()

    class BlockedCapture(MockCapture):
        def read(self):
            entered.set()
            release.wait(2)
            return frame()

    capture = BlockedCapture()
    runtime = CameraSourceRuntime("cam", lambda: capture)
    subscription = runtime.subscribe("peer")
    try:
        assert runtime.start() and entered.wait(1)
        assert not runtime.stop(0.01)
        assert runtime.state.cleanup_pending and not runtime.start()
        release.set()
        assert runtime.stop()
        assert subscription.get(0) is None
        assert runtime.latest_frame is None
    finally:
        release.set()
        runtime.stop()


def test_two_tasks_share_frames_and_stop_independently(source):
    runtime, captures = source
    seen_a, seen_b = [], []

    def processor_a(value):
        value.identity.pipeline_ingest_ts = 123
        return value.frame_number

    first = TaskRuntime("first", runtime, processor_a, on_result=lambda result, _: seen_a.append(result))
    second = TaskRuntime("second", runtime, lambda value: value.identity.pipeline_ingest_ts,
                         on_result=lambda result, _: seen_b.append(result))
    supervisor = TaskSupervisor()
    supervisor.register(first)
    supervisor.register(second)
    try:
        assert first.start() and second.start()
        assert first.status.state == "waiting_source"
        assert runtime.start()
        captures[0].frames.put(frame())
        wait_until(lambda: len(seen_a) == len(seen_b) == 1)
        assert seen_a == [1] and seen_b == [None]
        assert first.stop()
        captures[0].frames.put(frame(2))
        wait_until(lambda: len(seen_b) == 2)
        assert seen_a == [1] and captures[0].stops == 0
        assert len(supervisor.status()) == 2
        assert supervisor.stop_source("unrelated")
    finally:
        assert supervisor.stop_all()


@pytest.mark.parametrize("change", ["stop", "reconnect", "deployment"])
def test_task_rejects_late_result_across_all_identity_fences(source, change):
    runtime, captures = source
    entered, release = threading.Event(), threading.Event()
    results = []

    def process(value):
        entered.set()
        release.wait(2)
        return value.frame_number

    task = TaskRuntime("late", runtime, process, on_result=lambda value, _: results.append(value))
    try:
        assert runtime.start() and task.start()
        captures[0].frames.put(frame())
        assert entered.wait(1)
        if change == "stop":
            assert not task.stop(0.01)
            assert not task.start()
        elif change == "reconnect":
            runtime.request_reconnect()
        else:
            task.set_deployment_revision(1)
        release.set()
        wait_until(lambda: task.status.discarded_results == 1)
        assert results == []
        assert task.stop()
    finally:
        release.set()
        task.stop()


def test_cleanup_failure_retains_task_reservation_and_retries(source):
    runtime, _ = source
    attempts = []

    def cleanup():
        attempts.append(True)
        return len(attempts) > 1

    task = TaskRuntime("cleanup", runtime, lambda value: value, cleanup=cleanup)
    assert task.start()
    assert not task.stop()
    assert task.status.cleanup_pending
    assert not task.start()
    assert task.stop()
    assert len(attempts) == 2
    assert task.start() and task.stop()


def test_task_processor_failure_isolated_from_peer_and_source(source):
    runtime, captures = source
    results = []

    def fail(_):
        raise RuntimeError("bad task")

    broken = TaskRuntime("broken", runtime, fail)
    healthy = TaskRuntime("healthy", runtime, lambda value: value.frame_number,
                          on_result=lambda value, _: results.append(value))
    try:
        assert runtime.start() and broken.start() and healthy.start()
        captures[0].frames.put(frame())
        wait_until(lambda: broken.status.state == "failed" and results == [1])
        assert runtime.state.running
        assert not broken.start()
        captures[0].frames.put(frame(2))
        wait_until(lambda: results == [1, 2])
    finally:
        assert broken.stop() and healthy.stop()


def test_observer_failure_does_not_stop_capture(source):
    # This intentionally uses no fixture source: it exercises the independent
    # callback subscription, not a real recorder or video encoder.
    capture = MockCapture()

    def observe(_):
        raise RuntimeError("observer unavailable")

    runtime = CameraSourceRuntime("cam", lambda: capture, on_frame=observe)
    subscription = runtime.subscribe("analysis")
    try:
        assert runtime.start()
        capture.frames.put(frame())
        assert subscription.get(0.5)
        wait_until(lambda: runtime.state.observer_error is not None)
        assert runtime.state.running
    finally:
        assert runtime.stop()


def test_source_reconnect_cleanup_failure_does_not_open_new_capture(source):
    runtime, captures = source
    assert runtime.start()
    captures[0].fail_stop = True
    runtime.request_reconnect()
    wait_until(lambda: runtime.state.cleanup_pending)
    assert len(captures) == 1
    assert not runtime.start()
    captures[0].fail_stop = False
    assert runtime.stop()


def test_native_capture_is_not_released_while_read_is_still_running():
    entered, release = threading.Event(), threading.Event()

    class NativeCapture(MockCapture):
        def __init__(self):
            super().__init__()
            self.stop_requested = False
            self.inside_read = False
            self.released_during_read = False

        def set_reconnect_handler(self, callback):
            self.reconnect_handler = callback

        def request_stop(self):
            self.stop_requested = True

        def stop(self):
            self.released_during_read |= self.inside_read
            super().stop()

        def read(self):
            self.inside_read = True
            entered.set()
            release.wait(2)
            self.inside_read = False
            return frame()

    capture = NativeCapture()
    runtime = CameraSourceRuntime("cam", lambda: capture)
    try:
        assert runtime.start() and entered.wait(1)
        assert capture.reconnect_handler == runtime.request_reconnect
        assert not runtime.stop(0.01)
        assert capture.stop_requested
        assert capture.stops == 0
        assert not runtime.start()
        release.set()
        assert runtime.stop()
        assert capture.stops == 1
        assert not capture.released_during_read
        assert runtime.latest_frame is None
    finally:
        release.set()
        runtime.stop()


def test_duplicate_task_start_failure_cannot_close_original_subscription(source):
    runtime, captures = source
    results = []
    first = TaskRuntime("same", runtime, lambda value: value.frame_number,
                        on_result=lambda result, _: results.append(result))
    duplicate = TaskRuntime("same", runtime, lambda value: None)
    try:
        assert runtime.start() and first.start()
        assert not duplicate.start()
        assert duplicate.stop()
        captures[0].frames.put(frame())
        wait_until(lambda: results == [1])
    finally:
        assert first.stop() and duplicate.stop()


def test_ordered_window_overflow_is_explicit_and_not_hidden_by_idle(source):
    runtime, captures = source
    entered, release = threading.Event(), threading.Event()
    results = []

    def process(value):
        if value.frame_number == 1:
            entered.set()
            release.wait(2)
        return value.frame_number

    task = TaskRuntime(
        "ordered", runtime, process, mode="ordered_window", capacity=2,
        on_result=lambda result, _: results.append(result),
    )
    try:
        assert runtime.start() and task.start()
        captures[0].frames.put(frame())
        assert entered.wait(1)
        for number in range(2, 7):
            captures[0].frames.put(frame(number))
        wait_until(lambda: runtime.state.frames_published == 6)
        release.set()
        wait_until(lambda: task.status.state == "sampling_insufficient")
        time.sleep(0.12)  # includes one idle read iteration
        assert task.status.state == "sampling_insufficient"
        assert task.status.dropped_frames == 3
        assert task.status.processed_frames == 0 and results == []
        captures[0].frames.put(frame(7))
        wait_until(lambda: runtime.state.frames_published == 7)
        assert task.start()  # idempotent start must not clear the latch
        assert task.status.state == "sampling_insufficient"
        assert results == []
        assert task.stop() and task.start()
        captures[0].frames.put(frame(8))
        wait_until(lambda: results == [8])
        assert task.status.dropped_frames == 0
    finally:
        release.set()
        assert task.stop()


def test_stateful_task_requires_explicit_reset(source):
    runtime, _ = source
    with pytest.raises(ValueError, match="require an explicit reset"):
        TaskRuntime("stateful", runtime, lambda _: None, stateful=True)


def test_stateful_task_resets_before_start_source_and_deployment_contexts(source):
    runtime, captures = source
    history, reasons, results = [], [], []

    def reset(reason):
        reasons.append(reason)
        history.clear()

    def process(value):
        history.append(value.frame_number)
        return list(history)

    task = TaskRuntime(
        "stateful", runtime, process, stateful=True, reset=reset,
        on_result=lambda result, _: results.append(result),
    )
    try:
        assert runtime.start() and task.start()
        captures[0].frames.put(frame(1))
        wait_until(lambda: len(results) == 1)
        captures[0].frames.put(frame(2))
        wait_until(lambda: len(results) == 2)
        task.set_deployment_revision(1)
        captures[0].frames.put(frame(3))
        wait_until(lambda: len(results) == 3)
        runtime.request_reconnect()
        wait_until(lambda: len(captures) == 2 and runtime.state.connected)
        captures[1].frames.put(frame(4))
        wait_until(lambda: len(results) == 4)
        assert task.stop() and task.start()
        captures[1].frames.put(frame(5))
        wait_until(lambda: len(results) == 5)
        assert results == [[1], [1, 2], [3], [4], [5]]
        assert reasons == ["task_started", "deployment_changed", "source_changed", "task_started"]
        assert task.snapshot()["stateful"]
    finally:
        assert task.stop()


def test_stateful_late_mutation_is_reset_before_next_source_result(source):
    runtime, captures = source
    entered, release = threading.Event(), threading.Event()
    history, results = [], []

    def process(value):
        if value.frame_number == 1:
            entered.set()
            release.wait(2)
        history.append(value.frame_number)
        return list(history)

    task = TaskRuntime(
        "stateful", runtime, process, stateful=True,
        reset=lambda _: history.clear(), on_result=lambda result, _: results.append(result),
    )
    try:
        assert runtime.start() and task.start()
        captures[0].frames.put(frame(1))
        assert entered.wait(1)
        runtime.request_reconnect()
        wait_until(lambda: len(captures) == 2 and runtime.state.connected)
        release.set()
        wait_until(lambda: task.status.discarded_results == 1)
        captures[1].frames.put(frame(2))
        wait_until(lambda: results == [[2]])
        assert history == [2]
    finally:
        release.set()
        assert task.stop()


def test_state_reset_failure_is_fail_closed(source):
    runtime, captures = source
    calls = []

    def reset(_):
        raise RuntimeError("state could not be reset")

    task = TaskRuntime("broken-reset", runtime, lambda value: calls.append(value),
                       stateful=True, reset=reset)
    try:
        assert runtime.start() and task.start()
        captures[0].frames.put(frame())
        wait_until(lambda: task.status.state == "failed")
        assert calls == []
        assert task.status.cleanup_pending
        assert not task.start()
    finally:
        assert task.stop()


def test_supervisor_unregister_requires_all_cleanup_and_fences_old_references(source):
    runtime, _ = source
    cleanup_ok = False
    failed = TaskRuntime("failed-cleanup", runtime, lambda value: value,
                         cleanup=lambda: cleanup_ok)
    clean = TaskRuntime("clean", runtime, lambda value: value)
    supervisor = TaskSupervisor()
    supervisor.register(failed)
    supervisor.register(clean)
    assert failed.start() and clean.start()
    try:
        assert not supervisor.stop_source("cam")
        assert not supervisor.unregister_clean_source("cam", source=runtime)
        assert supervisor.get("failed-cleanup") is failed
        assert supervisor.get("clean") is clean
        cleanup_ok = True
        assert supervisor.stop_source("cam")
        assert supervisor.unregister_clean_source("cam", source=runtime)
        assert supervisor.list() == []
        assert not failed.start() and not clean.start()
        with pytest.raises(ValueError, match="detached task"):
            supervisor.register(clean)
        replacement_source = CameraSourceRuntime("cam", MockCapture)
        replacement = TaskRuntime("clean", replacement_source, lambda value: value)
        supervisor.register(replacement)
        assert supervisor.start("clean")
        assert supervisor.unregister_clean_source("cam", source=runtime)
        assert supervisor.get("clean") is replacement
        assert supervisor.stop_all()
        assert supervisor.unregister_clean_source("cam", source=replacement_source)
        assert supervisor.list() == []
    finally:
        cleanup_ok = True
        supervisor.stop_all()


def test_supervisor_register_reads_runtime_state_without_collection_lock(source):
    runtime, _ = source
    supervisor = TaskSupervisor()

    class CheckedTask(TaskRuntime):
        @property
        def registration_closed(self):
            assert not supervisor._lock._is_owned()
            return super().registration_closed

    task = CheckedTask("register-lock", runtime, lambda value: value)
    assert supervisor.register(task) is task


@pytest.mark.parametrize("lock_name", ["_lock", "_lifecycle_lock"])
def test_supervisor_register_rejects_busy_task_without_waiting_under_manager_lock(source, lock_name):
    runtime, _ = source
    supervisor = TaskSupervisor()
    task = TaskRuntime("busy-register", runtime, lambda value: value)
    manager_lock = threading.Lock()
    task_owned = threading.Event()
    callback_done = threading.Event()

    def callback_reads_manager():
        with getattr(task, lock_name):
            task_owned.set()
            with manager_lock:
                callback_done.set()

    thread = threading.Thread(target=callback_reads_manager, daemon=True)
    with manager_lock:
        thread.start()
        assert task_owned.wait(1)
        with pytest.raises(RuntimeError, match="busy; retry registration"):
            supervisor.register(task)
        assert supervisor.list() == []
    assert callback_done.wait(1)
    thread.join(1)
    assert supervisor.register(task) is task


def test_supervisor_start_does_not_resurrect_detached_lookup(source, monkeypatch):
    runtime, _ = source
    supervisor = TaskSupervisor()
    task = TaskRuntime("lookup-race", runtime, lambda value: value)
    supervisor.register(task)
    get = supervisor.get

    def lookup_then_detach(task_id):
        result = get(task_id)
        assert supervisor.unregister_clean_source("cam", source=runtime)
        return result

    monkeypatch.setattr(supervisor, "get", lookup_then_detach)
    assert not supervisor.start("lookup-race")
    assert supervisor.list() == []
    assert not task.start()


def test_supervisor_start_and_read_only_callback_do_not_invert_locks(source):
    runtime, captures = source
    supervisor = TaskSupervisor()
    entered, read_status, read_done = threading.Event(), threading.Event(), threading.Event()
    results = []

    def result_callback(value, _):
        entered.set()
        assert read_status.wait(1)
        assert supervisor.get("callback-lock").task_id == "callback-lock"
        supervisor.status()
        results.append(value)
        read_done.set()

    task = TaskRuntime("callback-lock", runtime, lambda value: value.frame_number,
                       on_result=result_callback)
    supervisor.register(task)
    try:
        assert runtime.start() and supervisor.start(task.task_id)
        captures[0].frames.put(frame())
        assert entered.wait(1)
        with ThreadPoolExecutor(max_workers=1) as executor:
            concurrent_start = executor.submit(supervisor.start, task.task_id)
            # The callback holds task._lock while start waits. Collection
            # reads must remain available to let that callback finish.
            read_status.set()
            assert read_done.wait(1)
            assert concurrent_start.result(timeout=1)
        assert results == [1]
    finally:
        read_status.set()
        assert supervisor.stop_all()


def test_source_reconnect_preserves_attempt_budget_and_exponential_backoff():
    captures, delays = [], []

    class BoundedCapture(MockCapture):
        max_reconnect_attempts = 2
        reconnect_delay = 0.02

        def connect(self):
            self.connects += 1
            self.state.connected = len(captures) == 1
            self.state.error = None if self.state.connected else "mock offline"
            return self.state.connected

    def factory():
        capture = BoundedCapture()
        captures.append(capture)
        return capture

    runtime = CameraSourceRuntime("cam", factory)
    runtime._wait_for_reconnect = lambda delay: delays.append(delay) or False
    try:
        assert runtime.start()
        runtime.request_reconnect()
        wait_until(lambda: runtime.state.state == "failed")
        assert len(captures) == 3
        assert delays == [0.02, 0.04]
        assert runtime.state.reconnect_attempts == 2
        assert runtime.state.reconnect_count == 2
        assert runtime.state.reconnect_max_attempts == 2
        assert runtime.state.cleanup_pending and not runtime.start()
        assert runtime.source_token is None
        assert captures[-1].stops == 1
    finally:
        assert runtime.stop()


def test_source_reconnect_success_resets_backoff_for_next_incident():
    captures, delays = [], []

    class RetryingCapture(MockCapture):
        max_reconnect_attempts = 2
        reconnect_delay = 0.02

        def connect(self):
            self.connects += 1
            self.state.connected = len(captures) != 2
            return self.state.connected

    def factory():
        capture = RetryingCapture()
        captures.append(capture)
        return capture

    runtime = CameraSourceRuntime("cam", factory)
    runtime._wait_for_reconnect = lambda delay: delays.append(delay) or False
    try:
        assert runtime.start()
        runtime.request_reconnect()
        wait_until(lambda: len(captures) == 3 and runtime.state.connected)
        assert runtime.state.reconnect_attempts == 0
        runtime.request_reconnect()
        wait_until(lambda: len(captures) == 4 and runtime.state.connected)
        assert delays == [0.02, 0.04, 0.02]
    finally:
        assert runtime.stop()


def test_observer_discards_old_queued_frames_and_late_error_after_reconnect():
    captures, observed = [], []
    entered, release = threading.Event(), threading.Event()

    def factory():
        capture = MockCapture()
        captures.append(capture)
        return capture

    def observer(value):
        observed.append(value.frame_number)
        if value.frame_number == 1:
            entered.set()
            release.wait(2)
            raise RuntimeError("old source observer failure")

    runtime = CameraSourceRuntime("cam", factory, on_frame=observer)
    try:
        assert runtime.start()
        captures[0].frames.put(frame(1))
        assert entered.wait(1)
        captures[0].frames.put(frame(2))
        wait_until(lambda: runtime.state.frames_published == 2)
        runtime.request_reconnect()
        wait_until(lambda: len(captures) == 2 and runtime.state.connected)
        release.set()
        captures[1].frames.put(frame(3))
        wait_until(lambda: 3 in observed)
        assert observed == [1, 3]
        assert runtime.state.observer_error is None
    finally:
        release.set()
        assert runtime.stop()


def test_capture_view_preserves_all_capture_diagnostics_without_sdk_status_calls():
    captures = []

    class DiagnosticCapture(MockCapture):
        def runtime_status(self):
            raise AssertionError("source diagnostics must not call SDK status methods")

    def factory():
        capture = DiagnosticCapture()
        degraded = not captures
        capture.state.backend = "mock-usb"
        capture.state.requested_pixel_format = "MJPG"
        capture.state.pixel_format = "YUY2" if degraded else "MJPG"
        capture.state.requested_resolution = (1920, 1080)
        capture.state.actual_resolution = (640, 480) if degraded else (1920, 1080)
        capture.state.requested_fps = 60.0
        capture.state.reported_fps = 30.0 if degraded else 60.0
        capture.state.actual_fps = 12.5 if degraded else 59.9
        capture.state.degraded = degraded
        capture.state.degradation_reason = "USB sampling below required mode" if degraded else None
        captures.append(capture)
        return capture

    runtime = CameraSourceRuntime("cam", factory)
    view = SourceCaptureView(runtime, "legacy-status")
    try:
        assert runtime.start()
        # CameraCapture.runtime_status only reads this in-memory dataclass;
        # this guards the complete legacy key contract without constructing or
        # opening a real capture adapter.
        expected = CameraCapture.runtime_status(SimpleNamespace(_state=captures[0].state))
        status = view.runtime_status()
        assert status == expected == runtime.runtime_status()
        assert status["degraded"]
        assert status["actual_resolution"] == (640, 480)
        assert status["actual_fps"] == 12.5
        status["actual_fps"] = 0
        status["actual_resolution"] = (0, 0)
        status["degraded"] = False
        assert view.runtime_status() == expected
        runtime.request_reconnect()
        wait_until(lambda: len(captures) == 2 and runtime.state.connected)
        current = view.runtime_status()
        assert current == CameraCapture.runtime_status(SimpleNamespace(_state=captures[1].state))
        assert not current["degraded"]
        assert current["degradation_reason"] is None
        assert current["actual_resolution"] == (1920, 1080)
        assert current["actual_fps"] == 59.9
        assert current["pixel_format"] == "MJPG"
    finally:
        view.stop()
        assert runtime.stop()
