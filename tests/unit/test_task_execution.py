"""Isolated worker lifecycle tests using fake processes, pipes and memory only."""

import hashlib
import json
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from argus.capture.camera import FrameData
from argus.core.analysis_task_contracts import get_default_config
from argus.core.frame_identity import build_frame_identity
from argus.core.task_execution import IsolatedTaskWorker, TaskExecutionError, validate_execution_manifest


def execution_manifest(template="fast_motion.v1", task_id="task-1"):
    config = get_default_config(template)

    def canonical(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest = {
        "schema_version": "argus.task-execution-bundle.v1", "bundle_id": "bundle-1",
        "task_id": task_id, "camera_id": "cam", "task_revision": 1,
        "task_config_sha256": hashlib.sha256(canonical(config).encode()).hexdigest(),
        "template_id": template, "authority": "managed", "stage": "shadow", "config": config, "models": {},
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest).encode()).hexdigest()
    return manifest


def synthetic_frame(number=1, timestamp=10.0, epoch=1):
    identity = build_frame_identity(
        camera_id="cam", source_declared="private-source", source_runtime="private-source",
        stream_id=None, frame_index=number, capture_monotonic_ts=timestamp,
        source_session_id="session", source_epoch=epoch,
    )
    return FrameData(np.zeros((4, 4, 3), dtype=np.uint8), "cam", timestamp, number, (4, 4), identity=identity)


class FakeMemory:
    def __init__(self, *, create, size):
        self.name, self.buf = "fake-memory", bytearray(size)
        self.closed = self.unlinked = False
        self.close_failure = False

    def close(self):
        if self.close_failure:
            raise OSError("memory close failed")
        self.closed = True

    def unlink(self):
        self.unlinked = True


class FakePipe:
    def __init__(self, context):
        self.context = context
        self.messages = deque()
        self.closed = False

    def poll(self, timeout):
        if self.messages:
            return True
        if self.context.block:
            self.context.polling.set()
            self.context.dead.wait(timeout)
        if self.closed:
            raise OSError("pipe closed")
        return bool(self.messages)

    def recv(self):
        return self.messages.popleft()

    def send(self, message):
        if self.closed:
            raise OSError("pipe closed")
        if self.context.infer_hang:
            return
        self.messages.append({"kind": "reset", "request_id": message["request_id"]}
                             if message["kind"] == "reset" else {
                                 "kind": "result", "request_id": message["request_id"],
                                 "source": message["source"], "data": {"state": "observed", "detections": []},
                             })

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self, context):
        self.context = context
        self.pid = None
        self.alive = self.closed = False

    def start(self):
        self.pid, self.alive = 10, True
        if not self.context.load_hang:
            self.context.pipe.messages.append({"kind": "ready", "provider": "classical-cpu"})

    def is_alive(self):
        return self.alive

    def terminate(self):
        if not self.context.kill_failure:
            self.alive = False
            self.context.dead.set()

    def kill(self):
        self.terminate()

    def join(self, timeout):
        self.context.joined += 1

    def close(self):
        assert not self.alive
        self.closed = True


class FakeContext:
    def __init__(self, *, load_hang=False, infer_hang=False, block=False, kill_failure=False):
        self.load_hang, self.infer_hang = load_hang, infer_hang
        self.block, self.kill_failure = block, kill_failure
        self.dead, self.polling = threading.Event(), threading.Event()
        self.joined = 0
        self.pipe, self.child = FakePipe(self), FakePipe(self)
        self.process = FakeProcess(self)

    def Pipe(self, *, duplex):
        return self.pipe, self.child

    def Process(self, **kwargs):
        assert kwargs["target"].__name__ == "_worker_main"
        return self.process


def worker(tmp_path, **kwargs):
    context = FakeContext(**kwargs)
    instance = IsolatedTaskWorker(execution_manifest(), tmp_path, context=context,
                                  memory_factory=FakeMemory, max_frame_bytes=1024)
    return instance, context


def test_manifest_detaches_and_checks_configuration_and_bundle_hash():
    manifest = execution_manifest()
    detached = validate_execution_manifest(manifest)
    detached["config"]["rules"]["confidence_threshold"] = 0.2
    assert manifest["config"]["rules"]["confidence_threshold"] == 0.5
    manifest["task_revision"] = True
    with pytest.raises(TaskExecutionError, match="identity"):
        validate_execution_manifest(manifest)


@pytest.mark.parametrize("change", [
    lambda value: value.update(stage="production"),
    lambda value: value.update(authority="legacy"),
    lambda value: value.update(manifest_sha256="0" * 64),
    lambda value: value.update(script="untrusted.py"),
    lambda value: value["config"]["rules"].update(confidence_threshold=0.1),
])
def test_manifest_rejects_authority_or_identity_drift(change):
    manifest = execution_manifest()
    change(manifest)
    with pytest.raises(TaskExecutionError):
        validate_execution_manifest(manifest)


def test_manifest_accepts_complete_dual_roles_and_rejects_missing_or_swapped_labels():
    manifest = execution_manifest("ppe_compliance_dual.v1")
    for role, labels in (("person_detector", ["person"]), ("head_detector", ["helmet", "bare_head"])):
        manifest["models"][role] = {
            "model_id": role, "artifact_sha256": "a" * 64, "artifact_path": f"/models/{role}.onnx",
            "name": role, "adapter_id": "onnx_detection.v1", "output_schema": "yolov8_detection.v1",
            "labels": labels, "input_size": [32, 32], "input_layout": "NCHW", "input_dtype": "float32",
            "color_order": "RGB",
        }

    def rehash(value):
        canonical = json.loads(json.dumps(value))
        canonical.pop("manifest_sha256")
        for model in canonical["models"].values():
            model.pop("artifact_path")
        value["manifest_sha256"] = hashlib.sha256(json.dumps(
            canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()

    rehash(manifest)
    assert set(validate_execution_manifest(manifest)["models"]) == {"person_detector", "head_detector"}
    missing = json.loads(json.dumps(manifest))
    del missing["models"]["head_detector"]
    rehash(missing)
    with pytest.raises(TaskExecutionError, match="roles"):
        validate_execution_manifest(missing)
    manifest["models"]["head_detector"]["labels"] = ["helmet"]
    rehash(manifest)
    with pytest.raises(TaskExecutionError, match="incompatible"):
        validate_execution_manifest(manifest)


def test_fake_worker_returns_exact_frame_and_bundle_identity_without_private_source(tmp_path):
    instance, context = worker(tmp_path)
    instance.start()
    result = instance.infer(synthetic_frame())
    assert result["bundle_id"] == "bundle-1" and result["task_revision"] == 1
    assert result["source"]["frame_index"] == 1
    assert result["source"]["pts"] is None
    assert "private-source" not in json.dumps(result)
    assert result["alerts_emitted"] is False and result["production_allowed"] is False
    instance.reset("source_changed")
    assert instance.stop() and context.process.closed and context.joined


@pytest.mark.parametrize("phase", ["load", "infer"])
def test_timeouts_kill_join_and_release_memory(tmp_path, phase):
    instance, context = worker(tmp_path, load_hang=phase == "load", infer_hang=phase == "infer")
    if phase == "infer":
        instance.start()
    with pytest.raises(TaskExecutionError) as failure:
        instance.start() if phase == "load" else instance.infer(synthetic_frame())
    assert failure.value.code == ("LOAD_TIMEOUT" if phase == "load" else "INFER_TIMEOUT")
    assert context.process.closed and not context.process.alive
    assert instance._memory is None and instance._connection is None


def test_failed_kill_retains_resources_until_explicit_cleanup_retry(tmp_path):
    instance, context = worker(tmp_path, load_hang=True, kill_failure=True)
    with pytest.raises(TaskExecutionError) as failure:
        instance.start()
    assert failure.value.code == "CLEANUP_PENDING"
    memory = instance._memory
    assert context.process.alive and not memory.closed
    with pytest.raises(TaskExecutionError):
        instance.start()
    context.kill_failure = False
    assert instance.stop() and memory.closed and memory.unlinked


def test_cleanup_failure_is_retryable_after_process_has_exited(tmp_path):
    instance, _ = worker(tmp_path)
    instance.start()
    memory = instance._memory
    memory.close_failure = True
    assert not instance.stop()
    assert instance.snapshot()["state"] == "cleanup_pending" and instance._memory is memory
    memory.close_failure = False
    assert instance.stop() and memory.unlinked


@pytest.mark.parametrize("phase", ["load", "infer"])
def test_stop_interrupts_sdk_wait_without_waiting_for_request_deadline(tmp_path, phase):
    instance, context = worker(tmp_path, block=True, load_hang=phase == "load", infer_hang=phase == "infer")
    if phase == "infer":
        instance.start()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(instance.start if phase == "load" else instance.infer,
                                 *([] if phase == "load" else [synthetic_frame()]))
        assert context.polling.wait(1)
        assert instance.cancel()
        with pytest.raises(TaskExecutionError):
            future.result(timeout=1)
    assert not context.process.alive
    with pytest.raises(TaskExecutionError):
        instance.start()


def test_frame_identity_mismatch_fails_and_cleans_worker(tmp_path):
    instance, _ = worker(tmp_path)
    instance.start()
    frame = synthetic_frame()
    frame.timestamp += 1
    with pytest.raises(TaskExecutionError) as failure:
        instance.infer(frame)
    assert failure.value.code == "INVALID_FRAME" and instance._memory is None


def test_partial_spawn_failure_still_kills_child_with_assigned_pid(tmp_path):
    instance, context = worker(tmp_path)

    def partially_start():
        context.process.pid, context.process.alive = 10, True
        raise OSError("spawn handshake failed after child creation")

    context.process.start = partially_start
    with pytest.raises(TaskExecutionError):
        instance.start()
    assert not context.process.alive and context.process.closed
    assert instance._memory is None


def test_late_response_or_eof_is_rejected_and_cleaned(tmp_path):
    instance, context = worker(tmp_path)
    instance.start()
    context.pipe.messages.append({"kind": "result", "request_id": -1, "source": {}})
    with pytest.raises(TaskExecutionError, match="identity mismatch"):
        instance.infer(synthetic_frame())
    assert instance._memory is None


def test_worker_ready_provider_cannot_claim_a_different_backend(tmp_path):
    instance, context = worker(tmp_path)
    context.pipe.messages.append({"kind": "ready", "provider": "CPUExecutionProvider"})
    with pytest.raises(TaskExecutionError, match="readiness"):
        instance.start()
    assert instance._memory is None and not context.process.alive


@pytest.mark.parametrize("command", ["infer", "reset"])
def test_boolean_request_id_cannot_match_the_first_integer_request(tmp_path, command):
    instance, context = worker(tmp_path)
    instance.start()
    frame = synthetic_frame()
    response = {"kind": "result" if command == "infer" else "reset", "request_id": True}
    if command == "infer":
        from argus.core.task_execution import frame_provenance

        response.update(source=frame_provenance(frame), data={"state": "observed"})
    context.pipe.messages.append(response)
    with pytest.raises(TaskExecutionError, match="identity mismatch"):
        instance.infer(frame) if command == "infer" else instance.reset("task_started")
    assert instance._memory is None and not context.process.alive


@pytest.mark.parametrize("data", [None, [], "observed", {}, {"state": "failed"}, {"state": None}, {"state": []}])
def test_malformed_result_is_failed_and_cleaned_instead_of_a_normal_observation(tmp_path, data):
    from argus.core.task_execution import frame_provenance

    instance, context = worker(tmp_path)
    instance.start()
    frame = synthetic_frame()
    context.pipe.messages.append({"kind": "result", "request_id": 1,
                                  "source": frame_provenance(frame), "data": data})
    with pytest.raises(TaskExecutionError) as failure:
        instance.infer(frame)
    assert failure.value.code == "INVALID_OUTPUT"
    assert instance._memory is None and not context.process.alive


def test_fixed_worker_entrypoint_passes_detached_pixels_and_closes_ipc(monkeypatch):
    from argus.core.task_execution import _worker_main

    frame = synthetic_frame()
    memory = FakeMemory(create=True, size=1024)
    np.ndarray(frame.frame.shape, dtype=np.uint8, buffer=memory.buf)[:] = 7
    calls, replies = [], []

    class Adapter:
        provider = "fake-provider"

        def reset(self, reason):
            calls.append(reason)

        def infer(self, pixels, timestamp):
            assert pixels[0, 0, 0] == 7 and timestamp == frame.timestamp
            pixels[:] = 11
            return {"state": "observed", "detections": []}

    commands = iter([
        {"kind": "reset", "request_id": 1, "reason": "task_started"},
        {"kind": "infer", "request_id": 2, "shape": list(frame.frame.shape),
         "source": {"capture_monotonic_ts": frame.timestamp}},
    ])

    class Connection:
        closed = False

        def recv(self):
            try:
                return next(commands)
            except StopIteration as exc:
                raise EOFError from exc

        def send(self, message):
            replies.append(message)

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr("argus.core.task_execution.shared_memory.SharedMemory", lambda **kwargs: memory)
    monkeypatch.setattr("argus.core.task_adapters.build_task_adapter", lambda *args: Adapter())
    _worker_main(connection, memory.name, 1024, execution_manifest(), "unused")
    assert calls == ["task_started"]
    assert [message["kind"] for message in replies] == ["ready", "reset", "result"]
    assert memory.buf[0] == 7 and memory.closed and connection.closed
