"""Smoke orchestration contracts only: never start CUDA, processes or cameras."""

from pathlib import Path
import multiprocessing
import sys
from unittest.mock import Mock
from types import SimpleNamespace

import pytest


@pytest.fixture
def smoke(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import run_gpu_task_restart_smoke

    return run_gpu_task_restart_smoke


def test_restart_smoke_import_does_not_initialize_model_sdks(smoke):
    assert not {"torch", "onnxruntime", "onnx", "ultralytics"} & sys.modules.keys()
    assert len(smoke.CHECKS) == len(set(smoke.CHECKS)) == 6


def test_controller_receipt_returns_exact_kind(smoke):
    connection = Mock()
    document = {"kind": "ready", "worker_pids": [1, 2]}
    connection.poll.return_value = True
    connection.recv.return_value = document
    assert smoke.receive(connection, "ready", 3) is document
    connection.poll.assert_called_once_with(3)


def test_controller_timeout_does_not_read_or_assume_success(smoke):
    connection = Mock()
    connection.poll.return_value = False
    with pytest.raises(RuntimeError, match="before its deadline"):
        smoke.receive(connection, "ready", 0.1)
    connection.recv.assert_not_called()


@pytest.mark.parametrize("response", [{"kind": "failed", "error_code": "UNAVAILABLE"}, None, {"kind": "closed"}])
def test_wrong_controller_receipt_cannot_be_applied(smoke, response):
    connection = Mock()
    connection.poll.return_value = True
    connection.recv.return_value = response
    with pytest.raises(RuntimeError, match="controller response failed"):
        smoke.receive(connection, "ready")


def _idle_without_models(gate):
    """An ordinary dummy process; no frame, model or GPU imports."""
    gate.wait(30)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job containment contract")
def test_windows_job_contains_and_cleans_only_its_owned_dummy_process(smoke):
    context = multiprocessing.get_context("spawn")
    gate = context.Event()
    process = context.Process(target=_idle_without_models, args=(gate,))
    job = None
    handles = []
    process.start()
    try:
        job = smoke.OwnedWindowsJob(process)
        handles = job.capture_worker_handles([process.pid])
        assert job.api.WaitForSingleObject(handles[0], 0) == 258  # WAIT_TIMEOUT: alive, not CUDA.
        job.close()
        assert job.api.WaitForSingleObject(handles[0], 5000) == 0
        process.join(5)
        assert not process.is_alive()
    finally:
        if job is not None:
            job.close()
            for handle in handles:
                job.api.CloseHandle(handle)
        if process.is_alive():
            process.terminate()
        process.join(5)
        assert not process.is_alive()
        process.close()


def _observation(task_id, frame_index, bundle="bundle"):
    return {"task_id": task_id, "bundle_id": bundle, "provider": "CUDAExecutionProvider",
        "production_allowed": False, "alerts_emitted": False,
        "source": {"frame_index": frame_index, "source_session_id": "session", "source_epoch": 1,
            "pts": None, "source_frame_index": None},
        "data": {"detections": [{"label": "synthetic_object", "confidence": 0.9}]}}


def _synthetic_observation_setup(smoke, monkeypatch, results):
    module = sys.modules["run_gpu_task_runtime_smoke"]
    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    source = SimpleNamespace(sequence=0, session="session")

    def publish():
        source.sequence += 1
        return source.sequence

    source.publish = publish
    system = SimpleNamespace(executor=SimpleNamespace(results=lambda task: results(task, source.sequence)))
    return module, system, source


def test_gpu_observe_retries_dropped_frame_but_requires_one_common_source_frame(smoke, monkeypatch):
    def results(task, sequence):
        if sequence < 2:
            return [] if task == "b" else [_observation(task, 1)]
        return [_observation(task, sequence)]

    module, system, source = _synthetic_observation_setup(smoke, monkeypatch, results)
    found = module.observe(system, source, {"a": ("bundle", 0.9), "b": ("bundle", 0.9)})
    assert source.sequence == 2
    assert {item["source"]["frame_index"] for item in found.values()} == {2}


def test_gpu_observe_never_combines_different_original_frames(smoke, monkeypatch):
    module, system, source = _synthetic_observation_setup(smoke, monkeypatch,
        lambda task, sequence: [_observation(task, sequence if task == "a" else sequence - 1)])
    with pytest.raises(RuntimeError, match="did not share a synthetic frame"):
        module.observe(system, source, {"a": ("bundle", 0.9), "b": ("bundle", 0.9)})
    assert 2 <= source.sequence <= 61


def test_gpu_observe_rejects_wrong_bundle_even_when_frame_ids_match(smoke, monkeypatch):
    module, system, _source = _synthetic_observation_setup(smoke, monkeypatch,
        lambda task, sequence: [_observation(task, sequence, "wrong" if task == "b" else "bundle")])
    with pytest.raises(RuntimeError, match="isolation failed"):
        module.observe(system, _source, {"a": ("bundle", 0.9), "b": ("bundle", 0.9)})
