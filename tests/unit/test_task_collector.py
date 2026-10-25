"""Image collector tests use in-memory frames and fake image bytes only."""

import threading
import time
import struct
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

from argus.capture.camera import FrameData
from argus.capture.source_runtime import CameraSourceRuntime
from argus.capture.task_collector import TaskFrameCollector, controlled_path, png_image_size
from argus.core.frame_identity import build_frame_identity


def png_image():
    def chunk(kind, value):
        return struct.pack(">I", len(value)) + kind + value + struct.pack(">I", zlib.crc32(kind + value))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 3, 2, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\0" * 20)) + chunk(b"IEND", b""))


@pytest.fixture
def source():
    runtime = CameraSourceRuntime("cam", lambda: pytest.fail("must not open a second capture"))
    runtime._state = "running"
    runtime._capture = SimpleNamespace(state=SimpleNamespace(connected=True))
    runtime.hub.activate("session", 1)
    return runtime


def binding():
    return {"task_id": "task-one", "camera_id": "cam", "template_id": "object_detection.v1",
            "task_revision": 1, "task_config_sha256": "a" * 64}


def frame(number=1, epoch=1):
    return FrameData(np.zeros((2, 3, 3), dtype=np.uint8), "cam", time.time(), number, (3, 2),
                     identity=build_frame_identity(camera_id="cam", source_declared="secret-url",
                         source_runtime="secret-url", stream_id=None, frame_index=number,
                         source_session_id="session", source_epoch=epoch, source_frame_index=99,
                         pts=123, time_base_num=1, time_base_den=90000))


def wait_for(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("mock collector did not reach expected state")
        time.sleep(0.005)


def collector(tmp_path, source, **kwargs):
    return TaskFrameCollector(SimpleNamespace(get_camera_source=lambda _: source), binding(),
                              tmp_path, "collection", interval=0, max_duration=0.5,
                              encoder=lambda _: png_image(), **kwargs)


def test_single_source_subscription_preserves_original_identity_and_source(source, tmp_path):
    peer = source.subscribe("existing-task")
    worker = collector(tmp_path, source, max_samples=1)
    worker.start()
    source.hub.publish(frame())
    wait_for(lambda: worker.snapshot()["state"] == "completed")
    assert worker.stop()
    result = worker.snapshot()
    sample = result["samples"][0]
    assert sample["identity"]["source_frame_index"] == 99
    assert sample["identity"]["pts"] == 123
    assert sample["image_size"] == [3, 2]
    assert "source_runtime" not in sample["identity"]
    assert sample["binding"] == binding()
    assert (tmp_path / sample["relative_path"]).read_bytes() == png_image()
    assert peer.get(0).identity.frame_index == 1
    assert source._state == "running"


def test_storage_budget_includes_previous_and_partial_files(source, tmp_path):
    task_directory = tmp_path / "task-one"
    task_directory.mkdir()
    (task_directory / "previous.png").write_bytes(b"x" * (1024 * 1024 - 3))
    worker = collector(tmp_path, source, max_storage_mb=1)
    worker.start()
    source.hub.publish(frame())
    wait_for(lambda: worker.snapshot()["state"] == "failed")
    assert worker.stop()
    assert worker.snapshot()["error_code"] == "STORAGE_LIMIT_REACHED"
    assert not worker.snapshot()["samples"]


def test_late_encoded_frame_after_reconnect_is_discarded(source, tmp_path):
    started, release = threading.Event(), threading.Event()

    def encode(_):
        started.set()
        assert release.wait(2)
        return png_image()

    worker = TaskFrameCollector(SimpleNamespace(get_camera_source=lambda _: source), binding(),
                               tmp_path, "late", max_duration=0.2, interval=0, encoder=encode)
    worker.start()
    source.hub.publish(frame())
    assert started.wait(2)
    source.hub.activate("session", 2)
    release.set()
    wait_for(lambda: worker.snapshot()["discarded_frames"] == 1)
    assert worker.stop()
    assert not worker.snapshot()["samples"]


def test_failed_manifest_write_retries_identical_receipt_without_duplicate_images(source, tmp_path):
    attempts = []

    def finish(receipt):
        attempts.append(receipt)
        if len(attempts) == 1:
            raise RuntimeError("database temporarily unavailable")

    worker = collector(tmp_path, source, max_samples=1, on_finish=finish)
    worker.start()
    source.hub.publish(frame())
    wait_for(lambda: worker.snapshot()["error_code"] == "MANIFEST_COMMIT_FAILED")
    assert worker.stop()
    assert worker.retry_finalize()
    assert attempts[0] == attempts[1]
    assert len(worker.snapshot()["samples"]) == 1
    assert not worker.snapshot()["persistence_pending"]


def test_stop_timeout_retains_worker_until_encoder_finishes(source, tmp_path):
    started, release = threading.Event(), threading.Event()

    def encode(_):
        started.set()
        assert release.wait(2)
        return png_image()

    worker = TaskFrameCollector(SimpleNamespace(get_camera_source=lambda _: source), binding(),
                               tmp_path, "blocked", max_duration=0.2, interval=0, encoder=encode)
    worker.start()
    source.hub.publish(frame())
    assert started.wait(2)
    assert not worker.stop(timeout=0)
    release.set()
    assert worker.stop()
    assert not worker.snapshot()["samples"]


@pytest.mark.parametrize("relative", ["../image.png", "task/../image.png", "C:/image.png", "task\\..\\image.png"])
def test_image_paths_reject_traversal_and_absolute_paths(tmp_path, relative):
    with pytest.raises(ValueError):
        controlled_path(tmp_path, relative)


def test_snapshot_mutation_cannot_change_owned_sample_binding(source, tmp_path):
    worker = collector(tmp_path, source, max_samples=1)
    worker.start()
    source.hub.publish(frame())
    wait_for(lambda: worker.snapshot()["state"] == "completed")
    assert worker.stop()
    changed = worker.snapshot()
    changed["samples"][0]["binding"]["task_revision"] = 99
    assert worker.snapshot()["samples"][0]["binding"]["task_revision"] == 1


def test_png_validation_rejects_crc_correct_but_invalid_compressed_pixels():
    raw = png_image()
    offset = raw.index(b"IDAT")
    prefix = raw[:offset - 4]
    suffix = raw[-12:]
    invalid = b"not-zlib-data"
    corrupt = prefix + struct.pack(">I", len(invalid)) + b"IDAT" + invalid + struct.pack(">I", zlib.crc32(b"IDAT" + invalid)) + suffix
    with pytest.raises(ValueError, match="compressed"):
        png_image_size(corrupt)


def test_duplicate_start_does_not_open_or_replace_subscription(source, tmp_path):
    worker = collector(tmp_path, source)
    worker.start()
    subscription = worker._subscription
    with pytest.raises(RuntimeError, match="once"):
        worker.start()
    assert worker._subscription is subscription
    assert worker.stop()


def test_subscription_close_failure_still_joins_worker_and_confirms_cleanup(source, tmp_path, monkeypatch):
    worker = collector(tmp_path, source)
    worker.start()
    original_close = worker._subscription.close
    calls = []

    def fail_once():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("transient subscription close failure")
        original_close()

    monkeypatch.setattr(worker._subscription, "close", fail_once)
    assert worker.stop()
    assert not worker._thread.is_alive()
    assert worker.snapshot()["cleanup_confirmed"] is True
