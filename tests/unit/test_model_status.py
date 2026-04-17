"""Tests for ModelStatus state transitions and serialization."""

from __future__ import annotations

import time

from argus.core.model_status import ModelStatus


class TestModelStatusLifecycle:
    def test_initial_state(self):
        s = ModelStatus(name="anomaly", camera_id="cam1")
        assert s.loaded is False
        assert s.backend == "none"
        assert s.consecutive_failures == 0
        assert s.total_inferences == 0

    def test_mark_loaded_sets_backend(self):
        s = ModelStatus(name="anomaly", camera_id="cam1")
        s.mark_loaded(backend="torch-cuda", model_path="/m/model.pt",
                      image_size=(256, 256))
        assert s.loaded is True
        assert s.backend == "torch-cuda"
        assert s.model_path == "/m/model.pt"
        assert s.image_size == (256, 256)

    def test_mark_load_failed(self):
        s = ModelStatus(name="anomaly", camera_id="cam1")
        s.mark_loaded(backend="openvino")
        s.mark_load_failed("dynamic shape error")
        assert s.loaded is False
        assert s.last_error == "dynamic shape error"
        assert s.last_error_ts is not None

    def test_inference_success_resets_consecutive(self):
        s = ModelStatus(name="anomaly", camera_id="cam1")
        s.mark_inference_failure("Broadcast failed")
        s.mark_inference_failure("Broadcast failed")
        s.mark_inference_failure("Broadcast failed")
        assert s.consecutive_failures == 3
        s.mark_inference_success()
        assert s.consecutive_failures == 0
        assert s.total_inferences == 4
        assert s.total_failures == 3

    def test_last_error_preserved_after_success(self):
        """A currently-green model can still expose its last error so
        operators see 'intermittent issue' rather than 'always fine'."""
        s = ModelStatus(name="anomaly", camera_id="cam1")
        s.mark_inference_failure("boom")
        s.mark_inference_success()
        assert s.consecutive_failures == 0
        assert s.last_error == "boom"  # preserved
        assert s.last_success_ts is not None

    def test_to_dict_is_json_serialisable(self):
        import json

        s = ModelStatus(name="yolo", camera_id="c1", image_size=(640, 640))
        s.mark_loaded(backend="cuda", model_path="yolo11n.pt")
        s.set_extra(classes=[0, 15, 16], tracking=True)
        s.mark_inference_success()
        d = s.to_dict()
        # round-trips JSON with no tuples leaking through
        json.dumps(d)
        assert d["image_size"] == [640, 640]
        assert d["extra"]["classes"] == [0, 15, 16]
        assert d["total_inferences"] == 1

    def test_thread_safety_concurrent_marks(self):
        """Concurrent marks shouldn't corrupt counters."""
        import threading

        s = ModelStatus(name="a", camera_id="c")

        def worker():
            for _ in range(500):
                s.mark_inference_success()

        ts = [threading.Thread(target=worker) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert s.total_inferences == 2000
        assert s.total_failures == 0
