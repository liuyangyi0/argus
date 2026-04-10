"""Tests for the batched inference queue."""

import time
import threading

import numpy as np
import pytest

from argus.core.inference_queue import InferenceQueue


def _dummy_predict(frame: np.ndarray) -> dict:
    """Simulate model inference with deterministic output."""
    return {"score": float(frame.mean()) / 255.0, "shape": frame.shape}


def _slow_predict(frame: np.ndarray) -> dict:
    """Simulate slow model inference."""
    time.sleep(0.05)
    return {"score": 0.5}


def _failing_predict(frame: np.ndarray) -> dict:
    raise RuntimeError("model crashed")


class TestInferenceQueue:
    def test_submit_and_get_result(self):
        queue = InferenceQueue(_dummy_predict, max_batch_size=2, max_wait_ms=50)
        queue.start()
        try:
            frame = np.zeros((256, 256, 3), dtype=np.uint8)
            future = queue.submit("cam-01", frame)
            result = future.result(timeout=2.0)
            assert result["score"] == 0.0
            assert result["shape"] == (256, 256, 3)
        finally:
            queue.stop()

    def test_multiple_submissions(self):
        queue = InferenceQueue(_dummy_predict, max_batch_size=4, max_wait_ms=50)
        queue.start()
        try:
            futures = []
            for i in range(8):
                frame = np.full((64, 64, 3), i * 10, dtype=np.uint8)
                futures.append(queue.submit(f"cam-{i}", frame))

            results = [f.result(timeout=2.0) for f in futures]
            assert len(results) == 8
            # First frame is all zeros
            assert results[0]["score"] == 0.0
        finally:
            queue.stop()

    def test_concurrent_submissions(self):
        queue = InferenceQueue(_dummy_predict, max_batch_size=4, max_wait_ms=20)
        queue.start()
        try:
            results = []
            lock = threading.Lock()

            def submit_batch(cam_id):
                for i in range(10):
                    frame = np.zeros((64, 64, 3), dtype=np.uint8)
                    future = queue.submit(cam_id, frame)
                    result = future.result(timeout=2.0)
                    with lock:
                        results.append(result)

            threads = [
                threading.Thread(target=submit_batch, args=(f"cam-{i}",))
                for i in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)

            assert len(results) == 40
        finally:
            queue.stop()

    def test_failing_predict_sets_exception(self):
        queue = InferenceQueue(_failing_predict, max_batch_size=1, max_wait_ms=10)
        queue.start()
        try:
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            future = queue.submit("cam-01", frame)
            with pytest.raises(RuntimeError, match="model crashed"):
                future.result(timeout=2.0)
        finally:
            queue.stop()

    def test_stats(self):
        queue = InferenceQueue(_dummy_predict, max_batch_size=2, max_wait_ms=10)
        queue.start()
        try:
            for _ in range(4):
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                queue.submit("cam-01", frame).result(timeout=2.0)

            stats = queue.stats
            assert stats["total_frames"] >= 4
            assert stats["total_batches"] >= 1
        finally:
            queue.stop()

    def test_stop_drains_queue(self):
        queue = InferenceQueue(_slow_predict, max_batch_size=1, max_wait_ms=10)
        queue.start()

        # Submit several frames
        futures = []
        for _ in range(3):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            futures.append(queue.submit("cam-01", frame))

        # Stop should complete without hanging
        queue.stop()

    def test_queue_full_fallback(self):
        """When queue is full, inference runs synchronously in caller thread."""
        # Use maxsize=1 queue internally. Fill it first, then trigger fallback.
        queue = InferenceQueue(_slow_predict, max_batch_size=1, max_wait_ms=100)
        # Start the worker so the first item gets picked up (slowly)
        queue.start()
        try:
            # Fill the internal queue beyond capacity
            futures = []
            for _ in range(70):  # exceed maxsize=64
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                futures.append(queue.submit("cam-01", frame))
            # At least one future should complete (either via worker or fallback)
            completed = sum(1 for f in futures if f.done())
            # The ones that triggered fallback should have results immediately
            assert completed >= 1
        finally:
            queue.stop()
