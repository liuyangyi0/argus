"""Asynchronous inference queue for batched model execution.

Collects frames from multiple cameras and runs YOLO/anomaly inference
in batches to maximize GPU/CPU utilization.  Falls back to single-frame
inference when the queue is empty (latency-first mode).

Architecture inspired by NVIDIA Triton Inference Server's dynamic
batching and Frigate's detector queue.

Usage::

    queue = InferenceQueue(detector, max_batch_size=4, max_wait_ms=10)
    queue.start()

    # From any camera thread:
    future = queue.submit(camera_id, frame)
    result = future.result(timeout=1.0)  # blocks until batch runs

    queue.stop()
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Callable

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class _InferenceRequest:
    """A pending inference request waiting to be batched."""

    camera_id: str
    frame: np.ndarray
    future: Future = field(default_factory=Future)
    enqueued_at: float = field(default_factory=time.monotonic)


class InferenceQueue:
    """Thread-safe batched inference queue.

    Parameters
    ----------
    predict_fn : callable
        Function that accepts a single frame (np.ndarray) and returns a
        result.  Used as fallback when batch function is not available.
    max_batch_size : int
        Maximum frames to batch before running inference.
    max_wait_ms : float
        Maximum milliseconds to wait for a full batch before running a
        partial batch.
    predict_batch_fn : callable, optional
        Function that accepts a list of frames (list[np.ndarray]) and
        returns a list of results (one per frame).  When provided,
        enables true batched inference for better throughput.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], Any],
        max_batch_size: int = 4,
        max_wait_ms: float = 10.0,
        predict_batch_fn: Callable[[list[np.ndarray]], list[Any]] | None = None,
    ):
        self._predict_fn = predict_fn
        self._predict_batch_fn = predict_batch_fn
        self._max_batch_size = max_batch_size
        self._max_wait_seconds = max_wait_ms / 1000.0
        self._queue: Queue[_InferenceRequest] = Queue(maxsize=64)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._total_batches = 0
        self._total_frames = 0
        self._batched_frames = 0  # frames processed via batch fn

    def start(self) -> None:
        """Start the background inference worker thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="argus-inference-queue",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def submit(self, camera_id: str, frame: np.ndarray) -> Future:
        """Submit a frame for inference, returning a Future for the result.

        If the queue is full, runs inference synchronously in the
        caller's thread (backpressure strategy).
        """
        request = _InferenceRequest(camera_id=camera_id, frame=frame)

        try:
            self._queue.put_nowait(request)
        except Exception:
            # Queue full: run synchronously as fallback
            try:
                result = self._predict_fn(request.frame)
                request.future.set_result(result)
            except Exception as e:
                request.future.set_exception(e)

        return request.future

    def _worker_loop(self) -> None:
        """Background thread: collect batch -> run inference -> deliver results."""
        while not self._stop_event.is_set():
            batch: list[_InferenceRequest] = []

            # Wait for first request (blocks up to 100ms)
            try:
                first = self._queue.get(timeout=0.1)
                batch.append(first)
            except Empty:
                continue

            # Collect more requests up to batch size or wait timeout
            deadline = time.monotonic() + self._max_wait_seconds
            while len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    req = self._queue.get(timeout=remaining)
                    batch.append(req)
                except Empty:
                    break

            # Run inference
            self._total_batches += 1
            self._total_frames += len(batch)

            if self._predict_batch_fn is not None and len(batch) > 1:
                # True batch inference
                self._run_batch(batch)
            else:
                # Single-frame inference (fallback)
                self._run_sequential(batch)

        # Drain remaining requests on shutdown
        while not self._queue.empty():
            try:
                req = self._queue.get_nowait()
                req.future.cancel()
            except Empty:
                break

    def _run_batch(self, batch: list[_InferenceRequest]) -> None:
        """Run true batched inference using predict_batch_fn."""
        frames = [req.frame for req in batch]
        try:
            results = self._predict_batch_fn(frames)  # type: ignore[misc]
            self._batched_frames += len(batch)
            if len(results) != len(batch):
                logger.warning(
                    "inference_queue.batch_size_mismatch",
                    expected=len(batch),
                    got=len(results),
                )
                # Pad with None or trim
                for i, req in enumerate(batch):
                    if req.future.cancelled():
                        continue
                    if i < len(results):
                        req.future.set_result(results[i])
                    else:
                        req.future.set_exception(
                            RuntimeError("Batch result missing for this frame")
                        )
            else:
                for req, result in zip(batch, results):
                    if not req.future.cancelled():
                        req.future.set_result(result)
        except Exception as e:
            # Batch failed — fall back to sequential
            logger.warning(
                "inference_queue.batch_failed_fallback",
                batch_size=len(batch),
                error=str(e),
            )
            self._run_sequential(batch)

    def _run_sequential(self, batch: list[_InferenceRequest]) -> None:
        """Run inference one frame at a time."""
        for req in batch:
            if req.future.cancelled():
                continue
            try:
                result = self._predict_fn(req.frame)
                if not req.future.cancelled():
                    req.future.set_result(result)
            except Exception as e:
                if not req.future.cancelled():
                    req.future.set_exception(e)

    @property
    def pending(self) -> int:
        """Number of frames waiting in the queue."""
        return self._queue.qsize()

    @property
    def stats(self) -> dict:
        """Return queue statistics."""
        return {
            "total_batches": self._total_batches,
            "total_frames": self._total_frames,
            "batched_frames": self._batched_frames,
            "avg_batch_size": (
                self._total_frames / self._total_batches
                if self._total_batches > 0
                else 0.0
            ),
            "pending": self.pending,
            "batch_enabled": self._predict_batch_fn is not None,
        }
