"""Write-behind buffer for InferenceRecord DB persistence (Section 7).

Batches per-frame inference results and bulk-inserts them to the
inference_records table at configurable intervals.  This avoids
per-frame DB writes (20 writes/sec at 5 FPS x 4 cameras) and
instead does ~1200-row bulk inserts every 60 seconds.

Works alongside the existing disk-based InferenceRecordStore
which handles JSON + raw frame persistence for audit.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import structlog

from argus.storage.database import Database
from argus.storage.models import InferenceRecord

logger = structlog.get_logger()


class InferenceBuffer:
    """Thread-safe write-behind buffer for inference record DB persistence.

    Usage:
        buffer = InferenceBuffer(database, flush_seconds=60, max_size=1000)
        buffer.start()
        ...
        buffer.append(InferenceRecord(...))  # called per-frame from pipeline
        ...
        buffer.stop()  # flushes remaining records
    """

    def __init__(
        self,
        database: Database,
        flush_seconds: int = 60,
        max_size: int = 1000,
    ):
        self._db = database
        self._flush_seconds = flush_seconds
        self._buffer: deque[InferenceRecord] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._total_flushed = 0

    def start(self) -> None:
        """Start the background flush thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._flush_loop,
            name="inference-buffer",
            daemon=False,
        )
        self._thread.start()
        logger.info(
            "inference_buffer.started",
            flush_seconds=self._flush_seconds,
            max_size=self._buffer.maxlen,
        )

    def stop(self) -> None:
        """Stop the flush thread, draining remaining records."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=15.0)
            self._thread = None
        # Final flush
        self._flush()

    def append(self, record: InferenceRecord) -> None:
        """Add an inference record to the buffer (thread-safe).

        If the buffer is full (maxlen reached), the oldest record
        is automatically dropped (deque behavior).
        """
        with self._lock:
            self._buffer.append(record)

    def _flush_loop(self) -> None:
        """Background loop that flushes the buffer at intervals."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self._flush_seconds)
            self._flush()

    def _flush(self) -> None:
        """Drain the buffer and bulk-insert to DB."""
        with self._lock:
            if not self._buffer:
                return
            batch = list(self._buffer)
            self._buffer.clear()

        try:
            count = self._db.save_inference_batch(batch)
            self._total_flushed += count
            logger.debug(
                "inference_buffer.flushed",
                count=count,
                total=self._total_flushed,
            )
        except Exception:
            logger.error(
                "inference_buffer.flush_failed",
                batch_size=len(batch),
                exc_info=True,
            )

    @property
    def pending_count(self) -> int:
        """Number of records waiting to be flushed."""
        with self._lock:
            return len(self._buffer)

    @property
    def total_flushed(self) -> int:
        """Total records flushed to DB since start."""
        return self._total_flushed
