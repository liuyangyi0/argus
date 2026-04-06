"""Background persistence for audit-grade inference records (5.2).

Follows the same Queue + background thread pattern as AlertDispatcher.
Only records with final_decision of INFO or ALERT are submitted.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from argus.core.inference_record import InferenceRecord

logger = structlog.get_logger()


class InferenceRecordStore:
    """Asynchronous writer for InferenceRecord + raw frame persistence.

    Records are enqueued via submit() and written to disk by a background
    daemon thread.  Directory layout:
        {base_dir}/YYYY-MM-DD/{camera_id}/{frame_id}.json
        {base_dir}/YYYY-MM-DD/{camera_id}/{frame_id}.jpg     (raw frame)
        {base_dir}/YYYY-MM-DD/{camera_id}/{frame_id}_heatmap.png  (if score > threshold)
    """

    def __init__(self, base_dir: Path, max_queue: int = 500):
        self._base_dir = Path(base_dir)
        self._queue: Queue[tuple[InferenceRecord, np.ndarray | None]] = Queue(
            maxsize=max_queue
        )
        self._thread: threading.Thread | None = None  # non-daemon to drain on shutdown
        self._stop = threading.Event()

    def start(self) -> None:
        """Start the background writer thread."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="inference-store",
            daemon=False,
        )
        self._thread.start()
        logger.info("inference_store.started", base_dir=str(self._base_dir))

    def stop(self) -> None:
        """Stop the writer thread, draining remaining items."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    def submit(self, record: InferenceRecord, raw_frame: np.ndarray | None = None) -> None:
        """Enqueue a record for background persistence."""
        try:
            self._queue.put_nowait((record, raw_frame))
        except Full:
            logger.warning(
                "inference_store.queue_full",
                camera_id=record.camera_id,
                frame_id=record.frame_id,
            )

    def _writer_loop(self) -> None:
        """Background loop writing records to disk."""
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except Empty:
                continue

            record, raw_frame = item
            try:
                self._write_record(record, raw_frame)
            except Exception as e:
                logger.error(
                    "inference_store.write_failed",
                    frame_id=record.frame_id,
                    error=str(e),
                )

        # Drain remaining items on shutdown
        while not self._queue.empty():
            try:
                record, raw_frame = self._queue.get_nowait()
                self._write_record(record, raw_frame)
            except Exception:
                break

    def _write_record(self, record: InferenceRecord, raw_frame: np.ndarray | None) -> None:
        """Write a single record + optional frame to disk."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_dir = self._base_dir / date_str / record.camera_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write JSON record
        json_path = out_dir / f"{record.frame_id}.json"
        data = record.to_dict()

        # Save raw frame as JPEG for audit
        if raw_frame is not None:
            import cv2

            frame_path = out_dir / f"{record.frame_id}.jpg"
            cv2.imwrite(str(frame_path), raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            data["raw_frame_path"] = str(frame_path)

        json_path.write_text(json.dumps(data, indent=2, default=str))
