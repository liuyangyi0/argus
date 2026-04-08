"""Shadow inference runner for model evaluation.

Runs a candidate model in parallel with production. Logs scores but
NEVER triggers alerts. Used during the shadow stage of the release pipeline.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import structlog

from argus.storage.models import ShadowInferenceLog

logger = structlog.get_logger()

_BATCH_FLUSH_SIZE = 50


class ShadowRunner:
    """Runs shadow model inference alongside production for comparison."""

    def __init__(
        self,
        shadow_model_path: Path,
        shadow_version_id: str,
        production_version_id: str | None,
        camera_id: str,
        session_factory,
        sample_rate: int = 5,
        threshold: float = 0.5,
    ):
        self._shadow_model_path = shadow_model_path
        self._shadow_version_id = shadow_version_id
        self._production_version_id = production_version_id
        self._camera_id = camera_id
        self._session_factory = session_factory
        self._sample_rate = max(1, sample_rate)
        self._threshold = threshold

        self._frame_counter = 0
        self._lock = threading.Lock()
        self._pending_logs: list[ShadowInferenceLog] = []
        self._detector = None  # Lazy-loaded
        self._load_failures = 0
        self._max_load_retries = 3
        self._retry_cooldown_frames = 500  # frames between retries
        self._frames_since_last_failure = 0

    def _ensure_loaded(self) -> bool:
        """Lazy-load the shadow model on first use, with limited retries."""
        if self._detector is not None:
            return True
        if self._load_failures >= self._max_load_retries:
            return False

        # Cooldown: wait N frames between retries after first failure
        if self._load_failures > 0:
            self._frames_since_last_failure += 1
            if self._frames_since_last_failure < self._retry_cooldown_frames:
                return False
            self._frames_since_last_failure = 0

        try:
            from argus.anomaly.detector import AnomalibDetector

            self._detector = AnomalibDetector(
                model_path=self._shadow_model_path,
                threshold=self._threshold,
            )
            logger.info(
                "shadow_runner.loaded",
                camera_id=self._camera_id,
                shadow_version=self._shadow_version_id,
            )
            return True
        except Exception as e:
            self._load_failures += 1
            self._frames_since_last_failure = 0
            logger.error(
                "shadow_runner.load_failed",
                error=str(e),
                attempt=self._load_failures,
                max_retries=self._max_load_retries,
            )
            return False

    def run_shadow(
        self,
        frame: np.ndarray,
        production_score: float | None = None,
        production_alerted: bool = False,
    ) -> None:
        """Run shadow inference if this is an Nth frame.

        This method is called from the inference loop and must be fast.
        Shadow results are buffered and flushed in batches.
        """
        with self._lock:
            self._frame_counter += 1
            if self._frame_counter % self._sample_rate != 0:
                return

        if not self._ensure_loaded():
            return

        try:
            t0 = time.perf_counter()
            result = self._detector.predict(frame)
            latency_ms = (time.perf_counter() - t0) * 1000

            shadow_score = result.anomaly_score if result else 0.0
            shadow_would_alert = shadow_score >= self._threshold

            log = ShadowInferenceLog(
                timestamp=datetime.now(timezone.utc),
                camera_id=self._camera_id,
                shadow_version_id=self._shadow_version_id,
                production_version_id=self._production_version_id,
                shadow_score=shadow_score,
                production_score=production_score,
                shadow_would_alert=shadow_would_alert,
                production_alerted=production_alerted,
                latency_ms=round(latency_ms, 1),
            )

            flush_batch = None
            with self._lock:
                self._pending_logs.append(log)
                if len(self._pending_logs) >= _BATCH_FLUSH_SIZE:
                    flush_batch = self._pending_logs[:]
                    self._pending_logs.clear()

            if flush_batch is not None:
                self._write_logs(flush_batch)

        except Exception as e:
            logger.warning("shadow_runner.predict_error", error=str(e))

    def flush(self) -> None:
        """Flush any remaining buffered logs to the database."""
        with self._lock:
            if not self._pending_logs:
                return
            batch = self._pending_logs[:]
            self._pending_logs.clear()
        self._write_logs(batch)

    def _write_logs(self, logs: list[ShadowInferenceLog]) -> None:
        """Write a batch of logs to the database (no lock held)."""
        try:
            with self._session_factory() as session:
                session.add_all(logs)
                session.commit()
            logger.debug(
                "shadow_runner.flushed",
                count=len(logs),
                camera_id=self._camera_id,
            )
        except Exception as e:
            logger.error("shadow_runner.flush_failed", error=str(e))

    @property
    def shadow_version_id(self) -> str:
        return self._shadow_version_id

    @property
    def frame_count(self) -> int:
        return self._frame_counter
