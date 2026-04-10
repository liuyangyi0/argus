"""Active learning uncertainty sampler for operator labeling.

Tracks prediction uncertainty (Shannon entropy) across frames and pushes
high-uncertainty frames to a labeling queue for operator annotation.
This closes the feedback loop: uncertain predictions → human labels →
retraining → improved model.

Integration: subscribes to FrameAnalyzed events via EventBus.
Publishes UncertainFrameDetected when entropy exceeds threshold.

Design: lightweight ring buffer + entropy calculator, no ML dependencies.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import structlog

from argus.core.event_bus import EventBus, FrameAnalyzed, UncertainFrameDetected, RetrainingTriggered

# Optional DB import — active learning works without persistence
try:
    from argus.storage.database import Database
    from argus.storage.models import LabelingQueueRecord
except ImportError:
    Database = None  # type: ignore[assignment,misc]
    LabelingQueueRecord = None  # type: ignore[assignment,misc]

logger = structlog.get_logger()


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning uncertainty sampling."""

    enabled: bool = False
    # Entropy threshold: frames above this get pushed to labeling queue
    entropy_threshold: float = 0.8
    # Minimum interval between labeling pushes per camera (seconds)
    min_push_interval: float = 60.0
    # Maximum frames in the labeling queue before dropping oldest
    max_queue_size: int = 200
    # Directory to save uncertain frames
    output_dir: str = "data/active_learning"
    # Score range for "uncertain" region: frames with scores in
    # [uncertain_low, uncertain_high] have highest entropy
    uncertain_low: float = 0.4
    uncertain_high: float = 0.7
    # Event-driven retrain trigger: fire after this many labels accumulated
    retrain_trigger_count: int = 50


@dataclass
class UncertainFrame:
    """A frame queued for operator labeling."""

    camera_id: str
    frame_number: int
    anomaly_score: float
    entropy: float
    frame_path: str
    timestamp: float = field(default_factory=time.time)


class ActiveLearningSampler:
    """Entropy-based uncertainty sampler for active learning.

    Subscribes to EventBus and monitors anomaly scores. When a frame
    falls in the uncertain region (near the decision boundary), it
    computes Shannon entropy and saves the frame for operator labeling.

    Entropy computation:
    - Binary classification entropy: H = -p*log2(p) - (1-p)*log2(1-p)
    - p = anomaly_score (treated as P(anomaly))
    - Maximum entropy at p=0.5 (most uncertain)
    - Frames near threshold have highest information gain
    """

    def __init__(
        self,
        config: ActiveLearningConfig,
        event_bus: EventBus | None = None,
        database: object | None = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._db: Database | None = database if Database is not None else None  # type: ignore[assignment]
        self._queue: deque[UncertainFrame] = deque(maxlen=config.max_queue_size)
        self._lock = threading.Lock()
        self._last_push_time: dict[str, float] = {}  # camera_id → last push timestamp
        self._output_dir = Path(config.output_dir)
        self._labeled_count = 0
        self._labeled_since_retrain = 0

        # Recent score buffer for per-camera entropy window
        self._score_buffers: dict[str, deque[float]] = {}
        self._buffer_size = 30  # rolling window for score distribution entropy

        # Frame buffer: pipeline writes current frame here before publishing
        # FrameAnalyzed, so the sampler can save it if uncertainty is high.
        # Key: "camera_id:frame_number" -> frame (np.ndarray)
        self._frame_buffer: dict[str, np.ndarray] = {}

        if event_bus and config.enabled:
            event_bus.subscribe(FrameAnalyzed, self._on_frame_analyzed)
            logger.info(
                "active_learning.initialized",
                entropy_threshold=config.entropy_threshold,
                push_interval=config.min_push_interval,
                db_persistence=self._db is not None,
            )

    @staticmethod
    def binary_entropy(p: float) -> float:
        """Shannon entropy for binary classification: H(p) = -p*log2(p) - (1-p)*log2(1-p).

        Returns value in [0, 1] where 1.0 = maximum uncertainty (p=0.5).
        """
        p = max(min(p, 0.9999), 0.0001)  # clamp for log safety
        return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    def score_distribution_entropy(self, camera_id: str, score: float) -> float:
        """Compute entropy from recent score distribution for a camera.

        Uses a rolling window of scores to estimate the local prediction
        uncertainty. High variance around the threshold → high entropy.
        """
        if camera_id not in self._score_buffers:
            self._score_buffers[camera_id] = deque(maxlen=self._buffer_size)

        buf = self._score_buffers[camera_id]
        buf.append(score)

        if len(buf) < 5:
            return self.binary_entropy(score)

        # Combine binary entropy with score variance in the window
        binary_h = self.binary_entropy(score)
        scores = list(buf)
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

        # High variance near threshold → higher uncertainty
        # Normalize variance to [0, 1] range (max variance for uniform [0,1] is 1/12)
        normalized_var = min(variance / 0.083, 1.0)

        # Weighted combination: 70% binary entropy + 30% distribution uncertainty
        return 0.7 * binary_h + 0.3 * normalized_var

    def _on_frame_analyzed(self, event: FrameAnalyzed) -> None:
        """EventBus handler: evaluate frame uncertainty and potentially queue it."""
        score = event.anomaly_score
        cfg = self._config

        # Only process frames in the uncertain zone
        if score < cfg.uncertain_low or score > cfg.uncertain_high:
            return

        # Compute entropy
        entropy = self.score_distribution_entropy(event.camera_id, score)

        if entropy < cfg.entropy_threshold:
            return

        # Rate limit per camera
        now = time.time()
        last_push = self._last_push_time.get(event.camera_id, 0.0)
        if now - last_push < cfg.min_push_interval:
            return

        self._last_push_time[event.camera_id] = now

        # Save frame and emit event (frame saving is done by the pipeline caller)
        frame_path = str(
            self._output_dir / event.camera_id / f"uncertain_{event.frame_number}.jpg"
        )

        uncertain = UncertainFrame(
            camera_id=event.camera_id,
            frame_number=event.frame_number,
            anomaly_score=score,
            entropy=entropy,
            frame_path=frame_path,
        )

        with self._lock:
            self._queue.append(uncertain)

        # Auto-save frame from buffer
        buf_key = f"{event.camera_id}:{event.frame_number}"
        buffered_frame = self._frame_buffer.pop(buf_key, None)
        if buffered_frame is not None:
            self.save_frame(event.camera_id, buffered_frame, event.frame_number)

        # Persist to DB for dashboard labeling queue
        self._persist_to_db(uncertain)

        # Publish event for downstream consumers (dashboard, notifications)
        if self._event_bus:
            self._event_bus.publish(UncertainFrameDetected(
                camera_id=event.camera_id,
                frame_number=event.frame_number,
                anomaly_score=score,
                entropy=entropy,
                frame_path=frame_path,
            ))

        logger.info(
            "active_learning.uncertain_frame",
            camera_id=event.camera_id,
            frame_number=event.frame_number,
            score=round(score, 4),
            entropy=round(entropy, 4),
        )

    def buffer_frame(self, camera_id: str, frame_number: int, frame: np.ndarray) -> None:
        """Buffer a frame for potential saving if it turns out to be uncertain.

        Called by the pipeline BEFORE publishing FrameAnalyzed.
        Only the latest frame per camera is kept to limit memory usage.
        """
        key = f"{camera_id}:{frame_number}"
        # Evict old entries for this camera
        old_keys = [k for k in self._frame_buffer if k.startswith(f"{camera_id}:")]
        for k in old_keys:
            del self._frame_buffer[k]
        self._frame_buffer[key] = frame

    def _persist_to_db(self, uncertain: UncertainFrame) -> None:
        """Persist uncertain frame to database labeling queue."""
        if self._db is None or LabelingQueueRecord is None:
            return
        try:
            record = LabelingQueueRecord(
                camera_id=uncertain.camera_id,
                frame_number=uncertain.frame_number,
                frame_path=uncertain.frame_path,
                anomaly_score=uncertain.anomaly_score,
                entropy=uncertain.entropy,
            )
            self._db.save_labeling_entry(record)
        except Exception:
            logger.warning(
                "active_learning.db_persist_failed",
                camera_id=uncertain.camera_id,
                frame_number=uncertain.frame_number,
                exc_info=True,
            )

    def save_frame(self, camera_id: str, frame: np.ndarray, frame_number: int) -> str | None:
        """Save an uncertain frame to disk for labeling.

        Called by the pipeline when a frame matches the uncertainty criteria.
        Returns the saved file path, or None if the frame wasn't in the queue.
        """
        with self._lock:
            match = None
            for uf in self._queue:
                if uf.camera_id == camera_id and uf.frame_number == frame_number:
                    match = uf
                    break
            if match is None:
                return None

        # Ensure output directory exists
        save_dir = self._output_dir / camera_id
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"uncertain_{frame_number}.jpg"

        cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return str(save_path)

    def get_queue(self, camera_id: str | None = None) -> list[UncertainFrame]:
        """Get queued uncertain frames, optionally filtered by camera."""
        with self._lock:
            if camera_id:
                return [uf for uf in self._queue if uf.camera_id == camera_id]
            return list(self._queue)

    def pop_labeled(
        self,
        camera_id: str,
        frame_number: int,
        label: str = "anomaly",
    ) -> UncertainFrame | None:
        """Remove a frame from the queue after operator labels it.

        Increments the labeled counter. When enough labels accumulate
        (>= retrain_threshold), publishes a retraining trigger event.
        """
        with self._lock:
            for i, uf in enumerate(self._queue):
                if uf.camera_id == camera_id and uf.frame_number == frame_number:
                    del self._queue[i]
                    self._labeled_count += 1
                    self._labeled_since_retrain += 1
                    break
            else:
                return None

        # Event-driven retrain trigger: fire when enough labels accumulated
        if self._labeled_since_retrain >= self._config.retrain_trigger_count:
            self._labeled_since_retrain = 0
            if self._event_bus:
                self._event_bus.publish(RetrainingTriggered(
                    reason="active_learning",
                    labeled_count=self._labeled_count,
                    cameras=list(self._score_buffers.keys()),
                ))
            logger.info(
                "active_learning.retrain_triggered",
                labeled_count=self._labeled_count,
                trigger_count=self._config.retrain_trigger_count,
            )

        return uf

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def labeled_count(self) -> int:
        return self._labeled_count

    def get_stats(self) -> dict:
        """Return active learning statistics for dashboard."""
        return {
            "enabled": self._config.enabled,
            "queue_size": len(self._queue),
            "max_queue_size": self._config.max_queue_size,
            "entropy_threshold": self._config.entropy_threshold,
            "cameras_tracked": len(self._score_buffers),
            "labeled_count": self._labeled_count,
            "labeled_since_retrain": self._labeled_since_retrain,
            "retrain_trigger_count": self._config.retrain_trigger_count,
        }
