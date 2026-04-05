"""Adaptive anomaly detection threshold.

Automatically adjusts the anomaly detection threshold based on recent score
statistics using an exponentially weighted moving average (EWMA). This prevents
fixed thresholds from causing excessive false positives (threshold too low) or
missed detections (threshold too high) as environmental conditions change.

The threshold is clamped within a bounded range around the base threshold to
prevent dangerous drift.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import structlog

logger = structlog.get_logger()


@dataclass
class ThresholdState:
    """Current state of the adaptive threshold."""

    base_threshold: float
    current_threshold: float
    score_mean: float
    score_std: float
    score_p95: float
    sample_count: int
    last_update_frame: int


class AdaptiveThreshold:
    """Automatically adjust anomaly detection threshold based on recent scores.

    Uses exponentially weighted moving average (EWMA) of score statistics
    to adapt the threshold within a bounded range around the base threshold.

    Prevents threshold drift by clamping to [base - max_delta, base + max_delta].
    """

    def __init__(
        self,
        base_threshold: float = 0.7,
        max_delta: float = 0.1,
        window_size: int = 300,
        update_interval: int = 50,
        ewma_alpha: float = 0.1,
    ):
        self._base = base_threshold
        self._max_delta = max_delta
        self._window_size = window_size
        self._update_interval = update_interval
        self._alpha = ewma_alpha
        self._scores: deque[float] = deque(maxlen=window_size)
        self._current = base_threshold
        self._frame_count = 0
        self._last_update_frame = 0
        self._score_mean = 0.0
        self._score_std = 0.0
        self._score_p95 = 0.0

    def record_score(self, score: float) -> None:
        """Record a new anomaly score from a processed frame."""
        if not math.isfinite(score):
            return
        self._scores.append(score)
        self._frame_count += 1

        if self._frame_count - self._last_update_frame >= self._update_interval:
            self._recompute()

    def get_threshold(self) -> float:
        """Get the current adaptive threshold."""
        return self._current

    def _recompute(self) -> None:
        """Recompute threshold from accumulated scores.

        new_threshold = P95 of scores + margin
        margin = max(2.0 * std, 0.05)
        Clamp to [base - max_delta, base + max_delta]
        EWMA smooth: current = alpha * new + (1-alpha) * current
        """
        if len(self._scores) < 2:
            return

        scores = sorted(self._scores)
        n = len(scores)

        # Compute statistics
        total = sum(scores)
        mean = total / n
        variance = sum((s - mean) ** 2 for s in scores) / n
        std = math.sqrt(variance)

        # P95
        p95_idx = int(n * 0.95)
        p95_idx = min(p95_idx, n - 1)
        p95 = scores[p95_idx]

        # New threshold candidate
        margin = max(2.0 * std, 0.05)
        new_threshold = p95 + margin

        # Clamp to bounded range
        lower = self._base - self._max_delta
        upper = self._base + self._max_delta
        new_threshold = max(lower, min(new_threshold, upper))

        # EWMA smoothing
        self._current = self._alpha * new_threshold + (1.0 - self._alpha) * self._current

        # Clamp final result as well
        self._current = max(lower, min(self._current, upper))

        # Store stats
        self._score_mean = mean
        self._score_std = std
        self._score_p95 = p95
        self._last_update_frame = self._frame_count

        logger.debug(
            "adaptive_threshold.recomputed",
            current=round(self._current, 4),
            p95=round(p95, 4),
            mean=round(mean, 4),
            std=round(std, 4),
            samples=n,
        )

    def get_state(self) -> ThresholdState:
        """Get current threshold state for dashboard display."""
        return ThresholdState(
            base_threshold=self._base,
            current_threshold=self._current,
            score_mean=self._score_mean,
            score_std=self._score_std,
            score_p95=self._score_p95,
            sample_count=len(self._scores),
            last_update_frame=self._last_update_frame,
        )

    def reset(self) -> None:
        """Reset to base threshold."""
        self._scores.clear()
        self._current = self._base
        self._frame_count = 0
        self._last_update_frame = 0
        self._score_mean = 0.0
        self._score_std = 0.0
        self._score_p95 = 0.0
