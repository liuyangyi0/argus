"""Model drift detection using Kolmogorov-Smirnov test.

Compares rolling window of recent anomaly scores against the reference
distribution from training/calibration time. Alerts when distributions
diverge significantly.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import structlog
from scipy.stats import ks_2samp

logger = structlog.get_logger()


@dataclass
class DriftStatus:
    """Current drift detection status."""

    is_drifted: bool = False
    ks_statistic: float = 0.0
    p_value: float = 1.0
    reference_mean: float = 0.0
    current_mean: float = 0.0
    samples_collected: int = 0
    last_check_time: float = 0.0


class DriftDetector:
    """Detects score distribution drift using KS test.

    Usage:
        detector = DriftDetector(reference_scores)
        for each frame:
            detector.update(anomaly_score)
        status = detector.get_status()
    """

    def __init__(
        self,
        reference_scores: np.ndarray | None = None,
        window_size: int = 5000,
        check_interval: int = 500,
        ks_threshold: float = 0.1,
        p_value_threshold: float = 0.01,
        on_drift_callback: callable | None = None,
    ):
        self._reference = np.sort(reference_scores) if reference_scores is not None else None
        self._window: deque[float] = deque(maxlen=window_size)
        self._check_interval = check_interval
        self._ks_threshold = ks_threshold
        self._p_value_threshold = p_value_threshold
        self._on_drift = on_drift_callback
        self._count_since_check = 0
        self._status = DriftStatus()
        if self._reference is not None:
            self._status.reference_mean = float(self._reference.mean())

    def set_reference(self, scores: np.ndarray) -> None:
        """Set reference distribution (from calibration)."""
        self._reference = np.sort(scores)
        self._status.reference_mean = float(scores.mean())

    def update(self, score: float) -> None:
        """Feed a new anomaly score. Periodically runs KS test."""
        self._window.append(score)
        self._count_since_check += 1

        if self._count_since_check >= self._check_interval and len(self._window) >= 100:
            self._run_check()
            self._count_since_check = 0

    def _run_check(self) -> None:
        """Run KS test between reference and current window."""
        if self._reference is None or len(self._window) < 100:
            return

        current = np.array(self._window)
        ks_stat, p_value = self._ks_2samp(self._reference, current)

        was_drifted = self._status.is_drifted
        self._status = DriftStatus(
            is_drifted=(ks_stat > self._ks_threshold and p_value < self._p_value_threshold),
            ks_statistic=float(ks_stat),
            p_value=float(p_value),
            reference_mean=float(self._reference.mean()),
            current_mean=float(current.mean()),
            samples_collected=len(self._window),
            last_check_time=time.time(),
        )

        # Log and fire the callback only on state transitions. Previously
        # this ran on every check (roughly every check_interval samples),
        # producing identical log lines whenever drift persisted — a
        # single training run could emit 800+ duplicate warnings.
        if self._status.is_drifted and not was_drifted:
            logger.warning(
                "drift.detected",
                ks=round(ks_stat, 4),
                p=round(p_value, 6),
                ref_mean=round(self._status.reference_mean, 4),
                cur_mean=round(self._status.current_mean, 4),
            )
            if self._on_drift is not None:
                try:
                    self._on_drift(self._status)
                except Exception as e:
                    logger.error("drift.callback_failed", error=str(e))
        elif not self._status.is_drifted and was_drifted:
            logger.info(
                "drift.cleared",
                ks=round(ks_stat, 4),
                ref_mean=round(self._status.reference_mean, 4),
                cur_mean=round(self._status.current_mean, 4),
            )

    @staticmethod
    def _ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        """KS two-sample test via scipy.

        Returns (ks_statistic, p_value).
        """
        result = ks_2samp(a, b)
        return float(result.statistic), float(result.pvalue)

    def get_status(self) -> DriftStatus:
        return self._status
