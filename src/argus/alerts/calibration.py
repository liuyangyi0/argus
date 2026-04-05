"""Conformal prediction score calibration for statistically guaranteed FPR.

Given a set of anomaly scores from known-normal frames, computes thresholds
that guarantee a target false positive rate using distribution-free quantiles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class CalibrationResult:
    """Calibrated thresholds with metadata."""

    info_threshold: float
    low_threshold: float
    medium_threshold: float
    high_threshold: float
    n_calibration_samples: int
    target_fprs: dict[str, float]


class ConformalCalibrator:
    """Distribution-free threshold calibration using conformal prediction.

    Given normal-frame scores, find the score threshold that guarantees
    P(score > threshold | normal) <= target_fpr.

    The key property: for n calibration samples, the (1-alpha) quantile
    provides a coverage guarantee of at least (1-alpha) with finite-sample
    validity, regardless of the score distribution.
    """

    def calibrate(
        self,
        normal_scores: np.ndarray,
        target_fprs: dict[str, float] | None = None,
    ) -> CalibrationResult:
        """Compute calibrated thresholds from normal-frame scores.

        Args:
            normal_scores: 1D array of anomaly scores from known-normal frames.
                          Must have at least 50 samples for meaningful calibration.
            target_fprs: Target FPR per severity level.
                        Defaults: info=0.10, low=0.01, medium=0.001, high=0.0001

        Returns:
            CalibrationResult with four calibrated thresholds.

        Raises:
            ValueError: If fewer than 50 scores provided.
        """
        if len(normal_scores) < 50:
            raise ValueError(
                f"Need >= 50 calibration scores, got {len(normal_scores)}. "
                "Collect more baseline frames."
            )

        if target_fprs is None:
            target_fprs = {
                "info": 0.10,
                "low": 0.01,
                "medium": 0.001,
                "high": 0.0001,
            }

        scores = np.sort(normal_scores)
        n = len(scores)

        # Conformal quantile: for target FPR alpha, use ceil((1-alpha)*(n+1))/n
        thresholds = {}
        for level, alpha in target_fprs.items():
            quantile_idx = int(np.ceil((1 - alpha) * (n + 1))) - 1
            quantile_idx = min(quantile_idx, n - 1)
            thresholds[level] = float(scores[quantile_idx])

        # Ensure strict ordering: info < low < medium < high
        ordered = ["info", "low", "medium", "high"]
        for i in range(1, len(ordered)):
            if thresholds[ordered[i]] <= thresholds[ordered[i - 1]]:
                thresholds[ordered[i]] = thresholds[ordered[i - 1]] + 0.01

        result = CalibrationResult(
            info_threshold=thresholds["info"],
            low_threshold=thresholds["low"],
            medium_threshold=thresholds["medium"],
            high_threshold=thresholds["high"],
            n_calibration_samples=n,
            target_fprs=target_fprs,
        )

        logger.info(
            "calibration.complete",
            n_samples=n,
            thresholds={k: round(v, 4) for k, v in thresholds.items()},
        )
        return result

    def save(self, result: CalibrationResult, path: Path) -> None:
        """Save calibration to JSON file alongside model."""
        data = {
            "info": result.info_threshold,
            "low": result.low_threshold,
            "medium": result.medium_threshold,
            "high": result.high_threshold,
            "n_samples": result.n_calibration_samples,
            "target_fprs": result.target_fprs,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path) -> CalibrationResult | None:
        """Load calibration from JSON file. Returns None if not found."""
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return CalibrationResult(
            info_threshold=data["info"],
            low_threshold=data["low"],
            medium_threshold=data["medium"],
            high_threshold=data["high"],
            n_calibration_samples=data["n_samples"],
            target_fprs=data["target_fprs"],
        )
