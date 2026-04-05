"""Model A/B comparison for TRN-008.

Compares two anomaly detection models on the same validation set,
measuring both detection quality (anomaly scores on normal images)
and inference latency. Lower scores on normal images = fewer false positives.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import structlog

from argus.anomaly.detector import AnomalibDetector

logger = structlog.get_logger()


@dataclass
class ComparisonResult:
    """Result of comparing two anomaly detection models."""

    model_a_path: str
    model_b_path: str
    model_a_scores: list[float]  # anomaly scores on validation set
    model_b_scores: list[float]
    model_a_mean: float
    model_b_mean: float
    model_a_std: float
    model_b_std: float
    model_a_max: float
    model_b_max: float
    model_a_p95: float
    model_b_p95: float
    model_a_latency_ms: float  # average inference latency
    model_b_latency_ms: float
    winner: str  # "A", "B", or "tie"
    reason: str  # explanation of why winner was chosen


class ModelComparator:
    """Compare two anomaly detection models on the same validation set.

    Winner criteria (lower is better for normal images):
    1. Lower max score (fewer false positives)
    2. Lower mean score (more confident on normal images)
    3. If scores similar (within 5%): prefer lower latency
    """

    SIMILARITY_THRESHOLD = 0.05  # 5% relative difference = "similar"

    def compare(
        self,
        model_a_path: Path,
        model_b_path: Path,
        val_dir: Path,
        image_size: tuple[int, int] = (256, 256),
    ) -> ComparisonResult:
        """Run both models on validation images and compare scores + latency."""
        images = self._load_images(val_dir, image_size)
        if not images:
            raise ValueError(f"No images found in {val_dir}")

        logger.info(
            "model_compare.start",
            model_a=str(model_a_path),
            model_b=str(model_b_path),
            num_images=len(images),
        )

        # Run model A
        detector_a = AnomalibDetector(
            model_path=model_a_path,
            image_size=image_size,
        )
        detector_a.load()
        scores_a, latency_a = self._benchmark(detector_a, images)

        # Run model B
        detector_b = AnomalibDetector(
            model_path=model_b_path,
            image_size=image_size,
        )
        detector_b.load()
        scores_b, latency_b = self._benchmark(detector_b, images)

        # Compute statistics
        stats_a = self._compute_stats(scores_a)
        stats_b = self._compute_stats(scores_b)

        # Determine winner
        winner, reason = self._pick_winner(stats_a, stats_b, latency_a, latency_b)

        result = ComparisonResult(
            model_a_path=str(model_a_path),
            model_b_path=str(model_b_path),
            model_a_scores=scores_a,
            model_b_scores=scores_b,
            model_a_mean=stats_a["mean"],
            model_b_mean=stats_b["mean"],
            model_a_std=stats_a["std"],
            model_b_std=stats_b["std"],
            model_a_max=stats_a["max"],
            model_b_max=stats_b["max"],
            model_a_p95=stats_a["p95"],
            model_b_p95=stats_b["p95"],
            model_a_latency_ms=latency_a,
            model_b_latency_ms=latency_b,
            winner=winner,
            reason=reason,
        )

        logger.info(
            "model_compare.done",
            winner=winner,
            reason=reason,
            a_max=round(stats_a["max"], 4),
            b_max=round(stats_b["max"], 4),
            a_latency=round(latency_a, 2),
            b_latency=round(latency_b, 2),
        )

        return result

    def _load_images(
        self, val_dir: Path, image_size: tuple[int, int]
    ) -> list[np.ndarray]:
        """Load and resize all images from validation directory."""
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        images = []
        for p in sorted(val_dir.iterdir()):
            if p.suffix.lower() in extensions:
                img = cv2.imread(str(p))
                if img is not None:
                    img = cv2.resize(img, image_size)
                    images.append(img)
        return images

    def _benchmark(
        self, detector: AnomalibDetector, images: list[np.ndarray]
    ) -> tuple[list[float], float]:
        """Run detector on all images, collect scores and average latency."""
        scores: list[float] = []
        total_time = 0.0

        for img in images:
            t0 = time.perf_counter()
            result = detector.predict(img)
            t1 = time.perf_counter()
            scores.append(result.anomaly_score)
            total_time += t1 - t0

        avg_latency_ms = (total_time / len(images)) * 1000.0 if images else 0.0
        return scores, avg_latency_ms

    def _compute_stats(self, scores: list[float]) -> dict[str, float]:
        """Compute summary statistics from a list of scores."""
        arr = np.array(scores)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "max": float(np.max(arr)),
            "p95": float(np.percentile(arr, 95)),
        }

    def _pick_winner(
        self,
        stats_a: dict[str, float],
        stats_b: dict[str, float],
        latency_a: float,
        latency_b: float,
    ) -> tuple[str, str]:
        """Determine winner based on max score, mean score, and latency.

        Returns (winner, reason) where winner is "A", "B", or "tie".
        """
        max_a, max_b = stats_a["max"], stats_b["max"]
        mean_a, mean_b = stats_a["mean"], stats_b["mean"]

        # Criterion 1: Lower max score (fewer false positives)
        if max_b > 0 and abs(max_a - max_b) / max(max_b, 1e-9) > self.SIMILARITY_THRESHOLD:
            if max_a < max_b:
                return "A", f"Lower max score ({max_a:.4f} vs {max_b:.4f})"
            return "B", f"Lower max score ({max_b:.4f} vs {max_a:.4f})"

        # Criterion 2: Lower mean score
        if mean_b > 0 and abs(mean_a - mean_b) / max(mean_b, 1e-9) > self.SIMILARITY_THRESHOLD:
            if mean_a < mean_b:
                return "A", f"Lower mean score ({mean_a:.4f} vs {mean_b:.4f})"
            return "B", f"Lower mean score ({mean_b:.4f} vs {mean_a:.4f})"

        # Criterion 3: Scores are similar — prefer lower latency
        if latency_a < latency_b:
            return "A", f"Similar scores, lower latency ({latency_a:.1f}ms vs {latency_b:.1f}ms)"
        if latency_b < latency_a:
            return "B", f"Similar scores, lower latency ({latency_b:.1f}ms vs {latency_a:.1f}ms)"

        return "tie", "Models are equivalent in scores and latency"
