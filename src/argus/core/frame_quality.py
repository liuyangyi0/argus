"""Frame quality assessment for anomaly detection pre-filtering.

Evaluates blur, exposure, noise, and entropy to determine whether a frame
is suitable for reliable anomaly detection. Low-quality frames receive a
confidence multiplier that reduces anomaly scores, cutting false positives
from degraded input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class QualityScore:
    """Frame quality assessment result."""

    acceptable: bool
    overall_score: float  # 0.0 (worst) to 1.0 (best)
    blur_score: float  # Laplacian variance normalized, higher = sharper
    exposure_score: float  # 0-1, 1 = well-exposed
    noise_score: float  # 0-1, 1 = low noise
    entropy: float  # Shannon entropy
    issues: list[str] = field(default_factory=list)


class FrameQualityAssessor:
    """Assess frame quality before anomaly detection.

    Low quality frames get a confidence multiplier applied to their
    anomaly scores, reducing false positives from bad input.
    """

    def __init__(
        self,
        blur_threshold: float = 50.0,
        entropy_min: float = 3.0,
        brightness_range: tuple[float, float] = (20.0, 240.0),
    ):
        self._blur_threshold = blur_threshold
        self._entropy_min = entropy_min
        self._brightness_low = brightness_range[0]
        self._brightness_high = brightness_range[1]

    def assess(self, frame: np.ndarray) -> QualityScore:
        """Evaluate frame quality. Returns QualityScore."""
        if frame is None or frame.size == 0:
            return QualityScore(
                acceptable=False,
                overall_score=0.0,
                blur_score=0.0,
                exposure_score=0.0,
                noise_score=0.0,
                entropy=0.0,
                issues=["empty frame"],
            )

        # Convert to grayscale
        if len(frame.shape) == 3 and frame.shape[2] >= 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif len(frame.shape) == 2:
            gray = frame
        else:
            gray = frame.reshape(frame.shape[0], frame.shape[1])

        gray = gray.astype(np.float64)

        issues: list[str] = []

        # 1. Blur detection — Laplacian variance, sigmoid normalization
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = 1.0 / (1.0 + math.exp(-(laplacian_var - self._blur_threshold) / (self._blur_threshold * 0.3)))
        if blur_score < 0.3:
            issues.append("image is blurry")

        # 2. Exposure — gray mean distance from ideal center
        mean_brightness = float(gray.mean())
        center = (self._brightness_low + self._brightness_high) / 2.0
        half_range = (self._brightness_high - self._brightness_low) / 2.0
        if half_range <= 0:
            exposure_score = 1.0
        else:
            distance = abs(mean_brightness - center) / half_range
            exposure_score = max(0.0, 1.0 - distance)
        if mean_brightness < self._brightness_low:
            issues.append("underexposed (too dark)")
        elif mean_brightness > self._brightness_high:
            issues.append("overexposed (too bright)")

        # 3. Noise estimation — high-pass filter energy ratio
        # Lower ratio of edge energy to total energy indicates noisier image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        high_pass = gray - blurred
        hp_energy = float(np.sum(high_pass ** 2))
        total_energy = float(np.sum(gray ** 2))
        if total_energy > 0:
            # A clean image with edges has moderate hp ratio;
            # a noisy image has very high hp ratio (noise everywhere)
            hp_ratio = hp_energy / total_energy
            # Invert: higher ratio = more noise = lower score
            # Typical clean image: ratio ~0.001-0.01; noisy: ~0.05+
            noise_score = 1.0 / (1.0 + math.exp((hp_ratio - 0.02) / 0.01))
        else:
            noise_score = 0.0
            issues.append("uniform black frame")

        # 4. Shannon entropy from gray histogram
        gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
        hist = cv2.calcHist([gray_uint8], [0], None, [256], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-12)
        nonzero = hist[hist > 0]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        if entropy < self._entropy_min:
            issues.append("low information content (low entropy)")

        # Normalize entropy to 0-1 (max theoretical = 8 bits)
        entropy_score = min(entropy / 8.0, 1.0)

        # Overall score — weighted average
        overall_score = (
            0.30 * blur_score
            + 0.25 * exposure_score
            + 0.20 * noise_score
            + 0.25 * entropy_score
        )
        overall_score = max(0.0, min(1.0, overall_score))

        acceptable = overall_score >= 0.4 and len(issues) <= 1

        return QualityScore(
            acceptable=acceptable,
            overall_score=overall_score,
            blur_score=blur_score,
            exposure_score=exposure_score,
            noise_score=noise_score,
            entropy=entropy,
            issues=issues,
        )

    def confidence_multiplier(self, quality: QualityScore) -> float:
        """Returns 0.0-1.0 multiplier to apply to anomaly scores.

        - Good quality (>0.8): 1.0 (no reduction)
        - Medium quality (0.5-0.8): 0.7-1.0 (proportional)
        - Poor quality (<0.5): 0.3-0.7 (heavy reduction)
        """
        score = quality.overall_score

        if score >= 0.8:
            return 1.0
        elif score >= 0.5:
            # Linear interpolation: 0.5 → 0.7, 0.8 → 1.0
            return 0.7 + (score - 0.5) / 0.3 * 0.3
        else:
            # Linear interpolation: 0.0 → 0.3, 0.5 → 0.7
            return 0.3 + score / 0.5 * 0.4
