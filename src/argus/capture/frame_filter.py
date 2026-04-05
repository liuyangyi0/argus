"""Frame quality filters for baseline capture (CAP-002~005, CAP-009).

Each filter is independently configurable and can be enabled/disabled.
Filters run in order: entropy -> exposure -> blur -> person -> dedup
(cheap filters first, expensive last).
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class FilterResult:
    """Result of frame quality evaluation."""

    accepted: bool
    reason: str = ""
    blur_score: float = 0.0
    brightness: float = 0.0
    entropy: float = 0.0
    ssim_score: float = 0.0
    person_count: int = 0


@dataclass
class FilterConfig:
    """Configuration for frame quality filters."""

    # CAP-002: Blur detection
    enable_blur_filter: bool = True
    blur_threshold: float = 100.0
    blur_adaptive: bool = True

    # CAP-003: Person detection
    enable_person_filter: bool = True
    person_confidence: float = 0.3

    # CAP-004: Exposure filtering
    enable_exposure_filter: bool = True
    brightness_min: float = 30.0
    brightness_max: float = 225.0
    saturation_max_ratio: float = 0.3
    min_std: float = 10.0

    # CAP-005: Frame deduplication
    enable_dedup_filter: bool = True
    dedup_ssim_threshold: float = 0.98

    # CAP-009: Encoder error / entropy
    enable_entropy_filter: bool = True
    entropy_min: float = 3.0


class FrameFilter:
    """Evaluates frame quality using a pipeline of configurable filters.

    Thread-safe: all mutable state is protected by a lock.
    """

    _BLUR_HISTORY_SIZE = 20
    _BLUR_ADAPTIVE_FACTOR = 0.3

    def __init__(
        self,
        config: FilterConfig | None = None,
        person_detector: object | None = None,
    ) -> None:
        self._config = config or FilterConfig()
        self._person_detector = person_detector
        self._lock = threading.Lock()
        self._blur_history: deque[float] = deque(maxlen=self._BLUR_HISTORY_SIZE)
        self._last_accepted_gray: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, frame: np.ndarray | None) -> FilterResult:
        """Run all enabled filters on *frame* and return a :class:`FilterResult`."""
        if frame is None or frame.size == 0:
            return FilterResult(accepted=False, reason="empty_frame")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        result = FilterResult(accepted=True)

        # Order: entropy -> exposure -> blur -> person -> dedup
        if self._config.enable_entropy_filter:
            if not self._check_entropy(gray, result):
                return result

        if self._config.enable_exposure_filter:
            if not self._check_exposure(gray, result):
                return result

        if self._config.enable_blur_filter:
            if not self._check_blur(gray, result):
                return result

        if self._config.enable_person_filter:
            self._check_persons(frame, result)

        if self._config.enable_dedup_filter:
            if not self._check_dedup(gray, result):
                return result

        # Frame accepted — store for future dedup comparison
        with self._lock:
            self._last_accepted_gray = cv2.resize(gray, (256, 256))

        return result

    def reset(self) -> None:
        """Reset internal state (blur history, last accepted frame for dedup)."""
        with self._lock:
            self._blur_history.clear()
            self._last_accepted_gray = None

    # ------------------------------------------------------------------
    # Private filter methods
    # ------------------------------------------------------------------

    def _check_entropy(self, gray: np.ndarray, result: FilterResult) -> bool:
        """CAP-009: Reject frames with Shannon entropy below threshold."""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        total = hist.sum()
        if total == 0:
            result.entropy = 0.0
            result.accepted = False
            result.reason = "entropy_too_low"
            return False

        probs = hist / total
        # Avoid log2(0) by filtering zeros
        nonzero = probs[probs > 0]
        entropy = -float(np.sum(nonzero * np.log2(nonzero)))
        result.entropy = entropy

        if entropy < self._config.entropy_min:
            result.accepted = False
            result.reason = "entropy_too_low"
            return False
        return True

    def _check_exposure(self, gray: np.ndarray, result: FilterResult) -> bool:
        """CAP-004: Reject under/over-exposed or uniform frames."""
        mean_val = float(gray.mean())
        std_val = float(gray.std())
        result.brightness = mean_val

        if mean_val < self._config.brightness_min:
            result.accepted = False
            result.reason = "too_dark"
            return False

        if mean_val > self._config.brightness_max:
            result.accepted = False
            result.reason = "too_bright"
            return False

        if std_val < self._config.min_std:
            result.accepted = False
            result.reason = "uniform_frame"
            return False

        total_pixels = gray.size
        saturated_high = int(np.count_nonzero(gray > 250))
        saturated_low = int(np.count_nonzero(gray < 5))
        saturated_ratio = (saturated_high + saturated_low) / total_pixels

        if saturated_ratio > self._config.saturation_max_ratio:
            result.accepted = False
            result.reason = "saturated"
            return False

        return True

    def _check_blur(self, gray: np.ndarray, result: FilterResult) -> bool:
        """CAP-002: Reject blurry frames using Laplacian variance."""
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        result.blur_score = variance

        with self._lock:
            threshold = self._config.blur_threshold

            if self._config.blur_adaptive and len(self._blur_history) >= 5:
                median_val = float(np.median(list(self._blur_history)))
                adaptive_threshold = median_val * self._BLUR_ADAPTIVE_FACTOR
                threshold = max(adaptive_threshold, threshold)

            self._blur_history.append(variance)

        if variance < threshold:
            result.accepted = False
            result.reason = "too_blurry"
            return False
        return True

    def _check_persons(self, frame: np.ndarray, result: FilterResult) -> None:
        """CAP-003: Detect persons in frame (informational, does not reject)."""
        if self._person_detector is None:
            return

        try:
            detections = self._person_detector(frame)
            count = 0
            # Support ultralytics-style results
            if hasattr(detections, "__iter__"):
                for det in detections:
                    if hasattr(det, "boxes"):
                        for box in det.boxes:
                            cls_id = int(box.cls[0]) if hasattr(box, "cls") else -1
                            conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                            if cls_id == 0 and conf >= self._config.person_confidence:
                                count += 1
            result.person_count = count
        except Exception:  # noqa: BLE001
            result.person_count = 0

    def _check_dedup(self, gray: np.ndarray, result: FilterResult) -> bool:
        """CAP-005: Reject near-duplicate frames via SSIM."""
        with self._lock:
            prev = self._last_accepted_gray

        if prev is None:
            # First frame — no duplicate possible; store later in evaluate()
            result.ssim_score = 0.0
            return True

        resized = cv2.resize(gray, (256, 256))
        ssim_val = self._compute_ssim(prev, resized)
        result.ssim_score = ssim_val

        if ssim_val >= self._config.dedup_ssim_threshold:
            result.accepted = False
            result.reason = "duplicate"
            return False
        return True

    # ------------------------------------------------------------------
    # SSIM helpers (no scipy dependency)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ssim(
        img1: np.ndarray,
        img2: np.ndarray,
        k1: float = 0.01,
        k2: float = 0.03,
        win_size: int = 7,
    ) -> float:
        """Compute mean SSIM between two single-channel images of equal size.

        Uses a uniform window (box filter) for speed.
        """
        L = 255.0  # noqa: N806
        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        mu1 = cv2.blur(img1, (win_size, win_size))
        mu2 = cv2.blur(img2, (win_size, win_size))

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.blur(img1 * img1, (win_size, win_size)) - mu1_sq
        sigma2_sq = cv2.blur(img2 * img2, (win_size, win_size)) - mu2_sq
        sigma12 = cv2.blur(img1 * img2, (win_size, win_size)) - mu1_mu2

        numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

        ssim_map = numerator / denominator
        return float(ssim_map.mean())
