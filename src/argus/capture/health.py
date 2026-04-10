"""Camera hardware health monitoring.

Detects 5 classes of camera issues that would otherwise cause
systematic false positives or missed detections:
1. Frame freeze (decoder hang, TCP still connected)
2. Lens contamination / fogging
3. Mechanical displacement
4. Flash / arc welding suppression
5. Auto-gain / exposure drift
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class HealthCheckResult:
    """Result of camera health analysis."""

    is_frozen: bool = False
    sharpness_score: float = 0.0
    sharpness_baseline: float = 0.0
    displacement_px: float = 0.0
    is_flash: bool = False
    brightness_mean: float = 0.0
    gain_drift_pct: float = 0.0
    suppress_detection: bool = False
    warnings: list[str] = field(default_factory=list)


class CameraHealthAnalyzer:
    """Analyzes per-frame camera health metrics.

    Each check costs <1ms. Total overhead: ~3-5ms per frame.
    """

    def __init__(
        self,
        freeze_window: int = 10,
        freeze_hash_threshold: float = 0.01,
        sharpness_drop_pct: float = 0.3,
        displacement_threshold_px: float = 20.0,
        flash_sigma: float = 3.0,
        brightness_window: int = 300,
        gain_drift_threshold_pct: float = 20.0,
    ):
        self._freeze_window = freeze_window
        self._freeze_hash_threshold = freeze_hash_threshold
        self._sharpness_drop_pct = sharpness_drop_pct
        self._displacement_threshold = displacement_threshold_px
        self._flash_sigma = flash_sigma
        self._gain_drift_threshold = gain_drift_threshold_pct

        self._frame_hashes: deque[int] = deque(maxlen=freeze_window)
        self._brightness_history: deque[float] = deque(maxlen=brightness_window)

        self._sharpness_baseline: float | None = None
        self._brightness_baseline: float | None = None
        self._reference_gray: np.ndarray | None = None
        self._cumulative_dx: float = 0.0
        self._cumulative_dy: float = 0.0
        self._prev_gray: np.ndarray | None = None
        self._calibration_count = 0
        self._calibration_sharpness: list[float] = []
        self._calibration_brightness: list[float] = []

    def analyze(self, frame: np.ndarray) -> HealthCheckResult:
        """Run all health checks on a single frame. <5ms total."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        result = HealthCheckResult()

        # 1. Frame freeze detection (pHash variance)
        frame_hash = self._compute_phash(gray)
        self._frame_hashes.append(frame_hash)
        if len(self._frame_hashes) >= self._freeze_window:
            hash_variance = np.var([h for h in self._frame_hashes])
            result.is_frozen = hash_variance < self._freeze_hash_threshold
            if result.is_frozen:
                result.warnings.append("frame_frozen")
                result.suppress_detection = True

        # 2. Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())
        result.sharpness_score = sharpness

        # 3. Brightness
        brightness = float(gray.mean())
        result.brightness_mean = brightness
        self._brightness_history.append(brightness)

        # Calibration phase (first 30 frames)
        if self._calibration_count < 30:
            self._calibration_sharpness.append(sharpness)
            self._calibration_brightness.append(brightness)
            self._calibration_count += 1
            if self._calibration_count == 30:
                self._sharpness_baseline = float(np.median(self._calibration_sharpness))
                self._brightness_baseline = float(np.median(self._calibration_brightness))
                self._reference_gray = gray.copy()
                logger.info(
                    "camera_health.calibrated",
                    sharpness=round(self._sharpness_baseline, 1),
                    brightness=round(self._brightness_baseline, 1),
                )
            self._prev_gray = gray.copy()
            return result

        result.sharpness_baseline = self._sharpness_baseline

        # 2b. Lens contamination check
        if sharpness < self._sharpness_baseline * (1 - self._sharpness_drop_pct):
            result.warnings.append("lens_contamination")

        # 3b. Flash detection
        if len(self._brightness_history) >= 10:
            recent = list(self._brightness_history)[-60:]
            mean_b = np.mean(recent[:-1]) if len(recent) > 1 else brightness
            std_b = float(np.std(recent[:-1])) if len(recent) > 1 else 1.0
            if std_b > 0 and abs(brightness - mean_b) > self._flash_sigma * std_b:
                result.is_flash = True
                result.warnings.append("flash_detected")
                result.suppress_detection = True

        # 3c. Gain drift
        if self._brightness_baseline and self._brightness_baseline > 0:
            drift = abs(brightness - self._brightness_baseline) / self._brightness_baseline * 100
            result.gain_drift_pct = drift
            if drift > self._gain_drift_threshold:
                result.warnings.append("gain_drift")

        # 4. Mechanical displacement (phase correlation)
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            try:
                shift, _ = cv2.phaseCorrelate(
                    self._prev_gray.astype(np.float64),
                    gray.astype(np.float64),
                )
                dx, dy = shift
                if abs(dx) < 5 and abs(dy) < 5:
                    self._cumulative_dx += dx
                    self._cumulative_dy += dy
                displacement = (self._cumulative_dx**2 + self._cumulative_dy**2) ** 0.5
                result.displacement_px = displacement
                if displacement > self._displacement_threshold:
                    result.warnings.append("mechanical_displacement")
            except cv2.error:
                logger.debug("health.optical_flow_failed", exc_info=True)

        self._prev_gray = gray.copy()
        return result

    def _compute_phash(self, gray: np.ndarray) -> int:
        """Compute simple perceptual hash (8x8 DCT-based)."""
        resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        mean_val = resized.mean()
        return int(np.packbits((resized > mean_val).flatten())[0])

    def reset(self) -> None:
        """Reset all baselines (e.g., after camera recalibration)."""
        self._frame_hashes.clear()
        self._brightness_history.clear()
        self._sharpness_baseline = None
        self._brightness_baseline = None
        self._reference_gray = None
        self._cumulative_dx = 0.0
        self._cumulative_dy = 0.0
        self._prev_gray = None
        self._calibration_count = 0
        self._calibration_sharpness.clear()
        self._calibration_brightness.clear()
