"""MOG2 background subtraction pre-filter.

Quickly determines whether a frame has meaningful changes compared to the
learned background model. Unchanged frames are skipped to save compute
on the downstream YOLO and Anomalib stages.

Includes optional phase-correlation stabilization to compensate for camera
micro-vibration common in nuclear plant environments (pumps, turbines).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class PreFilterResult:
    """Result of the pre-filter stage."""

    has_change: bool
    change_ratio: float  # fraction of pixels that changed (0.0 - 1.0)
    foreground_mask: np.ndarray | None = None  # binary mask of changed regions


class MOG2PreFilter:
    """Background subtraction pre-filter using MOG2.

    MOG2 models each pixel as a mixture of Gaussians and adaptively updates
    the background. When fewer than `change_pct_threshold` fraction of pixels
    differ from the background, the frame is considered unchanged.
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 25.0,
        detect_shadows: bool = True,
        change_pct_threshold: float = 0.005,
        learning_rate: float = -1,  # -1 = auto
        denoise: bool = True,
        enable_stabilization: bool = True,
    ):
        self.change_pct_threshold = change_pct_threshold
        self.learning_rate = learning_rate
        self.denoise = denoise
        self.enable_stabilization = enable_stabilization

        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Phase correlation stabilization state
        self._prev_gray: np.ndarray | None = None
        # Hanning window is pre-computed once and reused for phaseCorrelate
        self._hanning: np.ndarray | None = None

    def _align_frame(self, frame: np.ndarray) -> np.ndarray:
        """Compensate camera micro-vibration using phase correlation.

        Computes sub-pixel translation between consecutive frames in the
        frequency domain (FFT). Only compensates small shifts (<5px) that
        indicate vibration, not genuine scene motion.

        Args:
            frame: BGR image from camera.

        Returns:
            Stabilized frame (or original if stabilization is skipped).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.float64(gray)

        # Lazy-init Hanning window (must match frame dimensions)
        if self._hanning is None or self._hanning.shape != gray.shape:
            self._hanning = cv2.createHanningWindow(
                (gray.shape[1], gray.shape[0]), cv2.CV_64F,
            )

        if self._prev_gray is None:
            self._prev_gray = gray
            return frame

        (dx, dy), _response = cv2.phaseCorrelate(
            self._prev_gray, gray, self._hanning,
        )

        self._prev_gray = gray

        # MED-02: Validate phase correlation result (can return NaN/Inf on edge cases)
        if not (np.isfinite(dx) and np.isfinite(dy)):
            # Reset _prev_gray so the next frame re-initializes cleanly
            # instead of using stale reference that produced invalid correlation
            self._prev_gray = None
            return frame

        # Only compensate small vibration; large shifts are real motion
        if abs(dx) < 5.0 and abs(dy) < 5.0 and (abs(dx) > 0.3 or abs(dy) > 0.3):
            M = np.float64([[1, 0, -dx], [0, 1, -dy]])
            frame = cv2.warpAffine(
                frame, M, (frame.shape[1], frame.shape[0]),
                borderMode=cv2.BORDER_REPLICATE,
            )
            logger.debug(
                "stabilization.compensated",
                dx=round(dx, 2), dy=round(dy, 2),
            )

        return frame

    def process(self, frame: np.ndarray, learning_rate_override: float | None = None) -> PreFilterResult:
        """Apply background subtraction and determine if the frame has changed.

        Args:
            frame: BGR image from camera.
            learning_rate_override: If provided, override the default learning
                rate for this call. Use 0.0 to freeze the background model.

        Returns:
            PreFilterResult with change detection outcome.
        """
        # Phase correlation stabilization — compensate camera micro-vibration
        # HIGH-02: Skip stabilization when learning is frozen (lock active)
        # to avoid introducing shadow artifacts from alignment during freeze
        if self.enable_stabilization and learning_rate_override != 0.0:
            frame = self._align_frame(frame)

        # Median blur suppresses radiation-induced salt-and-pepper noise on CMOS
        # sensors in nuclear environments. 3x3 kernel adds <1ms latency.
        if self.denoise:
            frame = cv2.medianBlur(frame, 3)

        lr = learning_rate_override if learning_rate_override is not None else self.learning_rate
        fg_mask = self._subtractor.apply(frame, learningRate=lr)

        # Shadow pixels are marked as 127 by MOG2; treat only 255 as foreground
        binary_mask = (fg_mask == 255).astype(np.uint8) * 255

        # Morphological operations to reduce noise
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self._kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, self._kernel)

        total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
        change_pixels = cv2.countNonZero(binary_mask)
        change_ratio = change_pixels / total_pixels if total_pixels > 0 else 0.0

        has_change = change_ratio >= self.change_pct_threshold

        return PreFilterResult(
            has_change=has_change,
            change_ratio=change_ratio,
            foreground_mask=binary_mask if has_change else None,
        )

    def reset(self) -> None:
        """Reset the background model (e.g., after baseline update)."""
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self._subtractor.getHistory(),
            varThreshold=self._subtractor.getVarThreshold(),
            detectShadows=self._subtractor.getDetectShadows(),
        )
