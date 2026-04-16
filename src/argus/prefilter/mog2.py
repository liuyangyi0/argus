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

        # Use CUDA MOG2 when available (requires OpenCV compiled with CUDA)
        self._use_cuda_mog2 = False
        try:
            if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self._subtractor = cv2.cuda.createBackgroundSubtractorMOG2(
                    history=history,
                    varThreshold=var_threshold,
                    detectShadows=detect_shadows,
                )
                self._use_cuda_mog2 = True
                logger.info("mog2.cuda_enabled")
        except Exception:
            pass

        if not self._use_cuda_mog2:
            self._subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=detect_shadows,
            )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Reusable GpuMat + stream to avoid per-frame allocation overhead
        self._gpu_frame: object | None = None
        self._gpu_fg: object | None = None
        self._cuda_stream: object | None = None
        if self._use_cuda_mog2:
            self._gpu_frame = cv2.cuda_GpuMat()
            self._gpu_fg = cv2.cuda_GpuMat()
            self._cuda_stream = cv2.cuda.Stream()

        # Phase correlation stabilization state
        self._prev_gray: np.ndarray | None = None
        self._hanning: np.ndarray | None = None

    # Target width for the downscaled phase-correlation input.
    # FFT cost is O(N·log N); shrinking from 1280x720 to 320x180 is ~16x
    # faster (~2ms vs 32ms) while retaining sub-pixel vibration detection.
    _ALIGN_WIDTH = 320

    def _align_frame(self, frame: np.ndarray) -> np.ndarray:
        """Compensate camera micro-vibration using phase correlation.

        Computes sub-pixel translation between consecutive frames in the
        frequency domain (FFT).  The correlation is performed on a
        downscaled copy to keep latency under 3ms even at high resolution;
        the resulting shift is scaled back to original coordinates.

        Only compensates small shifts (<5px) that indicate vibration,
        not genuine scene motion.
        """
        h, w = frame.shape[:2]

        # Resize BEFORE cvtColor — processes 16x fewer pixels for gray conversion
        if w > self._ALIGN_WIDTH:
            scale = self._ALIGN_WIDTH / w
            small_h = int(h * scale)
            small = cv2.resize(frame, (self._ALIGN_WIDTH, small_h),
                               interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            scale = 1.0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self._hanning is None or self._hanning.shape != gray.shape:
            self._hanning = cv2.createHanningWindow(
                (gray.shape[1], gray.shape[0]), cv2.CV_32F,
            )

        if self._prev_gray is None:
            self._prev_gray = gray
            return frame

        (dx, dy), _response = cv2.phaseCorrelate(
            self._prev_gray, gray, self._hanning,
        )

        self._prev_gray = gray

        if not (np.isfinite(dx) and np.isfinite(dy)):
            self._prev_gray = None
            return frame

        # Scale shift back to original resolution
        dx /= scale
        dy /= scale

        if abs(dx) < 5.0 and abs(dy) < 5.0 and (abs(dx) > 0.3 or abs(dy) > 0.3):
            M = np.float64([[1, 0, -dx], [0, 1, -dy]])
            frame = cv2.warpAffine(
                frame, M, (w, h),
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
        if self._use_cuda_mog2:
            self._gpu_frame.upload(frame)
            self._subtractor.apply(
                self._gpu_frame, lr, self._gpu_fg, self._cuda_stream,
            )
            fg_mask = self._gpu_fg.download()
        else:
            fg_mask = self._subtractor.apply(frame, learningRate=lr)

        # Shadow pixels are marked as 127 by MOG2; treat only 255 as foreground
        binary_mask = (fg_mask == 255).astype(np.uint8) * 255

        # Chromaticity-based shadow suppression: large foreground regions that
        # preserve chromaticity (hue) are shadows, not real objects. This handles
        # crane shadows that MOG2's built-in shadow model may miss.
        if cv2.countNonZero(binary_mask) > 0 and len(frame.shape) == 3:
            binary_mask = self._suppress_chromaticity_shadows(frame, binary_mask)

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

    @staticmethod
    def _suppress_chromaticity_shadows(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Suppress shadow regions that preserve hue (chromaticity).

        Real shadows darken a region without changing its hue. Foreign objects
        introduce new colours. Low-saturation foreground pixels are treated as
        shadows or lighting changes. Blue-dominant pixels are treated as
        Cherenkov glow fluctuations.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]

        low_sat = s_channel < 40
        # Suppress: foreground pixels that are low-saturation (grey/shadow)
        shadow_suppress = mask.copy()
        shadow_suppress[low_sat] = 0

        # Also suppress blue-dominant regions (Cherenkov glow: peaked at 300-400nm → blue)
        b, g, r = cv2.split(frame)
        cherenkov_mask = (b.astype(np.int16) - np.maximum(g, r).astype(np.int16)) > 30
        shadow_suppress[cherenkov_mask] = 0

        return shadow_suppress

    def reset(self) -> None:
        """Reset the background model (e.g., after baseline update)."""
        h = self._subtractor.getHistory()
        vt = self._subtractor.getVarThreshold()
        ds = self._subtractor.getDetectShadows()
        if self._use_cuda_mog2:
            self._subtractor = cv2.cuda.createBackgroundSubtractorMOG2(
                history=h, varThreshold=vt, detectShadows=ds,
            )
        else:
            self._subtractor = cv2.createBackgroundSubtractorMOG2(
                history=h, varThreshold=vt, detectShadows=ds,
            )
