"""Sub-pixel phase-correlation alignment for camera micro-vibration.

Nuclear plant cameras mounted near pumps, fans, and turbines experience
1-2px frame-to-frame shifts from mechanical vibration. MOG2 and PatchCore
cannot distinguish this from real motion, making camera shake the single
largest source of false positives.

This module aligns each incoming frame to a periodically refreshed
reference using ``cv2.phaseCorrelate``, which recovers translation in the
frequency domain with sub-pixel accuracy. Correlation is performed on a
downscaled grayscale copy (default 1/4) to keep per-frame cost under a
few milliseconds even at 1080p.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass(frozen=True)
class _AlignmentCfg(Protocol):
    enabled: bool
    max_shift_px: float
    downsample: int
    ref_update_interval_s: float


class PhaseCorrelator:
    """Sub-pixel translation alignment via FFT-based phase correlation.

    The first frame passed to :meth:`align` is retained as the reference;
    subsequent frames are measured against it and warped by the recovered
    ``(dx, dy)`` translation. The reference is refreshed every
    ``ref_update_interval_s`` seconds so slow scene drift does not pin the
    correlator to a stale background.

    Shifts whose magnitude exceeds ``max_shift_px`` on either axis are
    treated as genuine motion (not vibration) and returned uncorrected.
    """

    def __init__(
        self,
        max_shift_px: float = 5.0,
        downsample: int = 4,
        ref_update_interval_s: float = 60.0,
    ) -> None:
        if max_shift_px <= 0:
            raise ValueError(f"max_shift_px must be positive, got {max_shift_px}")
        if downsample < 1:
            raise ValueError(f"downsample must be >= 1, got {downsample}")
        if ref_update_interval_s <= 0:
            raise ValueError(
                f"ref_update_interval_s must be positive, got {ref_update_interval_s}"
            )

        self.max_shift_px = float(max_shift_px)
        self.downsample = int(downsample)
        self.ref_update_interval_s = float(ref_update_interval_s)

        self._ref_small: np.ndarray | None = None
        self._hanning: np.ndarray | None = None
        self._ref_timestamp: float = 0.0

    def reset(self) -> None:
        """Drop the current reference so the next call establishes a fresh one."""
        self._ref_small = None
        self._hanning = None
        self._ref_timestamp = 0.0

    def align(
        self, frame: np.ndarray, *, now: float | None = None,
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """Align ``frame`` to the current reference.

        Returns the aligned frame and the measured ``(dx, dy)`` shift in
        original-resolution pixels. When no reference is set (first call
        or stale reference) the frame is adopted as the new reference and
        returned unchanged with ``(0.0, 0.0)``.

        When the measured shift exceeds ``max_shift_px`` the frame is
        returned unchanged so genuine motion passes through untouched; the
        caller receives the original measurement for logging.
        """
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty ndarray")

        t = time.monotonic() if now is None else float(now)
        small = self._downsample_gray(frame)

        ref_stale = (
            self._ref_small is not None
            and (t - self._ref_timestamp) >= self.ref_update_interval_s
        )
        if self._ref_small is None or ref_stale:
            self._ref_small = small
            self._ref_timestamp = t
            if self._hanning is None or self._hanning.shape != small.shape:
                self._hanning = cv2.createHanningWindow(
                    (small.shape[1], small.shape[0]), cv2.CV_32F,
                )
            return frame, (0.0, 0.0)

        if self._ref_small.shape != small.shape:
            self._ref_small = small
            self._ref_timestamp = t
            self._hanning = cv2.createHanningWindow(
                (small.shape[1], small.shape[0]), cv2.CV_32F,
            )
            return frame, (0.0, 0.0)

        (dx_small, dy_small), _response = cv2.phaseCorrelate(
            self._ref_small, small, self._hanning,
        )
        if not (np.isfinite(dx_small) and np.isfinite(dy_small)):
            return frame, (0.0, 0.0)

        dx = float(dx_small) * self.downsample
        dy = float(dy_small) * self.downsample

        if abs(dx) > self.max_shift_px or abs(dy) > self.max_shift_px:
            logger.debug(
                "alignment.outlier_skip",
                dx=round(dx, 3),
                dy=round(dy, 3),
                max_shift=self.max_shift_px,
            )
            return frame, (dx, dy)

        h, w = frame.shape[:2]
        translation = np.float64([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
        aligned = cv2.warpAffine(
            frame, translation, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned, (dx, dy)

    def _downsample_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale float32 at 1/downsample resolution."""
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            raise ValueError(f"frame must be 2D or 3D, got shape {frame.shape}")

        if self.downsample > 1:
            h, w = gray.shape[:2]
            target = (max(1, w // self.downsample), max(1, h // self.downsample))
            gray = cv2.resize(gray, target, interpolation=cv2.INTER_AREA)
        return gray.astype(np.float32)


def create_from_config(cfg: _AlignmentCfg) -> PhaseCorrelator:
    """Instantiate a :class:`PhaseCorrelator` from an ``AlignmentConfig``-shaped object."""
    return PhaseCorrelator(
        max_shift_px=cfg.max_shift_px,
        downsample=cfg.downsample,
        ref_update_interval_s=cfg.ref_update_interval_s,
    )
