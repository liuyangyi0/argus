"""Frame quality filtering for baseline capture.

Filters out low-quality frames (blurry, overexposed, duplicates, persons,
encoder errors) before saving to ensure clean training data.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import cv2
import numpy as np
import structlog

from argus.config.schema import CaptureQualityConfig

logger = structlog.get_logger()


@dataclass
class FrameQualityResult:
    """Result of a single frame quality check."""

    accepted: bool
    rejection_reason: str | None = None  # blur, exposure, duplicate, person, encoder_error
    blur_score: float = 0.0
    brightness_mean: float = 0.0
    brightness_std: float = 0.0
    ssim_to_prev: float | None = None
    entropy: float = 0.0


@dataclass
class CaptureStats:
    """Accumulates statistics across an entire capture session."""

    total_grabbed: int = 0
    null_frames: int = 0
    rejected_blur: int = 0
    rejected_exposure: int = 0
    rejected_duplicate: int = 0
    rejected_person: int = 0
    rejected_encoder: int = 0
    accepted: int = 0
    brightness_values: list[float] = field(default_factory=list)

    @property
    def total_rejected(self) -> int:
        return (
            self.rejected_blur
            + self.rejected_exposure
            + self.rejected_duplicate
            + self.rejected_person
            + self.rejected_encoder
        )

    def brightness_range(self) -> tuple[float, float]:
        if not self.brightness_values:
            return (0.0, 0.0)
        return (min(self.brightness_values), max(self.brightness_values))

    def record_rejection(self, reason: str) -> None:
        attr = f"rejected_{reason}"
        if hasattr(self, attr):
            setattr(self, attr, getattr(self, attr) + 1)

    def to_dict(self) -> dict:
        bmin, bmax = self.brightness_range()
        return {
            "total_grabbed": self.total_grabbed,
            "null_frames": self.null_frames,
            "accepted": self.accepted,
            "total_rejected": self.total_rejected,
            "rejected_blur": self.rejected_blur,
            "rejected_exposure": self.rejected_exposure,
            "rejected_duplicate": self.rejected_duplicate,
            "rejected_person": self.rejected_person,
            "rejected_encoder": self.rejected_encoder,
            "brightness_min": round(bmin, 1),
            "brightness_max": round(bmax, 1),
        }


class FrameQualityFilter:
    """Runs a chain of quality checks on captured frames.

    Checks are ordered cheapest-to-most-expensive and short-circuit on
    the first rejection.
    """

    def __init__(self, config: CaptureQualityConfig) -> None:
        self._config = config
        self._blur_scores: list[float] = []
        self._person_detector = None

    def check(
        self, frame: np.ndarray, prev_frame: np.ndarray | None = None
    ) -> FrameQualityResult:
        """Run all quality checks on a frame.

        Args:
            frame: BGR image (np.ndarray).
            prev_frame: Previous accepted frame for dedup comparison.

        Returns:
            FrameQualityResult with acceptance status and metrics.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute metrics we'll need regardless
        entropy = self._compute_entropy(gray)
        mean_val = float(gray.mean())
        std_val = float(gray.std())
        blur_score = self._compute_blur_score(gray)

        result = FrameQualityResult(
            accepted=True,
            blur_score=blur_score,
            brightness_mean=mean_val,
            brightness_std=std_val,
            entropy=entropy,
        )

        if not self._config.enabled:
            return result

        # 1. Encoder error detection (CAP-009) — cheapest
        if entropy < self._config.entropy_min:
            result.accepted = False
            result.rejection_reason = "encoder_error"
            return result

        # 2. Exposure check (CAP-004)
        if not self._check_exposure(gray, mean_val, std_val):
            result.accepted = False
            result.rejection_reason = "exposure"
            return result

        # 3. Blur detection (CAP-002)
        if not self._check_blur(blur_score):
            result.accepted = False
            result.rejection_reason = "blur"
            return result

        # 4. Frame deduplication (CAP-005)
        if prev_frame is not None:
            ssim_val = self._compute_ssim(gray, prev_frame)
            result.ssim_to_prev = ssim_val
            if ssim_val >= self._config.ssim_dedup_threshold:
                result.accepted = False
                result.rejection_reason = "duplicate"
                return result

        # 5. Person detection (CAP-003) — most expensive
        if self._check_person(frame):
            result.accepted = False
            result.rejection_reason = "person"
            return result

        # Frame accepted — record blur score for adaptive threshold
        self._blur_scores.append(blur_score)
        return result

    # ── Individual checks ──

    @staticmethod
    def _compute_entropy(gray: np.ndarray) -> float:
        """Shannon entropy of grayscale histogram."""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        total = hist.sum()
        if total == 0:
            return 0.0
        probs = hist[hist > 0] / total
        return float(-np.sum(probs * np.log2(probs)))

    @staticmethod
    def _compute_blur_score(gray: np.ndarray) -> float:
        """Laplacian variance — higher means sharper."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _check_exposure(
        self, gray: np.ndarray, mean_val: float, std_val: float
    ) -> bool:
        """Returns True if exposure is acceptable."""
        cfg = self._config
        if mean_val < cfg.brightness_min or mean_val > cfg.brightness_max:
            return False
        if std_val < cfg.brightness_std_min:
            return False
        # Saturated pixel check
        total_pixels = gray.size
        dark = int(np.count_nonzero(gray < 5))
        bright = int(np.count_nonzero(gray > 250))
        saturated_pct = (dark + bright) / total_pixels
        if saturated_pct > cfg.saturated_pixel_max_pct:
            return False
        return True

    def _check_blur(self, blur_score: float) -> bool:
        """Returns True if frame is sharp enough."""
        threshold = self._config.blur_threshold
        # Adaptive threshold after 10 accepted frames
        if len(self._blur_scores) >= 10:
            median_score = statistics.median(self._blur_scores)
            adaptive_floor = median_score * self._config.blur_adaptive_pct
            threshold = max(adaptive_floor, threshold)
        return blur_score >= threshold

    def _compute_ssim(self, gray: np.ndarray, prev_frame: np.ndarray) -> float:
        """SSIM between current and previous frame at reduced resolution."""
        size = self._config.ssim_resize
        a = cv2.resize(gray, (size, size))
        # prev_frame may be BGR or gray
        if len(prev_frame.shape) == 3:
            b = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            b = prev_frame
        b = cv2.resize(b, (size, size))

        from skimage.metrics import structural_similarity

        return float(structural_similarity(a, b, data_range=255))

    def _check_person(self, frame: np.ndarray) -> bool:
        """Returns True if persons are detected in the frame."""
        if self._person_detector is None:
            try:
                from argus.person.detector import YOLOPersonDetector

                self._person_detector = YOLOPersonDetector(
                    confidence=self._config.person_confidence,
                    skip_frame_on_person=True,
                )
            except Exception:
                logger.warning("quality.person_detector_unavailable")
                # Mark as permanently unavailable
                self._person_detector = False  # type: ignore[assignment]
                return False

        if self._person_detector is False:
            return False

        try:
            result = self._person_detector.detect(frame)
            return result.has_persons
        except Exception:
            logger.warning("quality.person_detection_failed", exc_info=True)
            return False
