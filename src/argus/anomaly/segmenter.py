"""SAM 2 instance segmentation for anomaly regions (D2).

Given anomaly peak points from heatmap analysis, segments individual objects
to provide precise boundaries for anomaly classification and reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class SegmentedObject:
    """A segmented object instance."""

    mask: np.ndarray  # Binary mask (H, W)
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area_px: int
    score: float


@dataclass
class SegmentationResult:
    """Result from instance segmentation."""

    objects: list[SegmentedObject] = field(default_factory=list)
    num_objects: int = 0


class InstanceSegmenter:
    """SAM 2 based instance segmentation for anomaly regions.

    Requires the segment-anything-2 package to be installed.
    Falls back to contour-based segmentation if SAM 2 is not available.
    """

    def __init__(self, model_size: str = "small"):
        self._model_size = model_size
        self._model = None
        self._loaded = False
        self._use_fallback = False

    def load(self) -> None:
        """Load SAM 2 model (lazy initialization)."""
        if self._loaded:
            return
        try:
            # Try to import SAM 2
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # SAM 2 model loading would go here
            # For now, use fallback since SAM 2 may not be installed
            self._use_fallback = True
            self._loaded = True
            logger.info("segmenter.using_fallback", msg="SAM 2 not configured, using contour fallback")
        except ImportError:
            self._use_fallback = True
            self._loaded = True
            logger.info("segmenter.sam2_not_available", msg="Using contour-based fallback")

    def segment(self, frame: np.ndarray, prompt_points: list[tuple[int, int]]) -> SegmentationResult:
        """Given anomaly peak points, segment objects.

        Args:
            frame: BGR image.
            prompt_points: List of (x, y) points indicating anomaly peaks.

        Returns:
            SegmentationResult with segmented objects.
        """
        if not self._loaded:
            self.load()

        if self._use_fallback:
            return self._segment_contour_fallback(frame, prompt_points)

        # SAM 2 inference would go here
        return SegmentationResult()

    def _segment_contour_fallback(
        self, frame: np.ndarray, prompt_points: list[tuple[int, int]]
    ) -> SegmentationResult:
        """Contour-based fallback when SAM 2 is not available."""
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        objects = []

        for px, py in prompt_points:
            # Extract local region around prompt point
            h, w = gray.shape[:2]
            r = 50  # search radius
            y1, y2 = max(0, py - r), min(h, py + r)
            x1, x2 = max(0, px - r), min(w, px + r)
            region = gray[y1:y2, x1:x2]

            if region.size == 0:
                continue

            # Threshold and find contours
            _, mask = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Find largest contour
                largest = max(contours, key=cv2.contourArea)
                area = int(cv2.contourArea(largest))
                if area > 10:
                    bx, by, bw, bh = cv2.boundingRect(largest)
                    full_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(
                        full_mask[y1:y2, x1:x2],
                        [largest], -1, 255, -1,
                    )
                    objects.append(SegmentedObject(
                        mask=full_mask,
                        bbox=(x1 + bx, y1 + by, bw, bh),
                        area_px=area,
                        score=1.0,
                    ))

        return SegmentationResult(objects=objects, num_objects=len(objects))
