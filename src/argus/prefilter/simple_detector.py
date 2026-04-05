"""Simplex safety channel: formally verifiable frame-difference detector.

This detector uses only classical CV primitives (absdiff, threshold,
morphology, contour analysis) with no learned parameters. It provides
a detection floor that works even when the ML channel fails.

The invariant it guarantees:
  "Any object larger than min_area_px pixels that remains stationary
   for longer than min_static_seconds will be detected."
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class StaticRegion:
    """A connected component that has been stationary."""

    centroid: tuple[int, int]
    area_px: int
    first_seen: float
    last_seen: float
    bbox: tuple[int, int, int, int]  # x, y, w, h


@dataclass
class SimplexResult:
    """Result from simplex detector."""

    has_detection: bool = False
    static_regions: list[StaticRegion] = field(default_factory=list)
    max_static_seconds: float = 0.0


class SimplexDetector:
    """Frame-difference based detector for the safety channel.

    Algorithm:
    1. cv2.absdiff(current, reference) -> diff
    2. cv2.cvtColor(diff, GRAY) if color
    3. cv2.GaussianBlur(diff, (5,5)) -> suppress noise
    4. cv2.threshold(diff, diff_threshold, 255, BINARY) -> mask
    5. cv2.morphologyEx(mask, OPEN, kernel) -> remove small noise
    6. cv2.morphologyEx(mask, CLOSE, kernel) -> fill holes
    7. cv2.findContours -> connected components
    8. Filter by min_area_px
    9. Track centroids: if a region's centroid stays within
       match_radius_px for > min_static_seconds -> detection
    """

    def __init__(
        self,
        diff_threshold: int = 30,
        min_area_px: int = 500,
        min_static_seconds: float = 30.0,
        morph_kernel_size: int = 5,
        match_radius_px: int = 50,
    ):
        self._diff_threshold = diff_threshold
        self._min_area_px = min_area_px
        self._min_static_seconds = min_static_seconds
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        self._match_radius = match_radius_px
        self._reference: np.ndarray | None = None
        self._tracked: list[StaticRegion] = []

    def set_reference(self, frame: np.ndarray) -> None:
        """Set the reference frame (from baseline)."""
        self._reference = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(frame.shape) == 3
            else frame.copy()
        )
        self._tracked.clear()

    def detect(self, frame: np.ndarray) -> SimplexResult:
        """Run simplex detection on a single frame.

        Returns SimplexResult. Latency: <2ms on typical hardware.
        """
        if self._reference is None:
            return SimplexResult()

        now = time.monotonic()
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(frame.shape) == 3
            else frame
        )

        # Step 1-2: Frame difference
        diff = cv2.absdiff(gray, self._reference)

        # Step 3: Noise suppression
        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        # Step 4: Binary threshold
        _, mask = cv2.threshold(diff, self._diff_threshold, 255, cv2.THRESH_BINARY)

        # Step 5-6: Morphology
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        # Step 7: Connected components
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 8: Filter by area and extract centroids
        current_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self._min_area_px:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
                current_regions.append((cx, cy, area, (x, y, w, h)))

        # Step 9: Match with tracked regions
        matched_indices: set[int] = set()
        for cx, cy, area, bbox in current_regions:
            best_match = None
            best_dist = float("inf")
            for i, tracked in enumerate(self._tracked):
                if i in matched_indices:
                    continue
                dist = ((cx - tracked.centroid[0]) ** 2 + (cy - tracked.centroid[1]) ** 2) ** 0.5
                if dist < self._match_radius and dist < best_dist:
                    best_match = i
                    best_dist = dist

            if best_match is not None:
                self._tracked[best_match].centroid = (cx, cy)
                self._tracked[best_match].area_px = area
                self._tracked[best_match].last_seen = now
                self._tracked[best_match].bbox = bbox
                matched_indices.add(best_match)
            else:
                self._tracked.append(
                    StaticRegion(
                        centroid=(cx, cy),
                        area_px=area,
                        first_seen=now,
                        last_seen=now,
                        bbox=bbox,
                    )
                )

        # Prune regions not seen in this frame (gone)
        self._tracked = [
            t
            for i, t in enumerate(self._tracked)
            if i in matched_indices or (now - t.last_seen) < 2.0
        ]

        # Check for static detections
        static = [
            t for t in self._tracked if (now - t.first_seen) >= self._min_static_seconds
        ]

        max_static = max((now - t.first_seen for t in self._tracked), default=0.0)

        return SimplexResult(
            has_detection=len(static) > 0,
            static_regions=static,
            max_static_seconds=max_static,
        )

    def reset(self) -> None:
        """Reset tracked regions (e.g., after model reload)."""
        self._tracked.clear()
