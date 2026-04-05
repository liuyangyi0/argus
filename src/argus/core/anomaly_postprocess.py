"""Anomaly map post-processing for cleaner detection boundaries.

Applies morphological operations, Gaussian smoothing, and contour filtering
to raw anomaly heatmaps. Extracts discrete anomaly regions with bounding
boxes, centroids, and scores for temporal tracking and alert annotation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AnomalyRegion:
    """A discrete anomaly region extracted from the heatmap."""

    x: int
    y: int
    width: int
    height: int
    area: int
    centroid_x: float
    centroid_y: float
    max_score: float
    mean_score: float


class AnomalyMapProcessor:
    """Post-process anomaly heatmaps for cleaner detection boundaries."""

    def __init__(
        self,
        morphology_kernel: int = 5,
        min_contour_area: int = 100,
        gaussian_sigma: float = 1.5,
    ):
        self._morphology_kernel = morphology_kernel
        self._min_contour_area = min_contour_area
        self._gaussian_sigma = gaussian_sigma

    def process(self, anomaly_map: np.ndarray) -> np.ndarray:
        """Apply morphological operations and contour filtering.

        Steps:
        1. Gaussian smoothing (reduce noise)
        2. Morphological close (fill small gaps in anomaly regions)
        3. Morphological open (remove small noise spots)
        4. Contour filtering (remove regions with area < min_contour_area)
        """
        if anomaly_map is None or anomaly_map.size == 0:
            return anomaly_map

        working = anomaly_map.astype(np.float32).copy()

        # Check if map is all zeros — skip processing
        if working.max() == 0:
            return working

        # 1. Gaussian smoothing
        ksize = max(3, int(self._gaussian_sigma * 4) | 1)  # ensure odd
        working = cv2.GaussianBlur(working, (ksize, ksize), self._gaussian_sigma)

        # 2. Morphological close (fill small gaps)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._morphology_kernel, self._morphology_kernel),
        )
        working = cv2.morphologyEx(working, cv2.MORPH_CLOSE, kernel)

        # 3. Morphological open (remove small noise)
        working = cv2.morphologyEx(working, cv2.MORPH_OPEN, kernel)

        # 4. Contour filtering — remove small regions
        # Threshold at a low level to find contours
        binary = (working > 0.01).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            mask = np.zeros_like(binary)
            large_contours = [c for c in contours if cv2.contourArea(c) >= self._min_contour_area]
            if large_contours:
                cv2.drawContours(mask, large_contours, -1, 255, thickness=cv2.FILLED)
            # Zero out regions that were filtered away
            working = working * (mask.astype(np.float32) / 255.0)

        return working

    def extract_regions(
        self, anomaly_map: np.ndarray, threshold: float = 0.5
    ) -> list[AnomalyRegion]:
        """Extract discrete anomaly regions from the heatmap.

        Returns list of AnomalyRegion with bounding box, area, centroid, max_score.
        Useful for temporal tracking and alert annotation.
        """
        if anomaly_map is None or anomaly_map.size == 0:
            return []

        working = anomaly_map.astype(np.float32)

        # Threshold the map
        binary = (working >= threshold).astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions: list[AnomalyRegion] = []
        for contour in contours:
            area = int(cv2.contourArea(contour))
            if area < self._min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Extract the region from the anomaly map for score computation
            region_mask = np.zeros_like(binary)
            cv2.drawContours(region_mask, [contour], -1, 255, thickness=cv2.FILLED)
            region_values = working[region_mask == 255]

            max_score = float(region_values.max()) if region_values.size > 0 else 0.0
            mean_score = float(region_values.mean()) if region_values.size > 0 else 0.0

            # Compute centroid using moments
            moments = cv2.moments(contour)
            if moments["m00"] > 0:
                centroid_x = moments["m10"] / moments["m00"]
                centroid_y = moments["m01"] / moments["m00"]
            else:
                centroid_x = x + w / 2.0
                centroid_y = y + h / 2.0

            regions.append(
                AnomalyRegion(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    area=area,
                    centroid_x=centroid_x,
                    centroid_y=centroid_y,
                    max_score=max_score,
                    mean_score=mean_score,
                )
            )

        # Sort by max_score descending (most anomalous first)
        regions.sort(key=lambda r: r.max_score, reverse=True)

        return regions
