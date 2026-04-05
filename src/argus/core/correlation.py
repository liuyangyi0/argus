"""Cross-camera anomaly correlation.

When cameras have overlapping fields of view, an anomaly seen by one
camera but not corroborated by another is likely a false positive
(e.g., lens-specific contamination, single-camera glare).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class CameraOverlapPair:
    """A pair of cameras with overlapping fields of view."""

    camera_a: str
    camera_b: str
    homography: list[list[float]]


@dataclass
class CorrelationResult:
    """Result of cross-camera correlation check."""

    corroborated: bool = True
    partner_camera: str | None = None
    partner_score_at_location: float = 0.0


class CrossCameraCorrelator:
    """Correlates anomalies across overlapping camera pairs.

    Usage:
    1. Configure camera pairs with homography matrices
    2. Feed anomaly results from each camera
    3. Query correlation before emitting alert
    """

    def __init__(self, pairs: list[CameraOverlapPair]):
        self._pairs: dict[str, list[tuple[str, np.ndarray]]] = {}
        for pair in pairs:
            H = np.array(pair.homography, dtype=np.float64)
            self._pairs.setdefault(pair.camera_a, []).append((pair.camera_b, H))
            H_inv = np.linalg.inv(H)
            self._pairs.setdefault(pair.camera_b, []).append((pair.camera_a, H_inv))

        self._recent_maps: dict[str, tuple[float, np.ndarray]] = {}

    def update(self, camera_id: str, anomaly_map: np.ndarray | None, timestamp: float) -> None:
        """Store the latest anomaly map for a camera."""
        if anomaly_map is not None:
            self._recent_maps[camera_id] = (timestamp, anomaly_map)

    def check(
        self,
        camera_id: str,
        anomaly_location: tuple[int, int],
        timestamp: float,
        max_age_seconds: float = 5.0,
    ) -> CorrelationResult:
        """Check if an anomaly is corroborated by partner cameras."""
        partners = self._pairs.get(camera_id, [])
        if not partners:
            return CorrelationResult()

        src_point = np.array([[anomaly_location]], dtype=np.float64)

        for partner_id, H in partners:
            partner_data = self._recent_maps.get(partner_id)
            if partner_data is None:
                continue

            partner_time, partner_map = partner_data
            if timestamp - partner_time > max_age_seconds:
                continue

            dst_point = cv2.perspectiveTransform(src_point, H)
            px, py = int(dst_point[0, 0, 0]), int(dst_point[0, 0, 1])

            h, w = partner_map.shape[:2]
            if 0 <= px < w and 0 <= py < h:
                r = 25
                y1, y2 = max(0, py - r), min(h, py + r)
                x1, x2 = max(0, px - r), min(w, px + r)
                region_score = float(partner_map[y1:y2, x1:x2].max())

                if region_score > 0.3:
                    return CorrelationResult(
                        corroborated=True,
                        partner_camera=partner_id,
                        partner_score_at_location=region_score,
                    )

        return CorrelationResult(
            corroborated=False,
            partner_camera=partners[0][0] if partners else None,
            partner_score_at_location=0.0,
        )

    def prune_stale(self, max_age: float = 30.0) -> None:
        """Remove stale anomaly maps."""
        now = time.time()
        stale = [k for k, (t, _) in self._recent_maps.items() if now - t > max_age]
        for k in stale:
            del self._recent_maps[k]
