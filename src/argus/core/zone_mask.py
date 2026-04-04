"""Polygon-based zone masking engine.

Manages include (detection) and exclude (ignore) zones as polygon masks.
Exclude zones black out regions that should never trigger detection — e.g.,
rotating fans, blinking indicator lights, or areas with known benign motion.

The combined mask is precomputed on initialization so per-frame application
is a single cv2.bitwise_and operation (<0.5ms overhead).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from argus.config.schema import ZoneConfig


class ZoneMaskEngine:
    """Applies include/exclude polygon masks to frames.

    If only exclude zones are defined, the entire frame is included
    except the excluded areas. If include zones are defined, only
    those areas are active. Exclude zones always override include zones
    where they overlap.
    """

    def __init__(
        self,
        zones: list[ZoneConfig],
        frame_height: int,
        frame_width: int,
    ):
        self._zones = list(zones)
        self._frame_h = frame_height
        self._frame_w = frame_width
        self._lock = threading.Lock()
        self._mask = self._build_mask(zones)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply the zone mask to a frame. Excluded regions become black."""
        with self._lock:
            mask = self._mask
        if mask is None:
            return frame
        return cv2.bitwise_and(frame, frame, mask=mask)

    def update_zones(self, zones: list[ZoneConfig]) -> None:
        """Hot-update zones without restarting the pipeline."""
        new_mask = self._build_mask(zones)
        with self._lock:
            self._zones = list(zones)
            self._mask = new_mask

    def get_include_zones(self) -> list[ZoneConfig]:
        """Return active include zones."""
        with self._lock:
            return [z for z in self._zones if z.zone_type == "include"]

    def get_exclude_zones(self) -> list[ZoneConfig]:
        """Return active exclude zones."""
        with self._lock:
            return [z for z in self._zones if z.zone_type == "exclude"]

    @property
    def has_zones(self) -> bool:
        with self._lock:
            return len(self._zones) > 0

    def _build_mask(self, zones: list[ZoneConfig]) -> np.ndarray | None:
        """Precompute the combined binary mask from zone polygons."""
        if not zones:
            return None

        include_zones = [z for z in zones if z.zone_type == "include"]
        exclude_zones = [z for z in zones if z.zone_type == "exclude"]

        h, w = self._frame_h, self._frame_w

        if include_zones:
            # Start with all-black (nothing included), then fill include zones white
            mask = np.zeros((h, w), dtype=np.uint8)
            for zone in include_zones:
                pts = np.array(zone.polygon, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
        else:
            # No include zones: start with all-white (everything included)
            mask = np.full((h, w), 255, dtype=np.uint8)

        # Exclude zones always punch holes (override includes)
        for zone in exclude_zones:
            pts = np.array(zone.polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 0)

        # If the resulting mask is all-white, return None (no masking needed)
        if cv2.countNonZero(mask) == h * w:
            return None

        return mask
