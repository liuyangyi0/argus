"""Phase-1 pixel-level speed estimation for tracked anomalies.

Converts the temporal tracker's per-frame pixel displacement into
calibrated speed metrics.  Phase 1 works without full camera calibration
— an optional ``pixel_scale_mm_per_px`` provides approximate real-world
speed.  Phase 2 (future) will use the full calibration matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from argus.core.temporal_tracker import TrackedAnomaly

logger = structlog.get_logger()


@dataclass
class SpeedEstimate:
    """Speed estimate for a tracked anomaly."""

    speed_px_per_sec: float
    vx_px_per_sec: float
    vy_px_per_sec: float
    direction_deg: float  # 0=up, 90=right, 180=down, 270=left
    speed_ms: float | None = None
    is_falling: bool = False
    matches_freefall: bool = False
    smoothed: bool = False


class PixelSpeedEstimator:
    """Estimate object speed from temporal tracker data.

    Phase 1: converts pixel-per-frame velocity to pixel-per-second
    using the camera FPS.  Optionally applies a simple pixel→mm scale
    factor for approximate m/s output.

    Parameters
    ----------
    fps:
        Camera frames per second (used to convert px/frame → px/sec).
    smoothing_window:
        Number of trajectory-history points for moving-average smoothing.
    pixel_scale_mm_per_px:
        Optional mm-per-pixel factor for approximate real-world speed.
    """

    def __init__(
        self,
        fps: float = 15.0,
        smoothing_window: int = 5,
        pixel_scale_mm_per_px: float | None = None,
        calibration: object | None = None,
    ) -> None:
        self._fps = max(fps, 1.0)
        self._smoothing_window = max(smoothing_window, 1)
        self._pixel_scale = pixel_scale_mm_per_px
        self._calibration = calibration  # CameraCalibration instance (phase 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, track: TrackedAnomaly) -> SpeedEstimate:
        """Compute speed estimate for a tracked anomaly."""

        history = getattr(track, "trajectory_history", [])

        if len(history) >= self._smoothing_window + 1:
            vx, vy = self._smooth_velocity(history)
            smoothed = True
        else:
            dx, dy = track.velocity
            vx = dx * self._fps
            vy = dy * self._fps
            smoothed = False

        speed_px = math.hypot(vx, vy)

        # Direction in degrees (0=up, clockwise)
        direction = math.degrees(math.atan2(vx, -vy)) % 360.0

        # Real-world speed: prefer calibration, fall back to pixel_scale
        speed_ms: float | None = None
        if self._calibration is not None and hasattr(self._calibration, "pixel_distance_to_mm"):
            # Phase 2: use full calibration for accurate m/s
            cx, cy = track.centroid_x, track.centroid_y
            dx_frame, dy_frame = track.velocity
            dist_mm = self._calibration.pixel_distance_to_mm(
                cx, cy, cx + dx_frame, cy + dy_frame,
            )
            speed_ms = dist_mm * self._fps / 1000.0  # mm/frame * fps → m/s
        elif self._pixel_scale is not None and self._pixel_scale > 0:
            speed_ms = speed_px * self._pixel_scale / 1000.0  # mm→m

        is_falling = self._detect_falling(vy)
        matches_ff = self._check_freefall_acceleration(history, self._pixel_scale)

        return SpeedEstimate(
            speed_px_per_sec=speed_px,
            vx_px_per_sec=vx,
            vy_px_per_sec=vy,
            direction_deg=direction,
            speed_ms=speed_ms,
            is_falling=is_falling,
            matches_freefall=matches_ff,
            smoothed=smoothed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _smooth_velocity(
        self, history: list[tuple[float, float, float]]
    ) -> tuple[float, float]:
        """Moving-average velocity from the last *smoothing_window* points."""

        window = history[-self._smoothing_window - 1 :]
        if len(window) < 2:
            return 0.0, 0.0

        total_vx = 0.0
        total_vy = 0.0
        count = 0
        for i in range(1, len(window)):
            t0, x0, y0 = window[i - 1]
            t1, x1, y1 = window[i]
            dt = t1 - t0
            if dt <= 0:
                continue
            total_vx += (x1 - x0) / dt
            total_vy += (y1 - y0) / dt
            count += 1

        if count == 0:
            return 0.0, 0.0
        return total_vx / count, total_vy / count

    def _detect_falling(self, vy_px_per_sec: float) -> bool:
        """Positive vy in image coordinates means downward movement."""
        return vy_px_per_sec > 10.0  # small threshold to ignore noise

    def _check_freefall_acceleration(
        self,
        history: list[tuple[float, float, float]],
        pixel_scale: float | None,
    ) -> bool:
        """Check whether vertical acceleration matches gravity.

        Needs at least 3 trajectory points and a pixel scale to convert
        pixel acceleration into m/s².  Returns ``False`` when calibration
        is unavailable.
        """

        if pixel_scale is None or pixel_scale <= 0 or len(history) < 3:
            return False

        # Compute acceleration from 3 evenly-spaced recent points
        n = len(history)
        p0 = history[max(0, n - 3)]
        p1 = history[max(0, n - 2)]
        p2 = history[n - 1]

        dt1 = p1[0] - p0[0]
        dt2 = p2[0] - p1[0]
        if dt1 <= 0 or dt2 <= 0:
            return False

        vy1 = (p1[2] - p0[2]) / dt1  # px/sec
        vy2 = (p2[2] - p1[2]) / dt2
        dt_avg = (dt1 + dt2) / 2.0
        if dt_avg <= 0:
            return False

        accel_px = (vy2 - vy1) / dt_avg  # px/sec²
        accel_ms2 = accel_px * pixel_scale / 1000.0  # m/s²

        # Match gravity within ±30 %
        g = 9.81
        return abs(accel_ms2 - g) / g < 0.30
