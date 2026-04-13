"""Trajectory fitting and origin/landing point estimation.

Fits physics models (free-fall, projectile with air drag) to tracked
object positions and extrapolates to estimate the release origin and
water-surface landing point.

Two trajectory models are supported:

1. **Free-fall** — vertical drop with no horizontal velocity:
   ``z(t) = z₀ − ½gt²``

2. **Projectile with drag** — 3-D ballistic trajectory with linear air
   drag correction for small objects:
   ``z(t) = z₀ + vz₀·t − (m/k)[t − (m/k)(1 − e^{−kt/m})]``

The better-fitting model (by R²) is selected automatically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from argus.physics.calibration import CameraCalibration, WorldPoint

logger = structlog.get_logger()

# Gravity constant (m/s²)
_G = 9.81


@dataclass
class TrajectoryPoint:
    """A single point on the tracked trajectory in world coordinates."""

    timestamp: float  # seconds (absolute or relative)
    x_mm: float
    y_mm: float
    z_mm: float


@dataclass
class TrajectoryFit:
    """Result of fitting a physics model to trajectory data."""

    model_type: str  # "free_fall" or "projectile"
    r_squared: float  # goodness of fit (0–1)
    # Fitted initial conditions
    origin: TrajectoryPoint  # extrapolated release point (t=0)
    landing: TrajectoryPoint  # extrapolated water-surface impact
    # Fitted parameters
    initial_velocity_ms: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (vx, vy, vz) m/s
    drag_coefficient: float = 0.0  # effective k/m (1/s) for drag model
    # Residual error
    mean_residual_mm: float = 0.0
    num_points: int = 0
    # Predicted positions for validation
    predicted_positions: list[TrajectoryPoint] = field(default_factory=list)


class TrajectoryAnalyzer:
    """Fit physics models to tracked trajectory and estimate origin/landing.

    Parameters
    ----------
    gravity_ms2:
        Local gravitational acceleration.
    pool_surface_z_mm:
        Z coordinate of the water surface in world frame.
    min_points:
        Minimum trajectory points before fitting.
    use_drag_model:
        Whether to attempt drag-corrected model fitting.
    """

    def __init__(
        self,
        gravity_ms2: float = 9.81,
        pool_surface_z_mm: float = 0.0,
        min_points: int = 5,
        use_drag_model: bool = True,
    ) -> None:
        self._g = gravity_ms2
        self._pool_z = pool_surface_z_mm
        self._min_points = min_points
        self._use_drag = use_drag_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_trajectory(self, points: list[TrajectoryPoint]) -> TrajectoryFit | None:
        """Fit physics models to trajectory points and return the best fit.

        Returns None if there are fewer than *min_points* observations.
        """
        if len(points) < self._min_points:
            return None

        # Normalise timestamps to start at 0
        t0 = points[0].timestamp
        ts = np.array([p.timestamp - t0 for p in points])
        xs = np.array([p.x_mm for p in points])
        ys = np.array([p.y_mm for p in points])
        zs = np.array([p.z_mm for p in points])

        # Fit free-fall model
        ff_fit = self._fit_freefall(ts, xs, ys, zs)

        # Fit projectile model (parabolic, no drag)
        proj_fit = self._fit_projectile(ts, xs, ys, zs)

        # Fit drag-corrected model if enabled (for small objects where drag matters)
        drag_fit = None
        if self._use_drag and len(ts) >= 6:
            drag_fit = self._fit_drag_projectile(ts, xs, ys, zs)

        # Select the model with highest R²
        best = ff_fit
        for candidate in (proj_fit, drag_fit):
            if candidate is not None and (best is None or candidate.r_squared > best.r_squared):
                best = candidate

        if best is not None:
            best.num_points = len(points)
            logger.info(
                "trajectory.fit_complete",
                model=best.model_type,
                r_squared=round(best.r_squared, 4),
                points=len(points),
                residual_mm=round(best.mean_residual_mm, 1),
            )
        return best

    def estimate_origin(self, fit: TrajectoryFit) -> TrajectoryPoint:
        """Return the extrapolated release point (already stored in fit)."""
        return fit.origin

    def estimate_landing(self, fit: TrajectoryFit) -> TrajectoryPoint:
        """Return the extrapolated water-surface impact (already stored in fit)."""
        return fit.landing

    # ------------------------------------------------------------------
    # Free-fall model: z(t) = z0 - 0.5*g*t², x/y constant
    # ------------------------------------------------------------------

    def _fit_freefall(
        self,
        ts: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
    ) -> TrajectoryFit | None:
        """Fit a pure free-fall model (no horizontal velocity)."""
        if len(ts) < 3:
            return None

        # z(t) = z0 - 0.5*g*t²  →  z = z0 - 0.5*g*t²
        # Least-squares for z0: z + 0.5*g*t² = z0
        z_corrected = zs + 0.5 * self._g * (ts ** 2) * 1000.0  # g in m/s² → mm
        z0 = float(np.mean(z_corrected))
        x0 = float(np.mean(xs))
        y0 = float(np.mean(ys))

        # Predicted
        z_pred = z0 - 0.5 * self._g * 1000.0 * (ts ** 2)
        residuals = zs - z_pred
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((zs - np.mean(zs)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

        mean_res = float(np.mean(np.abs(residuals)))

        # Check if horizontal displacement is small (confirms free-fall)
        x_range = float(np.max(xs) - np.min(xs))
        y_range = float(np.max(ys) - np.min(ys))
        if x_range > 200 or y_range > 200:  # >200mm horizontal → likely projectile
            r_squared *= 0.5  # penalise

        # Origin at t=0
        origin = TrajectoryPoint(timestamp=0.0, x_mm=x0, y_mm=y0, z_mm=z0)

        # Landing: solve z0 - 0.5*g*t² = pool_z → t = sqrt(2*(z0-pool_z)/(g*1000))
        dz = z0 - self._pool_z
        g_scaled = self._g * 1000.0
        if dz > 1e-6 and g_scaled > 1e-6:
            t_land = math.sqrt(2.0 * dz / g_scaled)
        else:
            t_land = ts[-1]
        landing = TrajectoryPoint(
            timestamp=t_land,
            x_mm=x0,
            y_mm=y0,
            z_mm=self._pool_z,
        )

        return TrajectoryFit(
            model_type="free_fall",
            r_squared=max(r_squared, 0.0),
            origin=origin,
            landing=landing,
            initial_velocity_ms=(0.0, 0.0, 0.0),
            mean_residual_mm=mean_res,
        )

    # ------------------------------------------------------------------
    # Projectile model: 3-D ballistic with optional drag
    # ------------------------------------------------------------------

    def _fit_projectile(
        self,
        ts: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
    ) -> TrajectoryFit | None:
        """Fit a 3-D projectile model.

        x(t) = x0 + vx*t
        y(t) = y0 + vy*t
        z(t) = z0 + vz*t - 0.5*g*t²  (no drag)

        Uses least-squares regression per axis.
        """
        if len(ts) < 3:
            return None

        n = len(ts)

        # --- X axis: x(t) = x0 + vx*t ---
        A_x = np.column_stack([np.ones(n), ts])
        coeff_x, _, _, _ = np.linalg.lstsq(A_x, xs, rcond=None)
        x0, vx = float(coeff_x[0]), float(coeff_x[1])

        # --- Y axis: y(t) = y0 + vy*t ---
        A_y = np.column_stack([np.ones(n), ts])
        coeff_y, _, _, _ = np.linalg.lstsq(A_y, ys, rcond=None)
        y0, vy = float(coeff_y[0]), float(coeff_y[1])

        # --- Z axis: z(t) = z0 + vz*t - 0.5*g*t² ---
        # Rewrite: z + 0.5*g*t² = z0 + vz*t
        z_adj = zs + 0.5 * self._g * 1000.0 * (ts ** 2)
        A_z = np.column_stack([np.ones(n), ts])
        coeff_z, _, _, _ = np.linalg.lstsq(A_z, z_adj, rcond=None)
        z0, vz = float(coeff_z[0]), float(coeff_z[1])

        # Predicted values
        x_pred = x0 + vx * ts
        y_pred = y0 + vy * ts
        z_pred = z0 + vz * ts - 0.5 * self._g * 1000.0 * (ts ** 2)

        # 3-D residuals
        residuals = np.sqrt((xs - x_pred) ** 2 + (ys - y_pred) ** 2 + (zs - z_pred) ** 2)
        mean_res = float(np.mean(residuals))

        # R² for the Z axis (most important for trajectory)
        ss_res = float(np.sum((zs - z_pred) ** 2))
        ss_tot = float(np.sum((zs - np.mean(zs)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

        # Origin at t=0
        origin = TrajectoryPoint(timestamp=0.0, x_mm=x0, y_mm=y0, z_mm=z0)

        # Landing: solve z0 + vz*t - 0.5*g*1000*t² = pool_z
        # Quadratic: -0.5*g*1000*t² + vz*t + (z0 - pool_z) = 0
        a_coeff = -0.5 * self._g * 1000.0
        b_coeff = vz
        c_coeff = z0 - self._pool_z

        disc = b_coeff ** 2 - 4 * a_coeff * c_coeff
        if disc >= 0 and abs(a_coeff) > 1e-9:
            sqrt_disc = math.sqrt(disc)
            t1 = (-b_coeff + sqrt_disc) / (2 * a_coeff)
            t2 = (-b_coeff - sqrt_disc) / (2 * a_coeff)
            # Select smallest positive root (first impact)
            positive_roots = [t for t in [t1, t2] if t > 0]
            t_land = min(positive_roots) if positive_roots else ts[-1]
        else:
            t_land = ts[-1]

        landing = TrajectoryPoint(
            timestamp=max(t_land, 0.0),
            x_mm=x0 + vx * t_land,
            y_mm=y0 + vy * t_land,
            z_mm=self._pool_z,
        )

        # Convert velocities to m/s (from mm/s)
        vx_ms = vx / 1000.0
        vy_ms = vy / 1000.0
        vz_ms = vz / 1000.0

        return TrajectoryFit(
            model_type="projectile",
            r_squared=max(r_squared, 0.0),
            origin=origin,
            landing=landing,
            initial_velocity_ms=(vx_ms, vy_ms, vz_ms),
            mean_residual_mm=mean_res,
        )

    # ------------------------------------------------------------------
    # Drag-corrected projectile: z(t) = z0 + (vt/k)(1 - e^{-kt}) - (g/k)t + (g/k²)(1-e^{-kt})
    # where k = drag_coeff (1/s), vt = terminal velocity
    # ------------------------------------------------------------------

    def _fit_drag_projectile(
        self,
        ts: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
    ) -> TrajectoryFit | None:
        """Fit drag-corrected projectile for small objects where air resistance matters."""
        try:
            from scipy.optimize import curve_fit
        except ImportError:
            return None

        if len(ts) < 6:
            return None

        g_mm = self._g * 1000.0  # mm/s²

        # X and Y axes: linear (drag on horizontal assumed negligible for short falls)
        n = len(ts)
        A_x = np.column_stack([np.ones(n), ts])
        coeff_x, _, _, _ = np.linalg.lstsq(A_x, xs, rcond=None)
        x0, vx = float(coeff_x[0]), float(coeff_x[1])

        A_y = np.column_stack([np.ones(n), ts])
        coeff_y, _, _, _ = np.linalg.lstsq(A_y, ys, rcond=None)
        y0, vy = float(coeff_y[0]), float(coeff_y[1])

        # Z axis: drag model z(t) = z0 + vz0/k*(1 - e^{-kt}) - g/k*t + g/k²*(1-e^{-kt})
        def z_drag(t: np.ndarray, z0_: float, vz0_: float, k_: float) -> np.ndarray:
            k_ = max(k_, 0.01)  # prevent division by zero
            exp_kt = np.exp(-k_ * t)
            return z0_ + (vz0_ / k_) * (1 - exp_kt) - (g_mm / k_) * t + (g_mm / (k_ * k_)) * (1 - exp_kt)

        try:
            popt, _ = curve_fit(
                z_drag, ts, zs,
                p0=[zs[0], 0.0, 1.0],
                maxfev=2000,
                bounds=([zs[0] - 5000, -50000, 0.01], [zs[0] + 5000, 50000, 100.0]),
            )
        except (RuntimeError, ValueError):
            return None

        z0_fit, vz0_fit, k_fit = popt
        z_pred = z_drag(ts, z0_fit, vz0_fit, k_fit)

        ss_res = float(np.sum((zs - z_pred) ** 2))
        ss_tot = float(np.sum((zs - np.mean(zs)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

        x_pred = x0 + vx * ts
        y_pred = y0 + vy * ts
        residuals = np.sqrt((xs - x_pred) ** 2 + (ys - y_pred) ** 2 + (zs - z_pred) ** 2)
        mean_res = float(np.mean(residuals))

        origin = TrajectoryPoint(timestamp=0.0, x_mm=x0, y_mm=y0, z_mm=z0_fit)

        # Landing: numerically solve z_drag(t) = pool_z
        from scipy.optimize import brentq
        try:
            t_land = brentq(lambda t: z_drag(np.array([t]), z0_fit, vz0_fit, k_fit)[0] - self._pool_z,
                            0.001, 30.0)
        except ValueError:
            t_land = ts[-1]

        landing = TrajectoryPoint(
            timestamp=t_land,
            x_mm=x0 + vx * t_land,
            y_mm=y0 + vy * t_land,
            z_mm=self._pool_z,
        )

        return TrajectoryFit(
            model_type="projectile_drag",
            r_squared=max(r_squared, 0.0),
            origin=origin,
            landing=landing,
            initial_velocity_ms=(vx / 1000.0, vy / 1000.0, vz0_fit / 1000.0),
            drag_coefficient=k_fit,
            mean_residual_mm=mean_res,
        )


def trajectory_points_from_pixel_history(
    history: list[tuple[float, float, float]],
    calibration: CameraCalibration,
) -> list[TrajectoryPoint]:
    """Convert pixel-space trajectory history to world-coordinate TrajectoryPoints.

    Parameters
    ----------
    history:
        List of (timestamp, pixel_x, pixel_y) from TemporalAnomalyTracker.
    calibration:
        Camera calibration for pixel→world mapping.

    Returns
    -------
    List of TrajectoryPoint in world coordinates.
    """
    points: list[TrajectoryPoint] = []
    for ts, px, py in history:
        wp = calibration.pixel_to_world(px, py)
        points.append(TrajectoryPoint(
            timestamp=ts,
            x_mm=wp.x_mm,
            y_mm=wp.y_mm,
            z_mm=wp.z_mm,
        ))
    return points
