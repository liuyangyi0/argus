"""Tests for the physics trajectory analysis module."""

from __future__ import annotations

import math

import pytest

from argus.physics.trajectory import TrajectoryAnalyzer, TrajectoryPoint


def _make_freefall_points(z0: float = 5000.0, g: float = 9.81, n: int = 20, dt: float = 0.05):
    """Generate synthetic free-fall trajectory points."""
    points = []
    for i in range(n):
        t = i * dt
        z = z0 - 0.5 * g * 1000.0 * t * t
        points.append(TrajectoryPoint(timestamp=t, x_mm=0.0, y_mm=0.0, z_mm=z))
    return points


def _make_projectile_points(
    x0=0.0, y0=0.0, z0=5000.0, vx=500.0, vy=0.0, vz=0.0,
    g=9.81, n=20, dt=0.05,
):
    """Generate synthetic projectile trajectory points."""
    points = []
    for i in range(n):
        t = i * dt
        x = x0 + vx * t
        y = y0 + vy * t
        z = z0 + vz * t - 0.5 * g * 1000.0 * t * t
        points.append(TrajectoryPoint(timestamp=t, x_mm=x, y_mm=y, z_mm=z))
    return points


class TestTrajectoryAnalyzer:
    def test_freefall_model_selection(self):
        """Pure free-fall data should select free_fall model."""
        analyzer = TrajectoryAnalyzer(gravity_ms2=9.81, pool_surface_z_mm=0.0, min_points=5)
        points = _make_freefall_points(z0=5000.0, n=15)
        fit = analyzer.fit_trajectory(points)

        assert fit is not None
        assert fit.model_type == "free_fall"
        assert fit.r_squared > 0.95

    def test_projectile_model_selection(self):
        """Data with horizontal velocity should prefer projectile model."""
        analyzer = TrajectoryAnalyzer(gravity_ms2=9.81, pool_surface_z_mm=0.0, min_points=5)
        points = _make_projectile_points(vx=1000.0, n=15)
        fit = analyzer.fit_trajectory(points)

        assert fit is not None
        assert fit.model_type == "projectile"

    def test_origin_estimation(self):
        """Origin should be close to the actual release point."""
        analyzer = TrajectoryAnalyzer(gravity_ms2=9.81, pool_surface_z_mm=0.0, min_points=5)
        points = _make_freefall_points(z0=5000.0, n=20)
        fit = analyzer.fit_trajectory(points)

        assert fit is not None
        origin = fit.origin
        assert origin.z_mm == pytest.approx(5000.0, abs=100.0)
        assert origin.x_mm == pytest.approx(0.0, abs=50.0)

    def test_landing_at_pool_surface(self):
        """Landing point should be at pool surface Z."""
        analyzer = TrajectoryAnalyzer(gravity_ms2=9.81, pool_surface_z_mm=0.0, min_points=5)
        points = _make_freefall_points(z0=5000.0, n=20)
        fit = analyzer.fit_trajectory(points)

        assert fit is not None
        landing = fit.landing
        assert landing.z_mm == pytest.approx(0.0, abs=1.0)

    def test_insufficient_points_returns_none(self):
        """With fewer than min_points, should return None."""
        analyzer = TrajectoryAnalyzer(min_points=10)
        points = _make_freefall_points(n=5)
        fit = analyzer.fit_trajectory(points)

        assert fit is None

    def test_identical_z_values_no_crash(self):
        """All points at same Z should not crash (ss_tot=0 edge case)."""
        analyzer = TrajectoryAnalyzer(min_points=3)
        points = [
            TrajectoryPoint(timestamp=i * 0.1, x_mm=i * 10.0, y_mm=0.0, z_mm=100.0)
            for i in range(5)
        ]
        fit = analyzer.fit_trajectory(points)
        # Should not crash; fit may have low R² but should not raise
        assert fit is not None or fit is None  # either outcome is acceptable

    def test_projectile_landing_positive_root(self):
        """Quadratic root selection should pick the smallest positive root."""
        analyzer = TrajectoryAnalyzer(gravity_ms2=9.81, pool_surface_z_mm=0.0, min_points=5)
        # Object starts at 3000mm, thrown upward slightly then falls
        points = _make_projectile_points(z0=3000.0, vz=500.0, n=20, dt=0.05)
        fit = analyzer.fit_trajectory(points)

        assert fit is not None
        assert fit.landing.z_mm == pytest.approx(0.0, abs=1.0)
        assert fit.landing.timestamp > 0  # landing must be in the future
