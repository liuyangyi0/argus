"""Tests for the physics speed estimation module."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import pytest

from argus.physics.speed import PixelSpeedEstimator, SpeedEstimate


@dataclass
class FakeTrack:
    """Minimal TrackedAnomaly for testing."""

    centroid_x: float = 100.0
    centroid_y: float = 100.0
    velocity: tuple[float, float] = (0.0, 0.0)
    trajectory_history: list[tuple[float, float, float]] = field(default_factory=list)
    max_score: float = 0.8


class TestPixelSpeedEstimator:
    def test_stationary_object(self):
        """Stationary object should have speed ≈ 0."""
        est = PixelSpeedEstimator(fps=15.0)
        track = FakeTrack(velocity=(0.0, 0.0))
        result = est.estimate(track)

        assert result.speed_px_per_sec == pytest.approx(0.0, abs=0.1)
        assert result.is_falling is False

    def test_horizontal_motion(self):
        """Horizontal motion: vx > 0, vy ≈ 0."""
        est = PixelSpeedEstimator(fps=15.0)
        track = FakeTrack(velocity=(10.0, 0.0))  # 10 px/frame right
        result = est.estimate(track)

        assert result.vx_px_per_sec == pytest.approx(150.0, rel=0.01)
        assert result.vy_px_per_sec == pytest.approx(0.0, abs=0.1)
        assert result.speed_px_per_sec == pytest.approx(150.0, rel=0.01)
        assert result.is_falling is False

    def test_vertical_fall(self):
        """Downward movement should be detected as falling."""
        est = PixelSpeedEstimator(fps=15.0)
        track = FakeTrack(velocity=(0.0, 20.0))  # 20 px/frame downward
        result = est.estimate(track)

        assert result.vy_px_per_sec == pytest.approx(300.0, rel=0.01)
        assert result.is_falling is True
        assert result.direction_deg == pytest.approx(180.0, abs=5.0)

    def test_smoothed_velocity_from_history(self):
        """With enough trajectory history, should use smoothed velocity."""
        est = PixelSpeedEstimator(fps=15.0, smoothing_window=3)
        now = time.time()
        history = [
            (now - 0.3, 100.0, 100.0),
            (now - 0.2, 110.0, 100.0),
            (now - 0.1, 120.0, 100.0),
            (now, 130.0, 100.0),
        ]
        track = FakeTrack(velocity=(10.0, 0.0), trajectory_history=history)
        result = est.estimate(track)

        assert result.smoothed is True
        assert result.vx_px_per_sec > 0

    def test_pixel_scale_converts_to_ms(self):
        """With pixel_scale_mm_per_px, speed_ms should be populated."""
        est = PixelSpeedEstimator(fps=10.0, pixel_scale_mm_per_px=2.0)
        track = FakeTrack(velocity=(5.0, 0.0))  # 5 px/frame = 50 px/s
        result = est.estimate(track)

        # 50 px/s * 2.0 mm/px / 1000 = 0.1 m/s
        assert result.speed_ms is not None
        assert result.speed_ms == pytest.approx(0.1, rel=0.05)

    def test_no_pixel_scale_no_ms(self):
        """Without pixel_scale, speed_ms should be None."""
        est = PixelSpeedEstimator(fps=10.0)
        track = FakeTrack(velocity=(5.0, 0.0))
        result = est.estimate(track)

        assert result.speed_ms is None

    def test_empty_history_uses_instantaneous(self):
        """With empty trajectory_history, should use instantaneous velocity."""
        est = PixelSpeedEstimator(fps=10.0, smoothing_window=5)
        track = FakeTrack(velocity=(3.0, 4.0), trajectory_history=[])
        result = est.estimate(track)

        assert result.smoothed is False
        assert result.speed_px_per_sec == pytest.approx(50.0, rel=0.01)  # 5 px/frame * 10 fps

    def test_freefall_check_without_scale(self):
        """Freefall check should return False without pixel_scale."""
        est = PixelSpeedEstimator(fps=30.0)
        now = time.time()
        history = [
            (now - 0.2, 100, 100),
            (now - 0.1, 100, 150),
            (now, 100, 220),
        ]
        track = FakeTrack(velocity=(0, 70), trajectory_history=history)
        result = est.estimate(track)

        assert result.matches_freefall is False
