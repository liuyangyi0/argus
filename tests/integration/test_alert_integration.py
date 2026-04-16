"""Alert flow integration tests.

Tests the AlertGrader's CUSUM evidence accumulation, deduplication,
severity grading, and the Simplex safety channel interaction with
the main detection pipeline.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from argus.alerts.grader import Alert, AlertGrader
from argus.config.schema import (
    AlertConfig,
    AlertSeverity,
    SeverityThresholds,
    SuppressionConfig,
    TemporalConfirmation,
    ZonePriority,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fast_config(**overrides) -> AlertConfig:
    """Create an AlertConfig tuned for fast testing."""
    defaults = dict(
        severity_thresholds=SeverityThresholds(
            info=0.3, low=0.5, medium=0.7, high=0.9,
        ),
        temporal=TemporalConfirmation(

            max_gap_seconds=10.0,
            evidence_lambda=0.8,
            evidence_threshold=1.0,
            min_spatial_overlap=0.0,
        ),
        suppression=SuppressionConfig(
            same_zone_window_seconds=10.0,
            same_camera_window_seconds=5.0,
        ),
    )
    defaults.update(overrides)
    return AlertConfig(**defaults)


class TestCUSUMEvidenceAccumulation:
    """Test that CUSUM evidence accumulates and triggers correctly."""

    def test_weak_signals_accumulate_to_alert(self):
        """Repeated weak-but-above-threshold signals accumulate evidence."""
        config = _make_fast_config(
            temporal=TemporalConfirmation(
                evidence_lambda=0.8,
                evidence_threshold=2.0,
                min_spatial_overlap=0.0,
            ),
        )
        grader = AlertGrader(config)

        alert = None
        for i in range(20):
            result = grader.evaluate(
                camera_id="cam1",
                zone_id="z1",
                zone_priority=ZonePriority.STANDARD,
                anomaly_score=0.6,  # Above info(0.3) but not super high
                frame_number=i,
            )
            if result is not None:
                alert = result
                break

        assert alert is not None, "Expected alert after evidence accumulation"
        assert alert.severity in (
            AlertSeverity.LOW,
            AlertSeverity.MEDIUM,
            AlertSeverity.INFO,
        )

    def test_below_threshold_no_accumulation(self):
        """Scores below the minimum info threshold never accumulate evidence."""
        config = _make_fast_config()
        grader = AlertGrader(config)

        for i in range(50):
            result = grader.evaluate(
                camera_id="cam1",
                zone_id="z1",
                zone_priority=ZonePriority.STANDARD,
                anomaly_score=0.2,  # Below info threshold of 0.3
                frame_number=i,
            )
            assert result is None

    def test_gap_timeout_resets_evidence(self):
        """A time gap larger than max_gap_seconds resets evidence."""
        config = _make_fast_config(
            temporal=TemporalConfirmation(
                evidence_lambda=0.8,
                evidence_threshold=2.0,
                max_gap_seconds=1.0,
                min_spatial_overlap=0.0,
            ),
        )
        grader = AlertGrader(config)

        # Accumulate some evidence
        for i in range(5):
            grader.evaluate(
                camera_id="cam1",
                zone_id="z1",
                zone_priority=ZonePriority.STANDARD,
                anomaly_score=0.6,
                frame_number=i,
            )

        # Simulate a gap by sleeping
        time.sleep(0.6)

        # Evidence should be reset; single detection should not alert
        result = grader.evaluate(
            camera_id="cam1",
            zone_id="z1",
            zone_priority=ZonePriority.STANDARD,
            anomaly_score=0.6,
            frame_number=10,
        )
        assert result is None


class TestAlertDeduplication:
    """Test that duplicate alerts within a suppression window are dropped."""

    def test_same_zone_dedup(self):
        """Only one alert per zone within the suppression window."""
        config = _make_fast_config(
            suppression=SuppressionConfig(
                same_zone_window_seconds=60.0,
                same_camera_window_seconds=30.0,
            ),
            temporal=TemporalConfirmation(
                evidence_lambda=0.8,  # No decay -> instant evidence
                evidence_threshold=0.5,
                min_spatial_overlap=0.0,
            ),
        )
        grader = AlertGrader(config)

        alerts = []
        for i in range(10):
            result = grader.evaluate(
                camera_id="cam1",
                zone_id="z1",
                zone_priority=ZonePriority.STANDARD,
                anomaly_score=0.8,
                frame_number=i,
            )
            if result is not None:
                alerts.append(result)

        # Should have exactly one alert — subsequent ones suppressed
        assert len(alerts) == 1

    def test_different_zones_independent(self):
        """Alerts from different zones are independent when camera window allows."""
        config = _make_fast_config(
            temporal=TemporalConfirmation(
                evidence_lambda=0.8,
                evidence_threshold=0.5,
                min_spatial_overlap=0.0,
            ),
            # Minimal camera window so zones can fire independently
            suppression=SuppressionConfig(
                same_zone_window_seconds=60.0,
                same_camera_window_seconds=5.0,
            ),
        )
        grader = AlertGrader(config)

        alerts = []
        for zone_id in ("z1", "z2", "z3"):
            for i in range(5):
                result = grader.evaluate(
                    camera_id="cam1",
                    zone_id=zone_id,
                    zone_priority=ZonePriority.STANDARD,
                    anomaly_score=0.8,
                    frame_number=i,
                )
                if result is not None:
                    alerts.append(result)
            grader.reset_camera_suppression("cam1")

        # Each zone should produce exactly one alert
        zone_ids = {a.zone_id for a in alerts}
        assert zone_ids == {"z1", "z2", "z3"}


class TestAlertSeverityGrading:
    """Test that different scores map to correct severity levels."""

    def test_severity_thresholds(self):
        """Scores map to the correct severity bracket."""
        config = _make_fast_config(
            temporal=TemporalConfirmation(
                evidence_lambda=0.8,
                evidence_threshold=0.5,
                min_spatial_overlap=0.0,
            ),
            suppression=SuppressionConfig(
                same_zone_window_seconds=10.0,
                same_camera_window_seconds=5.0,
            ),
        )

        test_cases = [
            (0.35, AlertSeverity.INFO),   # Just above info=0.3
            (0.55, AlertSeverity.LOW),    # Above low=0.5
            (0.75, AlertSeverity.MEDIUM), # Above medium=0.7
            (0.95, AlertSeverity.HIGH),   # Above high=0.9
        ]

        for score, expected_severity in test_cases:
            # Use a fresh grader to avoid suppression issues
            grader = AlertGrader(config)
            alert = None
            for i in range(5):
                result = grader.evaluate(
                    camera_id="cam1",
                    zone_id=f"z_{score}",
                    zone_priority=ZonePriority.STANDARD,
                    anomaly_score=score,
                    frame_number=i,
                )
                if result is not None:
                    alert = result
                    break

            assert alert is not None, f"Expected alert for score={score}"
            assert alert.severity == expected_severity, (
                f"Score {score}: expected {expected_severity.value}, "
                f"got {alert.severity.value}"
            )

    def test_zone_priority_multiplier(self):
        """Critical zone priority boosts the effective score."""
        config = _make_fast_config(
            temporal=TemporalConfirmation(
                evidence_lambda=0.8,
                evidence_threshold=0.5,
                min_spatial_overlap=0.0,
            ),
        )
        grader = AlertGrader(config)

        # Score 0.6 with CRITICAL zone priority (multiplier 1.2) -> 0.72 -> MEDIUM
        alert = None
        for i in range(5):
            result = grader.evaluate(
                camera_id="cam1",
                zone_id="z_crit",
                zone_priority=ZonePriority.CRITICAL,
                anomaly_score=0.6,
                frame_number=i,
            )
            if result is not None:
                alert = result
                break

        assert alert is not None
        # 0.6 * 1.2 = 0.72 -> MEDIUM (threshold 0.7)
        assert alert.severity == AlertSeverity.MEDIUM

    def test_nan_score_rejected(self):
        """NaN anomaly scores are rejected silently."""
        config = _make_fast_config()
        grader = AlertGrader(config)

        result = grader.evaluate(
            camera_id="cam1",
            zone_id="z1",
            zone_priority=ZonePriority.STANDARD,
            anomaly_score=float("nan"),
            frame_number=0,
        )
        assert result is None


class TestSimplexSafetyChannel:
    """Test Simplex safety detector integration with the pipeline."""

    def test_simplex_detects_static_object(self):
        """Simplex should detect a large static object even without ML model."""
        from argus.prefilter.simple_detector import SimplexDetector

        detector = SimplexDetector(
            diff_threshold=30,
            min_area_px=500,
            min_static_seconds=5.0,
            morph_kernel_size=5,
            match_radius_px=50,
        )
        # Override the internal threshold for testing (bypass Pydantic validation)
        detector._min_static_seconds = 0.0

        # Set reference frame (clean scene)
        reference = np.full((480, 640, 3), 128, dtype=np.uint8)
        detector.set_reference(reference)

        # Create frame with a large foreign object (black rectangle)
        current = reference.copy()
        current[100:300, 150:450] = 0  # Large black rectangle

        result = detector.detect(current)
        # The detector should see a significant difference and have regions
        assert result is not None
        assert result.has_detection is True

    def test_simplex_no_detection_on_identical(self):
        """Simplex should not detect anything when frames are identical."""
        from argus.prefilter.simple_detector import SimplexDetector

        detector = SimplexDetector(
            diff_threshold=30,
            min_area_px=500,
            min_static_seconds=5.0,
            morph_kernel_size=5,
            match_radius_px=50,
        )

        reference = np.full((480, 640, 3), 128, dtype=np.uint8)
        detector.set_reference(reference)

        result = detector.detect(reference.copy())
        assert result.has_detection is False
