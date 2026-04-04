"""Tests for the alert grading system."""

import time

import numpy as np
import pytest

from argus.alerts.grader import AlertGrader
from argus.config.schema import (
    AlertConfig,
    AlertSeverity,
    SeverityThresholds,
    SuppressionConfig,
    TemporalConfirmation,
    ZonePriority,
)


def make_config(**overrides) -> AlertConfig:
    """Create an AlertConfig with optional overrides."""
    defaults = {
        "severity_thresholds": SeverityThresholds(info=0.5, low=0.7, medium=0.85, high=0.95),
        "temporal": TemporalConfirmation(min_consecutive_frames=3, max_gap_seconds=10.0),
        "suppression": SuppressionConfig(same_zone_window_seconds=300, same_camera_window_seconds=60),
    }
    defaults.update(overrides)
    return AlertConfig(**defaults)


class TestAlertGrader:
    def test_no_alert_below_threshold(self):
        """Scores below the minimum threshold should not produce alerts."""
        grader = AlertGrader(make_config())
        for _ in range(10):
            alert = grader.evaluate(
                camera_id="cam1",
                zone_id="z1",
                zone_priority=ZonePriority.STANDARD,
                anomaly_score=0.3,
                frame_number=1,
            )
            assert alert is None

    def test_temporal_confirmation_required(self):
        """Alert should only fire after min_consecutive_frames."""
        grader = AlertGrader(make_config(
            temporal=TemporalConfirmation(min_consecutive_frames=3, max_gap_seconds=10.0),
        ))

        # First two detections: no alert yet
        for i in range(2):
            alert = grader.evaluate(
                camera_id="cam1",
                zone_id="z1",
                zone_priority=ZonePriority.STANDARD,
                anomaly_score=0.8,
                frame_number=i,
            )
            assert alert is None

        # Third detection: alert fires
        alert = grader.evaluate(
            camera_id="cam1",
            zone_id="z1",
            zone_priority=ZonePriority.STANDARD,
            anomaly_score=0.8,
            frame_number=3,
        )
        assert alert is not None
        assert alert.severity == AlertSeverity.LOW

    def test_severity_mapping(self):
        """Scores should map to the correct severity levels."""
        config = make_config(
            temporal=TemporalConfirmation(min_consecutive_frames=1, max_gap_seconds=10.0),
            suppression=SuppressionConfig(same_zone_window_seconds=10, same_camera_window_seconds=5),
        )
        grader = AlertGrader(config)

        test_cases = [
            (0.75, AlertSeverity.LOW),
            (0.90, AlertSeverity.MEDIUM),
            (0.98, AlertSeverity.HIGH),
        ]

        for score, expected_severity in test_cases:
            grader.reset()  # Reset to avoid suppression
            alert = grader.evaluate(
                camera_id="cam1",
                zone_id=f"z_{score}",
                zone_priority=ZonePriority.STANDARD,
                anomaly_score=score,
                frame_number=1,
            )
            assert alert is not None, f"No alert for score {score}"
            assert alert.severity == expected_severity, (
                f"Score {score}: expected {expected_severity}, got {alert.severity}"
            )

    def test_zone_priority_multiplier(self):
        """Critical zones should amplify the score."""
        config = make_config(
            temporal=TemporalConfirmation(min_consecutive_frames=1, max_gap_seconds=10.0),
            suppression=SuppressionConfig(same_zone_window_seconds=10, same_camera_window_seconds=5),
        )
        grader = AlertGrader(config)

        # Score 0.72 * 1.2 (critical multiplier) = 0.864 → should be MEDIUM
        alert = grader.evaluate(
            camera_id="cam1",
            zone_id="z1",
            zone_priority=ZonePriority.CRITICAL,
            anomaly_score=0.72,
            frame_number=1,
        )
        assert alert is not None
        assert alert.severity == AlertSeverity.MEDIUM

    def test_suppression_deduplication(self):
        """Duplicate alerts within the suppression window should be suppressed."""
        config = make_config(
            temporal=TemporalConfirmation(min_consecutive_frames=1, max_gap_seconds=10.0),
            suppression=SuppressionConfig(same_zone_window_seconds=300, same_camera_window_seconds=60),
        )
        grader = AlertGrader(config)

        # First alert
        alert1 = grader.evaluate(
            camera_id="cam1",
            zone_id="z1",
            zone_priority=ZonePriority.STANDARD,
            anomaly_score=0.8,
            frame_number=1,
        )
        assert alert1 is not None

        # Second alert for same zone — should be suppressed
        alert2 = grader.evaluate(
            camera_id="cam1",
            zone_id="z1",
            zone_priority=ZonePriority.STANDARD,
            anomaly_score=0.9,
            frame_number=2,
        )
        assert alert2 is None

    def test_reset_clears_state(self):
        """Reset should allow new alerts from the same zone."""
        config = make_config(
            temporal=TemporalConfirmation(min_consecutive_frames=1, max_gap_seconds=10.0),
            suppression=SuppressionConfig(same_zone_window_seconds=300, same_camera_window_seconds=60),
        )
        grader = AlertGrader(config)

        alert1 = grader.evaluate(
            camera_id="cam1",
            zone_id="z1",
            zone_priority=ZonePriority.STANDARD,
            anomaly_score=0.8,
            frame_number=1,
        )
        assert alert1 is not None

        grader.reset()

        alert2 = grader.evaluate(
            camera_id="cam1",
            zone_id="z1",
            zone_priority=ZonePriority.STANDARD,
            anomaly_score=0.8,
            frame_number=2,
        )
        assert alert2 is not None

    def test_alert_has_correct_fields(self):
        """Alert should contain all expected fields."""
        config = make_config(
            temporal=TemporalConfirmation(min_consecutive_frames=1, max_gap_seconds=10.0),
        )
        grader = AlertGrader(config)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        alert = grader.evaluate(
            camera_id="cam1",
            zone_id="z1",
            zone_priority=ZonePriority.STANDARD,
            anomaly_score=0.8,
            frame_number=42,
            frame=frame,
        )
        assert alert is not None
        assert alert.camera_id == "cam1"
        assert alert.zone_id == "z1"
        assert alert.frame_number == 42
        assert alert.alert_id.startswith("ALT-")
        assert alert.snapshot is not None
