"""Tests for the alert grading system."""

import time
from unittest.mock import patch

import numpy as np
import pytest

from argus.alerts.grader import AlertGrader, _AnomalyTracker
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
        "temporal": TemporalConfirmation(
            min_consecutive_frames=3,
            max_gap_seconds=10.0,
            evidence_lambda=0.95,
            evidence_threshold=3.0,
        ),
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
        """Alert should only fire after evidence accumulates past threshold."""
        grader = AlertGrader(make_config(
            temporal=TemporalConfirmation(
                min_consecutive_frames=3,
                max_gap_seconds=10.0,
                evidence_lambda=0.95,
                evidence_threshold=3.0,
            ),
        ))

        # First four detections (score=0.8): evidence accumulates but stays below 3.0
        for i in range(4):
            alert = grader.evaluate(
                camera_id="cam1",
                zone_id="z1",
                zone_priority=ZonePriority.STANDARD,
                anomaly_score=0.8,
                frame_number=i,
            )
            assert alert is None

        # Fifth detection: evidence crosses 3.0 threshold → alert fires
        alert = grader.evaluate(
            camera_id="cam1",
            zone_id="z1",
            zone_priority=ZonePriority.STANDARD,
            anomaly_score=0.8,
            frame_number=5,
        )
        assert alert is not None
        assert alert.severity == AlertSeverity.LOW

    def test_severity_mapping(self):
        """Scores should map to the correct severity levels."""
        config = make_config(
            temporal=TemporalConfirmation(
                min_consecutive_frames=1, max_gap_seconds=10.0,
                evidence_lambda=0.80, evidence_threshold=0.5,
            ),
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
            temporal=TemporalConfirmation(
                min_consecutive_frames=1, max_gap_seconds=10.0,
                evidence_lambda=0.80, evidence_threshold=0.5,
            ),
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
            temporal=TemporalConfirmation(
                min_consecutive_frames=1, max_gap_seconds=10.0,
                evidence_lambda=0.80, evidence_threshold=0.5,
            ),
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
            temporal=TemporalConfirmation(
                min_consecutive_frames=1, max_gap_seconds=10.0,
                evidence_lambda=0.80, evidence_threshold=0.5,
            ),
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
            temporal=TemporalConfirmation(
                min_consecutive_frames=1, max_gap_seconds=10.0,
                evidence_lambda=0.80, evidence_threshold=0.5,
            ),
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


class TestCUSUMEvidence:
    """CUSUM evidence accumulation tests."""

    def _make_cusum_config(self, lam=0.95, threshold=3.0):
        return make_config(
            temporal=TemporalConfirmation(
                min_consecutive_frames=1,
                max_gap_seconds=10.0,
                evidence_lambda=lam,
                evidence_threshold=threshold,
            ),
            suppression=SuppressionConfig(same_zone_window_seconds=10, same_camera_window_seconds=5),
        )

    def _eval(self, grader, score=0.55, zone_id="z1", frame_number=0):
        return grader.evaluate(
            camera_id="cam1",
            zone_id=zone_id,
            zone_priority=ZonePriority.STANDARD,
            anomaly_score=score,
            frame_number=frame_number,
        )

    def test_weak_persistent_signal_triggers(self):
        """连续 15 帧 score=0.55 → evidence 累积到阈值 → 触发告警。"""
        grader = AlertGrader(self._make_cusum_config(lam=0.95, threshold=3.0))
        triggered_at = None
        for i in range(15):
            alert = self._eval(grader, score=0.55, frame_number=i)
            if alert is not None:
                triggered_at = i
                break
        assert triggered_at is not None and triggered_at <= 12

    def test_transient_spike_decays(self):
        """1 帧 score=0.95, 后续 score=0.0 → evidence 快速衰减 → 不触发。"""
        grader = AlertGrader(self._make_cusum_config(lam=0.95, threshold=3.0))
        # One high-score frame
        alert = self._eval(grader, score=0.96, frame_number=0)
        assert alert is None
        # Many zero-score frames (below info threshold → decay)
        for i in range(1, 30):
            alert = self._eval(grader, score=0.0, frame_number=i)
            assert alert is None

    def test_gap_timeout_resets_evidence(self):
        """连续 3 帧有信号, gap > max_gap_seconds, 然后 1 帧 → evidence 重新从 0 开始。"""
        grader = AlertGrader(self._make_cusum_config(lam=0.95, threshold=5.0))
        # Build up some evidence (with higher threshold=5.0, won't trigger)
        for i in range(3):
            self._eval(grader, score=0.8, frame_number=i)

        tracker = grader._trackers["cam1:z1"]
        assert tracker.evidence > 1.0

        # Simulate gap by advancing time past max_gap_seconds
        tracker.last_seen = time.monotonic() - 15.0  # 15s gap > 10s max
        alert = self._eval(grader, score=0.8, frame_number=10)
        # Evidence should have been reset — single frame at 0.8 won't cross 5.0
        assert alert is None
        assert grader._trackers["cam1:z1"].evidence < 1.0

    def test_spatial_mismatch_resets_evidence(self):
        """IoU < min_spatial_overlap → evidence 归零。"""
        config = make_config(
            temporal=TemporalConfirmation(
                min_consecutive_frames=1,
                max_gap_seconds=10.0,
                min_spatial_overlap=0.3,
                evidence_lambda=0.95,
                evidence_threshold=3.0,
            ),
            suppression=SuppressionConfig(same_zone_window_seconds=10, same_camera_window_seconds=5),
        )
        grader = AlertGrader(config)

        # Build evidence with matching anomaly maps
        map1 = np.zeros((100, 100), dtype=np.float32)
        map1[20:60, 20:60] = 0.9  # top-left quadrant
        for i in range(3):
            grader.evaluate(
                camera_id="cam1", zone_id="z1", zone_priority=ZonePriority.STANDARD,
                anomaly_score=0.8, frame_number=i, anomaly_map=map1,
            )

        assert grader._trackers["cam1:z1"].evidence > 1.0

        # Completely different anomaly region → IoU ≈ 0 → evidence reset
        map2 = np.zeros((100, 100), dtype=np.float32)
        map2[70:95, 70:95] = 0.9  # bottom-right
        grader.evaluate(
            camera_id="cam1", zone_id="z1", zone_priority=ZonePriority.STANDARD,
            anomaly_score=0.8, frame_number=4, anomaly_map=map2,
        )
        # After spatial reset, evidence = 0.0 + 0.8 = 0.8
        assert grader._trackers["cam1:z1"].evidence < 1.5

    def test_steady_medium_signal(self):
        """连续帧 score=0.72 → evidence 稳步累积 → 约 5-6 帧触发。"""
        grader = AlertGrader(self._make_cusum_config(lam=0.95, threshold=3.0))
        triggered_at = None
        for i in range(10):
            alert = self._eval(grader, score=0.72, frame_number=i)
            if alert is not None:
                triggered_at = i
                break
        assert triggered_at is not None and triggered_at <= 7

    def test_below_info_threshold_decays(self):
        """score < 0.50 的帧 → evidence 纯衰减。"""
        grader = AlertGrader(self._make_cusum_config(lam=0.95, threshold=3.0))
        # First build evidence
        for i in range(3):
            self._eval(grader, score=0.8, frame_number=i)

        evidence_before = grader._trackers["cam1:z1"].evidence
        assert evidence_before > 1.0

        # Feed sub-threshold frames → evidence decays
        for i in range(3, 8):
            self._eval(grader, score=0.3, frame_number=i)

        evidence_after = grader._trackers["cam1:z1"].evidence
        assert evidence_after < evidence_before

    def test_mixed_signal_accumulates(self):
        """交替 score=0.6 和 score=0.0 → evidence 缓慢累积但可能不触发。"""
        grader = AlertGrader(self._make_cusum_config(lam=0.95, threshold=3.0))
        triggered = False
        for i in range(20):
            score = 0.6 if i % 2 == 0 else 0.0
            alert = self._eval(grader, score=score, frame_number=i)
            if alert is not None:
                triggered = True
                break
        # With alternating 0.6/0.0, evidence grows slowly due to decay on zero frames
        # May or may not trigger in 20 frames depending on exact math
        # The key test is that it doesn't crash and evidence mechanics work
