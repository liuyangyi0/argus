"""Tests for the temporal anomaly tracker."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from argus.core.temporal_tracker import TemporalAnomalyTracker, TemporalAnalysis, TrackedAnomaly


@dataclass
class FakeRegion:
    """Minimal region for testing."""

    centroid_x: float
    centroid_y: float
    max_score: float


class TestTemporalAnomalyTracker:
    def test_new_region_creates_new_track(self):
        """A region with no prior tracks should create a new track."""
        tracker = TemporalAnomalyTracker()
        result = tracker.update([FakeRegion(100.0, 200.0, 0.8)], frame_number=1)

        assert result.new_tracks == 1
        assert len(result.active_tracks) == 1
        assert result.active_tracks[0].centroid_x == 100.0
        assert result.active_tracks[0].centroid_y == 200.0
        assert result.active_tracks[0].consecutive_frames == 1

    def test_same_position_updates_existing_track(self):
        """A region at the same position should update the existing track."""
        tracker = TemporalAnomalyTracker(match_distance=50.0)

        tracker.update([FakeRegion(100.0, 200.0, 0.8)], frame_number=1)
        result = tracker.update([FakeRegion(105.0, 205.0, 0.85)], frame_number=2)

        assert result.new_tracks == 0
        assert len(result.active_tracks) == 1
        assert result.active_tracks[0].consecutive_frames == 2
        assert result.active_tracks[0].max_score == 0.85

    def test_far_away_region_creates_separate_track(self):
        """A region far from existing tracks should create a new track."""
        tracker = TemporalAnomalyTracker(match_distance=50.0)

        tracker.update([FakeRegion(100.0, 200.0, 0.8)], frame_number=1)
        result = tracker.update(
            [FakeRegion(100.0, 200.0, 0.8), FakeRegion(500.0, 500.0, 0.9)],
            frame_number=2,
        )

        assert result.new_tracks == 1
        assert len(result.active_tracks) == 2

    def test_track_disappears_after_max_gap_frames(self):
        """A track not seen for max_gap_frames should be removed."""
        tracker = TemporalAnomalyTracker(max_gap_frames=3)

        tracker.update([FakeRegion(100.0, 200.0, 0.8)], frame_number=1)
        # Skip several frames with no regions
        result = tracker.update([], frame_number=5)

        assert result.lost_tracks == 1
        assert len(result.active_tracks) == 0

    def test_track_persists_within_gap_window(self):
        """A track should persist if within max_gap_frames even without matches."""
        tracker = TemporalAnomalyTracker(max_gap_frames=5)

        tracker.update([FakeRegion(100.0, 200.0, 0.8)], frame_number=1)
        # Skip 3 frames (within gap of 5)
        result = tracker.update([], frame_number=4)

        assert result.lost_tracks == 0
        assert len(result.active_tracks) == 1

    def test_stationary_detection(self):
        """A region that barely moves should be marked stationary."""
        tracker = TemporalAnomalyTracker(stationary_threshold=10.0)

        tracker.update([FakeRegion(100.0, 200.0, 0.8)], frame_number=1)
        result = tracker.update([FakeRegion(102.0, 201.0, 0.8)], frame_number=2)

        assert result.active_tracks[0].is_stationary is True

    def test_moving_detection(self):
        """A region that moves significantly should not be stationary."""
        tracker = TemporalAnomalyTracker(stationary_threshold=10.0, match_distance=100.0)

        tracker.update([FakeRegion(100.0, 200.0, 0.8)], frame_number=1)
        result = tracker.update([FakeRegion(140.0, 230.0, 0.8)], frame_number=2)

        assert result.active_tracks[0].is_stationary is False

    def test_severity_boost_zero_for_new_track(self):
        """A brand-new track should not have a severity boost."""
        tracker = TemporalAnomalyTracker()
        result = tracker.update([FakeRegion(100.0, 200.0, 0.8)], frame_number=1)

        assert result.severity_boost == 0.0

    def test_severity_boost_01_at_5_frames(self):
        """A stationary track at 5+ frames gets +0.1 boost."""
        tracker = TemporalAnomalyTracker(stationary_threshold=10.0)

        for i in range(5):
            result = tracker.update([FakeRegion(100.0, 200.0, 0.8)], frame_number=i + 1)

        assert result.severity_boost == pytest.approx(0.1)

    def test_severity_boost_02_at_15_frames(self):
        """A stationary track at 15+ frames gets +0.2 boost."""
        tracker = TemporalAnomalyTracker(stationary_threshold=10.0)

        for i in range(15):
            result = tracker.update([FakeRegion(100.0, 200.0, 0.8)], frame_number=i + 1)

        assert result.severity_boost == pytest.approx(0.2)

    def test_severity_boost_03_at_30_frames(self):
        """A stationary track at 30+ frames gets +0.3 boost."""
        tracker = TemporalAnomalyTracker(stationary_threshold=10.0)

        for i in range(30):
            result = tracker.update([FakeRegion(100.0, 200.0, 0.8)], frame_number=i + 1)

        assert result.severity_boost == pytest.approx(0.3)

    def test_no_boost_for_moving_track(self):
        """A moving track should not receive a severity boost even after many frames."""
        tracker = TemporalAnomalyTracker(stationary_threshold=5.0, match_distance=100.0)

        for i in range(30):
            # Move significantly each frame
            result = tracker.update(
                [FakeRegion(100.0 + i * 20.0, 200.0, 0.8)], frame_number=i + 1
            )

        assert result.severity_boost == 0.0

    def test_reset_clears_all_tracks(self):
        """Reset should remove all active tracks."""
        tracker = TemporalAnomalyTracker()

        tracker.update([FakeRegion(100.0, 200.0, 0.8)], frame_number=1)
        tracker.reset()
        result = tracker.update([FakeRegion(100.0, 200.0, 0.8)], frame_number=2)

        # After reset, this should be a new track
        assert result.new_tracks == 1
        assert result.active_tracks[0].consecutive_frames == 1

    def test_multiple_simultaneous_tracks(self):
        """Multiple regions in the same frame should create multiple tracks."""
        tracker = TemporalAnomalyTracker(match_distance=50.0)

        regions = [
            FakeRegion(100.0, 100.0, 0.8),
            FakeRegion(300.0, 300.0, 0.7),
            FakeRegion(500.0, 500.0, 0.9),
        ]
        result = tracker.update(regions, frame_number=1)

        assert result.new_tracks == 3
        assert len(result.active_tracks) == 3

    def test_temporal_analysis_counts_new_and_lost(self):
        """TemporalAnalysis should correctly count new and lost tracks."""
        tracker = TemporalAnomalyTracker(max_gap_frames=2)

        # Frame 1: create 2 tracks
        r1 = tracker.update(
            [FakeRegion(100.0, 100.0, 0.8), FakeRegion(400.0, 400.0, 0.7)],
            frame_number=1,
        )
        assert r1.new_tracks == 2
        assert r1.lost_tracks == 0

        # Frame 5: both tracks should be lost, one new
        r2 = tracker.update([FakeRegion(700.0, 700.0, 0.9)], frame_number=5)
        assert r2.lost_tracks == 2
        assert r2.new_tracks == 1
        assert len(r2.active_tracks) == 1

    def test_avg_score_computed_correctly(self):
        """Average score should reflect all matched scores."""
        tracker = TemporalAnomalyTracker(stationary_threshold=100.0)

        tracker.update([FakeRegion(100.0, 200.0, 0.6)], frame_number=1)
        tracker.update([FakeRegion(100.0, 200.0, 0.8)], frame_number=2)
        result = tracker.update([FakeRegion(100.0, 200.0, 1.0)], frame_number=3)

        assert result.active_tracks[0].avg_score == pytest.approx(0.8, abs=0.01)

    def test_persistence_seconds(self):
        """Persistence should reflect frame span divided by fps."""
        tracker = TemporalAnomalyTracker(fps=10.0, stationary_threshold=100.0)

        tracker.update([FakeRegion(100.0, 200.0, 0.8)], frame_number=1)
        result = tracker.update([FakeRegion(100.0, 200.0, 0.8)], frame_number=11)

        # Frames 1 to 11 = 11 frames span => 11/10 = 1.1 seconds
        assert result.active_tracks[0].persistence_seconds == pytest.approx(1.1)


class TestStationarySuppressionF5:
    """F5: long-settled tracks should flip to ``suppressed=True``."""

    def test_stationary_track_marked_suppressed_after_interval(self):
        """A track stationary for >= stationary_suppress_after_s flips to suppressed."""
        tracker = TemporalAnomalyTracker(
            stationary_threshold=10.0,
            stationary_suppress_after_s=10.0,
            stationary_suppress_enabled=True,
        )
        base = 1_000_000.0
        with patch("argus.core.temporal_tracker.time.time") as mock_time:
            # 15 updates spanning 14 seconds — plenty past the 10s window.
            mock_time.side_effect = [base + i * 1.0 for i in range(15)]
            for i in range(15):
                result = tracker.update(
                    [FakeRegion(100.0, 200.0, 0.8)], frame_number=i + 1,
                )
        assert result.active_tracks[0].suppressed is True
        # Sanity: the track is stationary throughout.
        assert result.active_tracks[0].is_stationary is True

    def test_moving_track_not_suppressed(self):
        """Fast-moving tracks never flip to suppressed regardless of duration."""
        tracker = TemporalAnomalyTracker(
            stationary_threshold=5.0,
            match_distance=200.0,
            stationary_suppress_after_s=1.0,
            stationary_suppress_enabled=True,
        )
        base = 2_000_000.0
        with patch("argus.core.temporal_tracker.time.time") as mock_time:
            mock_time.side_effect = [base + i * 1.0 for i in range(15)]
            for i in range(15):
                result = tracker.update(
                    [FakeRegion(100.0 + i * 40.0, 200.0, 0.8)], frame_number=i + 1,
                )
        assert result.active_tracks[0].is_stationary is False
        assert result.active_tracks[0].suppressed is False

    def test_suppression_clears_when_motion_resumes(self):
        """A track that starts moving again after suppression clears back to non-suppressed."""
        tracker = TemporalAnomalyTracker(
            stationary_threshold=10.0,
            match_distance=200.0,
            stationary_suppress_after_s=5.0,
            stationary_suppress_enabled=True,
        )
        base = 3_000_000.0
        phase1 = [base + i * 1.0 for i in range(10)]          # 10s stationary
        phase2 = [base + 10.0 + i * 1.0 for i in range(5)]    # moving phase
        with patch("argus.core.temporal_tracker.time.time") as mock_time:
            mock_time.side_effect = phase1 + phase2
            # Phase 1 — stationary for 10s, suppression kicks in at 5s.
            for i in range(10):
                result = tracker.update(
                    [FakeRegion(100.0, 200.0, 0.8)], frame_number=i + 1,
                )
            assert result.active_tracks[0].suppressed is True

            # Phase 2 — track moves significantly each frame; suppression clears.
            for i in range(5):
                result = tracker.update(
                    [FakeRegion(100.0 + (i + 1) * 50.0, 200.0, 0.8)],
                    frame_number=10 + i + 1,
                )
        assert result.active_tracks[0].is_stationary is False
        assert result.active_tracks[0].suppressed is False

    def test_suppression_disabled_never_marks_suppressed(self):
        """With stationary_suppress_enabled=False, suppressed stays False."""
        tracker = TemporalAnomalyTracker(
            stationary_threshold=10.0,
            stationary_suppress_after_s=1.0,
            stationary_suppress_enabled=False,
        )
        base = 4_000_000.0
        with patch("argus.core.temporal_tracker.time.time") as mock_time:
            mock_time.side_effect = [base + i * 1.0 for i in range(15)]
            for i in range(15):
                result = tracker.update(
                    [FakeRegion(100.0, 200.0, 0.8)], frame_number=i + 1,
                )
        assert result.active_tracks[0].suppressed is False
