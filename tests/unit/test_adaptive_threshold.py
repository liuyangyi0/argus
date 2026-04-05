"""Tests for the adaptive threshold system."""

from __future__ import annotations

import math

import pytest

from argus.core.adaptive_threshold import AdaptiveThreshold, ThresholdState


class TestAdaptiveThreshold:
    def test_initial_threshold_equals_base(self):
        """Initial threshold should be the base threshold."""
        at = AdaptiveThreshold(base_threshold=0.7)
        assert at.get_threshold() == 0.7

    def test_threshold_stays_within_bounds(self):
        """Threshold must stay within [base - delta, base + delta]."""
        at = AdaptiveThreshold(
            base_threshold=0.7, max_delta=0.1, update_interval=5, window_size=50
        )

        # Feed very high scores to push threshold up
        for _ in range(60):
            at.record_score(0.99)

        assert at.get_threshold() <= 0.7 + 0.1
        assert at.get_threshold() >= 0.7 - 0.1

    def test_low_scores_decrease_threshold(self):
        """Consistently low scores should decrease the threshold toward lower bound."""
        at = AdaptiveThreshold(
            base_threshold=0.7, max_delta=0.1, update_interval=10, window_size=50,
            ewma_alpha=0.5,
        )

        for _ in range(200):
            at.record_score(0.1)

        # Threshold should have moved below the base
        assert at.get_threshold() < 0.7

    def test_high_scores_increase_threshold(self):
        """Consistently high scores should increase the threshold toward upper bound."""
        at = AdaptiveThreshold(
            base_threshold=0.7, max_delta=0.1, update_interval=10, window_size=50,
            ewma_alpha=0.5,
        )

        for _ in range(200):
            at.record_score(0.95)

        # Threshold should have moved above the base
        assert at.get_threshold() > 0.7

    def test_reset_restores_base(self):
        """Reset should restore the base threshold."""
        at = AdaptiveThreshold(base_threshold=0.7, update_interval=5, ewma_alpha=0.5)

        for _ in range(100):
            at.record_score(0.95)

        at.reset()
        assert at.get_threshold() == 0.7

    def test_threshold_state_has_correct_values(self):
        """ThresholdState should reflect the current internal state."""
        at = AdaptiveThreshold(base_threshold=0.7, update_interval=5, window_size=50)

        for _ in range(10):
            at.record_score(0.5)

        state = at.get_state()
        assert state.base_threshold == 0.7
        assert isinstance(state.current_threshold, float)
        assert state.sample_count == 10
        assert state.score_mean == pytest.approx(0.5)
        assert state.score_std == pytest.approx(0.0)

    def test_ewma_smoothing_prevents_jumps(self):
        """EWMA should smooth threshold changes, preventing large jumps."""
        at = AdaptiveThreshold(
            base_threshold=0.7, max_delta=0.1, update_interval=5,
            window_size=50, ewma_alpha=0.1,
        )

        # First establish a baseline with low scores
        for _ in range(20):
            at.record_score(0.1)
        threshold_after_low = at.get_threshold()

        # Now feed high scores for one interval
        for _ in range(5):
            at.record_score(0.95)
        threshold_after_spike = at.get_threshold()

        # With alpha=0.1, the jump should be small
        jump = abs(threshold_after_spike - threshold_after_low)
        assert jump < 0.05  # EWMA dampens the change

    def test_get_threshold_returns_float_in_valid_range(self):
        """get_threshold should always return a float within the bounded range."""
        at = AdaptiveThreshold(base_threshold=0.5, max_delta=0.15)

        # Various score patterns
        for score in [0.0, 0.1, 0.5, 0.9, 1.0, 0.3, 0.7]:
            at.record_score(score)

        threshold = at.get_threshold()
        assert isinstance(threshold, float)
        assert 0.5 - 0.15 <= threshold <= 0.5 + 0.15

    def test_no_update_before_interval(self):
        """Threshold should not change if fewer than update_interval scores recorded."""
        at = AdaptiveThreshold(
            base_threshold=0.7, update_interval=50, ewma_alpha=0.5
        )

        for _ in range(49):
            at.record_score(0.1)

        # Should still be at base since update_interval not reached
        assert at.get_threshold() == 0.7

    def test_update_happens_at_interval(self):
        """Threshold should update once update_interval scores are recorded."""
        at = AdaptiveThreshold(
            base_threshold=0.7, max_delta=0.1, update_interval=10,
            window_size=50, ewma_alpha=0.5,
        )

        for _ in range(10):
            at.record_score(0.1)

        # After exactly update_interval scores, threshold should have changed
        assert at.get_threshold() != 0.7

    def test_nan_scores_ignored(self):
        """NaN scores should be silently ignored."""
        at = AdaptiveThreshold(base_threshold=0.7, update_interval=5)

        at.record_score(float("nan"))
        at.record_score(float("inf"))
        at.record_score(float("-inf"))

        state = at.get_state()
        assert state.sample_count == 0
        assert at.get_threshold() == 0.7

    def test_reset_clears_sample_count(self):
        """Reset should clear all accumulated scores."""
        at = AdaptiveThreshold(base_threshold=0.7, update_interval=5)

        for _ in range(20):
            at.record_score(0.5)

        at.reset()
        state = at.get_state()
        assert state.sample_count == 0
        assert state.last_update_frame == 0
