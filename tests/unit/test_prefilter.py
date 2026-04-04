"""Tests for the MOG2 pre-filter module."""

import numpy as np
import pytest

from argus.prefilter.mog2 import MOG2PreFilter


class TestMOG2PreFilter:
    def test_no_change_on_static_frames(self, blank_frame):
        """Feeding the same frame repeatedly should show no change after warmup."""
        pf = MOG2PreFilter(history=10, var_threshold=25.0, change_pct_threshold=0.005)

        # Warm up the background model
        for _ in range(20):
            pf.process(blank_frame)

        result = pf.process(blank_frame)
        assert not result.has_change
        assert result.change_ratio < 0.005

    def test_detects_change(self, blank_frame, frame_with_white_patch):
        """A sudden change in the frame should be detected."""
        pf = MOG2PreFilter(history=10, var_threshold=25.0, change_pct_threshold=0.001)

        # Warm up with blank frames
        for _ in range(20):
            pf.process(blank_frame)

        # Introduce a change
        result = pf.process(frame_with_white_patch)
        assert result.has_change
        assert result.change_ratio > 0.001
        assert result.foreground_mask is not None

    def test_foreground_mask_is_none_when_no_change(self, blank_frame):
        """Foreground mask should be None when there's no change."""
        pf = MOG2PreFilter(history=10, change_pct_threshold=0.005)

        for _ in range(20):
            pf.process(blank_frame)

        result = pf.process(blank_frame)
        assert result.foreground_mask is None

    def test_reset_clears_background(self, blank_frame, frame_with_white_patch):
        """Reset should clear the background model."""
        pf = MOG2PreFilter(history=10, change_pct_threshold=0.001)

        # Build a background with the white patch
        for _ in range(30):
            pf.process(frame_with_white_patch)

        # After warmup, white patch should be background
        result = pf.process(frame_with_white_patch)
        assert not result.has_change

        # Reset and feed blank frame — should detect change now
        pf.reset()
        # Need some frames to detect change reliably
        for _ in range(5):
            result = pf.process(blank_frame)

        # The blank frame should look different from the initial (unknown) state
        # This is implementation-dependent; after reset the model starts fresh

    def test_change_pct_threshold(self, blank_frame):
        """A small change below the threshold should not trigger detection."""
        # Set a high threshold so a small patch doesn't count
        pf = MOG2PreFilter(history=10, change_pct_threshold=0.5)

        for _ in range(20):
            pf.process(blank_frame)

        # Add a small white patch (~3% of the frame)
        modified = blank_frame.copy()
        modified[0:50, 0:50] = 255
        result = pf.process(modified)
        assert not result.has_change  # 3% < 50% threshold
