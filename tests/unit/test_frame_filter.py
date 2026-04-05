"""Tests for frame quality filters (CAP-002~005, CAP-009)."""

from __future__ import annotations

import numpy as np
import cv2
import pytest

from argus.capture.frame_filter import FilterConfig, FilterResult, FrameFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _black_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _white_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _uniform_gray_frame(value: int = 128, h: int = 480, w: int = 640) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _random_frame(h: int = 480, w: int = 640, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _sharp_frame(h: int = 480, w: int = 640, seed: int = 42) -> np.ndarray:
    """A frame with strong edges (high Laplacian variance)."""
    frame = _random_frame(h, w, seed)
    # Add strong edge pattern
    frame[::2, :, :] = 0
    frame[1::2, :, :] = 255
    return frame


def _blurry_frame(h: int = 480, w: int = 640, seed: int = 42) -> np.ndarray:
    """Heavily blurred frame with very low Laplacian variance."""
    frame = _random_frame(h, w, seed)
    return cv2.GaussianBlur(frame, (99, 99), 0)


# ---------------------------------------------------------------------------
# CAP-004: Exposure filtering
# ---------------------------------------------------------------------------

class TestExposureFilter:
    """Tests for exposure / brightness / uniformity filtering."""

    def test_black_frame_rejected_too_dark(self):
        ff = FrameFilter(FilterConfig(
            enable_blur_filter=False,
            enable_person_filter=False,
            enable_dedup_filter=False,
            enable_entropy_filter=False,
        ))
        result = ff.evaluate(_black_frame())
        assert not result.accepted
        assert result.reason == "too_dark"

    def test_white_frame_rejected_too_bright(self):
        ff = FrameFilter(FilterConfig(
            enable_blur_filter=False,
            enable_person_filter=False,
            enable_dedup_filter=False,
            enable_entropy_filter=False,
        ))
        result = ff.evaluate(_white_frame())
        assert not result.accepted
        assert result.reason == "too_bright"

    def test_uniform_gray_rejected_low_std(self):
        ff = FrameFilter(FilterConfig(
            enable_blur_filter=False,
            enable_person_filter=False,
            enable_dedup_filter=False,
            enable_entropy_filter=False,
        ))
        result = ff.evaluate(_uniform_gray_frame(128))
        assert not result.accepted
        assert result.reason == "uniform_frame"

    def test_random_frame_accepted(self):
        ff = FrameFilter(FilterConfig(
            enable_blur_filter=False,
            enable_person_filter=False,
            enable_dedup_filter=False,
            enable_entropy_filter=False,
        ))
        result = ff.evaluate(_random_frame())
        assert result.accepted


# ---------------------------------------------------------------------------
# CAP-002: Blur detection
# ---------------------------------------------------------------------------

class TestBlurFilter:
    """Tests for Laplacian-variance blur detection."""

    def test_blurry_frame_rejected(self):
        ff = FrameFilter(FilterConfig(
            enable_exposure_filter=False,
            enable_person_filter=False,
            enable_dedup_filter=False,
            enable_entropy_filter=False,
            blur_adaptive=False,
        ))
        result = ff.evaluate(_blurry_frame())
        assert not result.accepted
        assert result.reason == "too_blurry"
        assert result.blur_score < 100.0

    def test_sharp_frame_accepted(self):
        ff = FrameFilter(FilterConfig(
            enable_exposure_filter=False,
            enable_person_filter=False,
            enable_dedup_filter=False,
            enable_entropy_filter=False,
            blur_adaptive=False,
        ))
        result = ff.evaluate(_sharp_frame())
        assert result.accepted
        assert result.blur_score > 100.0


# ---------------------------------------------------------------------------
# CAP-009: Entropy / encoder error
# ---------------------------------------------------------------------------

class TestEntropyFilter:
    """Tests for Shannon entropy encoder-error detection."""

    def test_uniform_frame_low_entropy_rejected(self):
        ff = FrameFilter(FilterConfig(
            enable_blur_filter=False,
            enable_exposure_filter=False,
            enable_person_filter=False,
            enable_dedup_filter=False,
        ))
        frame = _uniform_gray_frame(100)
        result = ff.evaluate(frame)
        assert not result.accepted
        assert result.reason == "entropy_too_low"
        assert result.entropy < 3.0

    def test_random_frame_high_entropy_accepted(self):
        ff = FrameFilter(FilterConfig(
            enable_blur_filter=False,
            enable_exposure_filter=False,
            enable_person_filter=False,
            enable_dedup_filter=False,
        ))
        result = ff.evaluate(_random_frame())
        assert result.accepted
        assert result.entropy > 3.0


# ---------------------------------------------------------------------------
# CAP-005: Frame deduplication
# ---------------------------------------------------------------------------

class TestDedupFilter:
    """Tests for SSIM-based frame deduplication."""

    def test_near_duplicate_rejected(self):
        ff = FrameFilter(FilterConfig(
            enable_blur_filter=False,
            enable_exposure_filter=False,
            enable_person_filter=False,
            enable_entropy_filter=False,
        ))
        frame = _random_frame()
        # First frame should be accepted (no previous)
        r1 = ff.evaluate(frame)
        assert r1.accepted

        # Same frame again -> duplicate
        r2 = ff.evaluate(frame)
        assert not r2.accepted
        assert r2.reason == "duplicate"
        assert r2.ssim_score >= 0.98

    def test_different_frames_accepted(self):
        ff = FrameFilter(FilterConfig(
            enable_blur_filter=False,
            enable_exposure_filter=False,
            enable_person_filter=False,
            enable_entropy_filter=False,
        ))
        r1 = ff.evaluate(_random_frame(seed=1))
        assert r1.accepted

        r2 = ff.evaluate(_random_frame(seed=999))
        assert r2.accepted
        assert r2.ssim_score < 0.98


# ---------------------------------------------------------------------------
# Config: all filters disabled
# ---------------------------------------------------------------------------

class TestAllFiltersDisabled:
    """When all filters are off, every frame should be accepted."""

    def test_black_frame_accepted_when_all_disabled(self):
        cfg = FilterConfig(
            enable_blur_filter=False,
            enable_exposure_filter=False,
            enable_person_filter=False,
            enable_dedup_filter=False,
            enable_entropy_filter=False,
        )
        ff = FrameFilter(cfg)
        result = ff.evaluate(_black_frame())
        assert result.accepted

    def test_uniform_frame_accepted_when_all_disabled(self):
        cfg = FilterConfig(
            enable_blur_filter=False,
            enable_exposure_filter=False,
            enable_person_filter=False,
            enable_dedup_filter=False,
            enable_entropy_filter=False,
        )
        ff = FrameFilter(cfg)
        result = ff.evaluate(_uniform_gray_frame(128))
        assert result.accepted


# ---------------------------------------------------------------------------
# FilterResult metadata
# ---------------------------------------------------------------------------

class TestFilterResultMetadata:
    """Verify FilterResult contains correct metric values."""

    def test_metadata_populated(self):
        ff = FrameFilter(FilterConfig(
            enable_person_filter=False,
            enable_dedup_filter=False,
            blur_adaptive=False,
        ))
        frame = _random_frame()
        result = ff.evaluate(frame)
        # Entropy, brightness, blur_score should all be populated
        assert result.entropy > 0.0
        assert result.brightness > 0.0
        assert result.blur_score > 0.0

    def test_person_count_zero_without_detector(self):
        ff = FrameFilter(FilterConfig(
            enable_blur_filter=False,
            enable_exposure_filter=False,
            enable_dedup_filter=False,
            enable_entropy_filter=False,
        ))
        result = ff.evaluate(_random_frame())
        assert result.person_count == 0


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    """Verify reset() clears internal state."""

    def test_reset_clears_dedup_state(self):
        ff = FrameFilter(FilterConfig(
            enable_blur_filter=False,
            enable_exposure_filter=False,
            enable_person_filter=False,
            enable_entropy_filter=False,
        ))
        frame = _random_frame()
        ff.evaluate(frame)

        # Same frame would be duplicate
        r = ff.evaluate(frame)
        assert not r.accepted

        # After reset, same frame should be accepted again
        ff.reset()
        r2 = ff.evaluate(frame)
        assert r2.accepted

    def test_reset_clears_blur_history(self):
        ff = FrameFilter(FilterConfig(
            enable_exposure_filter=False,
            enable_person_filter=False,
            enable_dedup_filter=False,
            enable_entropy_filter=False,
            blur_adaptive=True,
        ))
        # Feed several frames to build blur history
        for _ in range(10):
            ff.evaluate(_sharp_frame())

        ff.reset()
        assert len(ff._blur_history) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: None frame, empty frame."""

    def test_none_frame_rejected(self):
        ff = FrameFilter()
        result = ff.evaluate(None)
        assert not result.accepted
        assert result.reason == "empty_frame"

    def test_empty_frame_rejected(self):
        ff = FrameFilter()
        result = ff.evaluate(np.array([], dtype=np.uint8))
        assert not result.accepted
        assert result.reason == "empty_frame"
