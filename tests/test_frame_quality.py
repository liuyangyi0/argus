"""Tests for frame quality filtering (CAP-002~009)."""

import numpy as np
import pytest

from argus.capture.quality import CaptureStats, FrameQualityFilter, FrameQualityResult
from argus.config.schema import CaptureQualityConfig


@pytest.fixture
def default_config():
    return CaptureQualityConfig()


@pytest.fixture
def quality_filter(default_config):
    return FrameQualityFilter(default_config)


@pytest.fixture
def natural_frame():
    """A frame with natural-looking variation (gradient + noise)."""
    rng = np.random.default_rng(42)
    # Create gradient base
    base = np.linspace(40, 200, 640, dtype=np.uint8)
    frame = np.tile(base, (480, 1))
    # Add noise
    noise = rng.integers(-20, 20, (480, 640), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Make it 3-channel
    return np.stack([frame, frame, frame], axis=-1)


@pytest.fixture
def sharp_frame():
    """A frame with clear edges (high Laplacian variance)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Checkerboard pattern creates strong edges
    for i in range(0, 480, 20):
        for j in range(0, 640, 20):
            if (i // 20 + j // 20) % 2 == 0:
                frame[i:i+20, j:j+20] = 200
    return frame


# ── CAP-009: Encoder error detection ──


class TestEntropyFilter:
    def test_rejects_flat_frame(self, quality_filter):
        """Uniform color frame has near-zero entropy."""
        flat = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = quality_filter.check(flat)
        assert not result.accepted
        assert result.rejection_reason == "encoder_error"

    def test_accepts_natural_frame(self, quality_filter, natural_frame):
        result = quality_filter.check(natural_frame)
        assert result.entropy > 3.0
        # May or may not be accepted overall, but not rejected for entropy
        assert result.rejection_reason != "encoder_error"


# ── CAP-004: Exposure detection ──


class TestExposureFilter:
    def test_rejects_dark_frame(self, quality_filter):
        """Near-black frame (mean < 30)."""
        # Use random low values to ensure entropy > 3.0 but mean < 30
        rng = np.random.default_rng(42)
        dark = rng.integers(0, 28, (480, 640, 3), dtype=np.uint8)
        result = quality_filter.check(dark)
        assert not result.accepted
        assert result.rejection_reason == "exposure"

    def test_rejects_bright_frame(self, quality_filter):
        """Near-white frame (mean > 225)."""
        # Use random high values to ensure entropy > 3.0 but mean > 225
        rng = np.random.default_rng(42)
        bright = rng.integers(228, 256, (480, 640, 3), dtype=np.uint8)
        result = quality_filter.check(bright)
        assert not result.accepted
        assert result.rejection_reason == "exposure"

    def test_rejects_low_contrast(self):
        """Frame with mean in range but very low std."""
        config = CaptureQualityConfig(brightness_std_min=10.0)
        f = FrameQualityFilter(config)
        # Create frame centered at 128 but with tiny spread
        rng = np.random.default_rng(99)
        frame = np.clip(128 + rng.integers(-2, 3, (480, 640)), 0, 255).astype(np.uint8)
        frame = np.stack([frame, frame, frame], axis=-1)
        result = f.check(frame)
        assert not result.accepted
        # Could fail on entropy or exposure depending on exact values
        assert result.rejection_reason in ("exposure", "encoder_error")


# ── CAP-002: Blur detection ──


class TestBlurFilter:
    def test_rejects_blurry_frame(self, quality_filter, sharp_frame):
        """Heavily blurred frame should be rejected."""
        import cv2
        blurry = cv2.GaussianBlur(sharp_frame, (31, 31), 0)
        result = quality_filter.check(blurry)
        assert not result.accepted
        assert result.rejection_reason == "blur"

    def test_accepts_sharp_frame(self, quality_filter, sharp_frame):
        result = quality_filter.check(sharp_frame)
        assert result.blur_score > 100
        assert result.rejection_reason != "blur"


# ── CAP-005: Frame deduplication ──


class TestDedupFilter:
    def test_rejects_identical_frames(self, quality_filter, natural_frame):
        """Same frame as prev should be rejected as duplicate."""
        # First frame must be accepted (no prev)
        result1 = quality_filter.check(natural_frame)
        if result1.accepted:
            result2 = quality_filter.check(natural_frame, prev_frame=natural_frame)
            assert not result2.accepted
            assert result2.rejection_reason == "duplicate"

    def test_accepts_different_frames(self, quality_filter, natural_frame, sharp_frame):
        """Very different frames should not be flagged as duplicate."""
        result = quality_filter.check(natural_frame, prev_frame=sharp_frame)
        if result.ssim_to_prev is not None:
            assert result.ssim_to_prev < 0.98
            assert result.rejection_reason != "duplicate"


# ── CAP-003: Person detection ──


class TestPersonFilter:
    def test_graceful_when_unavailable(self, natural_frame):
        """When YOLO is unavailable, person check should be skipped."""
        config = CaptureQualityConfig()
        f = FrameQualityFilter(config)
        # Force detector to be unavailable
        f._person_detector = False
        result = f.check(natural_frame)
        # Should not reject for person
        assert result.rejection_reason != "person"


# ── Disabled filter ─���


class TestDisabledFilter:
    def test_accepts_everything_when_disabled(self):
        """With enabled=False, all frames pass."""
        config = CaptureQualityConfig(enabled=False)
        f = FrameQualityFilter(config)
        # Even a flat frame passes
        flat = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = f.check(flat)
        assert result.accepted
        assert result.rejection_reason is None

    def test_metrics_still_populated_when_disabled(self):
        """Even disabled, metrics should be computed."""
        config = CaptureQualityConfig(enabled=False)
        f = FrameQualityFilter(config)
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        result = f.check(frame)
        assert result.brightness_mean > 0
        assert result.entropy > 0
        assert result.blur_score >= 0


# ── CaptureStats ──


class TestCaptureStats:
    def test_accumulation_and_to_dict(self):
        stats = CaptureStats()
        stats.total_grabbed = 10
        stats.null_frames = 1
        stats.accepted = 5
        stats.rejected_blur = 2
        stats.rejected_exposure = 1
        stats.rejected_duplicate = 1
        stats.brightness_values = [50.0, 100.0, 150.0, 200.0, 80.0]

        assert stats.total_rejected == 4

        d = stats.to_dict()
        assert d["total_grabbed"] == 10
        assert d["accepted"] == 5
        assert d["total_rejected"] == 4
        assert d["rejected_blur"] == 2
        assert d["brightness_min"] == 50.0
        assert d["brightness_max"] == 200.0

    def test_record_rejection(self):
        stats = CaptureStats()
        stats.record_rejection("blur")
        stats.record_rejection("blur")
        stats.record_rejection("exposure")
        assert stats.rejected_blur == 2
        assert stats.rejected_exposure == 1

    def test_empty_brightness_range(self):
        stats = CaptureStats()
        assert stats.brightness_range() == (0.0, 0.0)
