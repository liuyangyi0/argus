"""Tests for phase-correlation frame alignment."""

from __future__ import annotations

import numpy as np
import pytest

from argus.config.schema import AlignmentConfig
from argus.preprocessing.alignment import PhaseCorrelator, create_from_config


def _textured_frame(seed: int = 42, size: tuple[int, int] = (240, 320)) -> np.ndarray:
    """Build a BGR frame with enough texture for phase correlation to lock onto."""
    rng = np.random.default_rng(seed)
    gray = rng.integers(30, 220, size=size, dtype=np.uint8)
    h, w = size
    # Add strong feature blobs so FFT peak is unambiguous at sub-pixel level
    for cy, cx in ((60, 80), (160, 220), (100, 250)):
        yy, xx = np.ogrid[:h, :w]
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < 15 ** 2
        gray[mask] = 255
    return np.stack([gray, gray, gray], axis=-1)


def _shift_frame(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate ``frame`` by ``(dx, dy)`` pixels via OpenCV warpAffine."""
    import cv2

    h, w = frame.shape[:2]
    m = np.float64([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        frame, m, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


class TestPhaseCorrelatorIdentity:
    def test_first_call_adopts_reference_and_returns_zero_shift(self):
        pc = PhaseCorrelator()
        frame = _textured_frame()
        aligned, (dx, dy) = pc.align(frame, now=0.0)

        assert dx == 0.0
        assert dy == 0.0
        assert aligned is frame

    def test_identity_shift_is_near_zero(self):
        pc = PhaseCorrelator(downsample=2)
        frame = _textured_frame()
        pc.align(frame, now=0.0)

        aligned, (dx, dy) = pc.align(frame, now=0.1)
        assert abs(dx) < 0.5
        assert abs(dy) < 0.5
        assert aligned.shape == frame.shape


class TestPhaseCorrelatorTranslation:
    def test_detects_pure_translation(self):
        pc = PhaseCorrelator(downsample=2, max_shift_px=10.0)
        ref = _textured_frame()
        pc.align(ref, now=0.0)

        shifted = _shift_frame(ref, dx=3.0, dy=2.0)
        _aligned, (dx, dy) = pc.align(shifted, now=0.1)

        assert dx == pytest.approx(3.0, abs=0.6)
        assert dy == pytest.approx(2.0, abs=0.6)

    def test_aligned_frame_matches_reference(self):
        """After alignment, the warped frame should closely resemble the reference."""
        pc = PhaseCorrelator(downsample=2, max_shift_px=10.0)
        ref = _textured_frame()
        pc.align(ref, now=0.0)

        shifted = _shift_frame(ref, dx=2.0, dy=1.0)
        aligned, _shift = pc.align(shifted, now=0.1)

        # Compare central region to avoid warpAffine border artifacts
        ref_center = ref[20:-20, 20:-20].astype(np.float32)
        aligned_center = aligned[20:-20, 20:-20].astype(np.float32)
        shifted_center = shifted[20:-20, 20:-20].astype(np.float32)

        err_aligned = float(np.mean(np.abs(aligned_center - ref_center)))
        err_shifted = float(np.mean(np.abs(shifted_center - ref_center)))
        assert err_aligned < err_shifted


class TestPhaseCorrelatorOutlierSkip:
    def test_large_shift_returns_original_frame(self):
        pc = PhaseCorrelator(downsample=2, max_shift_px=5.0)
        ref = _textured_frame()
        pc.align(ref, now=0.0)

        far = _shift_frame(ref, dx=50.0, dy=50.0)
        aligned, (dx, dy) = pc.align(far, now=0.1)

        assert aligned is far
        # Measurement still reported so the caller can log the outlier
        assert abs(dx) > 5.0 or abs(dy) > 5.0


class TestPhaseCorrelatorReferenceRefresh:
    def test_reference_refreshes_after_interval(self):
        pc = PhaseCorrelator(downsample=2, ref_update_interval_s=10.0)
        ref_a = _textured_frame(seed=1)
        pc.align(ref_a, now=0.0)

        ref_b = _textured_frame(seed=7)
        aligned_b, (dx_b, dy_b) = pc.align(ref_b, now=11.0)
        assert dx_b == 0.0 and dy_b == 0.0
        assert aligned_b is ref_b

        shifted = _shift_frame(ref_b, dx=2.0, dy=0.0)
        _aligned, (dx, dy) = pc.align(shifted, now=11.5)
        assert dx == pytest.approx(2.0, abs=0.6)
        assert dy == pytest.approx(0.0, abs=0.6)

    def test_reference_retained_within_interval(self):
        pc = PhaseCorrelator(downsample=2, ref_update_interval_s=60.0)
        ref = _textured_frame()
        pc.align(ref, now=0.0)

        shifted = _shift_frame(ref, dx=1.5, dy=1.0)
        _aligned, (dx, dy) = pc.align(shifted, now=30.0)
        assert dx == pytest.approx(1.5, abs=0.6)
        assert dy == pytest.approx(1.0, abs=0.6)


class TestPhaseCorrelatorValidation:
    def test_rejects_invalid_params(self):
        with pytest.raises(ValueError):
            PhaseCorrelator(max_shift_px=0.0)
        with pytest.raises(ValueError):
            PhaseCorrelator(downsample=0)
        with pytest.raises(ValueError):
            PhaseCorrelator(ref_update_interval_s=0.0)

    def test_rejects_empty_frame(self):
        pc = PhaseCorrelator()
        with pytest.raises(ValueError):
            pc.align(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_shape_change_resets_reference(self):
        pc = PhaseCorrelator(downsample=2)
        small = _textured_frame(size=(120, 160))
        _aligned, shift_a = pc.align(small, now=0.0)
        assert shift_a == (0.0, 0.0)

        large = _textured_frame(size=(240, 320))
        aligned, shift_b = pc.align(large, now=0.1)
        # Different shape → adopt new reference, zero shift
        assert shift_b == (0.0, 0.0)
        assert aligned is large


class TestCreateFromConfig:
    def test_builds_correlator_from_config(self):
        cfg = AlignmentConfig(
            enabled=True,
            max_shift_px=3.0,
            downsample=2,
            ref_update_interval_s=30.0,
        )
        pc = create_from_config(cfg)
        assert isinstance(pc, PhaseCorrelator)
        assert pc.max_shift_px == 3.0
        assert pc.downsample == 2
        assert pc.ref_update_interval_s == 30.0


class TestPhaseCorrelatorReset:
    def test_reset_drops_reference(self):
        pc = PhaseCorrelator(downsample=2)
        frame = _textured_frame()
        pc.align(frame, now=0.0)
        pc.reset()

        # After reset, next call should adopt a new reference
        _aligned, (dx, dy) = pc.align(frame, now=0.1)
        assert dx == 0.0 and dy == 0.0
