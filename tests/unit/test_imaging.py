"""Tests for the imaging module (polarization, fusion)."""

from __future__ import annotations

import numpy as np
import pytest

from argus.imaging.polarization import PolarizationProcessor, PolarizationResult
from argus.imaging.fusion import ModalityFusion, FusedFrame


class TestPolarizationProcessor:
    def test_fallback_when_polanalyser_unavailable(self):
        """Without polanalyser, process should return the input frame as deglared."""
        proc = PolarizationProcessor()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = proc.process(frame)

        assert isinstance(result, PolarizationResult)
        assert result.deglared.shape == frame.shape
        # DoLP should be zeros when no polarization processing
        assert result.dolp.shape[:2] == frame.shape[:2]

    def test_grayscale_input(self):
        """Should handle grayscale (2D) input without crashing."""
        proc = PolarizationProcessor()
        frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        result = proc.process(frame)

        assert isinstance(result, PolarizationResult)


class TestModalityFusion:
    def test_3ch_visible_only(self):
        """3-channel fusion should output (H, W, 3) from visible input."""
        fusion = ModalityFusion(fusion_channels=3)
        visible = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = fusion.fuse(visible)

        assert isinstance(result, FusedFrame)
        assert result.channels == 3
        assert result.tensor.shape == (480, 640, 3)

    def test_4ch_with_dolp(self):
        """4-channel fusion should stack visible + DoLP."""
        fusion = ModalityFusion(fusion_channels=4)
        visible = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dolp = np.random.rand(480, 640).astype(np.float32)
        result = fusion.fuse(visible, dolp=dolp)

        assert result.channels == 4
        assert result.tensor.shape[2] == 4

    def test_5ch_with_dolp_and_nir(self):
        """5-channel fusion should stack visible + DoLP + NIR."""
        fusion = ModalityFusion(fusion_channels=5)
        visible = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dolp = np.random.rand(480, 640).astype(np.float32)
        nir = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        result = fusion.fuse(visible, dolp=dolp, nir=nir)

        assert result.channels == 5
        assert result.tensor.shape[2] == 5

    def test_missing_dolp_zero_fills(self):
        """With fusion_channels=4 but no DoLP, should zero-fill."""
        fusion = ModalityFusion(fusion_channels=4)
        visible = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = fusion.fuse(visible, dolp=None)

        assert result.channels == 4
        # 4th channel should be all zeros
        assert result.tensor[:, :, 3].sum() == 0
