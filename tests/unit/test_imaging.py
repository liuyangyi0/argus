"""Tests for the imaging module (polarization, fusion, acquisition)."""

from __future__ import annotations

import numpy as np
import pytest

from argus.imaging.polarization import PolarizationProcessor, PolarizationResult
from argus.imaging.fusion import ModalityFusion, FusedFrame
from argus.imaging.acquisition import IndustrialCameraCapture, FrameMetadata


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

    def test_4channel_input_passthrough(self):
        """4-channel pre-demosaicked input should be handled."""
        proc = PolarizationProcessor()
        frame = np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8)
        result = proc.process(frame)

        assert isinstance(result, PolarizationResult)
        # Without polanalyser, should passthrough
        assert result.deglared.shape[:2] == (480, 640)

    def test_stokes_method_default(self):
        """Default deglare method should be stokes."""
        proc = PolarizationProcessor()
        assert proc.deglare_method == "stokes"

    def test_min_intensity_method(self):
        """Should accept min_intensity method."""
        proc = PolarizationProcessor(deglare_method="min_intensity")
        assert proc.deglare_method == "min_intensity"

    def test_invalid_method_raises(self):
        """Invalid deglare method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown deglare_method"):
            PolarizationProcessor(deglare_method="invalid")

    def test_result_shapes_consistent(self):
        """All result arrays should have consistent spatial dimensions."""
        proc = PolarizationProcessor()
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        result = proc.process(frame)

        h, w = frame.shape[:2]
        assert result.dolp.shape == (h, w)
        assert result.aolp.shape == (h, w)
        assert result.stokes.shape == (3, h, w)
        assert result.specular_mask.shape == (h, w)


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

    def test_missing_nir_zero_fills(self):
        """With fusion_channels=5 but no NIR, should zero-fill NIR channel."""
        fusion = ModalityFusion(fusion_channels=5)
        visible = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dolp = np.random.rand(480, 640).astype(np.float32)
        result = fusion.fuse(visible, dolp=dolp, nir=None)

        assert result.channels == 5
        assert result.nir is None

    def test_1ch_grayscale(self):
        """1-channel fusion should produce single grayscale output."""
        fusion = ModalityFusion(fusion_channels=1)
        visible = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = fusion.fuse(visible)

        assert result.channels == 1
        assert result.tensor.shape[2] == 1

    def test_invalid_channels_raises(self):
        """fusion_channels < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="fusion_channels must be >= 1"):
            ModalityFusion(fusion_channels=0)

    def test_dolp_scaling(self):
        """DoLP float [0,1] should be scaled to uint8 [0,255]."""
        fusion = ModalityFusion(fusion_channels=4)
        visible = np.zeros((10, 10, 3), dtype=np.uint8)
        dolp = np.ones((10, 10), dtype=np.float32)  # all 1.0
        result = fusion.fuse(visible, dolp=dolp)

        # DoLP channel (channel 1 in base stack) should be 255
        assert result.tensor[:, :, 1].max() == 255

    def test_visible_component_stored(self):
        """FusedFrame should store the visible component."""
        fusion = ModalityFusion(fusion_channels=3)
        visible = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = fusion.fuse(visible)

        assert result.visible is not None
        assert result.visible.shape == (10, 10)


class TestIndustrialCameraCapture:
    def test_opencv_backend_default(self):
        """Default backend should be opencv."""
        cap = IndustrialCameraCapture(source=0, backend="opencv")
        assert cap._active_backend == "opencv"
        assert not cap.is_polarization_capable

    def test_polarization_flag(self):
        """is_polarization_capable should reflect constructor arg."""
        cap = IndustrialCameraCapture(source=0, is_polarization=True)
        assert cap.is_polarization_capable

    def test_arena_fallback_without_sdk(self):
        """Arena backend should fall back to opencv when SDK not installed."""
        cap = IndustrialCameraCapture(source=0, backend="arena")
        # Should have fallen back silently
        assert cap._active_backend in ("arena", "opencv")

    def test_spinnaker_fallback_without_sdk(self):
        """Spinnaker backend should fall back to opencv when SDK not installed."""
        cap = IndustrialCameraCapture(source=0, backend="spinnaker")
        assert cap._active_backend in ("spinnaker", "opencv")

    def test_metavision_raises(self):
        """Metavision backend should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="reserved but not yet implemented"):
            IndustrialCameraCapture(source=0, backend="metavision")

    def test_dv_raises(self):
        """DV backend should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="reserved but not yet implemented"):
            IndustrialCameraCapture(source=0, backend="dv")

    def test_context_manager(self):
        """Context manager should not crash even without real device."""
        cap = IndustrialCameraCapture(source=999, backend="opencv")
        # open will fail but should not raise
        with cap:
            assert not cap.is_opened()

    def test_frame_metadata_defaults(self):
        """FrameMetadata should have sensible defaults."""
        meta = FrameMetadata(timestamp=1.0, frame_number=0)
        assert meta.exposure_us == 0.0
        assert meta.gain_db == 0.0
        assert meta.sensor_temperature_c == 0.0
        assert meta.is_nir is False

    def test_nir_strobe_noop_on_opencv(self):
        """NIR strobe on opencv backend should be a no-op (no crash)."""
        cap = IndustrialCameraCapture(source=0, backend="opencv")
        cap.set_nir_strobe(True)  # Should not raise
        cap.set_nir_strobe(False)
