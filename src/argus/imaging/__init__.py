"""Multi-modal imaging: DoFP polarization + NIR strobe fusion."""
from __future__ import annotations

from argus.imaging.acquisition import FrameMetadata, IndustrialCameraCapture
from argus.imaging.fusion import FusedFrame, ModalityFusion
from argus.imaging.polarization import PolarizationProcessor, PolarizationResult

__all__ = [
    "PolarizationProcessor",
    "PolarizationResult",
    "IndustrialCameraCapture",
    "FrameMetadata",
    "ModalityFusion",
    "FusedFrame",
]
