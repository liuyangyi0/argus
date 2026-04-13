"""Multi-modal fusion: visible + polarization DoLP + NIR strobe.

Stacks available modalities into a single multi-channel tensor suitable for
downstream detection / classification networks.  Missing modalities are
zero-filled so the output shape is deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FusedFrame:
    """Output of the modality fusion step.

    Attributes:
        tensor: Fused image array with shape ``(H, W, C)`` in ``uint8``.
        channels: Number of channels in ``tensor``.
        visible: The visible-light component (or ``None``).
        dolp: The DoLP component (or ``None``).
        nir: The NIR component (or ``None``).
    """

    tensor: np.ndarray
    channels: int
    visible: Optional[np.ndarray] = None
    dolp: Optional[np.ndarray] = None
    nir: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Fusion processor
# ---------------------------------------------------------------------------


class ModalityFusion:
    """Fuse visible, DoLP, and NIR modalities into one tensor.

    Parameters:
        fusion_channels: Number of output channels.  Must be 1, 3, or a
            positive integer >= the number of provided modalities.
    """

    def __init__(self, fusion_channels: int = 3) -> None:
        if fusion_channels < 1:
            raise ValueError("fusion_channels must be >= 1")
        self.fusion_channels = fusion_channels
        logger.info("modality_fusion.init", fusion_channels=fusion_channels)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        visible: np.ndarray,
        dolp: Optional[np.ndarray] = None,
        nir: Optional[np.ndarray] = None,
    ) -> FusedFrame:
        """Fuse available modalities into a single multi-channel image.

        Args:
            visible: Visible-light image, ``(H, W)`` or ``(H, W, C)``.
            dolp: Degree-of-linear-polarization map ``(H, W)``, float
                in ``[0, 1]``.  Zero-filled if ``None``.
            nir: Near-infrared image ``(H, W)`` or ``(H, W, 1)``.
                Zero-filled if ``None``.

        Returns:
            A :class:`FusedFrame` with the stacked tensor.
        """
        h, w = visible.shape[:2]

        vis = self._to_single_channel_uint8(visible)
        dolp_u8 = self._dolp_to_uint8(dolp, h, w)
        nir_u8 = self._to_single_channel_uint8(nir) if nir is not None else np.zeros(
            (h, w), dtype=np.uint8
        )

        # Stack the three base channels: visible, dolp, nir
        base = np.stack([vis, dolp_u8, nir_u8], axis=-1)  # (H, W, 3)

        # Adapt to requested channel count
        tensor = self._adapt_channels(base, self.fusion_channels)

        logger.debug(
            "modality_fusion.fused",
            shape=tensor.shape,
            has_dolp=dolp is not None,
            has_nir=nir is not None,
        )

        return FusedFrame(
            tensor=tensor,
            channels=tensor.shape[-1],
            visible=vis,
            dolp=dolp_u8,
            nir=nir_u8 if nir is not None else None,
        )

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_single_channel_uint8(img: np.ndarray) -> np.ndarray:
        """Convert an image to a single-channel uint8 array."""
        if img.ndim == 3:
            if img.shape[-1] == 1:
                img = img[..., 0]
            else:
                import cv2

                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img.dtype == np.float32 or img.dtype == np.float64:
            # Assume [0, 1] float range
            if img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255)
            else:
                img = img.clip(0, 255)
            return img.astype(np.uint8)

        return img.astype(np.uint8)

    @staticmethod
    def _dolp_to_uint8(dolp: Optional[np.ndarray], h: int, w: int) -> np.ndarray:
        """Scale a [0, 1] DoLP map to uint8, or return zeros."""
        if dolp is None:
            return np.zeros((h, w), dtype=np.uint8)
        scaled = (np.clip(dolp, 0.0, 1.0) * 255.0).astype(np.uint8)
        if scaled.ndim == 3:
            scaled = scaled[..., 0]
        return scaled

    @staticmethod
    def _adapt_channels(base: np.ndarray, target: int) -> np.ndarray:
        """Pad or truncate a (H, W, 3) array to (H, W, target)."""
        _, _, c = base.shape
        if target == c:
            return base
        if target < c:
            return base[..., :target]
        # Pad with zeros
        h, w = base.shape[:2]
        padding = np.zeros((h, w, target - c), dtype=base.dtype)
        return np.concatenate([base, padding], axis=-1)
