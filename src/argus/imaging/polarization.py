"""DoFP (Division-of-Focal-Plane) polarization processing.

Uses the ``polanalyser`` library when available.  If the library is not
installed the processor falls back to passthrough mode: the raw frame is
returned as the deglared image and DoLP / AoLP / Stokes are zeroed out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Optional dependency
# ---------------------------------------------------------------------------
try:
    import polanalyser as pa  # type: ignore[import-untyped]

    _HAS_POLANALYSER = True
except ImportError:
    _HAS_POLANALYSER = False
    logger.warning(
        "polanalyser not installed — polarization processing disabled, "
        "falling back to passthrough mode"
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class PolarizationResult:
    """Output of a single polarization processing pass.

    Attributes:
        deglared: Specular-reflection-suppressed intensity image (H x W x C or H x W).
        dolp: Degree of Linear Polarization map, range [0, 1].
        aolp: Angle of Linear Polarization map, range [0, pi].
        stokes: Stokes vector stack [S0, S1, S2] with shape (3, H, W).
        specular_mask: Binary mask where DoLP exceeds the configured threshold.
    """

    deglared: np.ndarray
    dolp: np.ndarray
    aolp: np.ndarray
    stokes: np.ndarray
    specular_mask: np.ndarray


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------
class PolarizationProcessor:
    """Process raw DoFP polarization mosaic frames.

    Parameters:
        deglare_method: ``"stokes"`` uses ``I_diffuse = S0 * (1 - DoLP)``.
            ``"min_intensity"`` uses ``2 * min(I_0, I_45, I_90, I_135)``.
        dolp_threshold: DoLP value above which a pixel is considered specular.
    """

    def __init__(
        self,
        deglare_method: Literal["stokes", "min_intensity"] = "stokes",
        dolp_threshold: float = 0.3,
    ) -> None:
        if deglare_method not in ("stokes", "min_intensity"):
            raise ValueError(
                f"Unknown deglare_method '{deglare_method}'; "
                "expected 'stokes' or 'min_intensity'"
            )
        self.deglare_method = deglare_method
        self.dolp_threshold = dolp_threshold
        logger.info(
            "polarization_processor.init",
            deglare_method=deglare_method,
            dolp_threshold=dolp_threshold,
            has_polanalyser=_HAS_POLANALYSER,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, raw_frame: np.ndarray) -> PolarizationResult:
        """Demosaic a DoFP frame and compute polarization parameters.

        Args:
            raw_frame: Raw Bayer-like polarization mosaic (H x W) or
                pre-demosaicked 4-channel image (H x W x 4).

        Returns:
            A :class:`PolarizationResult` with deglared image, DoLP, AoLP,
            Stokes vector, and specular mask.
        """
        if not _HAS_POLANALYSER:
            return self._passthrough(raw_frame)

        # --- demosaic ------------------------------------------------
        if raw_frame.ndim == 2:
            channels = pa.demosaicing(raw_frame)  # (4, H, W) or (H, W, 4)
            # Normalise to (H, W, 4)
            if channels.ndim == 3 and channels.shape[0] == 4:
                channels = np.moveaxis(channels, 0, -1)
        elif raw_frame.ndim == 3 and raw_frame.shape[-1] == 4:
            channels = raw_frame
        else:
            logger.warning(
                "polarization.unexpected_shape",
                shape=raw_frame.shape,
            )
            return self._passthrough(raw_frame)

        # channels shape: (H, W, 4)  —  [I_0, I_45, I_90, I_135]
        i0 = channels[..., 0].astype(np.float64)
        i45 = channels[..., 1].astype(np.float64)
        i90 = channels[..., 2].astype(np.float64)
        i135 = channels[..., 3].astype(np.float64)

        # --- Stokes parameters ----------------------------------------
        s0 = 0.5 * (i0 + i45 + i90 + i135)
        s1 = i0 - i90
        s2 = i45 - i135
        stokes = np.stack([s0, s1, s2], axis=0)  # (3, H, W)

        # --- DoLP / AoLP ---------------------------------------------
        dolp = np.zeros_like(s0)
        nonzero = s0 > 0
        dolp[nonzero] = np.sqrt(s1[nonzero] ** 2 + s2[nonzero] ** 2) / s0[nonzero]
        dolp = np.clip(dolp, 0.0, 1.0)

        aolp = 0.5 * np.arctan2(s2, s1)  # range [-pi/2, pi/2]
        aolp = aolp % np.pi  # map to [0, pi]

        # --- deglare --------------------------------------------------
        if self.deglare_method == "stokes":
            deglared = self._deglare_stokes(s0, dolp)
        else:
            deglared = self._deglare_min_intensity(channels)

        specular_mask = (dolp >= self.dolp_threshold).astype(np.uint8)

        return PolarizationResult(
            deglared=deglared.astype(np.uint8) if deglared.max() > 1.0 else deglared,
            dolp=dolp.astype(np.float32),
            aolp=aolp.astype(np.float32),
            stokes=stokes.astype(np.float32),
            specular_mask=specular_mask,
        )

    # ------------------------------------------------------------------
    # Deglare strategies
    # ------------------------------------------------------------------
    @staticmethod
    def _deglare_stokes(s0: np.ndarray, dolp: np.ndarray) -> np.ndarray:
        """Remove specular component via Stokes decomposition.

        ``I_diffuse = S0 * (1 - DoLP)``
        """
        return s0 * (1.0 - dolp)

    @staticmethod
    def _deglare_min_intensity(channels: np.ndarray) -> np.ndarray:
        """Remove specular component via minimum-intensity estimate.

        ``I_diffuse = 2 * min(I_0, I_45, I_90, I_135)``
        """
        return 2.0 * np.min(channels.astype(np.float64), axis=-1)

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------
    @staticmethod
    def _passthrough(raw_frame: np.ndarray) -> PolarizationResult:
        """Return a no-op result when polanalyser is unavailable."""
        h, w = raw_frame.shape[:2]
        return PolarizationResult(
            deglared=raw_frame,
            dolp=np.zeros((h, w), dtype=np.float32),
            aolp=np.zeros((h, w), dtype=np.float32),
            stokes=np.zeros((3, h, w), dtype=np.float32),
            specular_mask=np.zeros((h, w), dtype=np.uint8),
        )
