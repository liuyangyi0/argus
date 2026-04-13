"""Industrial camera acquisition adapter.

Supports three backends:

* **opencv** — fully implemented via ``cv2.VideoCapture``.
* **arena** — Lucid Vision ARENA SDK (stub, falls back to opencv).
* **spinnaker** — FLIR Spinnaker SDK (stub, falls back to opencv).
* **metavision** — Prophesee event camera SDK (reserved, not implemented).
* **dv** — iniVation DV SDK (reserved, not implemented).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Metadata container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FrameMetadata:
    """Per-frame acquisition metadata.

    Attributes:
        timestamp: Monotonic capture timestamp in seconds.
        frame_number: Incrementing counter for the capture session.
        exposure_us: Exposure time in microseconds (0 if unavailable).
        is_nir: ``True`` when the frame was acquired under NIR strobe.
    """

    timestamp: float
    frame_number: int
    exposure_us: float = 0.0
    is_nir: bool = False


# ---------------------------------------------------------------------------
# Capture adapter
# ---------------------------------------------------------------------------


class IndustrialCameraCapture:
    """Unified capture interface for industrial cameras.

    Parameters:
        source: Device index (``int``) or URI string passed to the backend.
        backend: One of ``"opencv"``, ``"arena"``, ``"spinnaker"``.
        is_polarization: Hint that the sensor carries a DoFP polarization
            mosaic.  Stored as a property; does **not** alter capture
            behaviour.
    """

    def __init__(
        self,
        source: int | str = 0,
        backend: Literal["opencv", "arena", "spinnaker", "metavision", "dv"] = "opencv",
        is_polarization: bool = False,
    ) -> None:
        self._source = source
        self._requested_backend = backend
        self._active_backend = backend
        self._is_polarization = is_polarization
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_counter: int = 0

        if backend in ("arena", "spinnaker"):
            logger.warning(
                "acquisition.backend_stub",
                backend=backend,
                msg=f"{backend} SDK not integrated — falling back to opencv",
            )
            self._active_backend = "opencv"

        if backend in ("metavision", "dv"):
            raise NotImplementedError(
                f"Event camera backend '{backend}' is reserved but not yet implemented. "
                f"Install the corresponding SDK (metavision-sdk or dv-python) and implement "
                f"EventCameraCapture when hardware is available."
            )

        logger.info(
            "acquisition.init",
            source=source,
            requested_backend=backend,
            active_backend=self._active_backend,
            is_polarization=is_polarization,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_polarization_capable(self) -> bool:
        """Whether the connected sensor is flagged as a DoFP polarization camera."""
        return self._is_polarization

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Open the capture device.

        Returns:
            ``True`` if the device was opened successfully.
        """
        if self._cap is not None and self._cap.isOpened():
            logger.debug("acquisition.already_open", source=self._source)
            return True

        if isinstance(self._source, int):
            self._cap = cv2.VideoCapture(self._source, cv2.CAP_ANY)
        else:
            self._cap = cv2.VideoCapture(self._source)

        opened = self._cap.isOpened()
        if opened:
            self._frame_counter = 0
            logger.info("acquisition.opened", source=self._source)
        else:
            logger.error("acquisition.open_failed", source=self._source)
        return opened

    def close(self) -> None:
        """Release the capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._frame_counter = 0
            logger.info("acquisition.closed", source=self._source)

    def is_opened(self) -> bool:
        """Check whether the capture device is currently open."""
        return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    def grab_frame(self) -> tuple[np.ndarray | None, FrameMetadata]:
        """Grab a single frame from the capture device.

        Returns:
            A tuple of ``(frame, metadata)``.  ``frame`` is ``None`` when
            the grab fails.
        """
        if self._cap is None or not self._cap.isOpened():
            logger.error("acquisition.grab_not_open", source=self._source)
            return None, FrameMetadata(
                timestamp=time.monotonic(),
                frame_number=self._frame_counter,
            )

        ret, frame = self._cap.read()

        ts = time.monotonic()
        self._frame_counter += 1

        meta = FrameMetadata(
            timestamp=ts,
            frame_number=self._frame_counter,
            exposure_us=self._read_exposure_us(),
            is_nir=False,
        )

        if not ret or frame is None:
            logger.warning(
                "acquisition.grab_failed",
                source=self._source,
                frame_number=meta.frame_number,
            )
            return None, meta

        return frame, meta

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_exposure_us(self) -> float:
        """Best-effort read of the current exposure time in microseconds."""
        if self._cap is None:
            return 0.0
        val = self._cap.get(cv2.CAP_PROP_EXPOSURE)
        if val <= 0:
            return 0.0
        # OpenCV often returns the value in seconds or log-scale; convert
        # conservatively.  Positive values < 1 are assumed to be seconds.
        if val < 1.0:
            return val * 1_000_000.0
        return val

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> IndustrialCameraCapture:
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
