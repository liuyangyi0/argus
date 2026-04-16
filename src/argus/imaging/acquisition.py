"""Industrial camera acquisition adapter.

Supports three backends:

* **opencv** — fully implemented via ``cv2.VideoCapture``.
* **arena** — Lucid Vision ARENA SDK (DoFP polarization cameras).
* **spinnaker** — FLIR Spinnaker SDK (machine vision cameras).
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
# Optional SDK imports
# ---------------------------------------------------------------------------
try:
    from arena_api.system import system as _arena_system  # type: ignore[import-untyped]
    from arena_api import enums as _arena_enums  # type: ignore[import-untyped]

    _HAS_ARENA = True
except ImportError:
    _HAS_ARENA = False

try:
    import PySpin  # type: ignore[import-untyped]

    _HAS_PYSPIN = True
except ImportError:
    _HAS_PYSPIN = False


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
        gain_db: Analog gain in dB (0 if unavailable).
        sensor_temperature_c: Sensor temperature in celsius (0 if unavailable).
        is_nir: ``True`` when the frame was acquired under NIR strobe.
    """

    timestamp: float
    frame_number: int
    exposure_us: float = 0.0
    gain_db: float = 0.0
    sensor_temperature_c: float = 0.0
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

        # Arena SDK state
        self._arena_device = None
        self._arena_nodemap = None

        # Spinnaker SDK state
        self._spin_system = None
        self._spin_camera = None

        if backend == "arena" and not _HAS_ARENA:
            logger.warning(
                "acquisition.arena_not_installed",
                msg="arena_api not installed — falling back to opencv. "
                "Install with: pip install arena-api",
            )
            self._active_backend = "opencv"

        if backend == "spinnaker" and not _HAS_PYSPIN:
            logger.warning(
                "acquisition.pyspin_not_installed",
                msg="PySpin not installed — falling back to opencv. "
                "Install from FLIR Spinnaker SDK package.",
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
        if self._active_backend == "arena":
            return self._open_arena()
        elif self._active_backend == "spinnaker":
            return self._open_spinnaker()
        return self._open_opencv()

    def close(self) -> None:
        """Release the capture device."""
        if self._active_backend == "arena":
            self._close_arena()
        elif self._active_backend == "spinnaker":
            self._close_spinnaker()
        else:
            self._close_opencv()
        self._frame_counter = 0

    def is_opened(self) -> bool:
        """Check whether the capture device is currently open."""
        if self._active_backend == "arena":
            return self._arena_device is not None
        elif self._active_backend == "spinnaker":
            return self._spin_camera is not None and self._spin_camera.IsStreaming()
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
        if self._active_backend == "arena":
            return self._grab_arena()
        elif self._active_backend == "spinnaker":
            return self._grab_spinnaker()
        return self._grab_opencv()

    # ------------------------------------------------------------------
    # GPIO / NIR strobe
    # ------------------------------------------------------------------

    def set_nir_strobe(self, enabled: bool) -> None:
        """Enable or disable NIR strobe output via ExposureActive GPIO.

        This configures the camera to drive a GPIO line high during
        exposure, which triggers an external NIR LED driver.
        """
        if self._active_backend == "arena" and self._arena_device is not None:
            self._arena_set_nir_strobe(enabled)
        elif self._active_backend == "spinnaker" and self._spin_camera is not None:
            self._spinnaker_set_nir_strobe(enabled)
        else:
            logger.debug(
                "acquisition.nir_strobe_not_supported",
                backend=self._active_backend,
                msg="NIR strobe requires arena or spinnaker backend",
            )

    # ==================================================================
    # OpenCV backend
    # ==================================================================

    def _open_opencv(self) -> bool:
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
            logger.info("acquisition.opened", source=self._source, backend="opencv")
        else:
            logger.error("acquisition.open_failed", source=self._source, backend="opencv")
        return opened

    def _close_opencv(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("acquisition.closed", source=self._source, backend="opencv")

    def _grab_opencv(self) -> tuple[np.ndarray | None, FrameMetadata]:
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
            exposure_us=self._read_exposure_us_opencv(),
        )

        if not ret or frame is None:
            logger.warning(
                "acquisition.grab_failed",
                source=self._source,
                frame_number=meta.frame_number,
            )
            return None, meta

        return frame, meta

    def _read_exposure_us_opencv(self) -> float:
        if self._cap is None:
            return 0.0
        val = self._cap.get(cv2.CAP_PROP_EXPOSURE)
        if val <= 0:
            return 0.0
        if val < 1.0:
            return val * 1_000_000.0
        return val

    # ==================================================================
    # Arena (LUCID) backend
    # ==================================================================

    def _open_arena(self) -> bool:
        if self._arena_device is not None:
            logger.debug("acquisition.arena_already_open", source=self._source)
            return True

        try:
            _arena_system.destroy_device()
            devices = _arena_system.create_device()
            if not devices:
                logger.error("acquisition.arena_no_devices", msg="No LUCID devices found")
                return False

            # Select device by index or serial number
            if isinstance(self._source, int):
                if self._source >= len(devices):
                    logger.error(
                        "acquisition.arena_device_index_oob",
                        source=self._source,
                        device_count=len(devices),
                    )
                    return False
                self._arena_device = devices[self._source]
            else:
                # Match by serial number string
                target_serial = str(self._source)
                matched = [
                    d for d in devices
                    if str(d.nodemap["DeviceSerialNumber"].value) == target_serial
                ]
                if not matched:
                    logger.error(
                        "acquisition.arena_serial_not_found",
                        serial=target_serial,
                        available=[
                            str(d.nodemap["DeviceSerialNumber"].value) for d in devices
                        ],
                    )
                    return False
                self._arena_device = matched[0]

            self._arena_nodemap = self._arena_device.nodemap

            # Configure for polarization if needed
            if self._is_polarization:
                self._arena_configure_polarization()

            # Start stream
            self._arena_device.start_stream()
            self._frame_counter = 0
            logger.info(
                "acquisition.opened",
                source=self._source,
                backend="arena",
                serial=str(self._arena_nodemap["DeviceSerialNumber"].value),
                model=str(self._arena_nodemap["DeviceModelName"].value),
            )
            return True
        except Exception as e:
            logger.error("acquisition.arena_open_failed", error=str(e))
            self._arena_device = None
            return False

    def _arena_configure_polarization(self) -> None:
        """Configure DoFP polarization demosaicing on the camera side."""
        try:
            nm = self._arena_nodemap
            # Set pixel format to polarized output if available
            pixel_format = nm["PixelFormat"]
            polarized_formats = [
                f for f in pixel_format.symbolics
                if "Polar" in f or "BayerRG" in f
            ]
            if polarized_formats:
                pixel_format.value = polarized_formats[0]
                logger.info(
                    "acquisition.arena_polarization_format",
                    format=polarized_formats[0],
                )
        except Exception as e:
            logger.warning("acquisition.arena_polarization_config_failed", error=str(e))

    def _close_arena(self) -> None:
        if self._arena_device is not None:
            try:
                self._arena_device.stop_stream()
                _arena_system.destroy_device()
            except Exception as e:
                logger.warning("acquisition.arena_close_error", error=str(e))
            self._arena_device = None
            self._arena_nodemap = None
            logger.info("acquisition.closed", source=self._source, backend="arena")

    def _grab_arena(self) -> tuple[np.ndarray | None, FrameMetadata]:
        if self._arena_device is None:
            return None, FrameMetadata(
                timestamp=time.monotonic(),
                frame_number=self._frame_counter,
            )

        try:
            buffer = self._arena_device.get_buffer(timeout=5000)
            ts = time.monotonic()
            self._frame_counter += 1

            # Convert buffer to numpy array
            nparray = np.ctypeslib.as_array(buffer.pdata, shape=(buffer.height, buffer.width))
            # Copy because buffer is recycled
            frame = nparray.copy()

            # Handle multi-channel pixel formats
            bits_per_pixel = buffer.bits_per_pixel
            if bits_per_pixel > 8:
                # 4-channel polarization or 16-bit
                if self._is_polarization and bits_per_pixel >= 32:
                    frame = frame.view(np.uint8).reshape(buffer.height, buffer.width, -1)

            # Extract metadata from buffer/nodemap
            exposure_us = 0.0
            gain_db = 0.0
            temp_c = 0.0
            try:
                nm = self._arena_nodemap
                exposure_us = float(nm["ExposureTime"].value)
                gain_db = float(nm["Gain"].value)
                temp_c = float(nm["DeviceTemperature"].value)
            except Exception:
                pass

            meta = FrameMetadata(
                timestamp=ts,
                frame_number=self._frame_counter,
                exposure_us=exposure_us,
                gain_db=gain_db,
                sensor_temperature_c=temp_c,
            )

            self._arena_device.requeue_buffer(buffer)
            return frame, meta

        except Exception as e:
            logger.warning("acquisition.arena_grab_failed", error=str(e))
            return None, FrameMetadata(
                timestamp=time.monotonic(),
                frame_number=self._frame_counter,
            )

    def _arena_set_nir_strobe(self, enabled: bool) -> None:
        """Configure ExposureActive GPIO output for NIR LED triggering."""
        try:
            nm = self._arena_nodemap
            # Select Line1 as output (ExposureActive)
            nm["LineSelector"].value = "Line1"
            nm["LineMode"].value = "Output"
            nm["LineSource"].value = "ExposureActive" if enabled else "Off"
            logger.info(
                "acquisition.arena_nir_strobe",
                enabled=enabled,
                line="Line1",
                source="ExposureActive" if enabled else "Off",
            )
        except Exception as e:
            logger.warning("acquisition.arena_nir_strobe_failed", error=str(e))

    # ==================================================================
    # Spinnaker (FLIR) backend
    # ==================================================================

    def _open_spinnaker(self) -> bool:
        if self._spin_camera is not None and self._spin_camera.IsStreaming():
            logger.debug("acquisition.spinnaker_already_open", source=self._source)
            return True

        try:
            self._spin_system = PySpin.System.GetInstance()
            cam_list = self._spin_system.GetCameras()
            if cam_list.GetSize() == 0:
                logger.error("acquisition.spinnaker_no_cameras", msg="No FLIR cameras found")
                cam_list.Clear()
                return False

            # Select camera by index or serial number
            if isinstance(self._source, int):
                if self._source >= cam_list.GetSize():
                    logger.error(
                        "acquisition.spinnaker_index_oob",
                        source=self._source,
                        camera_count=cam_list.GetSize(),
                    )
                    cam_list.Clear()
                    return False
                self._spin_camera = cam_list.GetByIndex(self._source)
            else:
                target_serial = str(self._source)
                self._spin_camera = cam_list.GetBySerial(target_serial)

            self._spin_camera.Init()

            # Configure pixel format
            if self._is_polarization:
                self._spinnaker_configure_polarization()

            # Set acquisition mode to continuous
            nodemap = self._spin_camera.GetNodeMap()
            acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
            continuous = acquisition_mode.GetEntryByName("Continuous")
            acquisition_mode.SetIntValue(continuous.GetValue())

            self._spin_camera.BeginAcquisition()
            self._frame_counter = 0

            # Read camera info
            nodemap_tldevice = self._spin_camera.GetTLDeviceNodeMap()
            serial_node = PySpin.CStringPtr(nodemap_tldevice.GetNode("DeviceSerialNumber"))
            model_node = PySpin.CStringPtr(nodemap_tldevice.GetNode("DeviceModelName"))
            serial = serial_node.GetValue() if PySpin.IsReadable(serial_node) else "unknown"
            model = model_node.GetValue() if PySpin.IsReadable(model_node) else "unknown"

            logger.info(
                "acquisition.opened",
                source=self._source,
                backend="spinnaker",
                serial=serial,
                model=model,
            )
            cam_list.Clear()
            return True

        except PySpin.SpinnakerException as e:
            logger.error("acquisition.spinnaker_open_failed", error=str(e))
            self._close_spinnaker()
            return False

    def _spinnaker_configure_polarization(self) -> None:
        """Configure FLIR polarization camera pixel format."""
        try:
            nodemap = self._spin_camera.GetNodeMap()
            pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
            # Try polarized format first
            for fmt_name in ("PolarizedAngles_0d_45d_90d_135d_Mono8", "BayerRG8"):
                entry = pixel_format.GetEntryByName(fmt_name)
                if PySpin.IsReadable(entry):
                    pixel_format.SetIntValue(entry.GetValue())
                    logger.info("acquisition.spinnaker_polarization_format", format=fmt_name)
                    return
        except Exception as e:
            logger.warning("acquisition.spinnaker_polarization_config_failed", error=str(e))

    def _close_spinnaker(self) -> None:
        try:
            if self._spin_camera is not None:
                if self._spin_camera.IsStreaming():
                    self._spin_camera.EndAcquisition()
                self._spin_camera.DeInit()
                del self._spin_camera
                self._spin_camera = None
            if self._spin_system is not None:
                self._spin_system.ReleaseInstance()
                self._spin_system = None
            logger.info("acquisition.closed", source=self._source, backend="spinnaker")
        except Exception as e:
            logger.warning("acquisition.spinnaker_close_error", error=str(e))
            self._spin_camera = None
            self._spin_system = None

    def _grab_spinnaker(self) -> tuple[np.ndarray | None, FrameMetadata]:
        if self._spin_camera is None or not self._spin_camera.IsStreaming():
            return None, FrameMetadata(
                timestamp=time.monotonic(),
                frame_number=self._frame_counter,
            )

        try:
            image_result = self._spin_camera.GetNextImage(5000)
            ts = time.monotonic()
            self._frame_counter += 1

            if image_result.IsIncomplete():
                logger.warning(
                    "acquisition.spinnaker_incomplete",
                    status=image_result.GetImageStatus(),
                )
                image_result.Release()
                return None, FrameMetadata(
                    timestamp=ts,
                    frame_number=self._frame_counter,
                )

            # Convert to numpy
            frame = image_result.GetNDArray().copy()

            # Extract metadata
            exposure_us = 0.0
            gain_db = 0.0
            temp_c = 0.0
            try:
                nodemap = self._spin_camera.GetNodeMap()
                exp_node = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
                gain_node = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
                temp_node = PySpin.CFloatPtr(nodemap.GetNode("DeviceTemperature"))
                if PySpin.IsReadable(exp_node):
                    exposure_us = exp_node.GetValue()
                if PySpin.IsReadable(gain_node):
                    gain_db = gain_node.GetValue()
                if PySpin.IsReadable(temp_node):
                    temp_c = temp_node.GetValue()
            except Exception:
                pass

            meta = FrameMetadata(
                timestamp=ts,
                frame_number=self._frame_counter,
                exposure_us=exposure_us,
                gain_db=gain_db,
                sensor_temperature_c=temp_c,
            )

            image_result.Release()
            return frame, meta

        except PySpin.SpinnakerException as e:
            logger.warning("acquisition.spinnaker_grab_failed", error=str(e))
            return None, FrameMetadata(
                timestamp=time.monotonic(),
                frame_number=self._frame_counter,
            )

    def _spinnaker_set_nir_strobe(self, enabled: bool) -> None:
        """Configure strobe output for NIR LED triggering."""
        try:
            nodemap = self._spin_camera.GetNodeMap()
            # Select Line2 as strobe output
            line_selector = PySpin.CEnumerationPtr(nodemap.GetNode("LineSelector"))
            line2 = line_selector.GetEntryByName("Line2")
            line_selector.SetIntValue(line2.GetValue())

            line_mode = PySpin.CEnumerationPtr(nodemap.GetNode("LineMode"))
            output = line_mode.GetEntryByName("Output")
            line_mode.SetIntValue(output.GetValue())

            line_source = PySpin.CEnumerationPtr(nodemap.GetNode("LineSource"))
            if enabled:
                exposure_active = line_source.GetEntryByName("ExposureActive")
                line_source.SetIntValue(exposure_active.GetValue())
            else:
                off = line_source.GetEntryByName("Off")
                line_source.SetIntValue(off.GetValue())

            logger.info(
                "acquisition.spinnaker_nir_strobe",
                enabled=enabled,
                line="Line2",
            )
        except Exception as e:
            logger.warning("acquisition.spinnaker_nir_strobe_failed", error=str(e))

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> IndustrialCameraCapture:
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
