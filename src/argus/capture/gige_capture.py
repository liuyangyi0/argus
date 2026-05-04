"""GigE Vision camera capture via Hikrobot MVS SDK.

Provides the same interface as ``CameraCapture`` (connect / read / release /
stop) so the detection pipeline can use it as a drop-in replacement.

The MVS SDK grabs raw frames directly from the camera over GigE Vision,
bypassing any RTSP/GStreamer encoding chain.  This gives the AI pipeline
zero-codec-overhead access to original pixel data.

Browser preview is handled separately by go2rtc's ``exec:`` source which
launches a GStreamer pipeline to encode and push H.264 back to go2rtc.
"""

from __future__ import annotations

import ctypes
import socket
import struct
import threading
import time
from threading import Event

import cv2
import numpy as np
import structlog

from typing import TYPE_CHECKING

from argus.capture.camera import CameraState, FrameData
from argus.core.error_channel import (
    SEVERITY_ERROR,
    get_error_channel,
)

if TYPE_CHECKING:
    from argus.capture.frame_buffer import LatestFrameBuffer

logger = structlog.get_logger()

# GenICam auto-mode enum values
_AUTO_OFF = 0
_AUTO_CONTINUOUS = 2

# GigE Vision transport layer type
_TLTYPE_GIGE = 0x00000001

# ---------------------------------------------------------------------------
# MVS SDK lazy import — the SDK is only available on machines with the
# Hikrobot MVS runtime installed.  We import at module level so that
# *importing* this file on a dev machine without the SDK doesn't crash;
# the error surfaces at ``connect()`` time instead.
# ---------------------------------------------------------------------------
try:
    from ctypes import byref, cast, POINTER

    from MvCameraControl_class import (  # type: ignore[import-untyped]
        MvCamera,
        MV_CC_DEVICE_INFO,
        MV_CC_DEVICE_INFO_LIST,
        MV_FRAME_OUT,
        MV_GIGE_DEVICE_INFO,
        MV_ACCESS_Exclusive,
        PixelType_Gvsp_Mono8,
        PixelType_Gvsp_BayerBG8,
        PixelType_Gvsp_BayerGB8,
        PixelType_Gvsp_BayerGR8,
        PixelType_Gvsp_BayerRG8,
    )

    _MVS_AVAILABLE = True
except ImportError:
    _MVS_AVAILABLE = False


# Map common pixel format names to SDK enum values
_PIXEL_FORMAT_MAP: dict[str, int] = {}
if _MVS_AVAILABLE:
    _PIXEL_FORMAT_MAP = {
        "Mono8": PixelType_Gvsp_Mono8,
        "BayerBG8": PixelType_Gvsp_BayerBG8,
        "BayerGB8": PixelType_Gvsp_BayerGB8,
        "BayerGR8": PixelType_Gvsp_BayerGR8,
        "BayerRG8": PixelType_Gvsp_BayerRG8,
    }


# Map pixel format names to OpenCV Bayer demosaicing codes
_BAYER_CV_MAP = {
    "BayerBG8": cv2.COLOR_BAYER_BG2BGR,
    "BayerGB8": cv2.COLOR_BAYER_GB2BGR,
    "BayerGR8": cv2.COLOR_BAYER_GR2BGR,
    "BayerRG8": cv2.COLOR_BAYER_RG2BGR,
}


def _ip_to_int(ip: str) -> int:
    """Convert dotted-quad IP string to a 32-bit integer."""
    return struct.unpack("!I", socket.inet_aton(ip))[0]


class GigECapture:
    """Captures frames from a Hikrobot GigE Vision camera via MVS SDK.

    Parameters
    ----------
    camera_id:
        Unique identifier for logging and pipeline plumbing.
    source:
        Camera IP address (e.g. ``"192.168.66.223"``).
    fps_target:
        Target frame rate.  The camera's ``AcquisitionFrameRate`` is set
        to this value *and* the capture loop throttles to match.
    resolution:
        Desired ``(width, height)``.  Applied as camera ROI if the sensor
        is larger.
    exposure:
        Exposure time in microseconds (0 = auto).
    gain:
        Gain in dB (0 = auto).
    pixel_format:
        SDK pixel format name (default ``"Mono8"``).
    reconnect_delay:
        Seconds between reconnection attempts.
    max_reconnect_attempts:
        Maximum retries (-1 = infinite).
    """

    def __init__(
        self,
        camera_id: str,
        source: str,
        fps_target: int = 30,
        resolution: tuple[int, int] = (1280, 720),
        exposure: float = 0,
        gain: float = 0,
        pixel_format: str = "Mono8",
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = -1,
    ):
        self.camera_id = camera_id
        self.source = source
        self.fps_target = fps_target
        self.resolution = resolution
        self.exposure = exposure
        self.gain = gain
        self.pixel_format = pixel_format
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        self._cam: MvCamera | None = None  # type: ignore[name-defined]
        self._state = CameraState()
        self._stop_event = Event()
        self._frame_interval = 1.0 / fps_target if fps_target > 0 else 0
        self._is_mono8 = pixel_format == "Mono8"
        self._bayer_cv_code: int | None = _BAYER_CV_MAP.get(pixel_format)

        # Pre-allocate SDK structs (reused every read() call)
        self._frame_out: MV_FRAME_OUT | None = None  # type: ignore[name-defined]
        if _MVS_AVAILABLE:
            self._frame_out = MV_FRAME_OUT()

        # Pre-allocate raw pixel buffer (sized for max expected frame)
        w, h = resolution
        self._raw_buf = np.empty(w * h * 4, dtype=np.uint8)  # 4 bytes/px worst case

        # Non-blocking reconnection state
        self._reconnecting = False
        self._reconnect_lock = threading.Lock()

        # Async capture thread buffer (initialised by start_capture_thread)
        self._frame_buffer: LatestFrameBuffer | None = None

        # Protocol identifier for pipeline dispatch
        self.protocol: str = "gige"

    @property
    def state(self) -> CameraState:
        return self._state

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Open the GigE camera and start grabbing."""
        if not _MVS_AVAILABLE:
            self._state.error = "MVS SDK not installed (MvCameraControl_class not found)"
            logger.error("gige.sdk_missing", camera_id=self.camera_id)
            return False

        try:
            return self._connect_impl()
        except Exception as e:
            self._state.connected = False
            self._state.error = str(e)
            logger.error("gige.connect_error", camera_id=self.camera_id, error=str(e))
            return False

    def _connect_impl(self) -> bool:
        cam = MvCamera()

        # Enumerate GigE devices
        device_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(_TLTYPE_GIGE, device_list)
        if ret != 0:
            self._state.error = f"EnumDevices failed: 0x{ret:08x}"
            logger.error("gige.enum_failed", camera_id=self.camera_id, ret=hex(ret))
            return False

        # Find the camera by IP address
        target_ip = _ip_to_int(self.source)
        device_info = None
        for i in range(device_list.nDeviceNum):
            dev = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if dev.nTLayerType == _TLTYPE_GIGE:
                gige_info = cast(
                    byref(dev.SpecialInfo),
                    POINTER(MV_GIGE_DEVICE_INFO),
                ).contents
                if gige_info.nCurrentIp == target_ip:
                    device_info = dev
                    break

        if device_info is None:
            self._state.error = f"Camera not found at {self.source}"
            logger.error(
                "gige.not_found",
                camera_id=self.camera_id,
                ip=self.source,
                found=device_list.nDeviceNum,
            )
            return False

        # Create handle and open
        ret = cam.MV_CC_CreateHandle(device_info)
        if ret != 0:
            cam.MV_CC_DestroyHandle()
            self._state.error = f"CreateHandle failed: 0x{ret:08x}"
            return False

        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            cam.MV_CC_DestroyHandle()
            self._state.error = f"OpenDevice failed: 0x{ret:08x}"
            return False

        # Optimal packet size for GigE
        packet_size = cam.MV_CC_GetOptimalPacketSize()
        if packet_size > 0:
            cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)

        # Configure camera parameters
        self._configure_camera(cam)

        # Start grabbing
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            self._state.error = f"StartGrabbing failed: 0x{ret:08x}"
            return False

        self._cam = cam
        self._state.connected = True
        self._state.error = None
        logger.info(
            "gige.connected",
            camera_id=self.camera_id,
            ip=self.source,
            resolution=self.resolution,
            fps=self.fps_target,
            pixel_format=self.pixel_format,
        )
        return True

    def _configure_camera(self, cam: MvCamera) -> None:  # type: ignore[name-defined]
        """Apply camera parameters (best-effort, log failures)."""
        # Trigger mode off (continuous acquisition)
        cam.MV_CC_SetEnumValue("TriggerMode", 0)

        # Pixel format
        fmt = _PIXEL_FORMAT_MAP.get(self.pixel_format)
        if fmt is not None:
            ret = cam.MV_CC_SetEnumValue("PixelFormat", fmt)
            if ret != 0:
                logger.warning("gige.set_pixel_format_failed", ret=hex(ret))
        else:
            logger.warning(
                "gige.unknown_pixel_format",
                pixel_format=self.pixel_format,
                supported=list(_PIXEL_FORMAT_MAP.keys()),
            )

        # Resolution / ROI
        w, h = self.resolution
        cam.MV_CC_SetIntValue("Width", w)
        cam.MV_CC_SetIntValue("Height", h)

        # Exposure
        if self.exposure > 0:
            cam.MV_CC_SetEnumValue("ExposureAuto", _AUTO_OFF)
            cam.MV_CC_SetFloatValue("ExposureTime", float(self.exposure))
        else:
            cam.MV_CC_SetEnumValue("ExposureAuto", _AUTO_CONTINUOUS)

        # Gain
        if self.gain > 0:
            cam.MV_CC_SetEnumValue("GainAuto", _AUTO_OFF)
            cam.MV_CC_SetFloatValue("Gain", float(self.gain))
        else:
            cam.MV_CC_SetEnumValue("GainAuto", _AUTO_CONTINUOUS)

        # Frame rate
        cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(self.fps_target))

    # ------------------------------------------------------------------
    # Frame reading
    # ------------------------------------------------------------------

    def read(self) -> FrameData | None:
        """Read a single frame from the camera.

        Returns ``None`` if the camera is disconnected or the FPS
        throttle says to skip this call.
        """
        if self._cam is None or not self._state.connected:
            return None

        # FPS throttle — safety belt in case camera firmware ignores
        # AcquisitionFrameRate (the hardware rate-limit set in connect).
        if self._frame_interval > 0:
            now = time.monotonic()
            elapsed = now - self._state.last_frame_time
            remaining = self._frame_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

        frame_out = self._frame_out

        ret = self._cam.MV_CC_GetImageBuffer(frame_out, 1000)  # 1s timeout
        if ret != 0:
            self._state.connected = False
            self._state.error = f"GetImageBuffer failed: 0x{ret:08x}"
            self.request_reconnect()
            return None

        try:
            info = frame_out.stFrameInfo
            w = info.nWidth
            h = info.nHeight

            # Copy raw pixels into pre-allocated buffer (must copy before FreeImageBuffer)
            buf_size = info.nFrameLen
            data = self._raw_buf[:buf_size]
            ctypes.memmove(data.ctypes.data, frame_out.pBufAddr, buf_size)

            # Convert to BGR for consistency with OpenCV pipeline
            if self._is_mono8:
                frame = cv2.cvtColor(data.reshape((h, w)), cv2.COLOR_GRAY2BGR)
            elif self._bayer_cv_code is not None:
                frame = cv2.cvtColor(data.reshape((h, w)), self._bayer_cv_code)
            else:
                frame = cv2.cvtColor(data.reshape((h, w)), cv2.COLOR_GRAY2BGR)
        finally:
            self._cam.MV_CC_FreeImageBuffer(frame_out)

        now = time.monotonic()
        self._state.last_frame_time = now
        self._state.total_frames += 1

        return FrameData(
            frame=frame,
            camera_id=self.camera_id,
            timestamp=now,
            frame_number=self._state.total_frames,
            resolution=(w, h),
        )

    # ------------------------------------------------------------------
    # Reconnection (mirrors CameraCapture interface)
    # ------------------------------------------------------------------

    def request_reconnect(self) -> None:
        """Request a non-blocking reconnection attempt."""
        with self._reconnect_lock:
            if self._reconnecting:
                return
            self._reconnecting = True

        thread = threading.Thread(
            target=self._reconnect_background,
            name=f"reconnect-gige-{self.camera_id}",
            daemon=True,
        )
        thread.start()

    def _reconnect_background(self) -> None:
        """Background reconnection with exponential backoff."""
        try:
            self.release()
            delay = self.reconnect_delay
            attempt = 0

            while not self._stop_event.is_set():
                if (
                    self.max_reconnect_attempts >= 0
                    and attempt >= self.max_reconnect_attempts
                ):
                    logger.error(
                        "gige.reconnect_exhausted",
                        camera_id=self.camera_id,
                        attempts=attempt,
                    )
                    get_error_channel().emit(
                        severity=SEVERITY_ERROR,
                        source="gige_capture",
                        code="reconnect_exhausted",
                        message=(
                            f"GigE 摄像头 {self.camera_id} 重连次数耗尽 "
                            f"(attempts={attempt})"
                        ),
                        context={
                            "camera_id": self.camera_id,
                            "attempts": attempt,
                            "max_attempts": self.max_reconnect_attempts,
                        },
                    )
                    break

                attempt += 1
                self._state.reconnect_count += 1
                logger.info(
                    "gige.reconnecting",
                    camera_id=self.camera_id,
                    attempt=attempt,
                    delay=delay,
                )

                self._stop_event.wait(delay)
                if self._stop_event.is_set():
                    break

                if self.connect():
                    logger.info(
                        "gige.reconnected",
                        camera_id=self.camera_id,
                        attempts=attempt,
                    )
                    break

                delay = min(delay * 2, 60.0)
        finally:
            with self._reconnect_lock:
                self._reconnecting = False

    def reconnect(self) -> bool:
        """Attempt to reconnect with exponential backoff (blocking)."""
        self.release()

        attempt = 0
        delay = self.reconnect_delay

        while not self._stop_event.is_set():
            if (
                self.max_reconnect_attempts >= 0
                and attempt >= self.max_reconnect_attempts
            ):
                logger.error(
                    "gige.reconnect_exhausted",
                    camera_id=self.camera_id,
                    attempts=attempt,
                )
                get_error_channel().emit(
                    severity=SEVERITY_ERROR,
                    source="gige_capture",
                    code="reconnect_exhausted",
                    message=(
                        f"GigE 摄像头 {self.camera_id} 重连次数耗尽 "
                        f"(attempts={attempt})"
                    ),
                    context={
                        "camera_id": self.camera_id,
                        "attempts": attempt,
                        "max_attempts": self.max_reconnect_attempts,
                    },
                )
                return False

            attempt += 1
            self._state.reconnect_count += 1
            logger.info(
                "gige.reconnecting",
                camera_id=self.camera_id,
                attempt=attempt,
                delay=delay,
            )

            self._stop_event.wait(delay)
            if self._stop_event.is_set():
                return False

            if self.connect():
                return True

            delay = min(delay * 2, 60.0)

        return False

    # ------------------------------------------------------------------
    # Async capture thread (same interface as CameraCapture)
    # ------------------------------------------------------------------

    def start_capture_thread(self) -> None:
        """Start a dedicated capture thread for latest-frame buffering."""
        from argus.capture.frame_buffer import LatestFrameBuffer

        if self._frame_buffer is not None:
            return
        self._frame_buffer = LatestFrameBuffer()
        t = threading.Thread(
            target=self._capture_loop,
            name=f"capture-{self.camera_id}",
            daemon=True,
        )
        t.start()
        logger.info("gige.capture_thread_started", camera_id=self.camera_id)

    def _capture_loop(self) -> None:
        """Background loop: read frames and store the latest one."""
        while not self._stop_event.is_set():
            frame_data = self.read()
            if frame_data is not None:
                self._frame_buffer.put(frame_data)
            else:
                self._stop_event.wait(0.01)

    def read_latest(self) -> FrameData | None:
        """Return the most recently captured frame.

        Falls back to ``read()`` if no capture thread is running.
        """
        if self._frame_buffer is None:
            return self.read()
        return self._frame_buffer.get(timeout=5.0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Release the camera resource."""
        if self._cam is not None:
            try:
                self._cam.MV_CC_StopGrabbing()
            except Exception:
                pass
            try:
                self._cam.MV_CC_CloseDevice()
            except Exception:
                pass
            try:
                self._cam.MV_CC_DestroyHandle()
            except Exception:
                pass
            self._cam = None
        self._state.connected = False

    def stop(self) -> None:
        """Signal the camera to stop."""
        self._stop_event.set()
        self.release()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.stop()
