"""Camera runtime input planning."""

from __future__ import annotations

from typing import Final

from argus.camera.models import (
    CameraRuntimePlan,
    DetectionInput,
    Go2RTCStreamSpec,
    PreviewInput,
    SnapshotInput,
)
from argus.config.schema import CameraConfig
from argus.streaming.go2rtc_manager import gige_to_go2rtc_source, usb_to_go2rtc_source


class CameraRuntimePlanner:
    """Build immutable runtime input plans from persisted camera config."""

    DEFAULT_GO2RTC_RTSP_HOST: Final[str] = "127.0.0.1"
    DEFAULT_GO2RTC_RTSP_PORT: Final[int] = 8554

    @classmethod
    def build(
        cls,
        config: CameraConfig,
        *,
        go2rtc_rtsp_host: str = DEFAULT_GO2RTC_RTSP_HOST,
        go2rtc_rtsp_port: int = DEFAULT_GO2RTC_RTSP_PORT,
    ) -> CameraRuntimePlan:
        """Return a runtime plan without mutating ``config``."""
        camera_id = config.camera_id
        source = str(config.source)
        protocol = str(config.protocol or "rtsp")

        stream = cls._go2rtc_stream(
            config=config,
            camera_id=camera_id,
            source=source,
            protocol=protocol,
            capture_script=config.gige.capture_script,
            rtsp_host=go2rtc_rtsp_host,
            rtsp_port=go2rtc_rtsp_port,
        )

        detection = cls._detection_input(
            source=source,
            protocol=protocol,
            stream=stream,
        )
        preview = cls._preview_input(camera_id=camera_id, stream=stream)
        snapshot = SnapshotInput(
            mode="latest_frame_jpeg",
            path=f"/api/cameras/{camera_id}/snapshot",
        )

        return CameraRuntimePlan(
            camera_id=camera_id,
            original_source=source,
            original_protocol=protocol,
            detection=detection,
            preview=preview,
            snapshot=snapshot,
            go2rtc_stream=stream,
        )

    @classmethod
    def _go2rtc_stream(
        cls,
        *,
        config: CameraConfig,
        camera_id: str,
        source: str,
        protocol: str,
        capture_script: str | None,
        rtsp_host: str,
        rtsp_port: int,
    ) -> Go2RTCStreamSpec | None:
        runtime_rtsp_url = cls._go2rtc_rtsp_url(camera_id, rtsp_host, rtsp_port)

        if protocol == "rtsp":
            return Go2RTCStreamSpec(
                name=camera_id,
                source=source,
                source_protocol=protocol,
                runtime_rtsp_url=runtime_rtsp_url,
                registration="rest_api",
            )
        if protocol == "usb":
            usb_cfg = config.usb
            return Go2RTCStreamSpec(
                name=camera_id,
                source=usb_to_go2rtc_source(
                    source,
                    device_name=usb_cfg.device_name,
                    device_id=usb_cfg.device_id,
                    resolution=config.resolution,
                    fps=config.fps_target,
                    pixel_format=usb_cfg.pixel_format,
                ),
                source_protocol=protocol,
                runtime_rtsp_url=runtime_rtsp_url,
                registration="rest_api",
            )
        if protocol == "gige" and capture_script:
            return Go2RTCStreamSpec(
                name=camera_id,
                source=gige_to_go2rtc_source(capture_script),
                source_protocol=protocol,
                runtime_rtsp_url=runtime_rtsp_url,
                registration="initial_config",
            )
        return None

    @staticmethod
    def _detection_input(
        *,
        source: str,
        protocol: str,
        stream: Go2RTCStreamSpec | None,
    ) -> DetectionInput:
        if protocol == "usb" and stream is not None:
            return DetectionInput(
                source=stream.runtime_rtsp_url,
                protocol="rtsp",
                backend="opencv",
                via_go2rtc=True,
                stream_name=stream.name,
            )
        if protocol == "gige":
            return DetectionInput(
                source=source,
                protocol=protocol,
                backend="gige_sdk",
            )
        return DetectionInput(
            source=source,
            protocol=protocol,
            backend="opencv",
        )

    @staticmethod
    def _preview_input(
        *,
        camera_id: str,
        stream: Go2RTCStreamSpec | None,
    ) -> PreviewInput:
        fallback = f"/api/cameras/{camera_id}/stream"
        if stream is not None:
            return PreviewInput(
                mode="go2rtc",
                stream_name=stream.name,
                fallback_path=fallback,
            )
        return PreviewInput(
            mode="latest_frame_mjpeg",
            fallback_path=fallback,
        )

    @staticmethod
    def _go2rtc_rtsp_url(camera_id: str, host: str, port: int) -> str:
        return f"rtsp://{host}:{port}/{camera_id}"
