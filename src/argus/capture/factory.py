"""Factory for constructing capture adapters from camera runtime inputs."""

from __future__ import annotations

from argus.camera.models import CameraRuntimePlan, DetectionInput
from argus.capture.base import CaptureAdapter
from argus.capture.camera import CameraCapture
from argus.config.schema import CameraConfig


class CaptureFactory:
    """Create capture adapters while preserving legacy capture behavior."""

    @staticmethod
    def create(
        camera_config_or_detection_input: CameraConfig | CameraRuntimePlan | DetectionInput,
        camera_config: CameraConfig | None = None,
    ) -> CaptureAdapter:
        """Create a ``CameraCapture`` or ``GigECapture``.

        Pass a persisted ``CameraConfig`` to reproduce the existing protocol
        branch, or pass a runtime ``DetectionInput``/``CameraRuntimePlan`` with
        the original ``CameraConfig`` for camera metadata and tuning values.
        """
        if isinstance(camera_config_or_detection_input, CameraConfig):
            if camera_config is not None:
                raise ValueError("camera_config must not be provided when creating from CameraConfig")
            return CaptureFactory._from_config(camera_config_or_detection_input)

        if camera_config is None:
            raise ValueError("camera_config is required when creating from runtime detection input")

        detection = (
            camera_config_or_detection_input.detection
            if isinstance(camera_config_or_detection_input, CameraRuntimePlan)
            else camera_config_or_detection_input
        )
        return CaptureFactory._from_detection_input(detection, camera_config)

    @staticmethod
    def _from_config(camera_config: CameraConfig) -> CaptureAdapter:
        if camera_config.protocol == "gige":
            return CaptureFactory._create_gige(
                camera_config=camera_config,
                source=camera_config.source,
            )
        return CaptureFactory._create_opencv(
            camera_config=camera_config,
            source=camera_config.source,
            protocol=camera_config.protocol,
        )

    @staticmethod
    def _from_detection_input(
        detection: DetectionInput,
        camera_config: CameraConfig,
    ) -> CaptureAdapter:
        if detection.backend == "gige_sdk" or detection.protocol == "gige":
            return CaptureFactory._create_gige(
                camera_config=camera_config,
                source=detection.source,
            )
        if detection.backend == "opencv":
            return CaptureFactory._create_opencv(
                camera_config=camera_config,
                source=detection.source,
                protocol=detection.protocol,
            )
        raise ValueError(f"Unsupported detection backend: {detection.backend!r}")

    @staticmethod
    def _create_opencv(
        *,
        camera_config: CameraConfig,
        source: str,
        protocol: str,
    ) -> CaptureAdapter:
        return CameraCapture(
            camera_id=camera_config.camera_id,
            source=source,
            protocol=protocol,
            fps_target=camera_config.fps_target,
            resolution=camera_config.resolution,
            reconnect_delay=camera_config.reconnect_delay,
            max_reconnect_attempts=camera_config.max_reconnect_attempts,
        )

    @staticmethod
    def _create_gige(
        *,
        camera_config: CameraConfig,
        source: str,
    ) -> CaptureAdapter:
        from argus.capture.gige_capture import GigECapture

        gige_cfg = camera_config.gige
        return GigECapture(
            camera_id=camera_config.camera_id,
            source=source,
            fps_target=camera_config.fps_target,
            resolution=camera_config.resolution,
            exposure=gige_cfg.exposure,
            gain=gige_cfg.gain,
            pixel_format=gige_cfg.pixel_format,
            reconnect_delay=camera_config.reconnect_delay,
            max_reconnect_attempts=camera_config.max_reconnect_attempts,
        )


__all__ = ["CaptureFactory"]
