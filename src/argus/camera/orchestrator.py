"""Thin lifecycle facade for camera runtime stream coordination."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog

logger = structlog.get_logger()


class CameraOrchestrator:
    """Coordinate CameraManager lifecycle calls with desired stream state."""

    def __init__(self, camera_manager: Any, stream_registry: Any | None = None) -> None:
        self.camera_manager = camera_manager
        self.stream_registry = stream_registry

    def start(self, camera_id: str) -> bool:
        """Register the runtime stream before starting the camera pipeline."""
        cam_config = self._camera_config(camera_id)
        if cam_config is not None:
            self._ensure_registered(cam_config)
        return bool(self.camera_manager.start_camera(camera_id))

    def stop(self, camera_id: str) -> None:
        """Stop one camera pipeline."""
        self.camera_manager.stop_camera(camera_id)

    def restart(self, camera_id: str) -> bool:
        """Stop and then start one camera pipeline."""
        self.stop(camera_id)
        return self.start(camera_id)

    def delete(self, camera_id: str) -> bool:
        """Stop a camera, remove its manager config, and undeclare its stream."""
        cam_config = self._camera_config(camera_id)
        self.stop(camera_id)

        remove_config = getattr(self.camera_manager, "remove_camera_config", None)
        if callable(remove_config):
            remove_config(camera_id)

        if self.stream_registry is None:
            return False
        return bool(self.stream_registry.undeclare_camera(cam_config or camera_id))

    def reconcile_streams(self) -> dict[str, Any]:
        """Reconcile currently managed camera configs into go2rtc."""
        if self.stream_registry is None:
            return {}
        return dict(self.stream_registry.reconcile(self._camera_configs()))

    def runtime_camera_config(self, camera_config: Any) -> Any:
        """Return the camera config copy the capture pipeline should open."""
        if self.stream_registry is None:
            return camera_config

        resolution = self._ensure_registered(camera_config)
        if resolution is not None:
            return self.stream_registry.runtime_camera_config(
                camera_config,
                resolution,
            )
        return self.stream_registry.runtime_camera_config(camera_config)

    def _ensure_registered(self, camera_config: Any) -> Any | None:
        if self.stream_registry is None:
            return None
        try:
            return self.stream_registry.ensure_registered(
                camera_config,
                start_if_needed=True,
            )
        except Exception as exc:
            logger.warning(
                "camera_orchestrator.stream_register_failed",
                camera_id=getattr(camera_config, "camera_id", None),
                error=str(exc),
            )
            return None

    def _camera_config(self, camera_id: str) -> Any | None:
        get_config: Callable[[str], Any] | None = getattr(
            self.camera_manager,
            "get_camera_config",
            None,
        )
        if callable(get_config):
            config = get_config(camera_id)
            if config is not None:
                return config

        return next(
            (
                camera
                for camera in getattr(self.camera_manager, "_cameras", [])
                if getattr(camera, "camera_id", None) == camera_id
            ),
            None,
        )

    def _camera_configs(self) -> list[Any]:
        return list(getattr(self.camera_manager, "_cameras", []) or [])
