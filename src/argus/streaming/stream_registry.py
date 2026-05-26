"""Declarative desired-stream registry for go2rtc-backed camera inputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import structlog

from argus.streaming.go2rtc_manager import (
    CameraSourceResolution,
    gige_to_go2rtc_source,
    usb_to_go2rtc_source,
)

logger = structlog.get_logger()


_PLAN_CAMERA_FIELDS = (
    "cameras",
    "camera_configs",
    "camera_inputs",
    "inputs",
    "streams",
    "desired_streams",
)


@dataclass(frozen=True)
class StreamDeclaration:
    """Desired go2rtc stream derived from a camera config or planner output."""

    camera_id: str
    source: str
    protocol: str
    stream_name: str | None = None
    capture_script: str | None = None
    go2rtc_source: str | None = None
    camera: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.stream_name:
            object.__setattr__(self, "stream_name", self.camera_id)

    @property
    def registration_signature(self) -> tuple[str, str, str | None]:
        return (self.source, self.protocol, self.capture_script or self.go2rtc_source)

    @property
    def supported_by_go2rtc(self) -> bool:
        if self.protocol in {"rtsp", "usb"}:
            return True
        return self.protocol == "gige" and bool(self.capture_script or self.go2rtc_source)


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _has_attr(obj: Any, name: str) -> bool:
    if isinstance(obj, Mapping):
        return name in obj
    return hasattr(obj, name)


def _gige_capture_script(camera: Any) -> str | None:
    direct = _attr(camera, "capture_script")
    if direct:
        return str(direct)

    gige = _attr(camera, "gige")
    if gige is None:
        return None
    script = _attr(gige, "capture_script")
    return str(script) if script else None


def _usb_go2rtc_source(camera: Any, source: str, protocol: str) -> str | None:
    if protocol != "usb":
        return None

    go2rtc_stream = _attr(camera, "go2rtc_stream")
    explicit = _attr(go2rtc_stream, "source")
    if explicit:
        return str(explicit)

    usb_cfg = _attr(camera, "usb")
    pixel_format = _attr(usb_cfg, "pixel_format", None)
    device_name = _attr(usb_cfg, "device_name", None)
    device_id = _attr(usb_cfg, "device_id", None)
    resolution = _attr(camera, "resolution", None)
    fps = _attr(camera, "fps_target", None)
    return usb_to_go2rtc_source(
        source,
        device_name=device_name,
        device_id=device_id,
        resolution=resolution,
        fps=fps,
        pixel_format=pixel_format,
    )


def _is_iterable_plan(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(
        value,
        (str, bytes, bytearray, Mapping),
    )


def _plan_items(plan: Any) -> list[Any]:
    if plan is None:
        return []

    if _looks_like_camera(plan):
        return [plan]

    for field_name in _PLAN_CAMERA_FIELDS:
        if not _has_attr(plan, field_name):
            continue
        value = _attr(plan, field_name)
        if value is None:
            continue
        if isinstance(value, Mapping):
            return list(value.values())
        if _is_iterable_plan(value):
            return list(value)
        if _looks_like_camera(value):
            return [value]

    if isinstance(plan, Mapping):
        values = list(plan.values())
        if values and all(_looks_like_camera(value) for value in values):
            return values

    if _is_iterable_plan(plan):
        return list(plan)

    return []


def _looks_like_camera(value: Any) -> bool:
    return any(
        _has_attr(value, name)
        for name in ("camera_id", "source", "original_source", "runtime_source")
    )


def _declaration_from_camera(
    camera: Any,
    *,
    camera_id: str | None = None,
    source: str | None = None,
    protocol: str | None = None,
    stream_name: str | None = None,
    capture_script: str | None = None,
) -> StreamDeclaration | None:
    nested_camera = _attr(camera, "camera")
    if nested_camera is not None and not _looks_like_camera(camera):
        camera = nested_camera

    resolved_camera_id = camera_id or _attr(camera, "camera_id") or _attr(camera, "id")
    resolved_source = (
        source
        or _attr(camera, "source")
        or _attr(camera, "original_source")
        or _attr(camera, "runtime_source")
    )
    if not resolved_camera_id or not resolved_source:
        return None

    resolved_protocol = (
        protocol
        or _attr(camera, "protocol")
        or _attr(camera, "original_protocol")
        or _attr(camera, "runtime_protocol")
        or "rtsp"
    )
    go2rtc_stream = _attr(camera, "go2rtc_stream")
    resolved_stream_name = (
        stream_name
        or _attr(camera, "stream_name")
        or _attr(go2rtc_stream, "name")
        or resolved_camera_id
    )
    resolved_capture_script = capture_script or _gige_capture_script(camera)
    resolved_go2rtc_source = _usb_go2rtc_source(
        camera, str(resolved_source), str(resolved_protocol),
    )
    if resolved_go2rtc_source is None:
        resolved_go2rtc_source = _attr(go2rtc_stream, "source")

    return StreamDeclaration(
        camera_id=str(resolved_camera_id),
        source=str(resolved_source),
        protocol=str(resolved_protocol),
        stream_name=str(resolved_stream_name),
        capture_script=(
            str(resolved_capture_script) if resolved_capture_script else None
        ),
        go2rtc_source=str(resolved_go2rtc_source) if resolved_go2rtc_source else None,
        camera=camera,
    )


def _resolution_for(
    declaration: StreamDeclaration,
    rtsp_url: str | None,
) -> CameraSourceResolution:
    runtime_source = declaration.source
    runtime_protocol = declaration.protocol
    if declaration.protocol == "usb" and rtsp_url:
        runtime_source = rtsp_url
        runtime_protocol = "rtsp"

    return CameraSourceResolution(
        camera_id=declaration.camera_id,
        original_source=declaration.source,
        original_protocol=declaration.protocol,
        runtime_source=runtime_source,
        runtime_protocol=runtime_protocol,
        stream_name=str(declaration.stream_name),
        go2rtc_managed=rtsp_url is not None
        and declaration.protocol in {"rtsp", "usb", "gige"},
    )


def runtime_camera_config(camera: Any, resolution: CameraSourceResolution) -> Any:
    """Return a runtime camera config copy using the resolved capture source."""

    updates = {
        "source": resolution.runtime_source,
        "protocol": resolution.runtime_protocol,
    }
    if hasattr(camera, "model_copy"):
        return camera.model_copy(update=updates)
    if isinstance(camera, Mapping):
        return {**camera, **updates}
    return type(
        "RuntimeCameraConfig",
        (),
        {
            **getattr(camera, "__dict__", {}),
            **updates,
        },
    )()


class StreamRegistry:
    """Tracks desired camera streams and reconciles them into go2rtc.

    The registry is intentionally tolerant about input shape so it can accept
    today's ``CameraConfig`` objects and tomorrow's planner models without
    coupling this module to the planner package.
    """

    def __init__(self, manager: Any) -> None:
        self.manager = manager
        self._desired: dict[str, StreamDeclaration] = {}
        self._registered: dict[str, tuple[str, str, str | None]] = {}
        self._resolutions: dict[str, CameraSourceResolution] = {}

    @property
    def desired_streams(self) -> dict[str, StreamDeclaration]:
        return dict(self._desired)

    @property
    def resolutions(self) -> dict[str, CameraSourceResolution]:
        return dict(self._resolutions)

    def declare_plan(self, plan: Any) -> dict[str, StreamDeclaration]:
        """Replace desired streams with the cameras found in a plan-like object."""

        desired: dict[str, StreamDeclaration] = {}
        for item in _plan_items(plan):
            declaration = _declaration_from_camera(item)
            if declaration is not None:
                desired[str(declaration.stream_name)] = declaration
        self._desired = desired
        return self.desired_streams

    def declare_camera(
        self,
        camera: Any = None,
        *,
        camera_id: str | None = None,
        source: str | None = None,
        protocol: str | None = None,
        stream_name: str | None = None,
        capture_script: str | None = None,
    ) -> StreamDeclaration | None:
        """Declare a single desired stream without registering it immediately."""

        if camera is None:
            camera = {
                "camera_id": camera_id,
                "source": source,
                "protocol": protocol,
                "stream_name": stream_name,
                "capture_script": capture_script,
            }
        declaration = _declaration_from_camera(
            camera,
            camera_id=camera_id,
            source=source,
            protocol=protocol,
            stream_name=stream_name,
            capture_script=capture_script,
        )
        if declaration is None:
            return None
        self._desired[str(declaration.stream_name)] = declaration
        return declaration

    def ensure_registered(
        self,
        camera: Any = None,
        *,
        start_if_needed: bool = False,
        camera_id: str | None = None,
        source: str | None = None,
        protocol: str | None = None,
        stream_name: str | None = None,
        capture_script: str | None = None,
    ) -> CameraSourceResolution | None:
        """Declare and register one camera, retrying later if registration fails."""

        declaration = self.declare_camera(
            camera,
            camera_id=camera_id,
            source=source,
            protocol=protocol,
            stream_name=stream_name,
            capture_script=capture_script,
        )
        if declaration is None:
            return None
        if start_if_needed:
            self._ensure_started()
        return self._ensure_registered(declaration)

    def undeclare_camera(self, camera_or_id: Any, *, remove_runtime: bool = True) -> bool:
        """Remove a desired stream and optionally remove its go2rtc runtime stream."""

        stream_name = self._stream_name_for(camera_or_id)
        if stream_name is None:
            return False

        declaration = self._desired.pop(stream_name, None)
        if declaration is None:
            declaration = self._declaration_by_camera_id(str(stream_name))

        camera_id = declaration.camera_id if declaration else str(stream_name)
        self._resolutions.pop(camera_id, None)
        self._drop_resolution_for_stream(stream_name)

        was_registered = self._registered.pop(stream_name, None) is not None
        if remove_runtime and was_registered:
            self._remove_stream(stream_name)
        return declaration is not None or was_registered

    def reconcile(
        self,
        plan: Any = None,
        *,
        start_if_needed: bool = True,
        remove_stale: bool = True,
    ) -> dict[str, CameraSourceResolution]:
        """Make go2rtc match the currently declared desired stream set."""

        if plan is not None:
            self.declare_plan(plan)

        if start_if_needed:
            self._ensure_started()

        desired_names = set(self._desired)
        if remove_stale:
            for camera_id, resolution in list(self._resolutions.items()):
                if resolution.stream_name not in desired_names:
                    self._resolutions.pop(camera_id, None)
            for stale_name in list(self._registered):
                if stale_name not in desired_names:
                    self._registered.pop(stale_name, None)
                    self._drop_resolution_for_stream(stale_name)
                    self._remove_stream(stale_name)

        for declaration in self._desired.values():
            self._ensure_registered(declaration)

        desired_camera_ids = {
            declaration.camera_id for declaration in self._desired.values()
        }
        return {
            camera_id: resolution
            for camera_id, resolution in self._resolutions.items()
            if camera_id in desired_camera_ids
        }

    def runtime_camera_config(
        self,
        camera: Any,
        resolution: CameraSourceResolution | None = None,
    ) -> Any:
        """Return a runtime config copy for a declared camera."""

        if resolution is None:
            camera_id = str(_attr(camera, "camera_id", camera) or "")
            resolution = self._resolutions.get(camera_id)
        if resolution is not None:
            return runtime_camera_config(camera, resolution)

        declaration = _declaration_from_camera(camera) or StreamDeclaration(
            camera_id=str(_attr(camera, "camera_id", "") or ""),
            source=str(_attr(camera, "source", "") or ""),
            protocol=str(_attr(camera, "protocol", None) or "rtsp"),
        )
        return runtime_camera_config(camera, _resolution_for(declaration, None))

    def _ensure_started(self) -> None:
        if self._is_running():
            return
        initial_streams = self._initial_streams()
        self.manager.start(initial_streams=initial_streams or None)
        self._registered.clear()

    def _ensure_registered(
        self,
        declaration: StreamDeclaration,
    ) -> CameraSourceResolution:
        stream_name = str(declaration.stream_name)
        signature = declaration.registration_signature
        previous_signature = self._registered.get(stream_name)
        if previous_signature == signature:
            resolution = self._resolutions.get(declaration.camera_id)
            if resolution is not None:
                return resolution

        if previous_signature is not None and previous_signature != signature:
            self._registered.pop(stream_name, None)
            self._remove_stream(stream_name)

        rtsp_url: str | None = None
        if declaration.supported_by_go2rtc:
            try:
                register_source = declaration.source
                register_protocol = declaration.protocol
                if declaration.protocol == "usb" and declaration.go2rtc_source:
                    register_source = declaration.go2rtc_source
                    register_protocol = "rtsp"
                rtsp_url = self.manager.register_camera(
                    stream_name,
                    register_source,
                    register_protocol,
                )
            except Exception:
                logger.warning(
                    "go2rtc.register_failed",
                    camera_id=declaration.camera_id,
                    stream_name=stream_name,
                    exc_info=True,
                )

        resolution = _resolution_for(declaration, rtsp_url)
        self._resolutions[declaration.camera_id] = resolution

        if resolution.go2rtc_managed:
            self._registered[stream_name] = signature
            if declaration.protocol == "usb":
                logger.info(
                    "go2rtc.device_runtime_redirect",
                    camera_id=declaration.camera_id,
                    protocol=declaration.protocol,
                    original=declaration.source,
                    runtime=resolution.runtime_source,
                )
            elif declaration.protocol == "gige":
                logger.info(
                    "go2rtc.gige_preview_registered",
                    camera_id=declaration.camera_id,
                    preview_url=rtsp_url,
                )
        return resolution

    def _initial_streams(self) -> dict[str, str]:
        initial_streams: dict[str, str] = {}
        for declaration in self._desired.values():
            if declaration.protocol != "gige":
                continue
            if declaration.go2rtc_source:
                initial_streams[str(declaration.stream_name)] = declaration.go2rtc_source
            elif declaration.capture_script:
                initial_streams[str(declaration.stream_name)] = gige_to_go2rtc_source(
                    declaration.capture_script,
                )
        return initial_streams

    def _stream_name_for(self, camera_or_id: Any) -> str | None:
        if _looks_like_camera(camera_or_id):
            declaration = _declaration_from_camera(camera_or_id)
            if declaration is not None:
                return str(declaration.stream_name)
        value = str(camera_or_id or "")
        if not value:
            return None
        if value in self._desired or value in self._registered:
            return value
        declaration = self._declaration_by_camera_id(value)
        return str(declaration.stream_name) if declaration else value

    def _declaration_by_camera_id(self, camera_id: str) -> StreamDeclaration | None:
        for declaration in self._desired.values():
            if declaration.camera_id == camera_id:
                return declaration
        return None

    def _is_running(self) -> bool:
        return bool(getattr(self.manager, "running", False))

    def _remove_stream(self, stream_name: str) -> None:
        try:
            self.manager.remove_stream(stream_name)
        except Exception:
            logger.warning("go2rtc.remove_stream_failed", name=stream_name, exc_info=True)

    def _drop_resolution_for_stream(self, stream_name: str) -> None:
        for camera_id, resolution in list(self._resolutions.items()):
            if resolution.stream_name == stream_name:
                self._resolutions.pop(camera_id, None)
