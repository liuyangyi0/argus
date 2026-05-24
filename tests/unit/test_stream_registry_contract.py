"""Contract tests for the declarative go2rtc stream registry."""

from __future__ import annotations

from types import SimpleNamespace

from argus.config.schema import CameraConfig
from argus.streaming.stream_registry import StreamRegistry


_DEFAULT_REGISTER_RETURN = object()


class FakeGo2RTCManager:
    def __init__(
        self,
        *,
        running: bool = True,
        register_return: object = _DEFAULT_REGISTER_RETURN,
    ) -> None:
        self.running = running
        self.register_return = register_return
        self.raise_register = False
        self.start_calls: list[dict[str, str] | None] = []
        self.register_calls: list[tuple[str, str, str]] = []
        self.remove_calls: list[str] = []

    def start(self, *, initial_streams: dict[str, str] | None = None) -> None:
        self.start_calls.append(initial_streams)
        self.running = True

    def register_camera(self, camera_id: str, source: str, protocol: str) -> str | None:
        self.register_calls.append((camera_id, source, protocol))
        if self.raise_register:
            raise RuntimeError("register failed")
        if self.register_return is _DEFAULT_REGISTER_RETURN:
            return f"rtsp://127.0.0.1:8554/{camera_id}"
        return self.register_return  # type: ignore[return-value]

    def remove_stream(self, name: str) -> None:
        self.remove_calls.append(name)


def _camera(camera_id: str, source: str, protocol: str | None = "rtsp") -> SimpleNamespace:
    return SimpleNamespace(camera_id=camera_id, source=source, protocol=protocol)


def test_ensure_registered_is_idempotent_for_same_declaration():
    manager = FakeGo2RTCManager(running=True)
    registry = StreamRegistry(manager)
    camera = _camera("cam_01", "rtsp://192.168.1.10/s1")

    first = registry.ensure_registered(camera)
    second = registry.ensure_registered(camera)

    assert first is second
    assert manager.register_calls == [("cam_01", "rtsp://192.168.1.10/s1", "rtsp")]


def test_undeclare_camera_removes_registered_stream():
    manager = FakeGo2RTCManager(running=True)
    registry = StreamRegistry(manager)
    registry.ensure_registered(_camera("cam_01", "rtsp://192.168.1.10/s1"))

    removed = registry.undeclare_camera("cam_01")

    assert removed is True
    assert manager.remove_calls == ["cam_01"]
    assert registry.reconcile(start_if_needed=False) == {}


def test_reconcile_registers_plan_and_removes_stale_streams():
    manager = FakeGo2RTCManager(running=False)
    registry = StreamRegistry(manager)
    cam_rtsp = _camera("cam_rtsp", "rtsp://192.168.1.10/s1")
    cam_usb = _camera("cam_usb", "0", "usb")

    resolutions = registry.reconcile(SimpleNamespace(cameras=[cam_rtsp, cam_usb]))

    assert manager.start_calls == [None]
    assert manager.register_calls == [
        ("cam_rtsp", "rtsp://192.168.1.10/s1", "rtsp"),
        ("cam_usb", "0", "usb"),
    ]
    assert resolutions["cam_usb"].runtime_source == "rtsp://127.0.0.1:8554/cam_usb"
    assert resolutions["cam_usb"].runtime_protocol == "rtsp"

    manager.register_calls.clear()
    registry.reconcile(SimpleNamespace(cameras=[cam_usb]))

    assert manager.remove_calls == ["cam_rtsp"]
    assert manager.register_calls == []
    assert list(registry.resolutions) == ["cam_usb"]


def test_failed_registration_does_not_mark_stream_registered():
    manager = FakeGo2RTCManager(running=True)
    manager.raise_register = True
    registry = StreamRegistry(manager)
    camera = _camera("cam_01", "rtsp://192.168.1.10/s1")

    failed = registry.ensure_registered(camera)

    assert failed is not None
    assert failed.go2rtc_managed is False
    assert failed.runtime_source == "rtsp://192.168.1.10/s1"
    assert manager.register_calls == [("cam_01", "rtsp://192.168.1.10/s1", "rtsp")]

    manager.raise_register = False
    recovered = registry.ensure_registered(camera)

    assert recovered is not None
    assert recovered.go2rtc_managed is True
    assert manager.register_calls == [
        ("cam_01", "rtsp://192.168.1.10/s1", "rtsp"),
        ("cam_01", "rtsp://192.168.1.10/s1", "rtsp"),
    ]


def test_runtime_camera_config_preserves_original_camera_config():
    manager = FakeGo2RTCManager(running=True)
    registry = StreamRegistry(manager)
    camera = CameraConfig(camera_id="cam_usb", name="USB", source="0", protocol="usb")

    registry.ensure_registered(camera)
    runtime_camera = registry.runtime_camera_config(camera)

    assert camera.source == "0"
    assert camera.protocol == "usb"
    assert runtime_camera is not camera
    assert runtime_camera.source == "rtsp://127.0.0.1:8554/cam_usb"
    assert runtime_camera.protocol == "rtsp"


def test_reconcile_re_registers_after_go2rtc_restart():
    manager = FakeGo2RTCManager(running=False)
    registry = StreamRegistry(manager)
    registry.declare_camera(_camera("cam_01", "rtsp://192.168.1.10/s1"))

    registry.reconcile()
    registry.reconcile()

    assert manager.start_calls == [None]
    assert manager.register_calls == [("cam_01", "rtsp://192.168.1.10/s1", "rtsp")]

    manager.running = False
    registry.reconcile()

    assert manager.start_calls == [None, None]
    assert manager.register_calls == [
        ("cam_01", "rtsp://192.168.1.10/s1", "rtsp"),
        ("cam_01", "rtsp://192.168.1.10/s1", "rtsp"),
    ]
