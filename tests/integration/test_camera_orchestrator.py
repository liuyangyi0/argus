"""Integration tests for the camera lifecycle orchestrator facade."""

from __future__ import annotations

from types import SimpleNamespace

from argus.camera.orchestrator import CameraOrchestrator
from argus.streaming.stream_registry import StreamRegistry


class FakeGo2RTCManager:
    def __init__(self, events: list[tuple], *, running: bool = True) -> None:
        self.events = events
        self.running = running
        self.start_calls: list[dict[str, str] | None] = []
        self.register_calls: list[tuple[str, str, str]] = []
        self.remove_calls: list[str] = []

    def start(self, *, initial_streams: dict[str, str] | None = None) -> None:
        self.events.append(("go2rtc_start", initial_streams))
        self.start_calls.append(initial_streams)
        self.running = True

    def register_camera(self, stream_name: str, source: str, protocol: str) -> str:
        self.events.append(("register", stream_name, source, protocol))
        self.register_calls.append((stream_name, source, protocol))
        return f"rtsp://127.0.0.1:8554/{stream_name}"

    def remove_stream(self, stream_name: str) -> None:
        self.events.append(("remove_stream", stream_name))
        self.remove_calls.append(stream_name)


class FakeCameraManager:
    def __init__(self, cameras: list[SimpleNamespace], events: list[tuple]) -> None:
        self._cameras = cameras
        self.events = events
        self.stop_calls: list[str] = []
        self.start_calls: list[str] = []
        self.remove_calls: list[str] = []

    def get_camera_config(self, camera_id: str) -> SimpleNamespace | None:
        return next(
            (camera for camera in self._cameras if camera.camera_id == camera_id),
            None,
        )

    def start_camera(self, camera_id: str) -> bool:
        self.events.append(("manager_start", camera_id))
        self.start_calls.append(camera_id)
        return True

    def stop_camera(self, camera_id: str) -> None:
        self.events.append(("manager_stop", camera_id))
        self.stop_calls.append(camera_id)

    def remove_camera_config(self, camera_id: str) -> None:
        self.events.append(("manager_remove_config", camera_id))
        self.remove_calls.append(camera_id)
        self._cameras = [
            camera for camera in self._cameras if camera.camera_id != camera_id
        ]


def _camera(camera_id: str = "cam_01") -> SimpleNamespace:
    return SimpleNamespace(
        camera_id=camera_id,
        source=f"rtsp://example/{camera_id}",
        protocol="rtsp",
    )


def test_start_ensures_stream_registration_before_manager_start():
    events: list[tuple] = []
    camera = _camera()
    go2rtc = FakeGo2RTCManager(events, running=True)
    registry = StreamRegistry(go2rtc)
    manager = FakeCameraManager([camera], events)
    orchestrator = CameraOrchestrator(manager, registry)

    assert orchestrator.start("cam_01") is True

    assert events == [
        ("register", "cam_01", "rtsp://example/cam_01", "rtsp"),
        ("manager_start", "cam_01"),
    ]


def test_delete_removes_stream_declaration_and_runtime_stream():
    events: list[tuple] = []
    camera = _camera()
    go2rtc = FakeGo2RTCManager(events, running=True)
    registry = StreamRegistry(go2rtc)
    manager = FakeCameraManager([camera], events)
    orchestrator = CameraOrchestrator(manager, registry)
    registry.ensure_registered(camera)

    assert orchestrator.delete("cam_01") is True

    assert manager.stop_calls == ["cam_01"]
    assert manager.remove_calls == ["cam_01"]
    assert go2rtc.remove_calls == ["cam_01"]
    assert registry.desired_streams == {}
    assert registry.resolutions == {}


def test_reconcile_streams_re_registers_after_go2rtc_restart():
    events: list[tuple] = []
    camera = _camera()
    go2rtc = FakeGo2RTCManager(events, running=False)
    registry = StreamRegistry(go2rtc)
    manager = FakeCameraManager([camera], events)
    orchestrator = CameraOrchestrator(manager, registry)

    orchestrator.reconcile_streams()
    go2rtc.running = False
    orchestrator.reconcile_streams()

    assert go2rtc.start_calls == [None, None]
    assert go2rtc.register_calls == [
        ("cam_01", "rtsp://example/cam_01", "rtsp"),
        ("cam_01", "rtsp://example/cam_01", "rtsp"),
    ]
