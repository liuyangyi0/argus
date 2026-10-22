"""Contract tests for the declarative go2rtc stream registry."""

from __future__ import annotations

from types import SimpleNamespace

from argus.camera import CameraRuntimePlanner
from argus.config.schema import CameraConfig, GigEConfig, USBCaptureConfig
from argus.streaming.go2rtc_manager import gige_to_go2rtc_source, usb_to_go2rtc_source
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
        self.rtsp_port = 8554
        self.register_return = register_return
        self.raise_register = False
        self.start_calls: list[dict[str, str] | None] = []
        self.stop_calls = 0
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

    def stop(self) -> None:
        self.stop_calls += 1
        self.running = False

    def remove_stream(self, name: str) -> None:
        self.remove_calls.append(name)


def _camera(camera_id: str, source: str, protocol: str | None = "rtsp") -> SimpleNamespace:
    return SimpleNamespace(camera_id=camera_id, source=source, protocol=protocol)


def test_ensure_registered_is_idempotent_for_same_declaration():
    manager = FakeGo2RTCManager(running=True)
    registry = StreamRegistry(manager)
    camera = _camera("cam_01", "rtsp://192.0.2.10/s1")

    first = registry.ensure_registered(camera)
    second = registry.ensure_registered(camera)

    assert first is second
    assert manager.register_calls == [("cam_01", "rtsp://192.0.2.10/s1", "rtsp")]


def test_undeclare_camera_removes_registered_stream():
    manager = FakeGo2RTCManager(running=True)
    registry = StreamRegistry(manager)
    registry.ensure_registered(_camera("cam_01", "rtsp://192.0.2.10/s1"))

    removed = registry.undeclare_camera("cam_01")

    assert removed is True
    assert manager.remove_calls == ["cam_01"]
    assert registry.reconcile(start_if_needed=False) == {}


def test_reconcile_registers_plan_and_removes_stale_streams():
    manager = FakeGo2RTCManager(running=False)
    registry = StreamRegistry(manager)
    cam_rtsp = _camera("cam_rtsp", "rtsp://192.0.2.10/s1")
    cam_usb = _camera("cam_usb", "0", "usb")

    resolutions = registry.reconcile(SimpleNamespace(cameras=[cam_rtsp, cam_usb]))

    assert manager.start_calls == [None]
    assert manager.register_calls == [
        ("cam_rtsp", "rtsp://192.0.2.10/s1", "rtsp"),
        ("cam_usb", usb_to_go2rtc_source("0"), "rtsp"),
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
    camera = _camera("cam_01", "rtsp://192.0.2.10/s1")

    failed = registry.ensure_registered(camera)

    assert failed is not None
    assert failed.go2rtc_managed is False
    assert failed.runtime_source == "rtsp://192.0.2.10/s1"
    assert manager.register_calls == [("cam_01", "rtsp://192.0.2.10/s1", "rtsp")]

    manager.raise_register = False
    recovered = registry.ensure_registered(camera)

    assert recovered is not None
    assert recovered.go2rtc_managed is True
    assert manager.register_calls == [
        ("cam_01", "rtsp://192.0.2.10/s1", "rtsp"),
        ("cam_01", "rtsp://192.0.2.10/s1", "rtsp"),
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


def test_runtime_camera_config_does_not_rewrite_rtsp_detection_source():
    manager = FakeGo2RTCManager(running=True)
    registry = StreamRegistry(manager)
    camera = CameraConfig(
        camera_id="cam_rtsp",
        name="RTSP",
        source="rtsp://camera.local/stream",
        protocol="rtsp",
    )

    registry.ensure_registered(camera)
    runtime_camera = registry.runtime_camera_config(camera)

    assert camera.source == "rtsp://camera.local/stream"
    assert runtime_camera.source == "rtsp://camera.local/stream"
    assert runtime_camera.protocol == "rtsp"
    assert registry.resolutions["cam_rtsp"].stream_name == "cam_rtsp"


def test_file_camera_is_not_go2rtc_managed():
    manager = FakeGo2RTCManager(running=True)
    registry = StreamRegistry(manager)
    camera = CameraConfig(
        camera_id="file_cam",
        name="File",
        source="data/dev/demo.mp4",
        protocol="file",
    )

    resolution = registry.ensure_registered(camera)

    assert resolution is not None
    assert resolution.go2rtc_managed is False
    assert resolution.runtime_source == "data/dev/demo.mp4"
    assert manager.register_calls == []


def test_reconcile_re_registers_after_go2rtc_restart():
    manager = FakeGo2RTCManager(running=False)
    registry = StreamRegistry(manager)
    registry.declare_camera(_camera("cam_01", "rtsp://192.0.2.10/s1"))

    registry.reconcile()
    registry.reconcile()

    assert manager.start_calls == [None]
    assert manager.register_calls == [("cam_01", "rtsp://192.0.2.10/s1", "rtsp")]

    manager.running = False
    registry.reconcile()

    assert manager.start_calls == [None, None]
    assert manager.register_calls == [
        ("cam_01", "rtsp://192.0.2.10/s1", "rtsp"),
        ("cam_01", "rtsp://192.0.2.10/s1", "rtsp"),
    ]


def test_reconcile_accepts_runtime_plan_for_gige_initial_streams():
    manager = FakeGo2RTCManager(running=False)
    registry = StreamRegistry(manager)
    capture_script = "scripts/gige_capture.ps1"
    camera = CameraConfig(
        camera_id="gige_01",
        name="GigE",
        source="192.0.2.20",
        protocol="gige",
        gige=GigEConfig(capture_script=capture_script),
    )
    plan = CameraRuntimePlanner.build(camera)

    resolutions = registry.reconcile([plan])

    assert manager.start_calls == [{
        "gige_01": gige_to_go2rtc_source(capture_script),
    }]
    assert manager.register_calls == []
    assert resolutions["gige_01"].go2rtc_managed is True
    assert resolutions["gige_01"].runtime_source == "192.0.2.20"
    assert resolutions["gige_01"].runtime_protocol == "gige"
    assert resolutions["gige_01"].stream_name == "gige_01"


def test_usb_exec_source_is_loaded_in_initial_config_for_nvenc():
    manager = FakeGo2RTCManager(running=False)
    registry = StreamRegistry(manager)
    exec_source = (
        "exec:ffmpeg -f v4l2 -i /dev/video0 -c:v h264_nvenc "
        "-f rtsp {output}#video=h264"
    )
    camera = CameraConfig(
        camera_id="obsbot",
        name="OBSBOT",
        source="/dev/video0",
        protocol="usb",
        usb=USBCaptureConfig(
            pixel_format="mjpeg",
            go2rtc_source=exec_source,
        ),
    )

    resolutions = registry.reconcile([camera])

    assert manager.start_calls == [{"obsbot": exec_source}]
    assert manager.register_calls == []
    assert resolutions["obsbot"].go2rtc_managed is True
    assert resolutions["obsbot"].runtime_source == "rtsp://127.0.0.1:8554/obsbot"
    assert registry.runtime_camera_config(camera).protocol == "rtsp"

    changed_source = exec_source.replace("h264_nvenc", "h264_nvenc -preset p1")
    changed = camera.model_copy(update={
        "usb": camera.usb.model_copy(update={"go2rtc_source": changed_source}),
    })
    changed_resolutions = registry.reconcile([changed])

    assert manager.stop_calls == 1
    assert manager.start_calls[-1] == {"obsbot": changed_source}
    assert changed_resolutions["obsbot"].go2rtc_managed is True


def test_exec_source_restart_replays_every_dynamic_stream():
    manager = FakeGo2RTCManager(running=False)
    registry = StreamRegistry(manager)
    rtsp = CameraConfig(
        camera_id="rtsp",
        name="RTSP",
        source="rtsp://camera.invalid/live",
        protocol="rtsp",
    )
    first_source = "exec:ffmpeg -i /dev/video0 -c:v h264_nvenc {output}"
    usb = CameraConfig(
        camera_id="obsbot",
        name="OBSBOT",
        source="/dev/video0",
        protocol="usb",
        usb=USBCaptureConfig(go2rtc_source=first_source),
    )
    registry.reconcile([rtsp, usb])

    changed = usb.model_copy(update={
        "usb": usb.usb.model_copy(update={
            "go2rtc_source": first_source + "#video=h264",
        }),
    })
    registry.reconcile([rtsp, changed])

    assert manager.stop_calls == 1
    assert manager.register_calls == [
        ("rtsp", "rtsp://camera.invalid/live", "rtsp"),
        ("rtsp", "rtsp://camera.invalid/live", "rtsp"),
    ]
    assert registry.resolutions["rtsp"].go2rtc_managed is True


def test_delete_and_readd_exec_source_rewrites_initial_config():
    manager = FakeGo2RTCManager(running=False)
    registry = StreamRegistry(manager)
    exec_source = "exec:ffmpeg -i /dev/video0 -c:v h264_nvenc {output}"
    camera = CameraConfig(
        camera_id="obsbot",
        name="OBSBOT",
        source="/dev/video0",
        protocol="usb",
        usb=USBCaptureConfig(go2rtc_source=exec_source),
    )
    registry.reconcile([camera])

    assert registry.undeclare_camera(camera) is True
    restored = registry.ensure_registered(camera, start_if_needed=True)

    assert manager.stop_calls == 2
    assert manager.start_calls == [
        {"obsbot": exec_source},
        None,
        {"obsbot": exec_source},
    ]
    assert restored is not None and restored.go2rtc_managed is True
    assert manager.remove_calls == []


def test_reconcile_accepts_runtime_plan_for_usb_runtime_redirect():
    manager = FakeGo2RTCManager(running=True)
    registry = StreamRegistry(manager)
    camera = CameraConfig(camera_id="usb_01", name="USB", source="0", protocol="usb")
    plan = CameraRuntimePlanner.build(camera)

    resolutions = registry.reconcile([plan], start_if_needed=False)

    assert manager.register_calls == [(
        "usb_01",
        usb_to_go2rtc_source(
            "0",
            resolution=camera.resolution,
            fps=camera.fps_target,
            pixel_format=camera.usb.pixel_format,
        ),
        "rtsp",
    )]
    assert resolutions["usb_01"].runtime_source == "rtsp://127.0.0.1:8554/usb_01"
    assert resolutions["usb_01"].runtime_protocol == "rtsp"
