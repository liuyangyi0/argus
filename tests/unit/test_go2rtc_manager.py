"""Tests for the go2rtc bootstrap helper.

These cover the shared startup sequence used by both ``__main__`` and the
dashboard lifespan. The real ``Go2RTCManager`` is mocked so the test does
not need a go2rtc binary available on the test host.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from argus.config.schema import CameraConfig, USBCaptureConfig
from argus.streaming.go2rtc_manager import (
    CameraSourceResolution,
    gige_to_go2rtc_source,
    register_go2rtc_streams,
    runtime_camera_config,
    start_and_register_cameras,
    usb_to_go2rtc_source,
)


def _make_cam(
    camera_id: str,
    source: str,
    protocol: str | None,
    *,
    capture_script: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        camera_id=camera_id,
        source=source,
        protocol=protocol,
        gige=SimpleNamespace(capture_script=capture_script),
    )


def _manager(
    *,
    running: bool = False,
    register_return: str | None = "rtsp://127.0.0.1:8554/cam",
) -> MagicMock:
    m = MagicMock()
    type(m).running = property(lambda self: running)
    m.register_camera.return_value = register_return
    return m


class TestStartAndRegisterCameras:
    def test_usb_source_includes_explicit_high_fps_mode(self):
        source = usb_to_go2rtc_source(
            "0",
            resolution=(1920, 1080),
            fps=60,
            pixel_format="mjpeg",
        )

        assert source == (
            "ffmpeg:device?video=0&input_format=mjpeg"
            "&video_size=1920x1080&framerate=60#video=h264"
        )

    def test_usb_source_can_select_device_by_name_or_id(self):
        by_name = usb_to_go2rtc_source(
            "0",
            device_name="OBSBOT Meet 2 StreamCamera",
            resolution=(1920, 1080),
            fps=60,
            pixel_format="mjpeg",
        )
        by_id = usb_to_go2rtc_source(
            "0",
            device_id="@device_pnp_\\\\?\\usb#vid_3564&pid_3022",
            resolution=(1920, 1080),
            fps=60,
            pixel_format="mjpeg",
        )

        assert "video=OBSBOT+Meet+2+StreamCamera" in by_name
        assert "video=%40device_pnp_" in by_id
        assert "video=0" not in by_name
        assert "video=0" not in by_id

    def test_starts_when_not_running(self):
        mgr = _manager(running=False)
        cameras = [_make_cam("cam_01", "rtsp://192.168.1.10/s1", "rtsp")]

        resolutions = register_go2rtc_streams(mgr, cameras)

        mgr.start.assert_called_once_with(initial_streams=None)
        mgr.register_camera.assert_called_once_with("cam_01", "rtsp://192.168.1.10/s1", "rtsp")
        assert resolutions["cam_01"].runtime_source == "rtsp://192.168.1.10/s1"
        assert resolutions["cam_01"].runtime_protocol == "rtsp"

    def test_skips_start_when_already_running(self):
        """Idempotency guard: re-invocation during dashboard lifespan must not restart."""
        mgr = _manager(running=True)
        cameras = [_make_cam("cam_01", "rtsp://192.168.1.10/s1", "rtsp")]

        start_and_register_cameras(mgr, cameras)

        mgr.start.assert_not_called()
        mgr.register_camera.assert_called_once()

    def test_repeated_stream_sync_reuses_manager_registry(self):
        mgr = _manager(running=True)
        cameras = [_make_cam("cam_01", "rtsp://192.168.1.10/s1", "rtsp")]

        register_go2rtc_streams(mgr, cameras)
        register_go2rtc_streams(mgr, cameras)

        mgr.start.assert_not_called()
        mgr.register_camera.assert_called_once_with("cam_01", "rtsp://192.168.1.10/s1", "rtsp")

    def test_usb_camera_source_is_not_rewritten(self):
        """USB config stays persisted while runtime config uses the RTSP re-stream."""
        mgr = _manager(running=False, register_return="rtsp://127.0.0.1:8554/cam_usb")
        cam = CameraConfig(camera_id="cam_usb", name="USB", source="0", protocol="usb")

        resolutions = register_go2rtc_streams(mgr, [cam])
        runtime_cam = runtime_camera_config(cam, resolutions["cam_usb"])

        mgr.register_camera.assert_called_once_with(
            "cam_usb",
            usb_to_go2rtc_source(
                "0",
                resolution=cam.resolution,
                fps=cam.fps_target,
                pixel_format=cam.usb.pixel_format,
            ),
            "rtsp",
        )
        assert cam.source == "0"
        assert cam.protocol == "usb"
        assert runtime_cam is not cam
        assert runtime_cam.source == "rtsp://127.0.0.1:8554/cam_usb"
        assert runtime_cam.protocol == "rtsp"

    def test_usb_camera_can_be_registered_by_stable_device_id(self):
        mgr = _manager(running=False, register_return="rtsp://127.0.0.1:8554/cam_usb")
        cam = CameraConfig(
            camera_id="cam_usb",
            name="USB",
            source="0",
            protocol="usb",
            usb=USBCaptureConfig(device_id="@device_pnp_usb_vid_3564_pid_3022"),
        )

        register_go2rtc_streams(mgr, [cam])

        mgr.register_camera.assert_called_once_with(
            "cam_usb",
            usb_to_go2rtc_source(
                "0",
                device_id="@device_pnp_usb_vid_3564_pid_3022",
                resolution=cam.resolution,
                fps=cam.fps_target,
                pixel_format=cam.usb.pixel_format,
            ),
            "rtsp",
        )

    def test_rtsp_camera_source_is_not_rewritten(self):
        """RTSP cameras keep their original source even if register_camera returned a URL."""
        mgr = _manager(running=False, register_return="rtsp://127.0.0.1:8554/whatever")
        cam = _make_cam("cam_01", "rtsp://192.168.1.10/s1", "rtsp")

        resolutions = register_go2rtc_streams(mgr, [cam])

        assert cam.source == "rtsp://192.168.1.10/s1"
        assert cam.protocol == "rtsp"
        assert resolutions["cam_01"].runtime_source == "rtsp://192.168.1.10/s1"
        assert resolutions["cam_01"].runtime_protocol == "rtsp"

    def test_usb_not_rewritten_when_register_returns_none(self):
        """If go2rtc refuses to register the USB stream, leave the camera alone."""
        mgr = _manager(running=False, register_return=None)
        cam = _make_cam("cam_usb", "/dev/video0", "usb")

        resolutions = register_go2rtc_streams(mgr, [cam])

        mgr.register_camera.assert_called_once_with(
            "cam_usb",
            usb_to_go2rtc_source("/dev/video0"),
            "rtsp",
        )
        assert cam.source == "/dev/video0"
        assert cam.protocol == "usb"
        assert resolutions["cam_usb"].runtime_source == "/dev/video0"
        assert resolutions["cam_usb"].runtime_protocol == "usb"
        assert resolutions["cam_usb"].go2rtc_managed is False

    def test_skips_cameras_with_missing_fields(self):
        mgr = _manager(running=False)
        cameras = [
            _make_cam("", "rtsp://x", "rtsp"),
            _make_cam("cam_01", "", "rtsp"),
            _make_cam("cam_02", "rtsp://y", "rtsp"),
        ]

        resolutions = register_go2rtc_streams(mgr, cameras)

        mgr.register_camera.assert_called_once_with("cam_02", "rtsp://y", "rtsp")
        assert list(resolutions) == ["cam_02"]

    def test_mixed_usb_and_rtsp(self):
        mgr = _manager(running=False)
        mgr.register_camera.side_effect = [
            "rtsp://192.168.1.10/s1",       # cam_rtsp — unchanged
            "rtsp://127.0.0.1:8554/cam_u",  # cam_usb — redirect
        ]
        cam_rtsp = _make_cam("cam_rtsp", "rtsp://192.168.1.10/s1", "rtsp")
        cam_usb = _make_cam("cam_usb", "/dev/video0", "usb")

        resolutions = register_go2rtc_streams(mgr, [cam_rtsp, cam_usb])

        assert cam_rtsp.source == "rtsp://192.168.1.10/s1"
        assert cam_rtsp.protocol == "rtsp"
        assert cam_usb.source == "/dev/video0"
        assert cam_usb.protocol == "usb"
        assert resolutions["cam_rtsp"].runtime_source == "rtsp://192.168.1.10/s1"
        assert resolutions["cam_usb"].runtime_source == "rtsp://127.0.0.1:8554/cam_u"
        assert resolutions["cam_usb"].runtime_protocol == "rtsp"

    def test_empty_camera_list_still_starts_go2rtc(self):
        """Dashboard-only mode may pass in an empty list — go2rtc should still come up."""
        mgr = _manager(running=False)

        start_and_register_cameras(mgr, [])

        mgr.start.assert_called_once_with(initial_streams=None)
        mgr.register_camera.assert_not_called()

    def test_no_default_protocol_none_handled(self):
        """Cameras with protocol=None are treated as RTSP (matches old __main__ behavior)."""
        mgr = _manager(running=False)
        cam = SimpleNamespace(camera_id="cam_01", source="rtsp://x", protocol=None)

        start_and_register_cameras(mgr, [cam])

        mgr.register_camera.assert_called_once_with("cam_01", "rtsp://x", "rtsp")

    def test_gige_initial_stream_pre_registered_but_runtime_stays_gige(self):
        mgr = _manager(running=False, register_return="rtsp://127.0.0.1:8554/gige_01")
        cam = _make_cam(
            "gige_01",
            "192.168.1.20",
            "gige",
            capture_script="scripts/gige_capture.ps1",
        )

        resolutions = register_go2rtc_streams(mgr, [cam])

        mgr.start.assert_called_once_with(
            initial_streams={
                "gige_01": gige_to_go2rtc_source("scripts/gige_capture.ps1"),
            },
        )
        mgr.register_camera.assert_called_once_with("gige_01", "192.168.1.20", "gige")
        assert cam.source == "192.168.1.20"
        assert cam.protocol == "gige"
        assert resolutions["gige_01"].runtime_source == "192.168.1.20"
        assert resolutions["gige_01"].runtime_protocol == "gige"
        assert resolutions["gige_01"].go2rtc_managed is True

    def test_runtime_camera_config_accepts_resolution_object(self):
        cam = CameraConfig(camera_id="cam_usb", name="USB", source="0", protocol="usb")
        resolution = CameraSourceResolution(
            camera_id="cam_usb",
            original_source="0",
            original_protocol="usb",
            runtime_source="rtsp://127.0.0.1:8554/cam_usb",
            runtime_protocol="rtsp",
            stream_name="cam_usb",
            go2rtc_managed=True,
        )

        runtime_cam = runtime_camera_config(cam, resolution)

        assert cam.source == "0"
        assert cam.protocol == "usb"
        assert runtime_cam.source == "rtsp://127.0.0.1:8554/cam_usb"
        assert runtime_cam.protocol == "rtsp"
