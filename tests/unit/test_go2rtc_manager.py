"""Tests for the go2rtc bootstrap helper.

These cover the shared startup sequence used by both ``__main__`` and the
dashboard lifespan. The real ``Go2RTCManager`` is mocked so the test does
not need a go2rtc binary available on the test host.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from argus.streaming.go2rtc_manager import start_and_register_cameras


def _make_cam(camera_id: str, source: str, protocol: str) -> SimpleNamespace:
    return SimpleNamespace(camera_id=camera_id, source=source, protocol=protocol)


def _manager(*, running: bool = False, register_return: str | None = "rtsp://127.0.0.1:8554/cam") -> MagicMock:
    m = MagicMock()
    type(m).running = property(lambda self: running)
    m.register_camera.return_value = register_return
    return m


class TestStartAndRegisterCameras:
    def test_starts_when_not_running(self):
        mgr = _manager(running=False)
        cameras = [_make_cam("cam_01", "rtsp://192.168.1.10/s1", "rtsp")]

        start_and_register_cameras(mgr, cameras)

        mgr.start.assert_called_once()
        mgr.register_camera.assert_called_once_with("cam_01", "rtsp://192.168.1.10/s1", "rtsp")

    def test_skips_start_when_already_running(self):
        """Idempotency guard: re-invocation during dashboard lifespan must not restart."""
        mgr = _manager(running=True)
        cameras = [_make_cam("cam_01", "rtsp://192.168.1.10/s1", "rtsp")]

        start_and_register_cameras(mgr, cameras)

        mgr.start.assert_not_called()
        mgr.register_camera.assert_called_once()

    def test_usb_camera_source_is_rewritten(self):
        """USB cameras must be redirected in-place so the pipeline opens the RTSP re-stream."""
        mgr = _manager(running=False, register_return="rtsp://127.0.0.1:8554/cam_usb")
        cam = _make_cam("cam_usb", "/dev/video0", "usb")

        start_and_register_cameras(mgr, [cam])

        assert cam.source == "rtsp://127.0.0.1:8554/cam_usb"
        assert cam.protocol == "rtsp"

    def test_rtsp_camera_source_is_not_rewritten(self):
        """RTSP cameras keep their original source even if register_camera returned a URL."""
        mgr = _manager(running=False, register_return="rtsp://127.0.0.1:8554/whatever")
        cam = _make_cam("cam_01", "rtsp://192.168.1.10/s1", "rtsp")

        start_and_register_cameras(mgr, [cam])

        assert cam.source == "rtsp://192.168.1.10/s1"
        assert cam.protocol == "rtsp"

    def test_usb_not_rewritten_when_register_returns_none(self):
        """If go2rtc refuses to register the USB stream, leave the camera alone."""
        mgr = _manager(running=False, register_return=None)
        cam = _make_cam("cam_usb", "/dev/video0", "usb")

        start_and_register_cameras(mgr, [cam])

        assert cam.source == "/dev/video0"
        assert cam.protocol == "usb"

    def test_skips_cameras_with_missing_fields(self):
        mgr = _manager(running=False)
        cameras = [
            _make_cam("", "rtsp://x", "rtsp"),
            _make_cam("cam_01", "", "rtsp"),
            _make_cam("cam_02", "rtsp://y", "rtsp"),
        ]

        start_and_register_cameras(mgr, cameras)

        mgr.register_camera.assert_called_once_with("cam_02", "rtsp://y", "rtsp")

    def test_mixed_usb_and_rtsp(self):
        mgr = _manager(running=False)
        mgr.register_camera.side_effect = [
            "rtsp://192.168.1.10/s1",       # cam_rtsp — unchanged
            "rtsp://127.0.0.1:8554/cam_u",  # cam_usb — redirect
        ]
        cam_rtsp = _make_cam("cam_rtsp", "rtsp://192.168.1.10/s1", "rtsp")
        cam_usb = _make_cam("cam_usb", "/dev/video0", "usb")

        start_and_register_cameras(mgr, [cam_rtsp, cam_usb])

        assert cam_rtsp.source == "rtsp://192.168.1.10/s1"
        assert cam_rtsp.protocol == "rtsp"
        assert cam_usb.source == "rtsp://127.0.0.1:8554/cam_u"
        assert cam_usb.protocol == "rtsp"

    def test_empty_camera_list_still_starts_go2rtc(self):
        """Dashboard-only mode may pass in an empty list — go2rtc should still come up."""
        mgr = _manager(running=False)

        start_and_register_cameras(mgr, [])

        mgr.start.assert_called_once()
        mgr.register_camera.assert_not_called()

    def test_no_default_protocol_none_handled(self):
        """Cameras with protocol=None are treated as RTSP (matches old __main__ behavior)."""
        mgr = _manager(running=False)
        cam = SimpleNamespace(camera_id="cam_01", source="rtsp://x", protocol=None)

        start_and_register_cameras(mgr, [cam])

        mgr.register_camera.assert_called_once_with("cam_01", "rtsp://x", "rtsp")
