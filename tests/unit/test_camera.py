"""Tests for the camera capture module."""

import sys
import numpy as np
import pytest

from argus.capture.camera import CameraCapture, CameraState, FrameData


class TestCameraCapture:
    def test_initial_state_is_disconnected(self):
        """Camera should start in disconnected state."""
        cam = CameraCapture(camera_id="test", source="fake.mp4", protocol="file")
        assert not cam.state.connected
        assert cam.state.total_frames == 0

    def test_connect_nonexistent_file_fails(self):
        """Connecting to a non-existent file should fail gracefully."""
        cam = CameraCapture(camera_id="test", source="nonexistent.mp4", protocol="file")
        result = cam.connect()
        assert not result
        assert not cam.state.connected
        assert cam.state.error is not None

    def test_read_without_connect_returns_none(self):
        """Reading before connect should return None."""
        cam = CameraCapture(camera_id="test", source="fake.mp4", protocol="file")
        frame = cam.read()
        assert frame is None

    def test_stop_sets_disconnected(self):
        """Stop should disconnect the camera."""
        cam = CameraCapture(camera_id="test", source="fake.mp4", protocol="file")
        cam.stop()
        assert not cam.state.connected

    def test_context_manager(self):
        """Context manager should handle connect/stop."""
        with CameraCapture(
            camera_id="test", source="nonexistent.mp4", protocol="file"
        ) as cam:
            # connect will fail for nonexistent file, but shouldn't raise
            assert not cam.state.connected

    def test_windows_usb_prefers_dshow_then_falls_back(self, monkeypatch):
        """Windows USB capture should try DirectShow before MSMF/default."""
        monkeypatch.setattr("argus.capture.camera.sys.platform", "win32")

        attempts = []

        class FakeCapture:
            def __init__(self, opened):
                self._opened = opened

            def isOpened(self):
                return self._opened

            def release(self):
                pass

            def set(self, *_args, **_kwargs):
                return True

        def fake_videocapture(source, backend=None):
            attempts.append((source, backend))
            if len(attempts) == 1:
                return FakeCapture(False)
            return FakeCapture(True)

        monkeypatch.setattr("argus.capture.camera.cv2.VideoCapture", fake_videocapture)

        cam = CameraCapture(camera_id="usb_cam", source="0", protocol="usb")

        assert cam.connect() is True
        assert attempts[0][1] == getattr(__import__("cv2"), "CAP_DSHOW", attempts[0][1])

    def test_non_windows_usb_uses_default_backend(self, monkeypatch):
        """Non-Windows USB capture should keep using the default backend."""
        monkeypatch.setattr("argus.capture.camera.sys.platform", "linux")

        attempts = []

        class FakeCapture:
            def isOpened(self):
                return True

            def release(self):
                pass

            def set(self, *_args, **_kwargs):
                return True

        def fake_videocapture(source, backend=None):
            attempts.append((source, backend))
            return FakeCapture()

        monkeypatch.setattr("argus.capture.camera.cv2.VideoCapture", fake_videocapture)

        cam = CameraCapture(camera_id="usb_cam", source="0", protocol="usb")

        assert cam.connect() is True
        assert attempts == [(0, None)]


class TestFrameData:
    def test_frame_data_fields(self):
        """FrameData should store all required fields."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fd = FrameData(
            frame=frame,
            camera_id="cam1",
            timestamp=1000.0,
            frame_number=42,
            resolution=(640, 480),
        )
        assert fd.camera_id == "cam1"
        assert fd.frame_number == 42
        assert fd.resolution == (640, 480)
        assert fd.frame.shape == (480, 640, 3)
