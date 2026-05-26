"""Tests for the camera capture module."""

import threading
import time

import cv2
import numpy as np

from argus.capture.camera import CameraCapture, FrameData


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

    def test_stop_during_fps_throttle_does_not_read_released_capture(self):
        """Stopping while read() is throttled should wake it without touching _cap."""

        class FakeCapture:
            def release(self):
                pass

            def read(self):
                raise AssertionError("read should not run after stop")

        cam = CameraCapture(
            camera_id="test",
            source="fake.mp4",
            protocol="file",
            fps_target=1,
        )
        cam._cap = FakeCapture()
        cam._state.connected = True
        cam._state.last_frame_time = time.monotonic()

        result = []
        errors = []

        def read_once():
            try:
                result.append(cam.read())
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=read_once)
        thread.start()
        time.sleep(0.05)
        cam.stop()
        thread.join(timeout=1.0)

        assert not thread.is_alive()
        assert errors == []
        assert result == [None]

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

    def test_usb_mjpeg_resolution_and_fps_are_set_before_runtime_check(self, monkeypatch):
        """USB high-FPS mode should force MJPG before width/height/FPS."""
        attempts = []
        set_calls = []

        class FakeCapture:
            def isOpened(self):
                return True

            def release(self):
                pass

            def set(self, prop, value):
                set_calls.append((prop, value))
                return True

            def get(self, prop):
                values = {
                    cv2.CAP_PROP_FRAME_WIDTH: 1920,
                    cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                    cv2.CAP_PROP_FPS: 60,
                    cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
                }
                return values.get(prop, 0)

        def fake_videocapture(source, backend=None):
            attempts.append((source, backend))
            return FakeCapture()

        monkeypatch.setattr("argus.capture.camera.cv2.VideoCapture", fake_videocapture)

        cam = CameraCapture(
            camera_id="usb_cam",
            source="0",
            protocol="usb",
            fps_target=60,
            resolution=(1920, 1080),
            usb_backend="dshow",
            usb_pixel_format="mjpeg",
            usb_min_runtime_fps=50,
        )

        assert cam.connect() is True
        assert attempts == [(0, cv2.CAP_DSHOW)]
        props = [prop for prop, _value in set_calls]
        assert props.index(cv2.CAP_PROP_FOURCC) < props.index(cv2.CAP_PROP_FRAME_WIDTH)
        assert props.index(cv2.CAP_PROP_FRAME_WIDTH) < props.index(cv2.CAP_PROP_FRAME_HEIGHT)
        assert props.index(cv2.CAP_PROP_FRAME_HEIGHT) < props.index(cv2.CAP_PROP_FPS)
        assert cam.state.actual_resolution == (1920, 1080)
        assert cam.state.reported_fps == 60
        assert cam.state.actual_fps == 60
        assert cam.state.requested_pixel_format == "MJPG"
        assert cam.state.pixel_format == "MJPG"
        assert cam.state.degraded is False

    def test_usb_measured_fps_below_minimum_is_degraded_even_when_reported_fps_is_high(
        self, monkeypatch,
    ):
        """A camera can advertise 60 FPS while decoded reads are slower."""

        class FakeCapture:
            def isOpened(self):
                return True

            def release(self):
                pass

            def set(self, *_args, **_kwargs):
                return True

            def get(self, prop):
                values = {
                    cv2.CAP_PROP_FRAME_WIDTH: 1920,
                    cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                    cv2.CAP_PROP_FPS: 60,
                    cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
                }
                return values.get(prop, 0)

        monkeypatch.setattr(
            "argus.capture.camera.cv2.VideoCapture",
            lambda *_args, **_kwargs: FakeCapture(),
        )
        monkeypatch.setattr(
            CameraCapture,
            "_measure_usb_read_fps",
            lambda self: 35.0,
        )

        cam = CameraCapture(
            camera_id="usb_cam",
            source="0",
            protocol="usb",
            fps_target=60,
            resolution=(1920, 1080),
            usb_backend="dshow",
            usb_pixel_format="mjpeg",
            usb_min_runtime_fps=50,
        )

        assert cam.connect() is True
        assert cam.state.reported_fps == 60
        assert cam.state.actual_fps == 35.0
        assert cam.state.degraded is True
        assert "fps 35.0 below required 50.0" in (cam.state.degradation_reason or "")

    def test_usb_runtime_fps_below_minimum_is_degraded(self, monkeypatch):
        """Runtime status should not silently accept a low-FPS USB mode."""

        class FakeCapture:
            def isOpened(self):
                return True

            def release(self):
                pass

            def set(self, *_args, **_kwargs):
                return True

            def get(self, prop):
                values = {
                    cv2.CAP_PROP_FRAME_WIDTH: 1920,
                    cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                    cv2.CAP_PROP_FPS: 30,
                    cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
                }
                return values.get(prop, 0)

        monkeypatch.setattr(
            "argus.capture.camera.cv2.VideoCapture",
            lambda *_args, **_kwargs: FakeCapture(),
        )

        cam = CameraCapture(
            camera_id="usb_cam",
            source="0",
            protocol="usb",
            fps_target=60,
            resolution=(1920, 1080),
            usb_backend="dshow",
            usb_pixel_format="mjpeg",
            usb_min_runtime_fps=50,
        )

        assert cam.connect() is True
        assert cam.state.degraded is True
        assert "below required 50.0" in (cam.state.degradation_reason or "")

    def test_usb_runtime_resolution_or_fourcc_mismatch_is_degraded(self, monkeypatch):
        """Startup validation should expose low mode/FourCC mismatches."""

        class FakeCapture:
            def isOpened(self):
                return True

            def release(self):
                pass

            def set(self, *_args, **_kwargs):
                return True

            def get(self, prop):
                values = {
                    cv2.CAP_PROP_FRAME_WIDTH: 640,
                    cv2.CAP_PROP_FRAME_HEIGHT: 480,
                    cv2.CAP_PROP_FPS: 60,
                    cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"YUY2"),
                }
                return values.get(prop, 0)

        monkeypatch.setattr(
            "argus.capture.camera.cv2.VideoCapture",
            lambda *_args, **_kwargs: FakeCapture(),
        )

        cam = CameraCapture(
            camera_id="usb_cam",
            source="0",
            protocol="usb",
            fps_target=60,
            resolution=(1920, 1080),
            usb_backend="dshow",
            usb_pixel_format="mjpeg",
            usb_min_runtime_fps=50,
        )

        assert cam.connect() is True
        assert cam.state.degraded is True
        reason = cam.state.degradation_reason or ""
        assert "resolution 640x480" in reason
        assert "pixel format YUY2 != requested MJPG" in reason


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
