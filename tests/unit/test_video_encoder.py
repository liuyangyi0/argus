"""Tests for MP4 encoder helpers."""

import subprocess
from types import SimpleNamespace

import numpy as np

from argus.core.video_encoder import _create_video_writer, _remux_faststart


def test_create_video_writer_prefers_ffmpeg(monkeypatch, tmp_path):
    calls = []

    class FakeStdin:
        def __init__(self):
            self.writes = []
            self.closed = False

        def write(self, data):
            self.writes.append(data)

        def close(self):
            self.closed = True

    class FakeProc:
        def __init__(self, cmd, **kwargs):
            self.cmd = cmd
            self.kwargs = kwargs
            self.stdin = FakeStdin()
            self.returncode = None
            calls.append(self)

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    def fail_opencv_writer(*_args, **_kwargs):
        raise AssertionError("OpenCV writer should not be tried when ffmpeg opens")

    monkeypatch.setattr("argus.core.video_encoder.shutil.which", lambda name: f"/bin/{name}")
    monkeypatch.setattr("argus.core.video_encoder.subprocess.Popen", FakeProc)
    monkeypatch.setattr("argus.core.video_encoder.cv2.VideoWriter", fail_opencv_writer)

    writer = _create_video_writer(
        tmp_path / "recording.mp4",
        fps=12,
        width=16,
        height=8,
        crf=27,
        preset="fast",
    )
    writer.write(np.zeros((8, 16, 3), dtype=np.uint8))
    writer.release()

    assert len(calls) == 1
    proc = calls[0]
    assert proc.cmd[:2] == ["/bin/ffmpeg", "-y"]
    assert "-c:v" in proc.cmd
    assert proc.cmd[proc.cmd.index("-c:v") + 1] == "libx264"
    assert proc.cmd[proc.cmd.index("-crf") + 1] == "27"
    assert proc.cmd[proc.cmd.index("-preset") + 1] == "fast"
    assert proc.kwargs["stdout"] == subprocess.DEVNULL
    assert proc.kwargs["stderr"] == subprocess.DEVNULL
    assert proc.stdin.closed is True
    assert proc.stdin.writes


def test_remux_faststart_uses_pyav_template_api(monkeypatch, tmp_path):
    mp4_path = tmp_path / "recording.mp4"
    mp4_path.write_bytes(b"ftyp" + b"\0" * 64)
    stream = object()

    class FakeInput:
        streams = SimpleNamespace(video=[stream])

        def demux(self, _stream):
            return []

        def close(self):
            pass

    class FakeOutput:
        def __init__(self, path):
            self.path = path
            self.used_template_api = False

        def add_stream_from_template(self, in_stream):
            assert in_stream is stream
            self.used_template_api = True
            return object()

        def add_stream(self, **_kwargs):
            raise AssertionError("old add_stream(template=...) API should not be used")

        def mux(self, _packet):
            pass

        def close(self):
            assert self.used_template_api
            self.path.write_bytes(b"moov-faststart")

    def fake_open(path, mode="r", options=None):
        if mode == "w":
            assert options == {"movflags": "+faststart"}
            return FakeOutput(tmp_path / "recording.faststart.mp4")
        return FakeInput()

    monkeypatch.setattr("argus.core.video_encoder.av.open", fake_open)

    _remux_faststart(mp4_path)

    assert mp4_path.read_bytes() == b"moov-faststart"


def test_remux_faststart_skips_and_cleans_up_when_template_api_fails(
    monkeypatch, tmp_path,
):
    mp4_path = tmp_path / "recording.mp4"
    mp4_path.write_bytes(b"ftyp" + b"\0" * 64)
    tmp_faststart = tmp_path / "recording.faststart.mp4"

    class FakeInput:
        streams = SimpleNamespace(video=[object()])

        def close(self):
            pass

    class FakeOutput:
        def add_stream_from_template(self, _stream):
            raise TypeError("unsupported template API")

        def close(self):
            tmp_faststart.write_bytes(b"partial")

    def fake_open(path, mode="r", options=None):
        if mode == "w":
            return FakeOutput()
        return FakeInput()

    monkeypatch.setattr("argus.core.video_encoder.av.open", fake_open)

    _remux_faststart(mp4_path)

    assert mp4_path.read_bytes().startswith(b"ftyp")
    assert not tmp_faststart.exists()
