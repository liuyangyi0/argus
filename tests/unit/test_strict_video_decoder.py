from __future__ import annotations

import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from argus.validation import strict_video_decoder as decoder


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probe_payload(
    pts: list[int],
    *,
    width: int = 1,
    height: int = 1,
    nb_frames: str | None = None,
) -> dict[str, Any]:
    stream: dict[str, Any] = {
        "index": 0,
        "codec_type": "video",
        "width": width,
        "height": height,
        "time_base": "1/1000",
        "avg_frame_rate": "10/1",
        "r_frame_rate": "10/1",
    }
    if nb_frames is not None:
        stream["nb_frames"] = nb_frames
    return {
        "streams": [stream],
        "frames": [
            {
                "media_type": "video",
                "stream_index": 0,
                "pts": value,
                "pts_time": f"{value / 1000:.6f}",
                "pkt_duration": (
                    pts[index + 1] - value
                    if index + 1 < len(pts)
                    else 100
                ),
            }
            for index, value in enumerate(pts)
        ],
    }


def _patch_probe(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
    *,
    returncode: int = 0,
    stderr: bytes = b"",
) -> None:
    monkeypatch.setattr(
        decoder.shutil,
        "which",
        lambda name: f"/usr/bin/{name}",
    )
    completed = subprocess.CompletedProcess(
        args=["ffprobe"],
        returncode=returncode,
        stdout=json.dumps(payload).encode("utf-8"),
        stderr=stderr,
    )
    def run(command, **kwargs):
        if command[-2:] == ["-h", "full"]:
            return subprocess.CompletedProcess(command, 0, b"-fps_mode mode  frame sync method\n", b"")
        return completed

    monkeypatch.setattr(decoder.subprocess, "run", run)


class _FakePopen:
    def __init__(
        self,
        raw_output: bytes,
        *,
        returncode: int,
        stderr_output: bytes,
        stderr_handle,
    ) -> None:
        self.stdout = io.BytesIO(raw_output)
        self.returncode = returncode
        self._stderr_handle = stderr_handle
        stderr_handle.write(stderr_output)
        stderr_handle.flush()

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def poll(self) -> int:
        return self.returncode

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None


def _patch_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
    raw_output: bytes,
    *,
    returncode: int = 0,
    stderr: bytes = b"",
) -> None:
    def make_process(*args, **kwargs):
        command = args[0]
        assert command[command.index("-fps_mode") + 1] == "passthrough"
        assert "-vsync" not in command
        assert command[command.index("-enc_time_base") + 1] == "1/1000"
        return _FakePopen(
            raw_output,
            returncode=returncode,
            stderr_output=stderr,
            stderr_handle=kwargs["stderr"],
        )

    monkeypatch.setattr(decoder.subprocess, "Popen", make_process)


@pytest.mark.parametrize("help_text,expected", [
    (b"-fps_mode mode  sync method\n-vsync mode  deprecated\n", ["-fps_mode", "passthrough"]),
    (b"-fps_mode[:<stream_spec>] mode  sync method\n", ["-fps_mode", "passthrough"]),
    (b"-vsync mode  sync method\n", ["-vsync", "0"]),
])
def test_frame_sync_uses_supported_passthrough_option(monkeypatch, help_text, expected):
    monkeypatch.setattr(decoder.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, help_text, b"",
    ))
    assert decoder._passthrough_sync_arguments("ffmpeg") == expected


@pytest.mark.parametrize("code,help_text", [(0, b"-r rate  force frame rate\n"), (1, b"-fps_mode mode\n")])
def test_missing_or_failed_sync_capability_check_fails_closed(monkeypatch, code, help_text):
    monkeypatch.setattr(decoder.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], code, help_text, b"",
    ))
    with pytest.raises(decoder.StrictDecodeError, match="frame-sync"):
        decoder._passthrough_sync_arguments("ffmpeg")


def test_probe_inventory_preserves_pts_and_detects_vfr() -> None:
    inventory = decoder._parse_probe_payload(_probe_payload([0, 40, 100]))

    assert inventory.time_base == "1/1000"
    assert inventory.variable_frame_rate is True
    assert inventory.distinct_pts_deltas == (40, 60)
    assert [item.pts for item in inventory.timestamps] == [0, 40, 100]
    assert [item.pts_time_seconds for item in inventory.timestamps] == [
        0.0,
        0.04,
        0.1,
    ]


@pytest.mark.parametrize(
    ("payload_mutator", "error"),
    [
        (
            lambda payload: payload["frames"][1].pop("pts"),
            "pts is missing or invalid",
        ),
        (
            lambda payload: payload["frames"][1].update({"pts": 0}),
            "strictly increasing",
        ),
        (
            lambda payload: payload["streams"][0].update({"nb_frames": "99"}),
            "frame count disagrees",
        ),
        (
            lambda payload: payload["frames"][1].update({"pts_time": "9.0"}),
            "pts_time disagrees",
        ),
    ],
)
def test_probe_inventory_rejects_timestamp_or_count_anomalies(
    payload_mutator,
    error: str,
) -> None:
    payload = _probe_payload([0, 100])
    payload_mutator(payload)

    with pytest.raises(decoder.StrictDecodeError, match=error):
        decoder._parse_probe_payload(payload)


def test_missing_ffprobe_command_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(decoder.shutil, "which", lambda name: None)

    with pytest.raises(decoder.StrictDecodeError, match="ffprobe"):
        decoder.StrictVideoDecoder(source, expected_sha256=_sha256(source))


@pytest.mark.parametrize(
    ("returncode", "stderr", "error"),
    [
        (1, b"invalid data", "ffprobe exited 1"),
        (0, b"corrupt packet", "ffprobe emitted decoder errors"),
    ],
)
def test_ffprobe_nonzero_or_error_stderr_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stderr: bytes,
    error: str,
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    _patch_probe(
        monkeypatch,
        _probe_payload([0]),
        returncode=returncode,
        stderr=stderr,
    )

    with pytest.raises(decoder.StrictDecodeError, match=error):
        decoder.StrictVideoDecoder(source, expected_sha256=_sha256(source))


def test_complete_ffmpeg_decode_cross_checks_frame_count_and_source_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    _patch_probe(monkeypatch, _probe_payload([0, 100], nb_frames="2"))
    _patch_ffmpeg(monkeypatch, bytes([1, 2, 3, 4, 5, 6]))

    strict = decoder.StrictVideoDecoder(
        source,
        expected_sha256=_sha256(source),
    )
    with strict:
        frames = list(strict)
    evidence = strict.evidence()

    assert len(frames) == 2
    assert frames[0].frame.tolist() == [[[1, 2, 3]]]
    assert frames[1].timestamp.pts == 100
    assert evidence["ffprobe"]["decoded_frame_inventory_count"] == 2
    assert evidence["ffmpeg"]["decoded_frame_count"] == 2
    assert evidence["ffmpeg"]["clean_eof"] is True
    assert evidence["source_sha256_matches"] is True
    assert evidence["sufficient_for_model_release"] is True


@pytest.mark.parametrize(
    ("raw_output", "returncode", "stderr", "error"),
    [
        (b"", 1, b"decode failed", "ffmpeg exited 1"),
        (b"", 0, b"corrupt packet", "ffmpeg emitted decoder errors"),
        (bytes([1, 2]), 0, b"", "partial raw frame"),
    ],
)
def test_ffmpeg_error_stderr_or_partial_frame_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_output: bytes,
    returncode: int,
    stderr: bytes,
    error: str,
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    _patch_probe(monkeypatch, _probe_payload([0], nb_frames="1"))
    _patch_ffmpeg(
        monkeypatch,
        raw_output,
        returncode=returncode,
        stderr=stderr,
    )
    strict = decoder.StrictVideoDecoder(
        source,
        expected_sha256=_sha256(source),
    )

    with pytest.raises(decoder.StrictDecodeError, match=error):
        with strict:
            list(strict)


def test_source_change_during_decode_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    _patch_probe(monkeypatch, _probe_payload([0], nb_frames="1"))
    _patch_ffmpeg(monkeypatch, bytes([1, 2, 3]))
    strict = decoder.StrictVideoDecoder(
        source,
        expected_sha256=_sha256(source),
    )

    with pytest.raises(decoder.StrictDecodeError, match="video changed"):
        with strict:
            first = next(strict)
            assert first.frame_index == 0
            source.write_bytes(b"changed")
            next(strict)
