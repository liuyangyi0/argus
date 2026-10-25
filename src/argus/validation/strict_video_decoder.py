"""Fail-closed FFmpeg video decoding for model-release evidence.

The normal offline corpus builder historically used ``cv2.VideoCapture``.
That API does not expose a trustworthy distinction between clean EOF and a
decoder failure, nor does it preserve per-frame presentation timestamps.  The
decoder in this module deliberately uses two independent FFmpeg entry points:

* ``ffprobe`` decodes the selected video stream and records every frame PTS;
* ``ffmpeg`` decodes the same stream to fixed-size BGR24 frames with ``-xerror``.

Evidence is sufficient only after both processes exit cleanly, their decoded
frame counts agree, no partial raw frame is observed, timestamps are valid,
and the source hash is unchanged.  No dependency on PyAV is required.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping

import numpy as np


_PROBE_TIMEOUT_SECONDS = 300
_TIMESTAMP_SCHEMA_VERSION = "argus.strict-video-frame-timestamps.v1"
_DECODER_EVIDENCE_SCHEMA_VERSION = "argus.strict-video-decoder-evidence.v1"


class StrictDecodeError(OSError):
    """Raised whenever release-grade video evidence cannot be proven."""


@dataclass(frozen=True)
class StrictFrameTimestamp:
    """Exact presentation timestamp for one decoded frame."""

    frame_index: int
    pts: int
    pts_time_seconds: float
    time_base: str
    duration_pts: int | None
    duration_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrictDecodedFrame:
    """One BGR frame paired with the PTS audited by ffprobe."""

    frame_index: int
    frame: np.ndarray
    timestamp: StrictFrameTimestamp


@dataclass(frozen=True)
class _ProbeInventory:
    stream_index: int
    width: int
    height: int
    time_base: str
    declared_frame_count: int | None
    nominal_frame_rate: float
    timestamps: tuple[StrictFrameTimestamp, ...]
    variable_frame_rate: bool
    distinct_pts_deltas: tuple[int, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_binary(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise StrictDecodeError(
            f"required release decoder command is unavailable: {name}"
        )
    return executable


def _passthrough_sync_arguments(executable: str) -> list[str]:
    """Keep every decoded frame on old and new FFmpeg, never guess CFR.

    FFmpeg 9 removed ``-vsync``. Older supported installations may lack
    ``-fps_mode``; discover the actual binary's advertised options rather than
    retrying an unsuccessful decode or silently accepting default frame sync.
    """
    try:
        result = subprocess.run(
            [executable, "-hide_banner", "-h", "full"], check=False,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StrictDecodeError(f"cannot inspect FFmpeg frame-sync capabilities: {exc}") from exc
    if result.returncode != 0:
        raise StrictDecodeError("FFmpeg frame-sync capability inspection failed")
    help_text = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    # Newer builds print e.g. ``-fps_mode[:<stream_spec>]``. The optional
    # per-stream suffix is documentation syntax, not part of the option name.
    options = {
        line.strip().split()[0].split("[", 1)[0].split(":", 1)[0]
        for line in help_text.splitlines() if line.strip()
    }
    if "-fps_mode" in options:
        return ["-fps_mode", "passthrough"]
    if "-vsync" in options:
        return ["-vsync", "0"]
    raise StrictDecodeError("FFmpeg advertises no supported lossless frame-sync option")


def _positive_fraction(raw: object, *, field: str) -> Fraction:
    if type(raw) is not str:
        raise StrictDecodeError(f"ffprobe {field} is missing or not a string")
    try:
        value = Fraction(raw)
    except (ValueError, ZeroDivisionError) as exc:
        raise StrictDecodeError(f"ffprobe {field} is invalid: {raw!r}") from exc
    if value <= 0:
        raise StrictDecodeError(f"ffprobe {field} must be positive: {raw!r}")
    return value


def _optional_positive_rate(raw: object) -> float | None:
    if type(raw) is not str:
        return None
    try:
        value = Fraction(raw)
    except (ValueError, ZeroDivisionError):
        return None
    if value <= 0:
        return None
    result = float(value)
    return result if math.isfinite(result) and result > 0.0 else None


def _strict_int(raw: object, *, field: str) -> int:
    if type(raw) is int:
        return raw
    if type(raw) is not str or not raw or raw.strip() != raw:
        raise StrictDecodeError(f"ffprobe {field} is missing or invalid")
    try:
        return int(raw)
    except ValueError as exc:
        raise StrictDecodeError(f"ffprobe {field} is not an integer: {raw!r}") from exc


def _optional_nonnegative_int(raw: object, *, field: str) -> int | None:
    if raw in (None, "", "N/A"):
        return None
    value = _strict_int(raw, field=field)
    if value < 0:
        raise StrictDecodeError(f"ffprobe {field} cannot be negative")
    return value


def _finite_float(raw: object, *, field: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise StrictDecodeError(f"ffprobe {field} is missing or invalid") from exc
    if not math.isfinite(value):
        raise StrictDecodeError(f"ffprobe {field} must be finite")
    return value


def _parse_probe_payload(payload: Mapping[str, Any]) -> _ProbeInventory:
    """Validate ffprobe JSON without accepting guessed timestamps."""

    if payload.get("error"):
        raise StrictDecodeError(f"ffprobe reported an error: {payload['error']!r}")
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise StrictDecodeError(
            "ffprobe must identify exactly one selected primary video stream"
        )
    stream = streams[0]
    if not isinstance(stream, Mapping):
        raise StrictDecodeError("ffprobe video stream metadata is malformed")
    if stream.get("codec_type") not in (None, "video"):
        raise StrictDecodeError("ffprobe selected stream is not video")

    stream_index = _strict_int(stream.get("index"), field="stream.index")
    width = _strict_int(stream.get("width"), field="stream.width")
    height = _strict_int(stream.get("height"), field="stream.height")
    if width <= 0 or height <= 0:
        raise StrictDecodeError("ffprobe video dimensions must be positive")
    if width * height > 200_000_000:
        raise StrictDecodeError("ffprobe video dimensions exceed the safety limit")

    time_base_fraction = _positive_fraction(
        stream.get("time_base"),
        field="stream.time_base",
    )
    time_base = f"{time_base_fraction.numerator}/{time_base_fraction.denominator}"
    declared_frame_count = _optional_nonnegative_int(
        stream.get("nb_frames"),
        field="stream.nb_frames",
    )

    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise StrictDecodeError("ffprobe produced no decoded video-frame inventory")
    timestamps: list[StrictFrameTimestamp] = []
    previous_pts: int | None = None
    for frame_index, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, Mapping):
            raise StrictDecodeError(
                f"ffprobe frame {frame_index} metadata is malformed"
            )
        if raw_frame.get("media_type") not in (None, "video"):
            raise StrictDecodeError(
                f"ffprobe frame {frame_index} is not a video frame"
            )
        raw_stream_index = raw_frame.get("stream_index")
        if raw_stream_index is not None and _strict_int(
            raw_stream_index,
            field=f"frames[{frame_index}].stream_index",
        ) != stream_index:
            raise StrictDecodeError(
                f"ffprobe frame {frame_index} belongs to a different stream"
            )

        pts = _strict_int(
            raw_frame.get("pts"),
            field=f"frames[{frame_index}].pts",
        )
        if previous_pts is not None and pts <= previous_pts:
            raise StrictDecodeError(
                "ffprobe frame PTS values must be strictly increasing; "
                f"frame {frame_index} has {pts} after {previous_pts}"
            )
        previous_pts = pts
        exact_seconds = float(pts * time_base_fraction)
        if not math.isfinite(exact_seconds):
            raise StrictDecodeError(
                f"ffprobe frame {frame_index} PTS converts to non-finite time"
            )
        reported_pts_time = _finite_float(
            raw_frame.get("pts_time"),
            field=f"frames[{frame_index}].pts_time",
        )
        # ffprobe renders ``pts_time`` as a decimal string, normally to six
        # places.  Permit only rendering/float error; a coarse stream time base
        # must not make a materially different reported timestamp acceptable.
        tolerance = max(1e-6, abs(exact_seconds) * 1e-12)
        if not math.isclose(
            reported_pts_time,
            exact_seconds,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise StrictDecodeError(
                f"ffprobe frame {frame_index} pts_time disagrees with PTS/time_base"
            )

        duration_pts = _optional_nonnegative_int(
            raw_frame.get("pkt_duration"),
            field=f"frames[{frame_index}].pkt_duration",
        )
        duration_seconds = (
            float(duration_pts * time_base_fraction)
            if duration_pts is not None
            else None
        )
        if duration_seconds is not None and (
            not math.isfinite(duration_seconds) or duration_seconds < 0.0
        ):
            raise StrictDecodeError(
                f"ffprobe frame {frame_index} duration is invalid"
            )
        timestamps.append(
            StrictFrameTimestamp(
                frame_index=frame_index,
                pts=pts,
                pts_time_seconds=exact_seconds,
                time_base=time_base,
                duration_pts=duration_pts,
                duration_seconds=duration_seconds,
            )
        )

    if declared_frame_count is not None and declared_frame_count != len(timestamps):
        raise StrictDecodeError(
            "container frame count disagrees with ffprobe decoded inventory "
            f"({declared_frame_count} != {len(timestamps)})"
        )

    pts_deltas = tuple(
        sorted({
            timestamps[index].pts - timestamps[index - 1].pts
            for index in range(1, len(timestamps))
        })
    )
    if any(delta <= 0 for delta in pts_deltas):
        raise StrictDecodeError("ffprobe produced a non-positive PTS delta")
    variable_frame_rate = len(pts_deltas) > 1

    nominal_frame_rate = _optional_positive_rate(stream.get("avg_frame_rate"))
    if nominal_frame_rate is None:
        nominal_frame_rate = _optional_positive_rate(stream.get("r_frame_rate"))
    if nominal_frame_rate is None and len(timestamps) >= 2:
        span = timestamps[-1].pts_time_seconds - timestamps[0].pts_time_seconds
        if span > 0.0:
            nominal_frame_rate = (len(timestamps) - 1) / span
    if nominal_frame_rate is None or not math.isfinite(nominal_frame_rate):
        raise StrictDecodeError("ffprobe produced no trustworthy nominal frame rate")

    return _ProbeInventory(
        stream_index=stream_index,
        width=width,
        height=height,
        time_base=time_base,
        declared_frame_count=declared_frame_count,
        nominal_frame_rate=nominal_frame_rate,
        timestamps=tuple(timestamps),
        variable_frame_rate=variable_frame_rate,
        distinct_pts_deltas=pts_deltas,
    )


class StrictVideoDecoder(Iterator[StrictDecodedFrame]):
    """Stream all frames while retaining release-grade decoder evidence."""

    def __init__(self, source: str | Path, *, expected_sha256: str) -> None:
        self.source = Path(source).resolve()
        if not self.source.is_file():
            raise FileNotFoundError(f"video not found: {self.source}")
        if (
            type(expected_sha256) is not str
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError("expected_sha256 must be a lowercase SHA256 digest")
        self.expected_sha256 = expected_sha256
        self.source_sha256_before = _sha256(self.source)
        if self.source_sha256_before != expected_sha256:
            raise StrictDecodeError(
                "video changed before strict decoding began: "
                f"{self.source}"
            )

        self.ffprobe_executable = _required_binary("ffprobe")
        self.ffmpeg_executable = _required_binary("ffmpeg")
        self._probe = self._probe_source()
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_file: BinaryIO | None = None
        self._decoded_frame_count = 0
        self._complete = False
        self._ffmpeg_returncode: int | None = None
        self._ffmpeg_stderr = ""
        self._source_sha256_after: str | None = None

    @property
    def timestamps(self) -> tuple[StrictFrameTimestamp, ...]:
        return self._probe.timestamps

    @property
    def nominal_frame_rate(self) -> float:
        return self._probe.nominal_frame_rate

    @property
    def declared_frame_count(self) -> int | None:
        return self._probe.declared_frame_count

    @property
    def probed_frame_count(self) -> int:
        return len(self._probe.timestamps)

    def _probe_source(self) -> _ProbeInventory:
        command = [
            self.ffprobe_executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_frames",
            "-show_error",
            "-show_entries",
            (
                "stream=index,codec_type,width,height,time_base,nb_frames,"
                "avg_frame_rate,r_frame_rate:"
                "frame=media_type,stream_index,pts,pts_time,pkt_duration"
            ),
            "-of",
            "json",
            str(self.source),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=_PROBE_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise StrictDecodeError(f"ffprobe could not audit {self.source}: {exc}") from exc
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        if completed.returncode != 0:
            raise StrictDecodeError(
                f"ffprobe exited {completed.returncode} for {self.source}: {stderr}"
            )
        if stderr:
            raise StrictDecodeError(
                f"ffprobe emitted decoder errors for {self.source}: {stderr}"
            )
        try:
            payload = json.loads(completed.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StrictDecodeError(
                f"ffprobe returned invalid JSON for {self.source}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise StrictDecodeError("ffprobe JSON root must be an object")
        return _parse_probe_payload(payload)

    def __enter__(self) -> StrictVideoDecoder:
        if self._process is not None:
            raise RuntimeError("strict decoder cannot be entered twice")
        sync_arguments = _passthrough_sync_arguments(self.ffmpeg_executable)
        self._stderr_file = tempfile.TemporaryFile(mode="w+b")
        command = [
            self.ffmpeg_executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-err_detect",
            "explode",
            "-noautorotate",
            "-i",
            str(self.source),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            *sync_arguments,
            # The rawvideo encoder otherwise defaults to 1/framerate. VFR
            # timestamps may collapse onto one encoder tick and make the
            # output muxer report duplicate DTS even with passthrough enabled.
            # Keep the audited source time base; do not rewrite source PTS.
            "-enc_time_base",
            self._probe.time_base,
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=self._stderr_file,
            )
        except OSError as exc:
            self._stderr_file.close()
            self._stderr_file = None
            raise StrictDecodeError(
                f"ffmpeg could not start strict decoding for {self.source}: {exc}"
            ) from exc
        return self

    def __iter__(self) -> StrictVideoDecoder:
        return self

    def __next__(self) -> StrictDecodedFrame:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("strict decoder must be used as a context manager")
        if self._complete:
            raise StopIteration

        frame_size = self._probe.width * self._probe.height * 3
        chunks: list[bytes] = []
        bytes_read = 0
        while bytes_read < frame_size:
            chunk = process.stdout.read(frame_size - bytes_read)
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)

        if bytes_read == 0:
            self._finalize_clean_eof()
            raise StopIteration
        if bytes_read != frame_size:
            self._wait_and_capture_stderr()
            raise StrictDecodeError(
                "ffmpeg produced a partial raw frame before termination "
                f"({bytes_read}/{frame_size} bytes) for {self.source}; "
                f"returncode={self._ffmpeg_returncode}, stderr={self._ffmpeg_stderr!r}"
            )
        if self._decoded_frame_count >= len(self._probe.timestamps):
            self._abort()
            raise StrictDecodeError(
                "ffmpeg decoded more frames than ffprobe inventoried for "
                f"{self.source}"
            )

        frame_index = self._decoded_frame_count
        raw = b"".join(chunks)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            self._probe.height,
            self._probe.width,
            3,
        ).copy()
        self._decoded_frame_count += 1
        return StrictDecodedFrame(
            frame_index=frame_index,
            frame=frame,
            timestamp=self._probe.timestamps[frame_index],
        )

    def _wait_and_capture_stderr(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            self._ffmpeg_returncode = process.wait(timeout=_PROBE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            raise StrictDecodeError(
                f"ffmpeg did not terminate after its output closed: {self.source}"
            ) from exc
        if self._stderr_file is not None:
            self._stderr_file.seek(0)
            self._ffmpeg_stderr = self._stderr_file.read().decode(
                "utf-8",
                errors="replace",
            ).strip()

    def _finalize_clean_eof(self) -> None:
        self._wait_and_capture_stderr()
        if self._ffmpeg_returncode != 0:
            raise StrictDecodeError(
                f"ffmpeg exited {self._ffmpeg_returncode} while decoding "
                f"{self.source}: {self._ffmpeg_stderr}"
            )
        if self._ffmpeg_stderr:
            raise StrictDecodeError(
                f"ffmpeg emitted decoder errors for {self.source}: "
                f"{self._ffmpeg_stderr}"
            )
        if self._decoded_frame_count != len(self._probe.timestamps):
            raise StrictDecodeError(
                "ffmpeg decoded frame count disagrees with ffprobe inventory "
                f"({self._decoded_frame_count} != {len(self._probe.timestamps)}) "
                f"for {self.source}"
            )
        self._source_sha256_after = _sha256(self.source)
        if self._source_sha256_after != self.source_sha256_before:
            raise StrictDecodeError(
                f"video changed while strict decoding was in progress: {self.source}"
            )
        self._complete = True

    def _abort(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None:
            self._abort()
        elif not self._complete:
            self._abort()
            raise StrictDecodeError(
                "strict decoder was not consumed through verified clean EOF"
            )
        if self._process is not None and self._process.stdout is not None:
            self._process.stdout.close()
        if self._stderr_file is not None:
            self._stderr_file.close()
        return False

    def evidence(self) -> dict[str, Any]:
        if not self._complete or self._source_sha256_after is None:
            raise StrictDecodeError(
                "strict decoder evidence requested before complete verified decode"
            )
        return {
            "schema_version": _DECODER_EVIDENCE_SCHEMA_VERSION,
            "backend": "ffprobe+ffmpeg-rawvideo",
            "strict_release_decoder": True,
            "source_sha256_before": self.source_sha256_before,
            "source_sha256_after": self._source_sha256_after,
            "source_sha256_matches": (
                self._source_sha256_after == self.expected_sha256
            ),
            "stream_index": self._probe.stream_index,
            "width": self._probe.width,
            "height": self._probe.height,
            "pixel_format": "bgr24",
            "time_base": self._probe.time_base,
            "timestamp_frame_count": len(self._probe.timestamps),
            "pts_strictly_increasing": True,
            "variable_frame_rate_detected": self._probe.variable_frame_rate,
            "vfr_detection_basis": "distinct_adjacent_pts_deltas",
            "distinct_pts_deltas": list(self._probe.distinct_pts_deltas),
            "nominal_frame_rate": self._probe.nominal_frame_rate,
            "container_declared_frame_count": self._probe.declared_frame_count,
            "container_declared_frame_count_matches": (
                self._probe.declared_frame_count is None
                or self._probe.declared_frame_count == self._decoded_frame_count
            ),
            "ffprobe": {
                "returncode": 0,
                "stderr_empty": True,
                "decoded_frame_inventory_count": len(self._probe.timestamps),
            },
            "ffmpeg": {
                "returncode": self._ffmpeg_returncode,
                "stderr_empty": not self._ffmpeg_stderr,
                "decoded_frame_count": self._decoded_frame_count,
                "decoded_frame_count_matches_ffprobe": (
                    self._decoded_frame_count == len(self._probe.timestamps)
                ),
                "partial_raw_frame_detected": False,
                "clean_eof": True,
                "xerror_enabled": True,
            },
            "decode_error_distinguishable_from_clean_eof": True,
            "variable_frame_rate_timestamps_preserved": True,
            "complete_decode_verified": True,
            "source_exhausted": True,
            "truncated": False,
            "sufficient_for_model_release": True,
        }


def write_frame_timestamp_jsonl(
    path: str | Path,
    timestamps: tuple[StrictFrameTimestamp, ...],
) -> dict[str, Any]:
    """Persist the complete PTS inventory and return hash-bound evidence."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        for timestamp in timestamps:
            handle.write(
                json.dumps(
                    timestamp.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    time_base = timestamps[0].time_base if timestamps else None
    return {
        "schema_version": _TIMESTAMP_SCHEMA_VERSION,
        "path": target.name,
        "sha256": _sha256(target),
        "frame_count": len(timestamps),
        "time_base": time_base,
        "record_format": "one_json_object_per_decoded_frame",
        "coordinates": "zero_based_decoded_presentation_order",
    }
