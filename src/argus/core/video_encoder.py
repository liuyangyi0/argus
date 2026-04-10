"""H.264 MP4 video encoder for alert recordings.

Provides frame-level encoding from JPEG bytes to H.264 MP4,
MP4 concatenation, and frame extraction.
"""

from __future__ import annotations

import struct
from pathlib import Path

import av
import cv2
import numpy as np
import structlog

logger = structlog.get_logger()

_BYTES_PER_MB = 1_048_576


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray | None:
    """Decode JPEG bytes to a BGR numpy array. Returns None on failure."""
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def jpeg_dimensions(jpeg_bytes: bytes) -> tuple[int, int] | None:
    """Parse JPEG header to extract (width, height) without full decode.

    Reads SOF0/SOF2 markers from the JPEG binary data.
    Returns None if the header cannot be parsed.
    """
    data = jpeg_bytes
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return None
    i = 2
    while i < len(data) - 1:
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        # SOF0 (0xC0) or SOF2 (0xC2) contain dimensions
        if marker in (0xC0, 0xC2) and i + 9 < len(data):
            height = struct.unpack(">H", data[i + 5: i + 7])[0]
            width = struct.unpack(">H", data[i + 7: i + 9])[0]
            if width > 0 and height > 0:
                return (width, height)
        if marker == 0xDA:  # Start of Scan — stop searching
            break
        if i + 3 < len(data):
            seg_len = struct.unpack(">H", data[i + 2: i + 4])[0]
            i += 2 + seg_len
        else:
            break
    return None


def _create_h264_output(
    path: Path, fps: int, width: int, height: int,
    crf: int = 23, preset: str = "veryfast",
) -> tuple[av.container.OutputContainer, av.stream.Stream]:
    """Create an H.264 MP4 output container + stream with standard settings."""
    container = av.open(str(path), mode="w", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf), "preset": preset, "profile": "baseline"}
    stream.gop_size = fps  # keyframe every 1 second
    return container, stream


class Mp4Encoder:
    """Encodes JPEG frames into an H.264 MP4 file.

    Usage::

        enc = Mp4Encoder(Path("out.mp4"), fps=15, width=1280, height=720)
        for jpeg_bytes in frames:
            enc.write_jpeg_frame(jpeg_bytes)
        file_size = enc.finalize()
    """

    def __init__(
        self,
        output_path: Path,
        fps: int,
        width: int = 1280,
        height: int = 720,
        crf: int = 23,
        preset: str = "veryfast",
    ):
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._width = width
        self._height = height
        self._crf = crf
        self._preset = preset
        self._frame_count = 0

        self._container, self._stream = _create_h264_output(
            self._output_path, fps, width, height, crf, preset,
        )

    def write_jpeg_frame(self, jpeg_bytes: bytes) -> None:
        """Decode a JPEG buffer and encode it as an H.264 frame."""
        bgr = decode_jpeg(jpeg_bytes)
        if bgr is None:
            logger.warning("video_encoder.skip_corrupt_frame", index=self._frame_count)
            return

        h, w = bgr.shape[:2]
        if w != self._width or h != self._height:
            bgr = cv2.resize(bgr, (self._width, self._height), interpolation=cv2.INTER_AREA)

        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
        frame.pts = self._frame_count
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self._frame_count += 1

    def finalize(self) -> int:
        """Flush encoder, close container. Returns file size in bytes."""
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()

        size = self._output_path.stat().st_size
        logger.info(
            "video_encoder.finalized",
            path=str(self._output_path),
            frames=self._frame_count,
            size_mb=round(size / _BYTES_PER_MB, 2),
        )
        return size

    @property
    def frame_count(self) -> int:
        return self._frame_count


def concat_mp4(
    pre_path: Path,
    post_path: Path,
    output_path: Path,
    crf: int = 23,
    preset: str = "veryfast",
    fps: int | None = None,
) -> int:
    """Concatenate two MP4 files by re-encoding into a single output.

    Both inputs are decoded and re-encoded into one continuous MP4.
    Returns the output file size in bytes.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    in_pre = av.open(str(pre_path))
    in_post = av.open(str(post_path))
    output = None
    try:
        pre_stream = in_pre.streams.video[0]
        post_stream = in_post.streams.video[0]
        if fps is None:
            fps = int(pre_stream.average_rate)

        output, out_stream = _create_h264_output(
            output_path, fps, pre_stream.width, pre_stream.height, crf, preset,
        )

        frame_idx = 0
        for frame in in_pre.decode(pre_stream):
            frame.pts = frame_idx
            for packet in out_stream.encode(frame):
                output.mux(packet)
            frame_idx += 1

        for frame in in_post.decode(post_stream):
            frame.pts = frame_idx
            for packet in out_stream.encode(frame):
                output.mux(packet)
            frame_idx += 1

        for packet in out_stream.encode():
            output.mux(packet)
    finally:
        in_pre.close()
        in_post.close()
        if output is not None:
            output.close()

    size = output_path.stat().st_size
    logger.info("video_encoder.concat", output=str(output_path), size_mb=round(size / _BYTES_PER_MB, 2))
    return size


def extract_frame_jpeg(
    mp4_path: Path, index: int, jpeg_quality: int = 85,
) -> bytes | None:
    """Extract a single frame from an MP4 file by frame index (0-based).

    Uses keyframe seeking for O(GOP_SIZE) decode instead of O(N).
    Returns JPEG bytes or None if index is out of range.
    """
    try:
        container = av.open(str(mp4_path))
    except Exception:
        logger.debug("video_encoder.open_failed", path=str(mp4_path), exc_info=True)
        return None

    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        # Seek to nearest keyframe before target frame
        if index > 0 and stream.average_rate:
            # Convert frame index to timestamp in stream time_base units
            target_pts = int(index * stream.time_base.denominator
                            / (stream.time_base.numerator * stream.average_rate))
            try:
                container.seek(target_pts, stream=stream)
            except av.error.FFmpegError:
                logger.debug("video_encoder.seek_failed", index=index, exc_info=True)

        # Decode forward from seek position, counting frames
        for i, frame in enumerate(container.decode(stream)):
            # After seek, we need to count from the actual position
            # Use sequential enumeration — seek puts us at or before target
            frame_num = i
            # If we seeked, we need the absolute frame number
            if index > 0 and stream.average_rate and frame.pts is not None:
                frame_num = round(float(frame.pts * stream.time_base * stream.average_rate))

            if frame_num == index:
                bgr = frame.to_ndarray(format="bgr24")
                ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                return buf.tobytes() if ok else None
            if frame_num > index:
                break
    finally:
        container.close()

    return None


def get_video_dimensions(mp4_path: Path) -> tuple[int, int] | None:
    """Return (width, height) of the first video stream, or None on error."""
    try:
        container = av.open(str(mp4_path))
        try:
            stream = container.streams.video[0]
            return (stream.width, stream.height)
        finally:
            container.close()
    except Exception:
        return None


def get_video_frame_count(mp4_path: Path) -> int | None:
    """Return the number of frames in the video, or None on error."""
    try:
        container = av.open(str(mp4_path))
        try:
            stream = container.streams.video[0]
            count = stream.frames
            if count == 0:
                # Count packets (no decode, much faster than frame decode)
                count = sum(1 for p in container.demux(stream) if p.size > 0)
            return count
        finally:
            container.close()
    except Exception:
        return None


def repair_video_timestamps(mp4_path: Path, fps: int, crf: int = 23, preset: str = "veryfast") -> bool:
    """Re-encode an MP4 in-place with correct PTS timestamps.

    Checks average_rate to decide if repair is needed, then streams
    decode→encode (one frame at a time) to avoid buffering the whole video.
    Returns True if repaired, False if skipped or failed.
    """
    tmp_path = mp4_path.with_suffix(".tmp.mp4")
    try:
        container = av.open(str(mp4_path))
        try:
            stream = container.streams.video[0]
            avg_rate = float(stream.average_rate) if stream.average_rate else 0
            if avg_rate > 0 and abs(avg_rate - fps) / fps < 0.5:
                return False  # already correct

            frame_count = 0
            out_container, out_stream = _create_h264_output(
                tmp_path, fps, stream.width, stream.height, crf, preset,
            )
            try:
                for frame in container.decode(video=0):
                    frame.pts = frame_count
                    for packet in out_stream.encode(frame):
                        out_container.mux(packet)
                    frame_count += 1

                for packet in out_stream.encode():
                    out_container.mux(packet)
            finally:
                out_container.close()
        finally:
            container.close()

        if frame_count == 0:
            tmp_path.unlink(missing_ok=True)
            return False

        tmp_path.replace(mp4_path)
        logger.info("video_encoder.repaired", path=str(mp4_path), frames=frame_count, fps=fps)
        return True
    except Exception:
        logger.warning("video_encoder.repair_failed", path=str(mp4_path), exc_info=True)
        tmp_path.unlink(missing_ok=True)
        return False
