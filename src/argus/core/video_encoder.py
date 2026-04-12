"""H.264 MP4 video encoder for alert recordings.

Provides frame-level encoding from JPEG bytes to H.264 MP4,
MP4 concatenation, and frame extraction.

Encoding uses cv2.VideoWriter which manages PTS timestamps
automatically — eliminates manual PTS bugs that caused playback
issues with PyAV (too-fast or single-frame playback).

Reading / seeking uses PyAV which has superior seeking performance.
"""

from __future__ import annotations

import io
from pathlib import Path

import av
import cv2
import numpy as np
import structlog
from PIL import Image

logger = structlog.get_logger()

_BYTES_PER_MB = 1_048_576

# Preferred H.264 fourcc codes in order of preference
_H264_FOURCCS = ["avc1", "H264", "h264", "x264"]


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray | None:
    """Decode JPEG bytes to a BGR numpy array. Returns None on failure."""
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def jpeg_dimensions(jpeg_bytes: bytes) -> tuple[int, int] | None:
    """Extract (width, height) from JPEG bytes without full decode.

    Uses Pillow's lazy header parsing — only reads markers, no pixel decode.
    """
    try:
        img = Image.open(io.BytesIO(jpeg_bytes))
        return img.size  # (width, height)
    except Exception:
        return None


def _create_video_writer(
    path: Path, fps: int, width: int, height: int,
) -> cv2.VideoWriter:
    """Create a cv2.VideoWriter with H.264 codec, falling back to mp4v.

    cv2.VideoWriter manages PTS automatically — no manual timestamp
    management needed, eliminating the class of bugs where PyAV's
    time_base negotiation with libx264 caused wrong playback speed.
    """
    for fourcc_str in _H264_FOURCCS:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
        if writer.isOpened():
            return writer
        writer.release()

    # Fallback: MPEG-4 (universally supported)
    logger.warning(
        "video_encoder.h264_unavailable",
        msg="H.264 codec not available in OpenCV build, falling back to mp4v",
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to create VideoWriter at {path} — no usable codec found"
        )
    return writer


def _remux_faststart(mp4_path: Path) -> None:
    """Re-mux an MP4 file with moov atom at the front (faststart).

    Required for HTTP Range seeking in browsers. Uses PyAV for
    lossless remux (no re-encoding — just moves the moov atom).
    Skips silently if the moov is already at the front.
    """
    try:
        # Quick check: read first 32 bytes to see if moov is already at front
        with open(mp4_path, "rb") as f:
            header = f.read(32)
        if b"moov" in header:
            return  # already faststart

        tmp_path = mp4_path.with_suffix(".faststart.mp4")
        inp = av.open(str(mp4_path))
        try:
            out = av.open(str(tmp_path), mode="w", options={"movflags": "+faststart"})
            try:
                in_stream = inp.streams.video[0]
                out_stream = out.add_stream(template=in_stream)
                for packet in inp.demux(in_stream):
                    if packet.dts is None:
                        continue
                    packet.stream = out_stream
                    out.mux(packet)
            finally:
                out.close()
        finally:
            inp.close()

        tmp_path.replace(mp4_path)
    except Exception:
        logger.debug("video_encoder.faststart_skip", path=str(mp4_path), exc_info=True)
        tmp_path = mp4_path.with_suffix(".faststart.mp4")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


class Mp4Encoder:
    """Encodes JPEG frames into an H.264 MP4 file.

    Uses cv2.VideoWriter for encoding — PTS timestamps are managed
    automatically by OpenCV/FFMPEG, eliminating manual PTS bugs.

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
        self._frame_count = 0

        self._writer = _create_video_writer(
            self._output_path, fps, width, height,
        )

    def write_jpeg_frame(self, jpeg_bytes: bytes) -> None:
        """Decode a JPEG buffer and write it as a video frame."""
        bgr = decode_jpeg(jpeg_bytes)
        if bgr is None:
            logger.warning("video_encoder.skip_corrupt_frame", index=self._frame_count)
            return

        h, w = bgr.shape[:2]
        if w != self._width or h != self._height:
            bgr = cv2.resize(bgr, (self._width, self._height), interpolation=cv2.INTER_AREA)

        self._writer.write(bgr)
        self._frame_count += 1

    def finalize(self) -> int:
        """Flush encoder, close writer, apply faststart. Returns file size in bytes."""
        self._writer.release()

        if self._output_path.exists():
            # Re-mux with moov atom at front for browser seeking
            _remux_faststart(self._output_path)

        size = self._output_path.stat().st_size if self._output_path.exists() else 0
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

    Uses PyAV for decoding (superior seeking) and cv2.VideoWriter
    for encoding (automatic PTS management).
    Returns the output file size in bytes.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    in_pre = av.open(str(pre_path))
    in_post = av.open(str(post_path))
    writer = None
    try:
        pre_stream = in_pre.streams.video[0]
        post_stream = in_post.streams.video[0]
        if fps is None:
            fps = int(pre_stream.average_rate) if pre_stream.average_rate else 15

        width, height = pre_stream.width, pre_stream.height
        writer = _create_video_writer(output_path, fps, width, height)

        for frame in in_pre.decode(pre_stream):
            bgr = frame.to_ndarray(format="bgr24")
            if bgr.shape[1] != width or bgr.shape[0] != height:
                bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(bgr)

        for frame in in_post.decode(post_stream):
            bgr = frame.to_ndarray(format="bgr24")
            if bgr.shape[1] != width or bgr.shape[0] != height:
                bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(bgr)
    finally:
        in_pre.close()
        in_post.close()
        if writer is not None:
            writer.release()

    # Apply faststart for browser seeking
    if output_path.exists():
        _remux_faststart(output_path)

    size = output_path.stat().st_size if output_path.exists() else 0
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

    Uses PyAV for decoding and cv2.VideoWriter for encoding to ensure
    correct PTS management. Checks average_rate to decide if repair is needed.
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

            width, height = stream.width, stream.height
            writer = _create_video_writer(tmp_path, fps, width, height)
            frame_count = 0
            try:
                for frame in container.decode(video=0):
                    bgr = frame.to_ndarray(format="bgr24")
                    writer.write(bgr)
                    frame_count += 1
            finally:
                writer.release()
        finally:
            container.close()

        if frame_count == 0:
            tmp_path.unlink(missing_ok=True)
            return False

        # Apply faststart
        _remux_faststart(tmp_path)

        tmp_path.replace(mp4_path)
        logger.info("video_encoder.repaired", path=str(mp4_path), frames=frame_count, fps=fps)
        return True
    except Exception:
        logger.warning("video_encoder.repair_failed", path=str(mp4_path), exc_info=True)
        tmp_path.unlink(missing_ok=True)
        return False
