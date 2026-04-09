"""Disk persistence for solidified alert recordings (FR-033).

Stores alert replay data (H.264 MP4 video + signal timeseries) under:
    {archive_dir}/{date}/{camera_id}/{alert_id}/
        recording.mp4   H.264 video (or pre.mp4 during recording)
        heatmaps/       JPEG heatmap overlays (optional)
        signals.json    Timeseries for all 5 replay tracks
        metadata.json   Recording metadata
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import av
import cv2
import numpy as np
import structlog

from argus.core.alert_ring_buffer import FrameSnapshot, SolidifiedRecording, RecordingStatus
from argus.core.video_encoder import (
    Mp4Encoder, concat_mp4, extract_frame_jpeg, jpeg_dimensions, _BYTES_PER_MB,
)

logger = structlog.get_logger()


def _find_video_file(rec_dir: Path) -> Path | None:
    """Return the active video file in a recording directory, or None."""
    for name in ("recording.mp4", "pre.mp4"):
        p = rec_dir / name
        if p.exists():
            return p
    return None


def _encode_heatmap_jpeg(heatmap_raw: np.ndarray, quality: int = 60) -> bytes | None:
    """Encode a raw anomaly map to a JET-colorized JPEG. Returns None on failure."""
    hmap = heatmap_raw
    if len(hmap.shape) == 2:
        hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    ok, buf = cv2.imencode(".jpg", hmap, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return buf.tobytes()


def _detect_frame_dimensions(frames: list[FrameSnapshot]) -> tuple[int, int]:
    """Detect (width, height) from the first valid JPEG frame header (no full decode)."""
    for snap in frames:
        if snap.frame_jpeg:
            dims = jpeg_dimensions(snap.frame_jpeg)
            if dims is not None:
                return dims
    return (1280, 720)  # fallback


class AlertRecordingStore:
    """Manages solidified alert recordings on disk as H.264 MP4 video."""

    def __init__(
        self,
        archive_dir: str = "data/recordings",
        video_crf: int = 23,
        video_preset: str = "veryfast",
    ):
        self._archive_dir = Path(archive_dir)
        self._video_crf = video_crf
        self._video_preset = video_preset
        self._path_cache: dict[str, Path] = {}  # alert_id -> rec_dir (O(1) lookup)

    def save(self, recording: SolidifiedRecording) -> tuple[str, int]:
        """Persist a solidified recording to disk as MP4 video.

        Returns (recording_dir_path, file_size_bytes).
        """
        date_str = datetime.fromtimestamp(
            recording.trigger_timestamp, tz=timezone.utc
        ).strftime("%Y-%m-%d")

        rec_dir = self._archive_dir / date_str / recording.camera_id / recording.alert_id
        rec_dir.mkdir(parents=True, exist_ok=True)

        # Determine video filename based on recording status
        # pre.mp4 during recording (post-trigger window still collecting)
        # recording.mp4 when complete
        is_recording = recording.status == RecordingStatus.RECORDING
        video_name = "pre.mp4" if is_recording else "recording.mp4"
        video_path = rec_dir / video_name

        # Detect frame dimensions from actual data
        width, height = _detect_frame_dimensions(recording.frames)

        # Encode frames to H.264 MP4
        file_size = 0
        if recording.frames:
            encoder = Mp4Encoder(
                video_path,
                fps=recording.fps,
                width=width,
                height=height,
                crf=self._video_crf,
                preset=self._video_preset,
            )
            for snap in recording.frames:
                encoder.write_jpeg_frame(snap.frame_jpeg)
            file_size = encoder.finalize()

        # Write heatmap overlays as JPEG sequence (sparse, used for Canvas overlay)
        has_heatmaps = any(snap.heatmap_raw is not None for snap in recording.frames)
        if has_heatmaps:
            heatmaps_dir = rec_dir / "heatmaps"
            heatmaps_dir.mkdir(parents=True, exist_ok=True)
            for i, snap in enumerate(recording.frames):
                if snap.heatmap_raw is not None:
                    heatmap_bytes = _encode_heatmap_jpeg(snap.heatmap_raw)
                    if heatmap_bytes:
                        (heatmaps_dir / f"{i:06d}.jpg").write_bytes(heatmap_bytes)

        # Build signals timeseries
        signals = self._build_signals(recording.frames)
        (rec_dir / "signals.json").write_text(
            json.dumps(signals, ensure_ascii=False), encoding="utf-8"
        )

        # Write metadata
        metadata = {
            "alert_id": recording.alert_id,
            "camera_id": recording.camera_id,
            "severity": recording.severity,
            "trigger_timestamp": recording.trigger_timestamp,
            "trigger_frame_index": recording.trigger_frame_index,
            "start_timestamp": recording.frames[0].timestamp if recording.frames else 0,
            "end_timestamp": recording.frames[-1].timestamp if recording.frames else 0,
            "frame_count": len(recording.frames),
            "fps": recording.fps,
            "linked_alert_id": recording.linked_alert_id,
            "status": recording.status.value,
            "video_file": video_name,
            "width": width,
            "height": height,
            "codec": "h264",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (rec_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )

        self._path_cache[recording.alert_id] = rec_dir

        logger.info(
            "alert_recording.saved",
            alert_id=recording.alert_id,
            camera_id=recording.camera_id,
            severity=recording.severity,
            frame_count=len(recording.frames),
            size_mb=round(file_size / _BYTES_PER_MB, 2),
            video=video_name,
        )

        return str(rec_dir), file_size

    def append_post_frames(self, alert_id: str, post_frames: list[FrameSnapshot]) -> bool:
        """Append post-trigger frames to an existing recording.

        Encodes post_frames as post.mp4, concatenates with pre.mp4 to produce
        recording.mp4, then cleans up temp files. Returns True on success.
        """
        rec_dir = self._find_recording_dir(alert_id)
        if rec_dir is None:
            logger.warning("alert_recording.append_not_found", alert_id=alert_id)
            return False

        metadata_path = rec_dir / "metadata.json"
        signals_path = rec_dir / "signals.json"
        pre_path = rec_dir / "pre.mp4"
        post_path = rec_dir / "post.mp4"
        final_path = rec_dir / "recording.mp4"

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        existing_count = metadata["frame_count"]
        width = metadata.get("width", 1280)
        height = metadata.get("height", 720)
        fps = metadata.get("fps", 15)

        # Encode post-trigger frames to post.mp4
        if post_frames:
            encoder = Mp4Encoder(
                post_path,
                fps=fps,
                width=width,
                height=height,
                crf=self._video_crf,
                preset=self._video_preset,
            )
            for snap in post_frames:
                encoder.write_jpeg_frame(snap.frame_jpeg)
            encoder.finalize()

        # Concatenate pre.mp4 + post.mp4 -> recording.mp4
        if pre_path.exists() and post_path.exists():
            file_size = concat_mp4(
                pre_path, post_path, final_path,
                crf=self._video_crf, preset=self._video_preset,
            )
            pre_path.unlink()
            post_path.unlink()
        elif pre_path.exists():
            pre_path.rename(final_path)
            file_size = final_path.stat().st_size
        else:
            logger.warning("alert_recording.no_pre_mp4", alert_id=alert_id)
            return False

        # Append heatmaps (continue numbering from existing_count)
        has_heatmaps = any(snap.heatmap_raw is not None for snap in post_frames)
        if has_heatmaps:
            heatmaps_dir = rec_dir / "heatmaps"
            heatmaps_dir.mkdir(parents=True, exist_ok=True)
            for i, snap in enumerate(post_frames):
                if snap.heatmap_raw is not None:
                    heatmap_bytes = _encode_heatmap_jpeg(snap.heatmap_raw)
                    if heatmap_bytes:
                        (heatmaps_dir / f"{existing_count + i:06d}.jpg").write_bytes(
                            heatmap_bytes
                        )

        # Update signals
        existing_signals = json.loads(signals_path.read_text(encoding="utf-8"))
        post_signals = self._build_signals(post_frames)
        for key in ("timestamps", "anomaly_scores", "simplex_scores"):
            existing_signals[key].extend(post_signals[key])
        for zone_id in post_signals["cusum_evidence"]:
            if zone_id not in existing_signals["cusum_evidence"]:
                existing_signals["cusum_evidence"][zone_id] = []
            existing_signals["cusum_evidence"][zone_id].extend(
                post_signals["cusum_evidence"][zone_id]
            )
        existing_signals["yolo_persons"].extend(post_signals["yolo_persons"])
        if "yolo_boxes" in post_signals:
            if "yolo_boxes" not in existing_signals:
                existing_signals["yolo_boxes"] = []
            existing_signals["yolo_boxes"].extend(post_signals["yolo_boxes"])
        if has_heatmaps:
            existing_signals["has_heatmaps"] = True
        signals_path.write_text(
            json.dumps(existing_signals, ensure_ascii=False), encoding="utf-8"
        )

        # Update metadata
        metadata["frame_count"] = existing_count + len(post_frames)
        metadata["end_timestamp"] = (
            post_frames[-1].timestamp if post_frames else metadata["end_timestamp"]
        )
        metadata["status"] = RecordingStatus.COMPLETE.value
        metadata["video_file"] = "recording.mp4"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )

        logger.info(
            "alert_recording.post_frames_appended",
            alert_id=alert_id,
            new_frames=len(post_frames),
            total_frames=metadata["frame_count"],
            size_mb=round(file_size / _BYTES_PER_MB, 2),
        )
        return True

    def load_metadata(self, alert_id: str) -> dict | None:
        """Load recording metadata for an alert."""
        rec_dir = self._find_recording_dir(alert_id)
        if rec_dir is None:
            return None
        meta_path = rec_dir / "metadata.json"
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def load_signals(self, alert_id: str) -> dict | None:
        """Load signal timeseries for an alert."""
        rec_dir = self._find_recording_dir(alert_id)
        if rec_dir is None:
            return None
        sig_path = rec_dir / "signals.json"
        if not sig_path.exists():
            return None
        return json.loads(sig_path.read_text(encoding="utf-8"))

    def load_frame(self, alert_id: str, index: int) -> bytes | None:
        """Extract a single JPEG frame by index from the MP4 video."""
        rec_dir = self._find_recording_dir(alert_id)
        if rec_dir is None:
            return None
        video_path = self._get_video_file(rec_dir)
        if video_path is None:
            return None
        return extract_frame_jpeg(video_path, index)

    def load_heatmap_frame(self, alert_id: str, index: int) -> bytes | None:
        """Load a single heatmap overlay JPEG by index."""
        rec_dir = self._find_recording_dir(alert_id)
        if rec_dir is None:
            return None
        heatmap_path = rec_dir / "heatmaps" / f"{index:06d}.jpg"
        if not heatmap_path.exists():
            return None
        return heatmap_path.read_bytes()

    def get_video_path(self, alert_id: str) -> Path | None:
        """Return the path to the MP4 video file for direct streaming."""
        rec_dir = self._find_recording_dir(alert_id)
        if rec_dir is None:
            return None
        return self._get_video_file(rec_dir)

    @property
    def archive_dir(self) -> Path:
        """Public access to the archive directory path."""
        return self._archive_dir

    def pin_frame(self, alert_id: str, index: int, label: str) -> bool:
        """Add a user-pinned key frame bookmark to recording metadata."""
        rec_dir = self._find_recording_dir(alert_id)
        if rec_dir is None:
            return False
        meta_path = rec_dir / "metadata.json"
        if not meta_path.exists():
            return False
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "pinned_frames" not in meta:
            meta["pinned_frames"] = []
        meta["pinned_frames"].append({"index": index, "label": label, "type": "user"})
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        return True

    def has_recording(self, alert_id: str) -> bool:
        """Check if a recording exists for the given alert."""
        return self._find_recording_dir(alert_id) is not None

    def reindex_moov(self) -> int:
        """Re-mux all existing MP4 files with moov atom at the beginning.

        Fixes videos encoded before the movflags +faststart fix.
        Returns number of files re-muxed.
        """
        if not self._archive_dir.exists():
            return 0

        count = 0
        for mp4_path in self._archive_dir.rglob("*.mp4"):
            # Check if moov is already at the front
            try:
                with open(mp4_path, "rb") as f:
                    data = f.read(64)
            except OSError:
                continue
            if len(data) < 16:
                continue
            ftyp_size = int.from_bytes(data[0:4], "big")
            pos = ftyp_size
            # Skip 'free' boxes after ftyp
            while pos + 8 <= len(data):
                box_type = data[pos + 4: pos + 8]
                box_size = int.from_bytes(data[pos: pos + 4], "big")
                if box_type == b"free":
                    pos += box_size
                    continue
                break
            if pos + 8 <= len(data) and data[pos + 4: pos + 8] == b"moov":
                continue  # already faststart, skip

            # Re-mux with faststart
            tmp_path = mp4_path.with_suffix(".tmp.mp4")
            inp = None
            out = None
            try:
                inp = av.open(str(mp4_path))
                in_stream = inp.streams.video[0]
                fps = int(in_stream.average_rate) if in_stream.average_rate else 15
                out = av.open(
                    str(tmp_path), mode="w",
                    options={"movflags": "+faststart"},
                )
                out_stream = out.add_stream("libx264", rate=fps)
                out_stream.width = in_stream.width
                out_stream.height = in_stream.height
                out_stream.pix_fmt = "yuv420p"
                out_stream.options = {
                    "crf": str(self._video_crf),
                    "preset": self._video_preset,
                    "profile": "baseline",
                }
                out_stream.gop_size = fps

                idx = 0
                for frame in inp.decode(in_stream):
                    frame.pts = idx
                    for pkt in out_stream.encode(frame):
                        out.mux(pkt)
                    idx += 1
                for pkt in out_stream.encode():
                    out.mux(pkt)
                out.close()
                out = None
                inp.close()
                inp = None

                # Replace original
                mp4_path.unlink()
                tmp_path.rename(mp4_path)
                count += 1
                logger.info("alert_recording.reindex_moov", path=str(mp4_path))
            except Exception as e:
                logger.warning("alert_recording.reindex_moov_failed", path=str(mp4_path), error=str(e))
            finally:
                if out is not None:
                    try:
                        out.close()
                    except Exception:
                        pass
                if inp is not None:
                    try:
                        inp.close()
                    except Exception:
                        pass
                if tmp_path.exists() and mp4_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass

        if count > 0:
            logger.info("alert_recording.reindex_moov_done", count=count)
        return count

    def cleanup_old(self, max_age_days: int = 30) -> int:
        """Remove recordings older than max_age_days.

        Recordings older than max_age_days are archived: only the trigger
        frame (extracted from MP4) and signals.json are kept.

        Returns the number of recordings archived.
        """
        if not self._archive_dir.exists():
            return 0

        cutoff = time.time() - (max_age_days * 86400)
        archived = 0

        for date_dir in sorted(self._archive_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
                if dir_date.timestamp() >= cutoff:
                    continue
            except ValueError:
                continue

            for camera_dir in date_dir.iterdir():
                if not camera_dir.is_dir():
                    continue
                for rec_dir in camera_dir.iterdir():
                    if not rec_dir.is_dir():
                        continue
                    self._archive_recording(rec_dir)
                    archived += 1

            if not any(date_dir.iterdir()):
                date_dir.rmdir()

        if archived > 0:
            logger.info("alert_recording.cleanup", archived=archived)
        return archived

    def disk_usage(self) -> int:
        """Total disk usage of all recordings in bytes."""
        if not self._archive_dir.exists():
            return 0
        return sum(f.stat().st_size for f in self._archive_dir.rglob("*") if f.is_file())

    @staticmethod
    def _get_video_file(rec_dir: Path) -> Path | None:
        """Return the active video file path (recording.mp4 or pre.mp4)."""
        return _find_video_file(rec_dir)

    def _find_recording_dir(self, alert_id: str) -> Path | None:
        """Find the recording directory for an alert_id."""
        cached = self._path_cache.get(alert_id)
        if cached is not None and cached.is_dir():
            return cached

        if not self._archive_dir.exists():
            return None
        for date_dir in sorted(self._archive_dir.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            for camera_dir in date_dir.iterdir():
                if not camera_dir.is_dir():
                    continue
                candidate = camera_dir / alert_id
                if candidate.is_dir():
                    self._path_cache[alert_id] = candidate
                    return candidate
        return None

    @staticmethod
    def _build_signals(frames: list[FrameSnapshot]) -> dict:
        """Build signal timeseries dict from frame snapshots."""
        all_zones: set[str] = set()
        for f in frames:
            all_zones.update(f.cusum_evidence.keys())

        signals = {
            "timestamps": [f.timestamp for f in frames],
            "anomaly_scores": [f.anomaly_score for f in frames],
            "simplex_scores": [f.simplex_score for f in frames],
            "cusum_evidence": {
                zone_id: [f.cusum_evidence.get(zone_id, 0.0) for f in frames]
                for zone_id in sorted(all_zones)
            },
            "yolo_persons": [
                {
                    "timestamp": f.timestamp,
                    "count": len(f.yolo_persons),
                    "bboxes": f.yolo_persons,
                }
                for f in frames
            ],
            "yolo_boxes": [f.yolo_boxes or [] for f in frames],
            "has_heatmaps": any(f.heatmap_raw is not None for f in frames),
        }
        return signals

    @staticmethod
    def _archive_recording(rec_dir: Path) -> None:
        """Downsample a recording for archival: extract trigger frame, delete video."""
        meta_path = rec_dir / "metadata.json"
        if not meta_path.exists():
            return

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        trigger_idx = metadata.get("trigger_frame_index", 0)

        # Extract trigger frame from video before deleting
        video_path = _find_video_file(rec_dir)
        trigger_jpeg = None
        if video_path is not None:
            trigger_jpeg = extract_frame_jpeg(video_path, trigger_idx)
            video_path.unlink()

        # Save trigger frame as standalone JPEG
        if trigger_jpeg:
            (rec_dir / "trigger_frame.jpg").write_bytes(trigger_jpeg)

        # Remove heatmaps directory
        heatmaps_dir = rec_dir / "heatmaps"
        if heatmaps_dir.exists():
            shutil.rmtree(heatmaps_dir)

        # Update metadata
        metadata["status"] = RecordingStatus.ARCHIVED.value
        metadata["video_file"] = None
        meta_path.write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )
