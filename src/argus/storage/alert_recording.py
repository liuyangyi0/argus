"""Disk persistence for solidified alert recordings (FR-033).

Stores alert replay data (JPEG frames + signal timeseries) under:
    {archive_dir}/{date}/{camera_id}/{alert_id}/
        frames/       JPEG frame sequence
        signals.json  Timeseries for all 5 replay tracks
        metadata.json Recording metadata
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import structlog

from argus.core.alert_ring_buffer import FrameSnapshot, SolidifiedRecording, RecordingStatus

logger = structlog.get_logger()


def _encode_heatmap_jpeg(heatmap_raw: np.ndarray, quality: int = 60) -> bytes | None:
    """Encode a raw anomaly map to a JET-colorized JPEG. Returns None on failure."""
    hmap = heatmap_raw
    if len(hmap.shape) == 2:
        hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    ok, buf = cv2.imencode(".jpg", hmap, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return buf.tobytes()


class AlertRecordingStore:
    """Manages solidified alert recordings on disk."""

    def __init__(self, archive_dir: str = "data/recordings"):
        self._archive_dir = Path(archive_dir)
        self._path_cache: dict[str, Path] = {}  # alert_id -> rec_dir (O(1) lookup)

    def save(self, recording: SolidifiedRecording) -> str:
        """Persist a solidified recording to disk.

        Returns the recording directory path.
        """
        date_str = datetime.fromtimestamp(
            recording.trigger_timestamp, tz=timezone.utc
        ).strftime("%Y-%m-%d")

        rec_dir = self._archive_dir / date_str / recording.camera_id / recording.alert_id
        frames_dir = rec_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Write frames + optional heatmap overlays (JPEG-encoded at save time)
        heatmaps_dir = rec_dir / "heatmaps"
        has_heatmaps = any(snap.heatmap_raw is not None for snap in recording.frames)
        if has_heatmaps:
            heatmaps_dir.mkdir(parents=True, exist_ok=True)

        total_size = 0
        for i, snap in enumerate(recording.frames):
            frame_path = frames_dir / f"{i:06d}.jpg"
            frame_path.write_bytes(snap.frame_jpeg)
            total_size += len(snap.frame_jpeg)
            if snap.heatmap_raw is not None and has_heatmaps:
                heatmap_bytes = _encode_heatmap_jpeg(snap.heatmap_raw)
                if heatmap_bytes:
                    (heatmaps_dir / f"{i:06d}.jpg").write_bytes(heatmap_bytes)
                    total_size += len(heatmap_bytes)

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
            size_mb=round(total_size / (1024 * 1024), 2),
        )

        return str(rec_dir)

    def append_post_frames(self, alert_id: str, post_frames: list[FrameSnapshot]) -> bool:
        """Append post-trigger frames to an existing recording.

        Updates frame files, signals, and metadata. Returns True on success.
        """
        rec_dir = self._find_recording_dir(alert_id)
        if rec_dir is None:
            logger.warning("alert_recording.append_not_found", alert_id=alert_id)
            return False

        frames_dir = rec_dir / "frames"
        metadata_path = rec_dir / "metadata.json"
        signals_path = rec_dir / "signals.json"

        # Read current metadata
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        existing_count = metadata["frame_count"]

        # Append new frames + heatmaps
        has_heatmaps = any(snap.heatmap_raw is not None for snap in post_frames)
        heatmaps_dir = rec_dir / "heatmaps"
        if has_heatmaps:
            heatmaps_dir.mkdir(parents=True, exist_ok=True)

        for i, snap in enumerate(post_frames):
            frame_path = frames_dir / f"{existing_count + i:06d}.jpg"
            frame_path.write_bytes(snap.frame_jpeg)
            if snap.heatmap_raw is not None and has_heatmaps:
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
        # Merge yolo_boxes for overlay toggle
        if "yolo_boxes" in post_signals:
            if "yolo_boxes" not in existing_signals:
                existing_signals["yolo_boxes"] = []
            existing_signals["yolo_boxes"].extend(post_signals["yolo_boxes"])
        # Update has_heatmaps flag if post frames have heatmaps
        if has_heatmaps:
            existing_signals["has_heatmaps"] = True
        signals_path.write_text(
            json.dumps(existing_signals, ensure_ascii=False), encoding="utf-8"
        )

        # Update metadata
        metadata["frame_count"] = existing_count + len(post_frames)
        metadata["end_timestamp"] = post_frames[-1].timestamp if post_frames else metadata["end_timestamp"]
        metadata["status"] = RecordingStatus.COMPLETE.value
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )

        logger.info(
            "alert_recording.post_frames_appended",
            alert_id=alert_id,
            new_frames=len(post_frames),
            total_frames=metadata["frame_count"],
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
        """Load a single JPEG frame by index."""
        rec_dir = self._find_recording_dir(alert_id)
        if rec_dir is None:
            return None
        frame_path = rec_dir / "frames" / f"{index:06d}.jpg"
        if not frame_path.exists():
            return None
        return frame_path.read_bytes()

    def list_frames(self, alert_id: str) -> list[str]:
        """List all frame filenames for an alert."""
        rec_dir = self._find_recording_dir(alert_id)
        if rec_dir is None:
            return []
        frames_dir = rec_dir / "frames"
        if not frames_dir.exists():
            return []
        return sorted(f.name for f in frames_dir.iterdir() if f.suffix == ".jpg")

    def load_heatmap_frame(self, alert_id: str, index: int) -> bytes | None:
        """Load a single heatmap overlay JPEG by index."""
        rec_dir = self._find_recording_dir(alert_id)
        if rec_dir is None:
            return None
        heatmap_path = rec_dir / "heatmaps" / f"{index:06d}.jpg"
        if not heatmap_path.exists():
            return None
        return heatmap_path.read_bytes()

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

    def cleanup_old(self, max_age_days: int = 30) -> int:
        """Remove recordings older than max_age_days.

        Recordings older than max_age_days are downsampled: only the trigger
        frame and signals.json are kept, video frames are deleted.

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

            # Archive each recording in this date directory
            for camera_dir in date_dir.iterdir():
                if not camera_dir.is_dir():
                    continue
                for rec_dir in camera_dir.iterdir():
                    if not rec_dir.is_dir():
                        continue
                    self._archive_recording(rec_dir)
                    archived += 1

            # Remove date dir if empty
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

    def _find_recording_dir(self, alert_id: str) -> Path | None:
        """Find the recording directory for an alert_id.

        Uses an in-memory cache for O(1) lookups. Falls back to
        filesystem scan on cache miss (cold start or restart).
        """
        # Cache hit
        cached = self._path_cache.get(alert_id)
        if cached is not None and cached.is_dir():
            return cached

        # Filesystem scan fallback
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
        # Collect all zone IDs across all frames
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
            # §1.3: Per-frame YOLO detection boxes for overlay toggle
            "yolo_boxes": [
                f.yolo_boxes or [] for f in frames
            ],
            "has_heatmaps": any(f.heatmap_raw is not None for f in frames),
        }
        return signals

    @staticmethod
    def _archive_recording(rec_dir: Path) -> None:
        """Downsample a recording for archival: keep trigger frame + signals only."""
        meta_path = rec_dir / "metadata.json"
        if not meta_path.exists():
            return

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        trigger_idx = metadata.get("trigger_frame_index", 0)
        frames_dir = rec_dir / "frames"

        # Keep only the trigger frame
        if frames_dir.exists():
            trigger_file = frames_dir / f"{trigger_idx:06d}.jpg"
            trigger_bytes = trigger_file.read_bytes() if trigger_file.exists() else None
            shutil.rmtree(frames_dir)
            if trigger_bytes is not None:
                frames_dir.mkdir()
                (frames_dir / f"{trigger_idx:06d}.jpg").write_bytes(trigger_bytes)

        # Update metadata
        metadata["status"] = RecordingStatus.ARCHIVED.value
        meta_path.write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )
