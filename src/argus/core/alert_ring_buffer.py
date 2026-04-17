"""Per-camera ring buffer for alert replay (FR-033).

Each camera's pipeline appends a FrameSnapshot every frame. When an alert
triggers, the buffer is solidified to disk with differential retention
based on severity level:

  HIGH:   front 60s + back 30s, 15 FPS (original)
  MEDIUM: front 60s + back 30s, 10 FPS
  LOW:    front 10s + back 10s, 5 FPS
  INFO:   trigger frame only (no video)

Memory budget: ~432 MB resident for 4 cameras at 15 FPS.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class FrameSnapshot:
    """Single frame with all concurrent signals for replay."""

    timestamp: float
    frame_jpeg: bytes  # JPEG compressed, ~80 KB at Q85 720p
    anomaly_score: float
    simplex_score: float | None
    cusum_evidence: dict[str, float]  # {zone_id: evidence_value}
    yolo_persons: list[dict]  # [{bbox: [x1,y1,x2,y2], confidence: float}]
    frame_number: int
    # §1.3 overlay data — stored separately for toggle support
    # Raw uint8 anomaly map (not JPEG-encoded) — encoding deferred to solidify time
    heatmap_raw: np.ndarray | None = None
    yolo_boxes: list[dict] | None = None  # all YOLO detections [{bbox, class, confidence}]
    # Active temporal tracks at this frame — serialized for trajectory replay.
    # Each dict: {"track_id": int, "centroid_x": float, "centroid_y": float,
    #             "max_score": float, "area_px": int}
    active_tracks: list[dict] = field(default_factory=list)


class RecordingStatus(str, Enum):
    """Lifecycle status of a solidified recording."""

    RECORDING = "recording"  # post-trigger window still collecting
    COMPLETE = "complete"
    ARCHIVED = "archived"  # downsampled for long-term storage


@dataclass
class SolidifiedRecording:
    """A fixed segment of frames + signals ready for disk persistence."""

    alert_id: str
    camera_id: str
    severity: str
    trigger_timestamp: float
    trigger_frame_index: int  # index within frames list
    frames: list[FrameSnapshot]
    fps: int  # effective FPS after downsampling
    linked_alert_id: str | None = None
    status: RecordingStatus = RecordingStatus.COMPLETE


# Severity-specific retention parameters
@dataclass(frozen=True)
class _RetentionPolicy:
    pre_seconds: int
    post_seconds: int
    target_fps: int  # 0 means "trigger frame only"


_RETENTION = {
    "high": _RetentionPolicy(pre_seconds=60, post_seconds=30, target_fps=15),
    "medium": _RetentionPolicy(pre_seconds=60, post_seconds=30, target_fps=10),
    "low": _RetentionPolicy(pre_seconds=10, post_seconds=10, target_fps=5),
    "info": _RetentionPolicy(pre_seconds=0, post_seconds=0, target_fps=0),  # trigger frame only
}


def compress_frame(frame: np.ndarray, quality: int = 85, max_height: int = 720) -> bytes:
    """Compress a BGR frame to JPEG bytes, scaling to max_height if needed."""
    h, w = frame.shape[:2]
    if h > max_height:
        scale = max_height / h
        frame = cv2.resize(frame, (int(w * scale), max_height), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return b""
    return buf.tobytes()


@dataclass
class _PendingCapture:
    """Tracks a post-trigger capture in progress."""

    alert_id: str
    severity: str
    trigger_timestamp: float
    deadline: float
    target_fps: int


class AlertFrameBuffer:
    """Thread-safe per-camera ring buffer for alert replay.

    Maintains a rolling window of FrameSnapshots. When an alert fires,
    `solidify()` extracts the relevant window and downsamples per severity.

    For alerts that need a post-trigger window (the alert is still
    "recording" for N seconds after trigger), use `start_post_capture()`
    which returns a PostTriggerCapture handle. Continue appending frames
    as normal; when the handle's window expires, call `finish_post_capture()`.
    """

    def __init__(self, fps: int = 15, pre_seconds: int = 60, post_seconds: int = 30):
        self._fps = max(fps, 1)
        self._pre_seconds = pre_seconds
        self._post_seconds = post_seconds
        # Buffer holds enough for the maximum window (pre + post)
        maxlen = self._fps * (pre_seconds + post_seconds)
        self._frames: deque[FrameSnapshot] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        # Active post-trigger captures awaiting completion
        self._pending_captures: dict[str, _PendingCapture] = {}

    def append(self, snapshot: FrameSnapshot) -> None:
        """Append a frame snapshot (called from pipeline thread, every frame)."""
        with self._lock:
            self._frames.append(snapshot)
            # Auto-clean orphaned pending captures that exceed max age (2× post_seconds)
            if self._pending_captures:
                now = time.time()
                max_age = 2 * (self._pre_seconds + self._post_seconds)
                orphaned = [
                    aid for aid, pc in self._pending_captures.items()
                    if now - pc.trigger_timestamp > max_age
                ]
                for aid in orphaned:
                    self._pending_captures.pop(aid, None)
                    logger.warning(
                        "ring_buffer.orphan_capture_cleaned",
                        alert_id=aid,
                        msg="Pending post-capture exceeded max age, cleaned up",
                    )

    def solidify(
        self,
        alert_id: str,
        camera_id: str,
        severity: str,
        trigger_timestamp: float,
        linked_alert_id: str | None = None,
    ) -> SolidifiedRecording | None:
        """Extract and downsample the buffer around the trigger point.

        For INFO severity, returns a single-frame recording.
        For other severities, returns the full pre-trigger window immediately.
        Post-trigger frames are NOT included here — use start_post_capture()
        for that.

        Returns None if the buffer is empty.
        """
        policy = _RETENTION.get(severity, _RETENTION["medium"])

        with self._lock:
            if not self._frames:
                return None

            # INFO: just the trigger frame
            if severity == "info":
                trigger_frame = self._find_closest_frame(trigger_timestamp)
                if trigger_frame is None:
                    return None
                return SolidifiedRecording(
                    alert_id=alert_id,
                    camera_id=camera_id,
                    severity=severity,
                    trigger_timestamp=trigger_timestamp,
                    trigger_frame_index=0,
                    frames=[trigger_frame],
                    fps=1,
                    linked_alert_id=linked_alert_id,
                )

            # Extract pre-trigger window
            pre_cutoff = trigger_timestamp - policy.pre_seconds
            pre_frames = [
                f for f in self._frames if pre_cutoff <= f.timestamp <= trigger_timestamp
            ]

        if not pre_frames:
            return None

        # Downsample if needed (HIGH keeps original FPS, MEDIUM/LOW downsample to 2)
        if policy.target_fps > 0 and policy.target_fps < self._fps:
            pre_frames = self._downsample(pre_frames, policy.target_fps)

        # Compute actual FPS from frame timestamps (not config target)
        if len(pre_frames) >= 2:
            duration = pre_frames[-1].timestamp - pre_frames[0].timestamp
            actual_fps = (len(pre_frames) - 1) / duration if duration > 0 else self._fps
            effective_fps = max(1, round(actual_fps))
        else:
            effective_fps = policy.target_fps if policy.target_fps > 0 else self._fps

        # Find trigger frame index in the downsampled list
        trigger_idx = len(pre_frames) - 1
        for i, f in enumerate(pre_frames):
            if f.timestamp >= trigger_timestamp:
                trigger_idx = i
                break

        return SolidifiedRecording(
            alert_id=alert_id,
            camera_id=camera_id,
            severity=severity,
            trigger_timestamp=trigger_timestamp,
            trigger_frame_index=trigger_idx,
            frames=pre_frames,
            fps=effective_fps,
            linked_alert_id=linked_alert_id,
            status=RecordingStatus.RECORDING,  # post-window not yet captured
        )

    def start_post_capture(
        self, alert_id: str, severity: str, trigger_timestamp: float
    ) -> None:
        """Register a pending post-trigger capture.

        After calling this, continue appending frames. When the post-trigger
        window expires, call finish_post_capture() to collect the frames.
        """
        policy = _RETENTION.get(severity, _RETENTION["medium"])
        if policy.post_seconds <= 0:
            return

        deadline = trigger_timestamp + policy.post_seconds
        with self._lock:
            self._pending_captures[alert_id] = _PendingCapture(
                alert_id=alert_id,
                severity=severity,
                trigger_timestamp=trigger_timestamp,
                deadline=deadline,
                target_fps=policy.target_fps if policy.target_fps > 0 else self._fps,
            )

    def finish_post_capture(self, alert_id: str) -> list[FrameSnapshot]:
        """Collect post-trigger frames for a completed capture.

        Returns the downsampled post-trigger frames, or empty list if
        the capture is not found or not yet ready.
        """
        with self._lock:
            pending = self._pending_captures.pop(alert_id, None)
            if pending is None:
                return []

            now = time.time()
            if now < pending.deadline:
                # Not ready yet — put it back
                self._pending_captures[alert_id] = pending
                return []

            post_frames = [
                f for f in self._frames
                if pending.trigger_timestamp < f.timestamp <= pending.deadline
            ]

        if not post_frames:
            return []

        if pending.target_fps < self._fps:
            post_frames = self._downsample(post_frames, pending.target_fps)

        return post_frames

    def get_pending_captures(self) -> list[str]:
        """Return alert_ids with pending post-trigger captures."""
        with self._lock:
            return list(self._pending_captures.keys())

    def check_expired_captures(self) -> list[str]:
        """Return alert_ids whose post-trigger window has expired."""
        now = time.time()
        with self._lock:
            return [
                aid for aid, pc in self._pending_captures.items()
                if now >= pc.deadline
            ]

    @property
    def frame_count(self) -> int:
        """Current number of frames in the buffer."""
        with self._lock:
            return len(self._frames)

    @property
    def memory_estimate_mb(self) -> float:
        """Rough memory estimate in MB."""
        with self._lock:
            total_bytes = sum(len(f.frame_jpeg) for f in self._frames)
        return total_bytes / (1024 * 1024)

    def _find_closest_frame(self, timestamp: float) -> FrameSnapshot | None:
        """Find the frame closest to the given timestamp (caller holds lock)."""
        if not self._frames:
            return None
        best = min(self._frames, key=lambda f: abs(f.timestamp - timestamp))
        return best

    @staticmethod
    def _downsample(frames: list[FrameSnapshot], target_fps: int) -> list[FrameSnapshot]:
        """Downsample a frame list to target_fps by uniform selection."""
        if not frames or target_fps <= 0:
            return frames

        duration = frames[-1].timestamp - frames[0].timestamp
        if duration <= 0:
            logger.warning(
                "ring_buffer.downsample_zero_duration",
                frame_count=len(frames),
                msg="All frames have same timestamp — returning single frame",
            )
            return frames[:1]

        actual_fps = len(frames) / duration
        if actual_fps <= target_fps:
            return frames  # already at or below target

        # Select frames at uniform intervals
        target_count = max(1, int(duration * target_fps))
        if target_count >= len(frames):
            return frames

        step = len(frames) / target_count
        return [frames[int(i * step)] for i in range(target_count)]
