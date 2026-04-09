"""Integration test for alert replay: ring buffer -> disk -> load roundtrip.

Verifies FR-033: solidified recordings are persisted as H.264 MP4 and
loadable via AlertRecordingStore, including post-trigger frame append and heatmaps.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from argus.core.alert_ring_buffer import (
    AlertFrameBuffer,
    FrameSnapshot,
    RecordingStatus,
)
from argus.storage.alert_recording import AlertRecordingStore


def _make_jpeg(width: int = 640, height: int = 480, value: int = 128) -> bytes:
    """Create a valid JPEG frame for testing."""
    frame = np.full((height, width, 3), value, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def _make_snapshot(
    ts: float,
    score: float = 0.5,
    frame_number: int = 0,
    with_heatmap: bool = False,
    simplex_score: float | None = None,
    cusum_evidence: dict | None = None,
    yolo_persons: list | None = None,
) -> FrameSnapshot:
    """Create a synthetic FrameSnapshot for testing with valid JPEG."""
    jpeg = _make_jpeg(value=min(255, int(score * 255)))
    heatmap = None
    if with_heatmap:
        heatmap = np.random.randint(0, 255, (64, 64), dtype=np.uint8)

    return FrameSnapshot(
        timestamp=ts,
        frame_jpeg=jpeg,
        anomaly_score=score,
        simplex_score=simplex_score,
        cusum_evidence=cusum_evidence or {"zone-0": score * 2},
        yolo_persons=yolo_persons or [],
        frame_number=frame_number,
        heatmap_raw=heatmap,
        yolo_boxes=[{"bbox": [10, 20, 30, 40], "class": "person", "confidence": 0.9}]
        if yolo_persons
        else [],
    )


class TestAlertReplayRoundtrip:
    """Test the full ring buffer -> disk -> load cycle."""

    def test_solidify_save_load(self, tmp_path: Path):
        """Solidify recording -> save as MP4 -> load back and verify."""
        buf = AlertFrameBuffer(fps=5, pre_seconds=10, post_seconds=5)
        store = AlertRecordingStore(archive_dir=str(tmp_path / "recordings"))

        # Simulate 10 seconds of frames (50 frames at 5 FPS)
        base_ts = time.time() - 10
        for i in range(50):
            ts = base_ts + i * 0.2  # 5 FPS
            score = 0.3 + (i / 50) * 0.6
            snap = _make_snapshot(
                ts=ts,
                score=score,
                frame_number=i,
                with_heatmap=True,
                simplex_score=score * 0.8,
            )
            buf.append(snap)

        trigger_ts = base_ts + 49 * 0.2
        recording = buf.solidify(
            alert_id="ALT-TEST-001",
            camera_id="CAM-01",
            severity="medium",
            trigger_timestamp=trigger_ts,
        )
        assert recording is not None
        assert recording.alert_id == "ALT-TEST-001"
        assert recording.severity == "medium"
        assert recording.fps == 10  # MEDIUM now 10 FPS (was 2)
        assert len(recording.frames) > 0
        assert recording.status == RecordingStatus.RECORDING

        # Save to disk as MP4
        rec_path, file_size = store.save(recording)
        assert Path(rec_path).exists()
        assert file_size > 0

        # Verify disk files: MP4 instead of frames/
        rec_dir = Path(rec_path)
        assert (rec_dir / "metadata.json").exists()
        assert (rec_dir / "signals.json").exists()
        assert (rec_dir / "pre.mp4").exists()  # Recording in progress
        assert not (rec_dir / "frames").exists()  # No JPEG frame dir

        # Heatmaps should be written as JPEG sequence
        assert (rec_dir / "heatmaps").is_dir()
        heatmap_files = sorted((rec_dir / "heatmaps").glob("*.jpg"))
        assert len(heatmap_files) > 0

        # Load metadata
        metadata = store.load_metadata("ALT-TEST-001")
        assert metadata is not None
        assert metadata["alert_id"] == "ALT-TEST-001"
        assert metadata["camera_id"] == "CAM-01"
        assert metadata["severity"] == "medium"
        assert metadata["fps"] == 10
        assert metadata["frame_count"] == len(recording.frames)
        assert metadata["status"] == "recording"
        assert metadata["codec"] == "h264"
        assert metadata["video_file"] == "pre.mp4"

        # Load signals
        signals = store.load_signals("ALT-TEST-001")
        assert signals is not None
        assert len(signals["timestamps"]) == len(recording.frames)
        assert len(signals["anomaly_scores"]) == len(recording.frames)
        assert len(signals["simplex_scores"]) == len(recording.frames)
        assert "zone-0" in signals["cusum_evidence"]
        assert len(signals["yolo_persons"]) == len(recording.frames)
        assert signals["has_heatmaps"] is True

        # Load a frame (extracted from MP4)
        frame_bytes = store.load_frame("ALT-TEST-001", 0)
        assert frame_bytes is not None
        assert len(frame_bytes) > 0
        # Verify it's a valid JPEG
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert img is not None

        # Load a heatmap
        heatmap_bytes = store.load_heatmap_frame("ALT-TEST-001", 0)
        assert heatmap_bytes is not None
        assert len(heatmap_bytes) > 0

    def test_post_capture_append(self, tmp_path: Path):
        """Verify post-trigger frames are appended via MP4 concat."""
        buf = AlertFrameBuffer(fps=5, pre_seconds=5, post_seconds=3)
        store = AlertRecordingStore(archive_dir=str(tmp_path / "recordings"))

        base_ts = time.time() - 8
        for i in range(25):
            snap = _make_snapshot(
                ts=base_ts + i * 0.2,
                score=0.4,
                frame_number=i,
                with_heatmap=True,
            )
            buf.append(snap)

        trigger_ts = base_ts + 24 * 0.2
        recording = buf.solidify(
            alert_id="ALT-TEST-002",
            camera_id="CAM-02",
            severity="high",
            trigger_timestamp=trigger_ts,
        )
        assert recording is not None
        initial_count = len(recording.frames)

        # Save initial recording
        store.save(recording)

        # Register post-capture
        buf.start_post_capture(
            alert_id="ALT-TEST-002",
            severity="high",
            trigger_timestamp=trigger_ts,
        )

        # Simulate post-trigger frames
        for i in range(15):
            post_ts = trigger_ts + 0.2 * (i + 1)
            snap = _make_snapshot(
                ts=post_ts,
                score=0.3,
                frame_number=25 + i,
                with_heatmap=True,
            )
            buf.append(snap)

        last_post_ts = trigger_ts + 0.2 * 15
        buf._pending_captures["ALT-TEST-002"].deadline = last_post_ts + 0.1

        post_frames = buf.finish_post_capture("ALT-TEST-002")
        assert len(post_frames) > 0

        # Append post-trigger frames (MP4 concat)
        result = store.append_post_frames("ALT-TEST-002", post_frames)
        assert result is True

        # Verify updated metadata
        metadata = store.load_metadata("ALT-TEST-002")
        assert metadata["frame_count"] == initial_count + len(post_frames)
        assert metadata["status"] == "complete"
        assert metadata["video_file"] == "recording.mp4"

        # Verify pre.mp4 and post.mp4 are cleaned up
        rec_dir = store._find_recording_dir("ALT-TEST-002")
        assert not (rec_dir / "pre.mp4").exists()
        assert not (rec_dir / "post.mp4").exists()
        assert (rec_dir / "recording.mp4").exists()

        # Verify signals are merged
        signals = store.load_signals("ALT-TEST-002")
        assert len(signals["timestamps"]) == metadata["frame_count"]
        assert len(signals["anomaly_scores"]) == metadata["frame_count"]

        # Verify post-trigger heatmaps exist
        heatmaps_dir = rec_dir / "heatmaps"
        if heatmaps_dir.exists():
            heatmap_files = sorted(heatmaps_dir.glob("*.jpg"))
            assert len(heatmap_files) > initial_count

    def test_info_severity_single_frame(self, tmp_path: Path):
        """INFO severity stores only trigger frame, no video."""
        buf = AlertFrameBuffer(fps=5, pre_seconds=10, post_seconds=5)
        store = AlertRecordingStore(archive_dir=str(tmp_path / "recordings"))

        base_ts = time.time() - 10
        for i in range(50):
            snap = _make_snapshot(ts=base_ts + i * 0.2, score=0.5, frame_number=i)
            buf.append(snap)

        recording = buf.solidify(
            alert_id="ALT-TEST-003",
            camera_id="CAM-03",
            severity="info",
            trigger_timestamp=base_ts + 49 * 0.2,
        )
        assert recording is not None
        assert len(recording.frames) == 1  # INFO: trigger frame only

        store.save(recording)
        metadata = store.load_metadata("ALT-TEST-003")
        assert metadata["frame_count"] == 1
        assert metadata["fps"] == 1

    def test_low_severity_short_window(self, tmp_path: Path):
        """LOW severity has shorter pre/post windows and 5 FPS."""
        buf = AlertFrameBuffer(fps=15, pre_seconds=60, post_seconds=30)
        store = AlertRecordingStore(archive_dir=str(tmp_path / "recordings"))

        base_ts = time.time() - 60
        for i in range(900):  # 60 seconds at 15 FPS
            snap = _make_snapshot(
                ts=base_ts + i / 15,
                score=0.7,
                frame_number=i,
            )
            buf.append(snap)

        recording = buf.solidify(
            alert_id="ALT-TEST-004",
            camera_id="CAM-04",
            severity="low",
            trigger_timestamp=base_ts + 899 / 15,
        )
        assert recording is not None
        # LOW: pre 10s at 5 FPS -> ~50 frames
        assert recording.fps == 5
        assert len(recording.frames) <= 60  # rough upper bound

        store.save(recording)
        metadata = store.load_metadata("ALT-TEST-004")
        assert metadata["severity"] == "low"
        assert metadata["fps"] == 5

    def test_has_recording_check(self, tmp_path: Path):
        """AlertRecordingStore.has_recording() works after save."""
        store = AlertRecordingStore(archive_dir=str(tmp_path / "recordings"))
        assert store.has_recording("ALT-NONEXISTENT") is False

        buf = AlertFrameBuffer(fps=5, pre_seconds=5, post_seconds=0)
        base_ts = time.time() - 5
        for i in range(25):
            buf.append(_make_snapshot(ts=base_ts + i * 0.2, frame_number=i))

        recording = buf.solidify(
            alert_id="ALT-TEST-005",
            camera_id="CAM-05",
            severity="high",
            trigger_timestamp=base_ts + 24 * 0.2,
        )
        store.save(recording)
        assert store.has_recording("ALT-TEST-005") is True

    def test_cleanup_old_recordings(self, tmp_path: Path):
        """Old recordings are archived (video deleted, trigger frame extracted)."""
        store = AlertRecordingStore(archive_dir=str(tmp_path / "recordings"))
        buf = AlertFrameBuffer(fps=5, pre_seconds=5, post_seconds=0)

        # Create a recording with a fake old timestamp (40 days ago)
        old_ts = time.time() - 40 * 86400
        for i in range(25):
            buf.append(_make_snapshot(ts=old_ts + i * 0.2, frame_number=i))

        recording = buf.solidify(
            alert_id="ALT-OLD-001",
            camera_id="CAM-OLD",
            severity="medium",
            trigger_timestamp=old_ts + 24 * 0.2,
        )
        store.save(recording)

        assert store.has_recording("ALT-OLD-001")
        metadata_before = store.load_metadata("ALT-OLD-001")
        assert metadata_before["frame_count"] > 1

        # Run cleanup with 30-day cutoff
        archived = store.cleanup_old(max_age_days=30)
        assert archived >= 1

        # Metadata should still exist but status = archived
        metadata_after = store.load_metadata("ALT-OLD-001")
        assert metadata_after["status"] == "archived"

        # MP4 should be deleted, trigger_frame.jpg should exist
        rec_dir = store._find_recording_dir("ALT-OLD-001")
        assert not (rec_dir / "pre.mp4").exists()
        assert not (rec_dir / "recording.mp4").exists()
        assert (rec_dir / "trigger_frame.jpg").exists()
        assert (rec_dir / "signals.json").exists()
