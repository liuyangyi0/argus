"""Integration test for alert replay: ring buffer → disk → load roundtrip.

Verifies FR-033: solidified recordings are persisted and loadable via
AlertRecordingStore, including post-trigger frame append and heatmaps.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from argus.core.alert_ring_buffer import (
    AlertFrameBuffer,
    FrameSnapshot,
    RecordingStatus,
)
from argus.storage.alert_recording import AlertRecordingStore


def _make_snapshot(
    ts: float,
    score: float = 0.5,
    frame_number: int = 0,
    with_heatmap: bool = False,
    simplex_score: float | None = None,
    cusum_evidence: dict | None = None,
    yolo_persons: list | None = None,
) -> FrameSnapshot:
    """Create a synthetic FrameSnapshot for testing."""
    # Minimal valid JPEG (1x1 white pixel)
    jpeg = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.\' ',#\x1c\x1c(7),01444\x1f\'9=82<.342"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
        b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
        b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04"
        b"\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"
        b"\x22q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16"
        b"\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz"
        b"\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99"
        b"\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7"
        b"\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5"
        b"\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1"
        b"\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa"
        b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd2\x8a(\x03\xff\xd9"
    )
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
    """Test the full ring buffer → disk → load cycle."""

    def test_solidify_save_load(self, tmp_path: Path):
        """Solidify recording → save to disk → load back and verify."""
        buf = AlertFrameBuffer(fps=5, pre_seconds=10, post_seconds=5)
        store = AlertRecordingStore(archive_dir=str(tmp_path / "recordings"))

        # Simulate 10 seconds of frames (50 frames at 5 FPS)
        base_ts = time.time() - 10
        for i in range(50):
            ts = base_ts + i * 0.2  # 5 FPS
            score = 0.3 + (i / 50) * 0.6  # gradually increasing
            snap = _make_snapshot(
                ts=ts,
                score=score,
                frame_number=i,
                with_heatmap=True,
                simplex_score=score * 0.8,
            )
            buf.append(snap)

        # Trigger alert at the last frame
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
        assert recording.fps == 2  # MEDIUM downsamples to 2 FPS
        assert len(recording.frames) > 0
        assert recording.status == RecordingStatus.RECORDING

        # Save to disk
        rec_path = store.save(recording)
        assert Path(rec_path).exists()

        # Verify disk files
        rec_dir = Path(rec_path)
        assert (rec_dir / "metadata.json").exists()
        assert (rec_dir / "signals.json").exists()
        assert (rec_dir / "frames").is_dir()
        frame_files = sorted((rec_dir / "frames").glob("*.jpg"))
        assert len(frame_files) == len(recording.frames)

        # Heatmaps should be written
        assert (rec_dir / "heatmaps").is_dir()
        heatmap_files = sorted((rec_dir / "heatmaps").glob("*.jpg"))
        assert len(heatmap_files) > 0

        # Load metadata
        metadata = store.load_metadata("ALT-TEST-001")
        assert metadata is not None
        assert metadata["alert_id"] == "ALT-TEST-001"
        assert metadata["camera_id"] == "CAM-01"
        assert metadata["severity"] == "medium"
        assert metadata["fps"] == 2
        assert metadata["frame_count"] == len(recording.frames)
        assert metadata["status"] == "recording"

        # Load signals
        signals = store.load_signals("ALT-TEST-001")
        assert signals is not None
        assert len(signals["timestamps"]) == len(recording.frames)
        assert len(signals["anomaly_scores"]) == len(recording.frames)
        assert len(signals["simplex_scores"]) == len(recording.frames)
        assert "zone-0" in signals["cusum_evidence"]
        assert len(signals["yolo_persons"]) == len(recording.frames)
        assert signals["has_heatmaps"] is True

        # Load a frame
        frame_bytes = store.load_frame("ALT-TEST-001", 0)
        assert frame_bytes is not None
        assert len(frame_bytes) > 0

        # Load a heatmap
        heatmap_bytes = store.load_heatmap_frame("ALT-TEST-001", 0)
        assert heatmap_bytes is not None
        assert len(heatmap_bytes) > 0

    def test_post_capture_append(self, tmp_path: Path):
        """Verify post-trigger frames are appended correctly."""
        buf = AlertFrameBuffer(fps=5, pre_seconds=5, post_seconds=3)
        store = AlertRecordingStore(archive_dir=str(tmp_path / "recordings"))

        # Use recent timestamps — orphan cleanup checks real time.time(),
        # so past timestamps would be cleaned before we access them.
        # All 25 frames span 25*0.2 = 5s; trigger is at frame 24 = base+4.8s.
        # We need trigger_ts < now so finish_post_capture deadline is expired.
        base_ts = time.time() - 8  # 8 seconds ago (within max_age = 2*(5+3) = 16s)
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

        # Set deadline so all post frames are within window and deadline is expired
        last_post_ts = trigger_ts + 0.2 * 15
        buf._pending_captures["ALT-TEST-002"].deadline = last_post_ts + 0.1

        post_frames = buf.finish_post_capture("ALT-TEST-002")
        assert len(post_frames) > 0

        # Append post-trigger frames
        result = store.append_post_frames("ALT-TEST-002", post_frames)
        assert result is True

        # Verify updated metadata
        metadata = store.load_metadata("ALT-TEST-002")
        assert metadata["frame_count"] == initial_count + len(post_frames)
        assert metadata["status"] == "complete"

        # Verify signals are merged
        signals = store.load_signals("ALT-TEST-002")
        assert len(signals["timestamps"]) == metadata["frame_count"]
        assert len(signals["anomaly_scores"]) == metadata["frame_count"]

        # Verify post-trigger heatmaps exist
        rec_dir = store._find_recording_dir("ALT-TEST-002")
        heatmaps_dir = rec_dir / "heatmaps"
        if heatmaps_dir.exists():
            heatmap_files = sorted(heatmaps_dir.glob("*.jpg"))
            assert len(heatmap_files) > initial_count  # post-frames added heatmaps

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
        """LOW severity has shorter pre/post windows and 2 FPS."""
        buf = AlertFrameBuffer(fps=5, pre_seconds=60, post_seconds=30)
        store = AlertRecordingStore(archive_dir=str(tmp_path / "recordings"))

        base_ts = time.time() - 60
        for i in range(300):  # 60 seconds at 5 FPS
            snap = _make_snapshot(
                ts=base_ts + i * 0.2,
                score=0.7,
                frame_number=i,
            )
            buf.append(snap)

        recording = buf.solidify(
            alert_id="ALT-TEST-004",
            camera_id="CAM-04",
            severity="low",
            trigger_timestamp=base_ts + 299 * 0.2,
        )
        assert recording is not None
        # LOW: pre 10s + trigger at 2 FPS → ~20 frames
        assert recording.fps == 2
        assert len(recording.frames) <= 25  # rough upper bound

        store.save(recording)
        metadata = store.load_metadata("ALT-TEST-004")
        assert metadata["severity"] == "low"
        assert metadata["fps"] == 2

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
        """Old recordings are archived (video deleted, trigger frame kept)."""
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

        # Verify it exists
        assert store.has_recording("ALT-OLD-001")
        metadata_before = store.load_metadata("ALT-OLD-001")
        assert metadata_before["frame_count"] > 1

        # Run cleanup with 30-day cutoff
        archived = store.cleanup_old(max_age_days=30)
        assert archived >= 1

        # Metadata should still exist but status = archived
        metadata_after = store.load_metadata("ALT-OLD-001")
        assert metadata_after["status"] == "archived"

        # Only trigger frame should remain
        rec_dir = store._find_recording_dir("ALT-OLD-001")
        frames = list((rec_dir / "frames").glob("*.jpg"))
        assert len(frames) <= 1
