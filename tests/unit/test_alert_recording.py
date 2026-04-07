"""Tests for AlertRecordingStore disk persistence (FR-033)."""

import json
import time
import tempfile
from pathlib import Path

import pytest

from argus.core.alert_ring_buffer import (
    FrameSnapshot,
    RecordingStatus,
    SolidifiedRecording,
)
from argus.storage.alert_recording import AlertRecordingStore


def _make_recording(
    alert_id: str = "ALT-TEST-001",
    camera_id: str = "cam_01",
    severity: str = "medium",
    frame_count: int = 10,
) -> SolidifiedRecording:
    """Create a test recording."""
    ts = time.time()
    frames = [
        FrameSnapshot(
            timestamp=ts - frame_count + i,
            frame_jpeg=b"\xff\xd8\xff\xe0" + bytes(100),
            anomaly_score=0.1 + i * 0.05,
            simplex_score=None,
            cusum_evidence={"cam_01:default": 0.5 + i * 0.1},
            yolo_persons=[],
            frame_number=i,
        )
        for i in range(frame_count)
    ]
    return SolidifiedRecording(
        alert_id=alert_id,
        camera_id=camera_id,
        severity=severity,
        trigger_timestamp=ts,
        trigger_frame_index=frame_count - 1,
        frames=frames,
        fps=2,
    )


class TestAlertRecordingStore:
    def test_save_and_load_metadata(self, tmp_path):
        store = AlertRecordingStore(archive_dir=str(tmp_path))
        recording = _make_recording()
        rec_path = store.save(recording)

        assert Path(rec_path).exists()
        metadata = store.load_metadata("ALT-TEST-001")
        assert metadata is not None
        assert metadata["alert_id"] == "ALT-TEST-001"
        assert metadata["camera_id"] == "cam_01"
        assert metadata["severity"] == "medium"
        assert metadata["frame_count"] == 10
        assert metadata["fps"] == 2

    def test_save_and_load_signals(self, tmp_path):
        store = AlertRecordingStore(archive_dir=str(tmp_path))
        recording = _make_recording()
        store.save(recording)

        signals = store.load_signals("ALT-TEST-001")
        assert signals is not None
        assert len(signals["timestamps"]) == 10
        assert len(signals["anomaly_scores"]) == 10
        assert "cam_01:default" in signals["cusum_evidence"]

    def test_load_frame(self, tmp_path):
        store = AlertRecordingStore(archive_dir=str(tmp_path))
        recording = _make_recording(frame_count=5)
        store.save(recording)

        frame = store.load_frame("ALT-TEST-001", 0)
        assert frame is not None
        assert isinstance(frame, bytes)

        # Out of range
        assert store.load_frame("ALT-TEST-001", 100) is None

    def test_list_frames(self, tmp_path):
        store = AlertRecordingStore(archive_dir=str(tmp_path))
        recording = _make_recording(frame_count=5)
        store.save(recording)

        frames = store.list_frames("ALT-TEST-001")
        assert len(frames) == 5
        assert frames[0] == "000000.jpg"

    def test_has_recording(self, tmp_path):
        store = AlertRecordingStore(archive_dir=str(tmp_path))
        assert not store.has_recording("NONEXISTENT")

        store.save(_make_recording())
        assert store.has_recording("ALT-TEST-001")

    def test_append_post_frames(self, tmp_path):
        store = AlertRecordingStore(archive_dir=str(tmp_path))
        recording = _make_recording(frame_count=5)
        store.save(recording)

        ts = time.time()
        post_frames = [
            FrameSnapshot(
                timestamp=ts + i,
                frame_jpeg=b"\xff\xd8\xff\xe0" + bytes(50),
                anomaly_score=0.3,
                simplex_score=None,
                cusum_evidence={"cam_01:default": 0.8},
                yolo_persons=[],
                frame_number=5 + i,
            )
            for i in range(3)
        ]

        result = store.append_post_frames("ALT-TEST-001", post_frames)
        assert result is True

        metadata = store.load_metadata("ALT-TEST-001")
        assert metadata["frame_count"] == 8
        assert metadata["status"] == "complete"

    def test_disk_usage(self, tmp_path):
        store = AlertRecordingStore(archive_dir=str(tmp_path))
        assert store.disk_usage() == 0

        store.save(_make_recording())
        assert store.disk_usage() > 0

    def test_nonexistent_alert(self, tmp_path):
        store = AlertRecordingStore(archive_dir=str(tmp_path))
        assert store.load_metadata("NOPE") is None
        assert store.load_signals("NOPE") is None
        assert store.load_frame("NOPE", 0) is None
        assert store.list_frames("NOPE") == []
