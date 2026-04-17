"""Tests for AlertRecordingStore disk persistence with H.264 MP4 (FR-033)."""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from argus.core.alert_ring_buffer import (
    FrameSnapshot,
    RecordingStatus,
    SolidifiedRecording,
)
from argus.storage.alert_recording import (
    AlertRecordingStore,
    _aggregate_trajectories,
)


def _make_jpeg(width: int = 640, height: int = 480, value: int = 128) -> bytes:
    """Create a valid JPEG frame for testing."""
    frame = np.full((height, width, 3), value, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def _make_recording(
    alert_id: str = "ALT-TEST-001",
    camera_id: str = "cam_01",
    severity: str = "medium",
    frame_count: int = 10,
    status: RecordingStatus = RecordingStatus.RECORDING,
) -> SolidifiedRecording:
    """Create a test recording with valid JPEG frames."""
    ts = time.time()
    frames = [
        FrameSnapshot(
            timestamp=ts - frame_count + i,
            frame_jpeg=_make_jpeg(value=i * 20),
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
        fps=15,
        status=status,
    )


class TestAlertRecordingStore:
    def test_save_and_load_metadata(self, tmp_path):
        store = AlertRecordingStore(archive_dir=str(tmp_path))
        recording = _make_recording()
        rec_path, file_size = store.save(recording)

        assert Path(rec_path).exists()
        assert file_size > 0

        metadata = store.load_metadata("ALT-TEST-001")
        assert metadata is not None
        assert metadata["alert_id"] == "ALT-TEST-001"
        assert metadata["camera_id"] == "cam_01"
        assert metadata["severity"] == "medium"
        assert metadata["frame_count"] == 10
        assert metadata["fps"] == 15
        assert metadata["codec"] == "h264"
        assert metadata["video_file"] == "pre.mp4"  # status=RECORDING

    def test_save_complete_recording(self, tmp_path):
        store = AlertRecordingStore(archive_dir=str(tmp_path))
        recording = _make_recording(status=RecordingStatus.COMPLETE)
        rec_path, _ = store.save(recording)

        metadata = store.load_metadata("ALT-TEST-001")
        assert metadata["video_file"] == "recording.mp4"
        assert (Path(rec_path) / "recording.mp4").exists()

    def test_save_creates_mp4(self, tmp_path):
        store = AlertRecordingStore(archive_dir=str(tmp_path))
        recording = _make_recording()
        rec_path, _ = store.save(recording)

        rec_dir = Path(rec_path)
        assert (rec_dir / "pre.mp4").exists()
        assert not (rec_dir / "frames").exists()  # No JPEG frame dir
        assert (rec_dir / "signals.json").exists()
        assert (rec_dir / "metadata.json").exists()

    def test_save_and_load_signals(self, tmp_path):
        store = AlertRecordingStore(archive_dir=str(tmp_path))
        recording = _make_recording()
        store.save(recording)

        signals = store.load_signals("ALT-TEST-001")
        assert signals is not None
        assert len(signals["timestamps"]) == 10
        assert len(signals["anomaly_scores"]) == 10
        assert "cam_01:default" in signals["cusum_evidence"]

    def test_load_frame_from_mp4(self, tmp_path):
        store = AlertRecordingStore(archive_dir=str(tmp_path))
        recording = _make_recording(frame_count=5)
        store.save(recording)

        frame = store.load_frame("ALT-TEST-001", 0)
        assert frame is not None
        assert isinstance(frame, bytes)
        assert len(frame) > 100  # Valid JPEG

        # Verify it's a valid JPEG
        arr = np.frombuffer(frame, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert img is not None

    def test_trajectory_points_in_signals(self, tmp_path):
        """_build_signals should emit trajectory_points keyed by track_id."""
        store = AlertRecordingStore(archive_dir=str(tmp_path))
        ts = time.time()
        # Frame 0: track 1 at (10,10), track 2 at (100,100)
        # Frame 1: only track 1 at (20,10)
        # Frame 2: track 1 gone, track 2 at (110,105)
        frames = [
            FrameSnapshot(
                timestamp=ts, frame_jpeg=_make_jpeg(), anomaly_score=0.1,
                simplex_score=None, cusum_evidence={}, yolo_persons=[],
                frame_number=0,
                active_tracks=[
                    {"track_id": 1, "centroid_x": 10.0, "centroid_y": 10.0,
                     "max_score": 0.5, "area_px": 100},
                    {"track_id": 2, "centroid_x": 100.0, "centroid_y": 100.0,
                     "max_score": 0.3, "area_px": 50},
                ],
            ),
            FrameSnapshot(
                timestamp=ts + 1, frame_jpeg=_make_jpeg(), anomaly_score=0.2,
                simplex_score=None, cusum_evidence={}, yolo_persons=[],
                frame_number=1,
                active_tracks=[
                    {"track_id": 1, "centroid_x": 20.0, "centroid_y": 10.0,
                     "max_score": 0.6, "area_px": 110},
                ],
            ),
            FrameSnapshot(
                timestamp=ts + 2, frame_jpeg=_make_jpeg(), anomaly_score=0.3,
                simplex_score=None, cusum_evidence={}, yolo_persons=[],
                frame_number=2,
                active_tracks=[
                    {"track_id": 2, "centroid_x": 110.0, "centroid_y": 105.0,
                     "max_score": 0.4, "area_px": 60},
                ],
            ),
        ]
        recording = SolidifiedRecording(
            alert_id="ALT-TRAJ-001", camera_id="cam_01", severity="medium",
            trigger_timestamp=ts, trigger_frame_index=2, frames=frames, fps=15,
            status=RecordingStatus.COMPLETE,
        )
        store.save(recording)
        signals = store.load_signals("ALT-TRAJ-001")

        traj = signals["trajectory_points"]
        assert set(traj.keys()) == {"1", "2"}
        # Track 1 appears on frames 0, 1 (not 2)
        assert [p["frame_index"] for p in traj["1"]] == [0, 1]
        assert traj["1"][0]["x"] == 10.0 and traj["1"][0]["y"] == 10.0
        assert traj["1"][1]["x"] == 20.0
        # Track 2 appears on frames 0, 2 (not 1)
        assert [p["frame_index"] for p in traj["2"]] == [0, 2]

    def test_aggregate_trajectories_offset(self):
        """_aggregate_trajectories should honour frame_index_offset."""
        ts = time.time()
        frames = [
            FrameSnapshot(
                timestamp=ts + i, frame_jpeg=b"", anomaly_score=0.1,
                simplex_score=None, cusum_evidence={}, yolo_persons=[],
                frame_number=i,
                active_tracks=[{"track_id": 7, "centroid_x": float(i),
                                "centroid_y": 0.0, "max_score": 0.5, "area_px": 1}],
            )
            for i in range(3)
        ]
        out = _aggregate_trajectories(frames, frame_index_offset=10)
        assert [p["frame_index"] for p in out["7"]] == [10, 11, 12]

    def test_append_post_frames_trajectory_continuity(self, tmp_path):
        """trajectory_points should keep frame_index continuous across pre/post."""
        store = AlertRecordingStore(archive_dir=str(tmp_path))
        ts = time.time()
        pre_frames = [
            FrameSnapshot(
                timestamp=ts + i, frame_jpeg=_make_jpeg(), anomaly_score=0.1,
                simplex_score=None, cusum_evidence={}, yolo_persons=[],
                frame_number=i,
                active_tracks=[{"track_id": 1, "centroid_x": float(i),
                                "centroid_y": 0.0, "max_score": 0.5, "area_px": 1}],
            )
            for i in range(3)
        ]
        recording = SolidifiedRecording(
            alert_id="ALT-APPEND-001", camera_id="cam_01", severity="medium",
            trigger_timestamp=ts, trigger_frame_index=2, frames=pre_frames, fps=15,
            status=RecordingStatus.RECORDING,
        )
        store.save(recording)

        post_frames = [
            FrameSnapshot(
                timestamp=ts + 3 + i, frame_jpeg=_make_jpeg(), anomaly_score=0.1,
                simplex_score=None, cusum_evidence={}, yolo_persons=[],
                frame_number=3 + i,
                active_tracks=[{"track_id": 1, "centroid_x": 3.0 + i,
                                "centroid_y": 0.0, "max_score": 0.5, "area_px": 1}],
            )
            for i in range(2)
        ]
        assert store.append_post_frames("ALT-APPEND-001", post_frames)

        signals = store.load_signals("ALT-APPEND-001")
        indices = [p["frame_index"] for p in signals["trajectory_points"]["1"]]
        assert indices == [0, 1, 2, 3, 4]

    def test_has_recording(self, tmp_path):
        store = AlertRecordingStore(archive_dir=str(tmp_path))
        assert not store.has_recording("NONEXISTENT")

        store.save(_make_recording())
        assert store.has_recording("ALT-TEST-001")

    def test_append_post_frames(self, tmp_path):
        store = AlertRecordingStore(archive_dir=str(tmp_path))
        recording = _make_recording(frame_count=5)
        rec_path, _ = store.save(recording)

        ts = time.time()
        post_frames = [
            FrameSnapshot(
                timestamp=ts + i,
                frame_jpeg=_make_jpeg(value=200 + i * 10),
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
        assert metadata["video_file"] == "recording.mp4"

        # Verify pre.mp4 and post.mp4 are cleaned up
        rec_dir = Path(rec_path)
        assert not (rec_dir / "pre.mp4").exists()
        assert not (rec_dir / "post.mp4").exists()
        assert (rec_dir / "recording.mp4").exists()

    def test_get_video_path(self, tmp_path):
        store = AlertRecordingStore(archive_dir=str(tmp_path))
        recording = _make_recording()
        store.save(recording)

        video_path = store.get_video_path("ALT-TEST-001")
        assert video_path is not None
        assert video_path.exists()
        assert video_path.suffix == ".mp4"

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

    def test_cleanup_old(self, tmp_path):
        store = AlertRecordingStore(archive_dir=str(tmp_path))
        recording = _make_recording()
        rec_path, _ = store.save(recording)

        # Force the date directory to look old by renaming
        rec_dir = Path(rec_path)
        date_dir = rec_dir.parent.parent
        old_date_dir = date_dir.parent / "2020-01-01"
        date_dir.rename(old_date_dir)

        archived = store.cleanup_old(max_age_days=1)
        assert archived == 1

        # Verify video is deleted but trigger frame is preserved
        old_rec_dir = old_date_dir / "cam_01" / "ALT-TEST-001"
        assert not (old_rec_dir / "pre.mp4").exists()
        assert not (old_rec_dir / "recording.mp4").exists()
        assert (old_rec_dir / "trigger_frame.jpg").exists()
        assert (old_rec_dir / "signals.json").exists()
