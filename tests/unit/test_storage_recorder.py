"""Tests for the continuous recording and retention modules."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pytest

from argus.storage.continuous_recorder import ContinuousRecorder
from argus.storage.database import Database
from argus.storage.models import AlertRecordingRecord
from argus.storage.retention import RetentionManager


class TestContinuousRecorder:
    def test_start_stop_lifecycle(self, tmp_path):
        """Recorder should start and stop cleanly."""
        rec = ContinuousRecorder(
            camera_id="test-cam",
            output_dir=tmp_path,
            encoding_fps=5,
        )
        rec.start()
        assert rec.is_recording is True

        rec.stop()
        assert rec.is_recording is False

    def test_push_frame_nonblocking(self, tmp_path):
        """push_frame should not block even if queue is full."""
        rec = ContinuousRecorder(
            camera_id="test-cam",
            output_dir=tmp_path,
            encoding_fps=5,
        )
        # Don't start the recording thread — queue will fill up
        rec._recording = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Push more frames than queue capacity (60) — should not block
        for _ in range(100):
            rec.push_frame(frame, time.time())

        rec._recording = False

    def test_push_frame_when_not_recording(self, tmp_path):
        """push_frame should silently return if not recording."""
        rec = ContinuousRecorder(
            camera_id="test-cam",
            output_dir=tmp_path,
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rec.push_frame(frame, time.time())  # Should not raise


class TestRetentionManager:
    def test_parse_date_dir_valid(self):
        """Valid YYYY-MM-DD directory names should parse correctly."""
        result = RetentionManager._parse_date_dir("2026-04-13")
        assert result is not None
        assert result.year == 2026
        assert result.month == 4
        assert result.day == 13

    def test_parse_date_dir_invalid(self):
        """Invalid directory names should return None."""
        assert RetentionManager._parse_date_dir("not-a-date") is None
        assert RetentionManager._parse_date_dir("camera-001") is None
        assert RetentionManager._parse_date_dir("") is None

    def test_delete_old_date_dirs(self, tmp_path):
        """Should delete directories older than cutoff, keep newer ones."""
        mgr = RetentionManager(
            continuous_recording_dir=tmp_path,
            alert_recording_dir=tmp_path / "alerts",
        )

        # Create old and new directories
        old_dir = tmp_path / "2020-01-01"
        old_dir.mkdir()
        (old_dir / "test.mp4").touch()

        new_dir = tmp_path / "2099-12-31"
        new_dir.mkdir()
        (new_dir / "test.mp4").touch()

        cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
        deleted = mgr._delete_old_date_dirs(tmp_path, cutoff)

        assert deleted == 1
        assert not old_dir.exists()
        assert new_dir.exists()

    def test_delete_handles_missing_root(self, tmp_path):
        """Should handle non-existent root directory gracefully."""
        mgr = RetentionManager(
            continuous_recording_dir=tmp_path / "nonexistent",
            alert_recording_dir=tmp_path / "alerts",
        )
        cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
        deleted = mgr._delete_old_date_dirs(tmp_path / "nonexistent", cutoff)
        assert deleted == 0

    def test_alert_retention_removes_recording_metadata(self, tmp_path):
        """Alert recording directory retention should remove DB metadata too."""
        db = Database(database_url=f"sqlite:///{tmp_path / 'retention.db'}")
        db.initialize()
        try:
            recordings_root = tmp_path / "recordings"
            rec_dir = recordings_root / "2020-01-01" / "cam_01" / "ALT-RET"
            rec_dir.mkdir(parents=True)
            (rec_dir / "metadata.json").write_text(
                '{"alert_id": "ALT-RET", "camera_id": "cam_01"}',
                encoding="utf-8",
            )
            (rec_dir / "recording.mp4").write_bytes(b"mp4")
            db.save_alert_recording(
                AlertRecordingRecord(
                    alert_id="ALT-RET",
                    camera_id="cam_01",
                    severity="high",
                    recording_path=str(rec_dir),
                    start_timestamp=1.0,
                    end_timestamp=2.0,
                    trigger_timestamp=1.5,
                    frame_count=10,
                    fps=5,
                    file_size_bytes=3,
                )
            )

            mgr = RetentionManager(
                continuous_recording_dir=tmp_path / "continuous",
                alert_recording_dir=recordings_root,
                alert_retention_days=1,
                database=db,
            )

            deleted = mgr._cleanup_alert_evidence()

            assert deleted == 1
            assert not rec_dir.exists()
            assert db.get_alert_recording("ALT-RET") is None
        finally:
            db.close()


class TestContinuousRecordingConfig:
    def test_archive_enabled_without_path_raises(self):
        """archive_enabled=True with archive_path=None should fail validation."""
        from argus.config.schema import ContinuousRecordingConfig

        with pytest.raises(Exception):  # Pydantic ValidationError
            ContinuousRecordingConfig(archive_enabled=True, archive_path=None)
