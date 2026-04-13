"""Tests for the continuous recording and retention modules."""

from __future__ import annotations

import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from argus.storage.continuous_recorder import ContinuousRecorder, ContinuousRecordingManager
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


class TestContinuousRecordingConfig:
    def test_archive_enabled_without_path_raises(self):
        """archive_enabled=True with archive_path=None should fail validation."""
        from argus.config.schema import ContinuousRecordingConfig

        with pytest.raises(Exception):  # Pydantic ValidationError
            ContinuousRecordingConfig(archive_enabled=True, archive_path=None)
