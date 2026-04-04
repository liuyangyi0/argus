"""Tests for false positive feedback loop."""

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from argus.alerts.feedback import FeedbackManager
from argus.storage.database import Database


@pytest.fixture
def db(tmp_path):
    database = Database(database_url=f"sqlite:///{tmp_path / 'test.db'}")
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def feedback(db, tmp_path):
    return FeedbackManager(
        database=db,
        baselines_dir=tmp_path / "baselines",
        alerts_dir=tmp_path / "alerts",
    )


class TestFeedbackManager:
    def test_feedback_stats_empty(self, feedback):
        """Empty database should return zero stats."""
        stats = feedback.get_feedback_stats()
        assert stats["total_alerts"] == 0
        assert stats["false_positives"] == 0
        assert stats["false_positive_rate"] == 0

    def test_feedback_stats_with_data(self, feedback, db):
        """Should calculate correct FP rate."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-001", now, "cam_01", "z1", "low", 0.72)
        db.save_alert("ALT-002", now, "cam_01", "z1", "medium", 0.88)
        db.save_alert("ALT-003", now, "cam_01", "z1", "high", 0.96)

        # Mark one as FP, one as acknowledged
        db.mark_false_positive("ALT-001")
        db.acknowledge_alert("ALT-002", "operator")

        stats = feedback.get_feedback_stats()
        assert stats["total_alerts"] == 3
        assert stats["false_positives"] == 1
        assert stats["acknowledged"] == 1
        assert stats["false_positive_rate"] == pytest.approx(1 / 3, abs=0.01)

    def test_export_false_positives(self, feedback, db, tmp_path):
        """Should export FP snapshots to baselines dir."""
        now = datetime.now(tz=timezone.utc)

        # Create a fake snapshot file
        alerts_dir = tmp_path / "alerts" / "2026-03-22" / "cam_01"
        alerts_dir.mkdir(parents=True)
        snapshot = alerts_dir / "ALT-001_snapshot.jpg"
        cv2.imwrite(str(snapshot), np.zeros((50, 50, 3), dtype=np.uint8))

        # Save alert with snapshot path and mark as FP
        db.save_alert("ALT-001", now, "cam_01", "z1", "low", 0.72, snapshot_path=str(snapshot))
        db.mark_false_positive("ALT-001")

        exported = feedback.export_false_positives("cam_01")
        assert exported == 1

        # Check the exported file exists
        fp_dir = tmp_path / "baselines" / "cam_01" / "default" / "false_positives"
        assert fp_dir.is_dir()
        assert len(list(fp_dir.glob("*.jpg"))) == 1

    def test_export_no_false_positives(self, feedback, db):
        """Should return 0 when no FPs exist."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-001", now, "cam_01", "z1", "low", 0.72)

        exported = feedback.export_false_positives("cam_01")
        assert exported == 0

    def test_filter_stats_by_camera(self, feedback, db):
        """Should filter stats by camera."""
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-001", now, "cam_01", "z1", "low", 0.72)
        db.save_alert("ALT-002", now, "cam_02", "z1", "high", 0.96)

        stats = feedback.get_feedback_stats(camera_id="cam_01")
        assert stats["total_alerts"] == 1
