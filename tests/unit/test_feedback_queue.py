"""Tests for FeedbackManager queue logic, uncertain type, and passive feedback (Phase 2)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from argus.alerts.feedback import FeedbackManager
from argus.config.schema import FeedbackConfig
from argus.storage.database import Database
from argus.storage.models import FeedbackStatus, FeedbackType


@pytest.fixture
def db(tmp_path):
    database = Database(database_url=f"sqlite:///{tmp_path / 'test.db'}")
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def snapshot_file(tmp_path):
    """Create a fake snapshot JPEG for testing."""
    alerts_dir = tmp_path / "alerts" / "2026-03-15" / "cam_01"
    alerts_dir.mkdir(parents=True)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    path = alerts_dir / "ALT-001_snapshot.jpg"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def feedback_config(tmp_path):
    return FeedbackConfig(
        auto_baseline_on_fp=True,
        auto_validation_on_confirmed=True,
        validation_dir=tmp_path / "validation",
    )


@pytest.fixture
def mgr(db, tmp_path, feedback_config):
    return FeedbackManager(
        database=db,
        baselines_dir=tmp_path / "baselines",
        alerts_dir=tmp_path / "alerts",
        config=feedback_config,
    )


class TestSubmitFeedback:
    """Test submit_feedback for each feedback type."""

    def test_false_positive_creates_record(self, mgr, db, snapshot_file):
        rec = mgr.submit_feedback(
            alert_id="ALT-001",
            feedback_type=FeedbackType.FALSE_POSITIVE,
            camera_id="cam_01",
            zone_id="default",
            category="lens_glare",
            notes="Glare from rain",
            anomaly_score=0.82,
            snapshot_path=str(snapshot_file),
        )
        assert rec.feedback_id is not None
        assert rec.status == FeedbackStatus.PENDING
        assert rec.feedback_type == FeedbackType.FALSE_POSITIVE
        assert rec.source == "manual"

        # Verify it's in DB
        fetched = db.get_feedback(rec.feedback_id)
        assert fetched is not None
        assert fetched.alert_id == "ALT-001"

    def test_false_positive_copies_to_baseline(self, mgr, tmp_path, snapshot_file):
        mgr.submit_feedback(
            alert_id="ALT-001",
            feedback_type=FeedbackType.FALSE_POSITIVE,
            camera_id="cam_01",
            snapshot_path=str(snapshot_file),
        )
        fp_dir = tmp_path / "baselines" / "cam_01" / "default" / "false_positives"
        fp_files = list(fp_dir.glob("fp_*.jpg"))
        assert len(fp_files) == 1
        # Check metadata sidecar
        meta_files = list(fp_dir.glob("*.meta.json"))
        assert len(meta_files) == 1

    def test_confirmed_copies_to_validation(self, mgr, tmp_path, snapshot_file):
        mgr.submit_feedback(
            alert_id="ALT-002",
            feedback_type=FeedbackType.CONFIRMED,
            camera_id="cam_01",
            snapshot_path=str(snapshot_file),
        )
        val_dir = tmp_path / "validation" / "cam_01" / "confirmed"
        confirmed_files = list(val_dir.glob("confirmed_*.jpg"))
        assert len(confirmed_files) == 1
        meta_files = list(val_dir.glob("*.meta.json"))
        assert len(meta_files) == 1

    def test_uncertain_no_copy(self, mgr, tmp_path, snapshot_file):
        mgr.submit_feedback(
            alert_id="ALT-003",
            feedback_type=FeedbackType.UNCERTAIN,
            camera_id="cam_01",
            snapshot_path=str(snapshot_file),
        )
        # No copy to baseline or validation
        baselines = tmp_path / "baselines"
        validation = tmp_path / "validation"
        assert not list(baselines.rglob("fp_*"))
        assert not list(validation.rglob("confirmed_*"))

    def test_fp_no_copy_when_disabled(self, db, tmp_path, snapshot_file):
        cfg = FeedbackConfig(auto_baseline_on_fp=False)
        mgr = FeedbackManager(
            database=db,
            baselines_dir=tmp_path / "baselines",
            alerts_dir=tmp_path / "alerts",
            config=cfg,
        )
        mgr.submit_feedback(
            alert_id="ALT-004",
            feedback_type=FeedbackType.FALSE_POSITIVE,
            camera_id="cam_01",
            snapshot_path=str(snapshot_file),
        )
        fp_dir = tmp_path / "baselines" / "cam_01" / "default" / "false_positives"
        assert not fp_dir.exists()

    def test_missing_snapshot_handled_gracefully(self, mgr):
        rec = mgr.submit_feedback(
            alert_id="ALT-005",
            feedback_type=FeedbackType.FALSE_POSITIVE,
            camera_id="cam_01",
            snapshot_path="/nonexistent/path.jpg",
        )
        assert rec.feedback_id is not None
        # No crash, record still created

    def test_no_snapshot_path(self, mgr):
        rec = mgr.submit_feedback(
            alert_id="ALT-006",
            feedback_type=FeedbackType.CONFIRMED,
            camera_id="cam_01",
        )
        assert rec.feedback_id is not None


class TestPassiveFeedback:
    """Test system-generated passive feedback."""

    def test_drift_feedback(self, mgr, db):
        rec = mgr.submit_passive_feedback(
            camera_id="cam_01",
            source="drift",
            notes="KS=0.15 p=0.003",
            model_version_id="cam_01-patchcore-v001",
        )
        assert rec.alert_id is None
        assert rec.source == "drift"
        assert rec.feedback_type == FeedbackType.CONFIRMED
        assert rec.submitted_by == "system"
        assert rec.status == FeedbackStatus.PENDING

        fetched = db.get_feedback(rec.feedback_id)
        assert fetched.model_version_at_time == "cam_01-patchcore-v001"

    def test_health_feedback(self, mgr, db):
        rec = mgr.submit_passive_feedback(
            camera_id="cam_02",
            source="health",
            notes="Camera disconnected. error=timeout",
        )
        assert rec.source == "health"
        assert rec.camera_id == "cam_02"


class TestQueueOperations:
    """Test queue query and batch processing."""

    def test_get_pending_for_training(self, mgr, db):
        mgr.submit_feedback(
            alert_id="A1", feedback_type=FeedbackType.FALSE_POSITIVE, camera_id="cam_01",
        )
        mgr.submit_feedback(
            alert_id="A2", feedback_type=FeedbackType.CONFIRMED, camera_id="cam_01",
        )
        mgr.submit_feedback(
            alert_id="A3", feedback_type=FeedbackType.UNCERTAIN, camera_id="cam_02",
        )

        pending = mgr.get_pending_for_training(camera_id="cam_01")
        assert len(pending) == 2

        all_pending = mgr.get_pending_for_training()
        assert len(all_pending) == 3

    def test_mark_batch_processed(self, mgr, db):
        rec1 = mgr.submit_feedback(
            alert_id="A1", feedback_type=FeedbackType.FALSE_POSITIVE, camera_id="cam_01",
        )
        rec2 = mgr.submit_feedback(
            alert_id="A2", feedback_type=FeedbackType.CONFIRMED, camera_id="cam_01",
        )

        count = mgr.mark_batch_processed(
            [rec1.feedback_id, rec2.feedback_id],
            model_version_id="cam_01-patchcore-v013",
        )
        assert count == 2

        # Verify records updated
        f1 = db.get_feedback(rec1.feedback_id)
        assert f1.status == FeedbackStatus.PROCESSED
        assert f1.trained_into == "cam_01-patchcore-v013"
        assert f1.processed_at is not None

        # No more pending
        assert len(mgr.get_pending_for_training(camera_id="cam_01")) == 0


class TestCollectConfirmed:
    """Test collect_confirmed_for_validation."""

    def test_collects_confirmed_snapshots(self, mgr, db, tmp_path, snapshot_file):
        mgr.submit_feedback(
            alert_id="A1",
            feedback_type=FeedbackType.CONFIRMED,
            camera_id="cam_01",
            snapshot_path=str(snapshot_file),
        )

        val_dir = tmp_path / "val_output"
        count = mgr.collect_confirmed_for_validation("cam_01", validation_dir=val_dir)
        assert count == 1

        confirmed_dir = val_dir / "cam_01" / "confirmed"
        assert len(list(confirmed_dir.glob("confirmed_*.jpg"))) == 1
        assert len(list(confirmed_dir.glob("*.meta.json"))) == 1

    def test_skips_fp_feedback(self, mgr, db, snapshot_file, tmp_path):
        """Only confirmed type should be collected."""
        mgr.submit_feedback(
            alert_id="A1",
            feedback_type=FeedbackType.FALSE_POSITIVE,
            camera_id="cam_01",
            snapshot_path=str(snapshot_file),
        )
        val_dir = tmp_path / "val_output"
        count = mgr.collect_confirmed_for_validation("cam_01", validation_dir=val_dir)
        assert count == 0


class TestFeedbackStatsWithQueue:
    """Test that get_feedback_stats includes queue data."""

    def test_stats_include_queue_summary(self, mgr, db):
        now = datetime.now(tz=timezone.utc)
        db.save_alert("ALT-001", now, "cam_01", "z1", "low", 0.72)
        db.mark_false_positive("ALT-001")

        mgr.submit_feedback(
            alert_id="ALT-001",
            feedback_type=FeedbackType.FALSE_POSITIVE,
            camera_id="cam_01",
        )

        stats = mgr.get_feedback_stats()
        assert stats["total_alerts"] == 1
        assert stats["false_positives"] == 1
        assert "feedback_queue" in stats
        assert stats["feedback_queue"]["total"] == 1


class TestHealthMonitorPassiveFeedback:
    """Test HealthMonitor integration with FeedbackManager."""

    def test_disconnect_generates_feedback(self, mgr, db):
        from argus.core.health import HealthMonitor

        monitor = HealthMonitor(feedback_manager=mgr)
        # First set camera as connected
        monitor.update_camera("cam_01", connected=True, frames_captured=100)
        # Then disconnect
        monitor.update_camera("cam_01", connected=False, error="RTSP timeout")

        pending = db.get_pending_feedback(camera_id="cam_01")
        assert len(pending) == 1
        assert pending[0].source == "health"
        assert "disconnected" in pending[0].notes.lower()

    def test_reconnect_allows_future_feedback(self, mgr, db):
        from argus.core.health import HealthMonitor

        monitor = HealthMonitor(feedback_manager=mgr)
        monitor.update_camera("cam_01", connected=True)
        monitor.update_camera("cam_01", connected=False, error="timeout")
        # Reconnect
        monitor.update_camera("cam_01", connected=True)
        # Disconnect again — should generate second feedback
        monitor.update_camera("cam_01", connected=False, error="timeout2")

        pending = db.get_pending_feedback(camera_id="cam_01")
        assert len(pending) == 2

    def test_no_duplicate_feedback_while_disconnected(self, mgr, db):
        from argus.core.health import HealthMonitor

        monitor = HealthMonitor(feedback_manager=mgr)
        monitor.update_camera("cam_01", connected=True)
        monitor.update_camera("cam_01", connected=False)
        monitor.update_camera("cam_01", connected=False)  # still disconnected

        pending = db.get_pending_feedback(camera_id="cam_01")
        assert len(pending) == 1  # only one feedback
