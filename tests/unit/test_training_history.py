"""Tests for TrainingRecord model and CaptureStats."""

from datetime import datetime, timedelta, timezone

import pytest

from argus.dashboard.routes.baseline import CaptureStats
from argus.storage.database import Database
from argus.storage.models import TrainingRecord


@pytest.fixture
def db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"
    database = Database(database_url=f"sqlite:///{db_path}")
    database.initialize()
    yield database
    database.close()


def _make_training_record(camera_id="cam_01", **overrides):
    """Helper to build a TrainingRecord instance."""
    defaults = {
        "camera_id": camera_id,
        "zone_id": "default",
        "model_type": "patchcore",
        "baseline_version": "v1",
        "baseline_count": 100,
        "trained_at": datetime.now(tz=timezone.utc),
        "status": "complete",
        "train_count": 80,
        "val_count": 20,
        "duration_seconds": 120.5,
        "quality_grade": "A",
        "val_score_mean": 0.15,
        "val_score_std": 0.05,
        "val_score_max": 0.42,
        "val_score_p95": 0.30,
        "threshold_recommended": 0.32,
        "model_path": "/data/models/cam_01/model.pt",
        "export_path": "/data/exports/cam_01/model.xml",
    }
    defaults.update(overrides)
    return TrainingRecord(**defaults)


class TestTrainingRecord:
    def test_create_and_query(self, db):
        """TrainingRecord can be created and queried."""
        db.save_training_record(_make_training_record())

        history = db.get_training_history()
        assert len(history) == 1
        assert history[0].camera_id == "cam_01"

    def test_save_stores_all_fields(self, db):
        """save_training_record stores all fields correctly."""
        record = _make_training_record(
            quality_grade="B",
            val_score_mean=0.22,
            val_score_std=0.08,
            val_score_max=0.55,
            val_score_p95=0.45,
            threshold_recommended=0.48,
            model_path="/models/full.pt",
            export_path="/exports/full.xml",
            error=None,
        )
        saved = db.save_training_record(record)

        fetched = db.get_training_record(saved.id)
        assert fetched is not None
        assert fetched.quality_grade == "B"
        assert fetched.val_score_mean == pytest.approx(0.22)
        assert fetched.val_score_std == pytest.approx(0.08)
        assert fetched.val_score_max == pytest.approx(0.55)
        assert fetched.val_score_p95 == pytest.approx(0.45)
        assert fetched.threshold_recommended == pytest.approx(0.48)
        assert fetched.model_path == "/models/full.pt"
        assert fetched.export_path == "/exports/full.xml"
        assert fetched.train_count == 80
        assert fetched.val_count == 20
        assert fetched.duration_seconds == pytest.approx(120.5)

    def test_get_training_history_descending_order(self, db):
        """get_training_history returns records in descending order by trained_at."""
        now = datetime.now(tz=timezone.utc)
        r1 = db.save_training_record(_make_training_record(
            trained_at=now - timedelta(hours=2),
        ))
        r2 = db.save_training_record(_make_training_record(
            trained_at=now,
        ))
        r3 = db.save_training_record(_make_training_record(
            trained_at=now - timedelta(hours=1),
        ))

        history = db.get_training_history()
        assert len(history) == 3
        assert history[0].id == r2.id
        assert history[1].id == r3.id
        assert history[2].id == r1.id

    def test_get_training_history_filters_by_camera(self, db):
        """get_training_history filters by camera_id."""
        now = datetime.now(tz=timezone.utc)
        db.save_training_record(_make_training_record(
            camera_id="cam_01", trained_at=now,
        ))
        db.save_training_record(_make_training_record(
            camera_id="cam_02", trained_at=now,
        ))
        db.save_training_record(_make_training_record(
            camera_id="cam_01",
            trained_at=now - timedelta(hours=1),
        ))

        cam1 = db.get_training_history(camera_id="cam_01")
        assert len(cam1) == 2
        assert all(r.camera_id == "cam_01" for r in cam1)

        cam2 = db.get_training_history(camera_id="cam_02")
        assert len(cam2) == 1
        assert cam2[0].camera_id == "cam_02"

    def test_get_training_history_respects_limit(self, db):
        """get_training_history respects the limit parameter."""
        now = datetime.now(tz=timezone.utc)
        for i in range(5):
            db.save_training_record(_make_training_record(
                trained_at=now - timedelta(hours=i),
            ))

        history = db.get_training_history(limit=3)
        assert len(history) == 3

    def test_get_training_record_not_found(self, db):
        """get_training_record returns None for nonexistent ID."""
        assert db.get_training_record(99999) is None

    def test_training_record_with_failed_status(self, db):
        """TrainingRecord can store failed training with error message."""
        saved = db.save_training_record(_make_training_record(
            status="failed",
            error="Out of memory",
            quality_grade=None,
            model_path=None,
        ))

        record = db.get_training_record(saved.id)
        assert record is not None
        assert record.status == "failed"
        assert record.error == "Out of memory"
        assert record.quality_grade is None


class TestCaptureStats:
    def test_retention_rate_calculated(self):
        """CaptureStats calculates retention_rate correctly."""
        stats = CaptureStats(
            total_frames=100,
            captured_frames=85,
            filtered_frames=15,
        )
        assert stats.retention_rate == pytest.approx(0.85)

    def test_retention_rate_zero_total(self):
        """CaptureStats handles zero total frames without division error."""
        stats = CaptureStats(
            total_frames=0,
            captured_frames=0,
            filtered_frames=0,
        )
        assert stats.retention_rate == 0.0

    def test_retention_rate_perfect(self):
        """CaptureStats with all frames captured has 1.0 retention."""
        stats = CaptureStats(
            total_frames=50,
            captured_frames=50,
            filtered_frames=0,
        )
        assert stats.retention_rate == pytest.approx(1.0)

    def test_filter_reasons_stored(self):
        """CaptureStats stores filter reasons correctly."""
        stats = CaptureStats(
            total_frames=100,
            captured_frames=90,
            filtered_frames=10,
            filter_reasons={"blur": 5, "exposure": 3, "no_frame": 2},
        )
        assert stats.filter_reasons["blur"] == 5
        assert stats.filter_reasons["exposure"] == 3
        assert sum(stats.filter_reasons.values()) == 10

    def test_brightness_range(self):
        """CaptureStats stores brightness range."""
        stats = CaptureStats(
            total_frames=100,
            captured_frames=100,
            filtered_frames=0,
            brightness_range=(45.2, 180.7),
        )
        assert stats.brightness_range == (45.2, 180.7)
