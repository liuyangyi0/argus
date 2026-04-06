"""Tests for FeedbackRecord ORM and database CRUD operations (Phase 1)."""

import uuid
from datetime import datetime, timezone

import pytest

from argus.storage.database import Database
from argus.storage.models import FeedbackRecord, FeedbackStatus, FeedbackType


@pytest.fixture
def db(tmp_path):
    database = Database(database_url=f"sqlite:///{tmp_path / 'test.db'}")
    database.initialize()
    yield database
    database.close()


def _make_feedback(
    camera_id: str = "cam_01",
    feedback_type: str = FeedbackType.FALSE_POSITIVE,
    **kwargs,
) -> FeedbackRecord:
    """Helper to create a FeedbackRecord with sensible defaults."""
    defaults = {
        "feedback_id": str(uuid.uuid4()),
        "camera_id": camera_id,
        "zone_id": "default",
        "feedback_type": feedback_type,
        "source": "manual",
        "submitted_by": "operator",
        "status": FeedbackStatus.PENDING,
    }
    defaults.update(kwargs)
    return FeedbackRecord(**defaults)


class TestFeedbackRecordCRUD:
    """Test basic save/get operations on FeedbackRecord."""

    def test_save_and_retrieve(self, db):
        rec = _make_feedback(
            alert_id="ALT-001",
            category="lens_glare",
            anomaly_score=0.82,
            model_version_at_time="cam_01-patchcore-20260301-0001",
            notes="Glare on lens after rain",
        )
        saved = db.save_feedback(rec)
        assert saved.id is not None

        fetched = db.get_feedback(rec.feedback_id)
        assert fetched is not None
        assert fetched.feedback_id == rec.feedback_id
        assert fetched.alert_id == "ALT-001"
        assert fetched.category == "lens_glare"
        assert fetched.anomaly_score == pytest.approx(0.82)
        assert fetched.model_version_at_time == "cam_01-patchcore-20260301-0001"
        assert fetched.status == FeedbackStatus.PENDING

    def test_get_nonexistent_returns_none(self, db):
        assert db.get_feedback("nonexistent-id") is None

    def test_passive_feedback_no_alert(self, db):
        """Passive feedback (drift/health) has no alert_id."""
        rec = _make_feedback(
            feedback_type=FeedbackType.CONFIRMED,
            source="drift",
            alert_id=None,
            notes="KS=0.15 p=0.003",
        )
        db.save_feedback(rec)
        fetched = db.get_feedback(rec.feedback_id)
        assert fetched.alert_id is None
        assert fetched.source == "drift"

    def test_to_dict(self, db):
        rec = _make_feedback(alert_id="ALT-X")
        db.save_feedback(rec)
        fetched = db.get_feedback(rec.feedback_id)
        d = fetched.to_dict()
        assert d["feedback_id"] == rec.feedback_id
        assert d["alert_id"] == "ALT-X"
        assert d["status"] == "pending"
        assert "created_at" in d


class TestPendingFeedback:
    """Test get_pending_feedback filtering."""

    def test_returns_only_pending(self, db):
        db.save_feedback(_make_feedback(status=FeedbackStatus.PENDING))
        db.save_feedback(_make_feedback(status=FeedbackStatus.PROCESSED))
        db.save_feedback(_make_feedback(status=FeedbackStatus.SKIPPED))

        pending = db.get_pending_feedback()
        assert len(pending) == 1
        assert pending[0].status == FeedbackStatus.PENDING

    def test_filter_by_camera(self, db):
        db.save_feedback(_make_feedback(camera_id="cam_01"))
        db.save_feedback(_make_feedback(camera_id="cam_02"))
        db.save_feedback(_make_feedback(camera_id="cam_01"))

        pending = db.get_pending_feedback(camera_id="cam_01")
        assert len(pending) == 2

    def test_filter_by_type(self, db):
        db.save_feedback(_make_feedback(feedback_type=FeedbackType.FALSE_POSITIVE))
        db.save_feedback(_make_feedback(feedback_type=FeedbackType.CONFIRMED))
        db.save_feedback(_make_feedback(feedback_type=FeedbackType.UNCERTAIN))

        fp_only = db.get_pending_feedback(feedback_type=FeedbackType.FALSE_POSITIVE)
        assert len(fp_only) == 1
        assert fp_only[0].feedback_type == FeedbackType.FALSE_POSITIVE

    def test_ordered_by_created_at_asc(self, db):
        """Pending feedback should be ordered oldest-first for FIFO processing."""
        from datetime import timedelta

        t0 = datetime(2026, 3, 1, tzinfo=timezone.utc)
        db.save_feedback(_make_feedback(
            feedback_id="old", camera_id="cam_01",
        ))
        db.save_feedback(_make_feedback(
            feedback_id="new", camera_id="cam_01",
        ))

        pending = db.get_pending_feedback()
        assert len(pending) == 2
        # The one saved first should come first
        assert pending[0].feedback_id == "old"


class TestMarkProcessed:
    """Test mark_feedback_processed batch operation."""

    def test_mark_batch(self, db):
        ids = []
        for _ in range(3):
            rec = _make_feedback()
            db.save_feedback(rec)
            ids.append(rec.feedback_id)

        updated = db.mark_feedback_processed(ids, trained_into="cam_01-patchcore-20260401-0001")
        assert updated == 3

        for fid in ids:
            rec = db.get_feedback(fid)
            assert rec.status == FeedbackStatus.PROCESSED
            assert rec.trained_into == "cam_01-patchcore-20260401-0001"
            assert rec.processed_at is not None

    def test_mark_only_pending(self, db):
        """Already processed entries should not be re-processed."""
        rec = _make_feedback(status=FeedbackStatus.PROCESSED)
        db.save_feedback(rec)

        updated = db.mark_feedback_processed([rec.feedback_id], trained_into="v999")
        assert updated == 0

        fetched = db.get_feedback(rec.feedback_id)
        assert fetched.trained_into is None  # unchanged

    def test_empty_ids_returns_zero(self, db):
        assert db.mark_feedback_processed([], trained_into="v001") == 0

    def test_nonexistent_ids_ignored(self, db):
        updated = db.mark_feedback_processed(["no-such-id"], trained_into="v001")
        assert updated == 0


class TestSkipFeedback:
    def test_skip_pending(self, db):
        rec = _make_feedback()
        db.save_feedback(rec)

        updated = db.skip_feedback([rec.feedback_id], reason="low quality")
        assert updated == 1

        fetched = db.get_feedback(rec.feedback_id)
        assert fetched.status == FeedbackStatus.SKIPPED
        assert "low quality" in fetched.notes

    def test_skip_already_processed(self, db):
        rec = _make_feedback(status=FeedbackStatus.PROCESSED)
        db.save_feedback(rec)
        assert db.skip_feedback([rec.feedback_id]) == 0


class TestFeedbackSummary:
    def test_empty_summary(self, db):
        summary = db.get_feedback_summary()
        assert summary["total"] == 0
        assert summary["by_type"] == {}
        assert summary["by_status"] == {}

    def test_summary_counts(self, db):
        db.save_feedback(_make_feedback(feedback_type=FeedbackType.FALSE_POSITIVE))
        db.save_feedback(_make_feedback(feedback_type=FeedbackType.FALSE_POSITIVE))
        db.save_feedback(_make_feedback(feedback_type=FeedbackType.CONFIRMED))
        db.save_feedback(_make_feedback(
            feedback_type=FeedbackType.UNCERTAIN,
            status=FeedbackStatus.SKIPPED,
        ))

        summary = db.get_feedback_summary()
        assert summary["total"] == 4
        assert summary["by_type"]["false_positive"] == 2
        assert summary["by_type"]["confirmed"] == 1
        assert summary["by_type"]["uncertain"] == 1
        assert summary["by_status"]["pending"] == 3
        assert summary["by_status"]["skipped"] == 1

    def test_summary_filter_by_camera(self, db):
        db.save_feedback(_make_feedback(camera_id="cam_01"))
        db.save_feedback(_make_feedback(camera_id="cam_02"))

        summary = db.get_feedback_summary(camera_id="cam_01")
        assert summary["total"] == 1


class TestListFeedback:
    def test_list_with_filters(self, db):
        db.save_feedback(_make_feedback(
            camera_id="cam_01",
            feedback_type=FeedbackType.FALSE_POSITIVE,
        ))
        db.save_feedback(_make_feedback(
            camera_id="cam_01",
            feedback_type=FeedbackType.CONFIRMED,
        ))
        db.save_feedback(_make_feedback(
            camera_id="cam_02",
            feedback_type=FeedbackType.FALSE_POSITIVE,
        ))

        all_entries = db.list_feedback()
        assert len(all_entries) == 3

        cam1 = db.list_feedback(camera_id="cam_01")
        assert len(cam1) == 2

        fp_only = db.list_feedback(feedback_type=FeedbackType.FALSE_POSITIVE)
        assert len(fp_only) == 2

        cam1_fp = db.list_feedback(
            camera_id="cam_01", feedback_type=FeedbackType.FALSE_POSITIVE
        )
        assert len(cam1_fp) == 1

    def test_list_pagination(self, db):
        for _ in range(5):
            db.save_feedback(_make_feedback())

        page1 = db.list_feedback(limit=2, offset=0)
        page2 = db.list_feedback(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].feedback_id != page2[0].feedback_id
