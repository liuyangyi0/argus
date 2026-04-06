"""Tests for InferenceBuffer and InferenceRecord DB operations (Phase 4)."""

import time

import pytest

from argus.storage.database import Database
from argus.storage.inference_buffer import InferenceBuffer
from argus.storage.models import InferenceRecord


@pytest.fixture
def db(tmp_path):
    database = Database(database_url=f"sqlite:///{tmp_path / 'test.db'}")
    database.initialize()
    yield database
    database.close()


def _make_record(
    camera_id: str = "cam_01",
    frame_number: int = 1,
    score: float = 0.42,
    **kwargs,
) -> InferenceRecord:
    defaults = {
        "camera_id": camera_id,
        "zone_id": "default",
        "frame_number": frame_number,
        "timestamp": time.time(),
        "anomaly_score": score,
        "was_alert": False,
    }
    defaults.update(kwargs)
    return InferenceRecord(**defaults)


class TestInferenceRecordDB:
    """Test direct DB operations on InferenceRecord."""

    def test_save_batch(self, db):
        records = [_make_record(frame_number=i) for i in range(5)]
        count = db.save_inference_batch(records)
        assert count == 5

    def test_empty_batch(self, db):
        assert db.save_inference_batch([]) == 0

    def test_get_stats(self, db):
        records = [
            _make_record(score=0.3, model_version_id="v1"),
            _make_record(score=0.5, model_version_id="v1"),
            _make_record(score=0.8, model_version_id="v1"),
        ]
        db.save_inference_batch(records)

        stats = db.get_inference_stats("cam_01", model_version_id="v1")
        assert stats["total_frames"] == 3
        assert stats["avg_score"] == pytest.approx(0.5333, abs=0.01)
        assert stats["max_score"] == pytest.approx(0.8)

    def test_get_stats_empty(self, db):
        stats = db.get_inference_stats("cam_99")
        assert stats["total_frames"] == 0

    def test_delete_old_records(self, db):
        # Insert records with old timestamp (90 days ago)
        old_ts = time.time() - 90 * 86400
        records = [_make_record(timestamp=old_ts, frame_number=i) for i in range(3)]
        db.save_inference_batch(records)

        # Insert recent records
        recent = [_make_record(frame_number=i + 100) for i in range(2)]
        db.save_inference_batch(recent)

        deleted = db.delete_old_inference_records(days=30)
        assert deleted == 3

        stats = db.get_inference_stats("cam_01")
        assert stats["total_frames"] == 2


class TestInferenceBuffer:
    """Test the write-behind buffer."""

    def test_append_and_manual_flush(self, db):
        buf = InferenceBuffer(db, flush_seconds=3600, max_size=100)
        for i in range(5):
            buf.append(_make_record(frame_number=i))

        assert buf.pending_count == 5
        buf._flush()
        assert buf.pending_count == 0
        assert buf.total_flushed == 5

        stats = db.get_inference_stats("cam_01")
        assert stats["total_frames"] == 5

    def test_overflow_drops_oldest(self, db):
        buf = InferenceBuffer(db, max_size=3)
        for i in range(5):
            buf.append(_make_record(frame_number=i, score=i * 0.1))

        assert buf.pending_count == 3  # only last 3 kept
        buf._flush()

        stats = db.get_inference_stats("cam_01")
        assert stats["total_frames"] == 3

    def test_start_stop_flushes(self, db):
        buf = InferenceBuffer(db, flush_seconds=0.1, max_size=100)
        buf.start()

        for i in range(3):
            buf.append(_make_record(frame_number=i))

        # Wait for auto-flush
        time.sleep(0.3)
        buf.stop()

        stats = db.get_inference_stats("cam_01")
        assert stats["total_frames"] == 3

    def test_stop_drains_remaining(self, db):
        buf = InferenceBuffer(db, flush_seconds=3600, max_size=100)
        buf.start()

        for i in range(3):
            buf.append(_make_record(frame_number=i))

        # Stop immediately — should drain buffer
        buf.stop()

        stats = db.get_inference_stats("cam_01")
        assert stats["total_frames"] == 3
