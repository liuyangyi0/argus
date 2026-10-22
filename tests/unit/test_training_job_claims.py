"""Adversarial training claims and stale-worker boundaries (no GPU needed)."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError

from argus.anomaly.job_executor import TrainingJobExecutor
from argus.storage.database import Database
from argus.storage.models import BackboneRecord, TrainingJobRecord


@pytest.fixture
def database(tmp_path):
    db = Database(database_url=f"sqlite:///{(tmp_path / 'claims.db').as_posix()}")
    db.initialize()
    db.save_training_job(TrainingJobRecord(
        job_id="claim-job", job_type="anomaly_head", camera_id="camera-01",
        zone_id="default", trigger_type="manual", status="pending_confirmation",
    ))
    return db


def test_status_compare_and_swap_is_one_conditional_update(database):
    statements = []

    def capture(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    engine = database._engine
    event.listen(engine, "before_cursor_execute", capture)
    try:
        assert database.update_training_job(
            "claim-job", expected_status="pending_confirmation",
            status="queued", confirmed_by="real-operator",
        )
    finally:
        event.remove(engine, "before_cursor_execute", capture)

    assert len(statements) == 1
    assert statements[0].lstrip().upper().startswith("UPDATE ")
    assert "status" in statements[0].split("WHERE", 1)[1]
    assert not database.update_training_job(
        "claim-job", expected_status="pending_confirmation", status="rejected",
    )
    assert database.get_training_job("claim-job").confirmed_by == "real-operator"


def test_concurrent_confirm_and_reject_have_only_one_winner(database):
    barrier = threading.Barrier(2)

    def transition(status):
        barrier.wait(timeout=5)
        return database.update_training_job(
            "claim-job", expected_status="pending_confirmation", status=status,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(transition, status) for status in ("queued", "rejected")]
        assert sorted(future.result(timeout=10) for future in futures) == [False, True]


class ClaimDatabase:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "queued"
        self.updates = []

    def update_training_job(self, _job_id, *, expected_status=None, **values):
        with self.lock:
            if expected_status is not None and self.status != expected_status:
                return False
            self.updates.append(values)
            self.status = values.get("status", self.status)
            return True


def test_two_executor_readers_only_train_one_job():
    db = ClaimDatabase()
    entered, finish = threading.Event(), threading.Event()
    calls = []
    job = SimpleNamespace(job_id="same-job", job_type="anomaly_head", camera_id="camera-01")

    def train(_job):
        calls.append(_job.job_id)
        entered.set()
        assert finish.wait(timeout=5)

    first, second = TrainingJobExecutor.__new__(TrainingJobExecutor), TrainingJobExecutor.__new__(TrainingJobExecutor)
    for executor in (first, second):
        executor._db = db
        executor._execute_head = train
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(first._execute_job, job)
        try:
            assert entered.wait(timeout=5)
            assert second._execute_job(job) is False
            assert calls == ["same-job"]
        finally:
            finish.set()
        assert running.result(timeout=5) is True


def test_late_failure_does_not_overwrite_an_already_terminal_job(monkeypatch):
    db = ClaimDatabase()
    executor = TrainingJobExecutor.__new__(TrainingJobExecutor)
    executor._db = db

    def fail_after_state_changed(_job):
        db.status = "complete"
        raise RuntimeError("late failure")

    executor._execute_head = fail_after_state_changed
    channel = Mock()
    monkeypatch.setattr("argus.anomaly.job_executor.get_error_channel", lambda: channel)
    executor._execute_job(SimpleNamespace(job_id="same-job", job_type="anomaly_head", camera_id="camera-01"))
    assert db.status == "complete"
    assert not any(item.get("status") == "failed" for item in db.updates)
    channel.emit.assert_not_called()


def test_unknown_claim_commit_never_starts_or_retries_training():
    db = ClaimDatabase()
    original_update = db.update_training_job

    def unknown_commit(*args, **kwargs):
        original_update(*args, **kwargs)
        raise RuntimeError("claim confirmation lost")

    db.update_training_job = unknown_commit
    executor = TrainingJobExecutor.__new__(TrainingJobExecutor)
    executor._db = db
    executor._execute_head = Mock()
    job = SimpleNamespace(job_id="same-job", job_type="anomaly_head", camera_id="camera-01")
    with pytest.raises(RuntimeError, match="confirmation lost"):
        executor._execute_job(job)
    assert db.status == "running"
    db.update_training_job = original_update
    assert executor._execute_job(job) is False
    executor._execute_head.assert_not_called()


@pytest.mark.parametrize("aware", [False, True])
def test_timeout_does_not_free_a_potentially_live_worker(aware):
    started_at = datetime.now(timezone.utc) - timedelta(hours=7)
    if not aware:
        started_at = started_at.replace(tzinfo=None)
    db = Mock()
    db.list_training_jobs.return_value = [SimpleNamespace(
        job_id="slow-job", started_at=started_at, error=None,
    )]
    db.update_training_job.return_value = True
    executor = TrainingJobExecutor.__new__(TrainingJobExecutor)
    executor._db = db

    assert executor.recover_stale_jobs() == 1
    update = db.update_training_job.call_args.kwargs
    assert update["expected_status"] == "running"
    assert "status" not in update and "completed_at" not in update
    assert "verify worker termination" in update["error"]


def test_timeout_annotation_losing_cas_does_not_count_as_recovered():
    db = Mock()
    db.list_training_jobs.return_value = [SimpleNamespace(
        job_id="finished-job", started_at=datetime.now(timezone.utc) - timedelta(hours=7), error=None,
    )]
    db.update_training_job.return_value = False
    executor = TrainingJobExecutor.__new__(TrainingJobExecutor)
    executor._db = db
    assert executor.recover_stale_jobs() == 0


def _backbone(version, job_id=None):
    return BackboneRecord(
        backbone_version_id=version, backbone_type="dinov2_vits14",
        checkpoint_path=f"{version}.pt", checkpoint_hash="a" * 64,
        dataset_hash="b" * 64, camera_ids_used='["camera-01"]',
        training_job_id=job_id, is_active=job_id is None,
    )


@pytest.mark.parametrize("status,accepted", [("running", True), ("failed", False), ("complete", False)])
def test_backbone_activation_and_completion_share_the_job_fence(database, status, accepted):
    database.update_training_job("claim-job", job_type="ssl_backbone", status=status)
    database.save_backbone(_backbone("old-backbone"))
    assert database.complete_backbone_training_job(
        _backbone("new-backbone", "claim-job"), job_id="claim-job",
        completed_at=datetime.now(timezone.utc), duration_seconds=1.0,
        metrics="{}", artifacts_path="new-backbone.pt",
    ) is accepted
    assert database.get_training_job("claim-job").status == ("complete" if accepted else status)
    assert database.get_active_backbone().backbone_version_id == (
        "new-backbone" if accepted else "old-backbone"
    )
    assert len(database.list_backbones()) == (2 if accepted else 1)


def test_backbone_registration_failure_rolls_back_job_and_previous_active_backbone(database):
    database.update_training_job("claim-job", job_type="ssl_backbone", status="running")
    database.save_backbone(_backbone("existing-backbone"))
    with pytest.raises(IntegrityError):
        database.complete_backbone_training_job(
            _backbone("existing-backbone", "claim-job"), job_id="claim-job",
            completed_at=datetime.now(timezone.utc), duration_seconds=1.0,
            metrics="{}", artifacts_path="existing-backbone.pt",
        )
    assert database.get_training_job("claim-job").status == "running"
    assert database.get_active_backbone().backbone_version_id == "existing-backbone"
    assert len(database.list_backbones()) == 1
