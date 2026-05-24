"""Regression tests for active-learning label lineage."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from argus.dashboard.routes import labeling, training_jobs
from argus.storage.database import Database
from argus.storage.models import (
    LabelingQueueRecord,
    TrainingJobStatus,
)


@pytest.fixture
def db(tmp_path):
    database = Database(database_url=f"sqlite:///{(tmp_path / 'argus.db').as_posix()}")
    database.initialize()
    return database


@pytest.fixture
def app(db):
    app = FastAPI()
    app.state.db = db
    app.state.database = db

    @app.middleware("http")
    async def _engineer_user(request, call_next):
        request.state.user = {"username": "engineer", "role": "engineer"}
        return await call_next(request)

    app.include_router(labeling.router, prefix="/labeling")
    app.include_router(training_jobs.router, prefix="/training-jobs")
    return app


def _save_labeled_entries(db: Database, tmp_path, count: int = 5) -> list[int]:
    ids: list[int] = []
    for i in range(count):
        frame = tmp_path / f"label_{i}.jpg"
        frame.write_bytes(b"jpg")
        record = db.save_labeling_entry(
            LabelingQueueRecord(
                camera_id="cam_01",
                zone_id="zone_a",
                frame_number=i,
                frame_path=str(frame),
                anomaly_score=0.5,
                entropy=0.7,
            )
        )
        labeled = db.label_entry(record.id, label="anomaly", labeled_by="tester")
        ids.append(labeled.id)
    return ids


def test_trigger_retrain_stores_label_ids_without_consuming(app, db, tmp_path):
    entry_ids = _save_labeled_entries(db, tmp_path)
    client = TestClient(app)

    response = client.post(
        "/labeling/trigger-retrain",
        json={"camera_id": "cam_01", "model_type": "patchcore"},
    )

    assert response.status_code == 200
    job_id = response.json()["data"]["job_id"]
    job = db.get_training_job(job_id)
    assert job.status == TrainingJobStatus.PENDING_CONFIRMATION.value
    assert json.loads(job.hyperparameters)["labeling_entry_ids"] == entry_ids

    reusable = db.get_labeled_entries(camera_id="cam_01", trained_into="")
    assert {entry.id for entry in reusable} == set(entry_ids)


def test_rejected_triggered_job_leaves_labels_reusable(app, db, tmp_path):
    entry_ids = _save_labeled_entries(db, tmp_path)
    client = TestClient(app)
    trigger_response = client.post(
        "/labeling/trigger-retrain",
        json={"camera_id": "cam_01", "model_type": "patchcore"},
    )
    job_id = trigger_response.json()["data"]["job_id"]

    reject_response = client.post(
        f"/training-jobs/{job_id}/reject",
        json={"rejected_by": "engineer", "reason": "not enough review"},
    )

    assert reject_response.status_code == 200
    job = db.get_training_job(job_id)
    assert job.status == TrainingJobStatus.REJECTED.value
    reusable = db.get_labeled_entries(camera_id="cam_01", trained_into="")
    assert {entry.id for entry in reusable} == set(entry_ids)
