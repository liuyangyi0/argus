"""Tests for /api/models/status aggregation endpoint."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from argus.core.health import HealthMonitor
from argus.core.model_status import ModelStatus
from argus.dashboard.app import create_app
from argus.storage.database import Database


@pytest.fixture
def db(tmp_path):
    database = Database(database_url=f"sqlite:///{tmp_path / 'test.db'}")
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def client(db, tmp_path):
    app = create_app(database=db, health_monitor=HealthMonitor(),
                     alerts_dir=str(tmp_path / "alerts"))
    return TestClient(app)


def _make_runner(camera_id: str, anomaly_status: ModelStatus,
                 yolo_status: ModelStatus) -> SimpleNamespace:
    pipeline = SimpleNamespace(
        _anomaly_detector=SimpleNamespace(status=anomaly_status),
        _object_detector=SimpleNamespace(status=yolo_status),
    )
    return SimpleNamespace(_pipeline=pipeline)


class TestModelsStatusRoute:
    def test_empty_when_no_camera_manager(self, client):
        """With no manager attached the endpoint returns an empty list."""
        resp = client.get("/api/models/status")
        assert resp.status_code == 200
        assert resp.json()["data"]["models"] == []

    def test_aggregates_across_cameras(self, client):
        cam1_anom = ModelStatus(name="anomaly", camera_id="cam1")
        cam1_anom.mark_loaded(backend="openvino", model_path="/m/a.xml")
        cam1_anom.mark_inference_failure("Broadcast check failed")
        cam1_anom.mark_inference_failure("Broadcast check failed")

        cam1_yolo = ModelStatus(name="yolo", camera_id="cam1")
        cam1_yolo.mark_loaded(backend="cuda", model_path="yolo11n.pt")
        cam1_yolo.mark_inference_success()

        cam2_anom = ModelStatus(name="anomaly", camera_id="cam2")
        cam2_anom.mark_loaded(backend="ssim-fallback")
        cam2_anom.set_extra(reason="no_model_found")

        cam2_yolo = ModelStatus(name="yolo", camera_id="cam2")
        cam2_yolo.mark_load_failed("missing weights")

        manager = MagicMock()
        manager._runners = {
            "cam1": _make_runner("cam1", cam1_anom, cam1_yolo),
            "cam2": _make_runner("cam2", cam2_anom, cam2_yolo),
        }
        client.app.state.camera_manager = manager

        resp = client.get("/api/models/status")
        assert resp.status_code == 200
        body = resp.json()["data"]["models"]
        assert len(body) == 4

        by_key = {(m["camera_id"], m["name"]): m for m in body}
        assert by_key[("cam1", "anomaly")]["backend"] == "openvino"
        assert by_key[("cam1", "anomaly")]["consecutive_failures"] == 2
        assert by_key[("cam1", "anomaly")]["last_error"] == "Broadcast check failed"
        assert by_key[("cam2", "anomaly")]["backend"] == "ssim-fallback"
        assert by_key[("cam2", "anomaly")]["extra"]["reason"] == "no_model_found"
        assert by_key[("cam2", "yolo")]["loaded"] is False

    def test_skips_runners_without_pipeline(self, client):
        manager = MagicMock()
        manager._runners = {"dead": SimpleNamespace(_pipeline=None)}
        client.app.state.camera_manager = manager

        resp = client.get("/api/models/status")
        assert resp.status_code == 200
        assert resp.json()["data"]["models"] == []
