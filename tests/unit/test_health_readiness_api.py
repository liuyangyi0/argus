"""Liveness and detection readiness must drive different operational actions."""

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from argus.core.health import HealthMonitor
from argus.dashboard.routes.system import router


def _client(monitor):
    app = FastAPI()
    app.state.health_monitor = monitor
    app.include_router(router, prefix="/api/system")
    return TestClient(app)


def test_missing_monitor_is_live_but_not_detection_ready():
    client = _client(None)
    live = client.get("/api/system/health/live")
    assert live.status_code == 200
    assert live.json()["data"]["alive"] is True
    ready = client.get("/api/system/health/ready")
    assert ready.status_code == 503
    assert ready.json()["code"] == 50300
    assert ready.json()["data"]["reasons"] == ["health_monitor_unavailable"]
    assert ready.headers["cache-control"] == "no-store"


def test_scene_mismatch_keeps_preview_capture_healthy_and_readiness_blocked():
    monitor = HealthMonitor()
    monitor.update_camera(
        "cam", connected=True, frames_captured=100,
        scene_guard_usable=False, scene_guard_state="moved",
        scene_guard_reason="camera_moved", model_ready=True,
    )
    client = _client(monitor)
    assert client.get("/api/system/health/live").status_code == 200
    legacy = client.get("/api/system/health").json()["data"]
    assert legacy["status"] == "healthy"
    assert legacy["cameras"][0]["capture_state"] == "receiving"
    ready = client.get("/api/system/health/ready")
    assert ready.status_code == 503
    camera = ready.json()["data"]["cameras"][0]
    assert camera["detection_state"] == "blocked"
    assert camera["scene_guard_state"] == "moved"
    assert camera["model_ready"] is True

    monitor.update_camera(
        "cam", connected=True, frames_captured=101,
        scene_guard_usable=True, scene_guard_state="stable", model_ready=True,
    )
    restored = client.get("/api/system/health/ready")
    assert restored.status_code == 200
    assert restored.json()["data"]["ready"] is True


def test_stale_capture_fails_readiness_without_failing_liveness(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("argus.core.health.time.monotonic", lambda: clock[0])
    monitor = HealthMonitor()
    monitor.update_camera(
        "cam", connected=True, frames_captured=100,
        scene_guard_usable=True, model_ready=True,
    )
    clock[0] = 41.0
    monitor.update_camera(
        "cam", connected=True, frames_captured=100,
        scene_guard_usable=True, model_ready=True,
    )
    client = _client(monitor)
    assert client.get("/api/system/health/live").status_code == 200
    ready = client.get("/api/system/health/ready")
    assert ready.status_code == 503
    assert ready.json()["data"]["cameras"][0]["capture_state"] == "stale"


@pytest.mark.parametrize("mode", ["collection", "training", "learning"])
def test_healthy_models_do_not_make_non_operational_modes_ready(mode):
    monitor = HealthMonitor()
    monitor.update_camera(
        "cam", connected=True, frames_captured=100,
        scene_guard_usable=True, model_ready=True, pipeline_mode=mode,
    )
    client = _client(monitor)
    assert client.get("/api/system/health/live").status_code == 200
    response = client.get("/api/system/health/ready")
    assert response.status_code == 503
    assert response.json()["data"]["cameras"][0]["pipeline_mode"] == mode
    assert response.json()["data"]["cameras"][0]["detection_state"] == "blocked"
