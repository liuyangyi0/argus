from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
import asyncio
import threading

import pytest
from fastapi.testclient import TestClient

from argus.config.schema import ArgusConfig
from argus.core.health import HealthMonitor
from argus.dashboard.app import create_app
from argus.storage.database import Database


def _client(config: ArgusConfig, camera_manager: object | None, db, health, alerts_dir):
    app = create_app(
        database=db,
        camera_manager=camera_manager,
        health_monitor=health,
        alerts_dir=str(alerts_dir),
        config=config,
    )
    return TestClient(app)


@pytest.fixture
def db(tmp_path):
    database = Database(database_url=f"sqlite:///{tmp_path / 'config-api.db'}")
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def health():
    return HealthMonitor()


@pytest.fixture
def alerts_dir(tmp_path):
    path = tmp_path / "alerts"
    path.mkdir()
    return path


def test_detection_params_accept_json_payload(db, health, alerts_dir):
    config = ArgusConfig()
    config.cameras = [SimpleNamespace(camera_id="cam_01")]

    camera_manager = MagicMock()
    camera_manager.update_pipeline_thresholds.return_value = (
        1,
        {
            "anomaly_threshold": True,
            "severity": True,
            "temporal": True,
            "suppression": True,
        },
    )
    client = _client(config, camera_manager, db, health, alerts_dir)

    response = client.post(
        "/api/config/detection-params",
        json={
            "anomaly_threshold": 0.42,
            "sev_info": 0.0,
            "sev_low": 0.35,
            "temp_gap": 1.5,
            "supp_zone": 3.0,
        },
    )

    assert response.status_code == 200
    assert config.alerts.severity_thresholds.info == 0.0
    assert config.alerts.severity_thresholds.low == 0.35
    assert config.alerts.temporal.max_gap_seconds == 1.5
    assert config.alerts.suppression.same_zone_window_seconds == 3.0
    camera_manager.update_pipeline_thresholds.assert_called_once_with(
        camera_ids=["cam_01"],
        anomaly_threshold=0.42,
        severity_changed=True,
        temporal_changed=True,
        suppression_changed=True,
    )
    data = response.json()["data"]
    assert data["anomaly_threshold"]["hot_reloaded"] is True
    assert data["severity"]["changed"] is True


def test_notifications_accept_json_payload(db, health, alerts_dir):
    config = ArgusConfig()
    client = _client(config, None, db, health, alerts_dir)

    response = client.post(
        "/api/config/notifications",
        json={
            "webhook_enabled": True,
            "webhook_url": "https://example.test/hook",
            "webhook_timeout": 7,
        },
    )

    assert response.status_code == 200
    assert config.alerts.webhook.enabled is True
    assert config.alerts.webhook.url == "https://example.test/hook"
    assert config.alerts.webhook.timeout == 7

    response = client.post(
        "/api/config/notifications",
        json={"webhook_enabled": False},
    )

    assert response.status_code == 200
    assert config.alerts.webhook.enabled is False


@pytest.mark.parametrize("key", ["auth.enabled", "classifier.min_anomaly_score_to_classify"])
def test_modules_reject_non_module_fields(key, db, health, alerts_dir):
    config = ArgusConfig()
    before = config.model_dump()
    response = _client(config, None, db, health, alerts_dir).post(
        "/api/config/modules", json={"key": key, "value": True},
    )
    assert response.status_code == 400
    assert config.model_dump() == before


def test_modules_require_literal_boolean(db, health, alerts_dir):
    config = ArgusConfig()
    response = _client(config, None, db, health, alerts_dir).post(
        "/api/config/modules", json={"key": "classifier.enabled", "value": "false"},
    )
    assert response.status_code == 422
    assert config.classifier.enabled is False


def test_modules_runtime_failure_is_not_applied_success(db, health, alerts_dir):
    config = ArgusConfig()
    manager = MagicMock()
    manager.reload_module_for_cameras.side_effect = RuntimeError("fake runtime failure")
    response = _client(config, manager, db, health, alerts_dir).post(
        "/api/config/modules", json={"key": "classifier.enabled", "value": True},
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["all_applied"] is False
    assert data["hot_reload_failed"] == 1
    assert data["restart_required"] is True
    assert config.classifier.enabled is True  # desired state, not applied state


def test_auxiliary_status_exposes_real_readiness_and_failure_reasons(db, health, alerts_dir):
    manager = MagicMock()
    detail = {"camera_id": "fake", "state": "failed", "ready": False,
              "loaded": False, "degraded": True, "reason": "fake unavailable", "busy": True}
    manager.get_runtime_component_summary.return_value = {
        "total_pipelines": 1, "segmenter_attached": 1, "segmenter_loaded": 0,
        "segmenter_ready": 0, "segmenter_failed": 1, "segmenter_degraded": 1,
        "segmenter_details": [detail],
    }
    response = _client(ArgusConfig(), manager, db, health, alerts_dir).get("/api/config/segmenter")
    runtime = response.json()["data"]["runtime"]
    assert runtime["pipelines_loaded"] == 0
    assert runtime["pipelines_ready"] == 0
    assert runtime["pipelines_failed"] == 1
    assert runtime["details"] == [detail]


@pytest.mark.asyncio
async def test_modules_serialize_desired_state_without_blocking_event_loop(monkeypatch):
    from argus.dashboard.routes import config as routes

    started, release = threading.Event(), threading.Event()
    applied = []

    def reload_modules(camera_ids, key, value):
        if value:
            started.set()
            assert release.wait(2), "test must release fake blocking SDK boundary"
        applied.append(value)
        return {"seen": 1, "succeeded": 1, "failed": 0}

    cfg = ArgusConfig()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        config=cfg, camera_manager=SimpleNamespace(reload_module_for_cameras=reload_modules),
        config_path=None,
    )))
    monkeypatch.setattr(routes, "_deny_without_permission", lambda *args: None)
    first = asyncio.create_task(routes.update_module_toggle(
        request, routes.ModuleToggleRequest(key="classifier.enabled", value=True),
    ))
    second = None
    try:
        assert await asyncio.wait_for(asyncio.to_thread(started.wait, 1), 1.5)
        assert not first.done()
        second = asyncio.create_task(routes.update_module_toggle(
            request, routes.ModuleToggleRequest(key="classifier.enabled", value=False),
        ))
        await asyncio.sleep(0)
        assert cfg.classifier.enabled is True
        assert applied == []
    finally:
        release.set()
        await first
        if second is not None:
            await second
    assert applied == [True, False]
    assert cfg.classifier.enabled is False


@pytest.mark.parametrize("payload", [
    {"max_points": 21}, {"min_anomaly_score": 0.0},
    {"min_mask_area_px": 0}, {"timeout_seconds": 61.0},
])
def test_segmenter_params_use_startup_schema_atomically(payload, db, health, alerts_dir):
    config = ArgusConfig()
    before = config.segmenter.model_dump()
    manager = MagicMock()
    response = _client(config, manager, db, health, alerts_dir).put(
        "/api/config/segmenter/params", json=payload,
    )
    assert response.status_code == 400
    assert config.segmenter.model_dump() == before
    manager.update_segmenter_params.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_request_keeps_config_ownership_until_worker_and_persist_finish(monkeypatch):
    from argus.dashboard.routes import config as routes

    started, release = threading.Event(), threading.Event()
    applied, persisted = [], []

    def reload_modules(camera_ids, key, value):
        applied.append(("camera-a", value))
        if value:
            started.set()
            assert release.wait(2)
        applied.append(("camera-b", value))
        return {"seen": 2, "succeeded": 2, "failed": 0}

    cfg = ArgusConfig()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        config=cfg, camera_manager=SimpleNamespace(reload_module_for_cameras=reload_modules),
        config_path="fake-config-no-disk-io.yaml",
    )))
    monkeypatch.setattr(routes, "_deny_without_permission", lambda *args: None)
    monkeypatch.setattr("argus.config.loader.save_config", lambda config, path: persisted.append(config.classifier.enabled))
    first = asyncio.create_task(routes.update_module_toggle(
        request, routes.ModuleToggleRequest(key="classifier.enabled", value=True),
    ))
    second = None
    try:
        assert await asyncio.wait_for(asyncio.to_thread(started.wait, 1), 1.5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(routes.update_module_toggle(
            request, routes.ModuleToggleRequest(key="classifier.enabled", value=False),
        ))
        await asyncio.sleep(0)
        assert cfg.classifier.enabled is True
        assert applied == [("camera-a", True)]
        assert persisted == []
    finally:
        release.set()
        if second is not None:
            await second
    assert applied == [("camera-a", True), ("camera-b", True), ("camera-a", False), ("camera-b", False)]
    assert persisted == [True, False]
    assert cfg.classifier.enabled is False


@pytest.mark.parametrize("changes", [
    {"high_risk_labels": ["wrench"], "suppress_labels": ["wrench"]},
    {"low_risk_labels": ["wrench"]},
])
def test_classifier_active_risk_bucket_conflicts_are_rejected_atomically(changes, db, health, alerts_dir):
    config = ArgusConfig()
    config.classifier.vocabulary = ["wrench", "shadow"]
    config.classifier.high_risk_labels = ["wrench"]
    config.classifier.low_risk_labels = ["shadow"]
    config.classifier.suppress_labels = []
    before = config.classifier.model_dump()
    manager = MagicMock()
    response = _client(config, manager, db, health, alerts_dir).put(
        "/api/config/classifier/vocabulary",
        json={"vocabulary": ["wrench", "shadow"], **changes},
    )
    assert response.status_code == 400
    assert config.classifier.model_dump() == before
    manager.update_classifier_vocabulary.assert_not_called()


def test_reload_rejects_auxiliary_drift_before_applying_any_camera(monkeypatch, db, health, alerts_dir):
    config, replacement = ArgusConfig(), ArgusConfig()
    replacement.classifier.enabled = True
    manager = MagicMock()
    client = _client(config, manager, db, health, alerts_dir)
    client.app.state.config_path = "fake-config-no-media.yaml"
    monkeypatch.setattr("argus.config.loader.load_config", lambda path: replacement)
    response = client.post("/api/config/reload")
    assert response.status_code == 409
    assert client.app.state.config is config
    manager.apply_runtime_camera_config.assert_not_called()


def test_reload_preserves_auxiliary_configuration_object_identity(monkeypatch, db, health, alerts_dir):
    config, replacement = ArgusConfig(), ArgusConfig()
    manager = MagicMock()
    client = _client(config, manager, db, health, alerts_dir)
    client.app.state.config_path = "fake-config-no-media.yaml"
    monkeypatch.setattr("argus.config.loader.load_config", lambda path: replacement)
    response = client.post("/api/config/reload")
    assert response.status_code == 200
    assert client.app.state.config is replacement
    for section in ("classifier", "segmenter", "physics"):
        assert getattr(replacement, section) is getattr(config, section)


@pytest.mark.asyncio
async def test_restart_cancellation_does_not_abandon_owned_restart(monkeypatch):
    from argus.dashboard.routes import config as routes

    entered, release = threading.Event(), threading.Event()
    completed = []

    def restart(camera_id):
        entered.set()
        assert release.wait(2)
        completed.append(camera_id)
        return True

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        camera_manager=SimpleNamespace(restart_camera=restart),
    )))
    monkeypatch.setattr(routes, "_deny_without_permission", lambda *args: None)
    caller = asyncio.create_task(routes.restart_camera(request, "fake-camera"))
    try:
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 1), 1.5)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        pending = list(request.app.state.auxiliary_config_tasks)
        assert len(pending) == 1
        release.set()
        await asyncio.wait_for(asyncio.gather(*pending), 1)
        assert completed == ["fake-camera"]
    finally:
        release.set()
