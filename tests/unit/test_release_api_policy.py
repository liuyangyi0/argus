"""Focused API policy tests for model release gates."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from argus.config.loader import load_config
from argus.dashboard.app import create_app
from argus.dashboard.routes.models import _get_release_anomaly_config
from argus.storage.database import Database
from argus.storage.model_registry import ModelRegistry
from argus.storage.models import ModelStage
from argus.storage.release_pipeline import ReleasePipeline


class LegacyCameraManager:
    def __init__(self) -> None:
        self._pipelines = {"cam_01": object()}
        self.reload_model = MagicMock(return_value=True)


class ReleaseStateCameraManager(LegacyCameraManager):
    def __init__(self) -> None:
        super().__init__()
        self.release_state_calls: list[str] = []

    def apply_model_release_state(self, camera_id: str) -> dict:
        self.release_state_calls.append(camera_id)
        return {
            "camera_id": camera_id,
            "call_count": len(self.release_state_calls),
        }


@pytest.fixture()
def db(tmp_path):
    database = Database(database_url=f"sqlite:///{tmp_path / 'api_policy.db'}")
    database.initialize()
    yield database
    database.close()


@pytest.fixture()
def alerts_dir(tmp_path):
    path = tmp_path / "alerts"
    path.mkdir()
    return path


def _make_app(db, camera_manager, alerts_dir):
    app = create_app(
        database=db,
        camera_manager=camera_manager,
        alerts_dir=str(alerts_dir),
    )
    app.state._release_pipeline = ReleasePipeline(
        db.get_session,
        min_shadow_days=0,
        min_canary_days=0,
    )
    return app


def _create_model_tree(root: Path, camera_id: str = "cam_01") -> tuple[Path, Path]:
    model_dir = root / "data" / "models" / camera_id / "default"
    model_dir.mkdir(parents=True)
    (model_dir / "model.xml").write_text("model", encoding="utf-8")

    baseline_dir = root / "baseline"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "img.png").write_bytes(b"img")
    return model_dir, baseline_dir


def _register_model(db, model_dir: Path, baseline_dir: Path, stage: str) -> str:
    registry = ModelRegistry(session_factory=db.get_session)
    version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")
    if stage in {
        ModelStage.SHADOW.value,
        ModelStage.CANARY.value,
        ModelStage.PRODUCTION.value,
    }:
        registry.promote(version_id, ModelStage.SHADOW.value, triggered_by="test")
    if stage in {ModelStage.CANARY.value, ModelStage.PRODUCTION.value}:
        registry.promote(
            version_id,
            ModelStage.CANARY.value,
            triggered_by="test",
            canary_camera_id="cam_01",
        )
    if stage == ModelStage.PRODUCTION.value:
        registry.promote(version_id, ModelStage.PRODUCTION.value, triggered_by="test")
    return version_id


def test_release_pipeline_uses_camera_anomaly_stage_waits():
    config = load_config(Path("configs/default.yaml"))
    config.cameras[0].anomaly.min_shadow_days = 0
    config.cameras[0].anomaly.min_canary_days = 0

    anomaly_cfg = _get_release_anomaly_config(config)

    assert anomaly_cfg is config.cameras[0].anomaly
    assert anomaly_cfg.min_shadow_days == 0
    assert anomaly_cfg.min_canary_days == 0


def test_promote_shadow_canary_production_returns_runtime_state(db, alerts_dir, tmp_path):
    model_dir, baseline_dir = _create_model_tree(tmp_path)
    registry = ModelRegistry(session_factory=db.get_session)
    version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")

    camera_manager = ReleaseStateCameraManager()
    client = TestClient(_make_app(db, camera_manager, alerts_dir))

    promotions = [
        {"target_stage": "shadow", "triggered_by": "tester"},
        {
            "target_stage": "canary",
            "triggered_by": "tester",
            "canary_camera_id": "cam_01",
        },
        {"target_stage": "production", "triggered_by": "tester"},
    ]

    for expected_call_count, body in enumerate(promotions, start=1):
        response = client.post(f"/api/models/{version_id}/promote", json=body)

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["runtime_synced"] is True
        assert data["runtime_state"] == {
            "camera_id": "cam_01",
            "call_count": expected_call_count,
        }

    assert camera_manager.release_state_calls == ["cam_01", "cam_01", "cam_01"]


def test_promote_defaults_triggered_by_to_authenticated_user(db, alerts_dir, tmp_path):
    model_dir, baseline_dir = _create_model_tree(tmp_path)
    registry = ModelRegistry(session_factory=db.get_session)
    version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")

    camera_manager = ReleaseStateCameraManager()
    client = TestClient(_make_app(db, camera_manager, alerts_dir))

    response = client.post(
        f"/api/models/{version_id}/promote",
        json={"target_stage": "shadow"},
    )

    assert response.status_code == 200
    event = registry.get_version_events(model_version_id=version_id, limit=1)[0]
    assert event.to_stage == "shadow"
    assert event.triggered_by == "system"


def test_retire_returns_runtime_state(db, alerts_dir, tmp_path):
    model_dir, baseline_dir = _create_model_tree(tmp_path)
    registry = ModelRegistry(session_factory=db.get_session)
    version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")

    camera_manager = ReleaseStateCameraManager()
    client = TestClient(_make_app(db, camera_manager, alerts_dir))

    response = client.post(
        f"/api/models/{version_id}/retire",
        json={"triggered_by": "tester"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["model"]["stage"] == "retired"
    assert data["runtime_synced"] is True
    assert data["runtime_state"] == {"camera_id": "cam_01", "call_count": 1}


def test_rollback_returns_runtime_state_and_model(db, alerts_dir, tmp_path):
    model_dir_1, baseline_dir_1 = _create_model_tree(tmp_path / "v1")
    version_id_1 = _register_model(
        db,
        model_dir_1,
        baseline_dir_1,
        ModelStage.PRODUCTION.value,
    )
    model_dir_2, baseline_dir_2 = _create_model_tree(tmp_path / "v2")
    version_id_2 = _register_model(
        db,
        model_dir_2,
        baseline_dir_2,
        ModelStage.PRODUCTION.value,
    )

    camera_manager = LegacyCameraManager()
    client = TestClient(_make_app(db, camera_manager, alerts_dir))

    response = client.post(f"/api/models/{version_id_2}/rollback")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["activated"] == version_id_1
    assert data["model"]["model_version_id"] == version_id_1
    assert data["runtime_synced"] is True
    assert data["runtime_state"] == "applied"
    camera_manager.reload_model.assert_called_once_with(
        "cam_01",
        str(model_dir_1 / "model.xml"),
        version_tag=version_id_1,
    )


@pytest.mark.parametrize(
    "stage",
    [
        None,
        ModelStage.CANDIDATE.value,
        ModelStage.SHADOW.value,
        ModelStage.CANARY.value,
    ],
)
def test_baseline_deploy_rejects_unregistered_candidate_shadow_canary(
    db,
    alerts_dir,
    tmp_path,
    monkeypatch,
    stage,
):
    monkeypatch.chdir(tmp_path)
    model_dir, baseline_dir = _create_model_tree(tmp_path)
    if stage is not None:
        _register_model(db, model_dir, baseline_dir, stage)

    camera_manager = LegacyCameraManager()
    client = TestClient(_make_app(db, camera_manager, alerts_dir))

    response = client.post(
        "/api/baseline/deploy",
        json={"camera_id": "cam_01", "model_path": str(model_dir / "model.xml")},
    )

    assert response.status_code == 400
    camera_manager.reload_model.assert_not_called()


def test_canary_camera_mismatch_returns_400(db, alerts_dir, tmp_path):
    model_dir, baseline_dir = _create_model_tree(tmp_path)
    registry = ModelRegistry(session_factory=db.get_session)
    version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")
    registry.promote(version_id, ModelStage.SHADOW.value, triggered_by="test")

    camera_manager = ReleaseStateCameraManager()
    client = TestClient(_make_app(db, camera_manager, alerts_dir))

    response = client.post(
        f"/api/models/{version_id}/promote",
        json={
            "target_stage": "canary",
            "triggered_by": "tester",
            "canary_camera_id": "other_cam",
        },
    )

    assert response.status_code == 400
    assert "canary_camera_id" in response.json()["msg"]
    assert camera_manager.release_state_calls == []


def test_active_production_cannot_be_retired_directly_via_api(db, alerts_dir, tmp_path):
    model_dir, baseline_dir = _create_model_tree(tmp_path)
    version_id = _register_model(
        db,
        model_dir,
        baseline_dir,
        ModelStage.PRODUCTION.value,
    )

    camera_manager = ReleaseStateCameraManager()
    client = TestClient(_make_app(db, camera_manager, alerts_dir))

    retire_response = client.post(
        f"/api/models/{version_id}/retire",
        json={"triggered_by": "tester"},
    )
    promote_response = client.post(
        f"/api/models/{version_id}/promote",
        json={"target_stage": "retired", "triggered_by": "tester"},
    )

    assert retire_response.status_code == 400
    assert "cannot be retired directly" in retire_response.json()["msg"]
    assert promote_response.status_code == 400
    assert "cannot be retired directly" in promote_response.json()["msg"]
    assert camera_manager.release_state_calls == []

    record = ModelRegistry(session_factory=db.get_session).get_by_version_id(version_id)
    assert record.stage == ModelStage.PRODUCTION.value
    assert record.is_active is True


def test_active_production_cannot_be_retired_through_promote_api(
    db,
    alerts_dir,
    tmp_path,
):
    model_dir, baseline_dir = _create_model_tree(tmp_path)
    version_id = _register_model(
        db,
        model_dir,
        baseline_dir,
        ModelStage.PRODUCTION.value,
    )

    camera_manager = ReleaseStateCameraManager()
    client = TestClient(_make_app(db, camera_manager, alerts_dir))

    response = client.post(
        f"/api/models/{version_id}/promote",
        json={"target_stage": "retired", "triggered_by": "tester"},
    )

    assert response.status_code == 400
    assert "cannot be retired directly" in response.json()["msg"]
    assert camera_manager.release_state_calls == []

    record = ModelRegistry(session_factory=db.get_session).get_by_version_id(version_id)
    assert record.stage == ModelStage.PRODUCTION.value
    assert record.is_active is True
