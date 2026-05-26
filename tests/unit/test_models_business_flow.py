"""End-to-end business flow tests for model training and release."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from argus.anomaly.job_executor import TrainingJobExecutor
from argus.anomaly.trainer import TrainingResult, TrainingStatus
from argus.capture.manager import CameraManager
from argus.config.schema import AlertConfig, CameraConfig
from argus.dashboard.app import create_app
from argus.storage.database import Database
from argus.storage.model_registry import ModelRegistry
from argus.storage.models import ModelStage
from argus.storage.release_pipeline import ReleasePipeline


class _PipelineDouble:
    def __init__(self, camera_config: CameraConfig, current_version: str | None):
        self.camera_config = camera_config
        self._model_version_id = current_version
        self._shadow_runner = None
        self.reload_anomaly_model = MagicMock(return_value=True)

    def set_model_version_id(self, version: str | None) -> None:
        self._model_version_id = version

    def set_shadow_runner(self, shadow_runner: object | None) -> dict:
        old_runner = self._shadow_runner
        old_version = getattr(old_runner, "shadow_version_id", None)
        new_version = getattr(shadow_runner, "shadow_version_id", None)
        self._shadow_runner = shadow_runner
        return {
            "shadow_previous_version": old_version,
            "shadow_model_version": new_version,
            "shadow_attached": shadow_runner is not None and old_version != new_version,
            "shadow_detached": old_runner is not None and shadow_runner is None,
        }


def _model_dir(root: Path, name: str) -> Path:
    model_dir = root / name
    model_dir.mkdir(parents=True)
    (model_dir / "model.xml").write_text("model", encoding="utf-8")
    return model_dir


def _baseline_dir(root: Path, camera_id: str = "cam_01") -> Path:
    baseline_dir = root / camera_id / "default"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "baseline.png").write_bytes(b"png")
    return baseline_dir


def _camera_manager(db: Database, current_version: str | None) -> CameraManager:
    cam_config = CameraConfig(
        camera_id="cam_01",
        name="Camera 01",
        source="0",
        protocol="file",
    )
    manager = CameraManager([cam_config], AlertConfig(), database=db)
    manager._pipelines["cam_01"] = _PipelineDouble(cam_config, current_version)
    manager._runners["cam_01"] = MagicMock()
    manager._notify_camera_status = MagicMock()
    return manager


def test_training_candidate_release_and_rollback_flow(tmp_path):
    database = Database(database_url=f"sqlite:///{tmp_path / 'argus.db'}")
    database.initialize()
    registry = ModelRegistry(session_factory=database.get_session)

    baselines_root = tmp_path / "baselines"
    baseline_dir = _baseline_dir(baselines_root)

    previous_model_dir = _model_dir(tmp_path / "models", "previous")
    previous_id = registry.register(
        previous_model_dir,
        baseline_dir,
        "cam_01",
        "patchcore",
    )
    registry.promote(previous_id, ModelStage.SHADOW.value, triggered_by="test")
    registry.promote(
        previous_id,
        ModelStage.CANARY.value,
        triggered_by="test",
        canary_camera_id="cam_01",
    )
    registry.promote(previous_id, ModelStage.PRODUCTION.value, triggered_by="test")

    camera_manager = _camera_manager(database, current_version=previous_id)
    alerts_dir = tmp_path / "alerts"
    alerts_dir.mkdir()
    app = create_app(
        database=database,
        camera_manager=camera_manager,
        alerts_dir=str(alerts_dir),
    )
    app.state._release_pipeline = ReleasePipeline(
        database.get_session,
        min_shadow_days=0,
        min_canary_days=0,
    )
    client = TestClient(app)

    create_response = client.post(
        "/api/training-jobs/",
        json={
            "job_type": "anomaly_head",
            "camera_id": "cam_01",
            "model_type": "patchcore",
            "hyperparameters": {"skip_baseline_validation": True},
        },
    )
    assert create_response.status_code == 201
    job_id = create_response.json()["data"]["job_id"]

    confirm_response = client.post(
        f"/api/training-jobs/{job_id}/confirm",
        json={"confirmed_by": "operator"},
    )
    assert confirm_response.status_code == 200

    trained_model_dir = _model_dir(tmp_path / "models", "trained")
    trainer = MagicMock()
    trainer.exports_dir = tmp_path / "exports"
    trainer.train.return_value = TrainingResult(
        status=TrainingStatus.COMPLETE,
        model_path=str(trained_model_dir),
        duration_seconds=1.5,
    )
    executor = TrainingJobExecutor(
        database=database,
        trainer=trainer,
        model_registry=registry,
        baselines_dir=baselines_root,
        model_packages_dir=tmp_path / "packages",
    )
    executor.execute(job_id)

    job_response = client.get(f"/api/training-jobs/{job_id}")
    assert job_response.status_code == 200
    job_data = job_response.json()["data"]
    assert job_data["status"] == "complete"
    candidate_id = job_data["model_version_id"]
    assert candidate_id

    list_response = client.get("/api/models/json", params={"camera_id": "cam_01"})
    assert list_response.status_code == 200
    models = list_response.json()["data"]["models"]
    candidate = next(m for m in models if m["model_version_id"] == candidate_id)
    assert candidate["stage"] == "candidate"
    assert candidate["is_active"] is False

    for body in (
        {"target_stage": "shadow", "triggered_by": "tester"},
        {
            "target_stage": "canary",
            "triggered_by": "tester",
            "canary_camera_id": "cam_01",
        },
        {"target_stage": "production", "triggered_by": "tester"},
    ):
        response = client.post(f"/api/models/{candidate_id}/promote", json=body)
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["runtime_synced"] is True
        assert data["runtime_state"]["camera_id"] == "cam_01"

    production = registry.get_by_version_id(candidate_id)
    assert production is not None
    assert production.stage == ModelStage.PRODUCTION.value
    assert production.is_active is True

    rollback_response = client.post(f"/api/models/{candidate_id}/rollback")
    assert rollback_response.status_code == 200
    rollback_data = rollback_response.json()["data"]
    assert rollback_data["activated"] == previous_id
    assert rollback_data["runtime_synced"] is True
    assert rollback_data["runtime_state"] == "applied"

    detail = client.get(f"/api/training-jobs/{job_id}").json()["data"]
    params = detail["hyperparameters"]
    if isinstance(params, str):
        params = json.loads(params)
    assert params["skip_baseline_validation"] is True


def test_training_job_actions_default_to_authenticated_user(tmp_path):
    database = Database(database_url=f"sqlite:///{tmp_path / 'training_audit.db'}")
    database.initialize()

    alerts_dir = tmp_path / "alerts"
    alerts_dir.mkdir()
    client = TestClient(create_app(database=database, alerts_dir=str(alerts_dir)))

    create_response = client.post(
        "/api/training-jobs/",
        json={"job_type": "ssl_backbone"},
    )
    assert create_response.status_code == 201
    job_id = create_response.json()["data"]["job_id"]

    detail = client.get(f"/api/training-jobs/{job_id}").json()["data"]
    assert detail["triggered_by"] == "system"

    confirm_response = client.post(f"/api/training-jobs/{job_id}/confirm", json={})
    assert confirm_response.status_code == 200
    confirmed = client.get(f"/api/training-jobs/{job_id}").json()["data"]
    assert confirmed["confirmed_by"] == "system"

    reject_create_response = client.post(
        "/api/training-jobs/",
        json={"job_type": "ssl_backbone"},
    )
    assert reject_create_response.status_code == 201
    reject_job_id = reject_create_response.json()["data"]["job_id"]

    reject_response = client.post(f"/api/training-jobs/{reject_job_id}/reject", json={})
    assert reject_response.status_code == 200
    rejected = client.get(f"/api/training-jobs/{reject_job_id}").json()["data"]
    assert rejected["error"] == "Rejected by system"


def test_reexport_api_invokes_trainer_with_requested_format(tmp_path, monkeypatch):
    database = Database(database_url=f"sqlite:///{tmp_path / 'reexport.db'}")
    database.initialize()
    registry = ModelRegistry(session_factory=database.get_session)

    model_dir = _model_dir(tmp_path / "models", "reexport-source")
    baseline_dir = _baseline_dir(tmp_path / "baselines")
    version_id = registry.register(
        model_dir,
        baseline_dir,
        "cam_01",
        "patchcore",
    )

    alerts_dir = tmp_path / "alerts"
    alerts_dir.mkdir()
    app = create_app(database=database, alerts_dir=str(alerts_dir))
    client = TestClient(app)

    trainer = MagicMock()
    trainer.reexport_model.return_value = {
        "status": "ok",
        "export_path": str(tmp_path / "exports" / "cam_01" / "default"),
        "format": "openvino",
        "quantization": "fp16",
    }
    monkeypatch.setattr(
        "argus.dashboard.routes.models._get_trainer",
        lambda _request: trainer,
    )

    response = client.post(
        f"/api/models/{version_id}/reexport",
        json={"export_format": "openvino", "quantization": "fp16"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "ok"
    trainer.reexport_model.assert_called_once_with(
        model_dir=model_dir,
        export_format="openvino",
        quantization="fp16",
        model_type="patchcore",
    )


def test_reexport_api_prefers_shared_app_state_trainer(tmp_path):
    database = Database(database_url=f"sqlite:///{tmp_path / 'shared_trainer.db'}")
    database.initialize()
    registry = ModelRegistry(session_factory=database.get_session)

    model_dir = _model_dir(tmp_path / "models", "shared-trainer-source")
    baseline_dir = _baseline_dir(tmp_path / "baselines")
    version_id = registry.register(
        model_dir,
        baseline_dir,
        "cam_01",
        "patchcore",
    )

    alerts_dir = tmp_path / "alerts"
    alerts_dir.mkdir()
    trainer = MagicMock()
    trainer.reexport_model.return_value = {
        "status": "ok",
        "export_path": str(tmp_path / "exports" / "cam_01" / "default"),
        "format": "openvino",
        "quantization": "fp16",
    }
    app = create_app(
        database=database,
        alerts_dir=str(alerts_dir),
        model_trainer=trainer,
    )
    client = TestClient(app)

    response = client.post(
        f"/api/models/{version_id}/reexport",
        json={"export_format": "openvino", "quantization": "fp16"},
    )

    assert response.status_code == 200
    trainer.reexport_model.assert_called_once_with(
        model_dir=model_dir,
        export_format="openvino",
        quantization="fp16",
        model_type="patchcore",
    )
