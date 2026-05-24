"""Runtime binding tests for model release-stage transitions."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from argus.capture.manager import CameraManager
from argus.config.schema import AlertConfig, CameraConfig
from argus.storage.models import Base, ModelRecord


@pytest.fixture()
def session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _model_dir(tmp_path: Path, name: str) -> Path:
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "model.xml").write_text("model", encoding="utf-8")
    return model_dir


def _add_model(
    session_factory,
    *,
    version: str,
    camera_id: str = "cam-01",
    stage: str,
    model_dir: Path,
    is_active: bool = False,
    canary_camera_id: str | None = None,
) -> None:
    with session_factory() as session:
        session.add(
            ModelRecord(
                model_version_id=version,
                camera_id=camera_id,
                model_type="patchcore",
                model_hash=f"hash-{version}",
                data_hash=f"data-{version}",
                stage=stage,
                component_type="full",
                model_path=str(model_dir),
                is_active=is_active,
                canary_camera_id=canary_camera_id,
            )
        )
        session.commit()


class _PipelineDouble:
    def __init__(
        self,
        camera_config: CameraConfig,
        current_version: str,
        shadow_runner: object | None = None,
    ) -> None:
        self.camera_config = camera_config
        self._model_version_id = current_version
        self._shadow_runner = shadow_runner
        self.reload_anomaly_model = MagicMock(return_value=True)
        self.set_model_version_id = MagicMock(side_effect=self._set_model_version_id)

    def _set_model_version_id(self, version: str | None) -> None:
        self._model_version_id = version

    def set_shadow_runner(self, shadow_runner: object | None) -> dict:
        old_runner = self._shadow_runner
        old_version = getattr(old_runner, "shadow_version_id", None)
        new_version = (
            getattr(shadow_runner, "shadow_version_id", None)
            if shadow_runner is not None
            else None
        )
        if old_version == new_version:
            return {
                "shadow_previous_version": old_version,
                "shadow_model_version": new_version,
                "shadow_attached": False,
                "shadow_detached": False,
            }

        self._shadow_runner = shadow_runner
        if old_runner is not None:
            old_runner.flush()
        return {
            "shadow_previous_version": old_version,
            "shadow_model_version": new_version,
            "shadow_attached": shadow_runner is not None,
            "shadow_detached": old_runner is not None and shadow_runner is None,
        }


def _manager(
    session_factory,
    *,
    current_version: str = "prod-v1",
    shadow_runner: object | None = None,
) -> tuple[CameraManager, _PipelineDouble, MagicMock, CameraConfig]:
    cam_config = CameraConfig(
        camera_id="cam-01",
        name="Camera 01",
        source="0",
        protocol="file",
    )
    db = SimpleNamespace(get_session=session_factory)
    manager = CameraManager([cam_config], AlertConfig(), database=db)
    pipeline = _PipelineDouble(cam_config, current_version, shadow_runner)
    runner = MagicMock()
    manager._pipelines["cam-01"] = pipeline
    manager._runners["cam-01"] = runner
    manager._notify_camera_status = MagicMock()
    return manager, pipeline, runner, cam_config


def test_startup_model_resolution_uses_canary_router(session_factory, tmp_path):
    prod_dir = _model_dir(tmp_path, "prod-v1")
    canary_dir = _model_dir(tmp_path, "canary-v2")
    _add_model(
        session_factory,
        version="prod-v1",
        stage="production",
        model_dir=prod_dir,
        is_active=True,
    )
    _add_model(
        session_factory,
        version="canary-v2",
        stage="canary",
        model_dir=canary_dir,
        canary_camera_id="cam-01",
    )

    manager, _, _, cam_config = _manager(session_factory)

    resolved = manager._resolve_model_path(cam_config)

    assert resolved == canary_dir / "model.xml"
    assert manager._model_version_id(cam_config, resolved) == "canary-v2"


def test_candidate_to_shadow_attaches_shadow_without_primary_reload(session_factory, tmp_path):
    prod_dir = _model_dir(tmp_path, "prod-v1")
    shadow_dir = _model_dir(tmp_path, "shadow-v2")
    _add_model(
        session_factory,
        version="prod-v1",
        stage="production",
        model_dir=prod_dir,
        is_active=True,
    )
    _add_model(
        session_factory,
        version="shadow-v2",
        stage="shadow",
        model_dir=shadow_dir,
    )
    manager, pipeline, _, _ = _manager(session_factory, current_version="prod-v1")

    state = manager.apply_model_release_state("cam-01")

    assert state["primary_model_version"] == "prod-v1"
    assert state["primary_model_stage"] == "production"
    assert state["primary_reloaded"] is False
    assert state["shadow_model_version"] == "shadow-v2"
    assert state["shadow_attached"] is True
    assert state["shadow_detached"] is False
    assert state["errors"] == []
    pipeline.reload_anomaly_model.assert_not_called()
    assert pipeline._shadow_runner.shadow_version_id == "shadow-v2"


def test_shadow_to_canary_reloads_primary_and_detaches_shadow(session_factory, tmp_path):
    prod_dir = _model_dir(tmp_path, "prod-v1")
    canary_dir = _model_dir(tmp_path, "canary-v2")
    _add_model(
        session_factory,
        version="prod-v1",
        stage="production",
        model_dir=prod_dir,
        is_active=True,
    )
    _add_model(
        session_factory,
        version="canary-v2",
        stage="canary",
        model_dir=canary_dir,
        canary_camera_id="cam-01",
    )
    old_shadow = MagicMock()
    old_shadow.shadow_version_id = "shadow-v2"
    manager, pipeline, _, _ = _manager(
        session_factory,
        current_version="prod-v1",
        shadow_runner=old_shadow,
    )

    state = manager.apply_model_release_state("cam-01")

    assert state["primary_model_version"] == "canary-v2"
    assert state["primary_model_stage"] == "canary"
    assert state["primary_reloaded"] is True
    assert state["shadow_model_version"] is None
    assert state["shadow_attached"] is False
    assert state["shadow_detached"] is True
    assert state["errors"] == []
    pipeline.reload_anomaly_model.assert_called_once_with(str(canary_dir / "model.xml"))
    old_shadow.flush.assert_called_once_with()


def test_canary_to_production_reloads_primary(session_factory, tmp_path):
    prod_dir = _model_dir(tmp_path, "prod-v2")
    _add_model(
        session_factory,
        version="prod-v2",
        stage="production",
        model_dir=prod_dir,
        is_active=True,
    )
    manager, pipeline, _, _ = _manager(session_factory, current_version="prod-v1")

    state = manager.apply_model_release_state("cam-01")

    assert state["primary_model_version"] == "prod-v2"
    assert state["primary_model_stage"] == "production"
    assert state["primary_reloaded"] is True
    assert state["shadow_attached"] is False
    assert state["shadow_detached"] is False
    assert state["errors"] == []
    pipeline.reload_anomaly_model.assert_called_once_with(str(prod_dir / "model.xml"))


def test_retire_without_production_reports_no_primary_and_detaches_shadow(
    session_factory, tmp_path
):
    retired_dir = _model_dir(tmp_path, "retired-v2")
    _add_model(
        session_factory,
        version="retired-v2",
        stage="retired",
        model_dir=retired_dir,
    )
    old_shadow = MagicMock()
    old_shadow.shadow_version_id = "shadow-v2"
    manager, pipeline, _, _ = _manager(
        session_factory,
        current_version="retired-v2",
        shadow_runner=old_shadow,
    )

    state = manager.apply_model_release_state("cam-01")

    assert state["primary_model_version"] is None
    assert state["primary_model_stage"] is None
    assert state["primary_reloaded"] is False
    assert state["shadow_model_version"] is None
    assert state["shadow_attached"] is False
    assert state["shadow_detached"] is True
    assert "primary_model_not_found" in state["errors"]
    pipeline.reload_anomaly_model.assert_not_called()
    old_shadow.flush.assert_called_once_with()
