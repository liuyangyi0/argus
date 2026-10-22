"""Runtime binding tests for model release-stage transitions."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from argus.capture.manager import CameraManager
from argus.config.schema import AlertConfig, CameraConfig, SceneGuardConfig
from argus.preprocessing.scene_profile import runtime_artifact_sha256
from argus.storage.models import (
    Base,
    ModelRecord,
    ModelReleaseOperation,
    ReleaseOperationStatus,
)
from argus.storage.release_evidence import canonical_json, canonical_sha256


@pytest.fixture()
def session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _model_dir(tmp_path: Path, name: str) -> Path:
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "model.xml").write_text("model", encoding="utf-8")
    (model_dir / "model.bin").write_bytes(b"weights")
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
    release_evidence_factory=None,
    authorize_runtime: bool = False,
) -> None:
    evidence = None
    if stage == "production" and release_evidence_factory is not None:
        evidence = release_evidence_factory(
            model_path=model_dir,
            model_version_id=version,
            registry_model_hash=f"hash-{version}",
            camera_id=camera_id,
        )
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
                release_evidence_json=(
                    canonical_json(evidence) if evidence is not None else None
                ),
                release_evidence_sha256=(
                    canonical_sha256(evidence) if evidence is not None else None
                ),
                release_revision=1 if authorize_runtime else 0,
            )
        )
        if authorize_runtime:
            artifact_sha256 = runtime_artifact_sha256(model_dir)
            session.add(
                ModelReleaseOperation(
                    operation_id=f"release-{version}",
                    model_version_id=version,
                    camera_id=camera_id,
                    from_stage="shadow",
                    target_stage=stage,
                    expected_revision=0,
                    applied_revision=1,
                    status=ReleaseOperationStatus.APPLIED.value,
                    requested_by="test-fixture",
                    runtime_state_json=json.dumps(
                        {
                            "identity": {
                                "camera_id": camera_id,
                                "model_version_id": version,
                                "target_stage": stage,
                                "runtime_artifact_sha256": artifact_sha256,
                            },
                        }
                    ),
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
        scene_guard=SceneGuardConfig(enabled=False),
    )
    db = SimpleNamespace(get_session=session_factory)
    manager = CameraManager([cam_config], AlertConfig(), database=db)
    pipeline = _PipelineDouble(cam_config, current_version, shadow_runner)
    runner = MagicMock()
    manager._pipelines["cam-01"] = pipeline
    manager._runners["cam-01"] = runner
    manager._notify_camera_status = MagicMock()
    return manager, pipeline, runner, cam_config


def test_startup_model_resolution_uses_canary_router(
    session_factory,
    tmp_path,
    release_evidence_factory,
    monkeypatch,
):
    prod_dir = _model_dir(tmp_path, "prod-v1")
    canary_dir = _model_dir(tmp_path, "canary-v2")
    _add_model(
        session_factory,
        version="prod-v1",
        stage="production",
        model_dir=prod_dir,
        is_active=True,
        release_evidence_factory=release_evidence_factory,
    )
    _add_model(
        session_factory,
        version="canary-v2",
        stage="canary",
        model_dir=canary_dir,
        canary_camera_id="cam-01",
        authorize_runtime=True,
    )
    unrelated_exports = tmp_path / "data" / "exports" / "cam-01"
    unrelated_exports.mkdir(parents=True)
    (unrelated_exports / "model.xml").write_text("unregistered", encoding="utf-8")
    (unrelated_exports / "model.bin").write_bytes(b"unregistered")
    monkeypatch.chdir(tmp_path)

    manager, _, _, cam_config = _manager(session_factory)
    monkeypatch.setattr(
        "argus.core.pipeline.DetectionPipeline._find_model_in_dir",
        lambda *_args, **_kwargs: pytest.fail(
            "registry-backed resolution must not scan exports or recurse"
        ),
    )

    resolved = manager._resolve_model_path(cam_config)

    assert resolved == canary_dir / "model.xml"
    assert manager._model_version_id(cam_config, resolved) == "canary-v2"


def test_startup_model_resolution_rejects_canary_without_applied_digest(
    session_factory,
    tmp_path,
):
    canary_dir = _model_dir(tmp_path, "canary-without-operation")
    _add_model(
        session_factory,
        version="canary-without-operation",
        stage="canary",
        model_dir=canary_dir,
        canary_camera_id="cam-01",
    )
    manager, _, _, cam_config = _manager(session_factory)

    assert manager._resolve_model_path(cam_config) is None
    assert manager._model_version_id(cam_config) == "patchcore:ssim_fallback"


def test_recovery_rejects_artifact_changed_after_applied_operation(
    session_factory,
    tmp_path,
):
    canary_dir = _model_dir(tmp_path, "canary-mutated")
    _add_model(
        session_factory,
        version="canary-mutated",
        stage="canary",
        model_dir=canary_dir,
        canary_camera_id="cam-01",
        authorize_runtime=True,
    )
    (canary_dir / "model.bin").write_bytes(b"tampered-after-release")
    manager, _, _, cam_config = _manager(session_factory)

    assert manager._resolve_model_path(cam_config) is None
    assert manager._resolve_authorized_model_identity("cam-01") is None


def test_registry_directory_rejects_nested_runtime_artifacts(
    session_factory,
    tmp_path,
):
    canary_dir = _model_dir(tmp_path, "canary-ambiguous")
    nested = canary_dir / "newer-export"
    nested.mkdir()
    (nested / "model.xml").write_text("unregistered", encoding="utf-8")
    (nested / "model.bin").write_bytes(b"unregistered")
    _add_model(
        session_factory,
        version="canary-ambiguous",
        stage="canary",
        model_dir=canary_dir,
        canary_camera_id="cam-01",
        authorize_runtime=True,
    )
    manager, _, _, cam_config = _manager(session_factory)

    # The persisted directory digest includes the nested model while the exact
    # root artifact digest does not, so ambiguity must fail closed.
    assert manager._resolve_model_path(cam_config) is None


def test_startup_model_resolution_rejects_legacy_active_production(
    session_factory,
    tmp_path,
):
    prod_dir = _model_dir(tmp_path, "legacy-prod")
    _add_model(
        session_factory,
        version="legacy-prod",
        stage="production",
        model_dir=prod_dir,
        is_active=True,
    )
    manager, _, _, cam_config = _manager(session_factory)

    assert manager._resolve_model_path(cam_config) is None
    assert manager._model_version_id(cam_config) == "patchcore:ssim_fallback"


def test_legacy_release_sync_cannot_attach_shadow(
    session_factory,
    tmp_path,
    release_evidence_factory,
):
    prod_dir = _model_dir(tmp_path, "prod-v1")
    shadow_dir = _model_dir(tmp_path, "shadow-v2")
    _add_model(
        session_factory,
        version="prod-v1",
        stage="production",
        model_dir=prod_dir,
        is_active=True,
        release_evidence_factory=release_evidence_factory,
    )
    _add_model(
        session_factory,
        version="shadow-v2",
        stage="shadow",
        model_dir=shadow_dir,
    )
    manager, pipeline, _, _ = _manager(session_factory, current_version="prod-v1")

    state = manager.apply_model_release_state("cam-01")

    assert state["applied"] is False
    assert state["primary_reloaded"] is False
    assert state["shadow_model_version"] is None
    assert state["shadow_attached"] is False
    assert state["shadow_detached"] is False
    assert state["errors"] == ["release_service_required"]
    pipeline.reload_anomaly_model.assert_not_called()
    assert pipeline._shadow_runner is None


def test_legacy_release_sync_cannot_reload_canary_or_detach_shadow(
    session_factory,
    tmp_path,
    release_evidence_factory,
):
    prod_dir = _model_dir(tmp_path, "prod-v1")
    canary_dir = _model_dir(tmp_path, "canary-v2")
    _add_model(
        session_factory,
        version="prod-v1",
        stage="production",
        model_dir=prod_dir,
        is_active=True,
        release_evidence_factory=release_evidence_factory,
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

    assert state["applied"] is False
    assert state["primary_reloaded"] is False
    assert state["shadow_model_version"] is None
    assert state["shadow_attached"] is False
    assert state["shadow_detached"] is False
    assert state["errors"] == ["release_service_required"]
    pipeline.reload_anomaly_model.assert_not_called()
    old_shadow.flush.assert_not_called()
    assert pipeline._shadow_runner is old_shadow


def test_legacy_release_sync_cannot_reload_production(
    session_factory,
    tmp_path,
    release_evidence_factory,
):
    prod_dir = _model_dir(tmp_path, "prod-v2")
    _add_model(
        session_factory,
        version="prod-v2",
        stage="production",
        model_dir=prod_dir,
        is_active=True,
        release_evidence_factory=release_evidence_factory,
    )
    manager, pipeline, _, _ = _manager(session_factory, current_version="prod-v1")

    state = manager.apply_model_release_state("cam-01")

    assert state["applied"] is False
    assert state["primary_reloaded"] is False
    assert state["shadow_attached"] is False
    assert state["shadow_detached"] is False
    assert state["errors"] == ["release_service_required"]
    pipeline.reload_anomaly_model.assert_not_called()


def test_legacy_release_sync_cannot_detach_shadow_for_retired_model(
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

    assert state["applied"] is False
    assert state["primary_model_version"] is None
    assert state["primary_model_stage"] is None
    assert state["primary_reloaded"] is False
    assert state["shadow_model_version"] is None
    assert state["shadow_attached"] is False
    assert state["shadow_detached"] is False
    assert state["errors"] == ["release_service_required"]
    pipeline.reload_anomaly_model.assert_not_called()
    old_shadow.flush.assert_not_called()
    assert pipeline._shadow_runner is old_shadow
