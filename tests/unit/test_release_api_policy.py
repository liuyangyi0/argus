"""Focused API policy tests for model release gates."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from argus.config.loader import load_config
from argus.config.schema import DashboardConfig
from argus.dashboard.app import create_app
from argus.dashboard.routes.models import _get_release_anomaly_config
from argus.preprocessing.scene_profile import runtime_artifact_sha256
from argus.storage.database import Database
from argus.storage.model_registry import ModelRegistry
from argus.storage.model_release_service import (
    RuntimeReleaseCommit,
    RuntimeReleaseIdentity,
    RuntimeReleasePreparation,
)
from argus.storage.models import (
    ModelReleaseOperation,
    ModelStage,
    ReleaseOperationStatus,
)
from argus.storage.release_pipeline import ReleasePipeline


class LegacyCameraManager:
    def __init__(self) -> None:
        self._pipelines = {"cam_01": object()}
        self.reload_model = MagicMock(return_value=True)


class ReleaseStateCameraManager(LegacyCameraManager):
    def __init__(self) -> None:
        super().__init__()
        self.release_state_calls: list[str] = []

    def prepare_model_release(self, plan) -> RuntimeReleasePreparation:
        owner = self

        class Transaction:
            def commit(self) -> RuntimeReleaseCommit:
                owner.release_state_calls.append(plan.camera_id)
                return RuntimeReleaseCommit(
                    True,
                    state={
                        "camera_id": plan.camera_id,
                        "call_count": len(owner.release_state_calls),
                        "primary_model_version": (
                            plan.model_version_id
                            if plan.target_stage in {"canary", "production"}
                            else None
                        ),
                        "primary_model_stage": plan.target_stage,
                        "shadow_model_version": (
                            plan.model_version_id
                            if plan.target_stage == "shadow"
                            else None
                        ),
                        "errors": [],
                    },
                )

            def abort(self) -> RuntimeReleaseCommit:
                return RuntimeReleaseCommit(True, state={"restored": True})

            def finalize(self) -> RuntimeReleaseCommit:
                return RuntimeReleaseCommit(True, state={"released": True})

        return RuntimeReleasePreparation(
            True,
            identity=RuntimeReleaseIdentity(
                camera_id=plan.camera_id,
                model_version_id=plan.model_version_id,
                target_stage=plan.target_stage,
                runtime_artifact_sha256=plan.runtime_artifact_sha256,
            ),
            transaction=Transaction(),
        )


class ReadinessCameraManager(ReleaseStateCameraManager):
    """Small public CameraRuntime fake for release-readiness API tests."""

    def __init__(
        self,
        *,
        model_version_id: str,
        artifact_sha256: str,
        release_revision: int,
        scene_usable: bool = True,
    ) -> None:
        super().__init__()
        self._runtime_state = {
            "primary_model_version": model_version_id,
            "shadow_model_version": None,
            "runtime_artifact_sha256": artifact_sha256,
            "scene_profile_id": "scene-profile-approved",
            "release_revision": release_revision,
            # A readiness response must not pass arbitrary/path-bearing fields.
            "model_path": "C:/secret/runtime/model.pt",
        }
        self._scene_status = {
            "enabled": True,
            "usable": scene_usable,
            "state": "stable" if scene_usable else "mismatch",
            "profile_id": "scene-profile-approved",
            "bound_model_version": model_version_id,
            "bound_model_artifact_sha256": artifact_sha256,
        }

    def get_runtime_model_state(self, camera_id: str) -> dict:
        assert camera_id == "cam_01"
        return dict(self._runtime_state)

    def get_scene_guard_status(self, camera_id: str) -> dict:
        assert camera_id == "cam_01"
        return dict(self._scene_status)


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
    camera_manager._db = db
    app = create_app(
        database=db,
        camera_manager=camera_manager,
        alerts_dir=str(alerts_dir),
        config=SimpleNamespace(dashboard=DashboardConfig(go2rtc_enabled=False), cameras=[]),
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
    (model_dir / "model.bin").write_bytes(b"weights")

    baseline_dir = root / "baseline"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "img.png").write_bytes(b"img")
    return model_dir, baseline_dir


def _register_model(
    db,
    model_dir: Path,
    baseline_dir: Path,
    stage: str,
    release_evidence_factory=None,
) -> str:
    registry = ModelRegistry(session_factory=db.get_session)
    pipeline = ReleasePipeline(db.get_session, min_shadow_days=0, min_canary_days=0)
    version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")
    if stage in {
        ModelStage.SHADOW.value,
        ModelStage.CANARY.value,
        ModelStage.PRODUCTION.value,
    }:
        pipeline.transition(
            model_version_id=version_id,
            target_stage=ModelStage.SHADOW.value,
            triggered_by="test-fixture",
        )
    if stage in {ModelStage.CANARY.value, ModelStage.PRODUCTION.value}:
        pipeline.transition(
            model_version_id=version_id,
            target_stage=ModelStage.CANARY.value,
            triggered_by="test-fixture",
            canary_camera_id="cam_01",
        )
    if stage == ModelStage.PRODUCTION.value:
        registered = registry.get_by_version_id(version_id)
        pipeline.transition(
            model_version_id=version_id,
            target_stage=ModelStage.PRODUCTION.value,
            triggered_by="test-fixture",
            release_evidence=release_evidence_factory(
                model_path=model_dir,
                model_version_id=version_id,
                registry_model_hash=registered.model_hash,
            ),
        )
    return version_id


def _fixture_transition(
    db,
    version_id: str,
    target_stage: str,
    *,
    canary_camera_id: str | None = None,
    release_evidence=None,
):
    return ReleasePipeline(
        db.get_session,
        min_shadow_days=0,
        min_canary_days=0,
    ).transition(
        model_version_id=version_id,
        target_stage=target_stage,
        triggered_by="test-fixture",
        canary_camera_id=canary_camera_id,
        release_evidence=release_evidence,
    )


def test_release_pipeline_uses_camera_anomaly_stage_waits():
    config = load_config(Path("configs/default.yaml"))
    config.cameras[0].anomaly.min_shadow_days = 0
    config.cameras[0].anomaly.min_canary_days = 0

    anomaly_cfg = _get_release_anomaly_config(config)

    assert anomaly_cfg is config.cameras[0].anomaly
    assert anomaly_cfg.min_shadow_days == 0
    assert anomaly_cfg.min_canary_days == 0


@pytest.mark.parametrize("body", [None, [], "production"])
def test_promote_rejects_non_object_json(db, alerts_dir, body):
    client = TestClient(
        _make_app(db, ReleaseStateCameraManager(), alerts_dir)
    )

    if body is None:
        response = client.post(
            "/api/models/unused/promote",
            content="null",
            headers={"content-type": "application/json"},
        )
    else:
        response = client.post("/api/models/unused/promote", json=body)

    assert response.status_code == 422
    assert response.json()["code"] == 42200


def test_promote_shadow_canary_then_v2_baseline_keeps_production_no_go(
    db,
    alerts_dir,
    tmp_path,
    release_baseline_factory,
    release_evidence_sources_factory,
    running_release_client,
    await_release_operation,
):
    model_dir, baseline_dir = _create_model_tree(tmp_path)
    baseline_dir = release_baseline_factory(root=baseline_dir)
    registry = ModelRegistry(session_factory=db.get_session)
    version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")

    camera_manager = ReleaseStateCameraManager()
    client = running_release_client(_make_app(db, camera_manager, alerts_dir))
    registered = registry.get_by_version_id(version_id)
    release_evidence_sources = release_evidence_sources_factory(
        root=tmp_path / "release-sources",
        model_path=model_dir,
        model_version_id=version_id,
        registry_model_hash=registered.model_hash,
        baseline_path=baseline_dir,
    )

    promotions = [
        {"target_stage": "shadow", "triggered_by": "tester"},
        {
            "target_stage": "canary",
            "triggered_by": "tester",
            "canary_camera_id": "cam_01",
        },
    ]

    for expected_call_count, body in enumerate(promotions, start=1):
        response = client.post(f"/api/models/{version_id}/promote", json=body)

        assert response.status_code == 202
        data = await_release_operation(client, response)
        assert data["runtime_synced"] is True
        assert data["runtime_state"]["camera_id"] == "cam_01"
        assert data["runtime_state"]["call_count"] == expected_call_count

    production_response = client.post(
        f"/api/models/{version_id}/promote",
        json={
            "target_stage": "production",
            "release_evidence_sources": release_evidence_sources,
        },
    )

    failed = await_release_operation(client, production_response, status="failed")
    assert "decoder evidence is insufficient" in failed["operation"]["error"]
    assert registry.get_by_version_id(version_id).stage == "canary"
    assert camera_manager.release_state_calls == ["cam_01", "cam_01"]


@pytest.mark.parametrize(
    "release_evidence",
    [None, {"approved": True}, {"operator_approval": {"operator_approved": 1}}],
)
def test_production_api_rejects_missing_or_client_supplied_evidence(
    db,
    alerts_dir,
    tmp_path,
    release_evidence,
):
    model_dir, baseline_dir = _create_model_tree(tmp_path)
    registry = ModelRegistry(session_factory=db.get_session)
    version_id = registry.register(
        model_dir,
        baseline_dir,
        "cam_01",
        "patchcore",
    )
    _fixture_transition(db, version_id, "shadow")
    _fixture_transition(
        db,
        version_id,
        "canary",
        canary_camera_id="cam_01",
    )
    camera_manager = ReleaseStateCameraManager()
    client = TestClient(_make_app(db, camera_manager, alerts_dir))
    body = {
        "target_stage": "production",
        "triggered_by": "tester",
        # The route must also ignore this attempted HTTP recovery bypass.
        "allow_bypass": True,
    }
    if release_evidence is not None:
        body["release_evidence"] = release_evidence

    response = client.post(f"/api/models/{version_id}/promote", json=body)

    assert response.status_code == 400
    record = registry.get_by_version_id(version_id)
    assert record.stage == "canary"
    assert record.is_active is False
    assert record.release_evidence_json is None
    assert camera_manager.release_state_calls == []


def test_production_api_rejects_even_structurally_valid_client_bundle(
    db,
    alerts_dir,
    tmp_path,
    release_evidence_factory,
):
    model_dir, baseline_dir = _create_model_tree(tmp_path)
    registry = ModelRegistry(session_factory=db.get_session)
    version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")
    _fixture_transition(db, version_id, "shadow")
    _fixture_transition(
        db,
        version_id,
        "canary",
        canary_camera_id="cam_01",
    )
    registered = registry.get_by_version_id(version_id)
    evidence = release_evidence_factory(
        model_path=model_dir,
        model_version_id=version_id,
        registry_model_hash=registered.model_hash,
    )
    camera_manager = ReleaseStateCameraManager()
    client = TestClient(_make_app(db, camera_manager, alerts_dir))

    response = client.post(
        f"/api/models/{version_id}/promote",
        json={"target_stage": "production", "release_evidence": evidence},
    )

    assert response.status_code == 400
    assert "not accepted" in response.json()["msg"]
    assert registry.get_by_version_id(version_id).stage == "canary"
    assert camera_manager.release_state_calls == []


def test_promote_ignores_client_triggered_by_and_uses_authenticated_user(
    db,
    alerts_dir,
    tmp_path,
    running_release_client,
    await_release_operation,
):
    model_dir, baseline_dir = _create_model_tree(tmp_path)
    registry = ModelRegistry(session_factory=db.get_session)
    version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")

    camera_manager = ReleaseStateCameraManager()
    client = running_release_client(_make_app(db, camera_manager, alerts_dir))

    response = client.post(
        f"/api/models/{version_id}/promote",
        json={"target_stage": "shadow", "triggered_by": "forged-user"},
    )

    assert response.status_code == 202
    await_release_operation(client, response)
    event = registry.get_version_events(model_version_id=version_id, limit=1)[0]
    assert event.to_stage == "shadow"
    assert event.triggered_by == "system"


@pytest.mark.parametrize("operation", ["reexport", "recalibrate"])
def test_released_artifacts_derive_new_candidates_without_mutating_source(
    db,
    alerts_dir,
    tmp_path,
    operation,
    monkeypatch,
):
    model_dir, baseline_dir = _create_model_tree(tmp_path)
    registry = ModelRegistry(session_factory=db.get_session)
    version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")
    _fixture_transition(db, version_id, "shadow")
    _fixture_transition(
        db,
        version_id,
        "canary",
        canary_camera_id="cam_01",
    )
    app = _make_app(db, ReleaseStateCameraManager(), alerts_dir)
    trainer = MagicMock()
    getattr(trainer, f"{operation}_model").return_value = {
        "status": "ok", "model_version_id": "new-candidate", "model_path": "new/model.xml",
    }
    app.state.model_trainer = trainer
    baseline_manager = MagicMock()
    baseline_manager.get_baseline_dir.return_value = baseline_dir
    app.state.baseline_manager = baseline_manager
    client = TestClient(app)
    original = (model_dir / "model.xml").read_bytes()

    response = client.post(f"/api/models/{version_id}/{operation}", json={})

    assert response.status_code == 200
    assert response.json()["data"]["model_version_id"] == "new-candidate"
    assert registry.get_by_version_id(version_id).stage == "canary"
    assert (model_dir / "model.xml").read_bytes() == original
    getattr(trainer, f"{operation}_model").assert_called_once()


def test_shadow_report_includes_gate_report(db, alerts_dir):
    camera_manager = ReleaseStateCameraManager()
    client = TestClient(_make_app(db, camera_manager, alerts_dir))

    response = client.get("/api/models/shadow-v1/shadow-report")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total_samples"] == 0
    assert data["gate_report"]["ready_for_canary"] is True
    assert any(
        check["name"] == "min_shadow_samples"
        for check in data["gate_report"]["checks"]
    )


def test_inactive_candidate_retire_skips_runtime_and_returns_audited_state(
    db, alerts_dir, tmp_path,
    running_release_client,
    await_release_operation,
):
    model_dir, baseline_dir = _create_model_tree(tmp_path)
    registry = ModelRegistry(session_factory=db.get_session)
    version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")

    camera_manager = ReleaseStateCameraManager()
    client = running_release_client(_make_app(db, camera_manager, alerts_dir))

    response = client.post(
        f"/api/models/{version_id}/retire",
        json={"triggered_by": "tester"},
    )

    assert response.status_code == 202
    data = await_release_operation(client, response)
    assert data["model"]["stage"] == "retired"
    assert data["runtime_synced"] is True
    assert data["runtime_state"]["runtime_impact"] == "none"
    assert data["runtime_state"]["reason"] == "inactive_candidate_retirement"
    assert camera_manager.release_state_calls == []


def test_rollback_legacy_endpoint_fails_closed_without_state_change(
    db,
    alerts_dir,
    tmp_path,
    release_evidence_factory,
):
    model_dir_1, baseline_dir_1 = _create_model_tree(tmp_path / "v1")
    version_id_1 = _register_model(
        db,
        model_dir_1,
        baseline_dir_1,
        ModelStage.PRODUCTION.value,
        release_evidence_factory,
    )
    model_dir_2, baseline_dir_2 = _create_model_tree(tmp_path / "v2")
    version_id_2 = _register_model(
        db,
        model_dir_2,
        baseline_dir_2,
        ModelStage.PRODUCTION.value,
        release_evidence_factory,
    )

    camera_manager = LegacyCameraManager()
    client = TestClient(_make_app(db, camera_manager, alerts_dir))

    response = client.post(f"/api/models/{version_id_2}/rollback")

    assert response.status_code == 409
    assert "回滚入口已安全关闭" in response.json()["msg"]
    registry = ModelRegistry(session_factory=db.get_session)
    assert registry.get_active("cam_01").model_version_id == version_id_2
    assert registry.get_by_version_id(version_id_1).stage == "retired"
    camera_manager.reload_model.assert_not_called()


def test_rollback_api_fails_closed_when_previous_sidecar_changed(
    db,
    alerts_dir,
    tmp_path,
    release_evidence_factory,
):
    model_dir_1, baseline_dir_1 = _create_model_tree(tmp_path / "v1")
    _register_model(
        db,
        model_dir_1,
        baseline_dir_1,
        ModelStage.PRODUCTION.value,
        release_evidence_factory,
    )
    model_dir_2, baseline_dir_2 = _create_model_tree(tmp_path / "v2")
    version_id_2 = _register_model(
        db,
        model_dir_2,
        baseline_dir_2,
        ModelStage.PRODUCTION.value,
        release_evidence_factory,
    )
    (model_dir_1 / "model.bin").write_bytes(b"changed after approval")
    camera_manager = LegacyCameraManager()
    client = TestClient(_make_app(db, camera_manager, alerts_dir))

    response = client.post(f"/api/models/{version_id_2}/rollback")

    assert response.status_code == 409
    assert "回滚入口已安全关闭" in response.json()["msg"]
    camera_manager.reload_model.assert_not_called()


def test_rollback_does_not_attempt_runtime_or_compensation(
    db,
    alerts_dir,
    tmp_path,
    release_evidence_factory,
):
    model_dir_1, baseline_dir_1 = _create_model_tree(tmp_path / "v1")
    version_id_1 = _register_model(
        db,
        model_dir_1,
        baseline_dir_1,
        ModelStage.PRODUCTION.value,
        release_evidence_factory,
    )
    model_dir_2, baseline_dir_2 = _create_model_tree(tmp_path / "v2")
    version_id_2 = _register_model(
        db,
        model_dir_2,
        baseline_dir_2,
        ModelStage.PRODUCTION.value,
        release_evidence_factory,
    )
    camera_manager = LegacyCameraManager()
    camera_manager.reload_model.side_effect = [False, True]
    client = TestClient(_make_app(db, camera_manager, alerts_dir))

    response = client.post(f"/api/models/{version_id_2}/rollback")

    assert response.status_code == 409
    registry = ModelRegistry(session_factory=db.get_session)
    active = registry.get_active("cam_01")
    assert active is not None
    assert active.model_version_id == version_id_2
    unchanged_target = registry.get_by_version_id(version_id_1)
    assert unchanged_target.stage == ModelStage.RETIRED.value
    assert unchanged_target.is_active is False
    assert camera_manager.reload_model.call_count == 0


def test_legacy_runtime_sync_is_rejected_without_registry_mutation(
    db,
    alerts_dir,
    tmp_path,
    monkeypatch,
    release_evidence_factory,
    release_baseline_factory,
    release_evidence_sources_factory,
    running_release_client,
    await_release_operation,
):
    old_dir, old_baseline = _create_model_tree(tmp_path / "old")
    old_id = _register_model(
        db,
        old_dir,
        old_baseline,
        ModelStage.PRODUCTION.value,
        release_evidence_factory,
    )
    new_dir, new_baseline = _create_model_tree(tmp_path / "new")
    new_baseline = release_baseline_factory(root=new_baseline)
    registry = ModelRegistry(session_factory=db.get_session)
    new_id = registry.register(new_dir, new_baseline, "cam_01", "patchcore")
    _fixture_transition(db, new_id, "shadow")
    _fixture_transition(
        db,
        new_id,
        "canary",
        canary_camera_id="cam_01",
    )
    registered = registry.get_by_version_id(new_id)
    trusted_evidence = release_evidence_factory(
        model_path=new_dir,
        model_version_id=new_id,
        registry_model_hash=registered.model_hash,
    )
    monkeypatch.setattr(
        "argus.storage.release_evidence.build_release_evidence_from_sources",
        lambda *_args, **_kwargs: trusted_evidence,
    )
    sources = release_evidence_sources_factory(
        root=tmp_path / "release-sources",
        model_path=new_dir,
        model_version_id=new_id,
        registry_model_hash=registered.model_hash,
        baseline_path=new_baseline,
    )
    camera_manager = LegacyCameraManager()
    camera_manager.reload_model.side_effect = [False, True]
    client = running_release_client(_make_app(db, camera_manager, alerts_dir))

    response = client.post(
        f"/api/models/{new_id}/promote",
        json={
            "target_stage": "production",
            "release_evidence_sources": sources,
        },
    )

    await_release_operation(client, response, status="failed")
    active = registry.get_active("cam_01")
    assert active is not None
    assert active.model_version_id == old_id
    failed = registry.get_by_version_id(new_id)
    assert failed.stage == ModelStage.CANARY.value
    assert failed.is_active is False
    assert failed.release_evidence_json is None
    camera_manager.reload_model.assert_not_called()


def test_legacy_runtime_callback_cannot_mutate_previous_artifact(
    db,
    alerts_dir,
    tmp_path,
    monkeypatch,
    release_evidence_factory,
    release_baseline_factory,
    release_evidence_sources_factory,
    running_release_client,
    await_release_operation,
):
    old_dir, old_baseline = _create_model_tree(tmp_path / "old")
    old_id = _register_model(
        db,
        old_dir,
        old_baseline,
        ModelStage.PRODUCTION.value,
        release_evidence_factory,
    )
    new_dir, new_baseline = _create_model_tree(tmp_path / "new")
    new_baseline = release_baseline_factory(root=new_baseline)
    registry = ModelRegistry(session_factory=db.get_session)
    new_id = registry.register(new_dir, new_baseline, "cam_01", "patchcore")
    _fixture_transition(db, new_id, "shadow")
    _fixture_transition(
        db,
        new_id,
        "canary",
        canary_camera_id="cam_01",
    )
    registered = registry.get_by_version_id(new_id)
    trusted_evidence = release_evidence_factory(
        model_path=new_dir,
        model_version_id=new_id,
        registry_model_hash=registered.model_hash,
    )
    monkeypatch.setattr(
        "argus.storage.release_evidence.build_release_evidence_from_sources",
        lambda *_args, **_kwargs: trusted_evidence,
    )
    sources = release_evidence_sources_factory(
        root=tmp_path / "release-sources",
        model_path=new_dir,
        model_version_id=new_id,
        registry_model_hash=registered.model_hash,
        baseline_path=new_baseline,
    )

    def fail_after_staling_previous(*_args, **_kwargs):
        (old_dir / "model.bin").write_bytes(b"stale during runtime switch")
        return False

    camera_manager = LegacyCameraManager()
    camera_manager.reload_model.side_effect = fail_after_staling_previous
    client = running_release_client(_make_app(db, camera_manager, alerts_dir))

    response = client.post(
        f"/api/models/{new_id}/promote",
        json={
            "target_stage": "production",
            "release_evidence_sources": sources,
        },
    )

    await_release_operation(client, response, status="failed")
    active = registry.get_active("cam_01")
    assert active is not None
    assert active.model_version_id == old_id
    failed = registry.get_by_version_id(new_id)
    assert failed.stage == ModelStage.CANARY.value
    assert failed.is_active is False
    assert failed.release_evidence_json is None
    camera_manager.reload_model.assert_not_called()
    assert (old_dir / "model.bin").read_bytes() != b"stale during runtime switch"


@pytest.mark.parametrize(
    "stage",
    [
        None,
        ModelStage.CANDIDATE.value,
        ModelStage.SHADOW.value,
        ModelStage.CANARY.value,
    ],
)
def test_baseline_deploy_rejects_unregistered_and_closes_registered_stages(
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

    assert response.status_code == (400 if stage is None else 409)
    if stage is not None:
        assert "路径部署入口已关闭" in response.json()["msg"]
    camera_manager.reload_model.assert_not_called()


def test_canary_camera_mismatch_returns_400(db, alerts_dir, tmp_path, running_release_client):
    model_dir, baseline_dir = _create_model_tree(tmp_path)
    registry = ModelRegistry(session_factory=db.get_session)
    version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")
    _fixture_transition(db, version_id, ModelStage.SHADOW.value)

    camera_manager = ReleaseStateCameraManager()
    client = running_release_client(_make_app(db, camera_manager, alerts_dir))

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


def test_active_production_cannot_be_retired_directly_via_api(
    db,
    alerts_dir,
    tmp_path,
    release_evidence_factory,
):
    model_dir, baseline_dir = _create_model_tree(tmp_path)
    version_id = _register_model(
        db,
        model_dir,
        baseline_dir,
        ModelStage.PRODUCTION.value,
        release_evidence_factory,
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
    release_evidence_factory,
):
    model_dir, baseline_dir = _create_model_tree(tmp_path)
    version_id = _register_model(
        db,
        model_dir,
        baseline_dir,
        ModelStage.PRODUCTION.value,
        release_evidence_factory,
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


def test_release_readiness_returns_404_for_unknown_model(db, alerts_dir):
    client = TestClient(_make_app(db, LegacyCameraManager(), alerts_dir))

    response = client.get("/api/models/missing-model/release-readiness")

    assert response.status_code == 404
    assert response.json()["code"] == 40400


def test_canary_release_readiness_is_explicit_no_go_without_server_evidence(
    db,
    alerts_dir,
    tmp_path,
):
    model_dir, baseline_dir = _create_model_tree(tmp_path)
    registry = ModelRegistry(session_factory=db.get_session)
    version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")
    _fixture_transition(db, version_id, ModelStage.SHADOW.value)
    _fixture_transition(
        db,
        version_id,
        ModelStage.CANARY.value,
        canary_camera_id="cam_01",
    )
    model = registry.get_by_version_id(version_id)
    artifact_sha256 = runtime_artifact_sha256(model_dir)
    manager = ReadinessCameraManager(
        model_version_id=version_id,
        artifact_sha256=artifact_sha256,
        release_revision=model.release_revision,
    )
    client = TestClient(_make_app(db, manager, alerts_dir))

    response = client.get(f"/api/models/{version_id}/release-readiness")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["model_version_id"] == version_id
    assert data["camera_id"] == "cam_01"
    assert data["target_stage"] == ModelStage.PRODUCTION.value
    assert data["status"] == "no_go"
    assert data["production_ready"] is False
    assert data["promotion_allowed"] is False
    assert data["desired_stage"] is None
    assert data["applied_stage"] == ModelStage.CANARY.value
    gates = {gate["key"]: gate for gate in data["gates"]}
    assert gates["artifact_sha256"]["passed"] is True
    assert gates["two_phase_runtime"]["passed"] is True
    assert gates["scene_guard_usable"]["passed"] is True
    assert gates["baseline_v3"]["passed"] is False
    assert gates["regression_metrics"]["passed"] is False
    assert gates["trusted_soak"]["passed"] is False
    assert any("not reconstructed" in reason for reason in data["reasons"])
    assert "model_path" not in response.text
    assert str(tmp_path) not in response.text


def test_release_readiness_reports_latest_pending_then_failed_operation(
    db,
    alerts_dir,
    tmp_path,
):
    model_dir, baseline_dir = _create_model_tree(tmp_path)
    registry = ModelRegistry(session_factory=db.get_session)
    version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")
    _fixture_transition(db, version_id, ModelStage.SHADOW.value)
    _fixture_transition(
        db,
        version_id,
        ModelStage.CANARY.value,
        canary_camera_id="cam_01",
    )
    with db.get_session() as session:
        session.add(ModelReleaseOperation(
            operation_id="release-readiness-pending",
            model_version_id=version_id,
            camera_id="cam_01",
            from_stage=ModelStage.CANARY.value,
            target_stage=ModelStage.PRODUCTION.value,
            expected_revision=2,
            status=ReleaseOperationStatus.PENDING.value,
            requested_by="operator",
        ))
        session.commit()

    client = TestClient(_make_app(db, LegacyCameraManager(), alerts_dir))
    pending_response = client.get(
        f"/api/models/{version_id}/release-readiness"
    )

    assert pending_response.status_code == 200
    pending = pending_response.json()["data"]
    assert pending["status"] == "pending"
    assert pending["operation_id"] == "release-readiness-pending"
    assert pending["desired_stage"] == ModelStage.PRODUCTION.value
    assert pending["desired_status"] == ReleaseOperationStatus.PENDING.value

    with db.get_session() as session:
        operation = session.query(ModelReleaseOperation).filter_by(
            operation_id="release-readiness-pending",
        ).one()
        operation.status = ReleaseOperationStatus.FAILED.value
        operation.error = "injected test failure"
        session.commit()

    failed_response = client.get(f"/api/models/{version_id}/release-readiness")

    assert failed_response.status_code == 200
    failed = failed_response.json()["data"]
    assert failed["status"] == "failed"
    assert failed["desired_status"] == ReleaseOperationStatus.FAILED.value
    assert any("latest production release operation failed" in reason.lower()
               for reason in failed["reasons"])


def test_applied_production_readiness_revalidates_evidence_and_runtime_identity(
    db,
    alerts_dir,
    tmp_path,
    release_evidence_factory,
):
    model_dir, baseline_dir = _create_model_tree(tmp_path)
    version_id = _register_model(
        db,
        model_dir,
        baseline_dir,
        ModelStage.PRODUCTION.value,
        release_evidence_factory,
    )
    registry = ModelRegistry(session_factory=db.get_session)
    model = registry.get_by_version_id(version_id)
    artifact_sha256 = runtime_artifact_sha256(model_dir)
    manager = ReadinessCameraManager(
        model_version_id=version_id,
        artifact_sha256=artifact_sha256,
        release_revision=model.release_revision,
    )
    client = TestClient(_make_app(db, manager, alerts_dir))

    response = client.get(f"/api/models/{version_id}/release-readiness")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "applied"
    assert data["applied_stage"] == ModelStage.PRODUCTION.value
    assert data["applied_revision"] == model.release_revision
    assert data["production_ready"] is True
    assert data["promotion_allowed"] is False
    assert data["reasons"] == []
    assert all(gate["passed"] is True for gate in data["gates"])
    assert data["runtime_state"]["primary_model_version"] == version_id
    assert "model_path" not in data["runtime_state"]
    assert str(tmp_path) not in response.text


def test_applied_production_keeps_applied_status_when_scene_guard_is_unsafe(
    db,
    alerts_dir,
    tmp_path,
    release_evidence_factory,
):
    model_dir, baseline_dir = _create_model_tree(tmp_path)
    version_id = _register_model(
        db,
        model_dir,
        baseline_dir,
        ModelStage.PRODUCTION.value,
        release_evidence_factory,
    )
    model = ModelRegistry(session_factory=db.get_session).get_by_version_id(version_id)
    manager = ReadinessCameraManager(
        model_version_id=version_id,
        artifact_sha256=runtime_artifact_sha256(model_dir),
        release_revision=model.release_revision,
        scene_usable=False,
    )
    client = TestClient(_make_app(db, manager, alerts_dir))

    response = client.get(f"/api/models/{version_id}/release-readiness")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "applied"
    assert data["applied_stage"] == ModelStage.PRODUCTION.value
    assert data["production_ready"] is False
    gates = {gate["key"]: gate for gate in data["gates"]}
    assert gates["scene_guard_usable"]["passed"] is False
