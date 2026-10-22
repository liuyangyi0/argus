"""Adversarial guards for one applied shadow/canary assignment per camera."""

import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from argus.anomaly.model_router import ModelRouter
from argus.capture.manager import CameraManager
from argus.config.schema import CameraConfig, SceneGuardConfig
from argus.storage.model_release_service import ModelReleaseService
from argus.storage.models import Base, ModelRecord, ModelReleaseOperation, ModelVersionEvent
from argus.storage.release_pipeline import ReleasePipeline, StageTransitionError


@pytest.fixture
def sessions():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine)
    engine.dispose()


def _add_model(sessions, version, stage, *, created_at=None):
    with sessions() as session:
        session.add(ModelRecord(
            model_version_id=version,
            camera_id="cam-1",
            model_type="patchcore",
            model_hash="a" * 64,
            data_hash="b" * 64,
            stage=stage,
            is_active=False,
            canary_camera_id="cam-1" if stage == "canary" else None,
            created_at=created_at or datetime.now(timezone.utc),
        ))
        if stage == "shadow":
            session.add(ModelVersionEvent(
                camera_id="cam-1",
                from_version=version,
                to_version=version,
                from_stage="candidate",
                to_stage="shadow",
                timestamp=datetime.now(timezone.utc) - timedelta(days=1),
                triggered_by="fixture",
            ))
        session.commit()


@pytest.mark.parametrize("target,source", [("shadow", "candidate"), ("canary", "shadow")])
def test_prepare_rejects_another_version_occupying_runtime_slot(sessions, target, source):
    _add_model(sessions, "already-applied", target)
    _add_model(sessions, "older-request", source)
    pipeline = ReleasePipeline(sessions, min_shadow_days=0, min_canary_days=0)

    with pytest.raises(StageTransitionError, match="runtime slot is occupied"):
        pipeline.prepare_transition(
            model_version_id="older-request",
            target_stage=target,
            canary_camera_id="cam-1" if target == "canary" else None,
        )


@pytest.mark.parametrize("target,source", [("shadow", "candidate"), ("canary", "shadow")])
def test_commit_rechecks_slot_after_prepare_and_rolls_back_cas(sessions, target, source):
    _add_model(sessions, "request", source)
    pipeline = ReleasePipeline(sessions, min_shadow_days=0, min_canary_days=0)
    plan = pipeline.prepare_transition(
        model_version_id="request",
        target_stage=target,
        canary_camera_id="cam-1" if target == "canary" else None,
    )
    _add_model(sessions, "won-during-runtime-prepare", target)

    with pytest.raises(StageTransitionError, match="runtime slot is occupied"):
        pipeline.commit_transition(plan, triggered_by="operator")

    with sessions() as session:
        requested = session.query(ModelRecord).filter_by(model_version_id="request").one()
        assert requested.stage == source
        assert requested.release_revision == 0


def test_two_prepared_shadow_assignments_cannot_both_commit(sessions):
    _add_model(sessions, "first", "candidate")
    _add_model(sessions, "second", "candidate")
    pipeline = ReleasePipeline(sessions)
    first = pipeline.prepare_transition(model_version_id="first", target_stage="shadow")
    second = pipeline.prepare_transition(model_version_id="second", target_stage="shadow")

    pipeline.commit_transition(first, triggered_by="operator")
    with pytest.raises(StageTransitionError, match="runtime slot is occupied"):
        pipeline.commit_transition(second, triggered_by="operator")


@pytest.mark.parametrize("stage", ["shadow", "canary"])
def test_router_refuses_legacy_duplicate_assignments_instead_of_picking_newest(sessions, stage):
    _add_model(sessions, "old", stage, created_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
    _add_model(sessions, "new", stage, created_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
    router = ModelRouter(sessions)

    if stage == "shadow":
        assert router.get_shadow_model_for_camera("cam-1") is None
    else:
        assert router.get_model_for_camera("cam-1") is None
        assert router.is_canary("cam-1") is False


@pytest.mark.parametrize("from_stage", ["shadow", "canary"])
def test_withdrawing_old_record_does_not_revoke_a_different_live_slot(sessions, from_stage):
    config = CameraConfig(
        camera_id="cam-1", name="camera", source="0", protocol="file",
        scene_guard=SceneGuardConfig(enabled=False),
    )
    live_shadow = SimpleNamespace(shadow_version_id="other-live-shadow")
    pipeline = SimpleNamespace(
        _model_version_id="other-live-primary",
        _anomaly_model_path=None,
        _scene_processing_lock=threading.RLock(),
        _shadow_runner_lock=threading.RLock(),
        get_shadow_runner=MagicMock(return_value=live_shadow),
        get_scene_guard_status=MagicMock(return_value={}),
        set_shadow_runner=MagicMock(),
    )
    manager = CameraManager.__new__(CameraManager)
    manager._pipelines = {"cam-1": pipeline}
    manager._runners = {}
    manager._cameras = [config]
    manager._db = SimpleNamespace(get_session=sessions)
    manager._get_active_model_record = MagicMock()
    prepared = manager.prepare_model_release(SimpleNamespace(
        camera_id="cam-1", model_version_id="old-record", target_stage="retired",
        from_stage=from_stage, runtime_artifact_sha256=None, expected_revision=0,
    ))

    assert prepared.success is True
    assert prepared.transaction.commit().success is True
    assert prepared.transaction.finalize().success is True
    pipeline.set_shadow_runner.assert_not_called()
    manager._get_active_model_record.assert_not_called()


def test_latest_operations_are_batched_and_tie_break_on_monotonic_id(sessions):
    same_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with sessions() as session:
        for operation_id, version, status in (
            ("old", "model-1", "failed"),
            ("latest", "model-1", "pending"),
            ("second-model", "model-2", "applied"),
        ):
            session.add(ModelReleaseOperation(
                operation_id=operation_id, model_version_id=version,
                camera_id="cam-1", from_stage="candidate", target_stage="shadow",
                expected_revision=0, requested_by="operator", status=status,
                requested_at=same_time,
            ))
            session.flush()
        session.commit()
    service = ModelReleaseService(
        sessions, ReleasePipeline(sessions), runtime_prepare=MagicMock(),
    )
    statements = []
    engine = sessions.kw["bind"]

    def count_select(_connection, _cursor, statement, *_args):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(engine, "before_cursor_execute", count_select)
    try:
        latest = service.get_latest_operations(["model-1", "model-2", "missing", "model-1"])
        assert latest["model-1"].operation_id == "latest"
        assert latest["model-2"].operation_id == "second-model"
        assert "missing" not in latest
        assert len(statements) == 1
        assert service.get_latest_operations([]) == {}
        assert len(statements) == 1
    finally:
        event.remove(engine, "before_cursor_execute", count_select)
