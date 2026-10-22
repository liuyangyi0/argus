"""ModelRegistry is registration/read-only at the release boundary."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from argus.storage import model_registry as registry_module
from argus.storage.model_registry import (
    ModelRegistry,
    ModelStageMutationDisabledError,
)
from argus.storage.models import AuditLog, Base, ModelRecord


@pytest.fixture
def registry():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return ModelRegistry(session_factory=sessionmaker(bind=engine))


@pytest.fixture
def model_dir(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    (path / "model.xml").write_text("fake model", encoding="utf-8")
    (path / "model.bin").write_bytes(b"fake weights")
    return path


@pytest.fixture
def baseline_dir(tmp_path):
    path = tmp_path / "baseline"
    path.mkdir()
    (path / "img.png").write_bytes(b"fake image")
    return path


def test_register_and_read_candidate(registry, model_dir, baseline_dir):
    version_id = registry.register(
        model_path=model_dir,
        baseline_dir=baseline_dir,
        camera_id="cam_01",
        model_type="patchcore",
        training_params={"epochs": 1},
    )

    record = registry.get_by_version_id(version_id)
    assert version_id.startswith("cam_01-patchcore-")
    assert record is not None
    assert record.stage == "candidate"
    assert record.is_active is False
    assert registry.get_active("cam_01") is None
    events = registry.get_version_events(model_version_id=version_id)
    assert len(events) == 1
    assert events[0].from_stage is None
    assert events[0].to_stage == "candidate"
    with registry._session_factory() as session:
        audit = session.query(AuditLog).filter_by(
            action="model_candidate_registered",
            target_id=version_id,
        ).one()
        assert audit.user == "model_registry"


def test_all_legacy_stage_mutators_fail_closed_without_state_change(
    registry,
    model_dir,
    baseline_dir,
):
    version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")

    calls = [
        lambda: registry.promote(version_id, "shadow", triggered_by="operator"),
        lambda: registry.activate(version_id, triggered_by="operator"),
        lambda: registry.retire(version_id, triggered_by="operator"),
        lambda: registry.rollback("cam_01", triggered_by="operator"),
        lambda: registry.compensate_failed_production_sync(
            version_id,
            None,
            triggered_by="operator",
            reason="runtime failed",
        ),
    ]

    for call in calls:
        with pytest.raises(
            ModelStageMutationDisabledError,
            match="ModelReleaseService",
        ):
            call()

    record = registry.get_by_version_id(version_id)
    assert record is not None
    assert record.stage == "candidate"
    assert record.is_active is False
    assert record.release_revision == 0
    events = registry.get_version_events(model_version_id=version_id)
    assert len(events) == 1
    assert events[0].to_stage == "candidate"
    assert events[0].triggered_by == "model_registry"


def test_legacy_active_row_without_evidence_is_not_runtime_active(
    registry,
    model_dir,
    baseline_dir,
):
    version_id = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")
    with registry._session_factory() as session:
        record = session.query(ModelRecord).filter_by(
            model_version_id=version_id
        ).one()
        record.stage = "production"
        record.is_active = True
        session.commit()

    assert registry.get_active("cam_01") is None


def test_model_hash_changes_on_retrain(registry, baseline_dir, tmp_path):
    first = tmp_path / "model1"
    first.mkdir()
    (first / "model.xml").write_text("version 1", encoding="utf-8")
    second = tmp_path / "model2"
    second.mkdir()
    (second / "model.xml").write_text("version 2", encoding="utf-8")

    first_id = registry.register(first, baseline_dir, "cam_01", "patchcore")
    second_id = registry.register(second, baseline_dir, "cam_01", "patchcore")

    assert first_id != second_id
    assert registry.get_by_version_id(first_id).model_hash != (
        registry.get_by_version_id(second_id).model_hash
    )


def test_register_retries_unique_id_collision_across_registry_instances(
    model_dir,
    baseline_dir,
    monkeypatch,
):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    class FixedDatetime:
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 5, 26, 4, 54, 0, tzinfo=tz or timezone.utc)

    suffixes = iter([
        "deadbeef000000000000000000000000",
        "deadbeef000000000000000000000000",
        "cafebabe000000000000000000000000",
    ])
    monkeypatch.setattr(registry_module, "datetime", FixedDatetime)
    monkeypatch.setattr(
        registry_module.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex=next(suffixes)),
    )

    first = ModelRegistry(session_factory=session_factory)
    second = ModelRegistry(session_factory=session_factory)
    first_id = first.register(model_dir, baseline_dir, "cam_01", "patchcore")
    second_id = second.register(model_dir, baseline_dir, "cam_01", "patchcore")

    assert first_id.endswith("deadbeef")
    assert second_id.endswith("cafebabe")
    assert len(first.list_models(camera_id="cam_01")) == 2


def test_list_models_filters_camera(registry, baseline_dir, tmp_path):
    first = tmp_path / "m1"
    first.mkdir()
    (first / "model.xml").write_text("cam01", encoding="utf-8")
    second = tmp_path / "m2"
    second.mkdir()
    (second / "model.xml").write_text("cam02", encoding="utf-8")
    registry.register(first, baseline_dir, "cam_01", "patchcore")
    registry.register(second, baseline_dir, "cam_02", "patchcore")

    assert len(registry.list_models()) == 2
    assert [row.camera_id for row in registry.list_models("cam_01")] == ["cam_01"]
    assert [row.camera_id for row in registry.list_models("cam_02")] == ["cam_02"]
    assert registry.list_models("cam_99") == []


def test_alert_carries_model_version():
    from argus.alerts.grader import Alert
    from argus.config.schema import AlertSeverity

    alert = Alert(
        alert_id="test",
        camera_id="cam_01",
        zone_id="z1",
        severity=AlertSeverity.INFO,
        anomaly_score=0.5,
        timestamp=0.0,
        frame_number=0,
        model_version_id="cam_01-patchcore-20260405",
    )
    assert alert.model_version_id == "cam_01-patchcore-20260405"
