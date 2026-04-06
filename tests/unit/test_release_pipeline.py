"""Tests for the four-stage model release pipeline."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from argus.storage.models import (
    Base,
    ModelRecord,
    ModelStage,
    ModelVersionEvent,
    ShadowInferenceLog,
)
from argus.storage.release_pipeline import (
    BackboneIncompatibleError,
    ReleasePipeline,
    StageTransitionError,
)


@pytest.fixture()
def session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    return factory


@pytest.fixture()
def pipeline(session_factory):
    return ReleasePipeline(
        session_factory, min_shadow_days=3, min_canary_days=7
    )


def _create_model(session_factory, camera_id="cam-01", stage="candidate", **kwargs):
    """Helper to create a ModelRecord."""
    defaults = {
        "model_version_id": f"{camera_id}-dinomaly2-{stage}-0001",
        "camera_id": camera_id,
        "model_type": "dinomaly2",
        "model_hash": "abcd1234",
        "data_hash": "efgh5678",
        "stage": stage,
        "component_type": "full",
    }
    defaults.update(kwargs)
    with session_factory() as session:
        record = ModelRecord(**defaults)
        session.add(record)
        session.commit()
        return defaults["model_version_id"]


class TestStageTransitions:
    """Test valid and invalid stage transitions."""

    def test_candidate_to_shadow(self, session_factory, pipeline):
        vid = _create_model(session_factory)
        result = pipeline.transition(vid, "shadow", "engineer_a", "starting shadow evaluation")
        assert result.stage == "shadow"

    def test_shadow_to_canary(self, session_factory, pipeline):
        vid = _create_model(session_factory, stage="shadow")
        # Need a shadow entry event with enough time elapsed
        now = datetime.now(timezone.utc)
        shadow_start = now - timedelta(days=5)
        with session_factory() as session:
            event = ModelVersionEvent(
                timestamp=shadow_start,
                camera_id="cam-01",
                from_version=vid,
                to_version=vid,
                from_stage="candidate",
                to_stage="shadow",
                triggered_by="engineer_a",
            )
            session.add(event)
            session.commit()

        result = pipeline.transition(
            vid, "canary", "engineer_b", canary_camera_id="cam-01", now=now,
        )
        assert result.stage == "canary"
        assert result.canary_camera_id == "cam-01"

    def test_canary_to_production(self, session_factory, pipeline):
        vid = _create_model(session_factory, stage="canary", canary_camera_id="cam-01")
        now = datetime.now(timezone.utc)
        canary_start = now - timedelta(days=10)
        with session_factory() as session:
            event = ModelVersionEvent(
                timestamp=canary_start,
                camera_id="cam-01",
                from_version=vid,
                to_version=vid,
                from_stage="shadow",
                to_stage="canary",
                triggered_by="engineer_b",
            )
            session.add(event)
            session.commit()

        result = pipeline.transition(vid, "production", "engineer_c", now=now)
        assert result.stage == "production"
        assert result.is_active is True
        assert result.canary_camera_id is None

    def test_any_to_retired(self, session_factory, pipeline):
        for stage in ["candidate", "shadow", "canary", "production"]:
            vid = _create_model(
                session_factory,
                stage=stage,
                model_version_id=f"cam-01-dinomaly2-{stage}-retire",
            )
            result = pipeline.transition(vid, "retired", "engineer_d")
            assert result.stage == "retired"
            assert result.is_active is False

    def test_shadow_pullback_to_candidate(self, session_factory, pipeline):
        vid = _create_model(session_factory, stage="shadow")
        result = pipeline.transition(vid, "candidate", "engineer_a", "bad shadow results")
        assert result.stage == "candidate"

    def test_canary_pullback_to_shadow(self, session_factory, pipeline):
        vid = _create_model(session_factory, stage="canary", canary_camera_id="cam-01")
        result = pipeline.transition(vid, "shadow", "engineer_a", "canary issues")
        assert result.stage == "shadow"


class TestInvalidTransitions:
    """Test that invalid transitions are rejected."""

    def test_candidate_to_production_skips_stages(self, session_factory, pipeline):
        vid = _create_model(session_factory)
        with pytest.raises(StageTransitionError, match="Invalid transition"):
            pipeline.transition(vid, "production", "engineer_a")

    def test_candidate_to_canary_skips_shadow(self, session_factory, pipeline):
        vid = _create_model(session_factory)
        with pytest.raises(StageTransitionError, match="Invalid transition"):
            pipeline.transition(vid, "canary", "engineer_a", canary_camera_id="cam-01")

    def test_production_to_shadow_not_allowed(self, session_factory, pipeline):
        vid = _create_model(session_factory, stage="production")
        with pytest.raises(StageTransitionError, match="Invalid transition"):
            pipeline.transition(vid, "shadow", "engineer_a")

    def test_retired_cannot_transition(self, session_factory, pipeline):
        vid = _create_model(session_factory, stage="retired")
        with pytest.raises(StageTransitionError, match="Invalid transition"):
            pipeline.transition(vid, "candidate", "engineer_a")


class TestTimingValidation:
    """Test minimum duration enforcement."""

    def test_shadow_too_short_for_canary(self, session_factory, pipeline):
        vid = _create_model(session_factory, stage="shadow")
        now = datetime.now(timezone.utc)
        # Shadow started only 1 day ago
        with session_factory() as session:
            event = ModelVersionEvent(
                timestamp=now - timedelta(days=1),
                camera_id="cam-01",
                from_version=vid,
                to_version=vid,
                from_stage="candidate",
                to_stage="shadow",
                triggered_by="engineer_a",
            )
            session.add(event)
            session.commit()

        with pytest.raises(StageTransitionError, match="Shadow period too short"):
            pipeline.transition(vid, "canary", "engineer_b", canary_camera_id="cam-01", now=now)

    def test_canary_too_short_for_production(self, session_factory, pipeline):
        vid = _create_model(session_factory, stage="canary", canary_camera_id="cam-01")
        now = datetime.now(timezone.utc)
        # Canary started only 2 days ago
        with session_factory() as session:
            event = ModelVersionEvent(
                timestamp=now - timedelta(days=2),
                camera_id="cam-01",
                from_version=vid,
                to_version=vid,
                from_stage="shadow",
                to_stage="canary",
                triggered_by="engineer_b",
            )
            session.add(event)
            session.commit()

        with pytest.raises(StageTransitionError, match="Canary period too short"):
            pipeline.transition(vid, "production", "engineer_c", now=now)

    def test_canary_requires_camera_id(self, session_factory, pipeline):
        vid = _create_model(session_factory, stage="shadow")
        now = datetime.now(timezone.utc)
        with session_factory() as session:
            event = ModelVersionEvent(
                timestamp=now - timedelta(days=5),
                camera_id="cam-01",
                from_version=vid,
                to_version=vid,
                from_stage="candidate",
                to_stage="shadow",
                triggered_by="engineer_a",
            )
            session.add(event)
            session.commit()

        with pytest.raises(ValueError, match="canary_camera_id required"):
            pipeline.transition(vid, "canary", "engineer_b", now=now)


class TestVersionEvents:
    """Test that version events are recorded correctly."""

    def test_transition_creates_event(self, session_factory, pipeline):
        vid = _create_model(session_factory)
        pipeline.transition(vid, "shadow", "engineer_a", "testing")

        with session_factory() as session:
            events = session.query(ModelVersionEvent).all()
            assert len(events) == 1
            e = events[0]
            assert e.from_stage == "candidate"
            assert e.to_stage == "shadow"
            assert e.triggered_by == "engineer_a"
            assert e.reason == "testing"

    def test_production_retires_old_model(self, session_factory, pipeline):
        # Create an existing production model
        old_vid = _create_model(
            session_factory,
            stage="production",
            model_version_id="cam-01-dinomaly2-old-0001",
            is_active=True,
        )

        # Create a canary model ready for production
        new_vid = _create_model(
            session_factory,
            stage="canary",
            model_version_id="cam-01-dinomaly2-new-0002",
            canary_camera_id="cam-01",
        )
        now = datetime.now(timezone.utc)
        with session_factory() as session:
            event = ModelVersionEvent(
                timestamp=now - timedelta(days=10),
                camera_id="cam-01",
                from_version=new_vid,
                to_version=new_vid,
                from_stage="shadow",
                to_stage="canary",
                triggered_by="engineer_b",
            )
            session.add(event)
            session.commit()

        pipeline.transition(new_vid, "production", "engineer_c", now=now)

        with session_factory() as session:
            old = session.query(ModelRecord).filter_by(model_version_id=old_vid).first()
            new = session.query(ModelRecord).filter_by(model_version_id=new_vid).first()
            assert old.stage == "retired"
            assert old.is_active is False
            assert new.stage == "production"
            assert new.is_active is True


class TestBackboneCompatibility:
    """Test backbone/head compatibility validation."""

    def test_compatible_head(self, session_factory):
        with session_factory() as session:
            record = ModelRecord(
                model_version_id="cam-01-head-0001",
                camera_id="cam-01",
                model_type="dinomaly2",
                model_hash="abc",
                data_hash="def",
                component_type="head",
                backbone_version_id="dinov2_vitb14-v1",
            )
            session.add(record)
            session.commit()

        with session_factory() as session:
            record = session.query(ModelRecord).first()
            # Should not raise
            ReleasePipeline.validate_backbone_compatibility(record, "dinov2_vitb14-v1")

    def test_incompatible_head(self, session_factory):
        with session_factory() as session:
            record = ModelRecord(
                model_version_id="cam-01-head-0002",
                camera_id="cam-01",
                model_type="dinomaly2",
                model_hash="abc",
                data_hash="def",
                component_type="head",
                backbone_version_id="dinov2_vitb14-v1",
            )
            session.add(record)
            session.commit()

        with session_factory() as session:
            record = session.query(ModelRecord).first()
            with pytest.raises(BackboneIncompatibleError, match="requires backbone"):
                ReleasePipeline.validate_backbone_compatibility(record, "dinov2_vitl14-v2")

    def test_full_model_skips_validation(self, session_factory):
        with session_factory() as session:
            record = ModelRecord(
                model_version_id="cam-01-full-0001",
                camera_id="cam-01",
                model_type="dinomaly2",
                model_hash="abc",
                data_hash="def",
                component_type="full",
            )
            session.add(record)
            session.commit()

        with session_factory() as session:
            record = session.query(ModelRecord).first()
            # Should not raise even with mismatched backbone
            ReleasePipeline.validate_backbone_compatibility(record, "anything")


class TestShadowStats:
    """Test shadow inference report generation."""

    def test_empty_shadow_stats(self, session_factory, pipeline):
        stats = pipeline.get_shadow_stats("nonexistent")
        assert stats["total_samples"] == 0
        assert stats["shadow_alert_rate"] == 0.0

    def test_shadow_stats_with_data(self, session_factory, pipeline):
        now = datetime.now(timezone.utc)
        with session_factory() as session:
            for i in range(10):
                log = ShadowInferenceLog(
                    timestamp=now - timedelta(hours=i),
                    camera_id="cam-01",
                    shadow_version_id="shadow-v1",
                    production_version_id="prod-v1",
                    shadow_score=0.6 + i * 0.01,
                    production_score=0.5 + i * 0.01,
                    shadow_would_alert=i >= 5,
                    production_alerted=i >= 7,
                    latency_ms=100.0 + i,
                )
                session.add(log)
            session.commit()

        stats = pipeline.get_shadow_stats("shadow-v1", camera_id="cam-01")
        assert stats["total_samples"] == 10
        assert stats["shadow_alert_rate"] == 0.5  # 5 out of 10
        assert stats["production_alert_rate"] == 0.3  # 3 out of 10
        assert stats["false_positive_delta"] == 2  # 5 - 3
        assert stats["avg_score_divergence"] > 0
        assert stats["avg_shadow_latency_ms"] > 0


class TestModelRouter:
    """Test canary routing logic."""

    def test_canary_camera_gets_canary_model(self, session_factory):
        from argus.anomaly.model_router import ModelRouter

        # Create production model
        with session_factory() as session:
            prod = ModelRecord(
                model_version_id="cam-01-prod-v1",
                camera_id="cam-01",
                model_type="dinomaly2",
                model_hash="abc",
                data_hash="def",
                stage="production",
                is_active=True,
                model_path="/models/prod",
            )
            canary = ModelRecord(
                model_version_id="cam-01-canary-v2",
                camera_id="cam-01",
                model_type="dinomaly2",
                model_hash="xyz",
                data_hash="uvw",
                stage="canary",
                canary_camera_id="cam-01",
                model_path="/models/canary",
            )
            session.add_all([prod, canary])
            session.commit()

        router = ModelRouter(session_factory)
        result = router.get_model_for_camera("cam-01")
        assert result.model_version_id == "cam-01-canary-v2"
        assert router.is_canary("cam-01")

    def test_non_canary_camera_gets_production(self, session_factory):
        from argus.anomaly.model_router import ModelRouter

        with session_factory() as session:
            prod = ModelRecord(
                model_version_id="cam-02-prod-v1",
                camera_id="cam-02",
                model_type="dinomaly2",
                model_hash="abc",
                data_hash="def",
                stage="production",
                is_active=True,
                model_path="/models/prod",
            )
            # Canary targets cam-01, not cam-02
            canary = ModelRecord(
                model_version_id="cam-02-canary-v2",
                camera_id="cam-02",
                model_type="dinomaly2",
                model_hash="xyz",
                data_hash="uvw",
                stage="canary",
                canary_camera_id="cam-01",
                model_path="/models/canary",
            )
            session.add_all([prod, canary])
            session.commit()

        router = ModelRouter(session_factory)
        result = router.get_model_for_camera("cam-02")
        assert result.model_version_id == "cam-02-prod-v1"
        assert not router.is_canary("cam-02")

    def test_no_model_returns_none(self, session_factory):
        from argus.anomaly.model_router import ModelRouter

        router = ModelRouter(session_factory)
        assert router.get_model_for_camera("cam-99") is None
        assert router.get_model_path("cam-99") is None
