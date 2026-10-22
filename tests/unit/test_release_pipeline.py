"""Tests for the four-stage model release pipeline."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from argus.storage.models import (
    Base,
    ModelRecord,
    ModelVersionEvent,
    ShadowInferenceLog,
)
from argus.storage.release_evidence import canonical_json, canonical_sha256
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


def _add_stage_entry(
    session_factory,
    *,
    model_version_id: str,
    to_stage: str,
    timestamp: datetime,
    camera_id: str = "cam-01",
) -> None:
    with session_factory() as session:
        session.add(ModelVersionEvent(
            timestamp=timestamp,
            camera_id=camera_id,
            from_version=model_version_id,
            to_version=model_version_id,
            from_stage="candidate" if to_stage == "shadow" else "shadow",
            to_stage=to_stage,
            triggered_by="engineer_a",
        ))
        session.commit()


def _add_shadow_logs(
    session_factory,
    *,
    model_version_id: str,
    now: datetime,
    count: int,
    camera_id: str = "cam-01",
    shadow_alert_every: int | None = None,
    production_alert_every: int | None = None,
    shadow_score: float = 0.42,
    production_score: float = 0.40,
    latency_ms: float = 80.0,
) -> None:
    with session_factory() as session:
        for i in range(count):
            session.add(ShadowInferenceLog(
                timestamp=now - timedelta(minutes=count - i),
                camera_id=camera_id,
                shadow_version_id=model_version_id,
                production_version_id="prod-v1",
                shadow_score=shadow_score,
                production_score=production_score,
                shadow_would_alert=(
                    shadow_alert_every is not None
                    and i % shadow_alert_every == 0
                ),
                production_alerted=(
                    production_alert_every is not None
                    and i % production_alert_every == 0
                ),
                latency_ms=latency_ms,
            ))
        session.commit()


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

    def test_canary_to_production(
        self,
        session_factory,
        pipeline,
        tmp_path,
        release_evidence_factory,
    ):
        model = tmp_path / "model.pt"
        model.write_bytes(b"runtime")
        vid = _create_model(
            session_factory,
            stage="canary",
            canary_camera_id="cam-01",
            model_path=str(model),
        )
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

        result = pipeline.transition(
            vid,
            "production",
            "engineer_c",
            now=now,
            release_evidence=release_evidence_factory(
                model_path=model,
                model_version_id=vid,
                registry_model_hash="abcd1234",
                camera_id="cam-01",
            ),
        )
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


class TestProductionReleaseEvidence:
    """Production is fail-closed and persists evidence atomically."""

    def test_transition_has_no_public_release_bypass(
        self,
        session_factory,
        pipeline,
    ):
        vid = _create_model(
            session_factory,
            stage="canary",
            canary_camera_id="cam-01",
        )

        with pytest.raises(TypeError, match="allow_bypass"):
            pipeline.transition(
                vid,
                "production",
                "operator",
                allow_bypass=True,
            )

    @staticmethod
    def _ready_canary(session_factory, tmp_path, *, version_id="model-v1"):
        model = tmp_path / f"{version_id}.pt"
        model.write_bytes(b"runtime artifact")
        now = datetime.now(timezone.utc)
        vid = _create_model(
            session_factory,
            stage="canary",
            canary_camera_id="cam-01",
            model_version_id=version_id,
            model_path=str(model),
        )
        _add_stage_entry(
            session_factory,
            model_version_id=vid,
            to_stage="canary",
            timestamp=now - timedelta(days=10),
        )
        return vid, model, now

    def test_missing_evidence_does_not_mutate_registry(
        self,
        session_factory,
        pipeline,
        tmp_path,
    ):
        old = _create_model(
            session_factory,
            stage="production",
            model_version_id="old-production",
            is_active=True,
        )
        vid, _, now = self._ready_canary(session_factory, tmp_path)

        with pytest.raises(StageTransitionError, match="release evidence rejected"):
            pipeline.transition(vid, "production", "operator", now=now)

        with session_factory() as session:
            old_record = session.query(ModelRecord).filter_by(
                model_version_id=old
            ).one()
            candidate = session.query(ModelRecord).filter_by(
                model_version_id=vid
            ).one()
            assert old_record.stage == "production"
            assert old_record.is_active is True
            assert candidate.stage == "canary"
            assert candidate.is_active is False
            assert candidate.release_evidence_json is None

    def test_valid_evidence_is_persisted_with_canonical_digest(
        self,
        session_factory,
        pipeline,
        tmp_path,
        release_evidence_factory,
    ):
        vid, model, now = self._ready_canary(session_factory, tmp_path)
        evidence = release_evidence_factory(
            model_path=model,
            model_version_id=vid,
            registry_model_hash="abcd1234",
            camera_id="cam-01",
        )

        result = pipeline.transition(
            vid,
            "production",
            "operator",
            now=now,
            release_evidence=evidence,
        )

        assert result.stage == "production"
        assert result.release_evidence_sha256 is not None
        assert result.to_dict()["release_evidence"] == evidence
        assert result.to_dict()["release_evidence_sha256"] == (
            result.release_evidence_sha256
        )
        assert result.to_dict()["release_evidence_integrity"] is True

    def test_commit_failure_rolls_back_old_and_new_model_state(
        self,
        session_factory,
        tmp_path,
        release_evidence_factory,
    ):
        old = _create_model(
            session_factory,
            stage="production",
            model_version_id="old-production",
            is_active=True,
        )
        vid, model, now = self._ready_canary(session_factory, tmp_path)
        evidence = release_evidence_factory(
            model_path=model,
            model_version_id=vid,
            registry_model_hash="abcd1234",
            camera_id="cam-01",
        )

        class FailingCommitSession:
            def __init__(self):
                self._session = session_factory()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                self._session.close()

            def __getattr__(self, name):
                return getattr(self._session, name)

            def commit(self):
                self._session.flush()
                raise RuntimeError("simulated commit failure")

            def rollback(self):
                self._session.rollback()

        failing = ReleasePipeline(
            FailingCommitSession,
            min_shadow_days=0,
            min_canary_days=0,
        )
        with pytest.raises(RuntimeError, match="simulated commit failure"):
            failing.transition(
                vid,
                "production",
                "operator",
                now=now,
                release_evidence=evidence,
            )

        with session_factory() as session:
            old_record = session.query(ModelRecord).filter_by(
                model_version_id=old
            ).one()
            candidate = session.query(ModelRecord).filter_by(
                model_version_id=vid
            ).one()
            assert old_record.stage == "production"
            assert old_record.is_active is True
            assert candidate.stage == "canary"
            assert candidate.is_active is False
            assert candidate.release_evidence_json is None


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

    def test_canary_camera_id_must_match_model_camera(self, session_factory, pipeline):
        vid = _create_model(session_factory, stage="shadow", camera_id="cam-01")
        now = datetime.now(timezone.utc)
        with session_factory() as session:
            session.add(ModelVersionEvent(
                timestamp=now - timedelta(days=5),
                camera_id="cam-01",
                from_version=vid,
                to_version=vid,
                from_stage="candidate",
                to_stage="shadow",
                triggered_by="engineer_a",
            ))
            session.commit()

        with pytest.raises(ValueError, match="canary_camera_id must match"):
            pipeline.transition(
                vid,
                "canary",
                "engineer_b",
                canary_camera_id="other-cam",
                now=now,
            )


class TestShadowEvidenceGates:
    """Test shadow evidence checks before canary promotion."""

    def _gated_pipeline(self, session_factory) -> ReleasePipeline:
        return ReleasePipeline(
            session_factory,
            min_shadow_days=0,
            min_canary_days=0,
            min_shadow_samples=3,
            max_shadow_alert_rate_delta=0.25,
            max_shadow_score_divergence=0.15,
            max_shadow_latency_ms=200.0,
        )

    def test_shadow_to_canary_requires_minimum_samples(self, session_factory):
        pipeline = self._gated_pipeline(session_factory)
        vid = _create_model(session_factory, stage="shadow")
        now = datetime.now(timezone.utc)
        _add_stage_entry(
            session_factory,
            model_version_id=vid,
            to_stage="shadow",
            timestamp=now - timedelta(days=1),
        )
        _add_shadow_logs(
            session_factory,
            model_version_id=vid,
            now=now,
            count=2,
        )

        with pytest.raises(StageTransitionError, match="Shadow evidence too sparse"):
            pipeline.transition(
                vid,
                "canary",
                "engineer_b",
                canary_camera_id="cam-01",
                now=now,
            )

    def test_shadow_to_canary_blocks_alert_rate_regression(self, session_factory):
        pipeline = self._gated_pipeline(session_factory)
        vid = _create_model(session_factory, stage="shadow")
        now = datetime.now(timezone.utc)
        _add_stage_entry(
            session_factory,
            model_version_id=vid,
            to_stage="shadow",
            timestamp=now - timedelta(days=1),
        )
        _add_shadow_logs(
            session_factory,
            model_version_id=vid,
            now=now,
            count=4,
            shadow_alert_every=1,
            production_alert_every=None,
        )

        with pytest.raises(StageTransitionError, match="alert-rate delta"):
            pipeline.transition(
                vid,
                "canary",
                "engineer_b",
                canary_camera_id="cam-01",
                now=now,
            )

    def test_shadow_to_canary_passes_with_clean_evidence(self, session_factory):
        pipeline = self._gated_pipeline(session_factory)
        vid = _create_model(session_factory, stage="shadow")
        now = datetime.now(timezone.utc)
        _add_stage_entry(
            session_factory,
            model_version_id=vid,
            to_stage="shadow",
            timestamp=now - timedelta(days=1),
        )
        _add_shadow_logs(
            session_factory,
            model_version_id=vid,
            now=now,
            count=4,
            shadow_alert_every=2,
            production_alert_every=2,
            shadow_score=0.43,
            production_score=0.40,
            latency_ms=90.0,
        )

        result = pipeline.transition(
            vid,
            "canary",
            "engineer_b",
            canary_camera_id="cam-01",
            now=now,
        )

        assert result.stage == "canary"
        assert result.canary_camera_id == "cam-01"

    def test_shadow_gate_report_exposes_readiness_checks(self, session_factory):
        pipeline = self._gated_pipeline(session_factory)
        vid = _create_model(session_factory, stage="shadow")
        now = datetime.now(timezone.utc)
        _add_shadow_logs(
            session_factory,
            model_version_id=vid,
            now=now,
            count=2,
        )

        report = pipeline.get_shadow_gate_report(vid, camera_id="cam-01")

        assert report["ready_for_canary"] is False
        sample_check = next(
            check for check in report["checks"]
            if check["name"] == "min_shadow_samples"
        )
        assert sample_check["value"] == 2
        assert sample_check["limit"] == 3
        assert sample_check["passed"] is False


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

    def test_production_retires_old_model(
        self,
        session_factory,
        pipeline,
        tmp_path,
        release_evidence_factory,
    ):
        # Create an existing production model
        old_vid = _create_model(
            session_factory,
            stage="production",
            model_version_id="cam-01-dinomaly2-old-0001",
            is_active=True,
        )
        stale_canary_vid = _create_model(
            session_factory,
            stage="canary",
            model_version_id="cam-01-dinomaly2-stale-canary-0002",
            canary_camera_id="cam-01",
        )

        # Create a canary model ready for production
        new_model = tmp_path / "new.pt"
        new_model.write_bytes(b"new runtime")
        new_vid = _create_model(
            session_factory,
            stage="canary",
            model_version_id="cam-01-dinomaly2-new-0003",
            canary_camera_id="cam-01",
            model_path=str(new_model),
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

        pipeline.transition(
            new_vid,
            "production",
            "engineer_c",
            now=now,
            release_evidence=release_evidence_factory(
                model_path=new_model,
                model_version_id=new_vid,
                registry_model_hash="abcd1234",
                camera_id="cam-01",
            ),
        )

        with session_factory() as session:
            old = session.query(ModelRecord).filter_by(model_version_id=old_vid).first()
            stale_canary = session.query(ModelRecord).filter_by(
                model_version_id=stale_canary_vid
            ).first()
            new = session.query(ModelRecord).filter_by(model_version_id=new_vid).first()
            assert old.stage == "retired"
            assert old.is_active is False
            assert stale_canary.stage == "retired"
            assert stale_canary.is_active is False
            assert stale_canary.canary_camera_id is None
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
        assert stats["alert_rate_delta"] == pytest.approx(0.2)
        assert stats["abs_alert_rate_delta"] == pytest.approx(0.2)
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

    def test_non_canary_camera_gets_production(
        self,
        session_factory,
        tmp_path,
        release_evidence_factory,
    ):
        from argus.anomaly.model_router import ModelRouter

        model_path = tmp_path / "prod.pt"
        model_path.write_bytes(b"approved production")
        evidence = release_evidence_factory(
            model_path=model_path,
            model_version_id="cam-02-prod-v1",
            registry_model_hash="abc",
            camera_id="cam-02",
        )
        with session_factory() as session:
            prod = ModelRecord(
                model_version_id="cam-02-prod-v1",
                camera_id="cam-02",
                model_type="dinomaly2",
                model_hash="abc",
                data_hash="def",
                stage="production",
                is_active=True,
                model_path=str(model_path),
                release_evidence_json=canonical_json(evidence),
                release_evidence_sha256=canonical_sha256(evidence),
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

    def test_production_with_stale_artifact_is_not_routed(
        self,
        session_factory,
        tmp_path,
        release_evidence_factory,
    ):
        from argus.anomaly.model_router import ModelRouter

        model_path = tmp_path / "prod.pt"
        model_path.write_bytes(b"approved")
        evidence = release_evidence_factory(
            model_path=model_path,
            model_version_id="cam-03-prod-v1",
            registry_model_hash="abc",
            camera_id="cam-03",
        )
        with session_factory() as session:
            session.add(ModelRecord(
                model_version_id="cam-03-prod-v1",
                camera_id="cam-03",
                model_type="dinomaly2",
                model_hash="abc",
                data_hash="def",
                stage="production",
                is_active=True,
                model_path=str(model_path),
                release_evidence_json=canonical_json(evidence),
                release_evidence_sha256=canonical_sha256(evidence),
            ))
            session.commit()
        model_path.write_bytes(b"changed")

        assert ModelRouter(session_factory).get_model_for_camera("cam-03") is None

    def test_no_model_returns_none(self, session_factory):
        from argus.anomaly.model_router import ModelRouter

        router = ModelRouter(session_factory)
        assert router.get_model_for_camera("cam-99") is None
        assert router.get_model_path("cam-99") is None
