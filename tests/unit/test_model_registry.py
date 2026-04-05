"""Tests for model registry (C4)."""

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from argus.storage.models import Base, ModelRecord
from argus.storage.model_registry import ModelRegistry


@pytest.fixture
def registry(tmp_path):
    """Create a test registry with in-memory SQLite."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return ModelRegistry(session_factory=SessionLocal)


@pytest.fixture
def model_dir(tmp_path):
    """Create a fake model directory."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "model.xml").write_text("fake model")
    return d


@pytest.fixture
def baseline_dir(tmp_path):
    """Create a fake baseline directory."""
    d = tmp_path / "baseline"
    d.mkdir()
    (d / "img.png").write_bytes(b"fake image")
    return d


class TestModelRegistry:

    def test_register_and_activate_model(self, registry, model_dir, baseline_dir):
        """Register a model and activate it."""
        vid = registry.register(
            model_path=model_dir,
            baseline_dir=baseline_dir,
            camera_id="cam_01",
            model_type="patchcore",
            training_params={"epochs": 1},
        )
        assert vid.startswith("cam_01-patchcore-")

        registry.activate(vid)
        active = registry.get_active("cam_01")
        assert active is not None
        assert active.model_version_id == vid
        assert active.is_active is True

    def test_alert_carries_model_version(self):
        """Alert dataclass should have model_version_id field."""
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

    def test_model_hash_changes_on_retrain(self, registry, baseline_dir, tmp_path):
        """Different model files → different hashes."""
        m1 = tmp_path / "model1"
        m1.mkdir()
        (m1 / "model.xml").write_text("version 1")

        m2 = tmp_path / "model2"
        m2.mkdir()
        (m2 / "model.xml").write_text("version 2")

        vid1 = registry.register(m1, baseline_dir, "cam_01", "patchcore")
        vid2 = registry.register(m2, baseline_dir, "cam_01", "patchcore")

        assert vid1 != vid2

    def test_rollback_to_previous(self, registry, baseline_dir, tmp_path):
        """Rollback should reactivate the previous model version."""
        m1 = tmp_path / "model_a"
        m1.mkdir()
        (m1 / "model.xml").write_text("v1")

        m2 = tmp_path / "model_b"
        m2.mkdir()
        (m2 / "model.xml").write_text("v2")

        vid1 = registry.register(m1, baseline_dir, "cam_01", "patchcore")
        registry.activate(vid1)

        vid2 = registry.register(m2, baseline_dir, "cam_01", "patchcore")
        registry.activate(vid2)

        # Current active should be vid2
        active = registry.get_active("cam_01")
        assert active.model_version_id == vid2

        # Rollback should reactivate vid1
        rolled_back = registry.rollback("cam_01")
        assert rolled_back is not None
        assert rolled_back.model_version_id == vid1

        active = registry.get_active("cam_01")
        assert active.model_version_id == vid1

    def test_rollback_no_previous(self, registry, model_dir, baseline_dir):
        """Rollback with only one model should return None."""
        vid = registry.register(model_dir, baseline_dir, "cam_01", "patchcore")
        registry.activate(vid)

        result = registry.rollback("cam_01")
        assert result is None

    def test_list_models_by_camera(self, registry, baseline_dir, tmp_path):
        """list_models should filter by camera_id correctly."""
        m1 = tmp_path / "m1"
        m1.mkdir()
        (m1 / "model.xml").write_text("cam01 model")

        m2 = tmp_path / "m2"
        m2.mkdir()
        (m2 / "model.xml").write_text("cam02 model")

        registry.register(m1, baseline_dir, "cam_01", "patchcore")
        registry.register(m2, baseline_dir, "cam_02", "patchcore")

        all_models = registry.list_models()
        assert len(all_models) == 2

        cam01_models = registry.list_models(camera_id="cam_01")
        assert len(cam01_models) == 1
        assert cam01_models[0].camera_id == "cam_01"

        cam02_models = registry.list_models(camera_id="cam_02")
        assert len(cam02_models) == 1
        assert cam02_models[0].camera_id == "cam_02"

        empty = registry.list_models(camera_id="cam_99")
        assert len(empty) == 0
