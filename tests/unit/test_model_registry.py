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
