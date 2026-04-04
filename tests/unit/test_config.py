"""Tests for configuration loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from argus.config.loader import load_config
from argus.config.schema import (
    ArgusConfig, AlertSeverity, ZonePriority,
    SeverityThresholds, MOG2Config, CameraConfig, TemporalConfirmation,
)


class TestConfigSchema:
    def test_default_config_is_valid(self):
        """Default config should pass validation."""
        config = ArgusConfig()
        assert config.node_id == "argus-edge-01"
        assert config.log_level == "INFO"

    def test_severity_thresholds_ordering(self):
        """Severity thresholds should be monotonically increasing."""
        config = ArgusConfig()
        t = config.alerts.severity_thresholds
        assert t.info < t.low < t.medium < t.high

    def test_zone_priority_values(self):
        """Zone priority enum values should be correct."""
        assert ZonePriority.CRITICAL.value == "critical"
        assert ZonePriority.STANDARD.value == "standard"
        assert ZonePriority.LOW_PRIORITY.value == "low_priority"

    def test_inverted_severity_thresholds_rejected(self):
        """Inverted severity thresholds must be rejected."""
        with pytest.raises(ValidationError, match="ordered"):
            SeverityThresholds(info=0.95, low=0.85, medium=0.70, high=0.50)

    def test_equal_severity_thresholds_rejected(self):
        """Equal severity thresholds must be rejected."""
        with pytest.raises(ValidationError, match="ordered"):
            SeverityThresholds(info=0.5, low=0.5, medium=0.5, high=0.5)

    def test_mog2_history_too_low_rejected(self):
        """MOG2 history below minimum must be rejected."""
        with pytest.raises(ValidationError):
            MOG2Config(history=1)

    def test_mog2_heartbeat_too_low_rejected(self):
        """Heartbeat frames below minimum must be rejected."""
        with pytest.raises(ValidationError):
            MOG2Config(heartbeat_frames=1)

    def test_fps_target_zero_rejected(self):
        """FPS target of 0 must be rejected."""
        with pytest.raises(ValidationError):
            CameraConfig(camera_id="c1", name="t", source="x", fps_target=0)

    def test_min_consecutive_frames_zero_rejected(self):
        """min_consecutive_frames of 0 must be rejected."""
        with pytest.raises(ValidationError):
            TemporalConfirmation(min_consecutive_frames=0)

    def test_valid_bounds_accepted(self):
        """Valid parameter values within bounds should be accepted."""
        t = SeverityThresholds(info=0.3, low=0.5, medium=0.7, high=0.9)
        assert t.info == 0.3
        m = MOG2Config(history=100, heartbeat_frames=50)
        assert m.history == 100


class TestConfigLoader:
    def test_load_valid_yaml(self, tmp_path):
        """Should load and validate a YAML config file."""
        config_data = {
            "node_id": "test-node",
            "cameras": [
                {
                    "camera_id": "cam_01",
                    "name": "Test Cam",
                    "source": "test.mp4",
                    "protocol": "file",
                }
            ],
        }
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_file)
        assert config.node_id == "test-node"
        assert len(config.cameras) == 1
        assert config.cameras[0].camera_id == "cam_01"

    def test_load_missing_file_raises(self):
        """Should raise FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_load_empty_yaml(self, tmp_path):
        """Empty YAML should produce default config."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        config = load_config(config_file)
        assert config.node_id == "argus-edge-01"

    def test_load_default_config_file(self):
        """The default.yaml shipped with the project should load correctly."""
        config = load_config("configs/default.yaml")
        assert config.node_id == "argus-edge-01"
        assert len(config.cameras) == 1
        assert config.cameras[0].camera_id == "cam_01"
