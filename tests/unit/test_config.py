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
    AnomalyConfig, CalibrationConfig, SimplexConfig,
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

    def test_evidence_threshold_too_low_rejected(self):
        """evidence_threshold below minimum must be rejected."""
        with pytest.raises(ValidationError):
            TemporalConfirmation(evidence_threshold=0.1)

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
        assert len(config.cameras) >= 1
        assert config.cameras[0].camera_id == "c"


class TestDinomaly2Config:
    """B1-6: Dinomaly2 configuration tests."""

    def test_dinomaly2_config_valid(self):
        """model_type=dinomaly2 + backbone + layers 正确解析。"""
        config = AnomalyConfig(
            model_type="dinomaly2",
            dinomaly_backbone="dinov2_vitb14",
            dinomaly_encoder_layers=[2, 5, 8, 11],
        )
        assert config.model_type == "dinomaly2"
        assert config.dinomaly_backbone == "dinov2_vitb14"

    def test_dinomaly2_few_shot_minimum(self):
        """dinomaly_few_shot_images validates range."""
        config = AnomalyConfig(model_type="dinomaly2", dinomaly_few_shot_images=8)
        assert config.dinomaly_few_shot_images == 8

    def test_dinomaly2_multi_class_default_false(self):
        """dinomaly_multi_class defaults to False."""
        config = AnomalyConfig(model_type="dinomaly2")
        assert config.dinomaly_multi_class is False


class TestQuantizationConfig:
    """B2-4: Quantization configuration tests."""

    def test_quantization_config_values(self):
        """fp32/fp16/int8 都是合法值。"""
        for q in ("fp32", "fp16", "int8"):
            config = AnomalyConfig(quantization=q)
            assert config.quantization == q

    def test_int8_calibration_images_range(self):
        """calibration_images validates range 50-1000."""
        config = AnomalyConfig(quantization="int8", quantization_calibration_images=100)
        assert config.quantization_calibration_images == 100

        with pytest.raises(ValidationError):
            AnomalyConfig(quantization="int8", quantization_calibration_images=10)


class TestCUSUMConfig:
    """A1-3: CUSUM evidence config tests."""

    def test_evidence_lambda_range(self):
        """evidence_lambda must be 0.80-0.99."""
        config = TemporalConfirmation(evidence_lambda=0.95)
        assert config.evidence_lambda == 0.95

        with pytest.raises(ValidationError):
            TemporalConfirmation(evidence_lambda=0.5)

    def test_evidence_threshold_range(self):
        """evidence_threshold must be 0.5-20.0."""
        config = TemporalConfirmation(evidence_threshold=5.0)
        assert config.evidence_threshold == 5.0


class TestCalibrationConfig:
    """A2-2: Calibration config tests."""

    def test_calibration_default_disabled(self):
        """Calibration should be disabled by default."""
        config = CalibrationConfig()
        assert config.enabled is False

    def test_calibration_fprs(self):
        """FPR values should be within range."""
        config = CalibrationConfig(
            enabled=True,
            target_fpr_info=0.10,
            target_fpr_low=0.01,
            target_fpr_medium=0.001,
            target_fpr_high=0.0001,
        )
        assert config.target_fpr_info == 0.10


class TestSimplexConfig:
    """A3-2: Simplex config tests."""

    def test_simplex_default_enabled(self):
        config = SimplexConfig()
        assert config.enabled is True

    def test_simplex_diff_threshold_range(self):
        with pytest.raises(ValidationError):
            SimplexConfig(diff_threshold=5)  # below 10
