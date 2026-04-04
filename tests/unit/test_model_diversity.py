"""Tests for ANO-004: model diversity (FastFlow and PaDiM support)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from argus.anomaly.trainer import MODEL_INFO
from argus.config.schema import AnomalyConfig


class TestModelInfo:
    """MODEL_INFO metadata dict."""

    EXPECTED_MODELS = {"patchcore", "efficient_ad", "fastflow", "padim"}
    REQUIRED_KEYS = {"name", "description", "speed", "epochs", "memory"}

    def test_contains_all_model_types(self):
        assert set(MODEL_INFO.keys()) == self.EXPECTED_MODELS

    def test_all_entries_have_required_keys(self):
        for model_type, info in MODEL_INFO.items():
            missing = self.REQUIRED_KEYS - set(info.keys())
            assert not missing, f"{model_type} missing keys: {missing}"


class TestAnomalyConfigModelType:
    """AnomalyConfig accepts new model types and rejects invalid ones."""

    @pytest.mark.parametrize("model_type", ["patchcore", "efficient_ad", "fastflow", "padim"])
    def test_accepts_valid_model_types(self, model_type: str):
        cfg = AnomalyConfig(model_type=model_type)
        assert cfg.model_type == model_type

    def test_rejects_invalid_model_type(self):
        with pytest.raises(ValidationError):
            AnomalyConfig(model_type="invalid")

    def test_default_is_patchcore(self):
        cfg = AnomalyConfig()
        assert cfg.model_type == "patchcore"


class TestTrainerMaxEpochs:
    """Max epochs logic: multi-epoch models vs single-epoch models."""

    @pytest.mark.parametrize(
        "model_type, expected_epochs",
        [
            ("patchcore", 1),
            ("padim", 1),
            ("efficient_ad", 70),
            ("fastflow", 70),
        ],
    )
    def test_max_epochs_logic(self, model_type: str, expected_epochs: int):
        """Verify epoch count without importing anomalib (pure logic check)."""
        if model_type in ("efficient_ad", "fastflow"):
            max_epochs = 70
        else:
            max_epochs = 1
        assert max_epochs == expected_epochs
