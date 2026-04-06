"""Tests for the BackboneManager and HeadDetector."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from argus.anomaly.backbone_manager import BackboneManager, HeadDetector
from argus.storage.release_pipeline import BackboneIncompatibleError


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset BackboneManager singleton between tests."""
    BackboneManager.reset()
    yield
    BackboneManager.reset()


@pytest.fixture(autouse=True)
def mock_torch():
    """Mock torch module for all tests, restoring original after."""
    original_torch = sys.modules.get("torch")
    mock = MagicMock()
    mock.load.return_value = MagicMock(name="mock_model")
    mock.hub.load.return_value = MagicMock(name="hub_model")
    sys.modules["torch"] = mock
    yield mock
    # Restore original torch (if it was imported before)
    if original_torch is not None:
        sys.modules["torch"] = original_torch
    else:
        sys.modules.pop("torch", None)


class TestBackboneManagerSingleton:

    def test_singleton_returns_same_instance(self):
        a = BackboneManager.get_instance()
        b = BackboneManager.get_instance()
        assert a is b

    def test_reset_creates_new_instance(self):
        a = BackboneManager.get_instance()
        BackboneManager.reset()
        b = BackboneManager.get_instance()
        assert a is not b

    def test_initial_state(self):
        manager = BackboneManager.get_instance()
        assert manager.version is None
        assert manager.is_loaded is False
        assert manager.get_backbone() is None


class TestBackboneLoading:

    def test_load_pt_file(self, mock_torch, tmp_path):
        model_file = tmp_path / "backbone.pt"
        model_file.write_bytes(b"fake weights")

        mock_model = MagicMock(name="loaded_model")
        mock_torch.load.return_value = mock_model

        manager = BackboneManager.get_instance()
        success = manager.load(model_file, "dinov2_vitb14-v1")

        assert success is True
        assert manager.version == "dinov2_vitb14-v1"
        assert manager.is_loaded is True
        assert manager.get_backbone() is mock_model

    def test_load_failure_returns_false(self, mock_torch, tmp_path):
        model_file = tmp_path / "broken.pt"
        model_file.write_bytes(b"corrupt")
        mock_torch.load.side_effect = RuntimeError("corrupt file")

        manager = BackboneManager.get_instance()
        success = manager.load(model_file, "v1")

        assert success is False
        assert manager.is_loaded is False


class TestBackboneUpgrade:

    def test_upgrade_swaps_atomically(self, mock_torch, tmp_path):
        old_file = tmp_path / "old.pt"
        old_file.write_bytes(b"old")
        new_file = tmp_path / "new.pt"
        new_file.write_bytes(b"new")

        old_model = MagicMock(name="old_model")
        new_model = MagicMock(name="new_model")
        mock_torch.load.side_effect = [old_model, new_model]

        manager = BackboneManager.get_instance()
        manager.load(old_file, "v1")
        assert manager.get_backbone() is old_model

        success = manager.upgrade(new_file, "v2")
        assert success is True
        assert manager.version == "v2"
        assert manager.get_backbone() is new_model

    def test_upgrade_rollback_on_failure(self, mock_torch, tmp_path):
        old_file = tmp_path / "old.pt"
        old_file.write_bytes(b"old")
        bad_file = tmp_path / "bad.pt"
        bad_file.write_bytes(b"bad")

        old_model = MagicMock(name="old_model")
        mock_torch.load.side_effect = [old_model, RuntimeError("corrupt")]

        manager = BackboneManager.get_instance()
        manager.load(old_file, "v1")

        success = manager.upgrade(bad_file, "v2")
        assert success is False
        assert manager.version == "v1"
        assert manager.get_backbone() is old_model


class TestHeadCompatibilityValidation:

    def test_compatible_head_passes(self):
        manager = BackboneManager.get_instance()
        manager._backbone_version = "dinov2_vitb14-v1"
        manager._loaded = True
        manager.validate_head_compatibility("dinov2_vitb14-v1")

    def test_incompatible_head_raises(self):
        manager = BackboneManager.get_instance()
        manager._backbone_version = "dinov2_vitb14-v1"
        manager._loaded = True

        with pytest.raises(BackboneIncompatibleError, match="requires backbone"):
            manager.validate_head_compatibility("dinov2_vitl14-v2")

    def test_no_backbone_loaded_raises(self):
        manager = BackboneManager.get_instance()
        with pytest.raises(BackboneIncompatibleError, match="no backbone is loaded"):
            manager.validate_head_compatibility("dinov2_vitb14-v1")

    def test_none_ref_skips_validation(self):
        manager = BackboneManager.get_instance()
        manager.validate_head_compatibility(None)


class TestHeadDetector:

    def test_load_validates_backbone(self, mock_torch, tmp_path):
        head_file = tmp_path / "head.pt"
        head_file.write_bytes(b"head weights")
        mock_torch.load.return_value = MagicMock()

        manager = BackboneManager.get_instance()
        manager._backbone_version = "dinov2_vitb14-v1"
        manager._loaded = True

        head = HeadDetector(
            head_path=head_file,
            camera_id="cam-01",
            backbone_ref="dinov2_vitb14-v1",
        )
        assert head.load() is True
        assert head.is_loaded is True

    def test_load_rejects_incompatible_backbone(self, tmp_path):
        head_file = tmp_path / "head.pt"
        head_file.write_bytes(b"head weights")

        manager = BackboneManager.get_instance()
        manager._backbone_version = "dinov2_vitb14-v1"
        manager._loaded = True

        head = HeadDetector(
            head_path=head_file,
            camera_id="cam-01",
            backbone_ref="dinov2_vitl14-v2",
        )
        with pytest.raises(BackboneIncompatibleError):
            head.load()

    def test_hot_reload_validates_backbone(self, mock_torch, tmp_path):
        old_file = tmp_path / "old_head.pt"
        old_file.write_bytes(b"old")
        new_file = tmp_path / "new_head.pt"
        new_file.write_bytes(b"new")
        mock_torch.load.return_value = MagicMock()

        manager = BackboneManager.get_instance()
        manager._backbone_version = "dinov2_vitb14-v1"
        manager._loaded = True

        head = HeadDetector(
            head_path=old_file,
            camera_id="cam-01",
            backbone_ref="dinov2_vitb14-v1",
        )
        head.load()
        assert head.hot_reload(new_file, backbone_ref="dinov2_vitb14-v1") is True

    def test_hot_reload_rejects_incompatible(self, mock_torch, tmp_path):
        head_file = tmp_path / "head.pt"
        head_file.write_bytes(b"head")
        new_file = tmp_path / "new_head.pt"
        new_file.write_bytes(b"new")
        mock_torch.load.return_value = MagicMock()

        manager = BackboneManager.get_instance()
        manager._backbone_version = "dinov2_vitb14-v1"
        manager._loaded = True

        head = HeadDetector(
            head_path=head_file,
            camera_id="cam-01",
            backbone_ref="dinov2_vitb14-v1",
        )
        head.load()

        with pytest.raises(BackboneIncompatibleError):
            head.hot_reload(new_file, backbone_ref="dinov2_vitl14-v2")
