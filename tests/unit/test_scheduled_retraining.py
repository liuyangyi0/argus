"""Tests for scheduled retraining task (C4 + A4 active learning loop)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from argus.config.schema import RetrainingConfig


class TestRetrainingConfig:
    """Test RetrainingConfig validation."""

    def test_defaults(self):
        cfg = RetrainingConfig()
        assert cfg.enabled is False
        assert cfg.interval_hours == 24
        assert cfg.min_new_baselines == 20
        assert cfg.auto_deploy is False
        assert cfg.auto_deploy_min_grade == "B"

    def test_valid_grades(self):
        for grade in ("A", "B", "C", "F"):
            cfg = RetrainingConfig(auto_deploy_min_grade=grade)
            assert cfg.auto_deploy_min_grade == grade

    def test_invalid_grade(self):
        with pytest.raises(Exception):
            RetrainingConfig(auto_deploy_min_grade="D")

    def test_interval_bounds(self):
        RetrainingConfig(interval_hours=1)
        RetrainingConfig(interval_hours=168)
        with pytest.raises(Exception):
            RetrainingConfig(interval_hours=0)


class TestCreateRetrainingTask:
    """Test the retraining task logic."""

    def _make_config(self, **overrides):
        """Create a mock config with retraining settings."""
        defaults = {
            "enabled": True,
            "interval_hours": 24,
            "min_new_baselines": 20,
            "auto_deploy": False,
            "auto_deploy_min_grade": "B",
        }
        defaults.update(overrides)
        cfg = MagicMock()
        cfg.retraining = RetrainingConfig(**defaults)
        return cfg

    def _make_camera(self, camera_id="cam_01", model_type="patchcore"):
        cam = MagicMock()
        cam.camera_id = camera_id
        cam.anomaly.model_type = model_type
        cam.anomaly.image_size = [256, 256]
        cam.anomaly.quantization = "fp16"
        return cam

    def test_disabled_config_skips_registration(self):
        from argus.core.scheduler import create_retraining_task

        scheduler = MagicMock()
        config = self._make_config(enabled=False)

        create_retraining_task(
            scheduler=scheduler,
            config=config,
            camera_configs=[],
            trainer=MagicMock(),
            model_registry=MagicMock(),
            baseline_manager=MagicMock(),
        )

        scheduler.add_interval_task.assert_not_called()

    def test_enabled_config_registers_task(self):
        from argus.core.scheduler import create_retraining_task

        scheduler = MagicMock()
        config = self._make_config(enabled=True, interval_hours=12)

        create_retraining_task(
            scheduler=scheduler,
            config=config,
            camera_configs=[self._make_camera()],
            trainer=MagicMock(),
            model_registry=MagicMock(),
            baseline_manager=MagicMock(),
        )

        scheduler.add_interval_task.assert_called_once()
        call_args = scheduler.add_interval_task.call_args
        assert call_args[0][0] == "scheduled_retraining"
        assert call_args[1]["hours"] == 12

    def test_skip_when_insufficient_baselines(self):
        from argus.core.scheduler import create_retraining_task

        scheduler = MagicMock()
        config = self._make_config(min_new_baselines=20)
        trainer = MagicMock()
        baseline_mgr = MagicMock()
        baseline_mgr.count_images.return_value = 10  # Below threshold

        create_retraining_task(
            scheduler=scheduler,
            config=config,
            camera_configs=[self._make_camera()],
            trainer=trainer,
            model_registry=MagicMock(),
            baseline_manager=baseline_mgr,
        )

        # Execute the registered callback
        callback = scheduler.add_interval_task.call_args[0][1]
        callback()

        trainer.train.assert_not_called()

    def test_triggers_training_when_threshold_met(self):
        from argus.core.scheduler import create_retraining_task

        scheduler = MagicMock()
        config = self._make_config(min_new_baselines=10)
        trainer = MagicMock()
        result = MagicMock()
        result.quality_grade = "B"
        result.status.value = "COMPLETE"
        trainer.train.return_value = result

        baseline_mgr = MagicMock()
        baseline_mgr.count_images.return_value = 50

        registry = MagicMock()
        registry.list_models.return_value = []  # No previous models

        create_retraining_task(
            scheduler=scheduler,
            config=config,
            camera_configs=[self._make_camera()],
            trainer=trainer,
            model_registry=registry,
            baseline_manager=baseline_mgr,
        )

        callback = scheduler.add_interval_task.call_args[0][1]
        callback()

        trainer.train.assert_called_once()

    def test_auto_deploy_on_good_grade(self):
        from argus.core.scheduler import create_retraining_task

        scheduler = MagicMock()
        config = self._make_config(
            min_new_baselines=5, auto_deploy=True, auto_deploy_min_grade="B"
        )
        trainer = MagicMock()
        result = MagicMock()
        result.quality_grade = "A"
        result.status.value = "COMPLETE"
        result.model_path = "/tmp/model"
        trainer.train.return_value = result

        baseline_mgr = MagicMock()
        baseline_mgr.count_images.return_value = 50

        registry = MagicMock()
        registry.list_models.return_value = []
        registry.register.return_value = "cam_01-patchcore-20260406-001"

        create_retraining_task(
            scheduler=scheduler,
            config=config,
            camera_configs=[self._make_camera()],
            trainer=trainer,
            model_registry=registry,
            baseline_manager=baseline_mgr,
        )

        callback = scheduler.add_interval_task.call_args[0][1]
        callback()

        registry.register.assert_called_once()
        registry.activate.assert_called_once_with("cam_01-patchcore-20260406-001", allow_bypass=True)

    def test_skip_deploy_on_bad_grade(self):
        from argus.core.scheduler import create_retraining_task

        scheduler = MagicMock()
        config = self._make_config(
            min_new_baselines=5, auto_deploy=True, auto_deploy_min_grade="B"
        )
        trainer = MagicMock()
        result = MagicMock()
        result.quality_grade = "F"
        result.status.value = "COMPLETE"
        trainer.train.return_value = result

        baseline_mgr = MagicMock()
        baseline_mgr.count_images.return_value = 50

        registry = MagicMock()
        registry.list_models.return_value = []

        create_retraining_task(
            scheduler=scheduler,
            config=config,
            camera_configs=[self._make_camera()],
            trainer=trainer,
            model_registry=registry,
            baseline_manager=baseline_mgr,
        )

        callback = scheduler.add_interval_task.call_args[0][1]
        callback()

        registry.register.assert_not_called()
        registry.activate.assert_not_called()

    def test_camera_failure_isolation(self):
        """One camera failing should not block others."""
        from argus.core.scheduler import create_retraining_task

        scheduler = MagicMock()
        config = self._make_config(min_new_baselines=5)
        trainer = MagicMock()
        trainer.train.side_effect = [RuntimeError("GPU error"), MagicMock(quality_grade="A")]

        baseline_mgr = MagicMock()
        baseline_mgr.count_images.return_value = 50

        registry = MagicMock()
        registry.list_models.return_value = []

        create_retraining_task(
            scheduler=scheduler,
            config=config,
            camera_configs=[self._make_camera("cam_01"), self._make_camera("cam_02")],
            trainer=trainer,
            model_registry=registry,
            baseline_manager=baseline_mgr,
        )

        callback = scheduler.add_interval_task.call_args[0][1]
        # Should not raise even though cam_01 fails
        callback()

        assert trainer.train.call_count == 2
