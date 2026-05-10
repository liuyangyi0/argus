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
        # P1 fix: scheduler must NEVER bypass the stage gate. Use the same
        # release pipeline path as a manual activation, identified by
        # triggered_by="auto_retrain" so audit logs can attribute it.
        registry.activate.assert_called_once_with(
            "cam_01-patchcore-20260406-001", triggered_by="auto_retrain",
        )

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

    def test_auto_deploy_respects_stage_gate(self):
        """P1: when registry refuses activation because the version is still
        a CANDIDATE, the scheduler must log + audit + continue, NOT bypass."""
        from argus.core.scheduler import create_retraining_task

        scheduler = MagicMock()
        audit_logger = MagicMock()
        scheduler._audit_logger = audit_logger
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
        registry.activate.side_effect = ValueError(
            "Cannot activate model at stage 'candidate'."
        )

        create_retraining_task(
            scheduler=scheduler,
            config=config,
            camera_configs=[self._make_camera()],
            trainer=trainer,
            model_registry=registry,
            baseline_manager=baseline_mgr,
        )

        callback = scheduler.add_interval_task.call_args[0][1]
        callback()  # Must NOT raise

        registry.activate.assert_called_once_with(
            "cam_01-patchcore-20260406-001", triggered_by="auto_retrain",
        )
        # Audit must record the blocked attempt for forensics
        audit_logger.log.assert_called_once()
        kwargs = audit_logger.log.call_args.kwargs
        assert kwargs["action"] == "auto_retrain_activation_blocked"
        assert kwargs["target_id"] == "cam_01-patchcore-20260406-001"

    def test_all_active_strategy_merges_eligible_versions(self):
        """P2 follow-up: dataset_strategy='all_active' must call trainer with
        baseline_dir_override pointing at a merged tmp dir built from every
        VERIFIED + ACTIVE version."""
        from argus.core.scheduler import create_retraining_task

        scheduler = MagicMock()
        config = self._make_config(min_new_baselines=5)
        config.retraining = RetrainingConfig(
            enabled=True, min_new_baselines=5, dataset_strategy="all_active",
        )
        trainer = MagicMock()
        trainer.train.return_value = MagicMock(quality_grade="A", status=MagicMock(value="COMPLETE"))

        baseline_mgr = MagicMock()
        baseline_mgr.count_images.return_value = 50
        # Multi-dir resolution returns two paths
        from pathlib import Path
        baseline_mgr.resolve_dataset_dirs.return_value = [Path("/tmp/v1"), Path("/tmp/v2")]
        baseline_mgr.count_images_multi.return_value = 80

        registry = MagicMock()
        registry.list_models.return_value = []

        # Lifecycle returns 2 eligible versions (VERIFIED + ACTIVE)
        lifecycle = MagicMock()
        v1 = MagicMock(); v1.version = "v001"
        v2 = MagicMock(); v2.version = "v003"
        lifecycle.get_eligible_versions.return_value = [v1, v2]

        with patch("argus.anomaly.trainer._DatasetMerger") as merger_cls, \
             patch("argus.anomaly.trainer._build_merger_items") as build_items:
            merger = MagicMock()
            merger.__enter__ = MagicMock(return_value=Path("/tmp/merged"))
            merger.__exit__ = MagicMock(return_value=False)
            merger_cls.return_value = merger
            build_items.return_value = [("a", Path("/tmp/v1")), ("b", Path("/tmp/v2"))]

            create_retraining_task(
                scheduler=scheduler,
                config=config,
                camera_configs=[self._make_camera()],
                trainer=trainer,
                model_registry=registry,
                baseline_manager=baseline_mgr,
                baseline_lifecycle=lifecycle,
            )
            callback = scheduler.add_interval_task.call_args[0][1]
            callback()

        # Lifecycle was queried with since=None (no since_last_train cap)
        lifecycle.get_eligible_versions.assert_called_once()
        kwargs = lifecycle.get_eligible_versions.call_args.kwargs
        assert kwargs["since"] is None

        trainer.train.assert_called_once()
        train_kwargs = trainer.train.call_args.kwargs
        assert train_kwargs["baseline_dir_override"] == Path("/tmp/merged")
        assert train_kwargs["image_count_override"] == 80
        assert "v001" in train_kwargs["baseline_versions_label"]
        assert "v003" in train_kwargs["baseline_versions_label"]

    def test_since_last_train_strategy_uses_last_completion(self):
        """since_last_train must filter eligible versions by the timestamp of
        the most recent successful TrainingRecord."""
        from argus.core.scheduler import create_retraining_task
        from datetime import datetime, timezone

        scheduler = MagicMock()
        config = self._make_config(min_new_baselines=5)
        config.retraining = RetrainingConfig(
            enabled=True, min_new_baselines=5, dataset_strategy="since_last_train",
        )
        trainer = MagicMock()
        trainer.train.return_value = MagicMock(quality_grade="A", status=MagicMock(value="COMPLETE"))

        baseline_mgr = MagicMock()
        baseline_mgr.count_images.return_value = 50

        registry = MagicMock()
        registry.list_models.return_value = []

        lifecycle = MagicMock()
        lifecycle.get_eligible_versions.return_value = []  # nothing newer → fallback

        # Database returns one COMPLETE training record
        last_train = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
        db = MagicMock()
        record = MagicMock()
        record.status = "COMPLETE"
        record.trained_at = last_train
        db.get_training_history.return_value = [record]

        create_retraining_task(
            scheduler=scheduler,
            config=config,
            camera_configs=[self._make_camera()],
            trainer=trainer,
            model_registry=registry,
            baseline_manager=baseline_mgr,
            baseline_lifecycle=lifecycle,
            database=db,
        )
        callback = scheduler.add_interval_task.call_args[0][1]
        callback()

        lifecycle.get_eligible_versions.assert_called_once()
        kwargs = lifecycle.get_eligible_versions.call_args.kwargs
        # since must be the last training timestamp
        assert kwargs["since"] == last_train

        # No eligible versions → trainer falls back to single-version legacy path
        # (no baseline_dir_override kwarg in train call)
        train_kwargs = trainer.train.call_args.kwargs
        assert "baseline_dir_override" not in train_kwargs

    def test_dataset_strategy_unwired_when_lifecycle_missing(self):
        """all_active without baseline_lifecycle must log warning + fall back."""
        from argus.core.scheduler import create_retraining_task

        scheduler = MagicMock()
        config = self._make_config(min_new_baselines=5)
        config.retraining = RetrainingConfig(
            enabled=True, min_new_baselines=5, dataset_strategy="all_active",
        )
        trainer = MagicMock()
        trainer.train.return_value = MagicMock(quality_grade="A", status=MagicMock(value="COMPLETE"))

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
            # no baseline_lifecycle → fallback to current_only
        )
        callback = scheduler.add_interval_task.call_args[0][1]
        callback()

        # Train still runs but on legacy single-version path
        train_kwargs = trainer.train.call_args.kwargs
        assert "baseline_dir_override" not in train_kwargs

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
