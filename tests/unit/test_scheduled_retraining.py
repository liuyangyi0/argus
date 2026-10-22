"""Tests for scheduled retraining task (C4 + A4 active learning loop)."""

from __future__ import annotations

from types import SimpleNamespace
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
        cam.anomaly.inference_backend = "openvino"
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

    def test_scheduled_retraining_marks_feedback_with_model_version(self):
        from argus.core.scheduler import create_retraining_task

        scheduler = MagicMock()
        config = self._make_config(min_new_baselines=10)
        trainer = MagicMock()
        result = MagicMock()
        result.quality_grade = "B"
        result.status.value = "COMPLETE"
        result.model_version_id = "model-version-123"
        trainer.train.return_value = result

        baseline_mgr = MagicMock()
        baseline_mgr.count_images.return_value = 50

        registry = MagicMock()
        registry.list_models.return_value = []

        feedback = MagicMock()
        feedback.get_pending_for_training.return_value = [
            MagicMock(feedback_id="fb-1"),
            MagicMock(feedback_id="fb-2"),
        ]

        create_retraining_task(
            scheduler=scheduler,
            config=config,
            camera_configs=[self._make_camera()],
            trainer=trainer,
            model_registry=registry,
            baseline_manager=baseline_mgr,
            feedback_manager=feedback,
        )

        callback = scheduler.add_interval_task.call_args[0][1]
        callback()

        feedback.get_pending_for_training.assert_called_once_with(camera_id="cam_01")
        feedback.mark_batch_processed.assert_called_once_with(
            ["fb-1", "fb-2"], "model-version-123",
        )

    def test_auto_deploy_quality_gate_only_registers_candidate(self, tmp_path):
        from argus.core.scheduler import create_retraining_task

        scheduler = MagicMock()
        config = self._make_config(
            min_new_baselines=5, auto_deploy=True, auto_deploy_min_grade="B"
        )
        trainer = MagicMock()
        result = MagicMock()
        result.quality_grade = "A"
        result.status.value = "COMPLETE"
        artifact = tmp_path / "model.pt"
        artifact.write_bytes(b"runtime artifact")
        result.model_path = str(artifact)
        result.model_version_id = None
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
        assert registry.register.call_args.kwargs["model_path"] == artifact
        assert registry.register.call_args.kwargs["baseline_dir"] == baseline_mgr.get_baseline_dir.return_value
        assert registry.register.call_args.kwargs["training_params"]["quality_grade"] == "A"
        # Scheduled work has no operator approval or runtime transaction.  It
        # may register the result, but publication remains a manual two-phase
        # ModelReleaseService command.
        registry.activate.assert_not_called()
        registry.promote.assert_not_called()

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

    def test_auto_deploy_fails_closed_before_registry_activation(self, tmp_path):
        """The scheduler audits release-required without calling a mutator."""
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
        artifact = tmp_path / "model.pt"
        artifact.write_bytes(b"runtime artifact")
        result.model_path = str(artifact)
        result.model_version_id = None
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
        callback()  # Must NOT raise

        registry.activate.assert_not_called()
        registry.promote.assert_not_called()
        # Audit records why a high-quality result remains a candidate.
        audit_logger.log.assert_called_once()
        kwargs = audit_logger.log.call_args.kwargs
        assert kwargs["action"] == "auto_retrain_release_required"
        assert kwargs["target_id"] == "cam_01-patchcore-20260406-001"
        assert "ModelReleaseService" in kwargs["detail"]

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
        v1 = MagicMock()
        v1.version = "v001"
        v2 = MagicMock()
        v2.version = "v003"
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

    @pytest.mark.parametrize(
        ("backend", "expected_format", "expected_quantization"),
        [("torch-cuda", "torch", "fp32"), ("torch-cpu", "torch", "fp32"),
         ("openvino", "openvino", "fp16")],
    )
    def test_runtime_backend_selects_matching_export_and_does_not_duplicate_registration(
        self, backend, expected_format, expected_quantization,
    ):
        from argus.anomaly.trainer import TrainingResult, TrainingStatus
        from argus.core.scheduler import create_retraining_task

        scheduler = MagicMock()
        config = self._make_config(min_new_baselines=5, auto_deploy=True)
        camera = self._make_camera()
        camera.anomaly.inference_backend = backend
        trainer = MagicMock()
        trainer.train.return_value = TrainingResult(
            status=TrainingStatus.COMPLETE, model_version_id="already-registered-run",
            quality_report=SimpleNamespace(grade="A"),
        )
        baseline_mgr = MagicMock()
        baseline_mgr.count_images.return_value = 50
        registry = MagicMock()
        registry.list_models.return_value = []

        create_retraining_task(
            scheduler=scheduler, config=config, camera_configs=[camera], trainer=trainer,
            model_registry=registry, baseline_manager=baseline_mgr,
        )
        scheduler.add_interval_task.call_args[0][1]()

        assert trainer.train.call_args.kwargs["export_format"] == expected_format
        assert trainer.train.call_args.kwargs["quantization"] == expected_quantization
        registry.register.assert_not_called()
        registry.activate.assert_not_called()
        registry.promote.assert_not_called()
        assert scheduler._audit_logger.log.call_args.kwargs["target_id"] == "already-registered-run"

    def test_persisted_training_counts_prevent_retraining_the_same_baselines(self):
        from argus.core.scheduler import create_retraining_task

        scheduler = MagicMock()
        trainer = MagicMock()
        baseline_mgr = MagicMock()
        baseline_mgr.count_images.return_value = 50
        registry = MagicMock()
        registry.list_models.return_value = [SimpleNamespace(
            training_params='{"train_count": 40, "val_count": 10}',
        )]

        create_retraining_task(
            scheduler=scheduler, config=self._make_config(min_new_baselines=5),
            camera_configs=[self._make_camera()], trainer=trainer,
            model_registry=registry, baseline_manager=baseline_mgr,
        )
        scheduler.add_interval_task.call_args[0][1]()

        trainer.train.assert_not_called()

    def test_missing_exact_artifact_is_not_registered_as_a_successful_run(self):
        from argus.anomaly.trainer import TrainingResult, TrainingStatus
        from argus.core.scheduler import create_retraining_task

        scheduler = MagicMock()
        trainer = MagicMock()
        trainer.train.return_value = TrainingResult(
            status=TrainingStatus.COMPLETE, model_path="/missing/model.pt",
            quality_report=SimpleNamespace(grade="A"),
        )
        baseline_mgr = MagicMock()
        baseline_mgr.count_images.return_value = 50
        registry = MagicMock()
        registry.list_models.return_value = []
        feedback = MagicMock()
        feedback.get_pending_for_training.return_value = [MagicMock(feedback_id="pending")]

        create_retraining_task(
            scheduler=scheduler, config=self._make_config(min_new_baselines=5, auto_deploy=True),
            camera_configs=[self._make_camera()], trainer=trainer,
            model_registry=registry, baseline_manager=baseline_mgr, feedback_manager=feedback,
        )
        scheduler.add_interval_task.call_args[0][1]()

        registry.register.assert_not_called()
        feedback.mark_batch_processed.assert_not_called()
