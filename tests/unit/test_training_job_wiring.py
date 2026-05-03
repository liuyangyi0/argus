"""Tests for training_job_wiring._on_model_trained (P1 fix 2026-05).

Training completion must register the model as CANDIDATE *only* — never
hot-reload it onto the running pipeline. Promotion through the release
pipeline (shadow → canary → production) is the single supported path
for putting a new model on the inference hot path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


def test_on_model_trained_does_not_call_pipeline_reload(tmp_path):
    """Regression: prior behaviour invoked pipeline.reload_anomaly_model on
    every successful training run, slipping a CANDIDATE model into production
    with no audit / no operator gate."""
    from argus.runtime.training_job_wiring import register_training_job_processing

    pipeline = MagicMock()
    camera_manager = MagicMock()
    camera_manager.get_pipeline.return_value = pipeline

    scheduler = MagicMock()
    config = MagicMock()
    config.storage.backbones_dir = tmp_path
    config.storage.baselines_dir = tmp_path
    database = MagicMock()
    database.get_session = MagicMock()

    captured_callbacks: dict = {}

    class _ExecutorSpy:
        def __init__(self, **kwargs):
            captured_callbacks["on_model_trained"] = kwargs.get("on_model_trained")

    with patch(
        "argus.runtime.training_job_wiring.create_job_processing_task",
        lambda *a, **kw: None,
    ), patch(
        "argus.anomaly.job_executor.TrainingJobExecutor", _ExecutorSpy,
    ), patch(
        "argus.anomaly.backbone_trainer.BackboneTrainer", MagicMock,
    ), patch(
        "argus.storage.model_registry.ModelRegistry", MagicMock,
    ):
        register_training_job_processing(
            scheduler=scheduler,
            config=config,
            database=database,
            baseline_manager=MagicMock(),
            model_trainer=MagicMock(),
            camera_manager=camera_manager,
        )

    on_trained = captured_callbacks["on_model_trained"]
    assert on_trained is not None

    on_trained("cam_01", Path(tmp_path / "fake_model.xml"))

    # P1 contract: callback must NEVER touch the pipeline directly.
    pipeline.reload_anomaly_model.assert_not_called()
    camera_manager.get_pipeline.assert_not_called()
