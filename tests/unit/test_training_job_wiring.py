"""Training-job wiring must not expose an out-of-band model path callback.

Training completion must register the model as CANDIDATE *only* — never
hot-reload it onto the running pipeline. Promotion through the release
pipeline (shadow → canary → production) is the single supported path
for putting a new model on the inference hot path.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_training_executor_has_no_model_path_callback(tmp_path):
    """A callback could reload a candidate without a release operation."""
    from argus.runtime.training_job_wiring import register_training_job_processing

    camera_manager = MagicMock()

    scheduler = MagicMock()
    config = MagicMock()
    config.storage.backbones_dir = tmp_path
    config.storage.baselines_dir = tmp_path
    database = MagicMock()
    database.get_session = MagicMock()

    captured_kwargs: dict = {}

    class _ExecutorSpy:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    captured_interval: dict = {}

    def _capture_job_task(*args, **kwargs):
        captured_interval.update(kwargs)

    with patch(
        "argus.runtime.training_job_wiring.create_job_processing_task",
        _capture_job_task,
    ), patch(
        "argus.anomaly.job_executor.TrainingJobExecutor", _ExecutorSpy,
    ), patch.dict(
        sys.modules, {"argus.anomaly.backbone_trainer": SimpleNamespace(BackboneTrainer=MagicMock)},
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
            interval_seconds=2.0,
        )

    assert captured_interval["interval_seconds"] == 2.0
    assert "on_model_trained" not in captured_kwargs
    assert captured_kwargs["camera_manager"] is camera_manager
    camera_manager.get_pipeline.assert_not_called()
