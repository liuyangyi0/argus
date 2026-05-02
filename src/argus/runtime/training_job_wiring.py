"""Wire queued training jobs into the scheduler.

Extracted from ``argus.__main__`` so the entrypoint no longer carries the
construction recipe for ``BackboneTrainer`` / ``TrainingJobExecutor`` /
``ModelRegistry``. The entrypoint passes in ``baseline_manager`` and
``model_trainer`` so the scheduled retraining loop and the queued-job
executor share the same on-disk state.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from argus.capture.manager import CameraManager
from argus.core.scheduler import TaskScheduler, create_job_processing_task
from argus.storage.database import Database

logger = structlog.get_logger()


def register_training_job_processing(
    scheduler: TaskScheduler,
    *,
    config,
    database: Database,
    baseline_manager,
    model_trainer,
    camera_manager: CameraManager | None = None,
) -> None:
    """Attach queued training-job execution to ``scheduler``.

    ``baseline_manager`` and ``model_trainer`` are constructed once in the
    entrypoint and shared with the scheduled retraining task so both paths
    point at the same on-disk state (and any future in-memory caches).

    The ``on_model_trained`` callback hot-reloads the new model into the
    running detection pipeline for the trained camera, so a successful
    training job takes effect without restarting the process.
    """
    from argus.anomaly.backbone_trainer import BackboneTrainer
    from argus.anomaly.job_executor import TrainingJobExecutor
    from argus.storage.model_registry import ModelRegistry

    def _on_model_trained(camera_id: str, model_path: Path) -> None:
        if camera_manager is None:
            return
        pipeline = camera_manager.get_pipeline(camera_id)
        if pipeline is not None:
            pipeline.reload_anomaly_model(model_path)
            logger.info(
                "training.model_hot_reloaded",
                camera_id=camera_id,
                model_path=str(model_path),
            )

    # Silence unused-import warnings — caller owns these lifecycles now.
    _ = baseline_manager

    backbone_trainer = BackboneTrainer(output_dir=config.storage.backbones_dir)
    model_registry = ModelRegistry(session_factory=database.get_session)
    job_executor = TrainingJobExecutor(
        database=database,
        trainer=model_trainer,
        backbone_trainer=backbone_trainer,
        model_registry=model_registry,
        baselines_dir=config.storage.baselines_dir,
        on_model_trained=_on_model_trained,
    )
    create_job_processing_task(scheduler, job_executor)
