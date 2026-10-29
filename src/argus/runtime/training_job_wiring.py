"""Wire queued training jobs into the scheduler.

Extracted from ``argus.__main__`` so the entrypoint no longer carries the
construction recipe for ``BackboneTrainer`` / ``TrainingJobExecutor`` /
``ModelRegistry``. The entrypoint passes in ``baseline_manager`` and
``model_trainer`` so the scheduled retraining loop and the queued-job
executor share the same on-disk state.
"""

from __future__ import annotations

from argus.capture.manager import CameraManager
from argus.core.scheduler import TaskScheduler, create_job_processing_task
from argus.storage.database import Database


def register_training_job_processing(
    scheduler: TaskScheduler,
    *,
    config,
    database: Database,
    baseline_manager,
    model_trainer,
    camera_manager: CameraManager | None = None,
    interval_seconds: float = 60.0,
) -> None:
    """Attach queued training-job execution to ``scheduler``.

    ``baseline_manager`` and ``model_trainer`` are constructed once in the
    entrypoint and shared with the scheduled retraining task so both paths
    point at the same on-disk state (and any future in-memory caches).

    A successful training job only registers a candidate.  There is no model
    path callback: even an apparently informational hook could reload runtime
    out of band and bypass ModelReleaseService.
    """
    from argus.runtime.task_training_resources import training_compute_exclusion_required
    if training_compute_exclusion_required(config, database):
        # Exclusive task training owns compute in this process. Keep the old
        # durable queue intact, but do not import/start its training engines.
        return
    from argus.anomaly.backbone_trainer import BackboneTrainer
    from argus.anomaly.job_executor import TrainingJobExecutor
    from argus.anomaly.trainer import ModelTrainer
    from argus.storage.model_registry import ModelRegistry

    # Silence unused-import warnings — caller owns these lifecycles now.
    _ = baseline_manager

    backbone_trainer = BackboneTrainer(output_dir=config.storage.backbones_dir)
    model_registry = ModelRegistry(session_factory=database.get_session)
    if isinstance(model_trainer, ModelTrainer):
        # Registration must happen before a merged dataset's temporary
        # directory is released. Both scheduled and queued jobs share this
        # trainer and therefore this authoritative database identity.
        model_trainer.bind_model_registry(model_registry)
    job_executor = TrainingJobExecutor(
        database=database,
        trainer=model_trainer,
        backbone_trainer=backbone_trainer,
        model_registry=model_registry,
        baselines_dir=config.storage.baselines_dir,
        camera_manager=camera_manager,
    )
    create_job_processing_task(
        scheduler,
        job_executor,
        interval_seconds=interval_seconds,
    )
