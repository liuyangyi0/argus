"""Periodic task scheduler for baseline refresh and model retraining.

Uses APScheduler to run maintenance tasks on configurable intervals:
- Baseline refresh: Capture new normal images from cameras
- Model retraining: Retrain anomaly models with latest baselines
- Stale camera check: Alert on cameras not producing frames
- Database cleanup: Remove old alert records
- Disk space monitoring: Warn when storage is running low
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from argus.storage.models import TrainingJobRecord, TrainingJobStatus, TrainingJobType, TrainingTriggerType

if TYPE_CHECKING:
    from argus.core.health import HealthMonitor
    from argus.storage.database import Database

logger = structlog.get_logger()


class TaskScheduler:
    """Manages periodic maintenance tasks.

    Tasks are lightweight wrappers that can be registered and run
    on configurable intervals. Uses APScheduler if available,
    otherwise falls back to a simple threading.Timer approach.
    """

    def __init__(self):
        self._scheduler = None
        self._tasks: dict[str, dict] = {}

    def start(self) -> None:
        """Start the scheduler."""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler

            self._scheduler = BackgroundScheduler()
            self._scheduler.start()
            logger.info("scheduler.started", backend="apscheduler")
        except ImportError:
            logger.warning("scheduler.apscheduler_not_available", msg="Scheduler disabled")

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            logger.info("scheduler.stopped")

    def add_interval_task(
        self,
        task_id: str,
        func: callable,
        hours: float = 0,
        minutes: float = 0,
        seconds: float = 0,
    ) -> None:
        """Add a task that runs at fixed intervals."""
        if not self._scheduler:
            return

        kwargs = {}
        if hours:
            kwargs["hours"] = hours
        if minutes:
            kwargs["minutes"] = minutes
        if seconds:
            kwargs["seconds"] = seconds

        self._scheduler.add_job(
            func,
            trigger="interval",
            id=task_id,
            **kwargs,
            replace_existing=True,
        )
        self._tasks[task_id] = {"type": "interval", **kwargs}
        logger.info("scheduler.task_added", task_id=task_id, **kwargs)

    def remove_task(self, task_id: str) -> None:
        """Remove a scheduled task."""
        if self._scheduler:
            try:
                self._scheduler.remove_job(task_id)
            except Exception as e:
                logger.warning("scheduler.remove_failed", task_id=task_id, error=str(e))
        self._tasks.pop(task_id, None)


def create_maintenance_tasks(
    scheduler: TaskScheduler,
    database: Database | None = None,
    health_monitor: HealthMonitor | None = None,
    alerts_dir: str | Path = "data/alerts",
    retention_days: int = 90,
    inference_records_dir: str | Path = "data/inference_records",
    inference_retention_days: int = 7,
) -> None:
    """Register standard maintenance tasks."""
    alerts_path = Path(alerts_dir)
    inference_path = Path(inference_records_dir)

    # Check for stale cameras every 30 seconds
    if health_monitor:
        def check_stale():
            stale = health_monitor.check_stale_cameras(max_stale_seconds=60.0)
            if stale:
                logger.warning("maintenance.stale_cameras", cameras=stale)

        scheduler.add_interval_task("check_stale_cameras", check_stale, seconds=30)

    # Clean old alerts every 6 hours
    if database:
        def cleanup_alerts():
            try:
                count, image_paths = database.delete_old_alerts(days=retention_days)
                if count > 0:
                    # Delete image files from disk
                    deleted_files = 0
                    for img_path in image_paths:
                        try:
                            p = Path(img_path)
                            if p.exists():
                                p.unlink()
                                deleted_files += 1
                        except OSError as e:
                            logger.warning(
                                "maintenance.file_delete_failed",
                                path=img_path,
                                error=str(e),
                            )

                    # Remove empty date directories
                    _cleanup_empty_dirs(alerts_path)

                    logger.info(
                        "maintenance.cleanup_complete",
                        alerts_deleted=count,
                        files_deleted=deleted_files,
                        retention_days=retention_days,
                    )
                else:
                    logger.debug("maintenance.cleanup_nothing", retention_days=retention_days)
            except Exception as e:
                logger.error("maintenance.cleanup_failed", error=str(e))

        scheduler.add_interval_task("cleanup_alerts", cleanup_alerts, hours=6)

        def cleanup_inference_records():
            """Clean inference records from both DB and disk.

            Historical bug: this task only cleaned DB rows, never touched
            the date-partitioned directories under inference_records_dir,
            so at 30fps they grew unbounded until the partition hit 100%
            and inference_store.write_failed started firing. Now we also
            prune old date directories (``YYYY-MM-DD`` naming).
            """
            try:
                count = database.delete_old_inference_records(days=inference_retention_days)
                if count > 0:
                    logger.info(
                        "maintenance.inference_cleaned",
                        deleted=count,
                        retention_days=inference_retention_days,
                    )
            except Exception as e:
                logger.error("maintenance.inference_cleanup_failed", error=str(e))

            # Prune old date directories from disk
            if inference_path.is_dir():
                cutoff = (datetime.now() - timedelta(days=inference_retention_days)).strftime("%Y-%m-%d")
                removed_dirs = 0
                freed_bytes = 0
                for date_dir in inference_path.iterdir():
                    if not date_dir.is_dir() or not date_dir.name[:10].replace("-", "").isdigit():
                        continue
                    if date_dir.name >= cutoff:
                        continue
                    try:
                        size = sum(f.stat().st_size for f in date_dir.rglob("*") if f.is_file())
                        shutil.rmtree(date_dir, ignore_errors=True)
                        freed_bytes += size
                        removed_dirs += 1
                    except OSError as e:
                        logger.warning(
                            "maintenance.inference_dir_delete_failed",
                            path=str(date_dir),
                            error=str(e),
                        )
                if removed_dirs > 0:
                    logger.info(
                        "maintenance.inference_dirs_cleaned",
                        dirs=removed_dirs,
                        freed_mb=round(freed_bytes / (1024 * 1024), 1),
                        cutoff=cutoff,
                    )

        scheduler.add_interval_task("cleanup_inference", cleanup_inference_records, hours=6)

    # Monitor disk space every 5 minutes
    def check_disk_space():
        try:
            usage = shutil.disk_usage(alerts_path)
            free_mb = usage.free / (1024 * 1024)
            used_pct = (usage.used / usage.total) * 100
            if free_mb < 1000:  # Less than 1GB
                logger.error(
                    "maintenance.disk_critical",
                    free_mb=round(free_mb, 1),
                    used_pct=round(used_pct, 1),
                    path=str(alerts_path),
                )
            elif free_mb < 5000:  # Less than 5GB
                logger.warning(
                    "maintenance.disk_low",
                    free_mb=round(free_mb, 1),
                    used_pct=round(used_pct, 1),
                    path=str(alerts_path),
                )
        except OSError as e:
            logger.error("maintenance.disk_check_failed", error=str(e))

    scheduler.add_interval_task("check_disk_space", check_disk_space, minutes=5)


_GRADE_RANK = {"A": 4, "B": 3, "C": 2, "F": 1}


def create_job_processing_task(
    scheduler: TaskScheduler,
    job_executor,
) -> None:
    """Register a periodic task that processes queued training jobs.

    Jobs must be confirmed by a human before they enter 'queued' status.
    This task picks up queued jobs and executes them.
    """
    def _process_jobs():
        try:
            processed = job_executor.process_queued_jobs()
            if processed > 0:
                logger.info("scheduler.jobs_processed", count=processed)
        except Exception as e:
            logger.error("scheduler.job_processing_failed", error=str(e))

    scheduler.add_interval_task("process_training_jobs", _process_jobs, minutes=1)
    logger.info("scheduler.job_processing_registered")


def create_backbone_retraining_task(
    scheduler: TaskScheduler,
    config,
    database,
) -> None:
    """Register periodic backbone retraining check.

    Creates a pending TrainingJob when the backbone is due for retraining.
    Does NOT auto-execute — requires human confirmation.
    """
    retrain_cfg = config.retraining
    if not retrain_cfg.enabled:
        return

    interval_days = retrain_cfg.backbone_retrain_interval_days

    def _backbone_check():
        try:
            active_backbone = database.get_active_backbone()
            if active_backbone:
                age_days = (datetime.now(timezone.utc) - active_backbone.created_at).days
                if age_days < interval_days:
                    logger.debug(
                        "scheduler.backbone_not_due",
                        age_days=age_days,
                        interval=interval_days,
                    )
                    return

            existing = database.list_training_jobs(
                status=TrainingJobStatus.PENDING_CONFIRMATION.value,
                job_type=TrainingJobType.SSL_BACKBONE.value,
                limit=1,
            )
            if existing:
                return

            job_id = str(uuid.uuid4())[:12]
            record = TrainingJobRecord(
                job_id=job_id,
                job_type=TrainingJobType.SSL_BACKBONE.value,
                trigger_type=TrainingTriggerType.SCHEDULED.value,
                triggered_by="scheduler",
                confirmation_required=True,
                status=TrainingJobStatus.PENDING_CONFIRMATION.value,
                hyperparameters=json.dumps({
                    "backbone_type": retrain_cfg.backbone_type,
                    "epochs": 5,
                }),
            )
            database.save_training_job(record)
            logger.info(
                "scheduler.backbone_job_created",
                job_id=job_id,
                reason="backbone age exceeded interval",
            )

        except Exception as e:
            logger.error("scheduler.backbone_check_failed", error=str(e))

    scheduler.add_interval_task("backbone_retraining_check", _backbone_check, hours=24)
    logger.info(
        "scheduler.backbone_check_registered",
        interval_days=interval_days,
    )


def create_retraining_task(
    scheduler: TaskScheduler,
    config,
    camera_configs: list,
    trainer,
    model_registry,
    baseline_manager,
    feedback_manager=None,
) -> None:
    """Register a scheduled retraining task that closes the active learning loop.

    For each camera, checks if enough new baseline images have been added
    (e.g. from false-positive feedback) and triggers retraining if threshold met.
    Optionally auto-deploys if quality grade is sufficient.
    """
    if not config.retraining.enabled:
        return

    retrain_cfg = config.retraining
    min_grade_rank = _GRADE_RANK.get(retrain_cfg.auto_deploy_min_grade, 3)

    def _retraining_check():
        for cam in camera_configs:
            camera_id = cam.camera_id
            try:
                current_count = baseline_manager.count_images(camera_id, "default")
                if current_count < retrain_cfg.min_new_baselines:
                    logger.debug(
                        "retraining.skip_insufficient",
                        camera_id=camera_id,
                        images=current_count,
                        required=retrain_cfg.min_new_baselines,
                    )
                    continue

                # Check last training record to see if images have grown
                models = model_registry.list_models(camera_id=camera_id)
                last_image_count = 0
                if models:
                    # The latest model has metadata with image_count
                    last = models[0]
                    last_image_count = getattr(last, "image_count", 0) or 0

                new_images = current_count - last_image_count
                if new_images < retrain_cfg.min_new_baselines:
                    logger.debug(
                        "retraining.skip_no_new",
                        camera_id=camera_id,
                        new_images=new_images,
                        required=retrain_cfg.min_new_baselines,
                    )
                    continue

                # 痛点 2: dataset_strategy controls which baseline versions feed
                # the auto-retrain. current_only keeps the legacy single-version
                # behaviour; multi-version strategies are reserved for a follow-up
                # implementation that walks baseline_lifecycle / training_records.
                strategy = getattr(retrain_cfg, "dataset_strategy", "current_only")
                if strategy != "current_only":
                    logger.warning(
                        "retraining.dataset_strategy_unimplemented",
                        camera_id=camera_id,
                        requested=strategy,
                        fallback="current_only",
                    )

                logger.info(
                    "retraining.triggered",
                    camera_id=camera_id,
                    new_images=new_images,
                    total_images=current_count,
                    dataset_strategy=strategy,
                )

                result = trainer.train(
                    camera_id=camera_id,
                    zone_id="default",
                    model_type=cam.anomaly.model_type,
                    image_size=cam.anomaly.image_size[0]
                    if isinstance(cam.anomaly.image_size, (list, tuple))
                    else cam.anomaly.image_size,
                    export_format="openvino",
                    quantization=cam.anomaly.quantization,
                    anomaly_config=cam.anomaly,
                )

                grade = getattr(result, "quality_grade", "F") or "F"
                logger.info(
                    "retraining.complete",
                    camera_id=camera_id,
                    grade=grade,
                    status=getattr(result, "status", None),
                )

                # Mark pending feedback as processed for this camera
                if feedback_manager is not None:
                    try:
                        pending = feedback_manager.get_pending_for_training(
                            camera_id=camera_id,
                        )
                        if pending:
                            model_vid = getattr(result, "model_version_id", None) or "unknown"
                            ids = [fb.feedback_id for fb in pending]
                            count = feedback_manager.mark_batch_processed(ids, model_vid)
                            logger.info(
                                "retraining.feedback_processed",
                                camera_id=camera_id,
                                feedback_count=count,
                                model_version_id=model_vid,
                            )
                    except Exception as e:
                        logger.warning(
                            "retraining.feedback_processing_failed",
                            camera_id=camera_id,
                            error=str(e),
                        )

                # Auto-deploy if grade meets threshold
                if (
                    retrain_cfg.auto_deploy
                    and _GRADE_RANK.get(grade, 0) >= min_grade_rank
                    and getattr(result, "status", None) is not None
                    and result.status.value == "COMPLETE"
                ):
                    try:
                        version_id = model_registry.register(
                            camera_id=camera_id,
                            model_type=cam.anomaly.model_type,
                            model_path=str(getattr(result, "model_path", "")),
                            image_count=current_count,
                            quality_grade=grade,
                        )
                        model_registry.activate(version_id, allow_bypass=True)
                        logger.info(
                            "retraining.auto_deployed",
                            camera_id=camera_id,
                            version_id=version_id,
                            grade=grade,
                        )
                    except Exception as e:
                        logger.error(
                            "retraining.deploy_failed",
                            camera_id=camera_id,
                            error=str(e),
                        )
                elif retrain_cfg.auto_deploy:
                    logger.info(
                        "retraining.skip_deploy",
                        camera_id=camera_id,
                        grade=grade,
                        min_grade=retrain_cfg.auto_deploy_min_grade,
                    )

            except Exception as e:
                logger.error(
                    "retraining.camera_failed",
                    camera_id=camera_id,
                    error=str(e),
                )

        # Camera group retraining
        for group in getattr(config, "camera_groups", []):
            try:
                group_dir = baseline_manager.get_group_baseline_dir(
                    group.group_id, group.zone_id
                )
                if not group_dir.is_dir():
                    continue
                group_images = (
                    len(list(group_dir.glob("*.png")))
                    + len(list(group_dir.glob("*.jpg")))
                )
                if group_images < retrain_cfg.min_new_baselines:
                    continue

                logger.info(
                    "retraining.group_triggered",
                    group_id=group.group_id,
                    images=group_images,
                )
                # Use first member camera's anomaly config for model type
                group_cam = next(
                    (c for c in camera_configs if c.camera_id in group.camera_ids),
                    None,
                )
                if group_cam is None:
                    continue

                trainer.train(
                    camera_id=f"group:{group.group_id}",
                    zone_id=group.zone_id,
                    model_type=group_cam.anomaly.model_type,
                    image_size=group_cam.anomaly.image_size[0]
                    if isinstance(group_cam.anomaly.image_size, (list, tuple))
                    else group_cam.anomaly.image_size,
                    export_format="openvino",
                    quantization=group_cam.anomaly.quantization,
                    anomaly_config=group_cam.anomaly,
                    group_id=group.group_id,
                )
            except Exception as e:
                logger.error(
                    "retraining.group_failed",
                    group_id=group.group_id,
                    error=str(e),
                )

    scheduler.add_interval_task(
        "scheduled_retraining",
        _retraining_check,
        hours=retrain_cfg.interval_hours,
    )
    logger.info(
        "scheduler.retraining_registered",
        interval_hours=retrain_cfg.interval_hours,
        auto_deploy=retrain_cfg.auto_deploy,
        min_grade=retrain_cfg.auto_deploy_min_grade,
    )


def _cleanup_empty_dirs(base_dir: Path) -> None:
    """Remove empty subdirectories under base_dir."""
    if not base_dir.exists():
        return
    for child in sorted(base_dir.rglob("*"), reverse=True):
        if child.is_dir():
            try:
                child.rmdir()  # Only succeeds if empty
            except OSError:
                pass  # Not empty, skip
