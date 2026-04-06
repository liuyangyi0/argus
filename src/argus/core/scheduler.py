"""Periodic task scheduler for baseline refresh and model retraining.

Uses APScheduler to run maintenance tasks on configurable intervals:
- Baseline refresh: Capture new normal images from cameras
- Model retraining: Retrain anomaly models with latest baselines
- Stale camera check: Alert on cameras not producing frames
- Database cleanup: Remove old alert records
- Disk space monitoring: Warn when storage is running low
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

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
) -> None:
    """Register standard maintenance tasks."""
    alerts_path = Path(alerts_dir)

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


def create_retraining_task(
    scheduler: TaskScheduler,
    config,
    camera_configs: list,
    trainer,
    model_registry,
    baseline_manager,
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

                logger.info(
                    "retraining.triggered",
                    camera_id=camera_id,
                    new_images=new_images,
                    total_images=current_count,
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
                        model_registry.activate(version_id)
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
