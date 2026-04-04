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
