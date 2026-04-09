"""CLI entry point for the Argus detection system."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import threading
import time

import logging

import structlog

from argus.alerts.dispatcher import AlertDispatcher
from argus.alerts.grader import Alert
from argus.capture.manager import CameraManager
from argus.config.loader import load_config
from argus.config.schema import AlertSeverity
from argus.core.health import HealthMonitor
from argus.core.scheduler import (
    TaskScheduler,
    create_job_processing_task,
    create_maintenance_tasks,
    create_retraining_task,
)
from argus.dashboard.app import create_app
from argus.dashboard.tasks import TaskManager
from argus.storage.audit import AuditLogger
from argus.storage.backup import BackupManager
from argus.storage.alert_recording import AlertRecordingStore
from argus.storage.database import Database
from argus.storage.inference_store import InferenceRecordStore

logger = structlog.get_logger()


def _register_training_job_processing(
    scheduler: TaskScheduler,
    *,
    config,
    database: Database,
) -> None:
    """Attach queued training job execution to the scheduler."""
    from argus.anomaly.baseline import BaselineManager
    from argus.anomaly.backbone_trainer import BackboneTrainer
    from argus.anomaly.job_executor import TrainingJobExecutor
    from argus.anomaly.trainer import ModelTrainer
    from argus.storage.model_registry import ModelRegistry

    baseline_manager = BaselineManager(baselines_dir=config.storage.baselines_dir)
    trainer = ModelTrainer(
        baseline_manager=baseline_manager,
        models_dir=config.storage.models_dir,
        exports_dir=config.storage.exports_dir,
    )
    backbone_trainer = BackboneTrainer(output_dir=config.storage.backbones_dir)
    model_registry = ModelRegistry(session_factory=database.get_session)
    job_executor = TrainingJobExecutor(
        database=database,
        trainer=trainer,
        backbone_trainer=backbone_trainer,
        model_registry=model_registry,
        baselines_dir=config.storage.baselines_dir,
    )
    create_job_processing_task(scheduler, job_executor)


def _setup_file_logging(config, log_level: int) -> None:
    """Configure rotating file log output alongside console."""
    from logging.handlers import RotatingFileHandler
    from pathlib import Path

    log_cfg = config.logging
    log_dir = Path(log_cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create a rotating file handler
    file_handler = RotatingFileHandler(
        filename=str(log_dir / "argus.log"),
        maxBytes=log_cfg.max_file_size_mb * 1024 * 1024,
        backupCount=log_cfg.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)

    # Use JSON format for machine-parseable logs
    if log_cfg.log_format == "json":
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=[
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
            ],
        )
    else:
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=False),
            foreign_pre_chain=[
                structlog.stdlib.add_log_level,
            ],
        )

    file_handler.setFormatter(formatter)

    # Add handler to root logger so all libraries also log to file
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(log_level)


_SEVERITY_COLORS = {
    AlertSeverity.INFO: "\033[36m",    # cyan
    AlertSeverity.LOW: "\033[33m",     # yellow
    AlertSeverity.MEDIUM: "\033[91m",  # light red
    AlertSeverity.HIGH: "\033[31;1m",  # bold red
}
_RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(
        prog="argus",
        description="Argus - Nuclear power plant foreign object visual detection system",
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--camera",
        help="Run only the specified camera ID (default: all cameras)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable the web dashboard",
    )
    args = parser.parse_args()

    # Configure structured logging with file rotation
    log_level = logging.DEBUG if args.verbose else logging.INFO

    # Console output always enabled
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Set up file logging with rotation (after config is loaded)
    _setup_file_logging(config, log_level)

    # Filter cameras if specified
    cameras = config.cameras
    if args.camera:
        cameras = [c for c in cameras if c.camera_id == args.camera]
        if not cameras:
            print(f"Error: Camera '{args.camera}' not found in config", file=sys.stderr)
            sys.exit(1)

    if not cameras:
        print("Error: No cameras configured", file=sys.stderr)
        sys.exit(1)

    # Initialize subsystems
    db = Database(database_url=config.storage.database_url)
    db.initialize()

    # Create default admin user if no users exist
    if db.user_count() == 0:
        from argus.dashboard.auth import hash_password
        db.create_user("admin", hash_password("admin"), "admin", "管理员")
        logger.info("auth.default_user_created", username="admin", msg="Default admin user created (password: admin)")

    health = HealthMonitor()

    dispatcher = AlertDispatcher(
        config=config.alerts,
        database=db,
        alerts_dir=config.storage.alerts_dir,
    )

    def on_alert(alert: Alert):
        # Console output
        color = _SEVERITY_COLORS.get(alert.severity, "")
        print(
            f"{color}[ALERT] {alert.alert_id} | "
            f"{alert.severity.value.upper():6s} | "
            f"Camera: {alert.camera_id} | "
            f"Zone: {alert.zone_id} | "
            f"Score: {alert.anomaly_score:.3f}"
            f"{_RESET}"
        )
        # Persist and dispatch
        dispatcher.dispatch(alert)
        health.record_alert()

    # 5.2: InferenceRecord persistence for audit trail
    audit_log = AuditLogger(database=db)
    record_store = InferenceRecordStore(
        base_dir=config.storage.inference_records_dir,
    )
    record_store.start()

    recordings_dir = str(Path(config.storage.alerts_dir).parent / "recordings")
    # Get video encoding config from first camera's ring_buffer config
    _rb_cfg = config.cameras[0].ring_buffer if config.cameras else None
    alert_recording_store = AlertRecordingStore(
        archive_dir=recordings_dir,
        video_crf=_rb_cfg.video_crf if _rb_cfg else 23,
        video_preset=_rb_cfg.video_preset if _rb_cfg else "veryfast",
    )

    # Start camera manager with all cameras
    manager = CameraManager(
        cameras=cameras,
        alert_config=config.alerts,
        on_alert=on_alert,
        cross_camera_config=config.cross_camera,
        segmenter_config=config.segmenter,
        classifier_config=config.classifier,
        health_monitor=health,
        audit_logger=audit_log,
        record_store=record_store,
        database=db,
        alert_recording_store=alert_recording_store,
    )

    # Graceful shutdown
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False
        print("\nShutting down...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start dashboard in a background thread
    dashboard_thread = None
    app = None
    if not args.no_dashboard:
        task_mgr = TaskManager(max_concurrent=2)
        audit_logger = AuditLogger(database=db)
        backup_mgr = BackupManager(
            database_url=config.storage.database_url,
            backup_dir="data/backups",
            max_backups=10,
        )
        app = create_app(
            database=db,
            camera_manager=manager,
            health_monitor=health,
            alerts_dir=config.storage.alerts_dir,
            config=config,
            config_path=args.config,
            task_manager=task_mgr,
        )
        app.state.audit_logger = audit_logger
        app.state.recording_store = alert_recording_store  # FR-033: shared with pipelines
        if app.state.baseline_lifecycle is not None:
            app.state.baseline_lifecycle._audit = audit_logger
        app.state.backup_manager = backup_mgr

        def run_dashboard():
            import uvicorn
            uvicorn.run(
                app,
                host=config.dashboard.host,
                port=config.dashboard.port,
                log_level="warning",
            )

        dashboard_thread = threading.Thread(target=run_dashboard, name="argus-dashboard", daemon=True)
        dashboard_thread.start()
        logger.info(
            "dashboard.started",
            url=f"http://localhost:{config.dashboard.port}",
        )

    # Start scheduler for maintenance tasks
    scheduler = TaskScheduler()
    scheduler.start()
    create_maintenance_tasks(
        scheduler,
        database=db,
        health_monitor=health,
        alerts_dir=config.storage.alerts_dir,
    )
    _register_training_job_processing(scheduler, config=config, database=db)

    # Scheduled retraining (C4 + A4: active learning loop)
    if config.retraining.enabled:
        from argus.anomaly.trainer import ModelTrainer
        from argus.storage.model_registry import ModelRegistry
        from argus.anomaly.baseline import BaselineManager

        baseline_mgr = BaselineManager(baselines_dir=config.storage.baselines_dir)
        model_trainer = ModelTrainer(
            baseline_manager=baseline_mgr,
            exports_dir=config.storage.exports_dir,
        )
        model_reg = ModelRegistry(session_factory=db.get_session)
        create_retraining_task(
            scheduler=scheduler,
            config=config,
            camera_configs=cameras,
            trainer=model_trainer,
            model_registry=model_reg,
            baseline_manager=baseline_mgr,
        )

    # Scheduled automatic backup every 6 hours
    if not args.no_dashboard:
        def _scheduled_backup():
            try:
                result = backup_mgr.create_backup(include_models=False)
                logger.info("backup.scheduled_ok", size_mb=result.get("size_mb"), duration=result.get("duration_seconds"))
            except Exception as e:
                logger.error("backup.scheduled_failed", error=str(e))

        scheduler.add_interval_task("auto_backup", _scheduled_backup, hours=6)

    # Start cameras
    logger.info("argus.starting", cameras=len(cameras))
    started = manager.start_all()

    if not started:
        print("Error: No cameras could be started", file=sys.stderr)
        db.close()
        sys.exit(1)

    logger.info("argus.running", cameras=len(started), msg="Press Ctrl+C to stop")

    try:
        while running and manager.is_running:
            time.sleep(1.0)

            # Update health for each camera
            for status in manager.get_status():
                health.update_camera(
                    camera_id=status.camera_id,
                    connected=status.connected,
                    frames_captured=status.stats.frames_captured if status.stats else 0,
                    avg_latency_ms=status.stats.avg_latency_ms if status.stats else 0,
                )

            # Periodic status in verbose mode
            if args.verbose:
                h = health.get_health()
                connected = sum(1 for c in h.cameras if c.connected)
                logger.info(
                    "health",
                    status=h.status.value,
                    cameras=f"{connected}/{len(h.cameras)}",
                    alerts=h.total_alerts,
                    uptime=f"{h.uptime_seconds:.0f}s",
                )
    finally:
        # Stop go2rtc FIRST — daemon thread may not get cleanup time
        if app is not None:
            _go2rtc = getattr(app.state, "go2rtc", None)
            if _go2rtc is not None:
                try:
                    _go2rtc.close()
                except Exception:
                    pass
        scheduler.stop()
        manager.stop_all()
        record_store.stop()
        dispatcher.close()
        db.close()

        # Final summary
        h = health.get_health()
        print(f"\nSession summary:")
        print(f"  Uptime:    {h.uptime_seconds:.0f}s")
        print(f"  Cameras:   {len(h.cameras)}")
        print(f"  Alerts:    {h.total_alerts}")
        for cam in h.cameras:
            print(f"  {cam.camera_id}: {cam.frames_captured} frames, {cam.avg_latency_ms:.1f}ms avg")


if __name__ == "__main__":
    main()
