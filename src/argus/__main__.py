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
from argus.core.event_bus import EventBus
from argus.core.health import HealthMonitor
from argus.core.metrics import METRICS
from argus.core.scheduler import (
    TaskScheduler,
    create_backbone_retraining_task,
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
from argus.storage.inference_buffer import InferenceBuffer
from argus.storage.inference_store import InferenceRecordStore

logger = structlog.get_logger()


def _register_training_job_processing(
    scheduler: TaskScheduler,
    *,
    config,
    database: Database,
    camera_manager: CameraManager | None = None,
) -> None:
    """Attach queued training job execution to the scheduler."""
    from argus.anomaly.baseline import BaselineManager
    from argus.anomaly.backbone_trainer import BackboneTrainer
    from argus.anomaly.job_executor import TrainingJobExecutor
    from argus.anomaly.trainer import ModelTrainer
    from argus.storage.model_registry import ModelRegistry

    # Hot-reload callback: when training completes, load the new model
    # into the running detection pipeline for the trained camera.
    def _on_model_trained(camera_id: str, model_path: Path) -> None:
        if camera_manager is None:
            return
        pipeline = camera_manager.get_pipeline(camera_id)
        if pipeline is not None:
            pipeline.reload_anomaly_model(model_path)
            logger.info("training.model_hot_reloaded", camera_id=camera_id, model_path=str(model_path))

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
        on_model_trained=_on_model_trained,
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


def _log_gpu_environment() -> None:
    """Log GPU/CUDA availability at startup for operator visibility."""
    import cv2

    # PyTorch / CUDA
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_properties(0)
            logger.info(
                "env.cuda_available",
                device=torch.cuda.get_device_name(0),
                memory_mb=getattr(dev, 'total_memory', getattr(dev, 'total_mem', 0)) // (1024 * 1024),
                cuda_version=torch.version.cuda,
                torch_version=torch.__version__,
            )
        else:
            logger.warning("env.cuda_unavailable", msg="CUDA not available — inference will run on CPU (slow)")
    except ImportError:
        logger.warning("env.torch_missing", msg="PyTorch not installed — GPU acceleration disabled")

    # OpenCV CUDA
    cuda_count = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, "cuda") else 0
    if cuda_count > 0:
        logger.info("env.opencv_cuda", devices=cuda_count)
    else:
        logger.info("env.opencv_cuda_unavailable", msg="OpenCV CUDA not available — cv2 ops will use CPU")


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

    # ── GPU / CUDA environment check ──────────────────────────────────
    _log_gpu_environment()

    METRICS.ensure_initialized()
    METRICS.app_info.info({"version": "0.2.0", "node_id": config.node_id})

    event_bus = EventBus()

    # Initialize subsystems
    db = Database(database_url=config.storage.database_url)
    db.initialize()

    # Create default admin user if no users exist
    if db.user_count() == 0:
        from argus.dashboard.auth import hash_password
        db.create_user("admin", hash_password("admin"), "admin", "管理员")
        logger.info("auth.default_user_created", username="admin", msg="Default admin user created (password: admin)")

    # Write-behind buffer for batched inference record DB persistence
    inference_buffer = InferenceBuffer(database=db, flush_seconds=60, max_size=1000)
    inference_buffer.start()

    health = HealthMonitor()

    dispatcher = AlertDispatcher(
        config=config.alerts,
        database=db,
        alerts_dir=config.storage.alerts_dir,
        audio_config=getattr(config.dashboard, "audio_alerts", None),
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

    # Repair recordings with broken PTS timestamps in background
    def _bg_repair():
        try:
            repaired = alert_recording_store.repair_all()
            if repaired > 0:
                logger.info("startup.recordings_repaired", count=repaired)
        except Exception:
            logger.warning("startup.recording_repair_failed", exc_info=True)

    threading.Thread(target=_bg_repair, name="recording-repair", daemon=True).start()

    # ── go2rtc frame-source redirection (Frigate-style architecture) ──
    # USB cameras are exclusive devices — only one process can open them.
    # We let go2rtc own the USB device and re-stream via RTSP so the
    # pipeline reads from go2rtc instead of opening the device directly.
    # This also gives the browser WebRTC/MSE playback without MJPEG fallback.
    _go2rtc = None
    dashboard_cfg = getattr(config, "dashboard", None)
    if dashboard_cfg and dashboard_cfg.go2rtc_enabled:
        from argus.streaming.go2rtc_manager import Go2RTCManager, start_and_register_cameras
        _go2rtc = Go2RTCManager(
            api_port=dashboard_cfg.go2rtc_api_port,
            rtsp_port=dashboard_cfg.go2rtc_rtsp_port,
            binary_path=dashboard_cfg.go2rtc_binary,
        )
        try:
            start_and_register_cameras(_go2rtc, cameras)
        except Exception as exc:
            logger.warning("go2rtc.start_failed", error=str(exc))
            _go2rtc = None

    # Active learning sampler — subscribes to FrameAnalyzed events and
    # pushes high-uncertainty frames to the labeling queue for operators.
    from argus.core.active_learning import ActiveLearningSampler, ActiveLearningConfig
    active_learning_sampler = ActiveLearningSampler(
        config=ActiveLearningConfig(enabled=True),
        event_bus=event_bus,
        database=db,
    )
    # Attach sampler to event_bus so pipeline can buffer frames for saving
    event_bus._active_sampler = active_learning_sampler  # type: ignore[attr-defined]

    # External sensor fusion — shared between CameraManager (for grader)
    # and create_app (for HTTP API). Build it here so both sides point at
    # the same in-memory store.
    from argus.sensors.fusion import SensorFusion
    fusion_cfg = getattr(config, "sensor_fusion", None)
    sensor_fusion = SensorFusion(
        enabled=bool(getattr(fusion_cfg, "enabled", False)),
        default_valid_for_s=float(getattr(fusion_cfg, "default_valid_for_s", 60.0)),
    )

    # Start camera manager with all cameras
    manager = CameraManager(
        cameras=cameras,
        alert_config=config.alerts,
        on_alert=on_alert,
        cross_camera_config=config.cross_camera,
        segmenter_config=config.segmenter,
        classifier_config=config.classifier,
        physics_config=config.physics,
        imaging_config=config.imaging,
        health_monitor=health,
        audit_logger=audit_log,
        record_store=record_store,
        database=db,
        alert_recording_store=alert_recording_store,
        event_bus=event_bus,
        sensor_fusion=sensor_fusion,
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
            go2rtc_instance=_go2rtc,
            sensor_fusion=sensor_fusion,
        )
        app.state.audit_logger = audit_logger
        app.state.recording_store = alert_recording_store  # FR-033: shared with pipelines
        app.state.active_learning_sampler = active_learning_sampler
        app.state.inference_buffer = inference_buffer

        # Wire camera status changes → WebSocket broadcast so dashboard
        # updates immediately when cameras start/stop (not just on poll).
        ws_mgr = getattr(app.state, "ws_manager", None)
        if ws_mgr is not None:
            manager._on_status_change = ws_mgr.broadcast
        if app.state.baseline_lifecycle is not None:
            app.state.baseline_lifecycle._audit = audit_logger
        app.state.backup_manager = backup_mgr

        # M6: Continuous recording + retention (if enabled)
        recording_manager = None
        retention_manager = None
        if config.continuous_recording.enabled:
            from argus.storage.continuous_recorder import ContinuousRecordingManager
            from argus.storage.retention import RetentionManager

            recording_manager = ContinuousRecordingManager(
                output_dir=config.continuous_recording.output_dir,
                segment_duration_hours=config.continuous_recording.segment_duration_hours,
                encoding_crf=config.continuous_recording.encoding_crf,
                encoding_preset=config.continuous_recording.encoding_preset,
                encoding_fps=config.continuous_recording.encoding_fps,
            )
            for cam_cfg in config.cameras:
                recording_manager.start_camera(cam_cfg.camera_id, cam_cfg.resolution)
            logger.info("main.continuous_recording_started", cameras=len(config.cameras))

            retention_manager = RetentionManager(
                continuous_recording_dir=config.continuous_recording.output_dir,
                alert_recording_dir=Path("data/recordings"),
                local_retention_days=config.continuous_recording.local_retention_days,
                archive_enabled=config.continuous_recording.archive_enabled,
                archive_path=config.continuous_recording.archive_path,
                archive_retention_days=config.continuous_recording.archive_retention_days,
                cleanup_interval_hours=config.continuous_recording.cleanup_interval_hours,
            )
            retention_manager.start()
            logger.info("main.retention_manager_started")

        app.state.recording_manager = recording_manager
        app.state.retention_manager = retention_manager

        # Wire continuous recorder into camera manager for frame feeding
        if recording_manager is not None:
            manager._continuous_recording_mgr = recording_manager

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
    _register_training_job_processing(scheduler, config=config, database=db, camera_manager=manager)

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
        create_backbone_retraining_task(
            scheduler=scheduler,
            config=config,
            database=db,
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

    _go2rtc_check_counter = 0

    try:
        while running and manager.is_running:
            time.sleep(1.0)

            # Update health and Prometheus metrics for each camera
            for status in manager.get_status():
                health.update_camera(
                    camera_id=status.camera_id,
                    connected=status.connected,
                    frames_captured=status.stats.frames_captured if status.stats else 0,
                    avg_latency_ms=status.stats.avg_latency_ms if status.stats else 0,
                )
                METRICS.camera_status.labels(camera_id=status.camera_id).set(
                    1.0 if status.connected else 0.0,
                )

            # Check go2rtc health every 10 seconds and auto-restart if crashed
            _go2rtc_check_counter += 1
            if _go2rtc is not None and _go2rtc_check_counter % 10 == 0:
                if not _go2rtc.running:
                    logger.error("go2rtc.crashed", msg="Process died, attempting restart")
                    try:
                        _go2rtc.start()
                        logger.info("go2rtc.auto_restarted")
                    except Exception as e:
                        logger.error("go2rtc.restart_failed", error=str(e))

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
                    logger.debug("shutdown.go2rtc_close_failed", exc_info=True)
        scheduler.stop()
        manager.stop_all()
        # Stop continuous recording AFTER cameras (no more frames to push)
        if recording_manager is not None:
            recording_manager.stop_all()
        if retention_manager is not None:
            retention_manager.stop()
        record_store.stop()
        inference_buffer.stop()  # flush remaining records before DB close
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
