"""Multi-camera thread pool manager.

Orchestrates multiple camera pipelines, each running in its own thread.
Provides a centralized interface for starting, stopping, and monitoring
all cameras, and aggregates alerts from all pipelines.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import structlog

from argus.alerts.grader import Alert
from argus.config.schema import AlertConfig, CameraConfig
from argus.core.pipeline import DetectionPipeline, PipelineMode, PipelineStats

logger = structlog.get_logger()


@dataclass
class CameraStatus:
    """Status of a single camera pipeline."""

    camera_id: str
    name: str
    connected: bool = False
    running: bool = False
    stats: PipelineStats | None = None
    error: str | None = None


class CameraManager:
    """Manages multiple camera detection pipelines in a thread pool.

    Each camera gets its own DetectionPipeline running in a dedicated thread.
    Alerts from all cameras are routed through a single callback.

    Usage:
        manager = CameraManager(cameras, alert_config, on_alert=handle_alert)
        manager.start_all()
        # ... system runs ...
        manager.stop_all()
    """

    def __init__(
        self,
        cameras: list[CameraConfig],
        alert_config: AlertConfig,
        on_alert: Callable[[Alert], None] | None = None,
        on_status_change: Callable[[str, dict], None] | None = None,
    ):
        self._cameras = cameras
        self._alert_config = alert_config
        self._on_alert = on_alert
        self._on_status_change = on_status_change
        self._pipelines: dict[str, DetectionPipeline] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._alert_count = 0
        self._last_frame_counts: dict[str, int] = {}  # camera_id -> last known frame count

        # B1-5: Shared anomaly detector for dinomaly_multi_class mode
        self._shared_anomaly_detector = None

        # DET-012: Pre-load shared YOLO model for all pipelines
        self._shared_yolo = None
        if cameras:
            try:
                from argus.person.detector import get_shared_yolo

                model_name = cameras[0].person_filter.model_name
                self._shared_yolo = get_shared_yolo(model_name)
            except Exception as e:
                logger.warning("manager.shared_yolo_failed", error=str(e))

    def start_all(self) -> list[str]:
        """Start all camera pipelines. Returns list of successfully started camera IDs."""
        self._stop_event.clear()
        started = []

        for cam_config in self._cameras:
            if self._start_camera(cam_config):
                started.append(cam_config.camera_id)

        logger.info(
            "manager.started",
            total=len(self._cameras),
            started=len(started),
            camera_ids=started,
        )
        return started

    def stop_all(self) -> None:
        """Stop all camera pipelines and wait for threads to finish."""
        logger.info("manager.stopping", cameras=len(self._threads))
        self._stop_event.set()

        for camera_id, pipeline in self._pipelines.items():
            pipeline.shutdown()

        for camera_id, thread in self._threads.items():
            thread.join(timeout=10.0)
            if thread.is_alive():
                logger.warning("manager.thread_timeout", camera_id=camera_id)

        self._pipelines.clear()
        self._threads.clear()
        logger.info("manager.stopped")

    def start_camera(self, camera_id: str) -> bool:
        """Start a single camera by ID."""
        cam_config = next((c for c in self._cameras if c.camera_id == camera_id), None)
        if cam_config is None:
            logger.error("manager.camera_not_found", camera_id=camera_id)
            return False
        return self._start_camera(cam_config)

    def stop_camera(self, camera_id: str) -> None:
        """Stop a single camera by ID."""
        with self._lock:
            pipeline = self._pipelines.pop(camera_id, None)
            thread = self._threads.pop(camera_id, None)

        if pipeline:
            pipeline.shutdown()
        if thread:
            thread.join(timeout=10.0)

    def get_status(self) -> list[CameraStatus]:
        """Get status of all configured cameras."""
        statuses = []
        for cam_config in self._cameras:
            pipeline = self._pipelines.get(cam_config.camera_id)
            if pipeline:
                statuses.append(CameraStatus(
                    camera_id=cam_config.camera_id,
                    name=cam_config.name,
                    connected=pipeline._camera.state.connected,
                    running=cam_config.camera_id in self._threads,
                    stats=pipeline.stats,
                ))
            else:
                statuses.append(CameraStatus(
                    camera_id=cam_config.camera_id,
                    name=cam_config.name,
                ))
        return statuses

    def get_latest_frame(self, camera_id: str) -> np.ndarray | None:
        """Get the latest processed frame from a camera for live preview."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        return pipeline.get_latest_frame()

    def get_raw_frame(self, camera_id: str) -> np.ndarray | None:
        """Get unmasked raw frame for baseline capture."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        return pipeline.get_raw_frame()

    def set_pipeline_mode(self, camera_id: str, mode: PipelineMode) -> bool:
        """Set pipeline operating mode for a camera (DET-006)."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return False
        pipeline.set_mode(mode)
        return True

    def get_pipeline_mode(self, camera_id: str) -> str | None:
        """Get pipeline operating mode for a camera (DET-006)."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        return pipeline.mode.value

    def get_learning_progress(self, camera_id: str) -> dict | None:
        """Get learning mode progress for a camera (DET-010)."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        return pipeline.get_learning_progress()

    def get_diagnostics(self, camera_id: str, n: int = 50) -> list | None:
        """Get recent frame diagnostics for a camera (DET-008)."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        return pipeline.get_diagnostics_buffer().get_recent(n)

    def evaluate_threshold(self, camera_id: str, threshold: float) -> dict | None:
        """Evaluate how a new threshold would affect alert counts (DET-005)."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        return pipeline.get_diagnostics_buffer().evaluate_threshold(threshold)

    def get_detector_status(self, camera_id: str) -> dict | None:
        """Get anomaly detector status for a camera (DET-004)."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        status = pipeline.get_detector_status()
        return {
            "mode": status.mode,
            "model_path": status.model_path,
            "model_loaded": status.model_loaded,
            "threshold": status.threshold,
            "ssim_calibration_progress": status.ssim_calibration_progress,
            "ssim_calibrated": status.ssim_calibrated,
            "ssim_noise_floor": status.ssim_noise_floor,
        }

    def get_latest_anomaly_map(self, camera_id: str) -> np.ndarray | None:
        """Get the latest anomaly heatmap from a camera for overlay stream."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        return pipeline.get_latest_anomaly_map()

    def _notify_camera_status(self, camera_id: str) -> None:
        """Notify WebSocket subscribers of camera status change."""
        if not self._on_status_change:
            return
        try:
            pipeline = self._pipelines.get(camera_id)
            if not pipeline:
                return
            cam_config = next((c for c in self._cameras if c.camera_id == camera_id), None)
            s = pipeline.stats
            self._on_status_change("cameras", {
                "camera_id": camera_id,
                "name": cam_config.name if cam_config else camera_id,
                "connected": pipeline._camera.state.connected,
                "running": camera_id in self._threads,
                "stats": {
                    "frames_captured": s.frames_captured,
                    "frames_analyzed": s.frames_analyzed,
                    "anomalies_detected": s.anomalies_detected,
                    "alerts_emitted": s.alerts_emitted,
                    "avg_latency_ms": round(s.avg_latency_ms, 1),
                } if s else None,
            })
        except Exception as e:
            logger.debug("manager.notify_failed", error=str(e))

    @property
    def alert_count(self) -> int:
        return self._alert_count

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set() and len(self._threads) > 0

    def _start_camera(self, cam_config: CameraConfig) -> bool:
        """Initialize and start a single camera pipeline in a new thread."""
        camera_id = cam_config.camera_id

        def _alert_handler(alert: Alert):
            with self._lock:
                self._alert_count += 1
            if self._on_alert:
                self._on_alert(alert)

        pipeline = DetectionPipeline(
            camera_config=cam_config,
            alert_config=self._alert_config,
            on_alert=_alert_handler,
            shared_yolo_model=self._shared_yolo,
        )

        if not pipeline.initialize():
            logger.error("manager.init_failed", camera_id=camera_id)
            return False

        thread = threading.Thread(
            target=self._camera_loop,
            args=(camera_id, pipeline),
            name=f"argus-{camera_id}",
            daemon=True,
        )

        with self._lock:
            self._pipelines[camera_id] = pipeline
            self._threads[camera_id] = thread

        thread.start()
        logger.info("manager.camera_started", camera_id=camera_id, source=cam_config.source)
        return True

    def _camera_loop(self, camera_id: str, pipeline: DetectionPipeline) -> None:
        """Main loop for a single camera thread with frame-rate watchdog."""
        logger.info("camera_loop.started", camera_id=camera_id)
        last_frame_time = time.monotonic()
        # MED-03: Configurable watchdog timeout from camera config
        cam_config = next((c for c in self._cameras if c.camera_id == camera_id), None)
        watchdog_timeout = cam_config.watchdog_timeout if cam_config else 30.0

        try:
            while not self._stop_event.is_set():
                try:
                    alert = pipeline.run_once()

                    # Update watchdog timer on successful frame processing
                    if pipeline.stats.frames_captured > 0:
                        current = pipeline.stats.frames_captured
                        prev = self._last_frame_counts.get(camera_id, 0)
                        if current > prev:
                            last_frame_time = time.monotonic()
                            self._last_frame_counts[camera_id] = current
                            # Notify every 25 frames (~5s at 5fps) to avoid flooding
                            if current % 25 == 0:
                                self._notify_camera_status(camera_id)

                    # Watchdog: force reconnect on stale stream
                    if time.monotonic() - last_frame_time > watchdog_timeout:
                        logger.warning(
                            "watchdog.stale_stream",
                            camera_id=camera_id,
                            stale_seconds=watchdog_timeout,
                        )
                        pipeline._camera.reconnect()
                        last_frame_time = time.monotonic()

                except Exception as e:
                    logger.error(
                        "camera_loop.error",
                        camera_id=camera_id,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    self._stop_event.wait(1.0)
        except Exception as e:
            logger.error("camera_loop.fatal", camera_id=camera_id, error=str(e))
        finally:
            logger.info("camera_loop.stopped", camera_id=camera_id)
            with self._lock:
                self._threads.pop(camera_id, None)
