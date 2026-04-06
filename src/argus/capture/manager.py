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

from argus.alerts.grader import Alert, AlertSeverity
from argus.config.schema import AlertConfig, AnomalyConfig, CameraConfig, ClassifierConfig, CrossCameraConfig, SegmenterConfig
from argus.core.correlation import CameraOverlapPair, CrossCameraCorrelator
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
        cross_camera_config: CrossCameraConfig | None = None,
        segmenter_config: SegmenterConfig | None = None,
        classifier_config: ClassifierConfig | None = None,
    ):
        self._cameras = cameras
        self._alert_config = alert_config
        self._on_alert = on_alert
        self._on_status_change = on_status_change
        self._segmenter_config = segmenter_config
        self._classifier_config = classifier_config
        self._pipelines: dict[str, DetectionPipeline] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._alert_count = 0
        self._last_frame_counts: dict[str, int] = {}  # camera_id -> last known frame count

        # Backpressure tracking
        self._pending_frames: dict[str, int] = {}
        self._frames_dropped: dict[str, int] = {}
        self._max_pending: int = 30
        self._bp_lock = threading.Lock()
        self._last_bp_warn: dict[str, float] = {}  # rate-limit warnings

        # B1-5: Shared anomaly detector for dinomaly_multi_class mode
        self._shared_anomaly_detector = None
        self._shared_detector_lock = threading.Lock()

        # C3: Cross-camera anomaly correlation
        self._correlator: CrossCameraCorrelator | None = None
        self._cross_camera_config = cross_camera_config
        if cross_camera_config and cross_camera_config.enabled and cross_camera_config.overlap_pairs:
            pairs = [
                CameraOverlapPair(
                    camera_a=p.camera_a,
                    camera_b=p.camera_b,
                    homography=p.homography,
                )
                for p in cross_camera_config.overlap_pairs
            ]
            self._correlator = CrossCameraCorrelator(
                pairs=pairs,
                corroboration_threshold=cross_camera_config.corroboration_threshold,
            )
            logger.info(
                "manager.cross_camera_enabled",
                pairs=len(pairs),
                threshold=cross_camera_config.corroboration_threshold,
            )

        # DET-012: Pre-load shared YOLO model for all pipelines
        self._shared_yolo = None
        if cameras:
            try:
                from argus.person.detector import get_shared_yolo

                model_name = cameras[0].person_filter.model_name
                self._shared_yolo = get_shared_yolo(model_name)
            except Exception as e:
                logger.warning("manager.shared_yolo_failed", error=str(e))

    def _get_shared_detector(self, anomaly_config: AnomalyConfig) -> object:
        """Get or create the shared anomaly detector for multi-class Dinomaly mode.

        Thread-safe: uses a lock to ensure the detector is created only once.
        Returns an AnomalibDetector instance shared across all cameras.
        """
        if self._shared_anomaly_detector is not None:
            return self._shared_anomaly_detector

        with self._shared_detector_lock:
            # Double-check after acquiring lock
            if self._shared_anomaly_detector is not None:
                return self._shared_anomaly_detector

            from argus.anomaly.detector import AnomalibDetector

            # For shared multi-class mode, find a trained model from any camera
            model_path = None
            for cam in self._cameras:
                candidate = DetectionPipeline._find_model(cam.camera_id)
                if candidate is not None:
                    model_path = candidate
                    break

            self._shared_anomaly_detector = AnomalibDetector(
                model_path=model_path,
                threshold=anomaly_config.threshold,
                image_size=anomaly_config.image_size,
                ssim_baseline_frames=anomaly_config.ssim_baseline_frames,
                ssim_sensitivity=anomaly_config.ssim_sensitivity,
                ssim_midpoint=anomaly_config.ssim_midpoint,
                enable_calibration=anomaly_config.enable_calibration,
            )
            logger.info(
                "manager.shared_detector_created",
                model_path=str(model_path),
                model_type=anomaly_config.model_type,
            )
            return self._shared_anomaly_detector

    def _create_shadow_runner(self, cam_config) -> object | None:
        """Create a ShadowRunner if a shadow-stage model exists for this camera."""
        try:
            db = getattr(self, "_db", None)
            if db is None:
                return None

            from argus.storage.model_registry import ModelRegistry
            from argus.storage.models import ModelStage

            registry = ModelRegistry(session_factory=db.get_session)
            shadow_models = registry.get_by_stage(
                cam_config.camera_id, ModelStage.SHADOW.value,
            )
            if not shadow_models:
                return None

            shadow_record = shadow_models[0]  # Most recent shadow model
            if not shadow_record.model_path:
                return None

            from pathlib import Path

            from argus.anomaly.shadow_runner import ShadowRunner

            # Get current production model version
            active = registry.get_active(cam_config.camera_id)
            prod_version = active.model_version_id if active else None

            sample_rate = getattr(cam_config.anomaly, "shadow_sample_rate", 5)

            shadow_model_path = Path(shadow_record.model_path)
            if not shadow_model_path.exists():
                logger.warning(
                    "manager.shadow_model_missing",
                    camera_id=cam_config.camera_id,
                    model_path=str(shadow_model_path),
                    shadow_version=shadow_record.model_version_id,
                )
                return None

            runner = ShadowRunner(
                shadow_model_path=shadow_model_path,
                shadow_version_id=shadow_record.model_version_id,
                production_version_id=prod_version,
                camera_id=cam_config.camera_id,
                session_factory=db.get_session,
                sample_rate=sample_rate,
                threshold=cam_config.anomaly.threshold,
            )
            logger.info(
                "manager.shadow_runner_created",
                camera_id=cam_config.camera_id,
                shadow_version=shadow_record.model_version_id,
            )
            return runner
        except Exception as e:
            logger.warning(
                "manager.shadow_runner_failed",
                camera_id=cam_config.camera_id,
                error=str(e),
            )
            return None

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

        for camera_id, pipeline in list(self._pipelines.items()):
            pipeline.shutdown()

        for camera_id, thread in list(self._threads.items()):
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

    def is_anomaly_locked(self, camera_id: str) -> bool:
        """Check if a camera's pipeline has an active anomaly lock."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return False
        with pipeline._lock_state_lock:
            return pipeline._locked

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

    def get_drift_status(self, camera_id: str) -> dict | None:
        """Get drift detection status for a camera."""
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        status = pipeline.get_drift_status()
        if status is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "is_drifted": status.is_drifted,
            "ks_statistic": round(status.ks_statistic, 4),
            "p_value": round(status.p_value, 6),
            "reference_mean": round(status.reference_mean, 4),
            "current_mean": round(status.current_mean, 4),
            "samples_collected": status.samples_collected,
        }

    def get_backpressure_stats(self) -> dict[str, dict]:
        """Get per-camera backpressure statistics."""
        with self._bp_lock:
            return {
                cam_id: {
                    "pending": self._pending_frames.get(cam_id, 0),
                    "dropped": self._frames_dropped.get(cam_id, 0),
                    "backpressured": self._pending_frames.get(cam_id, 0) >= self._max_pending,
                }
                for cam_id in [c.camera_id for c in self._cameras]
            }

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
            # C3: Cross-camera correlation check before emitting
            if self._correlator is not None:
                anomaly_map = pipeline.get_latest_anomaly_map()
                location = self._anomaly_peak_location(anomaly_map)
                max_age = (
                    self._cross_camera_config.max_age_seconds
                    if self._cross_camera_config
                    else 5.0
                )
                result = self._correlator.check(
                    camera_id=alert.camera_id,
                    anomaly_location=location,
                    timestamp=time.time(),
                    max_age_seconds=max_age,
                )
                alert.corroborated = result.corroborated
                alert.correlation_partner = result.partner_camera
                if not result.corroborated:
                    downgrade = (
                        self._cross_camera_config.uncorroborated_severity_downgrade
                        if self._cross_camera_config
                        else 1
                    )
                    if downgrade > 0:
                        alert.severity = self._downgrade_severity(
                            alert.severity, downgrade
                        )
                    logger.info(
                        "manager.alert_uncorroborated",
                        camera_id=alert.camera_id,
                        partner=result.partner_camera,
                        severity=alert.severity.value,
                    )

            with self._lock:
                self._alert_count += 1
            if self._on_alert:
                self._on_alert(alert)

        # B1-5: Use shared detector for dinomaly multi-class mode
        shared_detector = None
        anomaly_cfg = cam_config.anomaly
        if (
            anomaly_cfg.model_type == "dinomaly2"
            and anomaly_cfg.dinomaly_multi_class
        ):
            shared_detector = self._get_shared_detector(anomaly_cfg)

        # Shadow runner: check for shadow-stage model to evaluate
        shadow_runner = self._create_shadow_runner(cam_config)

        pipeline = DetectionPipeline(
            camera_config=cam_config,
            alert_config=self._alert_config,
            on_alert=_alert_handler,
            shared_yolo_model=self._shared_yolo,
            on_drift=self._on_status_change,
            segmenter_config=self._segmenter_config,
            classifier_config=self._classifier_config,
            shared_anomaly_detector=shared_detector,
            shadow_runner=shadow_runner,
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

        # Initialize backpressure counters
        with self._bp_lock:
            self._pending_frames[camera_id] = 0
            self._frames_dropped[camera_id] = 0

        try:
            while not self._stop_event.is_set():
                try:
                    # Backpressure check: skip non-heartbeat frames when overloaded
                    with self._bp_lock:
                        pending = self._pending_frames.get(camera_id, 0)
                    is_heartbeat = (
                        pipeline.stats.frames_captured > 0
                        and cam_config is not None
                        and getattr(cam_config.mog2, "heartbeat_frames", 0) > 0
                        and pipeline.stats.frames_captured % cam_config.mog2.heartbeat_frames == 0
                    )

                    if pending >= self._max_pending and not is_heartbeat:
                        with self._bp_lock:
                            self._frames_dropped[camera_id] = self._frames_dropped.get(camera_id, 0) + 1
                        pipeline.stats.frames_dropped_backpressure += 1
                        now = time.monotonic()
                        last_warn = self._last_bp_warn.get(camera_id, 0)
                        if now - last_warn > 60.0:
                            logger.warning(
                                "backpressure.dropping_frame",
                                camera_id=camera_id,
                                pending=pending,
                                total_dropped=self._frames_dropped.get(camera_id, 0),
                            )
                            self._last_bp_warn[camera_id] = now
                        self._stop_event.wait(0.01)  # Brief yield
                        continue

                    with self._bp_lock:
                        self._pending_frames[camera_id] = self._pending_frames.get(camera_id, 0) + 1

                    try:
                        alert = pipeline.run_once()
                    finally:
                        with self._bp_lock:
                            self._pending_frames[camera_id] = max(0, self._pending_frames.get(camera_id, 0) - 1)

                    # C3: Feed anomaly map to cross-camera correlator
                    if self._correlator is not None:
                        anomaly_map = pipeline.get_latest_anomaly_map()
                        if anomaly_map is not None:
                            self._correlator.update(
                                camera_id, anomaly_map, time.time()
                            )

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

    @staticmethod
    def _anomaly_peak_location(anomaly_map: np.ndarray | None) -> tuple[int, int]:
        """Find the peak anomaly location in a heatmap. Returns (x, y)."""
        if anomaly_map is None or anomaly_map.size == 0:
            return (0, 0)
        idx = np.argmax(anomaly_map)
        h, w = anomaly_map.shape[:2]
        y, x = divmod(int(idx), w)
        return (x, y)

    _SEVERITY_ORDER = [
        AlertSeverity.INFO,
        AlertSeverity.LOW,
        AlertSeverity.MEDIUM,
        AlertSeverity.HIGH,
    ]

    @classmethod
    def _downgrade_severity(cls, severity: AlertSeverity, levels: int) -> AlertSeverity:
        """Downgrade alert severity by N levels (minimum INFO)."""
        try:
            idx = cls._SEVERITY_ORDER.index(severity)
        except ValueError:
            return severity
        new_idx = max(0, idx - levels)
        return cls._SEVERITY_ORDER[new_idx]
