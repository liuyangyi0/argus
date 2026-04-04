"""Three-stage detection pipeline orchestrator.

Connects the camera capture, MOG2 pre-filter, YOLO person filter, and
Anomalib anomaly detector into a single processing pipeline per camera.

Enhancements over basic pipeline:
- Heartbeat full-frame detection bypasses MOG2 every N frames to catch
  static objects that MOG2 absorbs into its background model.
- Anomaly region locking prevents MOG2 from short-circuiting detection
  on confirmed anomalous areas until operator clears them.
- Zone masking applies include/exclude polygon regions.
- Multi-zone support evaluates each include zone independently.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import structlog

from argus.alerts.grader import Alert, AlertGrader
from argus.anomaly.detector import AnomalibDetector, AnomalyResult, DetectorStatus
from argus.core.diagnostics import (
    DiagnosticsBuffer,
    FrameDiagnostics,
    FrameScoreRecord,
    StageResult,
)
from argus.capture.camera import CameraCapture, FrameData
from argus.config.schema import AlertConfig, CameraConfig, ZonePriority
from argus.core.zone_mask import ZoneMaskEngine
from argus.person.detector import ObjectDetectionResult, YOLOObjectDetector
from argus.prefilter.mog2 import MOG2PreFilter, PreFilterResult

logger = structlog.get_logger()

_SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3}


class PipelineMode(str, Enum):
    """Operating modes for the detection pipeline (DET-006)."""

    ACTIVE = "active"  # Normal detection + alerts
    MAINTENANCE = "maintenance"  # MOG2 frozen, detection + alerts continue
    LEARNING = "learning"  # Full pipeline runs but alerts suppressed


def _severity_rank(severity) -> int:
    """Return numeric rank for alert severity (higher = more severe)."""
    return _SEVERITY_ORDER.get(severity.value, 0)


@dataclass
class PipelineStats:
    """Runtime statistics for the pipeline."""

    frames_captured: int = 0
    frames_skipped_no_change: int = 0
    frames_skipped_person: int = 0
    frames_analyzed: int = 0
    frames_heartbeat: int = 0
    anomalies_detected: int = 0
    alerts_emitted: int = 0
    avg_latency_ms: float = 0.0
    _latency_sum: float = 0.0


class DetectionPipeline:
    """Single-camera three-stage detection pipeline.

    Stage 0 (Zone Mask): Apply include/exclude polygon masks
    Stage 1 (MOG2): Skip frames with no changes (bypassed by heartbeat)
    Stage 2 (YOLO): Filter out persons
    Stage 3 (Anomalib): Detect anomalies

    Anti-absorption features:
    - Heartbeat: Every N frames, skip MOG2 and force full Anomalib analysis
    - Anomaly lock: Confirmed anomaly regions bypass MOG2 until cleared
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        alert_config: AlertConfig,
        on_alert: Callable[[Alert], None] | None = None,
        shared_yolo_model: object | None = None,
    ):
        self.camera_config = camera_config
        self._on_alert = on_alert
        self.stats = PipelineStats()

        # Stage 0: Zone masking
        self._zone_mask = ZoneMaskEngine(
            zones=camera_config.zones,
            frame_height=camera_config.resolution[1],
            frame_width=camera_config.resolution[0],
        )

        # Stage 1: Pre-filter
        self._prefilter = MOG2PreFilter(
            history=camera_config.mog2.history,
            var_threshold=camera_config.mog2.var_threshold,
            detect_shadows=camera_config.mog2.detect_shadows,
            change_pct_threshold=camera_config.mog2.change_pct_threshold,
            denoise=camera_config.mog2.denoise,
            enable_stabilization=camera_config.mog2.enable_stabilization,
        )

        # Stage 2: Object detection (YOLO-003: multi-class + tracking)
        self._object_detector = YOLOObjectDetector(
            model_name=camera_config.person_filter.model_name,
            confidence=camera_config.person_filter.confidence,
            skip_frame_on_person=camera_config.person_filter.skip_frame_on_person,
            shared_model=shared_yolo_model,
            classes_to_detect=camera_config.person_filter.classes_to_detect,
            enable_tracking=camera_config.person_filter.enable_tracking,
        )

        # Stage 3: Anomaly detector — auto-discover trained model
        model_path = self._find_model(camera_config.camera_id)
        base_detector = AnomalibDetector(
            model_path=model_path,
            threshold=camera_config.anomaly.threshold,
            image_size=camera_config.anomaly.image_size,
            ssim_baseline_frames=camera_config.anomaly.ssim_baseline_frames,
            ssim_sensitivity=camera_config.anomaly.ssim_sensitivity,
            ssim_midpoint=camera_config.anomaly.ssim_midpoint,
        )
        if camera_config.anomaly.enable_multiscale:
            from argus.anomaly.detector import MultiScaleDetector

            self._anomaly_detector = MultiScaleDetector(
                base_detector=base_detector,
                tile_size=camera_config.anomaly.tile_size,
                tile_overlap=camera_config.anomaly.tile_overlap,
            )
            logger.info(
                "pipeline.multiscale_enabled",
                camera_id=camera_config.camera_id,
                tile_size=camera_config.anomaly.tile_size,
                tile_overlap=camera_config.anomaly.tile_overlap,
            )
        else:
            self._anomaly_detector = base_detector

        # Alert grading
        self._alert_grader = AlertGrader(config=alert_config)

        # Camera
        self._camera = CameraCapture(
            camera_id=camera_config.camera_id,
            source=camera_config.source,
            protocol=camera_config.protocol,
            fps_target=camera_config.fps_target,
            resolution=camera_config.resolution,
            reconnect_delay=camera_config.reconnect_delay,
            max_reconnect_attempts=camera_config.max_reconnect_attempts,
        )

        # Anti-absorption: heartbeat counter
        self._heartbeat_interval = camera_config.mog2.heartbeat_frames
        self._frame_counter = 0

        # Anti-absorption: anomaly region lock (from config)
        self._locked = False  # True when a high-confidence anomaly is active
        self._lock_score_threshold = camera_config.mog2.lock_score_threshold
        self._lock_clear_count = 0  # Consecutive normal frames needed to clear lock
        self._lock_clear_target = camera_config.mog2.lock_clear_frames

        # Thread safety for hot-updates
        self._config_lock = threading.Lock()

        # Thread safety for anomaly lock state (HIGH-03)
        self._lock_state_lock = threading.Lock()

        # Raw frame buffer for baseline capture, unmasked (CRIT-05)
        self._latest_raw_frame: np.ndarray | None = None
        self._latest_raw_frame_lock = threading.Lock()

        # Latest frame buffer for live preview (thread-safe)
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_lock = threading.Lock()

        # Latest anomaly heatmap for overlay stream (thread-safe)
        self._latest_anomaly_map: np.ndarray | None = None
        self._latest_anomaly_map_lock = threading.Lock()

        # DET-006: Pipeline operating mode
        self._mode = PipelineMode.ACTIVE
        self._mode_lock = threading.Lock()

        # DET-008: Per-frame diagnostics ring buffer
        self._diagnostics = DiagnosticsBuffer(maxlen=1500, score_maxlen=300)

        # DET-010: Auto learning mode state
        self._learning_start_time: float | None = None
        self._learning_duration: float = 0.0
        self._auto_learning_complete = False

    @staticmethod
    def _find_model(camera_id: str) -> Path | None:
        """Auto-discover the latest trained model for a camera.

        Search order: OpenVINO (.xml) > Torch (.pt) > Lightning (.ckpt)
        """
        models_base = Path("data/models") / camera_id
        if not models_base.exists():
            return None
        # Prefer OpenVINO exports (fastest inference)
        for pattern in ("model.xml", "model.pt", "model.ckpt"):
            matches = sorted(
                models_base.rglob(pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if matches:
                return matches[0]
        return None

    def initialize(self) -> bool:
        """Initialize all pipeline components. Returns True on success."""
        if not self._camera.connect():
            return False

        self._anomaly_detector.load()

        # DET-010: Auto-enter learning mode on first start
        fps = max(1, self.camera_config.fps_target)
        history = self.camera_config.mog2.history
        duration = max(history / fps * 3, 600)
        self._learning_duration = duration
        self._learning_start_time = time.monotonic()
        with self._mode_lock:
            self._mode = PipelineMode.LEARNING
        logger.info(
            "pipeline.learning_mode_auto",
            camera_id=self.camera_config.camera_id,
            duration_seconds=round(duration, 1),
        )

        logger.info(
            "pipeline.initialized",
            camera_id=self.camera_config.camera_id,
            anomaly_model_loaded=self._anomaly_detector.is_loaded,
            zones=len(self.camera_config.zones),
            heartbeat_interval=self._heartbeat_interval,
        )
        return True

    def process_frame(self, frame_data: FrameData) -> Alert | None:
        """Run the detection pipeline on a single frame."""
        start = time.monotonic()
        self.stats.frames_captured += 1
        self._frame_counter += 1
        frame = frame_data.frame

        # DET-008: Build diagnostics record for this frame
        diag = FrameDiagnostics(
            frame_number=frame_data.frame_number,
            timestamp=time.time(),
            camera_id=frame_data.camera_id,
        )

        # Validate frame before processing
        if frame is None or frame.size == 0 or len(frame.shape) != 3:
            logger.warning(
                "pipeline.invalid_frame",
                camera_id=frame_data.camera_id,
                frame_number=frame_data.frame_number,
                shape=getattr(frame, "shape", None),
            )
            return None

        # DET-010: Check learning mode auto-expiry
        if self._learning_start_time is not None:
            elapsed = time.monotonic() - self._learning_start_time
            if elapsed >= self._learning_duration:
                self._learning_start_time = None
                self._auto_learning_complete = True
                self.set_mode(PipelineMode.ACTIVE)
                logger.info("pipeline.learning_complete", camera_id=self.camera_config.camera_id)

        # DET-006: Snapshot current mode for this frame
        current_mode = self.mode
        diag.pipeline_mode = current_mode.value

        # Save raw frame before zone masking for baseline capture (CRIT-05)
        with self._latest_raw_frame_lock:
            self._latest_raw_frame = frame.copy()

        # Stage 0: Apply zone mask (exclude regions become black)
        t0 = time.monotonic()
        frame = self._zone_mask.apply(frame)
        diag.stages.append(StageResult(
            stage_name="zone_mask", duration_ms=(time.monotonic() - t0) * 1000
        ))

        # Save latest frame for live preview
        with self._latest_frame_lock:
            self._latest_frame = frame.copy()

        # Stage 1: Pre-filter (with heartbeat bypass and anomaly lock bypass)
        t1 = time.monotonic()
        is_heartbeat = (self._frame_counter % self._heartbeat_interval) == 0
        with self._lock_state_lock:
            locked_snapshot = self._locked
        skip_prefilter = is_heartbeat or locked_snapshot

        # DET-006: MAINTENANCE mode freezes MOG2 learning on all frames
        freeze_mog2 = current_mode == PipelineMode.MAINTENANCE

        mog2_skipped = False
        if not skip_prefilter:
            if freeze_mog2:
                prefilter_result: PreFilterResult = self._prefilter.process(
                    frame, learning_rate_override=0.0
                )
            else:
                prefilter_result = self._prefilter.process(frame)
            if not prefilter_result.has_change:
                self.stats.frames_skipped_no_change += 1
                mog2_skipped = True
                diag.stages.append(StageResult(
                    stage_name="mog2",
                    duration_ms=(time.monotonic() - t1) * 1000,
                    skipped=True,
                    skip_reason="no_change",
                ))
                diag.total_duration_ms = (time.monotonic() - start) * 1000
                self._diagnostics.append(diag)
                return None
        else:
            if locked_snapshot or freeze_mog2:
                self._prefilter.process(frame, learning_rate_override=0.0)
            else:
                self._prefilter.process(frame)
            if is_heartbeat:
                self.stats.frames_heartbeat += 1

        if not mog2_skipped:
            diag.stages.append(StageResult(
                stage_name="mog2",
                duration_ms=(time.monotonic() - t1) * 1000,
                metadata={
                    "heartbeat": is_heartbeat,
                    "locked": locked_snapshot,
                    "frozen": freeze_mog2,
                },
            ))

        # Stage 2: Object detection (YOLO-003: multi-class + tracking)
        t2 = time.monotonic()
        detection_result: ObjectDetectionResult = self._object_detector.detect(frame)
        if detection_result.has_persons and self.camera_config.person_filter.skip_frame_on_person:
            self.stats.frames_skipped_person += 1
            diag.stages.append(StageResult(
                stage_name="yolo",
                duration_ms=(time.monotonic() - t2) * 1000,
                skipped=True,
                skip_reason="person_detected",
                metadata={
                    "person_count": len(detection_result.persons),
                    "object_count": len(detection_result.objects),
                },
            ))
            diag.total_duration_ms = (time.monotonic() - start) * 1000
            self._diagnostics.append(diag)
            return None

        diag.stages.append(StageResult(
            stage_name="yolo",
            duration_ms=(time.monotonic() - t2) * 1000,
            metadata={
                "person_count": len(detection_result.persons),
                "object_count": len(detection_result.objects),
                "classes": [o.class_name for o in detection_result.non_person_objects],
                "track_ids": [o.track_id for o in detection_result.objects if o.track_id is not None],
            },
        ))

        analysis_frame = (
            detection_result.masked_frame if detection_result.masked_frame is not None else frame
        )

        # Stage 3: Anomaly detection
        t3 = time.monotonic()
        anomaly_result: AnomalyResult = self._anomaly_detector.predict(analysis_frame)
        self.stats.frames_analyzed += 1
        diag.stages.append(StageResult(
            stage_name="anomaly",
            duration_ms=(time.monotonic() - t3) * 1000,
            metadata={"score": round(anomaly_result.anomaly_score, 4)},
        ))
        diag.anomaly_score = anomaly_result.anomaly_score
        diag.is_anomalous = anomaly_result.is_anomalous

        # Cache anomaly heatmap for overlay stream
        with self._latest_anomaly_map_lock:
            self._latest_anomaly_map = (
                anomaly_result.anomaly_map.copy()
                if anomaly_result.anomaly_map is not None
                else None
            )

        # Update anomaly lock state
        self._update_lock_state(anomaly_result)

        # DET-006: LEARNING mode suppresses alerts
        if current_mode == PipelineMode.LEARNING:
            self._update_latency(start)
            diag.total_duration_ms = (time.monotonic() - start) * 1000
            self._diagnostics.append(diag)
            self._diagnostics.append_score(FrameScoreRecord(
                frame_number=frame_data.frame_number,
                timestamp=time.time(),
                anomaly_score=anomaly_result.anomaly_score,
                was_alert=False,
            ))
            return None

        if not anomaly_result.is_anomalous:
            self._update_latency(start)
            diag.total_duration_ms = (time.monotonic() - start) * 1000
            self._diagnostics.append(diag)
            self._diagnostics.append_score(FrameScoreRecord(
                frame_number=frame_data.frame_number,
                timestamp=time.time(),
                anomaly_score=anomaly_result.anomaly_score,
                was_alert=False,
            ))
            return None

        self.stats.anomalies_detected += 1

        # Multi-zone alert grading
        alert = self._evaluate_zones(frame_data, anomaly_result, frame)

        self._update_latency(start)

        if alert is not None:
            self.stats.alerts_emitted += 1
            if self._on_alert:
                self._on_alert(alert)

        diag.alert_emitted = alert is not None
        diag.total_duration_ms = (time.monotonic() - start) * 1000
        self._diagnostics.append(diag)
        self._diagnostics.append_score(FrameScoreRecord(
            frame_number=frame_data.frame_number,
            timestamp=time.time(),
            anomaly_score=anomaly_result.anomaly_score,
            was_alert=alert is not None,
        ))

        return alert

    def run_once(self) -> Alert | None:
        """Read one frame from the camera and process it."""
        frame_data = self._camera.read()
        if frame_data is None:
            if not self._camera.state.connected:
                self._camera.request_reconnect()
            return None
        return self.process_frame(frame_data)

    def get_latest_frame(self) -> np.ndarray | None:
        """Get a copy of the latest processed frame for live preview."""
        with self._latest_frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_raw_frame(self) -> np.ndarray | None:
        """Get unmasked raw frame for baseline capture."""
        with self._latest_raw_frame_lock:
            return self._latest_raw_frame.copy() if self._latest_raw_frame is not None else None

    def get_latest_anomaly_map(self) -> np.ndarray | None:
        """Get a copy of the latest anomaly heatmap for overlay stream."""
        with self._latest_anomaly_map_lock:
            return self._latest_anomaly_map.copy() if self._latest_anomaly_map is not None else None

    def clear_anomaly_lock(self) -> None:
        """Operator action: clear the anomaly region lock."""
        with self._lock_state_lock:
            self._locked = False
            self._lock_clear_count = 0
        logger.info("pipeline.lock_cleared", camera_id=self.camera_config.camera_id)

    @property
    def mode(self) -> PipelineMode:
        """Current pipeline operating mode (DET-006)."""
        with self._mode_lock:
            return self._mode

    def set_mode(self, mode: PipelineMode) -> None:
        """Set pipeline operating mode (DET-006)."""
        with self._mode_lock:
            old = self._mode
            self._mode = mode
        logger.info(
            "pipeline.mode_changed",
            camera_id=self.camera_config.camera_id,
            old=old.value,
            new=mode.value,
        )

    def get_learning_progress(self) -> dict:
        """Return learning mode progress for dashboard display (DET-010)."""
        if self._learning_start_time is None:
            return {"active": False, "complete": self._auto_learning_complete}
        elapsed = time.monotonic() - self._learning_start_time
        return {
            "active": True,
            "elapsed_seconds": round(elapsed, 1),
            "total_seconds": round(self._learning_duration, 1),
            "progress": min(1.0, elapsed / self._learning_duration) if self._learning_duration > 0 else 1.0,
            "complete": False,
        }

    def update_zones(self, zones: list) -> None:
        """Hot-update zone configuration without restart."""
        with self._config_lock:
            self.camera_config.zones = zones
            self._zone_mask.update_zones(zones)
        logger.info("pipeline.zones_updated", camera_id=self.camera_config.camera_id, zones=len(zones))

    def update_thresholds(self, anomaly_threshold: float | None = None) -> None:
        """Hot-update detection thresholds without restart."""
        with self._config_lock:
            if anomaly_threshold is not None:
                self._anomaly_detector.threshold = anomaly_threshold
        logger.info("pipeline.thresholds_updated", camera_id=self.camera_config.camera_id)

    def get_diagnostics_buffer(self) -> DiagnosticsBuffer:
        """Get the per-frame diagnostics ring buffer (DET-008)."""
        return self._diagnostics

    def get_detector_status(self) -> DetectorStatus:
        """Get anomaly detector operational status (DET-004)."""
        return self._anomaly_detector.get_status()

    def reload_anomaly_model(self, model_path: str | Path) -> bool:
        """Hot-reload the anomaly detection model without stopping the pipeline."""
        return self._anomaly_detector.hot_reload(Path(model_path))

    def shutdown(self) -> None:
        """Clean up all pipeline resources."""
        self._camera.stop()
        logger.info(
            "pipeline.shutdown",
            camera_id=self.camera_config.camera_id,
            stats={
                "captured": self.stats.frames_captured,
                "skipped_no_change": self.stats.frames_skipped_no_change,
                "skipped_person": self.stats.frames_skipped_person,
                "analyzed": self.stats.frames_analyzed,
                "heartbeats": self.stats.frames_heartbeat,
                "anomalies": self.stats.anomalies_detected,
                "alerts": self.stats.alerts_emitted,
                "avg_latency_ms": round(self.stats.avg_latency_ms, 1),
            },
        )

    def _evaluate_zones(
        self, frame_data: FrameData, anomaly_result: AnomalyResult, frame: np.ndarray
    ) -> Alert | None:
        """Evaluate anomaly against all configured include zones."""
        include_zones = self._zone_mask.get_include_zones()

        if not include_zones:
            # No zones configured: use default
            return self._alert_grader.evaluate(
                camera_id=frame_data.camera_id,
                zone_id="default",
                zone_priority=ZonePriority.STANDARD,
                anomaly_score=anomaly_result.anomaly_score,
                frame_number=frame_data.frame_number,
                frame=frame,
                anomaly_map=anomaly_result.anomaly_map,
            )

        # Evaluate all zones and return the highest-severity alert (HIGH-06)
        best_alert = None
        for zone in include_zones:
            alert = self._alert_grader.evaluate(
                camera_id=frame_data.camera_id,
                zone_id=zone.zone_id,
                zone_priority=zone.priority,
                anomaly_score=anomaly_result.anomaly_score,
                frame_number=frame_data.frame_number,
                frame=frame,
                anomaly_map=anomaly_result.anomaly_map,
            )
            if alert is not None:
                if best_alert is None or _severity_rank(alert.severity) > _severity_rank(best_alert.severity):
                    best_alert = alert

        return best_alert

    def _update_lock_state(self, anomaly_result: AnomalyResult) -> None:
        """Update the anomaly region lock based on detection results."""
        with self._lock_state_lock:
            if anomaly_result.anomaly_score >= self._lock_score_threshold:
                if not self._locked:
                    logger.info(
                        "pipeline.lock_engaged",
                        camera_id=self.camera_config.camera_id,
                        score=round(anomaly_result.anomaly_score, 3),
                    )
                self._locked = True
                self._lock_clear_count = 0
            elif self._locked:
                self._lock_clear_count += 1
                if self._lock_clear_count >= self._lock_clear_target:
                    self._locked = False
                    self._lock_clear_count = 0
                    logger.info(
                        "pipeline.lock_auto_cleared",
                        camera_id=self.camera_config.camera_id,
                    )

    def _update_latency(self, start: float) -> None:
        elapsed_ms = (time.monotonic() - start) * 1000
        n = self.stats.frames_analyzed
        self.stats._latency_sum += elapsed_ms
        self.stats.avg_latency_ms = self.stats._latency_sum / n if n > 0 else 0
