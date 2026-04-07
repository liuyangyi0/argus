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

import cv2
import numpy as np
import structlog

import uuid

from argus.alerts.grader import Alert, AlertGrader, CusumSnapshot, DetectionType
from argus.core.alert_ring_buffer import AlertFrameBuffer, FrameSnapshot, compress_frame
from argus.anomaly.detector import AnomalibDetector, AnomalyResult, DetectorStatus
from argus.anomaly.drift import DriftDetector, DriftStatus
from argus.core.diagnostics import (
    DiagnosticsBuffer,
    FrameDiagnostics,
    FrameScoreRecord,
    StageResult,
)
from argus.core.degradation import LockState
from argus.core.inference_record import (
    ConformalLevel,
    FinalDecision,
    InferenceRecord,
    PrefilterDecision,
)
from argus.capture.camera import CameraCapture, FrameData
from argus.config.schema import AlertConfig, CameraConfig, ClassifierConfig, SegmenterConfig, ZonePriority
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
    frames_dropped_backpressure: int = 0
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
        on_drift: Callable[[str, dict], None] | None = None,
        classifier_config: ClassifierConfig | None = None,
        segmenter_config: SegmenterConfig | None = None,
        model_version_id: str | None = None,
        shared_anomaly_detector: object | None = None,
        shadow_runner: object | None = None,
        feedback_manager: object | None = None,
    ):
        self.camera_config = camera_config
        self._on_alert = on_alert
        self._on_drift = on_drift
        self._model_version_id = model_version_id
        self._shadow_runner = shadow_runner
        self._feedback_manager = feedback_manager
        self.stats = PipelineStats()
        self._classifier_config = classifier_config
        self._segmenter_config = segmenter_config

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
        # Use shared detector for multi-class dinomaly mode, else create per-camera
        if shared_anomaly_detector is not None:
            base_detector = shared_anomaly_detector
            logger.info(
                "pipeline.using_shared_detector",
                camera_id=camera_config.camera_id,
            )
        else:
            model_path = self._find_model(camera_config.camera_id)
            base_detector = AnomalibDetector(
                model_path=model_path,
                threshold=camera_config.anomaly.threshold,
                image_size=camera_config.anomaly.image_size,
                ssim_baseline_frames=camera_config.anomaly.ssim_baseline_frames,
                ssim_sensitivity=camera_config.anomaly.ssim_sensitivity,
                ssim_midpoint=camera_config.anomaly.ssim_midpoint,
                enable_calibration=camera_config.anomaly.enable_calibration,
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

        # Simplex safety channel (A3): dual-channel architecture
        if camera_config.simplex.enabled:
            from argus.prefilter.simple_detector import SimplexDetector

            self._simplex = SimplexDetector(
                diff_threshold=camera_config.simplex.diff_threshold,
                min_area_px=camera_config.simplex.min_area_px,
                min_static_seconds=camera_config.simplex.min_static_seconds,
                morph_kernel_size=camera_config.simplex.morph_kernel_size,
                match_radius_px=camera_config.simplex.match_radius_px,
            )
        else:
            self._simplex = None
        self._simplex_reference_set = False

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

        # Anti-absorption: time-based heartbeat (MED-05)
        self._heartbeat_seconds = camera_config.mog2.heartbeat_frames / max(camera_config.fps_target, 1)
        self._last_heartbeat_time = 0.0
        self._frame_counter = 0

        # Anti-absorption: anomaly region lock with time-based clearing (HIGH-03)
        self._locked = False
        self._lock_score_threshold = camera_config.mog2.lock_score_threshold
        self._lock_clear_threshold = camera_config.mog2.lock_score_threshold * 0.8  # HIGH-05: hysteresis
        self._lock_clear_seconds = camera_config.mog2.lock_clear_frames / max(camera_config.fps_target, 1)
        self._lock_last_below_time: float | None = None  # when score first dropped below clear threshold

        # Track person filter degradation (HIGH-04)
        self._person_filter_warned = False

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

        # FR-033: Alert ring buffer for replay recordings
        rb_cfg = camera_config.ring_buffer
        if rb_cfg.enabled:
            self._alert_ring_buffer: AlertFrameBuffer | None = AlertFrameBuffer(
                fps=camera_config.fps_target,
                pre_seconds=rb_cfg.pre_trigger_seconds,
                post_seconds=rb_cfg.post_trigger_seconds,
            )
            self._ring_buffer_jpeg_quality = rb_cfg.jpeg_quality
        else:
            self._alert_ring_buffer = None
            self._ring_buffer_jpeg_quality = 85

        # FR-033: Score sparkline for video wall (30 seconds at camera FPS)
        from collections import deque as _deque
        self._score_sparkline: _deque[float] = _deque(maxlen=camera_config.fps_target * 30)

        # DET-010: Auto learning mode state
        self._learning_start_time: float | None = None
        self._learning_duration: float = 0.0
        self._auto_learning_complete = False

        # 5.1/5.2: Per-frame inference tracking for CameraInferenceRunner
        self._last_inference_record: InferenceRecord | None = None
        self._last_detection_failed: bool = False

        # Drift monitoring: KS test on anomaly score distribution
        drift_cfg = camera_config.drift
        if drift_cfg.enabled:
            self._drift_detector = DriftDetector(
                reference_scores=None,  # collected during initial operation
                window_size=drift_cfg.test_window,
                check_interval=drift_cfg.test_window,
                ks_threshold=drift_cfg.ks_threshold,
                p_value_threshold=drift_cfg.p_value_threshold,
            )
            self._drift_reference_scores: list[float] = []
            self._drift_reference_size = drift_cfg.reference_window
            self._drift_reference_ready = False
            self._drift_cooldown_seconds = drift_cfg.cooldown_minutes * 60
            self._drift_last_alert_time = 0.0
        else:
            self._drift_detector = None

        # D1: Open vocabulary classifier (optional)
        self._classifier = None
        if classifier_config and classifier_config.enabled:
            try:
                from argus.anomaly.classifier import OpenVocabClassifier

                self._classifier = OpenVocabClassifier(
                    model_name=classifier_config.model_name,
                    vocabulary=classifier_config.vocabulary,
                )
                self._classifier_min_score = classifier_config.min_anomaly_score_to_classify
                self._classifier_high_risk = set(classifier_config.high_risk_labels)
                self._classifier_low_risk = set(classifier_config.low_risk_labels)
                logger.info(
                    "pipeline.classifier_configured",
                    camera_id=camera_config.camera_id,
                    model=classifier_config.model_name,
                    vocab_size=len(classifier_config.vocabulary),
                )
            except Exception as e:
                logger.warning(
                    "pipeline.classifier_init_failed",
                    camera_id=camera_config.camera_id,
                    error=str(e),
                )

        # D2: Instance segmentation (optional)
        self._segmenter = None
        if segmenter_config and segmenter_config.enabled:
            try:
                from argus.anomaly.segmenter import InstanceSegmenter

                self._segmenter = InstanceSegmenter(
                    model_size=segmenter_config.model_size,
                    min_mask_area_px=segmenter_config.min_mask_area_px,
                    timeout_seconds=segmenter_config.timeout_seconds,
                )
                self._segmenter_max_points = segmenter_config.max_points
                self._segmenter_min_score = segmenter_config.min_anomaly_score
                logger.info(
                    "pipeline.segmenter_configured",
                    camera_id=camera_config.camera_id,
                    model_size=segmenter_config.model_size,
                    max_points=segmenter_config.max_points,
                )
            except Exception as e:
                logger.warning(
                    "pipeline.segmenter_init_failed",
                    camera_id=camera_config.camera_id,
                    error=str(e),
                )

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
            heartbeat_interval=self._heartbeat_seconds,
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

        try:
            return self._process_frame_inner(frame_data, frame, start, diag)
        except Exception as e:
            # CRIT-03: Clear buffers on exception to prevent memory leak
            with self._latest_raw_frame_lock:
                self._latest_raw_frame = None
            with self._latest_frame_lock:
                self._latest_frame = None
            logger.error(
                "pipeline.process_error",
                camera_id=frame_data.camera_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    def _process_frame_inner(
        self, frame_data: FrameData, frame: np.ndarray, start: float, diag: FrameDiagnostics
    ) -> Alert | None:
        """Inner frame processing logic (extracted for CRIT-03 exception safety)."""
        current_mode = self.mode
        # Save raw frame before zone masking (need copy since zone_mask mutates frame)
        with self._latest_raw_frame_lock:
            self._latest_raw_frame = frame

        # Stage 0: Apply zone mask
        t0 = time.monotonic()
        frame = self._zone_mask.apply(frame)
        diag.stages.append(StageResult(
            stage_name="zone_mask", duration_ms=(time.monotonic() - t0) * 1000
        ))

        # Save latest frame reference — get_latest_frame() copies on read
        with self._latest_frame_lock:
            self._latest_frame = frame

        # Stage 1: Pre-filter (with heartbeat bypass and anomaly lock bypass)
        t1 = time.monotonic()
        # MED-05: Time-based heartbeat instead of frame-count
        now = time.monotonic()
        is_heartbeat = (now - self._last_heartbeat_time) >= self._heartbeat_seconds
        if is_heartbeat:
            self._last_heartbeat_time = now

        # CRIT-01/02: Read lock state atomically for prefilter decision
        with self._lock_state_lock:
            is_locked = self._locked
        skip_prefilter = is_heartbeat or is_locked

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
            if is_locked:
                self._prefilter.process(frame, learning_rate_override=0.0)
            else:
                self._prefilter.process(frame)
            if is_heartbeat:
                self.stats.frames_heartbeat += 1

        # Stage 2: Person filter
        t2 = time.monotonic()
        detection_result: ObjectDetectionResult = self._object_detector.detect(frame)

        # HIGH-04: Warn once if person filter is unavailable
        if not detection_result.filter_available and not self._person_filter_warned:
            self._person_filter_warned = True
            logger.warning(
                "pipeline.person_filter_offline",
                camera_id=self.camera_config.camera_id,
                msg="YOLO person filter unavailable — frames will not be filtered for humans",
            )

        if detection_result.has_persons and self.camera_config.person_filter.skip_frame_on_person:
            self.stats.frames_skipped_person += 1
            # HIGH-03: Still update lock state even when person filter skips
            self._update_lock_state_time(time.monotonic())
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

        # Feed anomaly score to drift detector
        if self._drift_detector is not None:
            self._feed_drift_score(
                anomaly_result.anomaly_score, frame_data.camera_id
            )

        # CRIT-04: Log detection failures
        if anomaly_result.detection_failed:
            logger.error(
                "pipeline.detection_failed",
                camera_id=frame_data.camera_id,
                frame_number=frame_data.frame_number,
                msg="Anomaly detection returned failure — frame not analyzed",
            )

        # Stage 3b: Simplex safety channel (dual-channel architecture)
        # Runs in parallel with Anomalib as a lightweight safety backup.
        # If Anomalib is unavailable, Simplex becomes the primary detector.
        simplex_detected = False
        if self._simplex is not None:
            try:
                t3b = time.monotonic()
                # Set reference frame from the first processed frame
                if not self._simplex_reference_set:
                    self._simplex.set_reference(frame)
                    self._simplex_reference_set = True
                    logger.info(
                        "pipeline.simplex_reference_set",
                        camera_id=frame_data.camera_id,
                    )

                simplex_result = self._simplex.detect(frame)
                simplex_detected = simplex_result.has_detection
                simplex_duration_ms = (time.monotonic() - t3b) * 1000

                diag.stages.append(StageResult(
                    stage_name="simplex",
                    duration_ms=simplex_duration_ms,
                    metadata={
                        "has_detection": simplex_detected,
                        "max_static_seconds": round(simplex_result.max_static_seconds, 1),
                        "static_region_count": len(simplex_result.static_regions),
                    },
                ))

                # Dual-channel result merging:
                # 1. If both detect anomaly -> boost confidence (multiply by 1.1, cap at 1.0)
                # 2. If only Simplex detects -> mark as anomalous (safety fallback)
                # 3. If Anomalib model not loaded (SSIM fallback) -> Simplex is primary
                if simplex_detected:
                    if anomaly_result.is_anomalous:
                        # Both channels agree: boost score
                        boosted = min(anomaly_result.anomaly_score * 1.1, 1.0)
                        anomaly_result = AnomalyResult(
                            anomaly_score=boosted,
                            anomaly_map=anomaly_result.anomaly_map,
                            is_anomalous=True,
                            threshold=anomaly_result.threshold,
                            detection_failed=anomaly_result.detection_failed,
                        )
                        diag.anomaly_score = boosted
                        logger.debug(
                            "pipeline.simplex_boost",
                            camera_id=frame_data.camera_id,
                            original_score=round(boosted / 1.1, 4),
                            boosted_score=round(boosted, 4),
                        )
                    else:
                        # Only Simplex detected: use as safety fallback
                        # Set a minimum anomaly score so the alert system can grade it
                        fallback_score = max(anomaly_result.anomaly_score, 0.6)
                        anomaly_result = AnomalyResult(
                            anomaly_score=fallback_score,
                            anomaly_map=anomaly_result.anomaly_map,
                            is_anomalous=True,
                            threshold=anomaly_result.threshold,
                            detection_failed=anomaly_result.detection_failed,
                        )
                        diag.anomaly_score = fallback_score
                        diag.is_anomalous = True
                        logger.info(
                            "pipeline.simplex_safety_detection",
                            camera_id=frame_data.camera_id,
                            anomalib_score=round(anomaly_result.anomaly_score, 4),
                            static_regions=len(simplex_result.static_regions),
                            max_static_seconds=round(simplex_result.max_static_seconds, 1),
                        )
            except Exception as e:
                # Simplex failure must never crash the pipeline
                logger.warning(
                    "pipeline.simplex_error",
                    camera_id=frame_data.camera_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )

        # Cache anomaly heatmap for overlay stream
        with self._latest_anomaly_map_lock:
            self._latest_anomaly_map = (
                anomaly_result.anomaly_map.copy()
                if anomaly_result.anomaly_map is not None
                else None
            )

        # Update anomaly lock state
        self._update_lock_state(anomaly_result)

        # YOLO-004: Determine hybrid detection type
        detection_type = DetectionType.ANOMALY
        detected_objects: list[dict] = []
        if detection_result.non_person_objects:
            detected_objects = [
                {
                    "class_name": obj.class_name,
                    "confidence": round(obj.confidence, 3),
                    "track_id": obj.track_id,
                    "bbox": [obj.x1, obj.y1, obj.x2, obj.y2],
                }
                for obj in detection_result.non_person_objects
            ]
            if anomaly_result.is_anomalous:
                detection_type = DetectionType.HYBRID
            else:
                detection_type = DetectionType.OBJECT

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

        # Skip alert if neither Anomalib nor YOLO detected anything
        if not anomaly_result.is_anomalous and not detected_objects:
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

        # D1: Open vocabulary classification on anomaly region
        classification_result: tuple[str, float] | None = None
        if (
            self._classifier is not None
            and anomaly_result.is_anomalous
            and anomaly_result.anomaly_score >= self._classifier_min_score
        ):
            try:
                t_cls = time.monotonic()
                bbox = self._extract_anomaly_bbox(anomaly_result.anomaly_map, frame.shape)
                classification_result = self._classifier.classify(frame, bbox=bbox)
                cls_duration_ms = (time.monotonic() - t_cls) * 1000
                diag.stages.append(StageResult(
                    stage_name="classifier",
                    duration_ms=cls_duration_ms,
                    metadata={
                        "label": classification_result[0] if classification_result else None,
                        "confidence": round(classification_result[1], 3) if classification_result else None,
                        "bbox": list(bbox) if bbox else None,
                    },
                ))
                if classification_result:
                    logger.info(
                        "pipeline.classified",
                        camera_id=frame_data.camera_id,
                        label=classification_result[0],
                        confidence=round(classification_result[1], 3),
                    )
            except Exception as e:
                logger.warning(
                    "pipeline.classifier_error",
                    camera_id=frame_data.camera_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )

        # Multi-zone alert grading with semantic context
        alert = self._evaluate_zones(
            frame_data, anomaly_result, frame,
            detection_type=detection_type,
            detected_objects=detected_objects,
        )

        # D1: Attach classification to alert and adjust severity
        if alert is not None and classification_result is not None:
            label, conf = classification_result
            alert.classification_label = label
            alert.classification_confidence = conf
            if label in self._classifier_high_risk:
                # Escalate: bump severity up one level
                alert = self._adjust_alert_severity(alert, escalate=True)
                alert.severity_adjusted_by_classifier = True
                logger.info(
                    "pipeline.classifier_escalated",
                    camera_id=frame_data.camera_id,
                    label=label,
                    severity=alert.severity.value,
                )
            elif label in self._classifier_low_risk:
                # Suppress: downgrade severity one level
                alert = self._adjust_alert_severity(alert, escalate=False)
                alert.severity_adjusted_by_classifier = True
                logger.info(
                    "pipeline.classifier_suppressed",
                    camera_id=frame_data.camera_id,
                    label=label,
                    severity=alert.severity.value,
                )

        # D2: Instance segmentation on anomaly peaks
        if (
            alert is not None
            and self._segmenter is not None
            and anomaly_result.is_anomalous
            and anomaly_result.anomaly_score >= self._segmenter_min_score
            and anomaly_result.anomaly_map is not None
        ):
            try:
                from argus.anomaly.segmenter import extract_peak_points

                t_seg = time.monotonic()
                peak_points = extract_peak_points(
                    anomaly_result.anomaly_map,
                    max_points=self._segmenter_max_points,
                    min_score=self._segmenter_min_score,
                )
                if peak_points:
                    # Scale peak points from anomaly map to frame coordinates
                    map_h, map_w = anomaly_result.anomaly_map.shape[:2]
                    frame_h, frame_w = frame.shape[:2]
                    scaled_points = [
                        (int(px * frame_w / map_w), int(py * frame_h / map_h))
                        for px, py in peak_points
                    ]
                    seg_result = self._segmenter.segment(frame, scaled_points)
                    seg_duration_ms = (time.monotonic() - t_seg) * 1000

                    diag.stages.append(StageResult(
                        stage_name="segmenter",
                        duration_ms=seg_duration_ms,
                        metadata={
                            "peak_points": len(peak_points),
                            "segments": seg_result.num_objects,
                            "total_area_px": seg_result.total_area_px,
                        },
                    ))

                    if seg_result.num_objects > 0:
                        alert.segmentation_count = seg_result.num_objects
                        alert.segmentation_total_area_px = seg_result.total_area_px
                        alert.segmentation_objects = [
                            {
                                "bbox": list(obj.bbox),
                                "area_px": obj.area_px,
                                "centroid": list(obj.centroid),
                                "confidence": round(obj.confidence, 3),
                            }
                            for obj in seg_result.objects
                        ]
                        logger.debug(
                            "pipeline.segmented",
                            camera_id=frame_data.camera_id,
                            segments=seg_result.num_objects,
                            total_area_px=seg_result.total_area_px,
                        )
            except Exception as e:
                # Segmentation failure must never block the pipeline
                logger.warning(
                    "pipeline.segmenter_error",
                    camera_id=frame_data.camera_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )

        self._update_latency(start)

        # Shadow runner: parallel scoring for release pipeline evaluation
        if self._shadow_runner is not None:
            try:
                self._shadow_runner.run_shadow(
                    frame=frame_data.frame,
                    production_score=anomaly_result.anomaly_score,
                    production_alerted=alert is not None,
                )
            except Exception as e:
                logger.warning(
                    "pipeline.shadow_runner_failed",
                    camera_id=self.camera_config.camera_id,
                    error=str(e),
                )

        if alert is not None:
            alert.model_version_id = self._model_version_id
            self.stats.alerts_emitted += 1
            if self._on_alert:
                self._on_alert(alert)

        diag.alert_emitted = alert is not None
        diag.total_duration_ms = (time.monotonic() - start) * 1000
        self._last_detection_failed = anomaly_result.detection_failed

        # 5.2: Build InferenceRecord only for non-NORMAL frames (avoid
        # uuid4/time_ns/dict-comprehension overhead on every frame)
        if alert is not None:
            final_decision = FinalDecision.ALERT
        elif current_mode == PipelineMode.LEARNING and anomaly_result.is_anomalous:
            final_decision = FinalDecision.SUPPRESSED
        elif anomaly_result.is_anomalous:
            final_decision = FinalDecision.INFO
        else:
            final_decision = FinalDecision.NORMAL

        if final_decision != FinalDecision.NORMAL:
            frame_id = uuid.uuid4().hex
            diag.frame_id = frame_id

            if skip_prefilter:
                prefilter_decision = (
                    PrefilterDecision.SKIPPED_HEARTBEAT if is_heartbeat
                    else PrefilterDecision.SKIPPED_LOCK
                )
            else:
                prefilter_decision = PrefilterDecision.PASSED

            if self._anomaly_detector._calibration_scores is not None:
                cal_level = ConformalLevel.INFO
                thresholds = self._alert_grader._config.severity_thresholds
                if anomaly_result.anomaly_score >= thresholds.high:
                    cal_level = ConformalLevel.HIGH
                elif anomaly_result.anomaly_score >= thresholds.medium:
                    cal_level = ConformalLevel.MEDIUM
                elif anomaly_result.anomaly_score >= thresholds.low:
                    cal_level = ConformalLevel.LOW
            else:
                cal_level = ConformalLevel.NONE

            cusum_evidence = {}
            try:
                best_zone = (
                    self.camera_config.zones[0].zone_id
                    if self.camera_config.zones
                    else "default"
                )
                zone_key = f"{frame_data.camera_id}:{best_zone}"
                snap = self._alert_grader.get_cusum_state(zone_key)
                if snap is not None:
                    cusum_evidence[zone_key] = snap.evidence
            except Exception:
                pass

            self._last_inference_record = InferenceRecord(
                frame_id=frame_id,
                camera_id=frame_data.camera_id,
                timestamp_ns=time.time_ns(),
                model_version=self._model_version_id or "",
                prefilter_result=prefilter_decision,
                anomaly_score=anomaly_result.anomaly_score,
                cusum_evidence=cusum_evidence,
                conformal_level=cal_level,
                safety_channel_result=simplex_detected if self._simplex is not None else None,
                final_decision=final_decision,
                stage_durations_ms={s.stage_name: s.duration_ms for s in diag.stages},
            )
        else:
            self._last_inference_record = None

        self._diagnostics.append(diag)
        self._diagnostics.append_score(FrameScoreRecord(
            frame_number=frame_data.frame_number,
            timestamp=time.time(),
            anomaly_score=anomaly_result.anomaly_score,
            was_alert=alert is not None,
        ))

        # FR-033: Append frame to ring buffer + sparkline
        self._score_sparkline.append(anomaly_result.anomaly_score)
        if self._alert_ring_buffer is not None:
            try:
                # Build CUSUM evidence snapshot for all zones
                rb_cusum: dict[str, float] = {}
                for zone_cfg in self.camera_config.zones:
                    zone_key = f"{frame_data.camera_id}:{zone_cfg.zone_id}"
                    snap = self._alert_grader.get_cusum_state(zone_key)
                    if snap is not None:
                        rb_cusum[zone_key] = snap.evidence
                if not rb_cusum:
                    default_key = f"{frame_data.camera_id}:default"
                    snap = self._alert_grader.get_cusum_state(default_key)
                    if snap is not None:
                        rb_cusum[default_key] = snap.evidence

                # Build YOLO persons list from detection_result
                rb_persons: list[dict] = []
                if detection_result is not None and hasattr(detection_result, "persons"):
                    rb_persons = [
                        {
                            "bbox": [int(p.x1), int(p.y1), int(p.x2), int(p.y2)]
                            if hasattr(p, "x1") else [],
                            "confidence": getattr(p, "confidence", 0),
                        }
                        for p in detection_result.persons
                    ]

                # §1.3: Store raw anomaly map — JPEG encoding deferred to solidify
                rb_heatmap_raw: np.ndarray | None = None
                if anomaly_result.anomaly_map is not None:
                    try:
                        amap = anomaly_result.anomaly_map
                        if len(amap.shape) == 2:
                            rb_heatmap_raw = (np.clip(amap, 0, 1) * 255).astype(np.uint8)
                        else:
                            rb_heatmap_raw = amap.copy() if amap.dtype == np.uint8 else amap
                    except Exception:
                        pass

                # §1.3: Capture all YOLO detections (not just persons)
                rb_all_boxes: list[dict] | None = None
                if detection_result is not None and hasattr(detection_result, "objects"):
                    rb_all_boxes = [
                        {
                            "bbox": [int(getattr(o, "x1", 0)), int(getattr(o, "y1", 0)),
                                     int(getattr(o, "x2", 0)), int(getattr(o, "y2", 0))],
                            "class": getattr(o, "class_name", ""),
                            "confidence": round(getattr(o, "confidence", 0), 2),
                        }
                        for o in detection_result.objects
                    ]

                frame_snap = FrameSnapshot(
                    timestamp=time.time(),
                    frame_jpeg=compress_frame(
                        frame, quality=self._ring_buffer_jpeg_quality
                    ),
                    anomaly_score=anomaly_result.anomaly_score,
                    simplex_score=getattr(simplex_result, "max_score", None) if simplex_detected else None,
                    cusum_evidence=rb_cusum,
                    yolo_persons=rb_persons,
                    frame_number=frame_data.frame_number,
                    heatmap_raw=rb_heatmap_raw,
                    yolo_boxes=rb_all_boxes,
                )
                self._alert_ring_buffer.append(frame_snap)
            except Exception:
                logger.debug(
                    "pipeline.ring_buffer_append_failed",
                    camera_id=frame_data.camera_id,
                    exc_info=True,
                )

        # FR-033: Check for expired post-captures and solidify on alert
        if self._alert_ring_buffer is not None:
            # Flush any expired post-trigger captures
            try:
                for expired_id in self._alert_ring_buffer.check_expired_captures():
                    post_frames = self._alert_ring_buffer.finish_post_capture(expired_id)
                    if post_frames:
                        rec_store = getattr(self, "_recording_store", None)
                        if rec_store is not None:
                            rec_store.append_post_frames(expired_id, post_frames)
            except Exception:
                logger.debug("pipeline.post_capture_flush_failed", exc_info=True)

        if alert is not None and self._alert_ring_buffer is not None:
            try:
                # §1.2.2: Detect overlap — if there's a pending post-capture
                # from a recent alert, link the new alert to it (shared ring buffer)
                linked_id: str | None = None
                pending = self._alert_ring_buffer.get_pending_captures()
                if pending:
                    linked_id = pending[0]  # link to most recent pending
                    logger.info(
                        "pipeline.linked_alert",
                        new_alert=alert.alert_id,
                        linked_to=linked_id,
                        camera_id=alert.camera_id,
                    )

                recording = self._alert_ring_buffer.solidify(
                    alert_id=alert.alert_id,
                    camera_id=alert.camera_id,
                    severity=alert.severity.value,
                    trigger_timestamp=alert.timestamp,
                    linked_alert_id=linked_id,
                )
                if recording is not None:
                    self._alert_ring_buffer.start_post_capture(
                        alert_id=alert.alert_id,
                        severity=alert.severity.value,
                        trigger_timestamp=alert.timestamp,
                    )
                    # Store recording reference on alert for dispatcher to pick up
                    alert._solidified_recording = recording  # type: ignore[attr-defined]
            except Exception:
                logger.warning(
                    "pipeline.ring_buffer_solidify_failed",
                    camera_id=frame_data.camera_id,
                    alert_id=alert.alert_id,
                    exc_info=True,
                )

        return alert

    def run_once(self) -> Alert | None:
        """Read one frame from the camera and process it."""
        frame_data = self._camera.read()
        if frame_data is None:
            if not self._camera.state.connected:
                self._camera.request_reconnect()
            return None
        return self.process_frame(frame_data)

    def get_wall_status(self) -> dict:
        """Return video wall tile data for this camera (Phase 3)."""
        sparkline = list(self._score_sparkline)
        # Downsample sparkline to ~30 points (1 per second) for the wall tile
        if len(sparkline) > 30:
            step = len(sparkline) / 30
            sparkline = [sparkline[int(i * step)] for i in range(30)]
        return {
            "camera_id": self.camera_config.camera_id,
            "name": self.camera_config.name,
            "current_score": sparkline[-1] if sparkline else 0.0,
            "score_sparkline": [round(s, 3) for s in sparkline],
        }

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
            self._lock_last_below_time = None
        # HIGH-01: Reset MOG2 background model after lock release
        self._prefilter.reset()
        logger.info("pipeline.lock_cleared", camera_id=self.camera_config.camera_id)

    @property
    def lock_state(self) -> LockState:
        """Current anomaly region lock state as LockState enum."""
        with self._lock_state_lock:
            if self._locked:
                if self._lock_last_below_time is not None:
                    return LockState.CLEARING
                return LockState.LOCKED
            return LockState.UNLOCKED

    @property
    def last_heartbeat_time(self) -> float:
        """Timestamp of the last heartbeat bypass."""
        return self._last_heartbeat_time

    @property
    def last_inference_record(self) -> InferenceRecord | None:
        """Last InferenceRecord built during process_frame (None for NORMAL frames)."""
        return self._last_inference_record

    @property
    def last_detection_failed(self) -> bool:
        """Whether the last anomaly detection call failed."""
        return self._last_detection_failed

    def get_cusum_states(self) -> dict[str, CusumSnapshot]:
        """Return CUSUM evidence snapshots for all active zones."""
        return self._alert_grader.get_all_cusum_states()

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
        if self._shadow_runner is not None:
            try:
                self._shadow_runner.flush()
            except Exception as e:
                logger.warning(
                    "pipeline.shadow_flush_failed",
                    camera_id=self.camera_config.camera_id,
                    error=str(e),
                )
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
        self, frame_data: FrameData, anomaly_result: AnomalyResult, frame: np.ndarray,
        detection_type: str = "anomaly", detected_objects: list[dict] | None = None,
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
                detection_type=detection_type,
                detected_objects=detected_objects,
            )

        # Evaluate all zones and return highest-severity alert; break ties by score (HIGH-06)
        best_alert = None
        best_rank = -1
        for zone in include_zones:
            alert = self._alert_grader.evaluate(
                camera_id=frame_data.camera_id,
                zone_id=zone.zone_id,
                zone_priority=zone.priority,
                anomaly_score=anomaly_result.anomaly_score,
                frame_number=frame_data.frame_number,
                frame=frame,
                anomaly_map=anomaly_result.anomaly_map,
                detection_type=detection_type,
                detected_objects=detected_objects,
            )
            if alert is not None:
                rank = _severity_rank(alert.severity)
                if best_alert is None or rank > best_rank or (
                    rank == best_rank and alert.anomaly_score > best_alert.anomaly_score
                ):
                    best_alert = alert
                    best_rank = rank

        return best_alert

    def _update_lock_state(self, anomaly_result: AnomalyResult) -> None:
        """Update the anomaly region lock based on detection results.

        HIGH-05: Uses hysteresis — lock engages at lock_score_threshold,
        clears only when score drops below lock_clear_threshold (80% of lock threshold).
        HIGH-03: Uses time-based clearing instead of frame count.
        """
        now = time.monotonic()
        with self._lock_state_lock:
            if anomaly_result.anomaly_score >= self._lock_score_threshold:
                if not self._locked:
                    logger.info(
                        "pipeline.lock_engaged",
                        camera_id=self.camera_config.camera_id,
                        score=round(anomaly_result.anomaly_score, 3),
                    )
                self._locked = True
                self._lock_last_below_time = None  # reset clear timer
            elif self._locked:
                # HIGH-05: Use lower hysteresis threshold for clearing
                if anomaly_result.anomaly_score < self._lock_clear_threshold:
                    if self._lock_last_below_time is None:
                        self._lock_last_below_time = now
                    elif (now - self._lock_last_below_time) >= self._lock_clear_seconds:
                        self._locked = False
                        self._lock_last_below_time = None
                        logger.info(
                            "pipeline.lock_auto_cleared",
                            camera_id=self.camera_config.camera_id,
                        )

    def _update_lock_state_time(self, now: float) -> None:
        """Update lock clear timer without a detection result (e.g., person-skipped frames).

        HIGH-03: Ensures lock can still clear even when frames are skipped by person filter.
        """
        with self._lock_state_lock:
            if self._locked and self._lock_last_below_time is not None:
                if (now - self._lock_last_below_time) >= self._lock_clear_seconds:
                    self._locked = False
                    self._lock_last_below_time = None
                    logger.info(
                        "pipeline.lock_auto_cleared",
                        camera_id=self.camera_config.camera_id,
                    )

    def _update_latency(self, start: float) -> None:
        elapsed_ms = (time.monotonic() - start) * 1000
        n = self.stats.frames_analyzed
        self.stats._latency_sum += elapsed_ms
        self.stats.avg_latency_ms = self.stats._latency_sum / n if n > 0 else 0

    # --- D1: Classifier helpers ----------------------------------------------

    @staticmethod
    def _extract_anomaly_bbox(
        anomaly_map: np.ndarray | None, frame_shape: tuple
    ) -> tuple[int, int, int, int] | None:
        """Extract bounding box around the peak anomaly region from heatmap.

        Returns (x, y, w, h) in frame coordinates, or None if no heatmap.
        """
        if anomaly_map is None or anomaly_map.size == 0:
            return None

        import cv2

        # Normalize heatmap to 0-255
        hmap = anomaly_map
        if hmap.ndim == 3:
            hmap = hmap.mean(axis=2)
        hmin, hmax = hmap.min(), hmap.max()
        if hmax - hmin < 1e-6:
            return None
        normalized = ((hmap - hmin) / (hmax - hmin) * 255).astype(np.uint8)

        # Threshold at 50% of peak to find anomaly region
        _, binary = cv2.threshold(normalized, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Take the largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Scale from heatmap to frame coordinates
        h_scale = frame_shape[0] / hmap.shape[0]
        w_scale = frame_shape[1] / hmap.shape[1]
        x = int(x * w_scale)
        y = int(y * h_scale)
        w = max(int(w * w_scale), 1)
        h = max(int(h * h_scale), 1)

        # Pad by 10% for context
        pad_x = int(w * 0.1)
        pad_y = int(h * 0.1)
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(frame_shape[1] - x, w + 2 * pad_x)
        h = min(frame_shape[0] - y, h + 2 * pad_y)

        return x, y, w, h

    @staticmethod
    def _adjust_alert_severity(alert: Alert, escalate: bool) -> Alert:
        """Adjust alert severity up or down by one level based on classifier output."""
        from argus.config.schema import AlertSeverity

        levels = [AlertSeverity.INFO, AlertSeverity.LOW, AlertSeverity.MEDIUM, AlertSeverity.HIGH]
        current_idx = levels.index(alert.severity) if alert.severity in levels else 0
        if escalate:
            new_idx = min(current_idx + 1, len(levels) - 1)
        else:
            new_idx = max(current_idx - 1, 0)
        alert.severity = levels[new_idx]
        return alert

    # --- Drift monitoring ---------------------------------------------------

    def _feed_drift_score(self, score: float, camera_id: str) -> None:
        """Feed an anomaly score to the drift detector.

        During the reference window collection phase, scores are accumulated
        to build the reference distribution. Once the reference is ready,
        the DriftDetector runs the KS test periodically (every test_window
        scores) and emits warnings when drift is detected.
        """
        if not self._drift_reference_ready:
            # Still collecting reference distribution
            self._drift_reference_scores.append(score)
            if len(self._drift_reference_scores) >= self._drift_reference_size:
                ref = np.array(self._drift_reference_scores)
                self._drift_detector.set_reference(ref)
                self._drift_reference_ready = True
                logger.info(
                    "drift.reference_ready",
                    camera_id=camera_id,
                    samples=len(self._drift_reference_scores),
                    mean=round(float(ref.mean()), 4),
                    std=round(float(ref.std()), 4),
                )
                # Clear the list to free memory
                self._drift_reference_scores.clear()
            return

        # Reference is ready — feed score to detector
        self._drift_detector.update(score)
        status = self._drift_detector.get_status()

        if status.is_drifted:
            now = time.monotonic()
            if (now - self._drift_last_alert_time) >= self._drift_cooldown_seconds:
                self._drift_last_alert_time = now
                logger.warning(
                    "drift.detected",
                    camera_id=camera_id,
                    p_value=round(status.p_value, 6),
                    ks_stat=round(status.ks_statistic, 4),
                    reference_mean=round(status.reference_mean, 4),
                    current_mean=round(status.current_mean, 4),
                )
                # Broadcast via WebSocket on "health" topic
                if self._on_drift:
                    self._on_drift("health", {
                        "type": "drift_warning",
                        "camera_id": camera_id,
                        "p_value": round(status.p_value, 6),
                        "ks_statistic": round(status.ks_statistic, 4),
                        "reference_mean": round(status.reference_mean, 4),
                        "current_mean": round(status.current_mean, 4),
                        "samples_collected": status.samples_collected,
                    })
                # Passive feedback: submit drift event to feedback queue
                if self._feedback_manager is not None:
                    try:
                        self._feedback_manager.submit_passive_feedback(
                            camera_id=camera_id,
                            zone_id="all",
                            source="drift",
                            notes=(
                                f"KS={status.ks_statistic:.4f} "
                                f"p={status.p_value:.6f} "
                                f"ref_mean={status.reference_mean:.4f} "
                                f"cur_mean={status.current_mean:.4f}"
                            ),
                            model_version_id=self._model_version_id,
                        )
                    except Exception:
                        logger.warning(
                            "feedback.drift_submit_failed",
                            camera_id=camera_id,
                            exc_info=True,
                        )

    def get_drift_status(self) -> DriftStatus | None:
        """Get current drift detection status, or None if disabled."""
        if self._drift_detector is None:
            return None
        status = self._drift_detector.get_status()
        # Include reference readiness info
        if not self._drift_reference_ready:
            status.samples_collected = len(self._drift_reference_scores)
        return status
