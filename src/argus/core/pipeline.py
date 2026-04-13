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
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import structlog

import uuid

# Default inference executor — used when no shared executor is injected.
# Prefer injecting a shared executor from CameraManager for proper lifecycle management.
_DEFAULT_INFERENCE_EXECUTOR: ThreadPoolExecutor | None = None
_DEFAULT_EXECUTOR_LOCK = threading.Lock()


def _get_default_inference_executor() -> ThreadPoolExecutor:
    """Lazy-init a fallback executor for standalone pipeline usage."""
    global _DEFAULT_INFERENCE_EXECUTOR
    if _DEFAULT_INFERENCE_EXECUTOR is None:
        with _DEFAULT_EXECUTOR_LOCK:
            if _DEFAULT_INFERENCE_EXECUTOR is None:
                _DEFAULT_INFERENCE_EXECUTOR = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="inference"
                )
    return _DEFAULT_INFERENCE_EXECUTOR

from argus.alerts.grader import Alert, AlertGrader, CusumSnapshot, DetectionType
from types import MappingProxyType

from argus.core.event_bus import (
    EventBus,
    FrameAnalyzed,
    AlertRaised,
)
from argus.core.metrics import METRICS
from argus.core.alert_ring_buffer import AlertFrameBuffer, FrameSnapshot, RecordingStatus, compress_frame
from argus.storage.models import AlertRecordingRecord
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
from argus.core.model_discovery import find_runtime_model, find_runtime_model_in_dir
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


class PipelineStats:
    """Runtime statistics for the pipeline.

    Thread-safe: all increments and reads go through a lock to prevent
    torn reads when the dashboard snapshots stats from a different thread.
    """

    __slots__ = (
        "_lock", "frames_captured", "frames_skipped_no_change",
        "frames_skipped_person", "frames_analyzed", "frames_heartbeat",
        "frames_dropped_backpressure", "frames_timeout", "anomalies_detected",
        "alerts_emitted", "avg_latency_ms", "_latency_sum",
        "current_fps", "_fps_last_count", "_fps_last_time",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.frames_captured: int = 0
        self.frames_skipped_no_change: int = 0
        self.frames_skipped_person: int = 0
        self.frames_analyzed: int = 0
        self.frames_heartbeat: int = 0
        self.frames_dropped_backpressure: int = 0
        self.frames_timeout: int = 0
        self.anomalies_detected: int = 0
        self.alerts_emitted: int = 0
        self.avg_latency_ms: float = 0.0
        self._latency_sum: float = 0.0
        # Real-time FPS: updated every 2 seconds from frame count delta
        self.current_fps: float = 0.0
        self._fps_last_count: int = 0
        self._fps_last_time: float = 0.0

    def update_fps(self) -> None:
        """Recompute current_fps from frame count delta (call once per frame)."""
        now = time.time()
        with self._lock:
            if self._fps_last_time == 0.0:
                self._fps_last_time = now
                self._fps_last_count = self.frames_captured
                return
            elapsed = now - self._fps_last_time
            if elapsed >= 2.0:
                delta = self.frames_captured - self._fps_last_count
                self.current_fps = round(delta / elapsed, 1)
                self._fps_last_count = self.frames_captured
                self._fps_last_time = now

    def snapshot(self) -> dict:
        """Return a consistent point-in-time copy of all counters."""
        with self._lock:
            return {
                "captured": self.frames_captured,
                "skipped_no_change": self.frames_skipped_no_change,
                "skipped_person": self.frames_skipped_person,
                "analyzed": self.frames_analyzed,
                "heartbeats": self.frames_heartbeat,
                "anomalies": self.anomalies_detected,
                "alerts": self.alerts_emitted,
                "avg_latency_ms": round(self.avg_latency_ms, 1),
                "current_fps": self.current_fps,
            }


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

    _TIMEOUT_DEGRADE_THRESHOLD = 5

    def __init__(
        self,
        camera_config: CameraConfig,
        alert_config: AlertConfig,
        on_alert: Callable[[Alert], None] | None = None,
        shared_yolo_model: object | None = None,
        on_drift: Callable[[str, dict], None] | None = None,
        classifier_config: ClassifierConfig | None = None,
        segmenter_config: SegmenterConfig | None = None,
        physics_config: object | None = None,
        imaging_config: object | None = None,
        model_version_id: str | None = None,
        model_path: Path | None = None,
        shared_anomaly_detector: object | None = None,
        shadow_runner: object | None = None,
        feedback_manager: object | None = None,
        recording_store: object | None = None,
        database: object | None = None,
        event_bus: EventBus | None = None,
        inference_executor: ThreadPoolExecutor | None = None,
    ):
        self.camera_config = camera_config
        self._on_alert = on_alert
        self._on_drift = on_drift
        self._event_bus = event_bus
        self._inference_executor = inference_executor or _get_default_inference_executor()
        self._model_version_id = model_version_id
        self._shadow_runner = shadow_runner
        self._feedback_manager = feedback_manager
        self._recording_store = recording_store
        self._database = database
        self.stats = PipelineStats()
        self._classifier_config = classifier_config
        self._segmenter_config = segmenter_config
        self._physics_config = physics_config
        self._imaging_config = imaging_config

        # Pre-cache Prometheus metric children (avoids per-frame .labels() lock)
        _cid = camera_config.camera_id
        self._m_latency = METRICS.inference_latency.labels(camera_id=_cid)
        self._m_frames = METRICS.frames_processed.labels(camera_id=_cid)
        self._m_skipped_mog2 = METRICS.frames_skipped.labels(camera_id=_cid, reason="mog2_no_change")
        _zone_id = camera_config.zones[0].zone_id if camera_config.zones else "default"
        self._m_anomaly_score = METRICS.anomaly_score.labels(camera_id=_cid, zone_id=_zone_id)
        # Stage metrics: pre-cache known stages, with fallback for dynamic ones
        self._m_stages: dict[str, object] = {
            name: METRICS.pipeline_stage_seconds.labels(camera_id=_cid, stage=name)
            for name in ("zone_mask", "mog2", "yolo", "anomaly", "simplex")
        }

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

        # Stage 2: Object detection (YOLO-003: multi-class + tracking + SAHI)
        self._object_detector = YOLOObjectDetector(
            model_name=camera_config.person_filter.model_name,
            confidence=camera_config.person_filter.confidence,
            skip_frame_on_person=camera_config.person_filter.skip_frame_on_person,
            shared_model=shared_yolo_model,
            classes_to_detect=camera_config.person_filter.classes_to_detect,
            enable_tracking=camera_config.person_filter.enable_tracking,
            sahi_enabled=camera_config.person_filter.sahi_enabled,
            sahi_slice_size=camera_config.person_filter.sahi_slice_size,
            sahi_overlap_ratio=camera_config.person_filter.sahi_overlap_ratio,
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
            model_path = model_path or self._find_model(camera_config.camera_id)
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
                pyramid_mode=camera_config.anomaly.pyramid_mode,
                pyramid_sizes=camera_config.anomaly.pyramid_sizes,
            )
            logger.info(
                "pipeline.multiscale_enabled",
                camera_id=camera_config.camera_id,
                tile_size=camera_config.anomaly.tile_size,
                tile_overlap=camera_config.anomaly.tile_overlap,
                pyramid_mode=camera_config.anomaly.pyramid_mode,
                pyramid_sizes=camera_config.anomaly.pyramid_sizes,
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
        # P2 fix: cap heartbeat to max_gap * 0.8 so CUSUM evidence can accumulate
        # even in static scenes where MOG2 drops all non-heartbeat frames
        raw_heartbeat = camera_config.mog2.heartbeat_frames / max(camera_config.fps_target, 1)
        max_gap = alert_config.temporal.max_gap_seconds
        self._heartbeat_seconds = min(raw_heartbeat, max_gap * 0.8)
        self._last_heartbeat_time = 0.0
        self._frame_counter = 0

        # Low-light mode: bypass MOG2 + shorten heartbeat when scene is dark
        ll_cfg = camera_config.low_light
        self._low_light_enabled = ll_cfg.enabled
        self._low_light_threshold = ll_cfg.brightness_threshold
        self._low_light_heartbeat_seconds = alert_config.temporal.max_gap_seconds / 2
        self._was_low_light = False
        self._prev_brightness: float | None = None
        self._brightness_jump_threshold = ll_cfg.brightness_jump_threshold

        # CLAHE preprocessing for low-light enhancement (<1ms/frame)
        self._clahe_enabled = ll_cfg.clahe_enabled and ll_cfg.enabled
        self._clahe: cv2.CLAHE | None = None
        if self._clahe_enabled:
            self._clahe = cv2.createCLAHE(
                clipLimit=ll_cfg.clahe_clip_limit,
                tileGridSize=(ll_cfg.clahe_grid_size, ll_cfg.clahe_grid_size),
            )

        # Anti-absorption: anomaly region lock with time-based clearing (HIGH-03)
        self._locked = False
        self._lock_score_threshold = camera_config.mog2.lock_score_threshold
        self._lock_clear_threshold = camera_config.mog2.lock_score_threshold * 0.8  # HIGH-05: hysteresis
        self._lock_clear_seconds = camera_config.mog2.lock_clear_frames / max(camera_config.fps_target, 1)
        self._lock_last_below_time: float | None = None  # when score first dropped below clear threshold

        # Track person filter degradation (HIGH-04)
        self._person_filter_warned = False

        # Consecutive inference timeout counter for degradation detection
        self._consecutive_timeouts = 0

        # Thread safety for hot-updates
        self._config_lock = threading.Lock()

        # Cached per-zone anomaly masks (invalidated on update_zones, bounded to 64)
        self._zone_mask_cache: OrderedDict[tuple[str, int, int], np.ndarray] = OrderedDict()
        self._zone_mask_cache_limit = 64

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

        # DET-008: Per-frame diagnostics ring buffer (500 ≈ 100s @ 5 FPS)
        self._diagnostics = DiagnosticsBuffer(maxlen=500, score_maxlen=300)

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
        self._segmenter_consecutive_failures: int = 0

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
            self._drift_reference_start_time: float = 0.0
            # Max 10 minutes to collect reference; use partial data if timeout
            self._drift_reference_timeout = 600.0
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

        # P1: Physics speed estimator (optional)
        self._speed_estimator = None
        if self._physics_config and self._physics_config.speed_enabled:
            try:
                from argus.physics.speed import PixelSpeedEstimator

                self._speed_estimator = PixelSpeedEstimator(
                    fps=camera_config.fps_target,
                    smoothing_window=self._physics_config.speed_smoothing_window,
                    pixel_scale_mm_per_px=self._physics_config.pixel_scale_mm_per_px,
                )
                logger.info(
                    "pipeline.speed_estimator_configured",
                    camera_id=camera_config.camera_id,
                    fps=camera_config.fps_target,
                )
            except Exception as e:
                logger.warning(
                    "pipeline.speed_estimator_init_failed",
                    camera_id=camera_config.camera_id,
                    error=str(e),
                )

        # Event camera trigger callback (reserved interface)
        # When an event camera detects motion, it calls this to boost frame FPS
        self._event_trigger_callback: Callable | None = None

        # M1: Multi-modal imaging preprocessing (optional)
        self._imaging_processor = None
        if self._imaging_config and self._imaging_config.enabled and self._imaging_config.polarization_processing:
            try:
                from argus.imaging.polarization import PolarizationProcessor

                self._imaging_processor = PolarizationProcessor(
                    deglare_method=self._imaging_config.deglare_method,
                    dolp_threshold=self._imaging_config.dolp_threshold,
                )
                logger.info(
                    "pipeline.imaging_processor_configured",
                    camera_id=camera_config.camera_id,
                    method=imaging_config.deglare_method,
                )
            except Exception as e:
                logger.warning(
                    "pipeline.imaging_processor_init_failed",
                    camera_id=camera_config.camera_id,
                    error=str(e),
                )

        # Cached postprocessor for physics enrichment (avoid per-frame allocation)
        from argus.core.anomaly_postprocess import AnomalyMapProcessor
        _min_contour = camera_config.anomaly.min_contour_area if hasattr(camera_config.anomaly, "min_contour_area") else 50
        self._physics_postproc = AnomalyMapProcessor(min_contour_area=_min_contour)

        # Temporal anomaly tracker for physics enrichment
        from argus.core.temporal_tracker import TemporalAnomalyTracker

        self._physics_tracker = TemporalAnomalyTracker(
            match_distance=camera_config.tracker_match_distance,
            max_gap_frames=camera_config.tracker_max_gap_frames,
            stationary_threshold=camera_config.tracker_stationary_threshold,
            fps=camera_config.fps_target,
            trajectory_history_length=(
                self._physics_config.trajectory_history_length if self._physics_config else 300
            ),
        )

        # P2: Trajectory analyzer (optional, phase 2)
        self._trajectory_analyzer = None
        self._camera_calibration = None
        if self._physics_config and self._physics_config.trajectory_enabled:
            try:
                from argus.physics.trajectory import TrajectoryAnalyzer

                self._trajectory_analyzer = TrajectoryAnalyzer(
                    gravity_ms2=self._physics_config.gravity_ms2,
                    pool_surface_z_mm=self._physics_config.pool_surface_z_mm,
                    min_points=self._physics_config.min_trajectory_points,
                    use_drag_model=self._physics_config.use_drag_model,
                )
                logger.info(
                    "pipeline.trajectory_analyzer_configured",
                    camera_id=camera_config.camera_id,
                )
            except Exception as e:
                logger.warning(
                    "pipeline.trajectory_analyzer_init_failed",
                    camera_id=camera_config.camera_id,
                    error=str(e),
                )

        # P3: Camera calibration for localization (optional, phase 2)
        if self._physics_config and self._physics_config.localization_enabled and camera_config.calibration_file:
            try:
                from argus.physics.calibration import CameraCalibration

                self._camera_calibration = CameraCalibration.from_file(
                    camera_config.calibration_file,
                    pool_surface_z_mm=self._physics_config.pool_surface_z_mm,
                )
                logger.info(
                    "pipeline.calibration_loaded",
                    camera_id=camera_config.camera_id,
                    file=camera_config.calibration_file,
                )
            except Exception as e:
                logger.warning(
                    "pipeline.calibration_load_failed",
                    camera_id=camera_config.camera_id,
                    error=str(e),
                )

        # Wire calibration to speed estimator if both available
        if self._speed_estimator and self._camera_calibration:
            self._speed_estimator._calibration = self._camera_calibration

        # Load custom classifier vocabulary if configured
        if self._classifier and classifier_config:
            custom_vocab_path = getattr(classifier_config, "custom_vocabulary_path", None)
            if custom_vocab_path:
                self._classifier.load_vocabulary(custom_vocab_path)

    @staticmethod
    def _find_model_in_dir(base: Path, camera_id: str) -> Path | None:
        """Find the best model file in a directory and its exports counterpart.

        Search order: OpenVINO (.xml) > Torch (.pt)
        Also searches data/exports/{camera_id} for inference-ready formats.
        """
        return find_runtime_model_in_dir(base, camera_id)

    @staticmethod
    def _find_model(camera_id: str) -> Path | None:
        """Auto-discover the latest trained model for a camera.

        Search order: OpenVINO (.xml) > Torch (.pt)
        Search paths: data/exports/{camera_id} (preferred) then data/models/{camera_id}
        """
        return find_runtime_model(camera_id)

    @staticmethod
    def _find_baseline_dir(camera_id: str) -> Path | None:
        """Find the latest baseline directory for a camera."""
        base = Path("data/baselines") / camera_id
        if not base.is_dir():
            return None
        # Check for zone subdirectories (e.g. data/baselines/c/default/)
        for zone_dir in sorted(base.iterdir()):
            if not zone_dir.is_dir():
                continue
            # Find current version
            current_file = zone_dir / "current.txt"
            if current_file.exists():
                version = current_file.read_text().strip()
                version_dir = zone_dir / version
                if version_dir.is_dir():
                    return version_dir
            # Fall back to latest version directory
            versions = sorted([d for d in zone_dir.iterdir() if d.is_dir()])
            if versions:
                return versions[-1]
        return None

    def initialize(self) -> bool:
        """Initialize all pipeline components. Returns True on success."""
        if not self._camera.connect():
            return False

        self._anomaly_detector.load()

        # Calibrate raw scores if PostProcessor MinMax is broken
        if self._anomaly_detector.get_status().minmax_broken:
            baseline_dir = self._find_baseline_dir(self.camera_config.camera_id)
            self._anomaly_detector.calibrate_raw_scores(baseline_dir)

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
        self.stats.update_fps()
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
        # DET-010: Auto-exit learning mode after duration expires
        if (
            self._learning_start_time is not None
            and self.mode == PipelineMode.LEARNING
            and (time.monotonic() - self._learning_start_time) >= self._learning_duration
        ):
            with self._mode_lock:
                self._mode = PipelineMode.ACTIVE
            self._auto_learning_complete = True
            self._learning_start_time = None
            logger.info(
                "pipeline.learning_mode_complete",
                camera_id=frame_data.camera_id,
                msg="Learning mode expired, switching to ACTIVE",
            )

        current_mode = self.mode
        # Save raw frame reference before zone masking — apply() returns a new
        # array (cv2.bitwise_and), so the original frame is not mutated.
        with self._latest_raw_frame_lock:
            self._latest_raw_frame = frame

        # Stage 0: Apply zone mask
        t0 = time.monotonic()
        frame = self._zone_mask.apply(frame)
        diag.stages.append(StageResult(
            stage_name="zone_mask", duration_ms=(time.monotonic() - t0) * 1000
        ))

        # Stage 0.5: Multi-modal imaging preprocessing (if enabled)
        if self._imaging_processor is not None:
            try:
                t_img = time.monotonic()
                from argus.imaging.polarization import PolarizationProcessor
                pol_result = self._imaging_processor.process(frame)
                frame = pol_result.deglared  # Use reflection-removed image
                diag.stages.append(StageResult(
                    stage_name="imaging_deglare",
                    duration_ms=(time.monotonic() - t_img) * 1000,
                ))
            except Exception as e:
                logger.debug("pipeline.imaging_preprocessing_error", error=str(e))

        # Save latest frame reference — get_latest_frame() copies on read
        with self._latest_frame_lock:
            self._latest_frame = frame

        # Low-light detection: bypass MOG2 when scene is dark
        is_low_light = False
        brightness_jump = False
        if self._low_light_enabled:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(gray.mean())
            is_low_light = mean_brightness < self._low_light_threshold

            # Brightness jump detection: freeze MOG2 on sudden illumination changes
            if self._prev_brightness is not None:
                delta = abs(mean_brightness - self._prev_brightness)
                if delta >= self._brightness_jump_threshold:
                    brightness_jump = True
                    logger.info(
                        "pipeline.brightness_jump",
                        camera_id=frame_data.camera_id,
                        delta=round(delta, 1),
                        prev=round(self._prev_brightness, 1),
                        current=round(mean_brightness, 1),
                    )
            self._prev_brightness = mean_brightness

            if is_low_light and not self._was_low_light:
                logger.warning(
                    "pipeline.low_light_entered",
                    camera_id=frame_data.camera_id,
                    brightness=round(mean_brightness, 1),
                )
            elif not is_low_light and self._was_low_light:
                logger.info(
                    "pipeline.low_light_exited",
                    camera_id=frame_data.camera_id,
                    brightness=round(mean_brightness, 1),
                )
            self._was_low_light = is_low_light

            # CLAHE low-light enhancement: equalize frame before detection stages
            if is_low_light and self._clahe is not None:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        diag.low_light = is_low_light

        # Stage 1: Pre-filter (with heartbeat bypass and anomaly lock bypass)
        t1 = time.monotonic()
        # MED-05: Time-based heartbeat instead of frame-count
        now = time.monotonic()
        # Low-light: shorten heartbeat to max_gap_seconds/2 so evidence can accumulate
        effective_heartbeat = (
            self._low_light_heartbeat_seconds if is_low_light
            else self._heartbeat_seconds
        )
        is_heartbeat = (now - self._last_heartbeat_time) >= effective_heartbeat
        if is_heartbeat:
            self._last_heartbeat_time = now

        # CRIT-01/02: Read lock state atomically for prefilter decision
        with self._lock_state_lock:
            is_locked = self._locked
        skip_prefilter = is_heartbeat or is_locked or is_low_light

        # DET-006: MAINTENANCE mode freezes MOG2 learning on all frames
        # Also freeze on brightness jumps to prevent background model corruption
        freeze_mog2 = current_mode == PipelineMode.MAINTENANCE or brightness_jump

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
                self._m_skipped_mog2.inc()
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

        # Stage 2+3: YOLO person filter + Anomaly detection
        # When skip_frame_on_person=True (nuclear plant default), run both in
        # parallel via ThreadPoolExecutor to overlap YOLO 15-30ms with Anomalib.
        # Correctness note: when skip_on_person=True AND YOLO finds no persons,
        # the frame is never masked (there's nobody to blur), so anomaly running
        # on the original frame produces the same result as sequential mode.
        # When skip_frame_on_person=False, person masking produces masked_frame
        # needed by anomaly, so stages must remain sequential.
        t2 = time.monotonic()
        skip_on_person = self.camera_config.person_filter.skip_frame_on_person

        if skip_on_person:
            # Parallel: submit both to thread pool concurrently
            yolo_future: Future = self._inference_executor.submit(
                self._object_detector.detect, frame,
            )
            anomaly_future: Future = self._inference_executor.submit(
                self._anomaly_detector.predict, frame,
            )
            try:
                detection_result: ObjectDetectionResult = yolo_future.result(timeout=30.0)
                t2_yolo_done = time.monotonic()
                anomaly_result: AnomalyResult = anomaly_future.result(timeout=30.0)
                t3_anomaly_done = time.monotonic()
            except TimeoutError:
                self.stats.frames_timeout += 1
                self._consecutive_timeouts += 1
                if self._consecutive_timeouts >= self._TIMEOUT_DEGRADE_THRESHOLD:
                    logger.error(
                        "pipeline.inference_degraded",
                        camera_id=self.camera_config.camera_id,
                        consecutive=self._consecutive_timeouts,
                        msg="Inference repeatedly timing out — model may be stuck",
                    )
                else:
                    logger.error(
                        "pipeline.inference_timeout",
                        camera_id=self.camera_config.camera_id,
                        consecutive=self._consecutive_timeouts,
                        msg="Inference future timed out after 30s — skipping frame",
                    )
                if not yolo_future.cancel():
                    logger.warning(
                        "pipeline.future_cancel_failed",
                        camera_id=self.camera_config.camera_id,
                        stage="yolo",
                        msg="YOLO inference still running after timeout — worker thread busy",
                    )
                if not anomaly_future.cancel():
                    logger.warning(
                        "pipeline.future_cancel_failed",
                        camera_id=self.camera_config.camera_id,
                        stage="anomaly",
                        msg="Anomaly inference still running after timeout — worker thread busy",
                    )
                self._update_lock_state_time(time.monotonic())
                return None

            self._consecutive_timeouts = 0  # reset on successful inference

            # HIGH-04: Warn once if person filter is unavailable
            if not detection_result.filter_available and not self._person_filter_warned:
                self._person_filter_warned = True
                logger.warning(
                    "pipeline.person_filter_offline",
                    camera_id=self.camera_config.camera_id,
                    msg="YOLO person filter unavailable — frames will not be filtered for humans",
                )

            if detection_result.has_persons:
                self.stats.frames_skipped_person += 1
                self._update_lock_state_time(time.monotonic())
                return None
        else:
            # Sequential: YOLO first (need masked_frame for person blurring), then anomaly
            detection_result = self._object_detector.detect(frame)
            t2_yolo_done = time.monotonic()

            if not detection_result.filter_available and not self._person_filter_warned:
                self._person_filter_warned = True
                logger.warning(
                    "pipeline.person_filter_offline",
                    camera_id=self.camera_config.camera_id,
                    msg="YOLO person filter unavailable — frames will not be filtered for humans",
                )

            # When persons found but skip disabled: use masked_frame with persons blurred
            analysis_frame = (
                detection_result.masked_frame if detection_result.masked_frame is not None else frame
            )
            anomaly_result = self._anomaly_detector.predict(analysis_frame)
            t3_anomaly_done = time.monotonic()

        diag.stages.append(StageResult(
            stage_name="yolo",
            duration_ms=(t2_yolo_done - t2) * 1000,
            metadata={
                "person_count": len(detection_result.persons),
                "object_count": len(detection_result.objects),
                "classes": [o.class_name for o in detection_result.non_person_objects],
                "track_ids": [o.track_id for o in detection_result.objects if o.track_id is not None],
                "parallel": skip_on_person,
            },
        ))

        self.stats.frames_analyzed += 1
        diag.stages.append(StageResult(
            stage_name="anomaly",
            duration_ms=(t3_anomaly_done - t2) * 1000 if skip_on_person else (t3_anomaly_done - t2_yolo_done) * 1000,
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
        simplex_result = None
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
                        # Guarantee minimum score of 0.6 AND boost existing score by at least 0.05
                        # so Simplex detection always changes the outcome meaningfully
                        original_anomalib_score = anomaly_result.anomaly_score
                        fallback_score = max(anomaly_result.anomaly_score + 0.05, 0.6)
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
                            anomalib_score=round(original_anomalib_score, 4),
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

                    self._segmenter_consecutive_failures = 0
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
                self._segmenter_consecutive_failures += 1
                logger.warning(
                    "pipeline.segmenter_error",
                    camera_id=frame_data.camera_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    consecutive_failures=self._segmenter_consecutive_failures,
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

        # P1-P3: Physics enrichment (speed, trajectory, localization)
        if alert is not None and anomaly_result.is_anomalous:
            try:
                regions = self._physics_postproc.extract_regions(
                    anomaly_result.anomaly_map,
                    threshold=0.3,
                ) if anomaly_result.anomaly_map is not None else []
                temporal = self._physics_tracker.update(regions, self._frame_counter)

                # P1: Speed estimation
                if self._speed_estimator and temporal.active_tracks:
                    best_track = max(temporal.active_tracks, key=lambda t: t.max_score)
                    speed_est = self._speed_estimator.estimate(best_track)
                    alert.speed_px_per_sec = speed_est.speed_px_per_sec
                    alert.speed_ms = speed_est.speed_ms

                # P2: Trajectory fitting (phase 2)
                if (
                    self._trajectory_analyzer
                    and self._camera_calibration
                    and temporal.active_tracks
                ):
                    best_track = max(temporal.active_tracks, key=lambda t: t.max_score)
                    if len(best_track.trajectory_history) >= 5:
                        from argus.physics.trajectory import trajectory_points_from_pixel_history
                        world_pts = trajectory_points_from_pixel_history(
                            best_track.trajectory_history,
                            self._camera_calibration,
                        )
                        fit = self._trajectory_analyzer.fit_trajectory(world_pts)
                        if fit is not None:
                            alert.trajectory_model = fit.model_type
                            alert.origin_x_mm = fit.origin.x_mm
                            alert.origin_y_mm = fit.origin.y_mm
                            alert.origin_z_mm = fit.origin.z_mm
                            alert.landing_x_mm = fit.landing.x_mm
                            alert.landing_y_mm = fit.landing.y_mm
                            alert.landing_z_mm = fit.landing.z_mm
            except Exception as e:
                logger.debug(
                    "pipeline.physics_enrichment_error",
                    camera_id=frame_data.camera_id,
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

        cam_id = frame_data.camera_id

        # Prometheus metrics (children pre-cached at __init__ to avoid per-frame lock)
        self._m_latency.observe(diag.total_duration_ms / 1000.0)
        self._m_frames.inc()
        for s in diag.stages:
            m = self._m_stages.get(s.stage_name) or self._m_stages_fallback(s.stage_name)
            m.observe(s.duration_ms / 1000.0)
        self._m_anomaly_score.set(anomaly_result.anomaly_score)
        if alert is not None:
            METRICS.alerts_total.labels(
                camera_id=cam_id, severity=alert.severity.value,
            ).inc()

        # Event bus (skip allocation if no subscribers)
        if self._event_bus is not None and self._event_bus.has_subscribers(FrameAnalyzed):
            # Buffer frame for active learning sampler before publishing
            _al_sampler = getattr(self._event_bus, '_active_sampler', None)
            if _al_sampler is not None:
                _al_sampler.buffer_frame(cam_id, frame_data.frame_number, frame_data.frame)
            self._event_bus.publish(FrameAnalyzed(
                camera_id=cam_id,
                frame_number=frame_data.frame_number,
                anomaly_score=anomaly_result.anomaly_score,
                is_anomalous=anomaly_result.is_anomalous,
                stage_latencies=MappingProxyType({s.stage_name: s.duration_ms for s in diag.stages}),
                mog2_skipped=mog2_skipped,
                person_detected=detection_result.has_persons,
            ))
        if self._event_bus is not None and alert is not None and self._event_bus.has_subscribers(AlertRaised):
            self._event_bus.publish(AlertRaised(
                alert_id=alert.alert_id,
                camera_id=cam_id,
                zone_id=alert.zone_id,
                severity=alert.severity,
                anomaly_score=alert.anomaly_score,
                handling_policy=alert.handling_policy,
            ))

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

            if self._anomaly_detector.is_calibrated:
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
                logger.debug("pipeline.cusum_evidence_extraction_failed", exc_info=True)

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
        self._handle_ring_buffer(
            frame_data, frame, anomaly_result, detection_result,
            simplex_result, simplex_detected, alert,
        )

        return alert

    def _handle_ring_buffer(
        self,
        frame_data: FrameData,
        frame: np.ndarray,
        anomaly_result: AnomalyResult,
        detection_result: ObjectDetectionResult,
        simplex_result: object | None,
        simplex_detected: bool,
        alert: Alert | None,
    ) -> None:
        """Ring buffer frame append, post-capture flush, and alert solidify."""
        if self._alert_ring_buffer is None:
            return

        # Append current frame to ring buffer
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

            rb_persons: list[dict] = [
                {
                    "bbox": [int(p.x1), int(p.y1), int(p.x2), int(p.y2)],
                    "confidence": p.confidence,
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
                    logger.debug("pipeline.heatmap_extraction_failed", exc_info=True)

            rb_all_boxes: list[dict] = [
                {
                    "bbox": [int(o.x1), int(o.y1), int(o.x2), int(o.y2)],
                    "class": o.class_name,
                    "confidence": round(o.confidence, 2),
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

        # Flush any expired post-trigger captures
        try:
            for expired_id in self._alert_ring_buffer.check_expired_captures():
                post_frames = self._alert_ring_buffer.finish_post_capture(expired_id)
                if post_frames and self._recording_store is not None:
                    self._recording_store.append_post_frames(expired_id, post_frames)
                    if self._database is not None:
                        try:
                            meta = self._recording_store.load_metadata(expired_id)
                            self._database.update_alert_recording_status(
                                expired_id,
                                RecordingStatus.COMPLETE.value,
                                frame_count=meta["frame_count"] if meta else None,
                                end_timestamp=meta["end_timestamp"] if meta else None,
                            )
                        except Exception:
                            logger.debug(
                                "pipeline.recording_db_update_failed",
                                alert_id=expired_id,
                                exc_info=True,
                            )
        except Exception:
            logger.debug("pipeline.post_capture_flush_failed", exc_info=True)

        # Solidify ring buffer on alert
        if alert is not None:
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
                    # Persist recording to disk and insert DB record
                    rec_path = ""
                    file_size = 0
                    if self._recording_store is not None:
                        try:
                            rec_path, file_size = self._recording_store.save(recording)
                            logger.info(
                                "pipeline.recording_saved",
                                alert_id=alert.alert_id,
                                path=rec_path,
                            )
                        except Exception:
                            logger.warning(
                                "pipeline.recording_save_failed",
                                alert_id=alert.alert_id,
                                exc_info=True,
                            )
                    if self._database is not None and rec_path:
                        try:
                            db_rec = AlertRecordingRecord(
                                alert_id=recording.alert_id,
                                camera_id=recording.camera_id,
                                severity=recording.severity,
                                recording_path=rec_path,
                                start_timestamp=recording.frames[0].timestamp if recording.frames else 0,
                                end_timestamp=recording.frames[-1].timestamp if recording.frames else 0,
                                trigger_timestamp=recording.trigger_timestamp,
                                frame_count=len(recording.frames),
                                fps=recording.fps,
                                file_size_bytes=file_size,
                                linked_alert_id=recording.linked_alert_id,
                                status=recording.status.value,
                                video_codec="h264",
                            )
                            self._database.save_alert_recording(db_rec)
                        except Exception:
                            logger.warning(
                                "pipeline.recording_db_insert_failed",
                                alert_id=alert.alert_id,
                                exc_info=True,
                            )
                    alert._solidified_recording = recording
            except Exception:
                logger.warning(
                    "pipeline.ring_buffer_solidify_failed",
                    camera_id=frame_data.camera_id,
                    alert_id=alert.alert_id,
                    exc_info=True,
                )

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

    def _m_stages_fallback(self, stage_name: str) -> object:
        """Lazily create and cache a stage metric child for an unexpected stage name."""
        child = METRICS.pipeline_stage_seconds.labels(
            camera_id=self.camera_config.camera_id, stage=stage_name,
        )
        self._m_stages[stage_name] = child
        return child

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

    @property
    def segmenter_consecutive_failures(self) -> int:
        """Number of consecutive segmentation failures."""
        return self._segmenter_consecutive_failures

    def get_cusum_states(self) -> dict[str, CusumSnapshot]:
        """Return CUSUM evidence snapshots for all active zones."""
        return self._alert_grader.get_all_cusum_states()

    def set_event_trigger(self, callback: Callable | None) -> None:
        """Set event camera trigger callback (reserved interface).

        When an event camera adapter detects a motion burst, it calls
        this callback to signal the frame camera to boost its FPS.
        The callback receives ``(camera_id: str, event_rate: int)``.
        """
        self._event_trigger_callback = callback

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
            self._zone_mask_cache.clear()
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

    def set_model_version_id(self, model_version_id: str | None) -> None:
        """Update the model version tag used in alerts and inference records."""
        self._model_version_id = model_version_id

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
            stats=self.stats.snapshot(),
        )

    def _evaluate_zones(
        self, frame_data: FrameData, anomaly_result: AnomalyResult, frame: np.ndarray,
        detection_type: str = "anomaly", detected_objects: list[dict] | None = None,
    ) -> Alert | None:
        """Evaluate anomaly against all configured include zones.

        H4 fix: When anomaly_map is available and zones have polygons,
        compute per-zone score (max within zone region) instead of using
        the global score for all zones. This prevents zone A's anomaly
        from causing false CUSUM accumulation in zone B.
        """
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
            zone_score = anomaly_result.anomaly_score
            if anomaly_result.anomaly_map is not None and zone.polygon:
                zone_score = self._compute_zone_score(
                    anomaly_result.anomaly_map, zone.zone_id, zone.polygon, frame.shape
                )

            alert = self._alert_grader.evaluate(
                camera_id=frame_data.camera_id,
                zone_id=zone.zone_id,
                zone_priority=zone.priority,
                anomaly_score=zone_score,
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

    def _compute_zone_score(
        self,
        anomaly_map: np.ndarray,
        zone_id: str,
        polygon: list[tuple[int, int]],
        frame_shape: tuple,
    ) -> float:
        """Compute the max anomaly score within a zone polygon.

        Uses cached masks to avoid repeated cv2.fillPoly on every frame.
        Cache is invalidated on update_zones().
        """
        map_h, map_w = anomaly_map.shape[:2]
        cache_key = (zone_id, map_h, map_w)

        zone_mask = self._zone_mask_cache.get(cache_key)
        if zone_mask is not None:
            self._zone_mask_cache.move_to_end(cache_key)  # LRU touch
        else:
            frame_h, frame_w = frame_shape[:2]
            scale_x = map_w / frame_w
            scale_y = map_h / frame_h
            scaled_pts = np.array(
                [(int(x * scale_x), int(y * scale_y)) for x, y in polygon],
                dtype=np.int32,
            )
            zone_mask = np.zeros((map_h, map_w), dtype=np.uint8)
            cv2.fillPoly(zone_mask, [scaled_pts], 255)
            if len(self._zone_mask_cache) >= self._zone_mask_cache_limit:
                self._zone_mask_cache.popitem(last=False)  # evict LRU (oldest)
            self._zone_mask_cache[cache_key] = zone_mask

        zone_values = anomaly_map[zone_mask > 0]
        if zone_values.size == 0:
            return 0.0
        return float(zone_values.max())

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
                else:
                    # Score bounced back above clear threshold (middle zone) —
                    # reset the clear timer to prevent premature unlock
                    self._lock_last_below_time = None

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
        with self.stats._lock:
            self.stats._latency_sum += elapsed_ms
            n = self.stats.frames_analyzed
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
            now = time.monotonic()
            if self._drift_reference_start_time == 0.0:
                self._drift_reference_start_time = now

            self._drift_reference_scores.append(score)
            collected = len(self._drift_reference_scores)
            timed_out = (now - self._drift_reference_start_time) >= self._drift_reference_timeout
            enough = collected >= self._drift_reference_size

            # Use partial reference if timeout and have at least 100 samples
            if enough or (timed_out and collected >= 100):
                ref = np.array(self._drift_reference_scores)
                self._drift_detector.set_reference(ref)
                self._drift_reference_ready = True
                logger.info(
                    "drift.reference_ready",
                    camera_id=camera_id,
                    samples=collected,
                    target=self._drift_reference_size,
                    timed_out=timed_out,
                    mean=round(float(ref.mean()), 4),
                    std=round(float(ref.std()), 4),
                )
                self._drift_reference_scores.clear()
            elif timed_out:
                # Not enough samples even after timeout — reset and retry
                logger.warning(
                    "drift.reference_timeout_insufficient",
                    camera_id=camera_id,
                    samples=collected,
                    required=100,
                )
                self._drift_reference_scores.clear()
                self._drift_reference_start_time = 0.0
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
