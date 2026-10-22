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

import hashlib
import json
import math
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from copy import copy
from dataclasses import asdict, dataclass, replace
from enum import Enum
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Callable

import cv2
import numpy as np
import structlog

from argus.alerts.grader import Alert, AlertGrader, CusumSnapshot, DetectionType
from argus.anomaly.candidate_model_evidence import (
    CandidateModelEvidence,
    CandidateModelThresholds,
    SpatialTemporalConfirmer,
    build_calibration_id,
    evaluate_candidate_model_evidence,
    prepare_model_evidence_map,
)
from argus.anomaly.classification_policy import localized_classification
from argus.anomaly.detector import AnomalibDetector, AnomalyResult, DetectorStatus
from argus.anomaly.drift import DriftDetector, DriftStatus
from argus.capture.base import CaptureAdapter
from argus.capture.camera import CameraCapture, FrameData
from argus.config.schema import AlertConfig, CameraConfig, ClassifierConfig, SegmenterConfig, ZonePriority
from argus.contracts.validation import validate_anomaly_result
from argus.core.alert_ring_buffer import AlertFrameBuffer, FrameSnapshot, RecordingStatus, compress_frame
from argus.core.degradation import LockState
from argus.core.diagnostics import (
    DiagnosticsBuffer,
    FrameDiagnostics,
    FrameScoreRecord,
    StageResult,
)
from argus.core.error_channel import (
    SEVERITY_WARNING,
    get_error_channel,
)
from argus.core.event_bus import (
    AlertRaised,
    EventBus,
    FrameAnalyzed,
)
from argus.core.frame_identity import FrameIdentity, build_frame_identity
from argus.core.frame_quality import FrameQualityAssessor
from argus.core.inference_record import (
    ConformalLevel,
    FinalDecision,
    InferenceRecord,
    PrefilterDecision,
)
from argus.core.metrics import METRICS
from argus.core.model_discovery import find_all_models, find_runtime_model, find_runtime_model_in_dir
from argus.core.runtime import FrameOutcomeKind
from argus.core.zone_mask import ZoneMaskEngine
from argus.person.detector import ObjectDetection, ObjectDetectionResult, YOLOObjectDetector
from argus.prefilter.fast_motion import FastMotionDetector, FastMotionResult
from argus.prefilter.mog2 import MOG2PreFilter, PreFilterResult
from argus.preprocessing.scene_profile import (
    RUNTIME_ARTIFACT_HASH_SCHEME,
    normalize_runtime_artifact_sha256,
    runtime_artifact_sha256,
)
from argus.storage.models import AlertRecordingRecord

logger = structlog.get_logger()

_INDEPENDENT_VISUAL_DETECTION_TYPES = {
    "simplex_static": DetectionType.PERSISTENT_CHANGE,
    "static_foreign": DetectionType.PERSISTENT_CHANGE,
    "visible_water_leak": DetectionType.WATER_LEAK,
    "visible_plume": DetectionType.PLUME,
}

_CANDIDATE_EVIDENCE_ALGORITHM_CONTRACT = {
    "detector_model_output": "native-pre-argus-minmax-v2",
    "multiscale_merge": "all-components-finite-comparable-max-v1",
    "candidate_roi": "floor-ceil-clipped-p95-v1",
    "candidate_context": "expanded-ring-excluding-roi-v1",
    "temporal_confirmation": "processed-observation-spatial-iou-v1",
    "anomaly_attribution": "connected-component-partitioned-map-overlap-v3",
}


def _runtime_package_version(distribution: str) -> str | None:
    """Return a calibration-relevant dependency version when installed."""
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return None

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
                    max_workers=3, thread_name_prefix="inference"
                )
    return _DEFAULT_INFERENCE_EXECUTOR


_SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3}


def _updated_anomaly_result(result: object, **changes) -> AnomalyResult:
    """Replace an anomaly result while tolerating legacy duck-typed test doubles."""
    if isinstance(result, AnomalyResult):
        return replace(result, **changes)

    def _number(name: str, default: float | None) -> float | None:
        value = getattr(result, name, default)
        return float(value) if isinstance(value, int | float) else default

    def _boolean(name: str, default: bool) -> bool:
        value = getattr(result, name, default)
        return value if isinstance(value, bool) else default

    anomaly_map = getattr(result, "anomaly_map", None)
    model_map = getattr(result, "model_anomaly_map", None)
    output_space = getattr(result, "model_output_space", None)
    rebuilt = AnomalyResult(
        anomaly_score=float(_number("anomaly_score", 0.0)),
        anomaly_map=anomaly_map if isinstance(anomaly_map, np.ndarray) else None,
        is_anomalous=_boolean("is_anomalous", False),
        threshold=float(_number("threshold", 0.7)),
        detection_failed=_boolean("detection_failed", False),
        raw_score=_number("raw_score", None),
        model_score=_number("model_score", None),
        model_anomaly_map=model_map if isinstance(model_map, np.ndarray) else None,
        model_output_space=output_space if isinstance(output_space, str) else None,
        model_output_comparable=(
            getattr(result, "model_output_comparable", False) is True
        ),
        model_input_variant=(
            getattr(result, "model_input_variant", "scene_stable:unmasked")
            if isinstance(
                getattr(result, "model_input_variant", "scene_stable:unmasked"),
                str,
            )
            else "scene_stable:unmasked"
        ),
    )
    return replace(rebuilt, **changes)


def _model_artifact_sha256(model_path: Path | None) -> str | None:
    """Fingerprint the exact runtime artifact set used by candidate calibration."""
    if model_path is None:
        return None
    try:
        return runtime_artifact_sha256(model_path)
    except FileNotFoundError:
        return None


def _frame_content_sha256(frame: np.ndarray | None) -> str | None:
    """Fingerprint the exact decoded reference tensor used by SceneGuard."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None
    values = np.ascontiguousarray(frame)
    digest = hashlib.sha256()
    shape = json.dumps(list(values.shape), separators=(",", ":")).encode("ascii")
    dtype = values.dtype.str.encode("ascii")
    digest.update(len(shape).to_bytes(4, "big"))
    digest.update(shape)
    digest.update(len(dtype).to_bytes(4, "big"))
    digest.update(dtype)
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


@dataclass
class PreparedAnomalyModelRelease:
    """Preloaded detector swap that can be committed and rolled back atomically."""

    owner_id: int
    target_detector: object
    target_path: Path
    target_version: str
    target_artifact_sha256: str
    target_scene_profile_id: str | None
    previous_detector: object
    previous_path: Path | None
    previous_version: str | None
    previous_artifact_sha256: str | None
    committed: bool = False
    previous_scene_state: dict[str, object] | None = None
    rollback_error: str | None = None


class PipelineMode(str, Enum):
    """Operating modes for the detection pipeline (DET-006)."""

    ACTIVE = "active"  # Normal detection + alerts
    MAINTENANCE = "maintenance"  # MOG2 frozen, detection + alerts continue
    LEARNING = "learning"  # Full pipeline runs but alerts suppressed
    COLLECTION = "collection"  # Capture mode: skip YOLO + Anomaly, raw frame still flows
    TRAINING = "training"  # Training in progress: skip YOLO + Anomaly to free GPU


_DETECTION_SKIPPED_MODES = (PipelineMode.COLLECTION, PipelineMode.TRAINING)


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

    # Anomaly degradation: enter simplex-only fallback after this many
    # consecutive hard exceptions from `_predict_anomaly`. Single failures are
    # tolerated by the detector's own try/except — only a sustained outage of
    # the anomaly head trips this gate.
    _ANOMALY_DEGRADE_THRESHOLD = 5

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
        auto_discover_model: bool = True,
        shared_anomaly_detector: object | None = None,
        shadow_runner: object | None = None,
        feedback_manager: object | None = None,
        recording_store: object | None = None,
        database: object | None = None,
        event_bus: EventBus | None = None,
        inference_executor: ThreadPoolExecutor | None = None,
        encode_executor: ThreadPoolExecutor | None = None,
        sensor_fusion: object | None = None,
        degradation_publisher: Callable[[str, dict], None] | None = None,
        capture: CaptureAdapter | None = None,
        frame_identity_declared_source: str | None = None,
        frame_identity_runtime_source: str | None = None,
        frame_identity_stream_id: str | None = None,
        baselines_dir: str | Path = "data/baselines",
    ):
        self.camera_config = camera_config
        self._baselines_dir = Path(baselines_dir).expanduser()
        self._on_alert = on_alert
        self._on_drift = on_drift
        self._event_bus = event_bus
        self._degradation_publisher = degradation_publisher
        self._inference_executor = inference_executor or _get_default_inference_executor()
        # Ring-buffer JPEG encoding runs on a dedicated pool so it never
        # starves inference futures (encode = 5-15ms each; inference = ~30ms).
        # When no executor is injected (standalone usage), fall back to the
        # inference pool — callers doing heavy encoding should inject one.
        self._encode_executor = encode_executor or self._inference_executor
        self._model_version_id = model_version_id
        self._auto_discover_model = bool(auto_discover_model)
        self._shadow_runner = shadow_runner
        self._shadow_runner_lock = threading.RLock()
        self._feedback_manager = feedback_manager
        self._recording_store = recording_store
        self._database = database
        self.stats = PipelineStats()
        self._last_zone_alerts: list[Alert] = []
        self._last_emitted_alerts: list[Alert] = []
        self._classifier_config = classifier_config
        self._segmenter_config = segmenter_config
        self._auxiliary_errors: dict[str, str] = {}
        self._auxiliary_retired: dict[str, object] = {}
        self._auxiliary_shutdown = False
        self._shutdown_lock = threading.Lock()
        self._auxiliary_load_blocked: set[str] = set()
        self._physics_config = physics_config
        self._imaging_config = imaging_config
        self._frame_identity_declared_source = (
            str(frame_identity_declared_source)
            if frame_identity_declared_source is not None
            else str(camera_config.source)
        )
        self._frame_identity_runtime_source = (
            str(frame_identity_runtime_source)
            if frame_identity_runtime_source is not None
            else str(camera_config.source)
        )
        self._frame_identity_stream_id = frame_identity_stream_id

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
            for name in (
                "zone_mask",
                "scene_guard",
                "fast_motion",
                "persistent_change",
                "water_leak",
                "plume",
                "mog2",
                "yolo",
                "anomaly",
                "simplex",
            )
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

        # Stage 0.75: Phase-correlation alignment (cancels camera micro-vibration
        # before MOG2/anomaly detection; the #1 false-positive source for
        # fixed cameras near pumps, fans, and turbines).
        if camera_config.alignment.enabled:
            from argus.preprocessing.alignment import create_from_config

            self._aligner = create_from_config(camera_config.alignment)
            self._m_alignment_shift_dx = METRICS.alignment_shift_px.labels(
                camera_id=_cid, axis="dx",
            )
            self._m_alignment_shift_dy = METRICS.alignment_shift_px.labels(
                camera_id=_cid, axis="dy",
            )
        else:
            self._aligner = None
            self._m_alignment_shift_dx = None
            self._m_alignment_shift_dy = None

        # Fixed-reference scene guard.  This is deliberately separate from the
        # micro-vibration aligner above: it proves that the live camera still
        # sees an operator-approved scene before any old baseline/model is used.
        self._scene_guard = None
        self._scene_guard_lock = threading.RLock()
        self._scene_processing_lock = threading.RLock()
        self._scene_guard_last_result = None
        self._scene_guard_last_check_frame = 0
        self._scene_guard_last_check_monotonic = 0.0
        self._scene_guard_last_usable = False
        self._scene_guard_policy_recovery_required = False
        self._scene_guard_policy_recovery_count = 0
        self._scene_guard_reference_frame: np.ndarray | None = None
        self._scene_guard_reference_sha256: str | None = None
        self._scene_guard_reference_source: str | None = None
        self._scene_guard_profile_id: str | None = None
        self._scene_guard_bound_model_version: str | None = None
        self._scene_guard_bound_model_artifact_sha256: str | None = None
        self._scene_guard_reference_error: str | None = None
        self._scene_guard_runtime_error: str | None = None
        self._scene_guard_last_published_usable: bool | None = None
        # A scene profile cannot become trusted until initialize() proves that
        # the exact artifact remained stable while the inference engine loaded.
        self._model_artifact_sha256: str | None = None
        self._scene_sensitive_rois: list[tuple[int, int, int, int]] = []
        self._candidate_model_evidence_epoch = 1
        self._candidate_model_evidence_counter = 0
        self._candidate_model_frame_sequence = 0
        self._candidate_model_temporal: SpatialTemporalConfirmer | None = None
        self._initialize_scene_guard()

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
            camera_id=camera_config.camera_id,
        )

        # Stage 3: Anomaly detector — auto-discover trained model
        # Use shared detector for multi-class dinomaly mode, else create per-camera
        self._anomaly_inference_backend = camera_config.anomaly.inference_backend
        self._uses_shared_anomaly_detector = shared_anomaly_detector is not None
        if shared_anomaly_detector is not None:
            base_detector = shared_anomaly_detector
            logger.info(
                "pipeline.using_shared_detector",
                camera_id=camera_config.camera_id,
            )
        else:
            if model_path is None and self._auto_discover_model:
                model_path = self._find_model(
                    camera_config.camera_id,
                    backend=camera_config.anomaly.inference_backend,
                )
            base_detector = AnomalibDetector(
                model_path=model_path,
                threshold=camera_config.anomaly.threshold,
                image_size=camera_config.anomaly.image_size,
                inference_backend=camera_config.anomaly.inference_backend,
                ssim_baseline_frames=camera_config.anomaly.ssim_baseline_frames,
                ssim_sensitivity=camera_config.anomaly.ssim_sensitivity,
                ssim_midpoint=camera_config.anomaly.ssim_midpoint,
                ssim_global_change_suppress_fraction=(
                    camera_config.anomaly.ssim_global_change_suppress_fraction
                ),
                enable_calibration=camera_config.anomaly.enable_calibration,
                camera_id=camera_config.camera_id,
            )
        # A shared detector owns its artifact.  A per-camera discovery result
        # may point at a different file and must never be used to describe the
        # shared engine's calibration domain.
        runtime_model_path = None if shared_anomaly_detector is not None else model_path
        if runtime_model_path is None:
            detector_model_path = getattr(base_detector, "_model_path", None)
            if isinstance(detector_model_path, str | Path):
                runtime_model_path = Path(detector_model_path)
        self._anomaly_model_path = (
            Path(runtime_model_path) if runtime_model_path is not None else None
        )
        # The calibration identity must describe the artifact the inference
        # engine actually loaded.  A constructor-time hash is unsafe because a
        # symlink or model file can change before initialize() reads it.
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

        # Stage 3b: Ensemble detector — multi-model fusion for reduced false positives
        self._ensemble_detector: object | None = None
        if camera_config.anomaly.ensemble.enabled:
            self._ensemble_detector = self._try_create_ensemble(
                camera_config, model_path,
            )

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

        pc_cfg = camera_config.persistent_change
        if pc_cfg.enabled:
            from argus.anomaly.persistent_change import PersistentChangeScorer

            self._persistent_change: PersistentChangeScorer | None = PersistentChangeScorer(
                process_width=pc_cfg.process_width,
                reference_frames=pc_cfg.reference_frames,
                diff_threshold=pc_cfg.diff_threshold,
                min_area_px=pc_cfg.min_area_px,
                max_area_fraction=pc_cfg.max_area_fraction,
                max_change_fraction=pc_cfg.max_change_fraction,
                min_persistence_seconds=pc_cfg.min_persistence_seconds,
                min_persistence_frames=pc_cfg.min_persistence_frames,
                learning_alpha=pc_cfg.learning_alpha,
                morph_kernel_size=pc_cfg.morph_kernel_size,
                match_radius_px=pc_cfg.match_radius_px,
                max_missed_frames=pc_cfg.max_missed_frames,
                alert_score=pc_cfg.alert_score,
                max_candidates_per_frame=pc_cfg.max_candidates_per_frame,
            )
        else:
            self._persistent_change = None

        confirmation_cfg = camera_config.static_confirmation
        model_evidence_cfg = confirmation_cfg.model_evidence
        if confirmation_cfg.enabled and model_evidence_cfg.enabled:
            self._candidate_model_temporal = SpatialTemporalConfirmer(
                required_passes=model_evidence_cfg.temporal_required_passes,
                min_gap_seconds=model_evidence_cfg.temporal_min_gap_seconds,
                max_gap_seconds=model_evidence_cfg.temporal_max_gap_seconds,
                iou_threshold=model_evidence_cfg.temporal_iou_threshold,
            )
        if confirmation_cfg.enabled:
            from argus.anomaly.static_candidate_confirmation import (
                StaticCandidateConfirmationConfig as ConfirmationAlgorithmConfig,
            )
            from argus.anomaly.static_candidate_confirmation import StaticCandidateConfirmer

            self._static_candidate_confirmer: StaticCandidateConfirmer | None = (
                StaticCandidateConfirmer(
                    ConfirmationAlgorithmConfig(
                        min_bbox_size=confirmation_cfg.min_bbox_size,
                        min_local_mean_abs_difference=(
                            confirmation_cfg.min_local_mean_abs_difference
                        ),
                        changed_pixel_threshold=confirmation_cfg.changed_pixel_threshold,
                        global_illumination_min_delta=(
                            confirmation_cfg.global_illumination_min_delta
                        ),
                        global_changed_fraction_threshold=(
                            confirmation_cfg.global_changed_fraction_threshold
                        ),
                        max_global_model_residual=confirmation_cfg.max_global_model_residual,
                        min_structure_score=confirmation_cfg.min_structure_score,
                        sensitive_min_structure_score=(
                            confirmation_cfg.sensitive_min_structure_score
                        ),
                        external_anomaly_score_threshold=(
                            confirmation_cfg.external_anomaly_score_threshold
                        ),
                    )
                )
            )
        else:
            self._static_candidate_confirmer = None
        self._scene_sensitive_rois.extend(
            tuple(int(value) for value in roi)
            for roi in confirmation_cfg.sensitive_rois
        )

        water_cfg = camera_config.water_leak
        if water_cfg.enabled:
            from argus.anomaly.water_leak import WaterLeakHead

            self._water_leak: WaterLeakHead | None = WaterLeakHead(
                process_width=water_cfg.process_width,
                reference_frames=water_cfg.reference_frames,
                diff_threshold=water_cfg.diff_threshold,
                darkening_threshold=water_cfg.darkening_threshold,
                brightening_threshold=water_cfg.brightening_threshold,
                specular_value_threshold=water_cfg.specular_value_threshold,
                specular_saturation_max=water_cfg.specular_saturation_max,
                min_area_px=water_cfg.min_area_px,
                max_area_fraction=water_cfg.max_area_fraction,
                max_change_fraction=water_cfg.max_change_fraction,
                min_growth_area_px=water_cfg.min_growth_area_px,
                min_persistence_seconds=water_cfg.min_persistence_seconds,
                min_persistence_frames=water_cfg.min_persistence_frames,
                min_drip_length_px=water_cfg.min_drip_length_px,
                min_vertical_aspect=water_cfg.min_vertical_aspect,
                learning_alpha=water_cfg.learning_alpha,
                morph_kernel_size=water_cfg.morph_kernel_size,
                match_radius_px=water_cfg.match_radius_px,
                max_missed_frames=water_cfg.max_missed_frames,
                alert_score=water_cfg.alert_score,
                max_candidates_per_frame=water_cfg.max_candidates_per_frame,
            )
        else:
            self._water_leak = None

        plume_cfg = camera_config.plume
        if plume_cfg.enabled:
            from argus.anomaly.plume import PlumeHead

            self._plume: PlumeHead | None = PlumeHead(
                process_width=plume_cfg.process_width,
                reference_frames=plume_cfg.reference_frames,
                diff_threshold=plume_cfg.diff_threshold,
                brightening_threshold=plume_cfg.brightening_threshold,
                darkening_threshold=plume_cfg.darkening_threshold,
                max_saturation=plume_cfg.max_saturation,
                min_area_px=plume_cfg.min_area_px,
                max_area_fraction=plume_cfg.max_area_fraction,
                max_change_fraction=plume_cfg.max_change_fraction,
                min_growth_area_px=plume_cfg.min_growth_area_px,
                min_persistence_seconds=plume_cfg.min_persistence_seconds,
                min_persistence_frames=plume_cfg.min_persistence_frames,
                max_edge_density=plume_cfg.max_edge_density,
                max_bbox_fill_fraction=plume_cfg.max_bbox_fill_fraction,
                learning_alpha=plume_cfg.learning_alpha,
                morph_kernel_size=plume_cfg.morph_kernel_size,
                match_radius_px=plume_cfg.match_radius_px,
                max_missed_frames=plume_cfg.max_missed_frames,
                alert_score=plume_cfg.alert_score,
                max_candidates_per_frame=plume_cfg.max_candidates_per_frame,
            )
        else:
            self._plume = None

        fast_cfg = camera_config.fast_motion
        if fast_cfg.enabled:
            self._fast_motion: FastMotionDetector | None = FastMotionDetector(
                process_width=fast_cfg.process_width,
                diff_threshold=fast_cfg.diff_threshold,
                background_alpha=fast_cfg.background_alpha,
                min_area_px=fast_cfg.min_area_px,
                max_area_px=fast_cfg.max_area_px,
                min_streak_length_px=fast_cfg.min_streak_length_px,
                min_confidence=fast_cfg.min_confidence,
                max_motion_fraction=fast_cfg.max_motion_fraction,
                max_streak_frame_fraction=fast_cfg.max_streak_frame_fraction,
                streak_gap_px=fast_cfg.streak_gap_px,
                max_candidates_per_frame=fast_cfg.max_candidates_per_frame,
                fps_hint=camera_config.fps_target,
            )
        else:
            self._fast_motion = None

        # Alert grading
        self._alert_grader = AlertGrader(config=alert_config, sensor_fusion=sensor_fusion)

        # Camera — use injected capture when provided; otherwise keep the
        # existing protocol branch for backwards compatibility.
        if capture is not None:
            self._camera: CaptureAdapter = capture
        elif camera_config.protocol == "gige":
            from argus.capture.gige_capture import GigECapture

            gige_cfg = camera_config.gige
            self._camera: CaptureAdapter = GigECapture(
                camera_id=camera_config.camera_id,
                source=camera_config.source,
                fps_target=camera_config.fps_target,
                resolution=camera_config.resolution,
                exposure=gige_cfg.exposure,
                gain=gige_cfg.gain,
                pixel_format=gige_cfg.pixel_format,
                reconnect_delay=camera_config.reconnect_delay,
                max_reconnect_attempts=camera_config.max_reconnect_attempts,
                source_declared=self._frame_identity_declared_source,
                source_runtime=self._frame_identity_runtime_source,
                stream_id=self._frame_identity_stream_id,
            )
        else:
            camera_kwargs = {
                "camera_id": camera_config.camera_id,
                "source": camera_config.source,
                "protocol": camera_config.protocol,
                "fps_target": camera_config.fps_target,
                "resolution": camera_config.resolution,
                "reconnect_delay": camera_config.reconnect_delay,
                "max_reconnect_attempts": camera_config.max_reconnect_attempts,
                "source_declared": self._frame_identity_declared_source,
                "source_runtime": self._frame_identity_runtime_source,
                "stream_id": self._frame_identity_stream_id,
            }
            if camera_config.protocol == "usb":
                usb_cfg = camera_config.usb
                camera_kwargs.update(
                    usb_backend=usb_cfg.preferred_backend,
                    usb_pixel_format=usb_cfg.pixel_format,
                    usb_min_runtime_fps=usb_cfg.min_runtime_fps,
                )
            self._camera = CameraCapture(
                **camera_kwargs,
            )

        configure_identity = getattr(self._camera, "configure_frame_identity", None)
        if callable(configure_identity):
            configure_identity(
                source_declared=self._frame_identity_declared_source,
                source_runtime=self._frame_identity_runtime_source,
                stream_id=self._frame_identity_stream_id,
            )

        # Anti-absorption: time-based heartbeat (MED-05)
        # P2 fix: cap heartbeat to max_gap * 0.8 so CUSUM evidence can accumulate
        # even in static scenes where MOG2 drops all non-heartbeat frames
        raw_heartbeat = camera_config.mog2.heartbeat_frames / max(camera_config.fps_target, 1)
        max_gap = alert_config.temporal.max_gap_seconds
        self._heartbeat_seconds = min(raw_heartbeat, max_gap * 0.8)
        self._last_heartbeat_time = 0.0
        self._frame_counter = 0
        # Runtime-only identities for localized incidents whose upstream
        # detector does not provide a track_id (notably default YOLO/SAHI
        # configurations and appearance-map connected components).  Exact
        # per-frame bbox strings are intentionally not identities: normal box
        # jitter would create a fresh CUSUM tracker on every frame.
        self._incident_association_counter = 0
        self._incident_associations: dict[
            tuple[str, str, str], list[dict[str, object]]
        ] = {}
        self._incident_association_max_gap_seconds = float(max_gap)

        # Low-light mode: bypass MOG2 + shorten heartbeat when scene is dark
        ll_cfg = camera_config.low_light
        self._low_light_enabled = ll_cfg.enabled
        self._low_light_threshold = ll_cfg.brightness_threshold
        self._low_light_heartbeat_seconds = alert_config.temporal.max_gap_seconds / 2
        self._was_low_light = False
        self._prev_brightness: float | None = None
        self._brightness_jump_threshold = ll_cfg.brightness_jump_threshold
        self._fast_motion_recovery_seconds = ll_cfg.fast_motion_recovery_seconds
        self._fast_motion_suppress_until = 0.0

        # CLAHE preprocessing for low-light enhancement
        # Use CUDA CLAHE when available (13ms CPU → <1ms GPU)
        self._clahe_enabled = ll_cfg.clahe_enabled and ll_cfg.enabled
        self._clahe: object | None = None
        self._clahe_use_cuda = False
        if self._clahe_enabled:
            try:
                if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self._clahe = cv2.cuda.createCLAHE(
                        clipLimit=ll_cfg.clahe_clip_limit,
                        tileGridSize=(ll_cfg.clahe_grid_size, ll_cfg.clahe_grid_size),
                    )
                    self._clahe_use_cuda = True
                    self._clahe_gpu_mat = cv2.cuda_GpuMat()
                    self._clahe_cuda_stream = cv2.cuda.Stream()
            except Exception:
                pass
            if self._clahe is None:
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

        # Anomaly head degradation state (requirements §3.2 №9): when the
        # main anomaly model is unavailable (load failure or sustained
        # runtime errors) the pipeline drops the anomaly stage and lets the
        # simplex safety channel run solo, instead of crashing the camera.
        self._anomaly_degraded: bool = False
        self._anomaly_degradation_reason: str | None = None
        self._anomaly_degradation_started_at: float | None = None
        self._anomaly_failure_count: int = 0
        self._anomaly_degraded_lock = threading.Lock()

        # Thread safety for hot-updates
        self._config_lock = threading.Lock()

        # Cached per-zone anomaly masks (invalidated on update_zones, bounded to 64)
        self._zone_mask_cache: OrderedDict[
            tuple[str, int, int, int, int], np.ndarray
        ] = OrderedDict()
        self._zone_mask_cache_limit = 64

        # Thread safety for anomaly lock state (HIGH-03)
        self._lock_state_lock = threading.Lock()

        # Raw frame buffer for baseline capture, unmasked (CRIT-05)
        self._latest_raw_frame: np.ndarray | None = None
        self._latest_raw_frame_lock = threading.Lock()

        # Latest frame buffer for live preview (thread-safe)
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_lock = threading.Lock()

        # Latest frame identity for source/path diagnostics (thread-safe)
        self._latest_frame_identity: FrameIdentity | None = None
        self._latest_frame_identity_lock = threading.Lock()
        self._last_snapshot_identity: FrameIdentity | None = None
        self._last_snapshot_ts: float | None = None
        self._last_preview_stream_identity: FrameIdentity | None = None
        self._last_preview_stream_ts: float | None = None

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

        # COLLECTION/TRAINING watchdog: long-running tasks may crash without
        # resetting the mode; auto-recover after max_duration to avoid
        # permanent stuck state.
        self._mode_session_start_time: float | None = None
        self._mode_session_max_duration: float = 4 * 3600.0  # 4 hours

        # 5.1/5.2: Per-frame inference tracking for CameraInferenceRunner
        self._last_inference_record: InferenceRecord | None = None
        self._last_detection_failed: bool = False
        self._last_frame_outcome_kind = FrameOutcomeKind.NORMAL
        self._last_frame_outcome_reason: str | None = None
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
        self._classifier_vocabulary: set[str] = set()
        self._classifier_min_confidence = 0.9
        if classifier_config and classifier_config.enabled:
            try:
                from argus.anomaly.classifier import OpenVocabClassifier

                self._classifier = OpenVocabClassifier(
                    model_name=classifier_config.model_name,
                    vocabulary=classifier_config.vocabulary,
                    timeout_seconds=classifier_config.timeout_seconds,
                    runtime_key=f"{camera_config.camera_id}:classifier",
                )
                self._classifier_min_score = classifier_config.min_anomaly_score_to_classify
                self._classifier_high_risk = set(classifier_config.high_risk_labels)
                self._classifier_low_risk = set(classifier_config.low_risk_labels)
                self._classifier_suppress = set(getattr(classifier_config, "suppress_labels", []))
                self._classifier_vocabulary = set(classifier_config.vocabulary)
                self._classifier_min_confidence = classifier_config.min_confidence_to_affect_alert
                self._apply_classifier_policy()
                logger.info(
                    "pipeline.classifier_configured",
                    camera_id=camera_config.camera_id,
                    model=classifier_config.model_name,
                    vocab_size=len(classifier_config.vocabulary),
                )
            except Exception as e:
                self._auxiliary_errors["classifier"] = str(e)
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
                    runtime_key=f"{camera_config.camera_id}:segmenter",
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
                self._auxiliary_errors["segmenter"] = str(e)
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
        self._modality_fusion = None
        self._fusion_channels: int = 3
        self._latest_nir_frame: np.ndarray | None = None
        if self._imaging_config and self._imaging_config.enabled and self._imaging_config.polarization_processing:
            try:
                from argus.imaging.polarization import PolarizationProcessor

                self._imaging_processor = PolarizationProcessor(
                    deglare_method=self._imaging_config.deglare_method,
                    dolp_threshold=self._imaging_config.dolp_threshold,
                )
                self._fusion_channels = self._imaging_config.fusion_channels
                # Initialise modality fusion when extra channels are requested
                if self._fusion_channels > 3:
                    from argus.imaging.fusion import ModalityFusion

                    self._modality_fusion = ModalityFusion(
                        fusion_channels=self._fusion_channels,
                    )
                logger.info(
                    "pipeline.imaging_processor_configured",
                    camera_id=camera_config.camera_id,
                    method=imaging_config.deglare_method,
                    fusion_channels=self._fusion_channels,
                )
            except Exception as e:
                logger.warning(
                    "pipeline.imaging_processor_init_failed",
                    camera_id=camera_config.camera_id,
                    error=str(e),
                )

        # Frame quality assessment: reduce false positives from degraded input
        self._frame_quality_assessor = FrameQualityAssessor()

        # Cached postprocessor for physics enrichment (avoid per-frame allocation)
        from argus.core.anomaly_postprocess import AnomalyMapProcessor

        self._physics_postproc = AnomalyMapProcessor(
            min_contour_area=camera_config.anomaly.min_contour_area,
        )

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
            stationary_suppress_enabled=getattr(
                alert_config.temporal, "stationary_suppress_enabled", True,
            ),
            stationary_suppress_after_s=float(
                getattr(alert_config.temporal, "stationary_suppress_after_s", 10.0),
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
                try:
                    self._classifier.load_vocabulary(custom_vocab_path)
                    self._classifier_vocabulary = set(self._classifier.vocabulary)
                except Exception as exc:
                    self._auxiliary_errors["classifier"] = str(exc)
                    self._auxiliary_load_blocked.add("classifier")
                    logger.warning("pipeline.classifier_vocabulary_failed", error=str(exc))

    def set_nir_frame(self, nir_frame: np.ndarray) -> None:
        """Inject a NIR strobe frame for modality fusion.

        Called by the camera manager when a NIR frame is available
        (Phase 3: alternating visible/NIR capture).
        """
        self._latest_nir_frame = nir_frame

    @staticmethod
    def _find_model_in_dir(
        base: Path,
        camera_id: str,
        backend: str = "auto",
    ) -> Path | None:
        """Find the best model file in a directory and its exports counterpart.

        Search order: OpenVINO (.xml) > Torch (.pt)
        Also searches data/exports/{camera_id} for inference-ready formats.
        """
        return find_runtime_model_in_dir(base, camera_id, backend=backend)

    @staticmethod
    def _find_model(camera_id: str, backend: str = "auto") -> Path | None:
        """Auto-discover the latest trained model for a camera.

        Search order: OpenVINO (.xml) > Torch (.pt)
        Search paths: data/exports/{camera_id} (preferred) then data/models/{camera_id}
        """
        return find_runtime_model(camera_id, backend=backend)

    @staticmethod
    def _find_baseline_dir(
        camera_id: str, baselines_dir: str | Path = "data/baselines",
    ) -> Path | None:
        """Find the latest baseline directory for a camera."""
        base = Path(baselines_dir).expanduser() / camera_id
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

    def _configured_baseline_dir(self, camera_id: str) -> Path | None:
        """Look only inside this runtime's configured baseline root.

        Keep the historical one-argument lookup for standalone/default callers
        and their test adapters. A custom root never falls back to the default
        tree, including when the configured directory does not exist.
        """
        root = getattr(self, "_baselines_dir", Path("data/baselines"))
        if root == Path("data/baselines"):
            return self._find_baseline_dir(camera_id)
        return self._find_baseline_dir(camera_id, baselines_dir=root)

    def _initialize_scene_guard(self) -> None:
        """Create the fixed-reference scene gate without trusting live frames."""
        cfg = self.camera_config.scene_guard
        self._scene_guard_reference_edges: np.ndarray | None = None
        self._scene_guard_quick_hanning: np.ndarray | None = None
        self._scene_guard_last_quick_score: float | None = None
        self._scene_guard_last_quick_shift_px: float | None = None
        self._scene_guard_last_reconnect_count: int | None = None
        self._scene_guard_last_stream_key: tuple | None = None
        if not cfg.enabled:
            return

        from argus.preprocessing.scene_guard import SceneGuard
        from argus.preprocessing.scene_guard_policy import runtime_scene_guard_config

        # Use the same exact runtime thresholds as strict baseline construction.
        # This extraction does not change STABLE/ALIGNABLE policy or geometry.
        algorithm_config = runtime_scene_guard_config(cfg)
        reference, source, profile_id, metadata, error = (
            self._resolve_scene_guard_reference()
        )
        try:
            self._scene_guard = SceneGuard(
                reference_frame=reference,
                config=algorithm_config,
            )
        except Exception as exc:
            # A bad or featureless approved image is a configuration failure,
            # never a reason to adopt the first live frame as a replacement.
            self._scene_guard = SceneGuard(config=algorithm_config)
            reference = None
            error = f"approved reference is unusable: {exc}"

        self._scene_guard_reference_frame = (
            reference.copy() if reference is not None else None
        )
        self._scene_guard_reference_sha256 = _frame_content_sha256(reference)
        self._scene_guard_reference_source = source
        self._scene_guard_profile_id = profile_id
        bound_model_version = metadata.get("model_version_id")
        self._scene_guard_bound_model_version = (
            str(bound_model_version) if bound_model_version is not None else None
        )
        bound_model_artifact = metadata.get("model_artifact_sha256")
        self._scene_guard_bound_model_artifact_sha256 = (
            str(bound_model_artifact) if bound_model_artifact is not None else None
        )
        self._scene_guard_reference_error = error
        if reference is not None:
            self._scene_guard_reference_edges = self._scene_edge_signature(reference)
            self._scene_guard_quick_hanning = cv2.createHanningWindow(
                (
                    self._scene_guard_reference_edges.shape[1],
                    self._scene_guard_reference_edges.shape[0],
                ),
                cv2.CV_32F,
            )
        self._scene_sensitive_rois.extend(self._metadata_sensitive_rois(metadata))

        if error:
            logger.error(
                "pipeline.scene_guard_reference_unavailable",
                camera_id=self.camera_config.camera_id,
                error=error,
            )
        else:
            logger.info(
                "pipeline.scene_guard_configured",
                camera_id=self.camera_config.camera_id,
                source=source,
                profile_id=profile_id,
            )

    def _resolve_scene_guard_reference(
        self,
    ) -> tuple[
        np.ndarray | None,
        str | None,
        str | None,
        dict,
        str | None,
    ]:
        """Resolve a policy-authorized profile or an explicit validation reference."""
        cfg = self.camera_config.scene_guard
        camera_id = self.camera_config.camera_id

        try:
            from argus.preprocessing.scene_profile import SceneProfileStore

            store = SceneProfileStore(cfg.profiles_dir)
            active = store.get_active(camera_id)
        except Exception as exc:
            return None, "scene_profile", None, {}, f"active scene profile is invalid: {exc}"

        if active is not None:
            metadata = dict(active.metadata)
            if cfg.require_operator_approved_profile:
                if metadata.get("operator_approved") is not True:
                    return (
                        None,
                        f"profile:{active.profile_id}",
                        active.profile_id,
                        metadata,
                        "active scene profile is not explicitly operator-approved",
                    )
                validation_only = metadata.get("validation_only", False)
                if not isinstance(validation_only, bool):
                    return (
                        None,
                        f"profile:{active.profile_id}",
                        active.profile_id,
                        metadata,
                        "active scene profile has invalid validation_only authority metadata",
                    )
                if validation_only is True:
                    return (
                        None,
                        f"profile:{active.profile_id}",
                        active.profile_id,
                        metadata,
                        "active scene profile is validation-only and cannot authorize production",
                    )
            if cfg.require_profile_model_binding:
                bound_model_version = metadata.get("model_version_id")
                if not isinstance(bound_model_version, str) or not bound_model_version.strip():
                    return (
                        None,
                        f"profile:{active.profile_id}",
                        active.profile_id,
                        metadata,
                        "active scene profile has no model_version_id binding",
                    )
                if bound_model_version != self._model_version_id:
                    return (
                        None,
                        f"profile:{active.profile_id}",
                        active.profile_id,
                        metadata,
                        "scene profile/model mismatch: "
                        f"profile={bound_model_version}, runtime={self._model_version_id}",
                    )
                bound_artifact = metadata.get("model_artifact_sha256")
                try:
                    bound_artifact = normalize_runtime_artifact_sha256(bound_artifact)
                except ValueError:
                    return (
                        None,
                        f"profile:{active.profile_id}",
                        active.profile_id,
                        metadata,
                        "active scene profile has no valid model_artifact_sha256 binding",
                    )
                metadata["model_artifact_sha256"] = bound_artifact
                if (
                    metadata.get("model_artifact_hash_scheme")
                    != RUNTIME_ARTIFACT_HASH_SCHEME
                ):
                    return (
                        None,
                        f"profile:{active.profile_id}",
                        active.profile_id,
                        metadata,
                        "active scene profile does not use the Argus runtime artifact hash scheme",
                    )
                runtime_artifact = self._model_artifact_sha256
                if runtime_artifact is None:
                    return (
                        None,
                        f"profile:{active.profile_id}",
                        active.profile_id,
                        metadata,
                        "runtime model artifact fingerprint is unavailable until model load succeeds",
                    )
                if bound_artifact != runtime_artifact:
                    return (
                        None,
                        f"profile:{active.profile_id}",
                        active.profile_id,
                        metadata,
                        "scene profile/runtime artifact mismatch: "
                        f"profile={bound_artifact}, runtime={runtime_artifact}",
                    )
                baseline_error = self._validate_profile_active_baseline(active)
                if baseline_error is not None:
                    return (
                        None,
                        f"profile:{active.profile_id}",
                        active.profile_id,
                        metadata,
                        baseline_error,
                    )
            image = cv2.imread(str(active.reference_path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                return (
                    None,
                    f"profile:{active.profile_id}",
                    active.profile_id,
                    metadata,
                    "active scene profile reference cannot be decoded",
                )
            return (
                image,
                f"profile:{active.profile_id}",
                active.profile_id,
                metadata,
                None,
            )

        if cfg.require_operator_approved_profile:
            return (
                None,
                "scene_profile",
                None,
                {},
                "no active operator-approved scene profile is available",
            )

        if cfg.require_profile_model_binding:
            # A raw reference image or baseline has no immutable profile
            # identity and no model_version_id.  Falling through here would
            # silently defeat the exact profile/model binding policy whenever
            # an operator had not yet published a profile.
            return (
                None,
                "scene_profile",
                None,
                {},
                "no active scene profile is available for required model binding",
            )

        if cfg.reference_image:
            reference_path = Path(cfg.reference_image).expanduser()
            image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                return (
                    None,
                    f"image:{reference_path}",
                    None,
                    {},
                    f"configured scene reference cannot be decoded: {reference_path}",
                )
            return image, f"image:{reference_path}", None, {}, None

        if cfg.fallback_to_baseline:
            baseline_dir = self._configured_baseline_dir(camera_id)
            if baseline_dir is not None:
                candidates = sorted(
                    path
                    for path in baseline_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                )
                if candidates:
                    selected = candidates[len(candidates) // 2]
                    image = cv2.imread(str(selected), cv2.IMREAD_COLOR)
                    if image is None or image.size == 0:
                        return (
                            None,
                            f"baseline:{selected}",
                            None,
                            {},
                            f"baseline scene reference cannot be decoded: {selected}",
                        )
                    return image, f"baseline:{selected}", None, {}, None

        return (
            None,
            None,
            None,
            {},
            "no approved scene profile, reference image, or active baseline is available",
        )

    def _validate_profile_active_baseline(self, profile) -> str | None:
        """Prove that a profile originated from the currently active baseline."""
        baseline_dir = self._configured_baseline_dir(self.camera_config.camera_id)
        if baseline_dir is None:
            return "active scene profile cannot be tied to an active baseline"
        candidates = [baseline_dir / profile.source_baseline_name]
        candidates.extend(
            path
            for path in baseline_dir.iterdir()
            if path.is_file()
            and path not in candidates
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            except OSError:
                continue
            if digest == profile.source_baseline_sha256:
                return None
        return (
            "scene profile baseline hash is not present in the current active "
            f"baseline {baseline_dir.name}"
        )

    @staticmethod
    def _metadata_sensitive_rois(metadata: dict) -> list[tuple[int, int, int, int]]:
        """Extract well-formed reflective/display regions from profile metadata."""
        rois: list[tuple[int, int, int, int]] = []
        raw_rois = metadata.get("sensitive_rois", []) if isinstance(metadata, dict) else []
        if not isinstance(raw_rois, (list, tuple)):
            return rois
        for roi in raw_rois:
            if not isinstance(roi, (list, tuple)) or len(roi) != 4:
                continue
            try:
                values = tuple(int(value) for value in roi)
            except (TypeError, ValueError):
                continue
            x1, y1, x2, y2 = values
            if x2 > x1 and y2 > y1:
                rois.append(values)
        return rois

    @staticmethod
    def _scene_edge_signature(frame: np.ndarray, width: int = 320) -> np.ndarray:
        """Build a lighting-tolerant thumbnail used as an every-frame sentinel."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] > width:
            scale = width / gray.shape[1]
            gray = cv2.resize(
                gray,
                (width, max(1, int(round(gray.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.equalizeHist(gray)
        return cv2.Canny(gray, 50, 150)

    def _scene_quick_evidence(
        self,
        aligned_frame: np.ndarray,
    ) -> tuple[float | None, float | None]:
        """Return thumbnail edge agreement and estimated full-resolution shift.

        The phase-correlation shift is only a cheap trigger for the audited
        feature-geometry check; it never declares a scene safe or moved by
        itself.  This catches abrupt camera translations that an edge-overlap
        score with spatial tolerance can otherwise hide.
        """
        reference_edges = self._scene_guard_reference_edges
        if reference_edges is None:
            return None, None
        current_edges = self._scene_edge_signature(
            aligned_frame,
            width=reference_edges.shape[1],
        )
        if current_edges.shape != reference_edges.shape:
            return 0.0, None

        reference = reference_edges > 0
        current = current_edges > 0
        reference_count = int(np.count_nonzero(reference))
        current_count = int(np.count_nonzero(current))
        if reference_count < 100 or current_count < 100:
            return None, None

        shift_px: float | None = None
        try:
            shift, response = cv2.phaseCorrelate(
                reference_edges.astype(np.float32),
                current_edges.astype(np.float32),
                self._scene_guard_quick_hanning,
            )
            dx, dy = float(shift[0]), float(shift[1])
            if (
                np.isfinite(dx)
                and np.isfinite(dy)
                and np.isfinite(response)
                and float(response) >= 0.20
            ):
                reference_frame = self._scene_guard_reference_frame
                if reference_frame is not None:
                    scale_x = reference_frame.shape[1] / reference_edges.shape[1]
                    scale_y = reference_frame.shape[0] / reference_edges.shape[0]
                    shift_px = float(np.hypot(dx * scale_x, dy * scale_y))
        except cv2.error:
            shift_px = None

        kernel = np.ones((5, 5), dtype=np.uint8)
        reference_near = cv2.dilate(reference_edges, kernel) > 0
        current_near = cv2.dilate(current_edges, kernel) > 0
        reference_recall = float(np.count_nonzero(reference & current_near)) / reference_count
        current_precision = float(np.count_nonzero(current & reference_near)) / current_count
        return 0.5 * (reference_recall + current_precision), shift_px

    def set_scene_reference(
        self,
        frame: np.ndarray,
        *,
        source: str,
        profile_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Explicitly install a trusted reference (used by offline evaluation).

        Production code should publish an immutable :class:`SceneProfile`
        instead.  This method never reads from the live capture automatically.
        """
        if self._scene_guard is None:
            raise RuntimeError("scene guard is disabled for this camera")
        if not source or not str(source).strip():
            raise ValueError("scene reference source must be explicit")
        with self._scene_processing_lock:
            with self._scene_guard_lock:
                self._scene_guard.set_reference(frame)
                self._scene_guard_reference_frame = frame.copy()
                self._scene_guard_reference_sha256 = _frame_content_sha256(frame)
                self._scene_guard_reference_source = str(source)
                self._scene_guard_profile_id = profile_id
                self._scene_guard_bound_model_version = (
                    str((metadata or {}).get("model_version_id"))
                    if (metadata or {}).get("model_version_id") is not None
                    else None
                )
                self._scene_guard_bound_model_artifact_sha256 = (
                    str((metadata or {}).get("model_artifact_sha256"))
                    if (metadata or {}).get("model_artifact_sha256") is not None
                    else None
                )
                self._scene_guard_reference_error = None
                self._scene_guard_runtime_error = None
                self._scene_guard_reference_edges = self._scene_edge_signature(frame)
                self._scene_guard_quick_hanning = cv2.createHanningWindow(
                    (
                        self._scene_guard_reference_edges.shape[1],
                        self._scene_guard_reference_edges.shape[0],
                    ),
                    cv2.CV_32F,
                )
                self._scene_guard_last_result = None
                self._scene_guard_last_check_frame = 0
                self._scene_guard_last_check_monotonic = 0.0
                self._scene_guard_last_usable = False
                self._scene_guard_last_stream_key = None
                self._scene_guard_policy_recovery_required = False
                self._scene_guard_policy_recovery_count = 0
                self._scene_guard_last_quick_score = None
                self._scene_guard_last_quick_shift_px = None
                configured = [
                    tuple(int(value) for value in roi)
                    for roi in self.camera_config.static_confirmation.sensitive_rois
                ]
                self._scene_sensitive_rois = configured + self._metadata_sensitive_rois(
                    metadata or {}
                )
            self._reset_scene_sensitive_state()
        return self.get_scene_guard_status()

    def get_scene_guard_status(self) -> dict:
        """Return an auditable, thread-safe snapshot of the scene gate."""
        cfg = self.camera_config.scene_guard
        if self._scene_guard is None:
            if cfg.enabled:
                reason = (
                    self._scene_guard_runtime_error
                    or self._scene_guard_reference_error
                    or "scene guard is enabled but no verified reference is available"
                )
                return {
                    "enabled": True,
                    "state": "blocked",
                    "raw_state": "blocked",
                    "usable": False,
                    "reason": reason,
                    "profile_id": self._scene_guard_profile_id,
                    "bound_model_version": self._scene_guard_bound_model_version,
                    "bound_model_artifact_sha256": (
                        self._scene_guard_bound_model_artifact_sha256
                    ),
                    "runtime_model_artifact_sha256": self._model_artifact_sha256,
                    "reference_source": self._scene_guard_reference_source,
                    "reference_sha256": self._scene_guard_reference_sha256,
                    "reference_error": self._scene_guard_reference_error,
                    "runtime_error": self._scene_guard_runtime_error,
                    "check_interval_frames": cfg.check_interval_frames,
                    "check_interval_seconds": cfg.check_interval_seconds,
                    "last_checked_frame": self._scene_guard_last_check_frame or None,
                    "quick_match_score": self._scene_guard_last_quick_score,
                    "quick_shift_px": self._scene_guard_last_quick_shift_px,
                    "reference_available": False,
                    "policy_recovery_count": self._scene_guard_policy_recovery_count,
                    "required_recovery_count": cfg.consecutive_matches,
                    "recovery_progress": {
                        "current": self._scene_guard_policy_recovery_count,
                        "required": cfg.consecutive_matches,
                        "complete": False,
                    },
                    "metrics": {"reference_available": False},
                }
            return {
                "enabled": False,
                "state": "disabled",
                "raw_state": "disabled",
                "usable": True,
                "reason": "scene guard disabled",
                "profile_id": None,
                "bound_model_version": None,
                "bound_model_artifact_sha256": None,
                "runtime_model_artifact_sha256": self._model_artifact_sha256,
                "reference_source": None,
                "reference_sha256": None,
                "reference_error": None,
                "runtime_error": None,
                "check_interval_frames": cfg.check_interval_frames,
                "check_interval_seconds": cfg.check_interval_seconds,
                "last_checked_frame": None,
                "quick_match_score": None,
                "quick_shift_px": None,
                "reference_available": False,
                "policy_recovery_count": 0,
                "required_recovery_count": cfg.consecutive_matches,
                "recovery_progress": {
                    "current": cfg.consecutive_matches,
                    "required": cfg.consecutive_matches,
                    "complete": True,
                },
                "metrics": {"reference_available": False},
            }
        with self._scene_guard_lock:
            result = self._scene_guard_last_result
            if self._scene_guard_runtime_error:
                state = "blocked"
                raw_state = "blocked"
                reason = self._scene_guard_runtime_error
            else:
                state = result.state.value if result is not None else self._scene_guard.state.value
                raw_state = result.raw_state.value if result is not None else state
                reason = (
                    result.reason
                    if result is not None
                    else self._scene_guard_reference_error or "awaiting first scene verification"
                )
                if (
                    state == "alignable"
                    and not self.camera_config.scene_guard.allow_alignable
                ):
                    reason = (
                        f"{reason}; homography reuse is disabled until this camera is "
                        "site-calibrated"
                    )
                elif self._scene_guard_policy_recovery_required and result is not None:
                    reason = (
                        f"awaiting stable policy recovery "
                        f"({self._scene_guard_policy_recovery_count}/"
                        f"{self.camera_config.scene_guard.consecutive_matches}): {reason}"
                    )
            metrics = result.metrics.to_dict() if result is not None else {
                "reference_available": self._scene_guard_reference_frame is not None,
            }
            return {
                "enabled": True,
                "state": state,
                "raw_state": raw_state,
                "usable": self._scene_guard_last_usable,
                "reason": reason,
                "profile_id": self._scene_guard_profile_id,
                "bound_model_version": self._scene_guard_bound_model_version,
                "bound_model_artifact_sha256": (
                    self._scene_guard_bound_model_artifact_sha256
                ),
                "runtime_model_artifact_sha256": self._model_artifact_sha256,
                "reference_source": self._scene_guard_reference_source,
                "reference_sha256": self._scene_guard_reference_sha256,
                "reference_error": self._scene_guard_reference_error,
                "runtime_error": self._scene_guard_runtime_error,
                "check_interval_frames": cfg.check_interval_frames,
                "check_interval_seconds": cfg.check_interval_seconds,
                "last_checked_frame": self._scene_guard_last_check_frame or None,
                "quick_match_score": self._scene_guard_last_quick_score,
                "quick_shift_px": self._scene_guard_last_quick_shift_px,
                "reference_available": self._scene_guard_reference_frame is not None,
                "policy_recovery_count": self._scene_guard_policy_recovery_count,
                "required_recovery_count": cfg.consecutive_matches,
                "recovery_progress": {
                    "current": (
                        cfg.consecutive_matches
                        if self._scene_guard_last_usable
                        else self._scene_guard_policy_recovery_count
                    ),
                    "required": cfg.consecutive_matches,
                    "complete": self._scene_guard_last_usable,
                },
                "metrics": metrics,
            }

    def _scene_policy_usable(self, result, *, checked: bool) -> bool:
        """Apply the deployment policy on top of geometric match quality."""
        state = result.state.value
        raw_state = getattr(getattr(result, "raw_state", result.state), "value", state)
        cfg = self.camera_config.scene_guard

        if not cfg.allow_alignable:
            # SceneGuard's algorithm-level debounce deliberately regards
            # STABLE and ALIGNABLE as geometrically safe so deployments that
            # opt into homography reuse can recover.  The default deployment
            # policy is stricter: when alignment is disabled, only consecutive
            # fresh raw STABLE checks may release the unwarped fixed-scene
            # model.  Counting CALIBRATING frames without inspecting raw_state
            # would allow ALIGNABLE, ALIGNABLE, STABLE to release after a
            # single policy-valid frame.
            if state == "calibrating" and not self._scene_guard_policy_recovery_required:
                self._scene_guard_policy_recovery_required = True
                self._scene_guard_policy_recovery_count = 0
            if raw_state != "stable":
                self._scene_guard_policy_recovery_required = True
                self._scene_guard_policy_recovery_count = 0
                return False
            if self._scene_guard_policy_recovery_required:
                if not checked:
                    return False
                self._scene_guard_policy_recovery_count += 1
                if self._scene_guard_policy_recovery_count < cfg.consecutive_matches:
                    return False
                if state != "stable":
                    return False
                self._scene_guard_policy_recovery_required = False
                self._scene_guard_policy_recovery_count = 0
            return bool(result.usable and state == "stable")

        if not result.usable:
            return False
        if self._scene_guard_policy_recovery_required:
            if state != "stable":
                self._scene_guard_policy_recovery_count = 0
                return False
            # Cached results are not independent evidence.  Only a fresh full
            # feature-geometry evaluation may advance the recovery streak.
            if not checked:
                return False
            self._scene_guard_policy_recovery_count += 1
            if (
                self._scene_guard_policy_recovery_count
                < cfg.consecutive_matches
            ):
                return False
            self._scene_guard_policy_recovery_required = False
            self._scene_guard_policy_recovery_count = 0
        return True

    def _fill_scene_warp_border(
        self,
        aligned: np.ndarray,
        homography: np.ndarray | None,
        source_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Replace unobservable warp borders with approved reference pixels."""
        reference = self._scene_guard_reference_frame
        if reference is None or homography is None or aligned.shape != reference.shape:
            return aligned
        source_height, source_width = source_shape[:2]
        validity_source = np.full((source_height, source_width), 255, dtype=np.uint8)
        validity = cv2.warpPerspective(
            validity_source,
            homography,
            (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        invalid = validity == 0
        if not np.any(invalid):
            return aligned
        filled = aligned.copy()
        filled[invalid] = reference[invalid]
        return filled

    def _publish_scene_guard_availability(
        self,
        *,
        usable: bool,
        state: str,
        reason: str,
    ) -> None:
        """Publish scene isolation/recovery exactly once per transition."""
        previous = self._scene_guard_last_published_usable
        if previous is None and state == "calibrating":
            return
        if previous is usable:
            return
        self._scene_guard_last_published_usable = usable
        if previous is None and usable:
            return

        event = "recovered" if usable else "entered"
        payload = {
            "type": event,
            "component": "scene_guard",
            "camera_id": self.camera_config.camera_id,
            "state": state,
            "reason": reason,
        }
        logger.info(
            "pipeline.scene_guard_availability_changed",
            **payload,
        )
        if self._event_bus is not None:
            try:
                self._event_bus.publish(
                    "pipeline.scene_guard_recovered"
                    if usable
                    else "pipeline.scene_guard_blocked",
                    payload,
                )
            except Exception:
                logger.debug(
                    "pipeline.scene_guard_event_publish_failed",
                    camera_id=self.camera_config.camera_id,
                    exc_info=True,
                )
        if self._degradation_publisher is not None:
            try:
                self._degradation_publisher("system_degradation", payload)
            except Exception:
                logger.debug(
                    "pipeline.scene_guard_degradation_publish_failed",
                    camera_id=self.camera_config.camera_id,
                    exc_info=True,
                )
        if not usable:
            get_error_channel().emit(
                severity=SEVERITY_WARNING,
                source="scene_guard",
                code="scene_guard_blocked",
                message=f"摄像头 {self.camera_config.camera_id} 场景校验未通过",
                context=payload,
            )

    def _apply_scene_guard(
        self,
        frame: np.ndarray,
        frame_data: FrameData,
        diag: FrameDiagnostics,
    ) -> tuple[np.ndarray, bool]:
        """Verify/align one raw frame and report whether detection may proceed."""
        started = time.monotonic()
        guard = self._scene_guard
        if guard is None:
            if not self.camera_config.scene_guard.enabled:
                diag.scene_guard_state = "disabled"
                return frame, True

            previous_usable = self._scene_guard_last_usable
            reason = (
                self._scene_guard_runtime_error
                or self._scene_guard_reference_error
                or "scene guard is enabled but the runtime guard is unavailable"
            )
            self._scene_guard_runtime_error = reason
            self._scene_guard_last_usable = False
            self._scene_guard_policy_recovery_required = True
            self._scene_guard_policy_recovery_count = 0
            diag.scene_guard_state = "blocked"
            diag.stages.append(StageResult(
                stage_name="scene_guard",
                duration_ms=(time.monotonic() - started) * 1000,
                skipped=True,
                skip_reason="guard_unavailable",
                metadata={"enabled": True, "reason": reason},
            ))
            if previous_usable:
                self._reset_scene_sensitive_state()
            self._publish_scene_guard_availability(
                usable=False,
                state="blocked",
                reason=reason,
            )
            return frame, False

        checked = False
        quick_score: float | None = None
        quick_shift_px: float | None = None
        result = None
        aligned = None
        previous_usable = self._scene_guard_last_usable

        try:
            check_now = time.monotonic()
            reconnect_count = getattr(getattr(self._camera, "state", None), "reconnect_count", None)
            reconnected = bool(
                reconnect_count is not None
                and self._scene_guard_last_reconnect_count is not None
                and reconnect_count != self._scene_guard_last_reconnect_count
            )
            if reconnect_count is not None:
                self._scene_guard_last_reconnect_count = int(reconnect_count)
            identity = frame_data.identity
            stream_key = (
                getattr(identity, "source_runtime", None),
                getattr(identity, "stream_id", None),
                frame.shape[:2],
            )
            stream_changed = bool(
                self._scene_guard_last_stream_key is not None
                and stream_key != self._scene_guard_last_stream_key
            )
            self._scene_guard_last_stream_key = stream_key

            with self._scene_guard_lock:
                previous = self._scene_guard_last_result
                if reconnected or stream_changed:
                    # A capture reconnect or stream identity change may point
                    # at another physical camera while preserving resolution.
                    # Retain the immutable approved reference, but require the
                    # complete fresh geometry streak before releasing the old
                    # model again.
                    guard.require_reconfirmation()
                    self._scene_guard_policy_recovery_required = (
                        not self.camera_config.scene_guard.allow_alignable
                    )
                    self._scene_guard_policy_recovery_count = 0
                should_check = bool(
                    previous is None
                    or previous.state.value == "calibrating"
                    # Unsafe decisions must be re-evaluated on every incoming
                    # frame so recovery starts immediately.  Reusing a cached
                    # BLOCKED/MOVED/UNKNOWN result until the normal TTL expires
                    # would unnecessarily keep the camera isolated and would
                    # also prevent the guard's consecutive-safe-frame counter
                    # from advancing.
                    or not previous.usable
                    or (
                        self._scene_guard_policy_recovery_required
                        and previous.state.value == "stable"
                    )
                    or self._scene_guard_runtime_error is not None
                    or reconnected
                    or stream_changed
                    or check_now - self._scene_guard_last_check_monotonic
                    >= self.camera_config.scene_guard.check_interval_seconds
                    or self._frame_counter - self._scene_guard_last_check_frame
                    >= self.camera_config.scene_guard.check_interval_frames
                )

                if not should_check:
                    sentinel_frame = frame
                    if (
                        previous.state.value == "alignable"
                        and self.camera_config.scene_guard.allow_alignable
                        and previous.homography is not None
                    ):
                        reference = self._scene_guard_reference_frame
                        if reference is None:
                            should_check = True
                        else:
                            height, width = reference.shape[:2]
                            sentinel_frame = cv2.warpPerspective(
                                frame,
                                previous.homography,
                                (width, height),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE,
                            )
                            aligned = sentinel_frame
                    if not should_check:
                        quick_score, quick_shift_px = self._scene_quick_evidence(
                            sentinel_frame
                        )
                        # The sentinel is intentionally only a trigger.  A low
                        # score never decides MOVED by itself; it forces the
                        # auditable feature-geometry check on the same frame.
                        pose_shift_triggered = bool(
                            previous.state.value == "stable"
                            and quick_shift_px is not None
                            and quick_shift_px
                            > self.camera_config.scene_guard.stable_max_shift_px
                        )
                        if (
                            quick_score is None
                            or quick_score < 0.38
                            or pose_shift_triggered
                        ):
                            should_check = True

                if should_check:
                    result = guard.evaluate(frame, return_aligned=False)
                    checked = True
                    self._scene_guard_last_result = result
                    self._scene_guard_last_check_frame = self._frame_counter
                    self._scene_guard_last_check_monotonic = check_now
                    self._scene_guard_runtime_error = None
                    aligned = None
                else:
                    result = previous

                assert result is not None
                self._scene_guard_last_quick_score = quick_score
                self._scene_guard_last_quick_shift_px = quick_shift_px
                policy_usable = self._scene_policy_usable(result, checked=checked)
                self._scene_guard_last_usable = policy_usable

        except Exception as exc:
            logger.error(
                "pipeline.scene_guard_failed",
                camera_id=frame_data.camera_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            with self._scene_guard_lock:
                self._scene_guard_runtime_error = f"scene verification failed: {exc}"
                self._scene_guard_last_usable = False
                self._scene_guard_policy_recovery_required = True
                self._scene_guard_policy_recovery_count = 0
            diag.scene_guard_state = "blocked"
            diag.stages.append(StageResult(
                stage_name="scene_guard",
                duration_ms=(time.monotonic() - started) * 1000,
                skipped=True,
                skip_reason="verification_error",
                metadata={"error": str(exc), "checked": checked},
            ))
            if previous_usable:
                self._reset_scene_sensitive_state()
            self._publish_scene_guard_availability(
                usable=False,
                state="blocked",
                reason=self._scene_guard_runtime_error or "verification_error",
            )
            return frame, False

        policy_usable = self._scene_guard_last_usable
        if previous_usable != policy_usable:
            self._reset_scene_sensitive_state()

        publish_reason = result.reason
        if result.state.value == "alignable" and not self.camera_config.scene_guard.allow_alignable:
            publish_reason = f"{publish_reason}; homography reuse disabled"
        self._publish_scene_guard_availability(
            usable=policy_usable,
            state=result.state.value,
            reason=publish_reason,
        )

        diag.scene_guard_state = result.state.value
        scene_guard_metadata = {
            "state": result.state.value,
            "raw_state": result.raw_state.value,
            "algorithm_usable": result.usable,
            "usable": policy_usable,
            "allow_alignable": self.camera_config.scene_guard.allow_alignable,
            "reason": result.reason,
            "checked": checked,
            "geometry_normalization_applied": False,
            "warp_border_policy": "not_applicable",
            "quick_match_score": (
                round(quick_score, 4) if quick_score is not None else None
            ),
            "quick_shift_px": (
                round(quick_shift_px, 3)
                if quick_shift_px is not None
                else None
            ),
            "profile_id": self._scene_guard_profile_id,
            "reference_source": self._scene_guard_reference_source,
            "metrics": result.metrics.to_dict(),
        }
        usable_frame = frame
        if (
            policy_usable
            and result.state.value == "alignable"
            and self.camera_config.scene_guard.allow_alignable
            and result.homography is not None
        ):
            if aligned is None:
                reference = self._scene_guard_reference_frame
                if reference is None:
                    return frame, False
                aligned = cv2.warpPerspective(
                    frame,
                    result.homography,
                    (reference.shape[1], reference.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            usable_frame = self._fill_scene_warp_border(
                aligned,
                result.homography,
                frame.shape,
            )
            scene_guard_metadata["geometry_normalization_applied"] = True
            scene_guard_metadata["warp_border_policy"] = (
                "approved_reference_fill_outside_source_footprint"
            )
        diag.stages.append(StageResult(
            stage_name="scene_guard",
            duration_ms=(time.monotonic() - started) * 1000,
            skipped=not policy_usable,
            skip_reason="" if policy_usable else result.state.value,
            metadata=scene_guard_metadata,
        ))
        if not policy_usable:
            return frame, False
        return usable_frame, True

    def _reset_scene_sensitive_state(self) -> None:
        """Drop temporal/background state whenever scene isolation changes."""
        self._reset_candidate_model_evidence_state()
        prefilter = getattr(self, "_prefilter", None)
        if prefilter is not None and hasattr(prefilter, "reset"):
            prefilter.reset()
        fast_motion = getattr(self, "_fast_motion", None)
        if fast_motion is not None:
            fast_motion.reset()
        reference = self._scene_guard_reference_frame
        detector_reference = reference
        if reference is not None:
            detector_reference = reference.copy()
            zone_mask = getattr(self, "_zone_mask", None)
            if zone_mask is not None:
                detector_reference = zone_mask.apply(detector_reference)
            imaging_processor = getattr(self, "_imaging_processor", None)
            if imaging_processor is not None:
                try:
                    detector_reference = imaging_processor.process(
                        detector_reference
                    ).deglared
                except Exception:
                    logger.debug(
                        "pipeline.scene_reference_imaging_failed",
                        camera_id=self.camera_config.camera_id,
                        exc_info=True,
                    )
        for name in ("_persistent_change", "_water_leak", "_plume"):
            detector = getattr(self, name, None)
            if detector is not None:
                if detector_reference is None:
                    detector.reset(clear_reference=True)
                else:
                    detector.set_reference(detector_reference)
        simplex = getattr(self, "_simplex", None)
        if simplex is not None:
            simplex.reset()
            if detector_reference is not None:
                simplex.set_reference(detector_reference)
        self._simplex_reference_set = bool(
            simplex is not None and detector_reference is not None
        )
        aligner = getattr(self, "_aligner", None)
        if aligner is not None:
            aligner.reset()
            if detector_reference is not None and hasattr(aligner, "set_reference"):
                aligner.set_reference(detector_reference, pinned=True)
        grader = getattr(self, "_alert_grader", None)
        if grader is not None:
            grader.reset()
        self._incident_associations.clear()
        physics_tracker = getattr(self, "_physics_tracker", None)
        if physics_tracker is not None:
            physics_tracker.reset()
        lock_state_lock = getattr(self, "_lock_state_lock", None)
        if lock_state_lock is None:
            self._locked = False
            self._lock_last_below_time = None
        else:
            with lock_state_lock:
                self._locked = False
                self._lock_last_below_time = None
        self._last_heartbeat_time = 0.0
        self._prev_brightness = None
        self._was_low_light = False
        self._fast_motion_suppress_until = 0.0
        self._last_inference_record = None
        self._last_grade_result = None
        latest_map_lock = getattr(self, "_latest_anomaly_map_lock", None)
        if latest_map_lock is not None:
            with latest_map_lock:
                self._latest_anomaly_map = None

    def _reset_candidate_model_evidence_state(self) -> None:
        """Start a new evidence epoch and prevent votes crossing context changes."""
        self._candidate_model_evidence_epoch = (
            int(getattr(self, "_candidate_model_evidence_epoch", 0)) + 1
        )
        self._candidate_model_evidence_counter = 0
        temporal = getattr(self, "_candidate_model_temporal", None)
        if temporal is not None:
            temporal.reset()

    @staticmethod
    def _candidate_roi_anomaly_score(
        anomaly_map: np.ndarray | None,
        bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
    ) -> float | None:
        """Return a candidate-local P95 model score, never a global frame score."""
        if anomaly_map is None:
            return None
        values = np.asarray(anomaly_map).squeeze()
        if values.ndim != 2 or values.size == 0:
            return None
        height, width = frame_shape[:2]
        if values.shape != (height, width):
            values = cv2.resize(values.astype(np.float32), (width, height))
        x1, y1, x2, y2 = bbox
        x1 = min(max(int(x1), 0), width)
        x2 = min(max(int(x2), 0), width)
        y1 = min(max(int(y1), 0), height)
        y2 = min(max(int(y2), 0), height)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = values[y1:y2, x1:x2]
        finite = roi[np.isfinite(roi)]
        if finite.size == 0:
            return None
        return float(np.percentile(finite, 95))

    def _confirm_persistent_change_candidates(
        self,
        *,
        current_frame: np.ndarray,
        persistent_result,
        anomaly_result: AnomalyResult,
        diag: FrameDiagnostics,
    ) -> list:
        """Apply explainable second-stage evidence to persistent candidates."""
        candidates = list(persistent_result.candidates)
        confirmer = self._static_candidate_confirmer
        temporal = self._candidate_model_temporal
        observation_number = int(
            getattr(diag, "candidate_model_observation_number", diag.frame_number)
        )
        if confirmer is None:
            return candidates

        # Calling this method for empty candidate frames is intentional: a
        # pending first vote must not survive a disappearance/early-return gap.
        if not candidates:
            if temporal is not None:
                temporal.finish_frame(
                    observed_track_ids=set(),
                    observation_number=observation_number,
                )
            return []

        started = time.monotonic()
        mode = self.camera_config.static_confirmation.mode
        evidence_cfg = self.camera_config.static_confirmation.model_evidence
        fusion_policy = evidence_cfg.fusion_policy
        reference = self._scene_guard_reference_frame
        effective_candidates = []
        would_accept_count = 0
        would_reject_count = 0
        evaluated_count = 0
        pending_count = 0
        unavailable_count = 0
        decisions: list[dict] = []
        observed_track_ids: set[int] = set()
        now = time.monotonic()

        output_space = anomaly_result.model_output_space
        runtime_calibration_id: str | None = None
        prepared_model_map = None
        model_thresholds = CandidateModelThresholds(
            global_score_min=evidence_cfg.global_score_min,
            roi_p95_min=evidence_cfg.roi_p95_min,
            roi_context_delta_min=evidence_cfg.roi_context_delta_min,
        )
        if evidence_cfg.enabled:
            binding = self._candidate_model_evidence_binding(
                output_space,
                model_input_variant=anomaly_result.model_input_variant,
            )
            runtime_calibration_id = (
                build_calibration_id(binding)
                if output_space is not None
                and self._model_artifact_sha256 is not None
                and self._scene_guard_reference_sha256 is not None
                else None
            )
            prepared_model_map = prepare_model_evidence_map(
                model_anomaly_map=anomaly_result.model_anomaly_map,
                model_score=anomaly_result.model_score,
                output_space=output_space,
                output_comparable=(
                    anomaly_result.model_output_comparable
                    and self._ensemble_detector is None
                    and not self._uses_shared_anomaly_detector
                ),
                detection_failed=anomaly_result.detection_failed,
            )

        if reference is None or reference.shape != current_frame.shape:
            reason = (
                "no_approved_reference"
                if reference is None
                else "reference_shape_mismatch"
            )
            for candidate in candidates:
                observed = bool(getattr(candidate, "observed_this_frame", True))
                if observed:
                    observed_track_ids.add(candidate.track_id)
                temporal_decision = None
                if temporal is not None:
                    x, y, width, height = candidate.bbox
                    temporal_decision = temporal.observe(
                        source_track_id=candidate.track_id,
                        bbox=(x, y, x + width, y + height),
                        now=now,
                        observation_number=observation_number,
                        evaluated=False,
                        passed=None,
                        status="upstream_skipped",
                        observed_this_frame=observed,
                    )
                evidence = {
                    "schema_version": 1,
                    "track_id": candidate.track_id,
                    "bbox": list(candidate.bbox),
                    "accepted": False,
                    "would_accept": None,
                    "attribution_disposition": "unavailable",
                    "decision_status": "upstream_skipped",
                    "effective": mode == "shadow",
                    "mode": mode,
                    "fusion_policy": fusion_policy,
                    "reason": reason,
                    "evidence_epoch": self._candidate_model_evidence_epoch,
                    "scene_profile_id": self._scene_guard_profile_id,
                    "model_version_id": self._model_version_id,
                    "model_evidence": (
                        CandidateModelEvidence(
                            status="upstream_skipped",
                            status_reason=reason,
                            would_accept=None,
                            calibration_id=runtime_calibration_id,
                            expected_calibration_id=evidence_cfg.calibration_id,
                            output_space=output_space,
                            output_comparable=(
                                prepared_model_map.output_comparable
                                if prepared_model_map is not None
                                else False
                            ),
                            map_shape=(
                                prepared_model_map.map_shape
                                if prepared_model_map is not None
                                else None
                            ),
                            model_score=(
                                prepared_model_map.model_score
                                if prepared_model_map is not None
                                else None
                            ),
                            model_input_variant=anomaly_result.model_input_variant,
                        ).to_dict()
                        if evidence_cfg.enabled
                        else None
                    ),
                    "temporal_evidence": (
                        temporal_decision.to_dict()
                        if temporal_decision is not None
                        else None
                    ),
                }
                decisions.append(evidence)
                unavailable_count += 1
                diag.add_candidate(
                    candidate_type="persistent_confirmation",
                    decision=(
                        "shadow_unavailable" if mode == "shadow" else "unavailable"
                    ),
                    stage="static_confirmation",
                    evidence=evidence,
                    suppression_reason=(
                        None if mode == "shadow" else f"static_confirmation:{reason}"
                    ),
                )
            if mode == "shadow":
                effective_candidates = candidates
        else:
            for candidate in candidates:
                x, y, width, height = candidate.bbox
                bbox = (x, y, x + width, y + height)
                observed = bool(getattr(candidate, "observed_this_frame", True))
                if observed:
                    observed_track_ids.add(candidate.track_id)

                # Keep the legacy display-map P95 only for explicitly configured
                # structure-only compatibility.  model_required never falls back
                # to a per-frame-normalized map.
                legacy_roi_score = None
                if fusion_policy == "structure_only":
                    legacy_roi_score = self._candidate_roi_anomaly_score(
                        anomaly_result.anomaly_map,
                        bbox,
                        current_frame.shape,
                    )

                structural_evaluated = False
                try:
                    decision = confirmer.confirm(
                        current_frame,
                        reference,
                        bbox,
                        sensitive_rois=self._scene_sensitive_rois,
                        external_anomaly_score=legacy_roi_score,
                    )
                    structural_evaluated = True
                    structural_accepted = decision.accepted
                    structural_reason = decision.reason.value
                    structural_evidence = {
                        "accepted": structural_accepted,
                        "reason": structural_reason,
                        "analysis_bbox": list(decision.analysis_bbox),
                        "in_sensitive_roi": decision.in_sensitive_roi,
                        "required_structure_score": decision.required_structure_score,
                        "metrics": asdict(decision.metrics),
                    }
                except Exception as exc:
                    structural_accepted = False
                    structural_reason = f"confirmation_error:{type(exc).__name__}"
                    structural_evidence = {
                        "accepted": False,
                        "reason": structural_reason,
                        "error": str(exc),
                    }

                model_evidence: CandidateModelEvidence | None = None
                if evidence_cfg.enabled:
                    self._candidate_model_evidence_counter += 1
                    sampled = (
                        (self._candidate_model_evidence_counter - 1)
                        % evidence_cfg.sample_every_n_candidates
                        == 0
                    )
                    if not sampled:
                        model_evidence = CandidateModelEvidence(
                            status="rate_limited",
                            status_reason="candidate_sampling_policy",
                            would_accept=None,
                            calibration_id=runtime_calibration_id,
                            expected_calibration_id=evidence_cfg.calibration_id,
                            output_space=output_space,
                            output_comparable=(
                                prepared_model_map.output_comparable
                                if prepared_model_map is not None
                                else False
                            ),
                            map_shape=(
                                prepared_model_map.map_shape
                                if prepared_model_map is not None
                                else None
                            ),
                            model_score=(
                                prepared_model_map.model_score
                                if prepared_model_map is not None
                                else None
                            ),
                            model_input_variant=anomaly_result.model_input_variant,
                        )
                    else:
                        model_evidence = evaluate_candidate_model_evidence(
                            model_anomaly_map=anomaly_result.model_anomaly_map,
                            bbox=bbox,
                            frame_shape=current_frame.shape,
                            model_score=anomaly_result.model_score,
                            output_space=output_space,
                            output_comparable=(
                                anomaly_result.model_output_comparable
                                and self._ensemble_detector is None
                            ),
                            calibration_id=runtime_calibration_id,
                            expected_calibration_id=evidence_cfg.calibration_id,
                            thresholds=model_thresholds,
                            context_ratio=evidence_cfg.context_ratio,
                            hot_pixel_threshold=evidence_cfg.hot_pixel_threshold,
                            detection_failed=anomaly_result.detection_failed,
                            prepared_map=prepared_model_map,
                            model_input_variant=anomaly_result.model_input_variant,
                        )

                temporal_decision = None
                if fusion_policy == "model_required":
                    model_evaluated = bool(
                        model_evidence is not None
                        and model_evidence.status == "evaluated"
                    )
                    combined_evaluated = structural_evaluated and model_evaluated
                    combined_pass = (
                        structural_accepted and bool(model_evidence.would_accept)
                        if combined_evaluated and model_evidence is not None
                        else None
                    )
                    combined_status = (
                        model_evidence.status
                        if model_evidence is not None and not model_evaluated
                        else "evaluated"
                        if structural_evaluated
                        else "inference_error"
                    )
                else:
                    combined_evaluated = structural_evaluated
                    combined_pass = structural_accepted if structural_evaluated else None
                    combined_status = "evaluated" if structural_evaluated else "inference_error"

                if temporal is not None:
                    temporal_decision = temporal.observe(
                        source_track_id=candidate.track_id,
                        bbox=bbox,
                        now=now,
                        observation_number=observation_number,
                        evaluated=combined_evaluated,
                        passed=combined_pass,
                        status=combined_status,
                        observed_this_frame=observed,
                    )

                if fusion_policy == "model_required":
                    final_would_accept = (
                        temporal_decision.would_accept
                        if temporal_decision is not None
                        else None
                    )
                else:
                    final_would_accept = (
                        structural_accepted if structural_evaluated else None
                    )
                accepted = final_would_accept is True
                is_pending = bool(
                    temporal_decision is not None
                    and temporal_decision.status == "evaluated"
                    and temporal_decision.streak > 0
                    and not temporal_decision.confirmed
                )
                if accepted:
                    attribution_disposition = "accepted"
                elif final_would_accept is None:
                    attribution_disposition = "unavailable"
                elif is_pending:
                    attribution_disposition = "pending"
                else:
                    attribution_disposition = "rejected"
                if final_would_accept is None:
                    unavailable_count += 1
                    decision_status = combined_status
                else:
                    evaluated_count += 1
                    if accepted:
                        would_accept_count += 1
                    elif is_pending:
                        pending_count += 1
                    else:
                        would_reject_count += 1
                    decision_status = "evaluated"

                reason = (
                    temporal_decision.status_reason
                    if fusion_policy == "model_required" and temporal_decision is not None
                    else structural_reason
                )
                evidence = {
                    "schema_version": 1,
                    "track_id": candidate.track_id,
                    "bbox": list(bbox),
                    "accepted": accepted,
                    "would_accept": final_would_accept,
                    "attribution_disposition": attribution_disposition,
                    "decision_status": decision_status,
                    "effective": accepted or mode == "shadow",
                    "mode": mode,
                    "fusion_policy": fusion_policy,
                    "reason": reason,
                    "observed_this_frame": observed,
                    "evidence_epoch": self._candidate_model_evidence_epoch,
                    "scene_guard_state": getattr(diag, "scene_guard_state", None),
                    "scene_profile_id": self._scene_guard_profile_id,
                    "model_version_id": self._model_version_id,
                    "pipeline_effective_score": anomaly_result.anomaly_score,
                    "structural_evidence": structural_evidence,
                    "model_evidence": (
                        model_evidence.to_dict() if model_evidence is not None else None
                    ),
                    "temporal_evidence": (
                        temporal_decision.to_dict()
                        if temporal_decision is not None
                        else None
                    ),
                }
                decisions.append(evidence)
                if accepted or mode == "shadow":
                    effective_candidates.append(candidate)
                if mode == "shadow":
                    if accepted:
                        candidate_decision = "shadow_accept"
                    elif final_would_accept is None:
                        candidate_decision = "shadow_unavailable"
                    elif is_pending:
                        candidate_decision = "shadow_pending"
                    else:
                        candidate_decision = "shadow_reject"
                elif accepted:
                    candidate_decision = "confirmed"
                elif final_would_accept is None:
                    candidate_decision = "unavailable"
                elif is_pending:
                    candidate_decision = "pending"
                else:
                    candidate_decision = "rejected"
                diag.add_candidate(
                    candidate_type="persistent_confirmation",
                    decision=candidate_decision,
                    stage="static_confirmation",
                    evidence=evidence,
                    suppression_reason=(
                        None
                        if accepted or mode == "shadow"
                        else f"static_confirmation:{reason}"
                    ),
                )

        if temporal is not None:
            temporal.finish_frame(
                observed_track_ids=observed_track_ids,
                observation_number=observation_number,
            )

        if effective_candidates:
            stage_skip_reason = ""
        elif unavailable_count == len(candidates):
            stage_skip_reason = "candidate_evidence_unavailable"
        elif pending_count:
            stage_skip_reason = "candidate_confirmation_pending"
        else:
            stage_skip_reason = "all_candidates_rejected"
        diag.stages.append(StageResult(
            stage_name="static_confirmation",
            duration_ms=(time.monotonic() - started) * 1000,
            skipped=not bool(effective_candidates),
            skip_reason=stage_skip_reason,
            metadata={
                "mode": mode,
                "fusion_policy": fusion_policy,
                "evidence_epoch": self._candidate_model_evidence_epoch,
                "candidate_count": len(candidates),
                "evaluated_count": evaluated_count,
                "would_accept_count": would_accept_count,
                "would_reject_count": would_reject_count,
                "pending_count": pending_count,
                "unavailable_count": unavailable_count,
                "effective_count": len(effective_candidates),
                "decisions": decisions,
            },
        ))
        return effective_candidates

    def _candidate_model_evidence_binding(
        self,
        output_space: str | None,
        *,
        model_input_variant: str = "scene_stable:unmasked",
    ) -> dict[str, object]:
        """Describe every runtime dimension that changes model-map thresholds."""
        detector_status = getattr(self._anomaly_detector, "status", None)
        backend = getattr(
            detector_status,
            "backend",
            self._anomaly_inference_backend,
        )
        anomaly_cfg = self.camera_config.anomaly
        evidence_cfg = self.camera_config.static_confirmation.model_evidence
        runtime_image_size = getattr(detector_status, "image_size", None)
        if not (
            isinstance(runtime_image_size, tuple | list)
            and len(runtime_image_size) == 2
            and all(isinstance(value, int) for value in runtime_image_size)
        ):
            runtime_image_size = anomaly_cfg.image_size
        input_contract = self._candidate_model_input_contract()
        input_contract_json = json.dumps(
            input_contract,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return {
            "schema_version": 2,
            "camera_id": self.camera_config.camera_id,
            "model_version_id": self._model_version_id,
            "model_artifact_sha256": self._model_artifact_sha256,
            "scene_profile_id": self._scene_guard_profile_id,
            "scene_reference_sha256": self._scene_guard_reference_sha256,
            "backend": str(backend),
            "output_space": output_space,
            "output_transform": "argus-model-evidence-v2",
            "algorithm_contract": dict(_CANDIDATE_EVIDENCE_ALGORITHM_CONTRACT),
            "runtime_dependencies": {
                "anomalib": _runtime_package_version("anomalib"),
                "numpy": _runtime_package_version("numpy"),
                "opencv_python": _runtime_package_version("opencv-python"),
                "openvino": _runtime_package_version("openvino"),
                "torch": _runtime_package_version("torch"),
            },
            "candidate_evidence_contract": {
                "roi_statistic": "native-map-p95-v1",
                "bbox_mapping": "floor-ceil-clipped-v1",
                "context_ratio": float(evidence_cfg.context_ratio),
                "hot_pixel_threshold": float(evidence_cfg.hot_pixel_threshold),
                "sample_every_n_candidates": int(
                    evidence_cfg.sample_every_n_candidates
                ),
                "temporal_required_passes": int(
                    evidence_cfg.temporal_required_passes
                ),
                "temporal_min_gap_seconds": float(
                    evidence_cfg.temporal_min_gap_seconds
                ),
                "temporal_max_gap_seconds": float(
                    evidence_cfg.temporal_max_gap_seconds
                ),
                "temporal_iou_threshold": float(
                    evidence_cfg.temporal_iou_threshold
                ),
                "anomaly_attribution": (
                    "connected-component-partitioned-map-overlap-v3"
                ),
                "anomaly_attribution_min_hot_coverage": float(
                    evidence_cfg.anomaly_attribution_min_hot_coverage
                ),
            },
            "model_input_variant": model_input_variant,
            "model_input_contract_sha256": hashlib.sha256(
                input_contract_json.encode("utf-8")
            ).hexdigest(),
            "model_image_size": list(runtime_image_size),
            "multiscale": bool(anomaly_cfg.enable_multiscale),
            "pyramid_mode": bool(anomaly_cfg.pyramid_mode),
            "tile_size": int(anomaly_cfg.tile_size),
            "tile_overlap": float(anomaly_cfg.tile_overlap),
            "pyramid_sizes": list(anomaly_cfg.pyramid_sizes),
            "ensemble": self._ensemble_detector is not None,
            "shared_detector": self._uses_shared_anomaly_detector,
        }

    def _apply_candidate_anomaly_attribution(
        self,
        anomaly_result: AnomalyResult,
        *,
        diag: FrameDiagnostics,
        frame_shape: tuple[int, ...],
        simplex_detected: bool,
        persistent_candidates: list | None = None,
    ) -> AnomalyResult:
        """Remove rejected candidate support from the duplicate appearance map.

        Candidate confirmation and the generic anomaly head observe the same
        model map.  A rejected reflection must therefore be removed from the
        appearance map even when another candidate in the frame is accepted.
        Accepted, pending, and unavailable component support wins every
        overlap, while model-hot pixels outside all candidates remain
        untouched.  The legacy whole-channel suppression is retained only
        when every record is explicitly rejected and every model-hot pixel
        belongs exclusively to a reject.
        """
        confirmation = self.camera_config.static_confirmation
        evidence_cfg = confirmation.model_evidence
        if (
            not confirmation.enabled
            or evidence_cfg.fusion_policy != "model_required"
            or not anomaly_result.is_anomalous
        ):
            return anomaly_result

        records = [
            record.evidence
            for record in diag.candidate_records
            if record.candidate_type == "persistent_confirmation"
            and isinstance(record.evidence, dict)
        ]
        if not records:
            return anomaly_result

        candidates_by_track_id = {}
        for candidate in persistent_candidates or []:
            try:
                track_id = int(candidate.track_id)
            except (AttributeError, OverflowError, TypeError, ValueError):
                continue
            candidates_by_track_id[track_id] = candidate

        accepted_records: list[dict] = []
        rejected_records: list[dict] = []
        pending_records: list[dict] = []
        unavailable_records: list[dict] = []
        for record in records:
            disposition = record.get("attribution_disposition")
            if disposition == "accepted":
                accepted_records.append(record)
            elif disposition == "rejected":
                rejected_records.append(record)
            elif disposition == "pending":
                pending_records.append(record)
            else:
                # Missing, unknown, or explicitly unavailable evidence cannot
                # safely authorize removal.  Treat its exact support as
                # protected if available.
                unavailable_records.append(record)

        reason = "independent_hotspot_preserved"
        hot_pixels = 0
        covered_hot_pixels = 0
        hot_coverage: float | None = None
        would_suppress = False
        would_remove_partial = False
        partial_removal_pixel_count = 0
        partial_removal_support_pixel_count = 0
        partial_removal_coverage: float | None = None
        partial_removal_reason = "no_rejected_candidate"
        display_removal_mask: np.ndarray | None = None
        protected_hot_pixels = 0
        independent_hot_pixels = 0
        overlap_hot_pixels = 0
        removal_occupancy_threshold = 0.95
        native_remaining_hot: np.ndarray | None = None
        native_support_projected_pixel_count = 0
        scalar_support_reconstructed = False

        if simplex_detected:
            reason = "simplex_safety_channel_detected"
            partial_removal_reason = "simplex_safety_channel_detected"
        elif not rejected_records:
            reason = (
                "confirmed_candidate_present"
                if accepted_records
                else "candidate_confirmation_pending"
                if pending_records
                else "candidate_evidence_unavailable"
            )
        else:
            prepared = prepare_model_evidence_map(
                model_anomaly_map=anomaly_result.model_anomaly_map,
                model_score=anomaly_result.model_score,
                output_space=anomaly_result.model_output_space,
                output_comparable=(
                    anomaly_result.model_output_comparable
                    and self._ensemble_detector is None
                    and not self._uses_shared_anomaly_detector
                ),
                detection_failed=anomaly_result.detection_failed,
            )
            if prepared.status != "ready" or prepared.values is None:
                reason = f"model_map_{prepared.status}"
                partial_removal_reason = reason
            elif (
                len(frame_shape) < 2
                or frame_shape[0] <= 0
                or frame_shape[1] <= 0
            ):
                reason = "frame_shape_invalid"
                partial_removal_reason = reason
            else:
                values = prepared.values
                hot = values >= evidence_cfg.hot_pixel_threshold
                hot_pixels = int(np.count_nonzero(hot))
                if hot_pixels == 0:
                    reason = "no_model_hot_support"
                    partial_removal_reason = reason
                # Local appearance-map attribution does not depend on the
                # native-map hot threshold: a low-native-score reject can still
                # be bright in the legacy per-frame-normalized display map.
                if rejected_records:
                    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
                    support_unavailable = False

                    frame_supports: dict[str, np.ndarray] = {}
                    record_groups = (
                        ("accepted", accepted_records),
                        ("rejected", rejected_records),
                        ("pending", pending_records),
                        ("unavailable", unavailable_records),
                    )
                    for disposition, grouped_records in record_groups:
                        for record in grouped_records:
                            try:
                                track_id = int(record["track_id"])
                            except (KeyError, OverflowError, TypeError, ValueError):
                                support_unavailable = True
                                break
                            candidate = candidates_by_track_id.get(track_id)
                            support_mask = getattr(candidate, "support_mask", None)
                            bbox = getattr(candidate, "bbox", None)
                            if not (
                                isinstance(support_mask, np.ndarray)
                                and support_mask.ndim == 2
                                and support_mask.size > 0
                                and isinstance(bbox, (tuple, list))
                                and len(bbox) == 4
                            ):
                                support_unavailable = True
                                break
                            try:
                                x, y, width, height = (
                                    int(value) for value in bbox
                                )
                            except (OverflowError, TypeError, ValueError):
                                support_unavailable = True
                                break
                            if (
                                width <= 0
                                or height <= 0
                                or support_mask.shape != (height, width)
                                or x < 0
                                or y < 0
                                or x + width > frame_w
                                or y + height > frame_h
                            ):
                                support_unavailable = True
                                break

                            # Keep the connected component exact in frame space;
                            # projection below applies disposition-specific
                            # ownership without filling a concave bbox interior.
                            frame_support = frame_supports.get(disposition)
                            if frame_support is None:
                                frame_support = np.zeros(
                                    (frame_h, frame_w), dtype=np.uint8
                                )
                                frame_supports[disposition] = frame_support
                            support_region = frame_support[
                                y : y + height, x : x + width
                            ]
                            np.maximum(
                                support_region,
                                support_mask.astype(np.uint8, copy=False),
                                out=support_region,
                            )
                        if support_unavailable:
                            break

                    if support_unavailable:
                        reason = "candidate_component_support_unavailable"
                        partial_removal_reason = reason
                    else:

                        def project_supports(
                            target_shape: tuple[int, int],
                        ) -> tuple[
                            np.ndarray,
                            np.ndarray,
                            np.ndarray,
                            np.ndarray,
                        ]:
                            target_h, target_w = target_shape
                            accepted = np.zeros(target_shape, dtype=bool)
                            rejected = np.zeros(target_shape, dtype=bool)
                            pending = np.zeros(target_shape, dtype=bool)
                            unavailable = np.zeros(target_shape, dtype=bool)
                            destinations = {
                                "accepted": accepted,
                                "rejected": rejected,
                                "pending": pending,
                                "unavailable": unavailable,
                            }
                            for (
                                disposition,
                                frame_support,
                            ) in frame_supports.items():
                                projected_fraction = cv2.resize(
                                    frame_support.astype(np.float32),
                                    (target_w, target_h),
                                    interpolation=cv2.INTER_AREA,
                                )
                                # A protective component owns a target pixel on
                                # any overlap.  A rejected component may clear a
                                # coarse target pixel only when it occupies
                                # nearly all of it; a tiny source overlap must
                                # never expand into whole-pixel suppression.
                                if disposition == "rejected":
                                    destinations[disposition] |= (
                                        projected_fraction
                                        >= removal_occupancy_threshold
                                    )
                                else:
                                    destinations[disposition] |= (
                                        projected_fraction > 0.0
                                    )
                            return accepted, rejected, pending, unavailable

                        (
                            accepted_model,
                            rejected_model,
                            pending_model,
                            unavailable_model,
                        ) = project_supports(values.shape)
                        protected_model = (
                            accepted_model | pending_model | unavailable_model
                        )
                        removable_model = rejected_model & ~protected_model
                        native_remaining_hot = hot & ~removable_model
                        covered_hot_pixels = int(
                            np.count_nonzero(hot & removable_model)
                        )
                        hot_coverage = (
                            covered_hot_pixels / hot_pixels
                            if hot_pixels > 0
                            else None
                        )
                        partial_removal_coverage = hot_coverage
                        protected_hot_pixels = int(
                            np.count_nonzero(hot & protected_model)
                        )
                        overlap_hot_pixels = int(
                            np.count_nonzero(hot & rejected_model & protected_model)
                        )
                        independent_hot_pixels = int(
                            np.count_nonzero(
                                hot
                                & ~(
                                    accepted_model
                                    | rejected_model
                                    | pending_model
                                    | unavailable_model
                                )
                            )
                        )

                        # Never discard an unassigned or protected hot pixel,
                        # even if a deployment lowers the legacy coverage knob.
                        all_hot_exclusively_rejected = bool(
                            covered_hot_pixels == hot_pixels
                            and protected_hot_pixels == 0
                            and independent_hot_pixels == 0
                        )
                        would_suppress = bool(
                            len(rejected_records) == len(records)
                            and hot_pixels > 0
                            and all_hot_exclusively_rejected
                            and hot_coverage is not None
                            and hot_coverage
                            >= evidence_cfg.anomaly_attribution_min_hot_coverage
                        )

                        appearance_map = anomaly_result.anomaly_map
                        if (
                            isinstance(appearance_map, np.ndarray)
                            and appearance_map.ndim == 2
                            and appearance_map.size > 0
                        ):
                            (
                                accepted_display,
                                rejected_display,
                                pending_display,
                                unavailable_display,
                            ) = project_supports(appearance_map.shape)
                            # Accepted/pending/unavailable ownership has priority
                            # at overlaps; only reject-exclusive support clears.
                            display_removal_mask = rejected_display & ~(
                                accepted_display
                                | pending_display
                                | unavailable_display
                            )
                            partial_removal_support_pixel_count = int(
                                np.count_nonzero(display_removal_mask)
                            )
                            partial_removal_pixel_count = int(
                                np.count_nonzero(
                                    display_removal_mask
                                    & np.not_equal(appearance_map, 0)
                                )
                            )
                            would_remove_partial = bool(
                                partial_removal_pixel_count > 0
                            )
                            partial_removal_reason = (
                                "full_suppression_selected"
                                if would_suppress
                                else "rejected_candidate_support_mapped"
                                if would_remove_partial
                                else "rejected_support_has_no_display_evidence"
                            )
                        elif would_suppress:
                            # Whole-channel suppression is still safe because it
                            # was proven from the stable native model map.
                            partial_removal_reason = "full_suppression_selected"
                        else:
                            partial_removal_reason = "appearance_map_unavailable"

                        if would_suppress:
                            reason = (
                                "all_hot_support_attributed_to_rejected_candidates"
                            )
                        elif would_remove_partial:
                            reason = "rejected_candidate_support_partially_removed"
                        else:
                            reason = "independent_hotspot_preserved"

        full_suppression_effective = bool(
            confirmation.mode == "enforce" and would_suppress
        )
        partial_removal_effective = bool(
            confirmation.mode == "enforce"
            and would_remove_partial
            and not would_suppress
        )
        effective = full_suppression_effective or partial_removal_effective
        score_before = float(anomaly_result.anomaly_score)
        score_after = score_before
        score_scale = 1.0
        display_peak_before: float | None = None
        display_peak_after: float | None = None
        cleaned_map: np.ndarray | None = None
        is_anomalous_after = anomaly_result.is_anomalous
        preserve_scalar_for_native_support = bool(
            protected_hot_pixels > 0 or independent_hot_pixels > 0
        )

        if full_suppression_effective:
            score_after = 0.0
            score_scale = 0.0
            is_anomalous_after = False
        elif partial_removal_effective:
            cleaned_map = anomaly_result.anomaly_map.copy()
            cleaned_map[display_removal_mask] = 0
            if (
                preserve_scalar_for_native_support
                and native_remaining_hot is not None
                and np.any(native_remaining_hot)
                and math.isfinite(score_before)
                and score_before > 0.0
            ):
                projected_native_support = cv2.resize(
                    native_remaining_hot.astype(np.float32),
                    (cleaned_map.shape[1], cleaned_map.shape[0]),
                    interpolation=cv2.INTER_AREA,
                ) > 0.0
                # Projection at a coarse-cell boundary must never paint back
                # support that was explicitly removed in display space.
                projected_native_support &= ~display_removal_mask
                native_support_projected_pixel_count = int(
                    np.count_nonzero(projected_native_support)
                )
                if native_support_projected_pixel_count > 0:
                    cleaned_map[projected_native_support] = np.maximum(
                        cleaned_map[projected_native_support],
                        score_before,
                    )
                    scalar_support_reconstructed = True
            before_finite = anomaly_result.anomaly_map[
                np.isfinite(anomaly_result.anomaly_map)
            ]
            after_finite = cleaned_map[np.isfinite(cleaned_map)]
            if before_finite.size:
                display_peak_before = max(float(np.max(before_finite)), 0.0)
            if after_finite.size:
                display_peak_after = max(float(np.max(after_finite)), 0.0)
            if preserve_scalar_for_native_support:
                score_scale = 1.0
            elif display_peak_before is not None and display_peak_before > 0.0:
                remaining_peak = display_peak_after or 0.0
                score_scale = min(
                    1.0,
                    max(0.0, remaining_peak / display_peak_before),
                )
            # Missing/non-positive source-map evidence cannot safely authorize
            # an additional scalar reduction beyond the local map cleanup.
            if math.isfinite(score_before):
                score_after = max(0.0, score_before * score_scale)
                if preserve_scalar_for_native_support:
                    is_anomalous_after = anomaly_result.is_anomalous
                else:
                    threshold = float(anomaly_result.threshold)
                    is_anomalous_after = bool(
                        not math.isfinite(threshold)
                        or score_after >= threshold
                    )

        attribution = {
            "schema_version": 1,
            "mode": confirmation.mode,
            "would_suppress": would_suppress,
            "effective": effective,
            "reason": reason,
            "hot_pixel_threshold": evidence_cfg.hot_pixel_threshold,
            "required_hot_coverage": (
                evidence_cfg.anomaly_attribution_min_hot_coverage
            ),
            "hot_pixels": hot_pixels,
            "covered_hot_pixels": covered_hot_pixels,
            "hot_coverage": hot_coverage,
            "candidate_count": len(records),
            "accepted_candidate_count": len(accepted_records),
            "rejected_candidate_count": len(rejected_records),
            "pending_candidate_count": len(pending_records),
            "unavailable_candidate_count": len(unavailable_records),
            "protected_hot_pixels": protected_hot_pixels,
            "independent_hot_pixels": independent_hot_pixels,
            "overlap_hot_pixels": overlap_hot_pixels,
            "would_remove_partial": would_remove_partial,
            "partial_removal_effective": partial_removal_effective,
            "partial_removal_pixel_count": partial_removal_pixel_count,
            "partial_removal_support_pixel_count": (
                partial_removal_support_pixel_count
            ),
            "partial_removal_coverage": partial_removal_coverage,
            "partial_removal_reason": partial_removal_reason,
            "appearance_score_before": score_before,
            "appearance_score_after": score_after,
            "appearance_score_scale": score_scale,
            "display_peak_before": display_peak_before,
            "display_peak_after": display_peak_after,
            "scalar_preserved_for_native_support": (
                preserve_scalar_for_native_support
            ),
            "native_support_projected_pixel_count": (
                native_support_projected_pixel_count
            ),
            "scalar_support_reconstructed": scalar_support_reconstructed,
            "removal_occupancy_threshold": removal_occupancy_threshold
            if rejected_records and not simplex_detected
            else None,
            "simplex_detected": simplex_detected,
            "model_input_variant": anomaly_result.model_input_variant,
            "support_contract": (
                "connected-component-partitioned-map-overlap-v3"
            ),
        }
        diag.add_candidate(
            candidate_type="anomaly_attribution",
            decision=(
                "suppressed"
                if full_suppression_effective
                else "partially_suppressed"
                if partial_removal_effective
                else "shadow_suppress"
                if would_suppress
                else "shadow_partial_suppress"
                if confirmation.mode == "shadow" and would_remove_partial
                else "preserved"
            ),
            stage="static_confirmation",
            evidence=attribution,
            suppression_reason=(
                "static_confirmation:duplicate_global_hotspot"
                if full_suppression_effective
                else "static_confirmation:rejected_candidate_support"
                if partial_removal_effective
                else None
            ),
        )
        if not effective:
            return anomaly_result
        diag.anomaly_score = score_after
        diag.is_anomalous = is_anomalous_after
        if full_suppression_effective:
            return replace(
                anomaly_result,
                anomaly_score=score_after,
                anomaly_map=None,
                is_anomalous=is_anomalous_after,
            )

        return replace(
            anomaly_result,
            anomaly_score=score_after,
            anomaly_map=cleaned_map,
            is_anomalous=is_anomalous_after,
        )

    def _candidate_model_input_contract(self) -> dict[str, object]:
        """Return the stable configuration contract that shapes model input."""

        def _dump(value: object | None) -> object:
            if value is None:
                return None
            model_dump = getattr(value, "model_dump", None)
            if callable(model_dump):
                return model_dump(mode="json")
            return {
                name: getattr(value, name)
                for name in (
                    "enabled",
                    "polarization_processing",
                    "deglare_method",
                    "dolp_threshold",
                    "fusion_channels",
                )
                if hasattr(value, name)
            }

        return {
            "schema_version": 2,
            "pipeline_order": (
                "scene_guard>zone_mask>imaging>alignment>low_light>"
                "conditional_person_mask"
            ),
            "scene_guard": _dump(self.camera_config.scene_guard),
            "scene_warp_contract": (
                "opencv-warpPerspective:INTER_LINEAR:BORDER_REPLICATE:"
                "argus-fill-scene-warp-border-v1"
            ),
            "camera_resolution": list(self.camera_config.resolution),
            "zones": [_dump(zone) for zone in self.camera_config.zones],
            "alignment": _dump(self.camera_config.alignment),
            "low_light": _dump(self.camera_config.low_light),
            "person_filter": _dump(self.camera_config.person_filter),
            "imaging": _dump(self._imaging_config),
            "polarization_runtime_enabled": bool(
                getattr(self, "_polarization_enabled", False)
            ),
        }

    def _record_candidate_model_evidence_skip(
        self,
        *,
        persistent_result,
        diag: FrameDiagnostics,
        status: str,
        reason: str,
    ) -> None:
        """Record a tri-state model observation for an upstream early return."""
        temporal = self._candidate_model_temporal
        if temporal is None or persistent_result is None:
            return
        candidates = list(getattr(persistent_result, "candidates", ()))
        now = time.monotonic()
        observation_number = int(
            getattr(diag, "candidate_model_observation_number", diag.frame_number)
        )
        for candidate in candidates:
            x, y, width, height = candidate.bbox
            observed = bool(getattr(candidate, "observed_this_frame", True))
            temporal_decision = temporal.observe(
                source_track_id=candidate.track_id,
                bbox=(x, y, x + width, y + height),
                now=now,
                observation_number=observation_number,
                evaluated=False,
                passed=None,
                status=status,
                observed_this_frame=observed,
            )
            diag.add_candidate(
                candidate_type="persistent_confirmation",
                decision="skipped",
                stage="static_confirmation",
                evidence={
                    "schema_version": 1,
                    "track_id": candidate.track_id,
                    "bbox": [x, y, x + width, y + height],
                    "accepted": False,
                    "would_accept": None,
                    "decision_status": status,
                    "effective": False,
                    "mode": self.camera_config.static_confirmation.mode,
                    "fusion_policy": (
                        self.camera_config.static_confirmation.model_evidence.fusion_policy
                    ),
                    "reason": reason,
                    "observed_this_frame": observed,
                    "evidence_epoch": self._candidate_model_evidence_epoch,
                    "scene_profile_id": self._scene_guard_profile_id,
                    "model_version_id": self._model_version_id,
                    "model_evidence": {
                        "status": status,
                        "status_reason": reason,
                        "would_accept": None,
                    },
                    "temporal_evidence": temporal_decision.to_dict(),
                },
                suppression_reason=reason,
            )
        temporal.finish_frame(
            observed_track_ids=set(),
            observation_number=observation_number,
        )

    def _try_create_ensemble(
        self,
        camera_config: CameraConfig,
        primary_model_path: Path | None,
    ) -> object | None:
        """Create a DetectorEnsemble if enough models are available.

        Discovers all model files for this camera, builds EnsembleConfig from
        the Pydantic config, and loads the ensemble. Falls back to None (single
        detector) if fewer than 2 models exist or loading fails.

        Safety gate: only models at the "production" stage in the model registry
        are eligible for the ensemble. If the registry is unavailable (no DB),
        falls back to filesystem discovery with a warning.
        """
        try:
            from argus.anomaly.ensemble import DetectorEnsemble, EnsembleConfig

            camera_id = camera_config.camera_id
            all_models = find_all_models(camera_id)

            # Safety gate: filter discovered models to only production-stage models.
            # This prevents candidate/shadow/canary models from bypassing the
            # four-stage release pipeline and entering the production ensemble.
            if self._database is not None:
                try:
                    from argus.storage.model_registry import ModelRegistry
                    registry = ModelRegistry(session_factory=self._database.get_session)
                    production_records = registry.get_runtime_approved_production(
                        camera_id
                    )
                    production_paths = {
                        Path(r.model_path).resolve()
                        for r in production_records
                        if r.model_path
                    }
                    filtered_models = [m for m in all_models if m.resolve() in production_paths]
                    logger.info(
                        "pipeline.ensemble_registry_filter",
                        camera_id=camera_id,
                        discovered=len(all_models),
                        production_approved=len(filtered_models),
                    )
                    all_models = filtered_models
                except Exception:
                    logger.warning(
                        "pipeline.ensemble_registry_filter_failed",
                        camera_id=camera_id,
                        exc_info=True,
                    )
                    # Registry validation is a trust boundary.  Keep only the
                    # explicitly supplied primary below; never fall through to
                    # unfiltered filesystem discovery.
                    all_models = []
            else:
                logger.warning(
                    "pipeline.ensemble_no_registry",
                    camera_id=camera_id,
                    reason=(
                        "database unavailable; unregistered ensemble members "
                        "are disabled"
                    ),
                )
                all_models = []

            # Include the primary model path if not already in the list
            if primary_model_path is not None:
                resolved_primary = primary_model_path.resolve()
                if resolved_primary not in {m for m in all_models}:
                    all_models.insert(0, resolved_primary)

            if len(all_models) < 2:
                logger.info(
                    "pipeline.ensemble_skipped",
                    camera_id=camera_id,
                    model_count=len(all_models),
                    reason="fewer than 2 models available",
                )
                return None

            ens_cfg = camera_config.anomaly.ensemble
            ensemble_config = EnsembleConfig(
                model_paths=[str(p) for p in all_models],
                method=ens_cfg.method,
                weights=ens_cfg.weights,
                image_size=camera_config.anomaly.image_size,
                threshold=camera_config.anomaly.threshold,
                bayesian_prior=ens_cfg.bayesian_prior,
                dynamic_fpr_weighting=ens_cfg.dynamic_fpr_weighting,
            )

            ensemble = DetectorEnsemble(ensemble_config)
            if not ensemble.load():
                logger.warning(
                    "pipeline.ensemble_load_failed",
                    camera_id=camera_id,
                )
                return None

            if ensemble.model_count < 2:
                logger.info(
                    "pipeline.ensemble_skipped",
                    camera_id=camera_id,
                    loaded=ensemble.model_count,
                    reason="fewer than 2 models loaded successfully",
                )
                return None

            logger.info(
                "pipeline.ensemble_active",
                camera_id=camera_id,
                model_count=ensemble.model_count,
                method=ens_cfg.method,
            )
            return ensemble

        except Exception:
            logger.exception(
                "pipeline.ensemble_creation_failed",
                camera_id=camera_config.camera_id,
            )
            return None

    def _categorize_alert(
        self,
        alert,
        simplex_result,
        detection_result,
        frame_shape: tuple | None = None,
    ) -> str:
        """Auto-categorize alert based on detection characteristics."""
        _ENV_LABELS = {
            "insect",
            "bird",
            "shadow",
            "reflection",
            "crane",
            "overhead-bridge",
            "scaffold",
        }

        alert_detection_type = getattr(alert, "detection_type", None)
        alert_detection_type = getattr(alert_detection_type, "value", alert_detection_type)
        if alert_detection_type == DetectionType.PROJECTILE.value:
            return "projectile"
        if alert_detection_type == DetectionType.PERSISTENT_CHANGE.value:
            return "static_foreign"
        if alert_detection_type == DetectionType.WATER_LEAK.value:
            return "water_leak"
        if alert_detection_type == DetectionType.PLUME.value:
            return "visible_plume"
        for obj in getattr(alert, "detected_objects", []) or []:
            class_name = (obj.get("class_name") or "").lower()
            if class_name == "fast_projectile":
                return "projectile"
            if class_name in {"simplex_static", "static_foreign"}:
                return "static_foreign"
            if class_name == "visible_water_leak":
                return "water_leak"
            if class_name == "visible_plume":
                return "visible_plume"

        if any(
            obj.get("object_safety_alert_contributed") is True
            for obj in (getattr(alert, "detected_objects", []) or [])
        ):
            return "equipment_displacement"

        speed_px_per_sec = getattr(alert, "speed_px_per_sec", None)
        if isinstance(speed_px_per_sec, int | float) and speed_px_per_sec > 50:
            trajectory_model = getattr(alert, "trajectory_model", None)
            if trajectory_model in ("free_fall", "projectile", "projectile_drag"):
                return "projectile"

        if self._person_overlaps_alert_zone(
            alert,
            detection_result,
            frame_shape=frame_shape,
        ):
            return "person_intrusion"

        classification_label = getattr(alert, "classification_label", None)
        if (
            isinstance(classification_label, str)
            and classification_label.lower() in _ENV_LABELS
        ):
            return "environmental"

        # Only zone-scoped objects attached to this alert may influence its
        # category. The frame-global detector result can contain an object in a
        # different include zone.
        if getattr(alert, "detected_objects", None):
            return "equipment_displacement"

        return "scene_change"

    def _person_overlaps_alert_zone(
        self,
        alert,
        detection_result,
        *,
        frame_shape: tuple | None = None,
    ) -> bool:
        """Keep person categorization local to the zone that emitted the alert."""
        if not detection_result or not getattr(
            detection_result,
            "has_persons",
            False,
        ):
            return False
        persons = getattr(detection_result, "persons", None) or []
        if not persons:
            return False

        include_zones = self._zone_mask.get_include_zones()
        if not include_zones:
            return True
        alert_zone_id = str(getattr(alert, "zone_id", "") or "")
        alert_zone = next(
            (
                zone
                for zone in include_zones
                if str(zone.zone_id) == alert_zone_id
            ),
            None,
        )
        if alert_zone is None:
            return False
        if not alert_zone.polygon:
            return True

        if frame_shape is not None and len(frame_shape) >= 2:
            source_shape = (int(frame_shape[0]), int(frame_shape[1]))
        else:
            width, height = self.camera_config.resolution
            source_shape = (int(height), int(width))
        zone_mask = self._get_zone_polygon_mask(
            alert_zone.zone_id,
            alert_zone.polygon,
            source_shape,
            source_shape=source_shape,
        )
        for person in persons:
            detected_object = {
                "bbox": [
                    getattr(person, "x1", None),
                    getattr(person, "y1", None),
                    getattr(person, "x2", None),
                    getattr(person, "y2", None),
                ],
            }
            if self._object_overlaps_zone_mask(detected_object, zone_mask):
                return True
        return False

    def _predict_anomaly(
        self, frame: np.ndarray, fused_tensor: np.ndarray | None = None,
    ) -> AnomalyResult:
        """Run anomaly detection, using ensemble if active, else single detector.

        When the ensemble is active, converts EnsembleResult back to AnomalyResult
        for downstream compatibility. Falls back to single detector on any error.

        Requirements §3.2 №9: when the anomaly head is in degraded mode
        (sustained outage or load failure) we skip inference entirely and
        return a synthetic safe result so the simplex safety channel can
        still produce alerts without burning CPU on a broken model.

        Args:
            frame: 3-channel BGR visible-light frame.
            fused_tensor: Optional multi-channel tensor from ModalityFusion
                (e.g. 4-ch RGB+DoLP or 5-ch RGB+DoLP+NIR).  Currently the
                detector truncates to 3 channels (Plan A); Plan B will pass
                the full tensor to a model trained on multi-channel input.
        """
        # Short-circuit when degraded — no inference, simplex carries the load.
        # detection_failed=False prevents the runner's reload churn from
        # firing repeatedly on a model we already gave up on.
        if self._anomaly_degraded:
            threshold = getattr(self._anomaly_detector, "threshold", 0.7)
            return AnomalyResult(
                anomaly_score=0.0,
                anomaly_map=None,
                is_anomalous=False,
                threshold=float(threshold),
                detection_failed=False,
            )

        try:
            if self._ensemble_detector is not None:
                try:
                    from argus.anomaly.ensemble import DetectorEnsemble

                    ensemble: DetectorEnsemble = self._ensemble_detector  # type: ignore[assignment]
                    ensemble_result = ensemble.predict(frame)
                    self._record_anomaly_inference_ok()
                    return AnomalyResult(
                        anomaly_score=ensemble_result.anomaly_score,
                        anomaly_map=ensemble_result.anomaly_map,
                        is_anomalous=ensemble_result.is_anomalous,
                        threshold=ensemble_result.threshold,
                    )
                except Exception:
                    logger.warning(
                        "pipeline.ensemble_predict_fallback",
                        camera_id=self.camera_config.camera_id,
                        msg="Ensemble predict failed — falling back to single detector",
                    )
            result = self._anomaly_detector.predict(frame, fused_tensor=fused_tensor)
            if result.detection_failed:
                reason = "detector_result_failed"
                status = getattr(self._anomaly_detector, "status", None)
                last_error = getattr(status, "last_error", None)
                if isinstance(last_error, str) and last_error:
                    reason = f"{reason}: {last_error}"
                self._record_anomaly_inference_failure(reason)
            else:
                self._record_anomaly_inference_ok()
            return result
        except Exception as exc:
            # Single-frame failures: tolerated by detector's own try/except;
            # if execution lands here it means the call raised hard. Count
            # it; trip degraded mode after _ANOMALY_DEGRADE_THRESHOLD in a
            # row. Until then keep returning a safe failed result so the
            # rest of the pipeline can proceed.
            self._record_anomaly_inference_failure(
                f"{type(exc).__name__}: {exc}"
            )
            threshold = getattr(self._anomaly_detector, "threshold", 0.7)
            return AnomalyResult(
                anomaly_score=0.0,
                anomaly_map=None,
                is_anomalous=False,
                threshold=float(threshold),
                detection_failed=True,
            )

    # ── Anomaly head degradation helpers (requirements §3.2 №9) ──

    def _enter_anomaly_degraded(
        self, reason: str, *, manual_recovery_hint: bool = False,
    ) -> None:
        """Mark the anomaly head as degraded. Idempotent and thread-safe.

        Once entered, the pipeline stops calling the anomaly detector and
        relies on the simplex safety channel for alerts. Recovery requires
        a process restart (no automatic backbone reload here — runner
        handles its own restart attempts via DegradationStateMachine).
        """
        with self._anomaly_degraded_lock:
            if self._anomaly_degraded:
                return
            self._anomaly_degraded = True
            self._anomaly_degradation_reason = reason
            self._anomaly_degradation_started_at = time.time()
            started_at = self._anomaly_degradation_started_at
        logger.error(
            "pipeline.anomaly_degraded_entered",
            camera_id=self.camera_config.camera_id,
            reason=reason,
            simplex_active=self._simplex is not None,
            manual_recovery_required=manual_recovery_hint,
        )
        # Surface to the global degradation bar when an EventBus / dashboard
        # is wired. Failure to publish must never crash the pipeline.
        if self._event_bus is not None:
            try:
                self._event_bus.publish(
                    "pipeline.anomaly_degraded",
                    {
                        "camera_id": self.camera_config.camera_id,
                        "reason": reason,
                        "simplex_active": self._simplex is not None,
                    },
                )
            except Exception:
                logger.debug(
                    "pipeline.anomaly_degraded_publish_failed",
                    camera_id=self.camera_config.camera_id,
                    exc_info=True,
                )
        # Push directly to the dashboard WebSocket subscribers when a
        # publisher is wired. Best-effort: any failure here must NOT impact
        # the inference loop.
        if self._degradation_publisher is not None:
            try:
                self._degradation_publisher(
                    "system_degradation",
                    {
                        "type": "entered",
                        "component": "anomaly",
                        "camera_id": self.camera_config.camera_id,
                        "reason": reason,
                        "started_at": started_at,
                        "simplex_active": self._simplex is not None,
                    },
                )
            except Exception as exc:
                logger.warning(
                    "pipeline.degradation_broadcast_failed",
                    camera_id=self.camera_config.camera_id,
                    error=str(exc),
                )
        # Mirror to the unified error channel so the system_errors banner
        # shows the same event alongside other backend failures. emit() is
        # internally swallowed; this can never crash the inference loop.
        get_error_channel().emit(
            severity=SEVERITY_WARNING,
            source="pipeline",
            code="anomaly_degraded",
            message=f"摄像头 {self.camera_config.camera_id} 进入异常降级",
            context={
                "camera_id": self.camera_config.camera_id,
                "reason": reason,
                "simplex_active": self._simplex is not None,
                "started_at": started_at,
                "manual_recovery_required": manual_recovery_hint,
            },
        )

    def _record_anomaly_inference_ok(self) -> None:
        """Reset the consecutive failure counter after a successful inference."""
        if self._anomaly_failure_count != 0:
            with self._anomaly_degraded_lock:
                self._anomaly_failure_count = 0

    def _record_anomaly_inference_failure(self, reason: str) -> None:
        """Count a hard failure; trip degraded mode at the threshold."""
        with self._anomaly_degraded_lock:
            self._anomaly_failure_count += 1
            count = self._anomaly_failure_count
            already_degraded = self._anomaly_degraded
        logger.warning(
            "pipeline.anomaly_inference_error",
            camera_id=self.camera_config.camera_id,
            reason=reason,
            consecutive=count,
            threshold=self._ANOMALY_DEGRADE_THRESHOLD,
        )
        if not already_degraded and count >= self._ANOMALY_DEGRADE_THRESHOLD:
            self._enter_anomaly_degraded(
                f"sustained_inference_errors: {count} consecutive — last={reason}",
                manual_recovery_hint=True,
            )

    def is_anomaly_degraded(self) -> bool:
        """Whether the anomaly head is in simplex-only fallback mode."""
        with self._anomaly_degraded_lock:
            return self._anomaly_degraded

    def get_anomaly_degradation_reason(self) -> str | None:
        """Reason for current anomaly degradation, or None if nominal."""
        with self._anomaly_degraded_lock:
            return self._anomaly_degradation_reason

    def get_anomaly_degradation_started_at(self) -> float | None:
        """Unix timestamp when the anomaly head entered degraded mode."""
        with self._anomaly_degraded_lock:
            return self._anomaly_degradation_started_at

    def set_degradation_publisher(
        self, publisher: Callable[[str, dict], None] | None,
    ) -> None:
        """Inject (or replace) the WebSocket broadcast callable.

        Used by ``__main__.py`` to wire ``ws_manager.broadcast`` into
        pipelines that were created before the dashboard initialised. The
        publisher is only invoked on state transitions, so calling this
        after the initial state has been set still surfaces future events.
        """
        self._degradation_publisher = publisher

    def initialize(self) -> bool:
        """Initialize all pipeline components. Returns True on success."""
        if not self._camera.connect():
            return False

        # For network streams (RTSP/GigE), start a dedicated capture thread
        # to drain the TCP buffer continuously.  read_latest() then always
        # returns the most recent frame, preventing slow-motion playback.
        # File and USB sources don't need this (no network buffer to drain).
        if self._camera.protocol in ("rtsp", "gige"):
            self._camera.start_capture_thread()

        # Requirements §3.2 №9: anomaly head load failure must not crash the
        # camera. Enter simplex-only degraded mode and continue, so the
        # safety channel can still raise alerts. The runner converts this
        # state into BACKBONE_FAILED via get_detector_status() if needed.
        preload_artifact_hash: str | None = None
        if self._anomaly_model_path is not None:
            try:
                preload_artifact_hash = _model_artifact_sha256(
                    self._anomaly_model_path
                )
            except (OSError, ValueError) as exc:
                logger.error(
                    "pipeline.model_fingerprint_before_load_failed",
                    camera_id=self.camera_config.camera_id,
                    model_path=str(self._anomaly_model_path),
                    error=str(exc),
                )
        # Never expose a constructor-time or previous-engine identity while the
        # detector is loading a possibly changed artifact.
        self._model_artifact_sha256 = None
        load_succeeded = False
        try:
            # ``load()`` deliberately returns False for non-model fallbacks
            # such as SSIM.  Only a literal True proves that the runtime
            # engine loaded the artifact fingerprinted on both sides of this
            # call; truthy mocks/objects and fallback modes must not authorize
            # an operator-approved scene profile.
            load_succeeded = self._anomaly_detector.load() is True
        except Exception as exc:
            self._enter_anomaly_degraded(
                f"load_failed: {type(exc).__name__}: {exc}",
                manual_recovery_hint=True,
            )

        postload_artifact_hash: str | None = None
        if self._anomaly_model_path is not None:
            try:
                postload_artifact_hash = _model_artifact_sha256(
                    self._anomaly_model_path
                )
            except (OSError, ValueError) as exc:
                logger.error(
                    "pipeline.model_fingerprint_after_load_failed",
                    camera_id=self.camera_config.camera_id,
                    model_path=str(self._anomaly_model_path),
                    error=str(exc),
                )
        if (
            load_succeeded
            and preload_artifact_hash is not None
            and preload_artifact_hash == postload_artifact_hash
        ):
            self._model_artifact_sha256 = postload_artifact_hash
        else:
            self._model_artifact_sha256 = None
            self._reset_candidate_model_evidence_state()
            if self._anomaly_model_path is not None:
                logger.warning(
                    "pipeline.model_fingerprint_unstable",
                    camera_id=self.camera_config.camera_id,
                    model_path=str(self._anomaly_model_path),
                    load_succeeded=load_succeeded,
                    before=preload_artifact_hash,
                    after=postload_artifact_hash,
                )

        # Constructor-time profile resolution is intentionally untrusted: the
        # detector had not loaded yet, so no runtime artifact identity existed.
        # Only the stable before/after fingerprint above may release the
        # operator-approved reference.  Load failure or any mismatch clears it.
        with self._scene_processing_lock:
            self._revalidate_scene_guard_reference()

        # Calibrate raw scores if PostProcessor MinMax is broken — skipped
        # in degraded mode because the detector is offline.
        if not self._anomaly_degraded:
            try:
                if self._anomaly_detector.get_status().minmax_broken:
                    baseline_dir = self._configured_baseline_dir(self.camera_config.camera_id)
                    self._anomaly_detector.calibrate_raw_scores(baseline_dir)
            except Exception as exc:
                # Calibration failure is not fatal but signals a sick detector.
                logger.warning(
                    "pipeline.calibrate_failed",
                    camera_id=self.camera_config.camera_id,
                    error=str(exc),
                )

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

        ensemble_active = self._ensemble_detector is not None
        # `is_loaded` may be unavailable on the detector when load() crashed
        # before assigning the attribute — guard with getattr.
        anomaly_model_loaded = bool(getattr(self._anomaly_detector, "is_loaded", False))
        logger.info(
            "pipeline.initialized",
            camera_id=self.camera_config.camera_id,
            anomaly_model_loaded=anomaly_model_loaded,
            anomaly_degraded=self._anomaly_degraded,
            ensemble_active=ensemble_active,
            ensemble_models=(
                self._ensemble_detector.model_count
                if ensemble_active else 0
            ),
            zones=len(self.camera_config.zones),
            heartbeat_interval=self._heartbeat_seconds,
        )
        return True

    def _identity_for_frame_data(
        self,
        frame_data: FrameData,
        pipeline_ingest_ts: float,
    ) -> FrameIdentity:
        identity = frame_data.identity
        if identity is None:
            identity = build_frame_identity(
                camera_id=frame_data.camera_id,
                source_declared=self._frame_identity_declared_source,
                source_runtime=self._frame_identity_runtime_source,
                stream_id=self._frame_identity_stream_id,
                frame_index=frame_data.frame_number,
                capture_monotonic_ts=frame_data.timestamp,
            )
        else:
            identity = identity.with_updates(pipeline_ingest_ts=pipeline_ingest_ts)
        if identity.pipeline_ingest_ts is None:
            identity = identity.with_updates(pipeline_ingest_ts=pipeline_ingest_ts)
        return identity

    def _set_latest_frame_identity(self, identity: FrameIdentity) -> None:
        with self._latest_frame_identity_lock:
            self._latest_frame_identity = identity

    def process_frame(self, frame_data: FrameData) -> Alert | None:
        """Run the detection pipeline on a single frame."""
        if self._auxiliary_shutdown:
            self._set_frame_outcome(FrameOutcomeKind.BLOCKED, "pipeline is stopped")
            return None
        start = time.monotonic()
        pipeline_ingest_ts = time.time()
        self.stats.frames_captured += 1
        self.stats.update_fps()
        self._frame_counter += 1
        self._last_zone_alerts = []
        self._last_emitted_alerts = []
        self._last_grade_result = None
        self._set_frame_outcome(FrameOutcomeKind.NORMAL)
        frame = frame_data.frame
        frame_data.identity = self._identity_for_frame_data(
            frame_data,
            pipeline_ingest_ts,
        )
        self._flush_expired_post_captures()

        # NIR strobe: if this frame was captured under NIR illumination,
        # store it for modality fusion and skip detection (NIR frames are
        # auxiliary — detection runs on visible-light frames only).
        if frame_data.is_nir:
            self._latest_nir_frame = frame
            self._set_frame_outcome(
                FrameOutcomeKind.BLOCKED,
                "auxiliary NIR frame is not eligible for alert inference",
            )
            return None

        # This counter advances for every visible frame offered to the pipeline,
        # independent of capture frame-number gaps or offline sample_every.  It
        # lets the temporal gate distinguish an actual early return from a
        # skipped source frame that was never part of the analysis sequence.
        self._candidate_model_frame_sequence += 1

        # DET-008: Build diagnostics record for this frame
        diag = FrameDiagnostics(
            frame_number=frame_data.frame_number,
            timestamp=time.time(),
            camera_id=frame_data.camera_id,
        )
        diag.candidate_model_observation_number = self._candidate_model_frame_sequence

        # Validate frame before processing
        if frame is None or frame.size == 0 or len(frame.shape) != 3:
            logger.warning(
                "pipeline.invalid_frame",
                camera_id=frame_data.camera_id,
                frame_number=frame_data.frame_number,
                shape=getattr(frame, "shape", None),
            )
            self._set_frame_outcome(FrameOutcomeKind.FAILED, "invalid camera frame")
            return None

        try:
            with self._scene_processing_lock:
                if self._auxiliary_shutdown:
                    self._set_frame_outcome(FrameOutcomeKind.BLOCKED, "pipeline is stopped")
                    return None
                analyzed_before = int(self.stats.frames_analyzed)
                alert = self._process_frame_inner(frame_data, frame, start, diag)
                if self._last_frame_outcome_kind is FrameOutcomeKind.NORMAL:
                    if alert is not None:
                        self._set_frame_outcome(FrameOutcomeKind.ALERT)
                    elif (
                        int(self.stats.frames_analyzed) > analyzed_before
                        and self._last_detection_failed
                    ):
                        self._set_frame_outcome(
                            FrameOutcomeKind.FAILED,
                            "anomaly detection failed",
                        )
                return alert
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
            self._set_frame_outcome(
                FrameOutcomeKind.FAILED,
                f"{type(e).__name__}: {e}",
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
            self._reset_fast_motion_after_detection_mode_entry(
                PipelineMode.LEARNING, PipelineMode.ACTIVE,
            )
            self._reset_candidate_model_evidence_state()
            self._auto_learning_complete = True
            self._learning_start_time = None
            logger.info(
                "pipeline.learning_mode_complete",
                camera_id=frame_data.camera_id,
                msg="Learning mode expired, switching to ACTIVE",
            )

        # COLLECTION/TRAINING watchdog: recover if a long-running task crashed
        # before resetting the mode (otherwise pipeline would skip detection forever).
        if (
            self._mode_session_start_time is not None
            and self.mode in _DETECTION_SKIPPED_MODES
            and (time.monotonic() - self._mode_session_start_time) >= self._mode_session_max_duration
        ):
            with self._mode_lock:
                stuck_mode = self._mode
                self._mode = PipelineMode.ACTIVE
                self._mode_session_start_time = None
            self._reset_fast_motion_after_detection_mode_entry(
                stuck_mode, PipelineMode.ACTIVE,
            )
            logger.warning(
                "pipeline.mode_session_timeout_recovered",
                camera_id=frame_data.camera_id,
                stuck_mode=stuck_mode.value,
                max_duration_seconds=self._mode_session_max_duration,
            )

        current_mode = self.mode
        diag.pipeline_mode = current_mode.value
        # Save raw frame reference before zone masking — apply() returns a new
        # array (cv2.bitwise_and), so the original frame is not mutated.
        with self._latest_raw_frame_lock:
            self._latest_raw_frame = frame

        # Stage -0.5: prove that the camera still sees the approved scene.
        # This must run on the untouched camera frame before zone masking,
        # deglare, CLAHE, or the adaptive micro-vibration aligner change its
        # geometry/photometry.  ALIGNABLE frames are warped into the approved
        # reference coordinates before the old model is allowed to run.
        scene_input = frame
        frame, scene_usable = self._apply_scene_guard(frame, frame_data, diag)
        if not scene_usable:
            with self._latest_frame_lock:
                self._latest_frame = scene_input
            if frame_data.identity is not None:
                self._set_latest_frame_identity(frame_data.identity)
            status = self.get_scene_guard_status()
            diag.final_decision = "skipped"
            diag.add_candidate(
                candidate_type="scene_identity",
                decision="blocked",
                stage="scene_guard",
                evidence=status,
                suppression_reason=f"scene_guard:{status.get('state', 'blocked')}",
            )
            diag.total_duration_ms = (time.monotonic() - start) * 1000
            self._diagnostics.append(diag)
            self._set_frame_outcome(
                FrameOutcomeKind.BLOCKED,
                str(status.get("reason") or "scene guard blocked inference"),
            )
            return None
        # Keep an unmasked, non-deglared canonical frame for the static
        # candidate confirmer.  It is compared with the approved raw reference
        # in the same coordinate and photometric domain.
        scene_confirmation_frame = frame
        scene_model_input_variant = (
            "scene_homography_aligned"
            if diag.scene_guard_state == "alignable"
            and self.camera_config.scene_guard.allow_alignable
            else "scene_stable"
        )
        model_preprocessing_valid = True
        imaging_model_input_variant = "imaging_disabled"
        alignment_model_input_variant = "micro_alignment_disabled"

        # Stage 0: Apply zone mask
        t0 = time.monotonic()
        frame = self._zone_mask.apply(frame)
        diag.stages.append(StageResult(
            stage_name="zone_mask", duration_ms=(time.monotonic() - t0) * 1000
        ))

        # Stage 0.5: Multi-modal imaging preprocessing (if enabled)
        fused_tensor: np.ndarray | None = None
        if self._imaging_processor is not None:
            try:
                t_img = time.monotonic()
                pol_result = self._imaging_processor.process(frame)
                frame = pol_result.deglared  # Use reflection-removed image
                imaging_model_input_variant = "imaging_deglared"

                # Fuse modalities when extra channels are configured
                if self._modality_fusion is not None:
                    nir = self._latest_nir_frame  # May be None until Phase 3
                    fused = self._modality_fusion.fuse(
                        visible=frame,
                        dolp=pol_result.dolp,
                        nir=nir,
                    )
                    fused_tensor = fused.tensor
                    logger.debug(
                        "pipeline.modality_fusion",
                        channels=fused.channels,
                        has_dolp=True,
                        has_nir=nir is not None,
                    )

                diag.stages.append(StageResult(
                    stage_name="imaging_deglare",
                    duration_ms=(time.monotonic() - t_img) * 1000,
                ))
            except Exception as e:
                imaging_model_input_variant = "imaging_failed"
                model_preprocessing_valid = False
                logger.debug("pipeline.imaging_preprocessing_error", error=str(e))

        # Stage 0.75: Phase-correlation alignment
        if self._aligner is not None:
            t_align = time.monotonic()
            try:
                frame, (align_dx, align_dy) = self._aligner.align(frame)
                alignment_model_input_variant = "micro_aligned"
                if (
                    abs(align_dx) <= self.camera_config.alignment.max_shift_px
                    and abs(align_dy) <= self.camera_config.alignment.max_shift_px
                    and (align_dx != 0.0 or align_dy != 0.0)
                ):
                    height, width = scene_confirmation_frame.shape[:2]
                    translation = np.float64(
                        [[1.0, 0.0, -align_dx], [0.0, 1.0, -align_dy]]
                    )
                    scene_confirmation_frame = cv2.warpAffine(
                        scene_confirmation_frame,
                        translation,
                        (width, height),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                diag.alignment_shift_px = (align_dx, align_dy)
                if self._m_alignment_shift_dx is not None:
                    self._m_alignment_shift_dx.observe(abs(align_dx))
                    self._m_alignment_shift_dy.observe(abs(align_dy))
                diag.stages.append(StageResult(
                    stage_name="alignment",
                    duration_ms=(time.monotonic() - t_align) * 1000,
                    metadata={
                        "dx": round(align_dx, 3),
                        "dy": round(align_dy, 3),
                    },
                ))
            except Exception as e:
                alignment_model_input_variant = "micro_alignment_failed"
                model_preprocessing_valid = False
                logger.debug(
                    "pipeline.alignment_error",
                    camera_id=frame_data.camera_id,
                    error=str(e),
                )

        # Save latest frame reference — get_latest_frame() copies on read
        with self._latest_frame_lock:
            self._latest_frame = frame
        if frame_data.identity is not None:
            self._set_latest_frame_identity(frame_data.identity)

        # COLLECTION/TRAINING modes skip detection stages entirely (痛点 1/3).
        # raw_frame and latest_frame are still updated above so the capture
        # thread / video wall keep working; YOLO + Anomalib + MOG2 consumption
        # are bypassed to free CPU/GPU and avoid contaminating alert history.
        if current_mode in _DETECTION_SKIPPED_MODES:
            if self._fast_motion is not None:
                self._fast_motion.reset()
            diag.stages.append(StageResult(
                stage_name="mode_skip",
                duration_ms=0.0,
                skipped=True,
                skip_reason=current_mode.value,
            ))
            diag.final_decision = "skipped"
            diag.add_candidate(
                candidate_type="frame",
                decision="skipped",
                stage="mode_skip",
                evidence={"pipeline_mode": current_mode.value},
                suppression_reason=f"pipeline_mode:{current_mode.value}",
            )
            diag.total_duration_ms = (time.monotonic() - start) * 1000
            self._diagnostics.append(diag)
            self._set_frame_outcome(
                FrameOutcomeKind.BLOCKED,
                f"pipeline mode {current_mode.value} suppresses detection",
            )
            return None

        # Single full-resolution BGR->GRAY conversion shared by brightness
        # detection, fast-motion gating, and MOG2 stabilization. Valid only for this frame.
        frame_gray: np.ndarray | None = None
        need_gray = self._low_light_enabled or getattr(
            self._prefilter, "enable_stabilization", False,
        )
        if need_gray:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._frame_gray = frame_gray

        # Low-light detection: bypass MOG2 when scene is dark
        is_low_light = False
        brightness_jump = False
        if self._low_light_enabled:
            gray = frame_gray if frame_gray is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(gray.mean())
            is_low_light = mean_brightness < self._low_light_threshold

            # Brightness jump detection: freeze MOG2 on sudden illumination changes
            if self._prev_brightness is not None:
                delta = abs(mean_brightness - self._prev_brightness)
                if delta >= self._brightness_jump_threshold:
                    brightness_jump = True
                    self._fast_motion_suppress_until = max(
                        self._fast_motion_suppress_until,
                        start + self._fast_motion_recovery_seconds,
                    )
                    logger.debug(
                        "pipeline.brightness_jump",
                        camera_id=frame_data.camera_id,
                        delta=round(delta, 1),
                        prev=round(self._prev_brightness, 1),
                        current=round(mean_brightness, 1),
                    )
            self._prev_brightness = mean_brightness

            low_light_entered = is_low_light and not self._was_low_light
            low_light_exited = not is_low_light and self._was_low_light
            if low_light_entered:
                self._fast_motion_suppress_until = max(
                    self._fast_motion_suppress_until,
                    start + self._fast_motion_recovery_seconds,
                )
                logger.warning(
                    "pipeline.low_light_entered",
                    camera_id=frame_data.camera_id,
                    brightness=round(mean_brightness, 1),
                )
            elif low_light_exited:
                self._fast_motion_suppress_until = max(
                    self._fast_motion_suppress_until,
                    start + self._fast_motion_recovery_seconds,
                )
                logger.info(
                    "pipeline.low_light_exited",
                    camera_id=frame_data.camera_id,
                    brightness=round(mean_brightness, 1),
                )
            self._was_low_light = is_low_light

            # CLAHE low-light enhancement: equalize frame before detection stages
            if is_low_light and self._clahe is not None:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                if self._clahe_use_cuda:
                    self._clahe_gpu_mat.upload(lab[:, :, 0])
                    gpu_result = self._clahe.apply(self._clahe_gpu_mat, self._clahe_cuda_stream)
                    self._clahe_cuda_stream.waitForCompletion()
                    lab[:, :, 0] = gpu_result.download()
                else:
                    lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                # CLAHE altered frame pixels — the cached gray no longer matches.
                frame_gray = None
                self._frame_gray = None
        diag.low_light = is_low_light

        fast_motion_observed = False
        if self._fast_motion is not None and current_mode in (
            PipelineMode.ACTIVE,
            PipelineMode.MAINTENANCE,
        ):
            t_fast = time.monotonic()
            exposure_recovering = start < self._fast_motion_suppress_until
            if is_low_light or brightness_jump or exposure_recovering:
                self._fast_motion.reset()
                if is_low_light:
                    skip_reason = "low_light"
                elif brightness_jump:
                    skip_reason = "brightness_jump"
                else:
                    skip_reason = "low_light_recovery"
                diag.stages.append(StageResult(
                    stage_name="fast_motion",
                    duration_ms=(time.monotonic() - t_fast) * 1000,
                    metadata={
                        "low_light": is_low_light,
                        "brightness_jump": brightness_jump,
                        "recovery_remaining_seconds": max(
                            0.0,
                            round(self._fast_motion_suppress_until - start, 3),
                        ),
                    },
                    skipped=True,
                    skip_reason=skip_reason,
                ))
                diag.add_candidate(
                    candidate_type="fast_motion",
                    decision="skipped",
                    stage="fast_motion",
                    evidence={
                        "low_light": is_low_light,
                        "brightness_jump": brightness_jump,
                        "exposure_recovering": exposure_recovering,
                    },
                    suppression_reason=skip_reason,
                )
            else:
                try:
                    fast_result = self._fast_motion.process(frame, timestamp=time.time())
                    fast_metadata = {
                        "candidates": len(fast_result.candidates),
                        "max_confidence": round(fast_result.max_confidence, 3),
                        "motion_fraction": round(fast_result.motion_fraction, 6),
                    }
                    fast_motion_observed = True
                    self._auxiliary_errors.pop("fast_motion", None)
                    diag.stages.append(StageResult(
                        stage_name="fast_motion",
                        duration_ms=(time.monotonic() - t_fast) * 1000,
                        metadata=fast_metadata,
                        skipped=not fast_result.has_detection,
                        skip_reason=fast_result.suppression_reason or "",
                    ))
                except Exception as exc:
                    # An unavailable auxiliary head is not a negative
                    # observation, and must not disable independent hazards.
                    fast_motion_observed = False
                    self._record_auxiliary_failure("fast_motion", exc, diag, t_fast)
                if fast_motion_observed and fast_result.has_detection:
                    # The handler may already have emitted/persisted an alert.
                    # Never turn its side-effect failure into a second decision
                    # through the ordinary path for this same frame.
                    return self._handle_fast_motion_detection(
                        frame_data=frame_data, frame=frame, start=start,
                        diag=diag, fast_result=fast_result,
                    )
        else:
            exposure_recovering = start < self._fast_motion_suppress_until

        persistent_change_result = None
        persistent_change_objects: list[dict] = []
        if self._persistent_change is not None and current_mode in (
            PipelineMode.ACTIVE,
            PipelineMode.MAINTENANCE,
        ):
            t_pc = time.monotonic()
            if is_low_light or brightness_jump or exposure_recovering:
                self._persistent_change.reset_tracks()
                if is_low_light:
                    pc_skip_reason = "low_light"
                elif brightness_jump:
                    pc_skip_reason = "brightness_jump"
                else:
                    pc_skip_reason = "low_light_recovery"
                diag.stages.append(StageResult(
                    stage_name="persistent_change",
                    duration_ms=(time.monotonic() - t_pc) * 1000,
                    skipped=True,
                    skip_reason=pc_skip_reason,
                    metadata={
                        "low_light": is_low_light,
                        "brightness_jump": brightness_jump,
                        "exposure_recovering": exposure_recovering,
                    },
                ))
                diag.add_candidate(
                    candidate_type="persistent_change",
                    decision="skipped",
                    stage="persistent_change",
                    evidence={
                        "low_light": is_low_light,
                        "brightness_jump": brightness_jump,
                        "exposure_recovering": exposure_recovering,
                    },
                    suppression_reason=pc_skip_reason,
                )
            else:
                try:
                    persistent_change_result = self._persistent_change.process(
                        frame,
                        timestamp=time.time(),
                    )
                    pc_evidence = persistent_change_result.evidence()
                    persistent_change_objects = [
                        self._independent_visual_payload(candidate)
                        for candidate in persistent_change_result.candidates
                    ]
                    diag.stages.append(StageResult(
                        stage_name="persistent_change",
                        duration_ms=(time.monotonic() - t_pc) * 1000,
                        skipped=not persistent_change_result.has_detection,
                        skip_reason=persistent_change_result.suppression_reason or "",
                        metadata={
                            "ready": persistent_change_result.ready,
                            "has_detection": persistent_change_result.has_detection,
                            "score": round(persistent_change_result.score, 4),
                            "candidate_count": len(persistent_change_result.candidates),
                            "active_track_count": len(
                                persistent_change_result.active_tracks
                            ),
                            "changed_fraction": round(
                                persistent_change_result.changed_fraction,
                                6,
                            ),
                        },
                    ))
                    if persistent_change_result.has_detection:
                        diag.add_candidate(
                            candidate_type="persistent_change",
                            decision="candidate",
                            stage="persistent_change",
                            evidence=pc_evidence,
                        )
                    elif persistent_change_result.suppression_reason:
                        diag.add_candidate(
                            candidate_type="persistent_change",
                            decision=(
                                "pending"
                                if persistent_change_result.ready
                                else "skipped"
                            ),
                            stage="persistent_change",
                            evidence=pc_evidence,
                            suppression_reason=(
                                persistent_change_result.suppression_reason
                            ),
                        )
                except Exception as e:
                    persistent_change_result = None
                    persistent_change_objects = []
                    logger.warning(
                        "pipeline.persistent_change_failed",
                        camera_id=frame_data.camera_id,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    diag.stages.append(StageResult(
                        stage_name="persistent_change",
                        duration_ms=(time.monotonic() - t_pc) * 1000,
                        skipped=True,
                        skip_reason="scorer_error",
                    ))
                    diag.add_candidate(
                        candidate_type="persistent_change",
                        decision="skipped",
                        stage="persistent_change",
                        evidence={"error": type(e).__name__},
                        suppression_reason="scorer_error",
                    )

        water_leak_result = None
        water_leak_objects: list[dict] = []
        if self._water_leak is not None and current_mode in (
            PipelineMode.ACTIVE,
            PipelineMode.MAINTENANCE,
        ):
            t_water = time.monotonic()
            if is_low_light or brightness_jump or exposure_recovering:
                self._water_leak.reset_tracks()
                if is_low_light:
                    water_skip_reason = "low_light"
                elif brightness_jump:
                    water_skip_reason = "brightness_jump"
                else:
                    water_skip_reason = "low_light_recovery"
                diag.stages.append(StageResult(
                    stage_name="water_leak",
                    duration_ms=(time.monotonic() - t_water) * 1000,
                    skipped=True,
                    skip_reason=water_skip_reason,
                    metadata={
                        "low_light": is_low_light,
                        "brightness_jump": brightness_jump,
                        "exposure_recovering": exposure_recovering,
                    },
                ))
                diag.add_candidate(
                    candidate_type="water_leak",
                    decision="skipped",
                    stage="water_leak",
                    evidence={
                        "low_light": is_low_light,
                        "brightness_jump": brightness_jump,
                        "exposure_recovering": exposure_recovering,
                    },
                    suppression_reason=water_skip_reason,
                )
            else:
                try:
                    water_leak_result = self._water_leak.process(
                        frame,
                        timestamp=time.time(),
                    )
                    water_evidence = water_leak_result.evidence()
                    water_leak_objects = [
                        self._independent_visual_payload(candidate)
                        for candidate in water_leak_result.candidates
                    ]
                    diag.stages.append(StageResult(
                        stage_name="water_leak",
                        duration_ms=(time.monotonic() - t_water) * 1000,
                        skipped=not water_leak_result.has_detection,
                        skip_reason=water_leak_result.suppression_reason or "",
                        metadata={
                            "ready": water_leak_result.ready,
                            "has_detection": water_leak_result.has_detection,
                            "score": round(water_leak_result.score, 4),
                            "candidate_count": len(water_leak_result.candidates),
                            "active_track_count": len(water_leak_result.active_tracks),
                            "changed_fraction": round(
                                water_leak_result.changed_fraction,
                                6,
                            ),
                        },
                    ))
                    if water_leak_result.has_detection:
                        diag.add_candidate(
                            candidate_type="water_leak",
                            decision="candidate",
                            stage="water_leak",
                            evidence=water_evidence,
                        )
                    elif water_leak_result.suppression_reason:
                        diag.add_candidate(
                            candidate_type="water_leak",
                            decision=(
                                "pending"
                                if water_leak_result.ready
                                else "skipped"
                            ),
                            stage="water_leak",
                            evidence=water_evidence,
                            suppression_reason=water_leak_result.suppression_reason,
                        )
                except Exception as e:
                    water_leak_result = None
                    water_leak_objects = []
                    logger.warning(
                        "pipeline.water_leak_failed",
                        camera_id=frame_data.camera_id,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    diag.stages.append(StageResult(
                        stage_name="water_leak",
                        duration_ms=(time.monotonic() - t_water) * 1000,
                        skipped=True,
                        skip_reason="scorer_error",
                    ))
                    diag.add_candidate(
                        candidate_type="water_leak",
                        decision="skipped",
                        stage="water_leak",
                        evidence={"error": type(e).__name__},
                        suppression_reason="scorer_error",
                    )

        plume_result = None
        plume_objects: list[dict] = []
        if self._plume is not None and current_mode in (
            PipelineMode.ACTIVE,
            PipelineMode.MAINTENANCE,
        ):
            t_plume = time.monotonic()
            if is_low_light or brightness_jump or exposure_recovering:
                self._plume.reset_tracks()
                if is_low_light:
                    plume_skip_reason = "low_light"
                elif brightness_jump:
                    plume_skip_reason = "brightness_jump"
                else:
                    plume_skip_reason = "low_light_recovery"
                diag.stages.append(StageResult(
                    stage_name="plume",
                    duration_ms=(time.monotonic() - t_plume) * 1000,
                    skipped=True,
                    skip_reason=plume_skip_reason,
                    metadata={
                        "low_light": is_low_light,
                        "brightness_jump": brightness_jump,
                        "exposure_recovering": exposure_recovering,
                    },
                ))
                diag.add_candidate(
                    candidate_type="plume",
                    decision="skipped",
                    stage="plume",
                    evidence={
                        "low_light": is_low_light,
                        "brightness_jump": brightness_jump,
                        "exposure_recovering": exposure_recovering,
                    },
                    suppression_reason=plume_skip_reason,
                )
            else:
                try:
                    plume_result = self._plume.process(
                        frame,
                        timestamp=time.time(),
                    )
                    plume_evidence = plume_result.evidence()
                    plume_objects = [
                        self._independent_visual_payload(candidate)
                        for candidate in plume_result.candidates
                    ]
                    diag.stages.append(StageResult(
                        stage_name="plume",
                        duration_ms=(time.monotonic() - t_plume) * 1000,
                        skipped=not plume_result.has_detection,
                        skip_reason=plume_result.suppression_reason or "",
                        metadata={
                            "ready": plume_result.ready,
                            "has_detection": plume_result.has_detection,
                            "score": round(plume_result.score, 4),
                            "candidate_count": len(plume_result.candidates),
                            "active_track_count": len(plume_result.active_tracks),
                            "changed_fraction": round(
                                plume_result.changed_fraction,
                                6,
                            ),
                            "suppressed_region_count": (
                                plume_result.suppressed_region_count
                            ),
                        },
                    ))
                    if plume_result.has_detection:
                        diag.add_candidate(
                            candidate_type="plume",
                            decision="candidate",
                            stage="plume",
                            evidence=plume_evidence,
                        )
                    elif plume_result.suppression_reason:
                        diag.add_candidate(
                            candidate_type="plume",
                            decision=(
                                "pending"
                                if plume_result.ready
                                else "skipped"
                            ),
                            stage="plume",
                            evidence=plume_evidence,
                            suppression_reason=plume_result.suppression_reason,
                        )
                except Exception as e:
                    plume_result = None
                    plume_objects = []
                    logger.warning(
                        "pipeline.plume_failed",
                        camera_id=frame_data.camera_id,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    diag.stages.append(StageResult(
                        stage_name="plume",
                        duration_ms=(time.monotonic() - t_plume) * 1000,
                        skipped=True,
                        skip_reason="scorer_error",
                    ))
                    diag.add_candidate(
                        candidate_type="plume",
                        decision="skipped",
                        stage="plume",
                        evidence={"error": type(e).__name__},
                        suppression_reason="scorer_error",
                    )

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

        calibrating_ssim = False
        ssim_fallback_active = False
        try:
            detector_status = self._anomaly_detector.get_status()
            ssim_fallback_active = (
                getattr(detector_status, "mode", None) == "ssim_fallback"
                and not bool(getattr(detector_status, "model_loaded", False))
            )
            calibrating_ssim = (
                ssim_fallback_active
                and not bool(getattr(detector_status, "ssim_calibrated", True))
            )
        except Exception:
            logger.debug(
                "pipeline.detector_status_unavailable",
                camera_id=frame_data.camera_id,
                exc_info=True,
            )

        # These independent heads have already completed (or failed/skipped)
        # before MOG2 and SSIM can return early.  A configured head without a
        # result is not negative evidence: only channels fully observed on this
        # frame may decay an existing CUSUM incident.
        early_observed_alert_channel_types: set[str] = set()
        if fast_motion_observed:
            early_observed_alert_channel_types.add(
                DetectionType.PROJECTILE.value
            )
        # Simplex runs after the heavyweight inference stage.  Therefore this
        # early branch can only have fully observed the composite persistent
        # channel when Simplex is disabled and the dedicated head returned a
        # result on this frame.
        persistent_heads_observed_early = bool(
            self._simplex is None and persistent_change_result is not None
        )
        if persistent_heads_observed_early:
            early_observed_alert_channel_types.add(
                DetectionType.PERSISTENT_CHANGE.value
            )
        if water_leak_result is not None:
            early_observed_alert_channel_types.add(
                DetectionType.WATER_LEAK.value
            )
        if plume_result is not None:
            early_observed_alert_channel_types.add(DetectionType.PLUME.value)

        # CRIT-01/02: Read lock state atomically for prefilter decision
        with self._lock_state_lock:
            is_locked = self._locked
        skip_prefilter = is_heartbeat or is_locked or is_low_light or calibrating_ssim

        # DET-006: MAINTENANCE mode freezes MOG2 learning on all frames
        # Also freeze on brightness jumps to prevent background model corruption
        freeze_mog2 = current_mode == PipelineMode.MAINTENANCE or brightness_jump

        mog2_skipped = False
        if not skip_prefilter:
            if freeze_mog2:
                prefilter_result: PreFilterResult = self._prefilter.process(
                    frame, learning_rate_override=0.0, gray_frame=frame_gray,
                )
            else:
                prefilter_result = self._prefilter.process(frame, gray_frame=frame_gray)
            if not prefilter_result.has_change:
                persistent_override = bool(
                    persistent_change_result is not None
                    and persistent_change_result.has_detection
                )
                water_leak_override = bool(
                    water_leak_result is not None
                    and water_leak_result.has_detection
                )
                plume_override = bool(
                    plume_result is not None
                    and plume_result.has_detection
                )
                if persistent_override or water_leak_override or plume_override:
                    diag.stages.append(StageResult(
                        stage_name="mog2",
                        duration_ms=(time.monotonic() - t1) * 1000,
                        skipped=False,
                        metadata={
                            "has_change": False,
                            "change_ratio": prefilter_result.change_ratio,
                            "persistent_change_override": persistent_override,
                            "persistent_change_score": (
                                round(persistent_change_result.score, 4)
                                if persistent_change_result is not None
                                else None
                            ),
                            "water_leak_override": water_leak_override,
                            "water_leak_score": (
                                round(water_leak_result.score, 4)
                                if water_leak_result is not None
                                else None
                            ),
                            "plume_override": plume_override,
                            "plume_score": (
                                round(plume_result.score, 4)
                                if plume_result is not None
                                else None
                            ),
                        },
                    ))
                else:
                    self.stats.frames_skipped_no_change += 1
                    mog2_skipped = True
                    self._m_skipped_mog2.inc()
                    diag.stages.append(StageResult(
                        stage_name="mog2",
                        duration_ms=(time.monotonic() - t1) * 1000,
                        skipped=True,
                        skip_reason="no_change",
                    ))
                    diag.final_decision = "skipped"
                    diag.add_candidate(
                        candidate_type="frame_change",
                        decision="skipped",
                        stage="mog2",
                        evidence={
                            "change_ratio": prefilter_result.change_ratio,
                            "heartbeat": is_heartbeat,
                            "locked": is_locked,
                        },
                        suppression_reason="no_change",
                    )
                    diag.total_duration_ms = (time.monotonic() - start) * 1000
                    self._diagnostics.append(diag)
                    self._decay_all_unobserved_alert_incidents(
                        frame_data.camera_id,
                        decay_detection_types=(
                            early_observed_alert_channel_types
                        ),
                    )
                    return None
        else:
            if is_locked:
                self._prefilter.process(frame, learning_rate_override=0.0, gray_frame=frame_gray)
            else:
                self._prefilter.process(frame, gray_frame=frame_gray)
            if is_heartbeat:
                self.stats.frames_heartbeat += 1

        if calibrating_ssim and (is_low_light or brightness_jump or exposure_recovering):
            diag.stages.append(StageResult(
                stage_name="anomaly",
                duration_ms=0.0,
                skipped=True,
                skip_reason="unstable_light_calibration",
                metadata={
                    "low_light": is_low_light,
                    "brightness_jump": brightness_jump,
                    "exposure_recovering": exposure_recovering,
                },
            ))
            diag.final_decision = "skipped"
            diag.add_candidate(
                candidate_type="frame_anomaly",
                decision="skipped",
                stage="anomaly",
                evidence={
                    "low_light": is_low_light,
                    "brightness_jump": brightness_jump,
                    "exposure_recovering": exposure_recovering,
                    "detector_mode": "ssim_fallback",
                },
                suppression_reason="unstable_light_calibration",
            )
            self._record_candidate_model_evidence_skip(
                persistent_result=persistent_change_result,
                diag=diag,
                status="upstream_skipped",
                reason="unstable_light_calibration",
            )
            diag.total_duration_ms = (time.monotonic() - start) * 1000
            self._diagnostics.append(diag)
            self._decay_all_unobserved_alert_incidents(
                frame_data.camera_id,
                decay_detection_types=early_observed_alert_channel_types,
            )
            self._set_frame_outcome(
                FrameOutcomeKind.BLOCKED,
                "unstable light blocked SSIM calibration inference",
            )
            return None

        # Frame quality assessment: pre-compute confidence multiplier
        # to reduce false positives from blurry, dark, or noisy frames.
        try:
            quality_score = self._frame_quality_assessor.assess(frame)
            quality_multiplier = self._frame_quality_assessor.confidence_multiplier(quality_score)
        except Exception:
            logger.debug(
                "pipeline.frame_quality_failed",
                camera_id=frame_data.camera_id,
                exc_info=True,
            )
            quality_multiplier = 1.0

        # Stage 2+3: YOLO person filter + Anomaly detection
        # When skip_frame_on_person=True (industrial facility default), run both in
        # parallel via ThreadPoolExecutor to overlap YOLO 15-30ms with Anomalib.
        # Correctness note: when skip_on_person=True AND YOLO finds no persons,
        # the frame is never masked (there's nobody to blur), so anomaly running
        # on the original frame produces the same result as sequential mode.
        # When skip_frame_on_person=False, person masking produces masked_frame
        # needed by anomaly, so stages must remain sequential.
        t2 = time.monotonic()
        self._last_detection_failed = False
        skip_on_person = self.camera_config.person_filter.skip_frame_on_person
        anomaly_model_skipped_for_person = False
        appearance_channel_observed = True
        person_model_input_variant = "unmasked"
        low_light_model_input_variant = (
            "low_light_clahe"
            if is_low_light and self._clahe is not None
            else "standard_light_input"
        )

        def appearance_unavailable_result(
            *,
            reason: str,
            mode: str,
            error: Exception | None = None,
        ) -> AnomalyResult:
            """Fail only the appearance channel and keep localized hazards live."""
            nonlocal model_preprocessing_valid, appearance_channel_observed
            model_preprocessing_valid = False
            appearance_channel_observed = False
            self._last_detection_failed = True
            evidence: dict[str, object] = {
                "mode": mode,
                "detection_failed": True,
                "independent_safety_channels_continued": True,
            }
            if reason == "appearance_inference_timeout":
                evidence.update({
                    "timeout_seconds": 30.0,
                    "consecutive_timeouts": self._consecutive_timeouts,
                })
            if error is not None:
                evidence.update({
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
            diag.add_candidate(
                candidate_type="appearance_inference",
                decision="unavailable",
                stage="anomaly",
                evidence=evidence,
                suppression_reason=reason,
            )
            return AnomalyResult(
                anomaly_score=0.0,
                anomaly_map=None,
                is_anomalous=False,
                threshold=float(self.camera_config.anomaly.threshold),
                detection_failed=True,
            )

        yolo_inference_unavailable = False
        if skip_on_person:
            # Parallel: submit both to thread pool concurrently
            yolo_future: Future = self._inference_executor.submit(
                self._object_detector.detect, frame,
            )
            anomaly_future: Future = self._inference_executor.submit(
                self._predict_anomaly, frame, fused_tensor,
            )
            try:
                detection_result: ObjectDetectionResult = yolo_future.result(timeout=30.0)
                t2_yolo_done = time.monotonic()
            except TimeoutError:
                yolo_inference_unavailable = True
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
                        msg=(
                            "YOLO timed out after 30s — blocking appearance "
                            "and continuing independent safety channels"
                        ),
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
                diag.add_candidate(
                    candidate_type="person_filter_inference",
                    decision="unavailable",
                    stage="inference",
                    evidence={
                        "timeout_seconds": 30.0,
                        "consecutive_timeouts": self._consecutive_timeouts,
                        "mode": "skip_on_person",
                        "failed_channel": "yolo",
                    },
                    suppression_reason="yolo_inference_timeout",
                )
                self._record_candidate_model_evidence_skip(
                    persistent_result=persistent_change_result,
                    diag=diag,
                    status="inference_error",
                    reason="yolo_inference_timeout",
                )
                t2_yolo_done = time.monotonic()
                t3_anomaly_done = t2_yolo_done
                detection_result = ObjectDetectionResult(
                    filter_available=False,
                )
                anomaly_result = appearance_unavailable_result(
                    reason="yolo_inference_timeout_appearance_blocked",
                    mode="skip_on_person",
                )
                fallback_future: Future = Future()
                fallback_future.set_result(anomaly_result)
                anomaly_future = fallback_future
            except Exception as exc:
                yolo_inference_unavailable = True
                self._consecutive_timeouts = 0
                logger.error(
                    "pipeline.yolo_inference_failed",
                    camera_id=self.camera_config.camera_id,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    msg=(
                        "YOLO inference failed — blocking appearance and "
                        "continuing independent safety channels"
                    ),
                )
                yolo_future.cancel()
                anomaly_future.cancel()
                diag.add_candidate(
                    candidate_type="person_filter_inference",
                    decision="unavailable",
                    stage="inference",
                    evidence={
                        "mode": "skip_on_person",
                        "failed_channel": "yolo",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    suppression_reason="yolo_inference_error",
                )
                self._record_candidate_model_evidence_skip(
                    persistent_result=persistent_change_result,
                    diag=diag,
                    status="inference_error",
                    reason="yolo_inference_error",
                )
                t2_yolo_done = time.monotonic()
                t3_anomaly_done = t2_yolo_done
                detection_result = ObjectDetectionResult(
                    filter_available=False,
                )
                anomaly_result = appearance_unavailable_result(
                    reason="yolo_inference_error_appearance_blocked",
                    mode="skip_on_person",
                    error=exc,
                )
                fallback_future = Future()
                fallback_future.set_result(anomaly_result)
                anomaly_future = fallback_future

            if (
                not yolo_inference_unavailable
                and not detection_result.filter_available
            ):
                yolo_inference_unavailable = True
                anomaly_future.cancel()
                diag.add_candidate(
                    candidate_type="person_filter_inference",
                    decision="unavailable",
                    stage="inference",
                    evidence={
                        "mode": "skip_on_person",
                        "failed_channel": "yolo",
                        "independent_safety_channels_continued": True,
                    },
                    suppression_reason="yolo_filter_unavailable",
                )
                self._record_candidate_model_evidence_skip(
                    persistent_result=persistent_change_result,
                    diag=diag,
                    status="inference_error",
                    reason="yolo_filter_unavailable",
                )
                detection_result = ObjectDetectionResult(
                    filter_available=False,
                )
                anomaly_result = appearance_unavailable_result(
                    reason="yolo_filter_unavailable_appearance_blocked",
                    mode="skip_on_person",
                )
                fallback_future = Future()
                fallback_future.set_result(anomaly_result)
                anomaly_future = fallback_future

            try:
                anomaly_result: AnomalyResult = anomaly_future.result(timeout=30.0)
                t3_anomaly_done = time.monotonic()
            except TimeoutError:
                self.stats.frames_timeout += 1
                self._consecutive_timeouts += 1
                logger.error(
                    "pipeline.appearance_inference_timeout",
                    camera_id=self.camera_config.camera_id,
                    consecutive=self._consecutive_timeouts,
                    msg=(
                        "Appearance inference timed out after 30s — "
                        "continuing independent safety channels"
                    ),
                )
                if not anomaly_future.cancel():
                    logger.warning(
                        "pipeline.future_cancel_failed",
                        camera_id=self.camera_config.camera_id,
                        stage="anomaly",
                        msg=(
                            "Anomaly inference still running after timeout — "
                            "worker thread busy"
                        ),
                    )
                t3_anomaly_done = time.monotonic()
                anomaly_result = appearance_unavailable_result(
                    reason="appearance_inference_timeout",
                    mode="skip_on_person",
                )
            except Exception as exc:
                self._consecutive_timeouts = 0
                logger.error(
                    "pipeline.appearance_inference_failed",
                    camera_id=self.camera_config.camera_id,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    msg=(
                        "Appearance inference failed — continuing independent "
                        "safety channels"
                    ),
                )
                t3_anomaly_done = time.monotonic()
                anomaly_result = appearance_unavailable_result(
                    reason="appearance_inference_error",
                    mode="skip_on_person",
                    error=exc,
                )
            else:
                if not yolo_inference_unavailable:
                    self._consecutive_timeouts = 0
                if (
                    not yolo_inference_unavailable
                    and anomaly_result.detection_failed
                ):
                    anomaly_result = appearance_unavailable_result(
                        reason="appearance_detection_failed",
                        mode="skip_on_person",
                    )

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
                anomaly_model_skipped_for_person = True
                appearance_channel_observed = False
                person_model_input_variant = "person_frame_model_skipped"
                original_person_frame_score = anomaly_result.anomaly_score
                original_person_frame_anomalous = anomaly_result.is_anomalous
                # ``skip_frame_on_person`` applies only to the appearance-model
                # channel.  Returning here used to drop independent Simplex,
                # water/plume and non-person YOLO hazards from the same frame.
                # Keep those safety channels live while making model evidence
                # explicitly unavailable for candidate calibration.
                anomaly_result = _updated_anomaly_result(
                    anomaly_result,
                    anomaly_score=0.0,
                    anomaly_map=None,
                    is_anomalous=False,
                    raw_score=None,
                    model_score=None,
                    model_anomaly_map=None,
                    model_output_space=None,
                    model_output_comparable=False,
                )
                diag.add_candidate(
                    candidate_type="person_filter",
                    decision="model_skipped",
                    stage="yolo",
                    evidence={
                        "person_count": len(detection_result.persons),
                        "object_count": len(detection_result.objects),
                        "discarded_anomaly_score": round(
                            original_person_frame_score,
                            4,
                        ),
                        "discarded_is_anomalous": original_person_frame_anomalous,
                        "independent_safety_channels_continued": True,
                    },
                    suppression_reason="person_detected:model_channel_only",
                )
        else:
            # Parallel: run YOLO and anomaly concurrently on the original
            # frame.  If YOLO detects persons AND anomaly fires, re-run
            # anomaly on the masked frame (rare case, preserves correctness).
            yolo_future = self._inference_executor.submit(
                self._object_detector.detect, frame,
            )
            anomaly_future = self._inference_executor.submit(
                self._predict_anomaly, frame, fused_tensor,
            )
            try:
                detection_result = yolo_future.result(timeout=30.0)
                t2_yolo_done = time.monotonic()
            except TimeoutError:
                yolo_inference_unavailable = True
                self.stats.frames_timeout += 1
                self._consecutive_timeouts += 1
                logger.error(
                    "pipeline.inference_timeout",
                    camera_id=self.camera_config.camera_id,
                    consecutive=self._consecutive_timeouts,
                    msg=(
                        "YOLO timed out after 30s — blocking appearance "
                        "and continuing independent safety channels"
                    ),
                )
                yolo_future.cancel()
                anomaly_future.cancel()
                diag.add_candidate(
                    candidate_type="person_filter_inference",
                    decision="unavailable",
                    stage="inference",
                    evidence={
                        "timeout_seconds": 30.0,
                        "consecutive_timeouts": self._consecutive_timeouts,
                        "mode": "rerun_masked",
                        "failed_channel": "yolo",
                    },
                    suppression_reason="yolo_inference_timeout",
                )
                self._record_candidate_model_evidence_skip(
                    persistent_result=persistent_change_result,
                    diag=diag,
                    status="inference_error",
                    reason="yolo_inference_timeout",
                )
                t2_yolo_done = time.monotonic()
                t3_anomaly_done = t2_yolo_done
                detection_result = ObjectDetectionResult(
                    filter_available=False,
                )
                anomaly_result = appearance_unavailable_result(
                    reason="yolo_inference_timeout_appearance_blocked",
                    mode="rerun_masked",
                )
                fallback_future = Future()
                fallback_future.set_result(anomaly_result)
                anomaly_future = fallback_future
            except Exception as exc:
                yolo_inference_unavailable = True
                self._consecutive_timeouts = 0
                logger.error(
                    "pipeline.yolo_inference_failed",
                    camera_id=self.camera_config.camera_id,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    msg=(
                        "YOLO inference failed — blocking appearance and "
                        "continuing independent safety channels"
                    ),
                )
                yolo_future.cancel()
                anomaly_future.cancel()
                diag.add_candidate(
                    candidate_type="person_filter_inference",
                    decision="unavailable",
                    stage="inference",
                    evidence={
                        "mode": "rerun_masked",
                        "failed_channel": "yolo",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    suppression_reason="yolo_inference_error",
                )
                self._record_candidate_model_evidence_skip(
                    persistent_result=persistent_change_result,
                    diag=diag,
                    status="inference_error",
                    reason="yolo_inference_error",
                )
                t2_yolo_done = time.monotonic()
                t3_anomaly_done = t2_yolo_done
                detection_result = ObjectDetectionResult(
                    filter_available=False,
                )
                anomaly_result = appearance_unavailable_result(
                    reason="yolo_inference_error_appearance_blocked",
                    mode="rerun_masked",
                    error=exc,
                )
                fallback_future = Future()
                fallback_future.set_result(anomaly_result)
                anomaly_future = fallback_future

            if (
                not yolo_inference_unavailable
                and not detection_result.filter_available
            ):
                yolo_inference_unavailable = True
                anomaly_future.cancel()
                diag.add_candidate(
                    candidate_type="person_filter_inference",
                    decision="unavailable",
                    stage="inference",
                    evidence={
                        "mode": "rerun_masked",
                        "failed_channel": "yolo",
                        "independent_safety_channels_continued": True,
                    },
                    suppression_reason="yolo_filter_unavailable",
                )
                self._record_candidate_model_evidence_skip(
                    persistent_result=persistent_change_result,
                    diag=diag,
                    status="inference_error",
                    reason="yolo_filter_unavailable",
                )
                detection_result = ObjectDetectionResult(
                    filter_available=False,
                )
                anomaly_result = appearance_unavailable_result(
                    reason="yolo_filter_unavailable_appearance_blocked",
                    mode="rerun_masked",
                )
                fallback_future = Future()
                fallback_future.set_result(anomaly_result)
                anomaly_future = fallback_future

            try:
                anomaly_result = anomaly_future.result(timeout=30.0)
                t3_anomaly_done = time.monotonic()
            except TimeoutError:
                self.stats.frames_timeout += 1
                self._consecutive_timeouts += 1
                logger.error(
                    "pipeline.appearance_inference_timeout",
                    camera_id=self.camera_config.camera_id,
                    consecutive=self._consecutive_timeouts,
                    msg=(
                        "Appearance inference timed out after 30s — "
                        "continuing independent safety channels"
                    ),
                )
                if not anomaly_future.cancel():
                    logger.warning(
                        "pipeline.future_cancel_failed",
                        camera_id=self.camera_config.camera_id,
                        stage="anomaly",
                        msg=(
                            "Anomaly inference still running after timeout — "
                            "worker thread busy"
                        ),
                    )
                t3_anomaly_done = time.monotonic()
                anomaly_result = appearance_unavailable_result(
                    reason="appearance_inference_timeout",
                    mode="rerun_masked",
                )
            except Exception as exc:
                self._consecutive_timeouts = 0
                logger.error(
                    "pipeline.appearance_inference_failed",
                    camera_id=self.camera_config.camera_id,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    msg=(
                        "Appearance inference failed — continuing independent "
                        "safety channels"
                    ),
                )
                t3_anomaly_done = time.monotonic()
                anomaly_result = appearance_unavailable_result(
                    reason="appearance_inference_error",
                    mode="rerun_masked",
                    error=exc,
                )
            else:
                if not yolo_inference_unavailable:
                    self._consecutive_timeouts = 0
                if (
                    not yolo_inference_unavailable
                    and anomaly_result.detection_failed
                ):
                    anomaly_result = appearance_unavailable_result(
                        reason="appearance_detection_failed",
                        mode="rerun_masked",
                    )

            if not detection_result.filter_available and not self._person_filter_warned:
                self._person_filter_warned = True
                logger.warning(
                    "pipeline.person_filter_offline",
                    camera_id=self.camera_config.camera_id,
                    msg="YOLO person filter unavailable — frames will not be filtered for humans",
                )

            # Re-run anomaly with person-masked frame only when both
            # persons and anomaly are detected (rare — avoids double cost
            # in the common case).
            if (
                detection_result.has_persons
                and detection_result.masked_frame is not None
                and anomaly_result.is_anomalous
            ):
                person_model_input_variant = "person_masked"
                masked_future: Future = self._inference_executor.submit(
                    self._predict_anomaly,
                    detection_result.masked_frame,
                )
                try:
                    anomaly_result = masked_future.result(timeout=30.0)
                except TimeoutError:
                    self.stats.frames_timeout += 1
                    self._consecutive_timeouts += 1
                    masked_future.cancel()
                    anomaly_result = appearance_unavailable_result(
                        reason="person_masked_inference_timeout",
                        mode="person_masked_rerun",
                    )
                except Exception as exc:
                    anomaly_result = appearance_unavailable_result(
                        reason="person_masked_inference_error",
                        mode="person_masked_rerun",
                        error=exc,
                    )
                else:
                    if anomaly_result.detection_failed:
                        anomaly_result = appearance_unavailable_result(
                            reason="person_masked_detection_failed",
                            mode="person_masked_rerun",
                        )
                t3_anomaly_done = time.monotonic()
            elif detection_result.has_persons:
                person_model_input_variant = "person_present_unmasked"
            if detection_result.has_persons:
                # A single no-person calibration must never be reused for a
                # dynamically masked (or person-contaminated) model input.
                # The generic anomaly channel remains available, but candidate
                # model evidence becomes tri-state unavailable for this frame.
                model_preprocessing_valid = False

        anomaly_result = _updated_anomaly_result(
            anomaly_result,
            model_input_variant=(
                f"{scene_model_input_variant}:{imaging_model_input_variant}:"
                f"{alignment_model_input_variant}:"
                f"{low_light_model_input_variant}:"
                f"{person_model_input_variant}"
            ),
            model_output_comparable=(
                anomaly_result.model_output_comparable
                and model_preprocessing_valid
            ),
        )

        diag.stages.append(StageResult(
            stage_name="yolo",
            duration_ms=(t2_yolo_done - t2) * 1000,
            metadata={
                "person_count": len(detection_result.persons),
                "object_count": len(detection_result.objects),
                "classes": [o.class_name for o in detection_result.non_person_objects],
                "track_ids": [o.track_id for o in detection_result.objects if o.track_id is not None],
                "parallel": True,
                "mode": "skip_on_person" if skip_on_person else "rerun_masked",
            },
        ))

        self.stats.frames_analyzed += 1
        diag.stages.append(StageResult(
            stage_name="anomaly",
            duration_ms=(t3_anomaly_done - t2) * 1000 if skip_on_person else (t3_anomaly_done - t2_yolo_done) * 1000,
            skipped=anomaly_model_skipped_for_person,
            skip_reason=(
                "person_detected:model_channel_only"
                if anomaly_model_skipped_for_person
                else ""
            ),
            metadata={"score": round(anomaly_result.anomaly_score, 4)},
        ))
        diag.anomaly_score = anomaly_result.anomaly_score
        diag.is_anomalous = anomaly_result.is_anomalous

        # Apply frame quality confidence multiplier to anomaly score.
        # Poor quality frames (blur, low exposure) get a reduced score
        # to cut false positives from degraded input.
        if quality_multiplier < 1.0:
            adjusted_score = anomaly_result.anomaly_score * quality_multiplier
            anomaly_result = _updated_anomaly_result(
                anomaly_result,
                anomaly_score=adjusted_score,
                is_anomalous=adjusted_score >= anomaly_result.threshold,
            )
            diag.anomaly_score = adjusted_score
            diag.is_anomalous = anomaly_result.is_anomalous
            logger.debug(
                "pipeline.quality_adjusted",
                camera_id=frame_data.camera_id,
                multiplier=round(quality_multiplier, 3),
                original=round(anomaly_result.anomaly_score / quality_multiplier, 4)
                if quality_multiplier > 0 else 0,
                adjusted=round(adjusted_score, 4),
            )

        # Snapshot the appearance-model channel before independent Simplex,
        # persistent-change, water, plume, or object-safety scores are fused.
        appearance_model_score = anomaly_result.anomaly_score
        appearance_decision_result = anomaly_result

        # Data contract validation at detector→grader boundary (advisory only)
        try:
            validate_anomaly_result(anomaly_result)
        except Exception:
            logger.debug("pipeline.contract_validation_failed", exc_info=True)

        # Feed anomaly score to drift detector
        if self._drift_detector is not None and not anomaly_model_skipped_for_person:
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
        simplex_score = 0.0
        simplex_objects: list[dict] = []
        simplex_map = None
        if self._simplex is not None:
            try:
                t3b = time.monotonic()
                # Set reference frame from the first processed frame
                if not self._simplex_reference_set:
                    self._simplex.set_reference(frame)
                    self._simplex_reference_set = True
                    logger.debug(
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

                # Keep Simplex as localized evidence.  A frame-global score
                # would otherwise accumulate in every include zone when the
                # anomaly head has no usable heatmap.
                if simplex_detected:
                    if appearance_decision_result.is_anomalous:
                        simplex_score = min(
                            appearance_decision_result.anomaly_score * 1.1,
                            1.0,
                        )
                        logger.debug(
                            "pipeline.simplex_boost",
                            camera_id=frame_data.camera_id,
                            original_score=round(
                                appearance_decision_result.anomaly_score,
                                4,
                            ),
                            boosted_score=round(simplex_score, 4),
                        )
                    else:
                        simplex_score = max(
                            appearance_decision_result.anomaly_score + 0.05,
                            0.6,
                        )
                        logger.debug(
                            "pipeline.simplex_safety_detection",
                            camera_id=frame_data.camera_id,
                            anomalib_score=round(
                                appearance_decision_result.anomaly_score,
                                4,
                            ),
                            static_regions=len(simplex_result.static_regions),
                            max_static_seconds=round(simplex_result.max_static_seconds, 1),
                        )
                    simplex_objects = [
                        self._simplex_visual_payload(region, simplex_score, index)
                        for index, region in enumerate(simplex_result.static_regions)
                    ]
                    simplex_map = self._localized_score_map_from_objects(
                        simplex_objects,
                        frame.shape,
                    )
                    diag.anomaly_score = max(
                        appearance_decision_result.anomaly_score,
                        simplex_score,
                    )
                    diag.is_anomalous = True
            except Exception as e:
                # Simplex failure must never crash the pipeline
                logger.warning(
                    "pipeline.simplex_error",
                    camera_id=frame_data.camera_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )

        confirmed_persistent_candidates = []
        persistent_score = 0.0
        if persistent_change_result is not None:
            confirmed_persistent_candidates = self._confirm_persistent_change_candidates(
                current_frame=scene_confirmation_frame,
                persistent_result=persistent_change_result,
                anomaly_result=appearance_decision_result,
                diag=diag,
            )
            persistent_change_objects = [
                self._independent_visual_payload(candidate)
                for candidate in confirmed_persistent_candidates
            ]
        else:
            persistent_change_objects = []
        appearance_decision_result = self._apply_candidate_anomaly_attribution(
            appearance_decision_result,
            diag=diag,
            frame_shape=scene_confirmation_frame.shape,
            # Simplex is now a localized incident.  It must not preserve an
            # unrelated rejected appearance hotspot elsewhere in the frame.
            simplex_detected=False,
            persistent_candidates=(
                list(persistent_change_result.candidates)
                if persistent_change_result is not None
                else []
            ),
        )
        persistent_change_detected = bool(confirmed_persistent_candidates)
        if persistent_change_detected:
            persistent_score = max(
                candidate.score for candidate in confirmed_persistent_candidates
            )
        persistent_map = self._confirmed_persistent_score_map(
            confirmed_persistent_candidates,
            frame.shape,
        )
        incident_support_maps: dict[str, np.ndarray] = {}
        for candidate, payload in zip(
            confirmed_persistent_candidates,
            persistent_change_objects,
            strict=True,
        ):
            candidate_map = self._confirmed_persistent_score_map(
                [candidate],
                frame.shape,
            )
            if candidate_map is not None:
                incident_support_maps[
                    self._incident_identity(
                        DetectionType.PERSISTENT_CHANGE,
                        payload,
                    )
                ] = candidate_map

        water_leak_detected = bool(
            water_leak_result is not None
            and water_leak_result.has_detection
        )
        plume_detected = bool(
            plume_result is not None
            and plume_result.has_detection
        )

        channel_scores = [appearance_decision_result.anomaly_score]
        channel_maps = [appearance_decision_result.anomaly_map]
        if simplex_detected:
            channel_scores.append(simplex_score)
            channel_maps.append(simplex_map)
        if persistent_change_detected:
            channel_scores.append(persistent_score)
            channel_maps.append(persistent_map)
        if water_leak_detected:
            channel_scores.append(float(water_leak_result.score))
            channel_maps.append(water_leak_result.anomaly_map)
        if plume_detected:
            channel_scores.append(float(plume_result.score))
            channel_maps.append(plume_result.anomaly_map)
        combined_score = max(
            (score for score in channel_scores if math.isfinite(score)),
            default=0.0,
        )
        independent_channel_detected = bool(
            simplex_detected
            or persistent_change_detected
            or water_leak_detected
            or plume_detected
        )
        anomaly_result = _updated_anomaly_result(
            appearance_decision_result,
            anomaly_score=combined_score,
            anomaly_map=self._merge_visual_maps_max(channel_maps),
            is_anomalous=(
                appearance_decision_result.is_anomalous
                or independent_channel_detected
            ),
        )
        diag.anomaly_score = combined_score
        diag.is_anomalous = anomaly_result.is_anomalous

        # Cache anomaly heatmap for overlay stream
        with self._latest_anomaly_map_lock:
            self._latest_anomaly_map = (
                anomaly_result.anomaly_map.copy()
                if anomaly_result.anomaly_map is not None
                else None
            )

        # Update anomaly lock state
        self._update_lock_state(
            anomaly_result,
            allow_engage=not ssim_fallback_active,
        )

        # YOLO-004: Determine hybrid detection type
        detection_type = DetectionType.ANOMALY
        detected_objects: list[dict] = []
        if detection_result.non_person_objects:
            detected_objects.extend(
                self._object_detection_payload(obj)
                for obj in detection_result.non_person_objects
            )
            if anomaly_result.is_anomalous:
                detection_type = DetectionType.HYBRID
            else:
                detection_type = DetectionType.OBJECT
        if simplex_objects:
            detected_objects.extend(simplex_objects)
            detection_type = DetectionType.PERSISTENT_CHANGE
        if persistent_change_objects:
            detected_objects.extend(persistent_change_objects)
            detection_type = DetectionType.PERSISTENT_CHANGE
        if water_leak_objects:
            detected_objects.extend(water_leak_objects)
            detection_type = DetectionType.WATER_LEAK
        if plume_objects:
            detected_objects.extend(plume_objects)
            detection_type = DetectionType.PLUME

        object_safety_score = self._object_safety_score(detected_objects)
        independent_visual_sources: list[tuple[float, np.ndarray | None]] = []
        if simplex_detected:
            independent_visual_sources.append((simplex_score, simplex_map))
        if persistent_change_detected:
            independent_visual_sources.append(
                (persistent_score, persistent_map)
            )
        if water_leak_detected:
            independent_visual_sources.append(
                (float(water_leak_result.score), water_leak_result.anomaly_map)
            )
        if plume_detected:
            independent_visual_sources.append(
                (float(plume_result.score), plume_result.anomaly_map)
            )
        finite_visual_sources = [
            (score, channel_map)
            for score, channel_map in independent_visual_sources
            if math.isfinite(score) and score > 0.0
        ]
        independent_visual_score = max(
            (score for score, _channel_map in finite_visual_sources),
            default=0.0,
        )
        independent_visual_map = self._merge_visual_maps_max(
            channel_map
            for _score, channel_map in finite_visual_sources
        )
        independent_hazard_active = bool(
            independent_visual_score > 0.0 or object_safety_score > 0.0
        )

        # Channel availability is tri-state.  Only a head that completed on
        # this frame may decay a prior incident as "not observed"; a failed or
        # intentionally skipped head is unknown and must preserve its CUSUM.
        observed_alert_channel_types: set[str] = set()
        if appearance_channel_observed:
            observed_alert_channel_types.update({
                DetectionType.ANOMALY.value,
                DetectionType.HYBRID.value,
            })
        if detection_result.filter_available:
            observed_alert_channel_types.add(DetectionType.OBJECT.value)
        if fast_motion_observed:
            observed_alert_channel_types.add(DetectionType.PROJECTILE.value)
        persistent_heads_observed = bool(
            (self._simplex is None or simplex_result is not None)
            and (
                self._persistent_change is None
                or persistent_change_result is not None
            )
        )
        if persistent_heads_observed:
            observed_alert_channel_types.add(
                DetectionType.PERSISTENT_CHANGE.value
            )
        if water_leak_result is not None:
            observed_alert_channel_types.add(DetectionType.WATER_LEAK.value)
        if plume_result is not None:
            observed_alert_channel_types.add(DetectionType.PLUME.value)

        def independent_hazard_result() -> AnomalyResult:
            """Drop appearance/Simplex score while preserving independent channels."""
            return _updated_anomaly_result(
                anomaly_result,
                anomaly_score=independent_visual_score,
                anomaly_map=independent_visual_map,
                is_anomalous=independent_visual_score > 0.0,
            )

        # DET-006: LEARNING mode suppresses alerts
        if current_mode == PipelineMode.LEARNING:
            self._update_latency(start)
            diag.final_decision = "suppressed"
            diag.add_candidate(
                candidate_type="frame_anomaly",
                decision="suppressed",
                stage="decision",
                evidence={
                    "pipeline_mode": current_mode.value,
                    "anomaly_score": round(anomaly_result.anomaly_score, 4),
                    "is_anomalous": anomaly_result.is_anomalous,
                    "detected_objects": detected_objects,
                    "persistent_change": (
                        persistent_change_result.evidence()
                        if persistent_change_result is not None
                        else None
                    ),
                    "water_leak": (
                        water_leak_result.evidence()
                        if water_leak_result is not None
                        else None
                    ),
                    "plume": (
                        plume_result.evidence()
                        if plume_result is not None
                        else None
                    ),
                },
                suppression_reason="learning_mode",
            )
            diag.total_duration_ms = (time.monotonic() - start) * 1000
            self._diagnostics.append(diag)
            self._diagnostics.append_score(FrameScoreRecord(
                frame_number=frame_data.frame_number,
                timestamp=time.time(),
                anomaly_score=anomaly_result.anomaly_score,
                was_alert=False,
            ))
            self._decay_all_unobserved_alert_incidents(
                frame_data.camera_id,
                decay_detection_types=observed_alert_channel_types,
            )
            self._set_frame_outcome(
                FrameOutcomeKind.BLOCKED,
                "learning mode suppresses alert emission",
            )
            return None

        # Skip alert if neither Anomalib nor YOLO detected anything
        if not anomaly_result.is_anomalous and not detected_objects:
            self._update_latency(start)
            diag.final_decision = "normal"
            diag.add_candidate(
                candidate_type="frame_anomaly",
                decision="normal",
                stage="decision",
                evidence={
                    "anomaly_score": round(anomaly_result.anomaly_score, 4),
                    "is_anomalous": anomaly_result.is_anomalous,
                    "detected_objects": detected_objects,
                    "persistent_change": (
                        persistent_change_result.evidence()
                        if persistent_change_result is not None
                        else None
                    ),
                    "water_leak": (
                        water_leak_result.evidence()
                        if water_leak_result is not None
                        else None
                    ),
                    "plume": (
                        plume_result.evidence()
                        if plume_result is not None
                        else None
                    ),
                },
            )
            diag.total_duration_ms = (time.monotonic() - start) * 1000
            self._diagnostics.append(diag)
            self._diagnostics.append_score(FrameScoreRecord(
                frame_number=frame_data.frame_number,
                timestamp=time.time(),
                anomaly_score=anomaly_result.anomaly_score,
                was_alert=False,
            ))
            self._decay_all_unobserved_alert_incidents(
                frame_data.camera_id,
                decay_detection_types=observed_alert_channel_types,
            )
            return None

        self.stats.anomalies_detected += 1

        # D1: Open vocabulary classification on anomaly region
        classification_result: tuple[str, float] | None = None
        classifier_multi_zone_unsafe = bool(
            self._classifier is not None
            and len(self._zone_mask.get_include_zones()) > 1
        )
        classifier_multi_incident_unsafe = bool(
            self._classifier is not None
            and len(self._appearance_map_components(
                appearance_decision_result.anomaly_map,
                frame.shape,
            )) > 1
        )
        classifier_localization_unsafe = bool(
            classifier_multi_zone_unsafe or classifier_multi_incident_unsafe
        )
        if classifier_localization_unsafe:
            # The current classifier returns one frame-global label/bbox.  Until
            # it emits per-component results, applying that label to multiple
            # zones/components can suppress or corroborate an unrelated
            # incident.
            diag.stages.append(StageResult(
                stage_name="classifier",
                duration_ms=0.0,
                skipped=True,
                skip_reason=(
                    "multi_zone_localization_unavailable"
                    if classifier_multi_zone_unsafe
                    else "multi_incident_localization_unavailable"
                ),
            ))
        if (
            self._classifier is not None
            and (self._classifier_config is None or self._classifier_config.enabled is True)
            and not classifier_localization_unsafe
            and "classifier" not in self._auxiliary_load_blocked
            and appearance_decision_result.is_anomalous
            and appearance_decision_result.anomaly_score
            >= self._classifier_min_score
        ):
            try:
                t_cls = time.monotonic()
                bbox = self._extract_anomaly_bbox(
                    appearance_decision_result.anomaly_map,
                    frame.shape,
                )
                classify_evidence = getattr(self._classifier, "classify_with_evidence", None)
                raw_classification = (
                    classify_evidence(frame, bbox=bbox)
                    if bbox is not None and callable(classify_evidence)
                    else None
                )
                vocabulary = getattr(self._classifier, "vocabulary", None)
                if not isinstance(vocabulary, (list, tuple, set, frozenset)):
                    vocabulary = self._classifier_vocabulary
                classification_result = localized_classification(
                    raw_classification,
                    vocabulary,
                    appearance_decision_result.anomaly_map,
                    frame.shape,
                    bbox,
                    self._classifier_min_confidence,
                )
                self._auxiliary_errors.pop("classifier", None)
                cls_duration_ms = (time.monotonic() - t_cls) * 1000
                diag.stages.append(StageResult(
                    stage_name="classifier",
                    duration_ms=cls_duration_ms,
                    metadata={
                        "label": classification_result[0] if classification_result else None,
                        "confidence": round(classification_result[1], 3) if classification_result else None,
                        "bbox": list(bbox) if bbox else None,
                        "evidence_accepted": classification_result is not None,
                        "untrusted_result_ignored": (
                            raw_classification is not None and classification_result is None
                        ),
                    },
                ))
                if classification_result:
                    logger.debug(
                        "pipeline.classified",
                        camera_id=frame_data.camera_id,
                        label=classification_result[0],
                        confidence=round(classification_result[1], 3),
                    )
            except Exception as e:
                classification_result = None
                self._record_auxiliary_failure("classifier", e, diag, t_cls)
                logger.warning(
                    "pipeline.classifier_error",
                    camera_id=frame_data.camera_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )

        # Suppress alert if classifier label matches suppress_labels
        if classification_result is not None:
            label, _conf = classification_result
            if (
                label in getattr(self, "_classifier_suppress", set())
                and label not in getattr(self, "_classifier_high_risk", set())
                and label not in getattr(self, "_classifier_low_risk", set())
            ):
                logger.debug(
                    "pipeline.classifier_label_suppressed",
                    camera_id=frame_data.camera_id,
                    label=label,
                )
                if independent_hazard_active:
                    diag.add_candidate(
                        candidate_type="classification",
                        decision="appearance_suppressed_independent_continued",
                        stage="classifier",
                        evidence={
                            "label": label,
                            "appearance_model_score": round(
                                appearance_model_score,
                                4,
                            ),
                            "independent_visual_score": round(
                                independent_visual_score,
                                4,
                            ),
                            "object_safety_score": round(object_safety_score, 4),
                        },
                    )
                    anomaly_result = independent_hazard_result()
                    appearance_decision_result = _updated_anomaly_result(
                        appearance_decision_result,
                        anomaly_score=0.0,
                        anomaly_map=None,
                        is_anomalous=False,
                    )
                    classification_result = None
                    if object_safety_score > 0.0 and not finite_visual_sources:
                        detection_type = DetectionType.OBJECT
                else:
                    diag.final_decision = "suppressed"
                    diag.add_candidate(
                        candidate_type="classification",
                        decision="suppressed",
                        stage="classifier",
                        evidence={
                            "label": label,
                            "anomaly_score": round(anomaly_result.anomaly_score, 4),
                            "is_anomalous": anomaly_result.is_anomalous,
                        },
                        suppression_reason="classifier_label",
                    )
                    self._update_latency(start)
                    diag.total_duration_ms = (time.monotonic() - start) * 1000
                    self._diagnostics.append(diag)
                    self._diagnostics.append_score(FrameScoreRecord(
                        frame_number=frame_data.frame_number,
                        timestamp=time.time(),
                        anomaly_score=anomaly_result.anomaly_score,
                        was_alert=False,
                    ))
                    self._decay_all_unobserved_alert_incidents(
                        frame_data.camera_id,
                        decay_detection_types=observed_alert_channel_types,
                    )
                    self._set_frame_outcome(
                        FrameOutcomeKind.BLOCKED,
                        "classifier suppression blocked an anomalous frame",
                    )
                    return None

        # F5: Stationary-object suppression. Run the temporal tracker up-front
        # (instead of only after a successful alert, as the physics enrichment
        # path does below) so we can decide whether every active track has
        # settled into a suppressed state. When every track in the frame is
        # suppressed — and therefore the only reason anomaly scoring would
        # re-fire is a long-settled object — we skip grader evaluation so the
        # operator is not re-alerted. A brand-new track (i.e. a freshly
        # entered object) cannot be suppressed in the same frame it is born,
        # so this still lets fresh anomalies through. The result is cached so
        # the physics-enrichment block does not re-run update() and create
        # duplicate tracks.
        pre_temporal = None
        pre_regions = []
        if (
            appearance_decision_result.is_anomalous
            and getattr(self, "_physics_tracker", None) is not None
            and getattr(self, "_physics_postproc", None) is not None
        ):
            try:
                pre_regions = (
                    self._physics_postproc.extract_regions(
                        appearance_decision_result.anomaly_map,
                        threshold=0.3,
                    )
                    if appearance_decision_result.anomaly_map is not None
                    else []
                )
                pre_temporal = self._physics_tracker.update(
                    pre_regions, self._frame_counter,
                )
            except Exception as e:
                logger.debug(
                    "pipeline.temporal_tracker_failed",
                    camera_id=frame_data.camera_id,
                    error=str(e),
                )
                pre_temporal = None

            if (
                pre_temporal is not None
                and pre_temporal.active_tracks
                and self._stationary_evidence_is_current(
                    appearance_decision_result.anomaly_map, pre_regions,
                    pre_temporal.active_tracks, self._frame_counter,
                )
            ):
                logger.debug(
                    "pipeline.alert_suppressed_stationary",
                    camera_id=frame_data.camera_id,
                    frame_number=frame_data.frame_number,
                    track_ids=[t.track_id for t in pre_temporal.active_tracks],
                )
                if independent_hazard_active:
                    diag.add_candidate(
                        candidate_type="stationary_track",
                        decision="appearance_suppressed_independent_continued",
                        stage="temporal_tracker",
                        evidence={
                            "track_ids": [
                                t.track_id for t in pre_temporal.active_tracks
                            ],
                            "active_track_count": len(pre_temporal.active_tracks),
                            "independent_visual_score": round(
                                independent_visual_score,
                                4,
                            ),
                            "object_safety_score": round(object_safety_score, 4),
                        },
                    )
                    anomaly_result = independent_hazard_result()
                    appearance_decision_result = _updated_anomaly_result(
                        appearance_decision_result,
                        anomaly_score=0.0,
                        anomaly_map=None,
                        is_anomalous=False,
                    )
                    pre_temporal = None
                    if object_safety_score > 0.0 and not finite_visual_sources:
                        detection_type = DetectionType.OBJECT
                else:
                    self._update_latency(start)
                    diag.final_decision = "suppressed"
                    diag.add_candidate(
                        candidate_type="stationary_track",
                        decision="suppressed",
                        stage="temporal_tracker",
                        evidence={
                            "track_ids": [
                                t.track_id for t in pre_temporal.active_tracks
                            ],
                            "active_track_count": len(pre_temporal.active_tracks),
                            "anomaly_score": round(anomaly_result.anomaly_score, 4),
                        },
                        suppression_reason="stationary_track_suppressed",
                    )
                    diag.total_duration_ms = (time.monotonic() - start) * 1000
                    self._diagnostics.append(diag)
                    self._diagnostics.append_score(FrameScoreRecord(
                        frame_number=frame_data.frame_number,
                        timestamp=time.time(),
                        anomaly_score=anomaly_result.anomaly_score,
                        was_alert=False,
                    ))
                    self._decay_all_unobserved_alert_incidents(
                        frame_data.camera_id,
                        decay_detection_types=observed_alert_channel_types,
                    )
                    self._set_frame_outcome(
                        FrameOutcomeKind.BLOCKED,
                        "stationary-track policy suppressed an anomalous frame",
                    )
                    return None

        # Multi-zone alert grading with semantic context.
        # Classification info is forwarded so the grader can use it as
        # corroboration for the F1 single-frame early-warning fast path.
        cls_label = classification_result[0] if classification_result else None
        cls_conf = classification_result[1] if classification_result else None
        alert = self._evaluate_zones(
            frame_data, anomaly_result, frame,
            detection_type=detection_type,
            detected_objects=detected_objects,
            classification_label=cls_label,
            classification_confidence=cls_conf,
            appearance_result=appearance_decision_result,
            incident_support_maps=incident_support_maps,
            decay_detection_types=observed_alert_channel_types,
        )
        frame_alerts = (
            [
                alert,
                *(getattr(alert, "_secondary_zone_alerts", None) or []),
            ]
            if alert is not None
            else []
        )

        # D1: Attach classification to alert and adjust severity
        appearance_alerts = [
            candidate
            for candidate in frame_alerts
            if getattr(
                getattr(candidate, "detection_type", None),
                "value",
                getattr(candidate, "detection_type", None),
            ) in {
                DetectionType.ANOMALY.value,
                DetectionType.HYBRID.value,
            }
        ]
        # Classifier/segmenter/physics still operate on one frame-global
        # appearance map.  Enrich exactly one localized appearance incident;
        # when several components emitted alerts, skipping is safer than
        # attaching all components' evidence to the first alert.
        appearance_alert = (
            appearance_alerts[0]
            if len(appearance_alerts) == 1
            else None
        )
        classified_alert = appearance_alert
        if classified_alert is not None and classification_result is not None:
            label, conf = classification_result
            classified_alert.classification_label = label
            classified_alert.classification_confidence = conf
            if label in self._classifier_high_risk:
                # Escalate: bump severity up one level
                self._adjust_alert_severity(classified_alert, escalate=True)
                classified_alert.severity_adjusted_by_classifier = True
                logger.info(
                    "pipeline.classifier_escalated",
                    camera_id=frame_data.camera_id,
                    label=label,
                    severity=classified_alert.severity.value,
                )
            elif label in self._classifier_low_risk and label not in self._classifier_suppress:
                # Suppress: downgrade severity one level
                self._adjust_alert_severity(classified_alert, escalate=False)
                classified_alert.severity_adjusted_by_classifier = True
                logger.info(
                    "pipeline.classifier_suppressed",
                    camera_id=frame_data.camera_id,
                    label=label,
                    severity=classified_alert.severity.value,
                )

        if frame_alerts:
            alert = max(
                frame_alerts,
                key=lambda candidate: (
                    _severity_rank(candidate.severity),
                    candidate.anomaly_score,
                ),
            )
            frame_alerts = [
                alert,
                *(candidate for candidate in frame_alerts if candidate is not alert),
            ]
            alert._secondary_zone_alerts = frame_alerts[1:]

        # D2: Instance segmentation on anomaly peaks
        if (
            appearance_alert is not None
            and self._segmenter is not None
            and (self._segmenter_config is None or self._segmenter_config.enabled is True)
            and "segmenter" not in self._auxiliary_load_blocked
            and appearance_decision_result.is_anomalous
            and appearance_decision_result.anomaly_score
            >= self._segmenter_min_score
            and appearance_decision_result.anomaly_map is not None
        ):
            try:
                from argus.anomaly.segmenter import extract_peak_points

                t_seg = time.monotonic()
                peak_points = extract_peak_points(
                    appearance_decision_result.anomaly_map,
                    max_points=self._segmenter_max_points,
                    min_score=self._segmenter_min_score,
                )
                if peak_points:
                    # Scale peak points from anomaly map to frame coordinates
                    map_h, map_w = appearance_decision_result.anomaly_map.shape[:2]
                    frame_h, frame_w = frame.shape[:2]
                    scaled_points = [
                        (int(px * frame_w / map_w), int(py * frame_h / map_h))
                        for px, py in peak_points
                    ]
                    seg_result = self._segmenter.segment(frame, scaled_points)
                    if getattr(seg_result, "degraded", False) is True:
                        raise RuntimeError("segmentation backend is degraded; evidence ignored")
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
                    self._auxiliary_errors.pop("segmenter", None)
                    if seg_result.num_objects > 0:
                        appearance_alert.segmentation_count = seg_result.num_objects
                        appearance_alert.segmentation_total_area_px = (
                            seg_result.total_area_px
                        )
                        appearance_alert.segmentation_objects = [
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
                self._record_auxiliary_failure("segmenter", e, diag, t_seg)
                logger.warning(
                    "pipeline.segmenter_error",
                    camera_id=frame_data.camera_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    consecutive_failures=self._segmenter_consecutive_failures,
                )

        self._update_latency(start)

        # Shadow runner: parallel scoring for release pipeline evaluation.
        # Hold the lock while the runner is used so release-stage swaps do not
        # flush an old runner that is still handling the current frame.
        with self._shadow_runner_lock:
            shadow_runner = self._shadow_runner
            if shadow_runner is not None:
                try:
                    shadow_runner.run_shadow(
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
        active_tracks_payload: list[dict] = []
        if appearance_alert is not None and appearance_decision_result.is_anomalous:
            try:
                # Reuse the temporal result computed up-front for F5
                # suppression; fall back to a fresh update only if the
                # pre-pass bailed out (e.g. exception path above).
                if pre_temporal is not None:
                    temporal = pre_temporal
                else:
                    regions = self._physics_postproc.extract_regions(
                        appearance_decision_result.anomaly_map,
                        threshold=0.3,
                    ) if appearance_decision_result.anomaly_map is not None else []
                    temporal = self._physics_tracker.update(regions, self._frame_counter)

                # Serialize active tracks for the ring buffer (drives replay trajectory)
                active_tracks_payload = [
                    {
                        "track_id": t.track_id,
                        "centroid_x": float(t.centroid_x),
                        "centroid_y": float(t.centroid_y),
                        "max_score": float(t.max_score),
                        "area_px": int(t.area_px),
                    }
                    for t in temporal.active_tracks
                ]

                # Per-track trajectory fits (used for persistence + API)
                trajectories_payload: list[dict] = []
                if (
                    self._trajectory_analyzer
                    and self._camera_calibration
                    and temporal.active_tracks
                ):
                    from argus.physics.trajectory import (
                        trajectory_points_from_pixel_history,
                    )
                    for track in temporal.active_tracks:
                        if len(track.trajectory_history) < 5:
                            continue
                        world_pts = trajectory_points_from_pixel_history(
                            track.trajectory_history,
                            self._camera_calibration,
                        )
                        fit = self._trajectory_analyzer.fit_trajectory(world_pts)
                        if fit is None:
                            continue
                        try:
                            ox, oy = self._camera_calibration.world_to_pixel(
                                fit.origin.x_mm, fit.origin.y_mm, fit.origin.z_mm,
                            )
                            lx, ly = self._camera_calibration.world_to_pixel(
                                fit.landing.x_mm, fit.landing.y_mm, fit.landing.z_mm,
                            )
                        except Exception:
                            ox = oy = lx = ly = None
                        per_track_speed = None
                        if self._speed_estimator:
                            try:
                                per_track_speed = self._speed_estimator.estimate(track)
                            except Exception:
                                per_track_speed = None
                        trajectories_payload.append({
                            "track_id": int(track.track_id),
                            "is_primary": False,
                            "model_type": fit.model_type,
                            "r_squared": float(fit.r_squared),
                            "speed_ms": (
                                float(per_track_speed.speed_ms)
                                if per_track_speed and per_track_speed.speed_ms is not None
                                else None
                            ),
                            "speed_px_per_sec": (
                                float(per_track_speed.speed_px_per_sec)
                                if per_track_speed else None
                            ),
                            "origin": {
                                "x_mm": float(fit.origin.x_mm),
                                "y_mm": float(fit.origin.y_mm),
                                "z_mm": float(fit.origin.z_mm),
                                "x_px": float(ox) if ox is not None else None,
                                "y_px": float(oy) if oy is not None else None,
                            },
                            "landing": {
                                "x_mm": float(fit.landing.x_mm),
                                "y_mm": float(fit.landing.y_mm),
                                "z_mm": float(fit.landing.z_mm),
                                "x_px": float(lx) if lx is not None else None,
                                "y_px": float(ly) if ly is not None else None,
                            },
                        })

                # Mark primary = track with highest max_score; mirror its scalars onto alert
                if trajectories_payload:
                    score_by_id = {
                        int(t.track_id): float(t.max_score)
                        for t in temporal.active_tracks
                    }
                    primary = max(
                        trajectories_payload,
                        key=lambda tr: score_by_id.get(tr["track_id"], 0.0),
                    )
                    primary["is_primary"] = True
                    appearance_alert.trajectory_model = primary["model_type"]
                    appearance_alert.origin_x_mm = primary["origin"]["x_mm"]
                    appearance_alert.origin_y_mm = primary["origin"]["y_mm"]
                    appearance_alert.origin_z_mm = primary["origin"]["z_mm"]
                    appearance_alert.landing_x_mm = primary["landing"]["x_mm"]
                    appearance_alert.landing_y_mm = primary["landing"]["y_mm"]
                    appearance_alert.landing_z_mm = primary["landing"]["z_mm"]
                    if primary["speed_ms"] is not None:
                        appearance_alert.speed_ms = primary["speed_ms"]
                    if primary["speed_px_per_sec"] is not None:
                        appearance_alert.speed_px_per_sec = primary[
                            "speed_px_per_sec"
                        ]
                    import json as _json
                    appearance_alert.trajectories_json = _json.dumps(
                        trajectories_payload
                    )
                elif self._speed_estimator and temporal.active_tracks:
                    # No fit available — still surface best-track speed
                    best_track = max(temporal.active_tracks, key=lambda t: t.max_score)
                    speed_est = self._speed_estimator.estimate(best_track)
                    appearance_alert.speed_px_per_sec = speed_est.speed_px_per_sec
                    appearance_alert.speed_ms = speed_est.speed_ms
            except Exception as e:
                logger.debug(
                    "pipeline.physics_enrichment_error",
                    camera_id=frame_data.camera_id,
                    error=str(e),
                )

        if alert is not None:
            # Persist trajectory centroid history for visualization.
            # Read-only access: calling update([]) here would double-advance
            # the stationary-suppress state machine and delete stale tracks.
            try:
                if (
                    hasattr(self, '_physics_tracker')
                    and self._physics_tracker is not None
                ):
                    histories = self._physics_tracker.get_trajectory_histories()
                    if histories and appearance_alert is not None:
                        appearance_alert.trajectory_points = histories[0]
            except Exception:
                pass

            for emitted_alert in frame_alerts:
                # Auto-categorize every alert produced by this frame.  The
                # method still returns the primary alert for API compatibility,
                # while secondary zone incidents are dispatched below.
                emitted_alert.category = self._categorize_alert(
                    emitted_alert,
                    simplex_result,
                    detection_result,
                    frame_shape=frame.shape,
                )

                # Requirements §3.2 №9: downgrade only a Simplex-only fallback.
                # Object safety, persistent change, water and plume are independent
                # channels and retain their own severity when appearance is degraded.
                raw_alert_detection_type = getattr(
                    emitted_alert,
                    "detection_type",
                    None,
                )
                alert_detection_type = getattr(
                    raw_alert_detection_type,
                    "value",
                    raw_alert_detection_type,
                )
                alert_has_object_safety = any(
                    obj.get("object_safety_alert_contributed") is True
                    for obj in (
                        getattr(emitted_alert, "detected_objects", []) or []
                    )
                )
                alert_has_independent_visual = any(
                    str(obj.get("class_name") or "").lower()
                    in {
                        "static_foreign",
                        "visible_water_leak",
                        "visible_plume",
                    }
                    for obj in (
                        getattr(emitted_alert, "detected_objects", []) or []
                    )
                )
                independent_alert_channel = bool(
                    alert_has_object_safety
                    or alert_has_independent_visual
                    or alert_detection_type == DetectionType.PROJECTILE.value
                )
                if (
                    self._anomaly_degraded
                    and simplex_detected
                    and not independent_alert_channel
                ):
                    from argus.alerts.grader import AlertSeverity, HANDLING_POLICIES

                    if emitted_alert.severity == AlertSeverity.HIGH:
                        emitted_alert.severity = AlertSeverity.MEDIUM
                        emitted_alert.handling_policy = HANDLING_POLICIES.get(
                            AlertSeverity.MEDIUM,
                            emitted_alert.handling_policy,
                        )
                        logger.info(
                            "pipeline.alert_severity_degraded",
                            camera_id=frame_data.camera_id,
                            original="high",
                            adjusted="medium",
                            reason="anomaly_head_degraded",
                        )

                emitted_alert.model_version_id = self._model_version_id
            self.stats.alerts_emitted += len(frame_alerts)

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
        for emitted_alert in frame_alerts:
            METRICS.alerts_total.labels(
                camera_id=cam_id,
                severity=emitted_alert.severity.value,
            ).inc()

        # Use the score that actually reached the grader for frame-level
        # diagnostics.  On a person frame the appearance model may be zeroed
        # while an explicitly allowlisted object supplies the safety score.
        grade_result = getattr(self, "_last_grade_result", None)
        decision_score = appearance_model_score
        if alert is not None and isinstance(alert.anomaly_score, int | float):
            decision_score = float(alert.anomaly_score)
        else:
            graded_score = getattr(grade_result, "score", None)
            if isinstance(graded_score, int | float) and math.isfinite(graded_score):
                decision_score = float(graded_score)
        decision_signal_active = bool(
            anomaly_result.is_anomalous
            or alert is not None
            or getattr(grade_result, "severity", None) is not None
        )
        diag.anomaly_score = decision_score
        diag.is_anomalous = decision_signal_active

        # Event bus (skip allocation if no subscribers)
        if self._event_bus is not None and self._event_bus.has_subscribers(FrameAnalyzed):
            # Buffer frame for active learning sampler before publishing
            _al_sampler = getattr(self._event_bus, '_active_sampler', None)
            if _al_sampler is not None:
                _al_sampler.buffer_frame(cam_id, frame_data.frame_number, frame_data.frame)
            self._event_bus.publish(FrameAnalyzed(
                camera_id=cam_id,
                frame_number=frame_data.frame_number,
                anomaly_score=decision_score,
                is_anomalous=decision_signal_active,
                stage_latencies=MappingProxyType({s.stage_name: s.duration_ms for s in diag.stages}),
                mog2_skipped=mog2_skipped,
                person_detected=detection_result.has_persons,
            ))
        if (
            self._event_bus is not None
            and frame_alerts
            and self._event_bus.has_subscribers(AlertRaised)
        ):
            for emitted_alert in frame_alerts:
                self._event_bus.publish(AlertRaised(
                    alert_id=emitted_alert.alert_id,
                    camera_id=cam_id,
                    zone_id=emitted_alert.zone_id,
                    severity=emitted_alert.severity,
                    anomaly_score=emitted_alert.anomaly_score,
                    handling_policy=emitted_alert.handling_policy,
                ))

        # 5.2: Build InferenceRecord only for non-NORMAL frames (avoid
        # uuid4/time_ns/dict-comprehension overhead on every frame)
        if alert is not None:
            final_decision = FinalDecision.ALERT
        elif current_mode == PipelineMode.LEARNING and decision_signal_active:
            final_decision = FinalDecision.SUPPRESSED
        elif decision_signal_active:
            final_decision = FinalDecision.INFO
        else:
            final_decision = FinalDecision.NORMAL

        candidate_cusum_evidence: dict = {}
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
            candidate_cusum_evidence = cusum_evidence

            self._last_inference_record = InferenceRecord(
                frame_id=frame_id,
                camera_id=frame_data.camera_id,
                timestamp_ns=time.time_ns(),
                model_version=self._model_version_id or "",
                prefilter_result=prefilter_decision,
                anomaly_score=decision_score,
                cusum_evidence=cusum_evidence,
                conformal_level=cal_level,
                safety_channel_result=simplex_detected if self._simplex is not None else None,
                final_decision=final_decision,
                stage_durations_ms={s.stage_name: s.duration_ms for s in diag.stages},
            )
        else:
            self._last_inference_record = None

        final_suppression_reason = None
        if final_decision == FinalDecision.SUPPRESSED:
            final_suppression_reason = "pipeline_mode_suppressed"
        elif final_decision == FinalDecision.INFO and alert is None:
            grade_result = getattr(self, "_last_grade_result", None)
            final_suppression_reason = (
                getattr(grade_result, "suppression_reason", None)
                or "grader_no_alert"
            )
        diag.final_decision = final_decision.value
        final_evidence = {
            "pipeline_mode": current_mode.value,
            "anomaly_score": round(decision_score, 4),
            "appearance_model_score": round(appearance_model_score, 4),
            "is_anomalous": decision_signal_active,
            "alert_emitted": alert is not None,
            "alert_count": len(frame_alerts),
            "detection_type": getattr(detection_type, "value", str(detection_type)),
            "detected_objects": detected_objects,
            "person_count": len(detection_result.persons),
            "object_count": len(detection_result.objects),
            "simplex_detected": simplex_detected,
            "mog2_skipped": mog2_skipped,
            "prefilter_bypassed": skip_prefilter,
        }
        if candidate_cusum_evidence:
            final_evidence["cusum_evidence"] = candidate_cusum_evidence
        if classification_result is not None:
            final_evidence["classification"] = {
                "label": classification_result[0],
                "confidence": round(classification_result[1], 3),
            }
        if persistent_change_result is not None:
            final_evidence["persistent_change"] = persistent_change_result.evidence()
        if water_leak_result is not None:
            final_evidence["water_leak"] = water_leak_result.evidence()
        if plume_result is not None:
            final_evidence["plume"] = plume_result.evidence()
        if frame_alerts:
            alert_summaries = [
                {
                    "alert_id": emitted_alert.alert_id,
                    "severity": getattr(
                        emitted_alert.severity,
                        "value",
                        emitted_alert.severity,
                    ),
                    "zone_id": emitted_alert.zone_id,
                    "detection_type": getattr(
                        emitted_alert.detection_type,
                        "value",
                        emitted_alert.detection_type,
                    ),
                }
                for emitted_alert in frame_alerts
            ]
            final_evidence["alert"] = alert_summaries[0]
            final_evidence["alerts"] = alert_summaries
            for emitted_alert, alert_summary in zip(
                frame_alerts,
                alert_summaries,
                strict=True,
            ):
                # Alert evidence is causal and incident-local. Full-frame head
                # results remain in FrameDiagnostics only; shallow-copying
                # them here would attach sibling water/plume/static candidates
                # to an unrelated alert and corrupt replay/report attribution.
                per_alert_pipeline_evidence = {
                    "pipeline_mode": current_mode.value,
                    "alert": alert_summary,
                    "alert_emitted": True,
                    "alert_count": 1,
                    "anomaly_score": round(emitted_alert.anomaly_score, 4),
                    "is_anomalous": True,
                    "detection_type": getattr(
                        emitted_alert.detection_type,
                        "value",
                        emitted_alert.detection_type,
                    ),
                    "detected_objects": list(
                        getattr(emitted_alert, "detected_objects", []) or []
                    ),
                    "zone_id": emitted_alert.zone_id,
                    "frame_number": frame_data.frame_number,
                    "model_version_id": self._model_version_id,
                }
                alert_evidence = dict(
                    getattr(emitted_alert, "evidence_json", {}) or {}
                )
                alert_evidence["model_version_id"] = getattr(
                    emitted_alert,
                    "model_version_id",
                    None,
                )
                alert_evidence["category"] = getattr(
                    emitted_alert,
                    "category",
                    None,
                )
                alert_evidence["localized_incidents"] = [
                    {
                        "class_name": detected_object.get("class_name"),
                        "bbox": detected_object.get("bbox"),
                        "score": (
                            detected_object.get("independent_visual_score")
                            if detected_object.get(
                                "independent_visual_alert_eligible"
                            )
                            is True
                            else detected_object.get("object_safety_score")
                        ),
                    }
                    for detected_object in (
                        getattr(emitted_alert, "detected_objects", []) or []
                    )
                    if (
                        detected_object.get(
                            "independent_visual_alert_eligible"
                        )
                        is True
                        or detected_object.get(
                            "object_safety_alert_contributed"
                        )
                        is True
                    )
                ]
                features = dict(alert_evidence.get("features") or {})
                features["pipeline"] = per_alert_pipeline_evidence
                alert_evidence["features"] = features
                emitted_alert.evidence_json = alert_evidence
                if not getattr(emitted_alert, "evidence_summary", ""):
                    emitted_alert.evidence_summary = (
                        f"{emitted_alert.detection_type}, "
                        f"severity={emitted_alert.severity.value}, "
                        f"score={emitted_alert.anomaly_score:.4f}, "
                        f"zone={emitted_alert.zone_id}, "
                        f"frame={frame_data.frame_number}"
                    )
        diag.add_candidate(
            candidate_type="frame_anomaly",
            decision=final_decision.value,
            stage="decision",
            evidence=final_evidence,
            suppression_reason=final_suppression_reason,
        )

        self._diagnostics.append(diag)
        self._diagnostics.append_score(FrameScoreRecord(
            frame_number=frame_data.frame_number,
            timestamp=time.time(),
            anomaly_score=decision_score,
            was_alert=alert is not None,
        ))

        # FR-033: Append frame to ring buffer + sparkline
        self._score_sparkline.append(decision_score)
        diagnostic_result = _updated_anomaly_result(
            anomaly_result,
            anomaly_score=decision_score,
            is_anomalous=decision_signal_active,
        )
        candidate_record_payloads = [
            record.to_dict() for record in diag.candidate_records
        ]
        if frame_alerts:
            for emitted_alert in frame_alerts:
                emitted_result = _updated_anomaly_result(
                    diagnostic_result,
                    anomaly_score=emitted_alert.anomaly_score,
                    anomaly_map=emitted_alert.heatmap,
                    is_anomalous=True,
                )
                self._handle_ring_buffer(
                    frame_data,
                    frame,
                    emitted_result,
                    detection_result,
                    simplex_result,
                    simplex_detected,
                    emitted_alert,
                    (
                        active_tracks_payload
                        if emitted_alert is appearance_alert
                        else []
                    ),
                    candidate_records=candidate_record_payloads,
                )
        else:
            self._handle_ring_buffer(
                frame_data,
                frame,
                diagnostic_result,
                detection_result,
                simplex_result,
                simplex_detected,
                None,
                active_tracks_payload,
                candidate_records=candidate_record_payloads,
            )

        self._last_emitted_alerts = list(frame_alerts)
        for emitted_alert in frame_alerts:
            if self._on_alert:
                self._on_alert(emitted_alert)
            self._persist_alert_recording(emitted_alert)

        return alert

    def _handle_fast_motion_detection(
        self,
        *,
        frame_data: FrameData,
        frame: np.ndarray,
        start: float,
        diag: FrameDiagnostics,
        fast_result: FastMotionResult,
    ) -> Alert | None:
        """Grade a fast fly-through detection through the normal alert path."""
        cfg = self.camera_config.fast_motion
        score = min(1.0, max(float(cfg.instant_alert_score), fast_result.max_confidence))
        anomaly_result = AnomalyResult(
            anomaly_score=score,
            anomaly_map=fast_result.anomaly_map,
            is_anomalous=True,
            threshold=float(cfg.min_confidence),
            detection_failed=False,
        )
        detected_objects = []
        for candidate in fast_result.candidates:
            payload = candidate.to_detected_object()
            payload["projectile_alert_score"] = min(
                1.0,
                max(
                    float(cfg.instant_alert_score),
                    float(candidate.confidence),
                ),
            )
            detected_objects.append(payload)

        objects: list[ObjectDetection] = []
        active_tracks_payload: list[dict] = []
        for candidate in fast_result.candidates:
            x1, y1, x2, y2 = candidate.bbox
            objects.append(ObjectDetection(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                confidence=float(candidate.confidence),
                class_id=-1,
                class_name="fast_projectile",
                track_id=candidate.track_id,
            ))
            active_tracks_payload.append({
                "track_id": int(candidate.track_id or 0),
                "centroid_x": float(candidate.centroid[0]),
                "centroid_y": float(candidate.centroid[1]),
                "max_score": float(candidate.confidence),
                "area_px": int(candidate.area_px),
                "speed_px_per_sec": (
                    float(candidate.speed_px_per_sec)
                    if candidate.speed_px_per_sec is not None
                    else None
                ),
            })

        detection_result = ObjectDetectionResult(
            persons=[],
            has_persons=False,
            objects=objects,
            has_objects=bool(objects),
            non_person_objects=objects,
            filter_available=True,
        )

        self.stats.frames_analyzed += 1
        self.stats.anomalies_detected += 1
        diag.anomaly_score = anomaly_result.anomaly_score
        diag.is_anomalous = True

        with self._latest_anomaly_map_lock:
            self._latest_anomaly_map = (
                anomaly_result.anomaly_map.copy()
                if anomaly_result.anomaly_map is not None
                else None
            )

        alert = self._evaluate_zones(
            frame_data,
            anomaly_result,
            frame,
            detection_type=DetectionType.PROJECTILE,
            detected_objects=detected_objects,
            # Heavyweight appearance/YOLO and the other independent heads are
            # intentionally skipped on this fast path. Only projectile
            # evidence was observed, so no other channel may decay here.
            decay_detection_types={DetectionType.PROJECTILE.value},
        )
        frame_alerts = (
            [
                alert,
                *(getattr(alert, "_secondary_zone_alerts", None) or []),
            ]
            if alert is not None
            else []
        )

        if alert is not None:
            best = max(fast_result.candidates, key=lambda item: item.confidence)
            candidates_by_track = {
                str(candidate.track_id): candidate
                for candidate in fast_result.candidates
                if candidate.track_id is not None
            }
            for emitted_alert in frame_alerts:
                alert_track_id = next(
                    (
                        item.get("track_id")
                        for item in emitted_alert.detected_objects
                        if item.get("track_id") is not None
                    ),
                    None,
                )
                incident_candidate = candidates_by_track.get(
                    str(alert_track_id),
                    best,
                )
                emitted_alert.category = "projectile"
                emitted_alert.trajectory_model = "projectile"
                emitted_alert.model_version_id = self._model_version_id
                emitted_alert.trajectory_points = incident_candidate.trajectory or [(
                    time.time(),
                    float(incident_candidate.centroid[0]),
                    float(incident_candidate.centroid[1]),
                )]
                if incident_candidate.speed_px_per_sec is not None:
                    emitted_alert.speed_px_per_sec = float(
                        incident_candidate.speed_px_per_sec
                    )
                alert_evidence = dict(
                    getattr(emitted_alert, "evidence_json", {}) or {}
                )
                alert_evidence["model_version_id"] = getattr(
                    emitted_alert,
                    "model_version_id",
                    None,
                )
                alert_evidence["category"] = emitted_alert.category
                features = dict(alert_evidence.get("features") or {})
                features["projectile"] = {
                    "confidence": float(incident_candidate.confidence),
                    "track_id": incident_candidate.track_id,
                    "speed_px_per_sec": (
                        float(incident_candidate.speed_px_per_sec)
                        if incident_candidate.speed_px_per_sec is not None
                        else None
                    ),
                }
                alert_evidence["features"] = features
                emitted_alert.evidence_json = alert_evidence
            self.stats.alerts_emitted += len(frame_alerts)

        self._update_latency(start)
        diag.alert_emitted = alert is not None
        diag.total_duration_ms = (time.monotonic() - start) * 1000
        self._last_detection_failed = False

        cam_id = frame_data.camera_id
        self._m_latency.observe(diag.total_duration_ms / 1000.0)
        self._m_frames.inc()
        for stage in diag.stages:
            metric = self._m_stages.get(stage.stage_name) or self._m_stages_fallback(stage.stage_name)
            metric.observe(stage.duration_ms / 1000.0)
        self._m_anomaly_score.set(anomaly_result.anomaly_score)
        for emitted_alert in frame_alerts:
            METRICS.alerts_total.labels(
                camera_id=cam_id,
                severity=emitted_alert.severity.value,
            ).inc()

        if self._event_bus is not None and self._event_bus.has_subscribers(FrameAnalyzed):
            self._event_bus.publish(FrameAnalyzed(
                camera_id=cam_id,
                frame_number=frame_data.frame_number,
                anomaly_score=anomaly_result.anomaly_score,
                is_anomalous=True,
                stage_latencies=MappingProxyType({
                    stage.stage_name: stage.duration_ms for stage in diag.stages
                }),
                mog2_skipped=True,
                person_detected=False,
            ))
        if (
            self._event_bus is not None
            and frame_alerts
            and self._event_bus.has_subscribers(AlertRaised)
        ):
            for emitted_alert in frame_alerts:
                self._event_bus.publish(AlertRaised(
                    alert_id=emitted_alert.alert_id,
                    camera_id=cam_id,
                    zone_id=emitted_alert.zone_id,
                    severity=emitted_alert.severity,
                    anomaly_score=emitted_alert.anomaly_score,
                    handling_policy=emitted_alert.handling_policy,
                ))

        frame_id = uuid.uuid4().hex
        diag.frame_id = frame_id
        final_decision = FinalDecision.ALERT if alert is not None else FinalDecision.INFO
        diag.final_decision = final_decision.value
        fast_evidence = {
            "pipeline_mode": self.mode.value,
            "anomaly_score": round(anomaly_result.anomaly_score, 4),
            "max_confidence": round(fast_result.max_confidence, 4),
            "candidate_count": len(fast_result.candidates),
            "detected_objects": detected_objects,
            "alert_emitted": alert is not None,
            "alert_count": len(frame_alerts),
        }
        if alert is not None:
            fast_evidence["alert"] = {
                "alert_id": getattr(alert, "alert_id", None),
                "severity": getattr(
                    getattr(alert, "severity", None),
                    "value",
                    getattr(alert, "severity", None),
                ),
                "zone_id": getattr(alert, "zone_id", None),
            }
            fast_evidence["alerts"] = [
                {
                    "alert_id": emitted_alert.alert_id,
                    "severity": getattr(
                        emitted_alert.severity,
                        "value",
                        emitted_alert.severity,
                    ),
                    "zone_id": emitted_alert.zone_id,
                }
                for emitted_alert in frame_alerts
            ]
        grade_result = getattr(self, "_last_grade_result", None)
        diag.add_candidate(
            candidate_type="fast_motion",
            decision=final_decision.value,
            stage="decision",
            evidence=fast_evidence,
            suppression_reason=(
                None if alert is not None
                else getattr(grade_result, "suppression_reason", None) or "grader_no_alert"
            ),
        )
        self._last_inference_record = InferenceRecord(
            frame_id=frame_id,
            camera_id=frame_data.camera_id,
            timestamp_ns=time.time_ns(),
            model_version=self._model_version_id or "fast_motion",
            prefilter_result=PrefilterDecision.PASSED,
            anomaly_score=anomaly_result.anomaly_score,
            cusum_evidence={},
            conformal_level=ConformalLevel.HIGH,
            safety_channel_result=True,
            final_decision=final_decision,
            stage_durations_ms={stage.stage_name: stage.duration_ms for stage in diag.stages},
        )

        self._diagnostics.append(diag)
        self._diagnostics.append_score(FrameScoreRecord(
            frame_number=frame_data.frame_number,
            timestamp=time.time(),
            anomaly_score=anomaly_result.anomaly_score,
            was_alert=alert is not None,
        ))
        self._score_sparkline.append(anomaly_result.anomaly_score)
        candidate_record_payloads = [
            record.to_dict() for record in diag.candidate_records
        ]
        if frame_alerts:
            for emitted_alert in frame_alerts:
                emitted_track_ids = {
                    int(item["track_id"])
                    for item in emitted_alert.detected_objects
                    if isinstance(item.get("track_id"), int)
                }
                emitted_result = _updated_anomaly_result(
                    anomaly_result,
                    anomaly_score=emitted_alert.anomaly_score,
                    anomaly_map=emitted_alert.heatmap,
                    is_anomalous=True,
                )
                self._handle_ring_buffer(
                    frame_data,
                    frame,
                    emitted_result,
                    detection_result,
                    None,
                    False,
                    emitted_alert,
                    [
                        track
                        for track in active_tracks_payload
                        if int(track["track_id"]) in emitted_track_ids
                    ],
                    candidate_records=candidate_record_payloads,
                )
        else:
            self._handle_ring_buffer(
                frame_data,
                frame,
                anomaly_result,
                detection_result,
                None,
                False,
                None,
                active_tracks_payload,
                candidate_records=candidate_record_payloads,
            )

        self._last_emitted_alerts = list(frame_alerts)
        for emitted_alert in frame_alerts:
            if self._on_alert:
                self._on_alert(emitted_alert)
            self._persist_alert_recording(emitted_alert)

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
        active_tracks: list[dict] | None = None,
        candidate_records: list[dict] | None = None,
    ) -> None:
        """Ring buffer frame append, post-capture flush, and alert solidify."""
        if alert is not None:
            alert._has_recording = False
            alert._replay_ready = False
            alert._recording_status = None
            alert._recording_unavailable_reason = None
            alert._recording_evidence_unavailable = False

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

            snap_ts = float(alert.timestamp) if alert is not None else time.time()
            snap_score = anomaly_result.anomaly_score
            snap_simplex = getattr(simplex_result, "max_score", None) if simplex_detected else None
            snap_frame_num = frame_data.frame_number
            quality = self._ring_buffer_jpeg_quality
            rb = self._alert_ring_buffer

            rb_active_tracks = list(active_tracks) if active_tracks else []
            rb_candidate_records = list(candidate_records) if candidate_records else []

            def _encode_and_append() -> None:
                snap = FrameSnapshot(
                    timestamp=snap_ts,
                    frame_jpeg=compress_frame(frame, quality=quality),
                    anomaly_score=snap_score,
                    simplex_score=snap_simplex,
                    cusum_evidence=rb_cusum,
                    yolo_persons=rb_persons,
                    frame_number=snap_frame_num,
                    heatmap_raw=rb_heatmap_raw,
                    yolo_boxes=rb_all_boxes,
                    active_tracks=rb_active_tracks,
                    identity=frame_data.identity,
                    candidate_records=rb_candidate_records,
                )
                rb.append(snap)

            if alert is not None:
                _encode_and_append()
            else:
                self._encode_executor.submit(_encode_and_append)
        except Exception:
            logger.debug(
                "pipeline.ring_buffer_append_failed",
                camera_id=frame_data.camera_id,
                exc_info=True,
            )

        self._flush_expired_post_captures()

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
                    alert._solidified_recording = recording
                    alert._recording_status = recording.status.value
                else:
                    alert._recording_evidence_unavailable = True
                    alert._recording_unavailable_reason = "ring_buffer_empty"
            except Exception:
                alert._recording_evidence_unavailable = True
                alert._recording_unavailable_reason = "ring_buffer_solidify_failed"
                logger.warning(
                    "pipeline.ring_buffer_solidify_failed",
                    camera_id=frame_data.camera_id,
                    alert_id=alert.alert_id,
                    exc_info=True,
                )

    def _persist_alert_recording(self, alert: Alert) -> None:
        """Persist replay artifacts after the primary alert evidence is dispatched."""
        recording = getattr(alert, "_solidified_recording", None)
        if recording is None:
            return

        if self._recording_store is None:
            alert._recording_evidence_unavailable = True
            alert._recording_unavailable_reason = "recording_store_unavailable"
            return

        rec_path = ""
        file_size = 0
        try:
            rec_path, file_size = self._recording_store.save(recording)
            logger.info(
                "pipeline.recording_saved",
                alert_id=alert.alert_id,
                path=rec_path,
            )
        except Exception:
            alert._has_recording = False
            alert._replay_ready = False
            alert._recording_evidence_unavailable = True
            alert._recording_unavailable_reason = "recording_save_failed"
            logger.warning(
                "pipeline.recording_save_failed",
                alert_id=alert.alert_id,
                exc_info=True,
            )
            return

        if rec_path:
            alert._has_recording = True
            alert._replay_ready = True
            alert._recording_status = recording.status.value
            alert._recording_unavailable_reason = None

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

    def run_once(self) -> Alert | None:
        """Read one frame from the camera and process it.

        Uses ``read_latest()`` when a capture thread is running to always
        process the most recent frame (avoids slow-motion from buffer
        accumulation).
        """
        with self._scene_processing_lock:
            if self._auxiliary_shutdown:
                self._set_frame_outcome(FrameOutcomeKind.BLOCKED, "pipeline is stopped")
                return None
        frame_data = self._camera.read_latest()
        with self._scene_processing_lock:
            if self._auxiliary_shutdown:
                self._set_frame_outcome(FrameOutcomeKind.BLOCKED, "pipeline is stopped")
                return None
        if frame_data is None:
            self._flush_expired_post_captures()
            if not self._camera.state.connected:
                self._camera.request_reconnect()
            return None
        return self.process_frame(frame_data)

    def _flush_expired_post_captures(self) -> None:
        """Finalize expired alert recordings, including pre-only recordings."""
        if self._alert_ring_buffer is None:
            return

        try:
            expired_ids = self._alert_ring_buffer.check_expired_captures()
        except Exception:
            logger.debug("pipeline.post_capture_check_failed", exc_info=True)
            return

        for expired_id in expired_ids:
            try:
                post_frames = self._alert_ring_buffer.finish_post_capture(expired_id)
                if self._recording_store is None:
                    continue

                if not self._recording_store.append_post_frames(expired_id, post_frames):
                    continue

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
                logger.debug(
                    "pipeline.post_capture_flush_failed",
                    alert_id=expired_id,
                    exc_info=True,
                )

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

    def get_active_tracks_snapshot(self) -> list[dict]:
        """Return the physics track state without exposing tracker internals."""
        tracker = getattr(self, "_physics_tracker", None)
        getter = getattr(tracker, "get_active_tracks_snapshot", None)
        if not callable(getter):
            return []
        tracks = getter()
        return tracks if isinstance(tracks, list) else []

    def get_model_statuses_snapshot(self) -> list[dict]:
        """Return operator-facing detector health as detached dictionaries."""
        result: list[dict] = []
        anomaly = getattr(self, "_anomaly_detector", None)
        anomaly_status = getattr(anomaly, "status", None)
        if anomaly_status is not None and hasattr(anomaly_status, "to_dict"):
            record = anomaly_status.to_dict()
            try:
                quality = self.get_input_quality_status()
            except Exception:
                logger.debug(
                    "pipeline.input_quality_status_snapshot_failed",
                    camera_id=self.camera_config.camera_id,
                    exc_info=True,
                )
            else:
                extra = dict(record.get("extra") or {})
                extra["input_quality"] = quality
                record["extra"] = extra
            result.append(record)

        object_detector = getattr(self, "_object_detector", None)
        object_status = getattr(object_detector, "status", None)
        if object_status is not None and hasattr(object_status, "to_dict"):
            result.append(object_status.to_dict())
        return result

    def _record_auxiliary_failure(
        self, name: str, exc: Exception, diag: FrameDiagnostics, started: float,
    ) -> None:
        reason = f"{type(exc).__name__}: {exc}"
        self._auxiliary_errors[name] = reason
        diag.stages.append(StageResult(
            stage_name=name,
            duration_ms=max(0.0, (time.monotonic() - started) * 1000),
            metadata={"unavailable": True, "error": reason},
            skipped=True,
            skip_reason="auxiliary_unavailable",
        ))
        diag.add_candidate(
            candidate_type=name, decision="unavailable", stage=name,
            evidence={"error": reason, "independent_channels_continued": True},
            suppression_reason="auxiliary_unavailable",
        )

    @staticmethod
    def _stationary_evidence_is_current(score_map, regions, tracks, frame_number: int) -> bool:
        """Historical/filtered-out tracks are never negative evidence for a new frame."""
        if not isinstance(score_map, np.ndarray) or score_map.ndim != 2 or not regions:
            return False
        if score_map.size == 0 or not np.all(np.isfinite(score_map)) or len(regions) != len(tracks):
            return False
        low, high = float(score_map.min()), float(score_map.max())
        if high - low <= 1e-6 or high <= 0.3:
            return False
        support = score_map >= 0.3
        covered = np.zeros(score_map.shape, dtype=bool)
        remaining = list(tracks)
        try:
            for region in regions:
                left, top = int(region.x), int(region.y)
                right, bottom = left + int(region.width), top + int(region.height)
                if not (0 <= left < right <= score_map.shape[1] and 0 <= top < bottom <= score_map.shape[0]):
                    return False
                match = next((track for track in remaining if (
                    track.suppressed is True and track.last_seen_frame == frame_number
                    and math.isclose(track.centroid_x, region.centroid_x, abs_tol=1e-6)
                    and math.isclose(track.centroid_y, region.centroid_y, abs_tol=1e-6)
                )), None)
                if match is None:
                    return False
                remaining.remove(match)
                covered[top:bottom, left:right] = True
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False
        return bool(np.any(support) and np.all(covered[support]))

    def _auxiliary_status(self, name: str, component: object | None) -> dict:
        status = {
            "state": "disabled" if component is None else "configured",
            "backend": None, "loaded": False, "ready": False,
            "degraded": False, "reason": None, "busy": False,
        }
        getter = getattr(component, "get_status", None)
        if callable(getter):
            try:
                value = getter()
                if isinstance(value, dict):
                    status.update(value)
            except Exception as exc:
                status.update(state="failed", degraded=True, reason=str(exc))
        error = getattr(self, "_auxiliary_errors", {}).get(name)
        if error:
            status.update(state="failed", ready=False, degraded=True, reason=error)
        retired = getattr(self, "_auxiliary_retired", {}).get(name)
        if component is None and retired is not None:
            try:
                retired_status = retired.get_status()
                draining = retired_status.get("busy") is not False
                stopped = retired_status.get("state") == "stopped" and not draining
            except Exception:
                draining = True
                stopped = False
            status.update(
                state="disabled" if stopped and not error else "draining" if draining else "failed",
                busy=draining, ready=False, degraded=not stopped or bool(error),
                reason=error or ("SDK shutdown is not confirmed" if not stopped else None),
            )
        return status

    def get_system_status_snapshot(self) -> dict:
        """Return detached System-page diagnostics and component readiness."""
        drift = {
            "is_drifted": False,
            "ks_statistic": 0,
            "p_value": 1,
            "reference_mean": 0,
            "current_mean": 0,
            "samples_collected": 0,
            "last_check_time": 0,
        }
        drift_detector = getattr(self, "_drift_detector", None)
        drift_getter = getattr(drift_detector, "get_status", None)
        if callable(drift_getter):
            status = drift_getter()
            drift = {
                "is_drifted": status.is_drifted,
                "ks_statistic": round(status.ks_statistic, 4),
                "p_value": round(status.p_value, 6),
                "reference_mean": round(status.reference_mean, 4),
                "current_mean": round(status.current_mean, 4),
                "samples_collected": status.samples_collected,
                "last_check_time": status.last_check_time,
            }

        camera_health = {
            "is_frozen": False,
            "sharpness_score": 0,
            "displacement_px": 0,
            "is_flash": False,
            "gain_drift_pct": 0,
            "suppress_detection": False,
            "warnings": [],
        }
        health_result = getattr(
            getattr(self, "_health_analyzer", None),
            "last_result",
            None,
        )
        if health_result is not None:
            camera_health = {
                "is_frozen": health_result.is_frozen,
                "sharpness_score": round(health_result.sharpness_score, 2),
                "displacement_px": round(health_result.displacement_px, 1),
                "is_flash": health_result.is_flash,
                "gain_drift_pct": round(health_result.gain_drift_pct, 1),
                "suppress_detection": health_result.suppress_detection,
                "warnings": list(health_result.warnings),
            }

        ensemble = {
            "model_count": 0,
            "loaded": False,
            "method": "none",
            "model_paths": [],
        }
        ensemble_getter = getattr(
            getattr(self, "_ensemble_detector", None),
            "get_status",
            None,
        )
        if callable(ensemble_getter):
            maybe_ensemble = ensemble_getter()
            if isinstance(maybe_ensemble, dict):
                ensemble = dict(maybe_ensemble)

        classifier = getattr(self, "_classifier", None)
        segmenter = getattr(self, "_segmenter", None)
        classifier_status = self._auxiliary_status("classifier", classifier)
        segmenter_status = self._auxiliary_status("segmenter", segmenter)
        return {
            "drift": drift,
            "camera_health": camera_health,
            "ensemble": ensemble,
            "anomaly_degradation": {
                "degraded": bool(self.is_anomaly_degraded()),
                "reason": self.get_anomaly_degradation_reason(),
                "since": self.get_anomaly_degradation_started_at(),
            },
            "components": {
                "classifier_attached": classifier is not None,
                "classifier_loaded": bool(
                    classifier_status.get("loaded") is True
                ),
                "segmenter_attached": segmenter is not None,
                "segmenter_loaded": bool(
                    segmenter_status.get("loaded") is True
                ),
                "classifier": classifier_status,
                "segmenter": segmenter_status,
                "auxiliary_failures": dict(getattr(self, "_auxiliary_errors", {})),
                "imaging_attached": getattr(self, "_imaging_processor", None) is not None,
            },
        }

    def get_runtime_model_state(self) -> dict:
        """Return model versions applied to the primary and shadow slots."""
        with self._scene_processing_lock:
            primary_version = self._model_version_id
            shadow_version = self.get_shadow_runner_version()
        return {
            "primary_model_version": primary_version,
            "shadow_model_version": shadow_version,
        }

    def get_anomaly_backend_status(self) -> dict:
        """Return the configured and actually loaded anomaly backends."""
        expected = self._anomaly_inference_backend
        if not isinstance(expected, str):
            expected = getattr(expected, "value", None)
        if not isinstance(expected, str):
            expected = "auto"

        anomaly = getattr(self, "_anomaly_detector", None)
        status = getattr(anomaly, "status", None)
        if status is None and hasattr(anomaly, "_base"):
            status = getattr(getattr(anomaly, "_base", None), "status", None)
        actual = getattr(status, "backend", None)
        if not isinstance(actual, str):
            actual = getattr(actual, "value", None)
        if not isinstance(actual, str):
            actual = None
        return {"actual": actual, "expected": expected}

    def get_anomaly_inference_handle(self) -> object | None:
        """Return the opaque detector used by explicit operator inference jobs.

        Dashboard routes should not discover detector fields themselves.  The
        returned handle is intentionally narrow in usage: callers may invoke
        its documented inference methods but must not mutate pipeline state.
        """
        return self._anomaly_detector

    def get_capture_status(self) -> dict:
        """Return capture and fast-motion runtime metadata for dashboard status."""
        capture = (
            self._camera.runtime_status()
            if hasattr(self._camera, "runtime_status")
            else {}
        )
        fast_cfg = self.camera_config.fast_motion
        stats_fps = self.stats.snapshot().get("current_fps")
        runtime_fps = stats_fps if stats_fps and stats_fps > 0 else capture.get("actual_fps")
        actual_resolution = capture.get("actual_resolution")
        fast_degraded = False
        reasons: list[str] = []
        if fast_cfg.enabled:
            if actual_resolution:
                actual_tuple = tuple(int(v) for v in actual_resolution)
                required_tuple = tuple(int(v) for v in fast_cfg.required_resolution)
                if actual_tuple != required_tuple:
                    fast_degraded = True
                    reasons.append(
                        f"resolution {actual_tuple[0]}x{actual_tuple[1]} "
                        f"below required {required_tuple[0]}x{required_tuple[1]}"
                    )
            if runtime_fps and runtime_fps < fast_cfg.min_runtime_fps:
                fast_degraded = True
                reasons.append(
                    f"fps {runtime_fps:.1f} below required {fast_cfg.min_runtime_fps:.1f}"
                )

        return {
            **capture,
            "degraded": bool(capture.get("degraded")) or fast_degraded,
            "degradation_reason": capture.get("degradation_reason") or "; ".join(reasons) or None,
            "fast_motion": {
                "enabled": fast_cfg.enabled,
                "active": fast_cfg.enabled and self.mode in (
                    PipelineMode.ACTIVE,
                    PipelineMode.MAINTENANCE,
                ),
                "process_width": fast_cfg.process_width,
                "max_motion_fraction": fast_cfg.max_motion_fraction,
                "max_streak_frame_fraction": fast_cfg.max_streak_frame_fraction,
                "streak_gap_px": fast_cfg.streak_gap_px,
                "min_runtime_fps": fast_cfg.min_runtime_fps,
                "required_resolution": tuple(fast_cfg.required_resolution),
                "current_fps": runtime_fps,
                "degraded": fast_degraded,
                "degradation_reason": "; ".join(reasons) or None,
            },
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

    def get_latest_frame_with_identity(
        self,
    ) -> tuple[np.ndarray | None, FrameIdentity | None]:
        """Get the latest preview frame and its identity."""
        with self._latest_frame_lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
        with self._latest_frame_identity_lock:
            identity = (
                self._latest_frame_identity.with_updates()
                if self._latest_frame_identity is not None
                else None
            )
        return frame, identity

    def get_latest_frame_identity(self) -> FrameIdentity | None:
        """Return identity for the latest frame visible to preview/snapshot."""
        with self._latest_frame_identity_lock:
            if self._latest_frame_identity is None:
                return None
            return self._latest_frame_identity.with_updates()

    def mark_snapshot_identity(
        self,
        snapshot_id: str,
        *,
        timestamp: float | None = None,
    ) -> FrameIdentity | None:
        """Record that the latest frame was served through the snapshot API."""
        at = time.time() if timestamp is None else timestamp
        with self._latest_frame_identity_lock:
            if self._latest_frame_identity is None:
                return None
            identity = self._latest_frame_identity.with_updates(
                snapshot_id=snapshot_id,
            )
            self._last_snapshot_identity = identity
            self._last_snapshot_ts = at
            return identity

    def mark_preview_stream_identity(
        self,
        *,
        timestamp: float | None = None,
    ) -> FrameIdentity | None:
        """Record that the latest frame was served through backend preview."""
        at = time.time() if timestamp is None else timestamp
        with self._latest_frame_identity_lock:
            if self._latest_frame_identity is None:
                self._last_preview_stream_ts = at
                return None
            identity = self._latest_frame_identity.with_updates(
                preview_stream_ts=at,
            )
            self._latest_frame_identity = identity
            self._last_preview_stream_identity = identity
            self._last_preview_stream_ts = at
            return identity

    def get_frame_identity_status(self) -> dict:
        """Return source and latest-frame identity diagnostics."""
        with self._latest_frame_identity_lock:
            latest = self._latest_frame_identity
            snapshot = self._last_snapshot_identity
            preview = self._last_preview_stream_identity
            return {
                "declared_source": self._frame_identity_declared_source,
                "runtime_source": self._frame_identity_runtime_source,
                "go2rtc_stream": self._frame_identity_stream_id,
                "latest": latest.to_dict() if latest is not None else None,
                "last_snapshot": snapshot.to_dict() if snapshot is not None else None,
                "last_preview_stream": preview.to_dict() if preview is not None else None,
                "last_frame_index": latest.frame_index if latest is not None else None,
                "last_capture_ts": (
                    latest.capture_wall_ts if latest is not None else None
                ),
                "last_pipeline_ts": (
                    latest.pipeline_ingest_ts if latest is not None else None
                ),
                "last_snapshot_ts": self._last_snapshot_ts,
                "last_snapshot_id": (
                    snapshot.snapshot_id if snapshot is not None else None
                ),
                "last_snapshot_frame_index": (
                    snapshot.frame_index if snapshot is not None else None
                ),
                "last_preview_stream_ts": self._last_preview_stream_ts,
                "last_preview_stream_frame_index": (
                    preview.frame_index if preview is not None else None
                ),
            }

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

    def _set_frame_outcome(
        self,
        kind: FrameOutcomeKind,
        reason: str | None = None,
    ) -> None:
        """Publish the explicit result of the current captured frame."""
        self._last_frame_outcome_kind = kind
        self._last_frame_outcome_reason = reason

    @property
    def last_frame_outcome_kind(self) -> FrameOutcomeKind:
        """Explicit state of the most recently captured frame."""
        return self._last_frame_outcome_kind

    @property
    def last_frame_outcome_reason(self) -> str | None:
        """Failure or suppression reason paired with the latest frame state."""
        return self._last_frame_outcome_reason

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

    def set_mode(self, mode: PipelineMode) -> PipelineMode:
        """Set pipeline operating mode (DET-006). Returns previous mode."""
        now = time.monotonic()
        with self._mode_lock:
            old = self._mode
            self._mode = mode
            if mode == PipelineMode.LEARNING:
                self._learning_start_time = now
                if self._learning_duration <= 0:
                    fps = max(1, self.camera_config.fps_target)
                    history = self.camera_config.mog2.history
                    self._learning_duration = max(history / fps * 3, 600)
                self._auto_learning_complete = False
            elif old == PipelineMode.LEARNING:
                self._learning_start_time = None
                if mode == PipelineMode.ACTIVE:
                    self._auto_learning_complete = True
            if mode in _DETECTION_SKIPPED_MODES:
                self._mode_session_start_time = now
            else:
                self._mode_session_start_time = None
        self._reset_fast_motion_after_detection_mode_entry(old, mode, now=now)
        if old != mode:
            self._reset_candidate_model_evidence_state()
            self._alert_grader.reset()
            self._incident_associations.clear()
        logger.info(
            "pipeline.mode_changed",
            camera_id=self.camera_config.camera_id,
            old=old.value,
            new=mode.value,
        )
        return old

    def _reset_fast_motion_after_detection_mode_entry(
        self,
        old: PipelineMode,
        new: PipelineMode,
        *,
        now: float | None = None,
    ) -> None:
        """Clear transient fast-motion state when detection resumes."""
        if old == new or new not in (PipelineMode.ACTIVE, PipelineMode.MAINTENANCE):
            return
        if self._fast_motion is None:
            return
        ts = time.monotonic() if now is None else now
        self._fast_motion.reset()
        self._fast_motion_suppress_until = max(
            self._fast_motion_suppress_until,
            ts + self._fast_motion_recovery_seconds,
        )
        logger.debug(
            "pipeline.fast_motion_mode_warmup",
            camera_id=self.camera_config.camera_id,
            old=old.value,
            new=new.value,
            seconds=round(self._fast_motion_recovery_seconds, 3),
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

    def update_thresholds(
        self,
        anomaly_threshold: float | None = None,
        *,
        severity_changed: bool = False,
        temporal_changed: bool = False,
        suppression_changed: bool = False,
    ) -> dict[str, bool]:
        """Hot-update detection thresholds without restart (痛点 9).

        AlertGrader holds a reference to the shared AlertConfig object, so
        modifications to ``alerts.severity_thresholds`` / ``alerts.temporal``
        / ``alerts.suppression`` are observed automatically. The boolean
        flags here only refresh pipeline-level derived caches (e.g.
        heartbeat seconds) that snapshot config values once.

        Returns a per-section ``{section: True}`` map that the route layer
        can surface to the UI as ``hot_reloaded`` indicators.
        """
        applied: dict[str, bool] = {}
        with self._config_lock:
            if anomaly_threshold is not None:
                self._anomaly_detector.threshold = anomaly_threshold
                applied["anomaly_threshold"] = True
            if temporal_changed:
                # Refresh derived heartbeat caches from the (already-mutated)
                # alert_config.temporal reference.
                try:
                    new_max_gap = self._alert_grader._config.temporal.max_gap_seconds
                    self._heartbeat_seconds = new_max_gap
                    self._low_light_heartbeat_seconds = new_max_gap / 2
                    applied["temporal"] = True
                except AttributeError:
                    applied["temporal"] = False
            if severity_changed:
                applied["severity"] = True  # AlertGrader reads via self._config ref
            if suppression_changed:
                applied["suppression"] = True  # ditto
        logger.info(
            "pipeline.thresholds_updated",
            camera_id=self.camera_config.camera_id,
            applied=list(applied.keys()),
        )
        return applied

    def get_diagnostics_buffer(self) -> DiagnosticsBuffer:
        """Get the per-frame diagnostics ring buffer (DET-008)."""
        return self._diagnostics

    def get_detector_status(self) -> DetectorStatus:
        """Get anomaly detector operational status (DET-004)."""
        return self._anomaly_detector.get_status()

    def get_input_quality_status(self) -> dict:
        """Return live image-quality gates that can limit detection."""
        detector_status = self.get_detector_status()
        ssim_calibrating = (
            getattr(detector_status, "mode", None) == "ssim_fallback"
            and not bool(getattr(detector_status, "ssim_calibrated", True))
        )
        now = time.monotonic()
        low_light = bool(self._low_light_enabled and self._was_low_light)
        exposure_recovering = bool(
            self._low_light_enabled and now < self._fast_motion_suppress_until
        )
        limited_reason = None
        if low_light:
            limited_reason = "low_light"
        elif exposure_recovering:
            limited_reason = "exposure_recovery"

        return {
            "low_light_enabled": bool(self._low_light_enabled),
            "low_light": low_light,
            "low_light_threshold": self._low_light_threshold if self._low_light_enabled else None,
            "last_brightness": (
                round(self._prev_brightness, 1)
                if self._prev_brightness is not None
                else None
            ),
            "exposure_recovering": exposure_recovering,
            "exposure_recovery_remaining_seconds": round(
                max(0.0, self._fast_motion_suppress_until - now),
                3,
            ),
            "detection_limited": limited_reason is not None,
            "detection_limited_reason": limited_reason,
            "fast_motion_suppressed": bool(
                self._fast_motion is not None and limited_reason is not None
            ),
            "ssim_calibration_blocked": bool(ssim_calibrating and limited_reason is not None),
            "ssim_calibration_blocked_reason": (
                limited_reason if ssim_calibrating and limited_reason is not None else None
            ),
        }

    def prepare_anomaly_model_release(
        self,
        model_path: str | Path,
        *,
        version_tag: str,
        expected_artifact_sha256: str,
    ) -> PreparedAnomalyModelRelease:
        """Load an exact registry artifact into standby without changing inference.

        Release preparation never calls model discovery.  ``model_path`` must
        be the exact deployable file authorized by the registry.  The artifact
        fingerprint is checked before and after loading, and the currently
        approved SceneGuard profile is evaluated against the proposed identity
        without replacing the live reference.
        """

        target_path = Path(model_path).expanduser()
        if not target_path.is_file() or target_path.suffix.lower() not in {".pt", ".xml"}:
            raise ValueError(
                "release runtime requires an exact registry model.pt/model.xml path"
            )
        if not version_tag:
            raise ValueError("release runtime requires a model version identity")
        if self._uses_shared_anomaly_detector:
            raise RuntimeError(
                "two-phase per-camera release is unavailable for a shared detector"
            )
        if self._ensemble_detector is not None:
            raise RuntimeError(
                "two-phase release is unavailable while detector ensemble is enabled"
            )

        before_hash = runtime_artifact_sha256(target_path)
        if before_hash != expected_artifact_sha256:
            raise ValueError(
                "registry artifact hash does not match the runtime release plan"
            )

        cfg = self.camera_config.anomaly
        base_detector = AnomalibDetector(
            model_path=target_path,
            threshold=cfg.threshold,
            image_size=cfg.image_size,
            inference_backend=cfg.inference_backend,
            ssim_baseline_frames=cfg.ssim_baseline_frames,
            ssim_sensitivity=cfg.ssim_sensitivity,
            ssim_midpoint=cfg.ssim_midpoint,
            ssim_global_change_suppress_fraction=(
                cfg.ssim_global_change_suppress_fraction
            ),
            enable_calibration=cfg.enable_calibration,
            camera_id=self.camera_config.camera_id,
        )
        if base_detector.load() is not True:
            raise RuntimeError("standby detector did not load the requested artifact")

        target_detector: object = base_detector
        if cfg.enable_multiscale:
            from argus.anomaly.detector import MultiScaleDetector

            target_detector = MultiScaleDetector(
                base_detector=base_detector,
                tile_size=cfg.tile_size,
                tile_overlap=cfg.tile_overlap,
                pyramid_mode=cfg.pyramid_mode,
                pyramid_sizes=cfg.pyramid_sizes,
            )

        after_hash = runtime_artifact_sha256(target_path)
        if after_hash != before_hash:
            raise RuntimeError("runtime artifact changed while standby detector loaded")

        # The resolver is read-only. Temporarily presenting the proposed
        # identity under the frame-processing lock allows reuse of the exact
        # profile authority rules without mutating SceneGuard trust state.
        with self._scene_processing_lock:
            previous_version = self._model_version_id
            previous_hash = self._model_artifact_sha256
            self._model_version_id = version_tag
            self._model_artifact_sha256 = after_hash
            try:
                reference, source, _profile_id, _metadata, error = (
                    self._resolve_scene_guard_reference()
                )
            finally:
                self._model_version_id = previous_version
                self._model_artifact_sha256 = previous_hash
        if self._scene_guard is not None and (
            error is not None or reference is None or source is None
        ):
            raise RuntimeError(
                "standby model is not authorized by the active scene profile: "
                + (error or "scene reference unavailable")
            )

        return PreparedAnomalyModelRelease(
            owner_id=id(self),
            target_detector=target_detector,
            target_path=target_path.resolve(),
            target_version=version_tag,
            target_artifact_sha256=after_hash,
            target_scene_profile_id=_profile_id,
            previous_detector=self._anomaly_detector,
            previous_path=self._anomaly_model_path,
            previous_version=self._model_version_id,
            previous_artifact_sha256=self._model_artifact_sha256,
        )

    def commit_anomaly_model_release(
        self,
        prepared: PreparedAnomalyModelRelease,
    ) -> bool:
        """Atomically expose a preloaded detector while retaining rollback state."""

        with self._scene_processing_lock:
            if prepared.owner_id != id(self) or prepared.committed:
                return False
            if (
                self._anomaly_detector is not prepared.previous_detector
                or self._anomaly_model_path != prepared.previous_path
                or self._model_version_id != prepared.previous_version
                or self._model_artifact_sha256 != prepared.previous_artifact_sha256
            ):
                return False
            if runtime_artifact_sha256(prepared.target_path) != prepared.target_artifact_sha256:
                return False

            prepared.previous_scene_state = self._snapshot_model_release_scene_state()
            # Mark the token before the first mutation. Revalidation/reset may
            # throw after the detector was exposed; abort must also compensate
            # that partially committed state, not mistake it for unused standby.
            prepared.committed = True
            try:
                if self._scene_guard is not None:
                    # set_reference/clear_reference replace reference arrays;
                    # a shallow clone retains OpenCV handles without mutating
                    # the rollback guard's reference/debounce attributes.
                    self._scene_guard = copy(self._scene_guard)
                self._anomaly_detector = prepared.target_detector
                self._anomaly_model_path = prepared.target_path
                self._model_version_id = prepared.target_version
                self._model_artifact_sha256 = prepared.target_artifact_sha256
                self._reset_candidate_model_evidence_state()
                self._revalidate_scene_guard_reference()
                if (
                    self._scene_guard is not None
                    and self._scene_guard_reference_error is not None
                ):
                    self.abort_anomaly_model_release(prepared)
                    return False
            except BaseException:
                self.abort_anomaly_model_release(prepared)
                raise
            return True

    def _snapshot_model_release_scene_state(self) -> dict[str, object]:
        """Retain immutable reference/status values while processing is locked."""
        return {
            name: value
            for name, value in vars(self).items()
            if name == "_scene_guard"
            or (name.startswith("_scene_guard_") and name != "_scene_guard_lock")
            or name == "_scene_sensitive_rois"
        }

    def _restore_model_release_scene_state(self, state: dict[str, object]) -> None:
        # Reference setters assign new arrays/lists instead of mutating the
        # retained values. Never copy synchronization primitives or cv2 handles.
        for name, value in state.items():
            setattr(self, name, value)

    def abort_anomaly_model_release(
        self,
        prepared: PreparedAnomalyModelRelease,
    ) -> bool:
        """Discard standby state or restore the previous committed detector."""

        with self._scene_processing_lock:
            if prepared.owner_id != id(self):
                return False
            if not prepared.committed:
                return prepared.rollback_error is None
            if (
                self._anomaly_detector is not prepared.target_detector
                and self._anomaly_detector is not prepared.previous_detector
            ):
                return False
            self._anomaly_detector = prepared.previous_detector
            self._anomaly_model_path = prepared.previous_path
            self._model_version_id = prepared.previous_version
            self._model_artifact_sha256 = prepared.previous_artifact_sha256
            if prepared.previous_scene_state is not None:
                self._restore_model_release_scene_state(prepared.previous_scene_state)
            prepared.committed = False
            try:
                if self._scene_guard is not None:
                    self._scene_guard = copy(self._scene_guard)
                self._reset_candidate_model_evidence_state()
                # Recheck disk authority; restoring an old cached trusted pose
                # must not authorize a profile that changed during preparation.
                self._revalidate_scene_guard_reference()
            except BaseException as exc:
                if prepared.previous_scene_state is not None:
                    self._restore_model_release_scene_state(prepared.previous_scene_state)
                prepared.rollback_error = f"scene rollback failed: {type(exc).__name__}: {exc}"
                self._scene_guard_runtime_error = prepared.rollback_error
                self._scene_guard_last_usable = False
                self._scene_guard_policy_recovery_required = True
                self._scene_guard_policy_recovery_count = 0
                try:
                    if self._scene_guard is not None:
                        self._scene_guard.require_reconfirmation()
                    self._reset_scene_sensitive_state()
                except BaseException:
                    logger.critical("pipeline.model_release_rollback_reset_failed", exc_info=True)
                logger.critical(
                    "pipeline.model_release_scene_rollback_failed",
                    camera_id=self.camera_config.camera_id,
                    error=prepared.rollback_error,
                )
                return False
            return True

    def is_anomaly_model_release_restored(
        self,
        prepared: PreparedAnomalyModelRelease,
    ) -> bool:
        """Confirm artifact identity separately from scene rollback readiness."""
        with self._scene_processing_lock:
            return (
                prepared.owner_id == id(self)
                and self._anomaly_detector is prepared.previous_detector
                and self._anomaly_model_path == prepared.previous_path
                and self._model_version_id == prepared.previous_version
                and self._model_artifact_sha256 == prepared.previous_artifact_sha256
            )

    def reload_anomaly_model(
        self,
        model_path: str | Path,
        *,
        version_tag: str | None = None,
    ) -> bool:
        """Atomically hot-reload a model and revalidate its scene binding.

        Frame processing uses the same lock, so no frame can observe a new
        detector together with the previous model-version/profile identity.
        A failed detector swap leaves both the old model and binding untouched.
        """
        with self._scene_processing_lock:
            if (
                self._scene_guard is not None
                and self.camera_config.scene_guard.require_profile_model_binding
                and not version_tag
            ):
                logger.error(
                    "pipeline.model_reload_missing_version",
                    camera_id=self.camera_config.camera_id,
                    model_path=str(model_path),
                )
                return False
            new_model_path = Path(model_path)
            try:
                before_reload_hash = _model_artifact_sha256(new_model_path)
            except (OSError, ValueError) as exc:
                logger.error(
                    "pipeline.model_reload_fingerprint_failed",
                    camera_id=self.camera_config.camera_id,
                    model_path=str(new_model_path),
                    phase="before_reload",
                    error=str(exc),
                )
                return False
            if before_reload_hash is None:
                logger.error(
                    "pipeline.model_reload_fingerprint_missing",
                    camera_id=self.camera_config.camera_id,
                    model_path=str(new_model_path),
                )
                return False
            # The detector contract returns a real bool.  Require literal
            # success so a truthy sentinel cannot advance runtime identity.
            if self._anomaly_detector.hot_reload(new_model_path) is not True:
                return False
            self._anomaly_model_path = new_model_path
            try:
                after_reload_hash = _model_artifact_sha256(new_model_path)
            except (OSError, ValueError) as exc:
                after_reload_hash = None
                logger.error(
                    "pipeline.model_reload_fingerprint_failed",
                    camera_id=self.camera_config.camera_id,
                    model_path=str(new_model_path),
                    phase="after_reload",
                    error=str(exc),
                )
            self._model_artifact_sha256 = (
                after_reload_hash
                if after_reload_hash == before_reload_hash
                else None
            )
            if self._model_artifact_sha256 is None:
                logger.error(
                    "pipeline.model_reload_fingerprint_unstable",
                    camera_id=self.camera_config.camera_id,
                    model_path=str(new_model_path),
                    before=before_reload_hash,
                    after=after_reload_hash,
                )
            # Even an unchanged version label can point at a different artifact
            # or output calibration after an operator reload.
            self._reset_candidate_model_evidence_state()
            if version_tag is not None:
                # RLock re-entry is intentional: profile mismatch clears the
                # approved reference and fails closed before processing resumes.
                self.set_model_version_id(version_tag)
            else:
                # Explicit validation configurations may opt out of version
                # binding, but a hot-loaded artifact must still re-run scene
                # profile authority checks before the processing lock opens.
                self._revalidate_scene_guard_reference()
            return True

    def reload_module(self, key: str, enabled: bool) -> bool:
        """Apply desired state, serializing swaps with frame processing.

        True means the requested runtime state was actually reached, not merely
        that an object was attached. A failed enable remains retryable.
        """
        with self._scene_processing_lock:
            if self._auxiliary_shutdown:
                return False
            if key == "physics.speed_enabled":
                try:
                    if not enabled:
                        self._speed_estimator = None
                    elif self._speed_estimator is None:
                        from argus.physics.speed import PixelSpeedEstimator
                        cfg = self._physics_config
                        if cfg is None:
                            return False
                        self._speed_estimator = PixelSpeedEstimator(
                            fps=self.camera_config.fps_target,
                            smoothing_window=cfg.speed_smoothing_window,
                            pixel_scale_mm_per_px=cfg.pixel_scale_mm_per_px,
                            calibration=self._camera_calibration,
                        )
                    self._auxiliary_errors.pop("speed", None)
                    return True
                except Exception as exc:
                    self._auxiliary_errors["speed"] = str(exc)
                    return False

            if key not in {"classifier.enabled", "segmenter.enabled"}:
                return False
            name = key.split(".", 1)[0]
            attr = f"_{name}"
            component = getattr(self, attr)
            if not enabled:
                return self._stop_auxiliary_module(name)

            created = component is None
            try:
                retired = self._auxiliary_retired.get(name)
                if retired is not None:
                    retired_status = retired.get_status()
                    if retired_status.get("busy") is not False or retired_status.get("state") != "stopped":
                        raise RuntimeError(f"{name} is still draining a stopped SDK task")
                    self._auxiliary_retired.pop(name, None)
                cfg = getattr(self, f"_{name}_config")
                if cfg is None:
                    raise RuntimeError(f"{name} configuration is unavailable")
                if component is None:
                    if name == "classifier":
                        from argus.anomaly.classifier import OpenVocabClassifier
                        component = OpenVocabClassifier(
                            model_name=cfg.model_name, vocabulary=cfg.vocabulary,
                            timeout_seconds=cfg.timeout_seconds,
                            runtime_key=f"{self.camera_config.camera_id}:classifier",
                        )
                    else:
                        from argus.anomaly.segmenter import InstanceSegmenter
                        component = InstanceSegmenter(
                            model_size=cfg.model_size,
                            min_mask_area_px=cfg.min_mask_area_px,
                            timeout_seconds=cfg.timeout_seconds,
                            runtime_key=f"{self.camera_config.camera_id}:segmenter",
                        )
                if name == "classifier" and cfg.custom_vocabulary_path:
                    component.load_vocabulary(cfg.custom_vocabulary_path)
                if component.load() is not True:
                    status = component.get_status()
                    raise RuntimeError(status.get("reason") or f"{name} model is not ready")
                if component.get_status().get("ready") is not True:
                    raise RuntimeError(f"{name} did not confirm target-model readiness")
                setattr(self, attr, component)
                if name == "classifier":
                    self._apply_classifier_policy()
                    self._classifier_vocabulary = set(component.vocabulary)
                else:
                    self._segmenter_max_points = cfg.max_points
                    self._segmenter_min_score = cfg.min_anomaly_score
                    self._segmenter_consecutive_failures = 0
                self._auxiliary_errors.pop(name, None)
                self._auxiliary_load_blocked.discard(name)
                return True
            except Exception as exc:
                self._auxiliary_errors[name] = str(exc)
                self._auxiliary_load_blocked.add(name)
                if created and component is not None:
                    # Retain the bounded in-flight slot across failed retries;
                    # replacing a timed-out instance could orphan SDK workers.
                    setattr(self, attr, component)
                logger.warning("pipeline.auxiliary_enable_failed", module=name, error=str(exc))
                return False

    def _apply_classifier_policy(self) -> None:
        cfg = self._classifier_config
        self._classifier_min_score = cfg.min_anomaly_score_to_classify
        self._classifier_min_confidence = cfg.min_confidence_to_affect_alert
        self._classifier_high_risk = set(cfg.high_risk_labels)
        low = set(cfg.low_risk_labels)
        suppress = set(cfg.suppress_labels)
        self._classifier_low_risk = low - self._classifier_high_risk - suppress
        self._classifier_suppress = suppress - self._classifier_high_risk - low

    def _stop_auxiliary_module(self, name: str) -> bool:
        """Detach evidence immediately, retaining cleanup ownership until confirmed."""
        component = getattr(self, f"_{name}", None) or self._auxiliary_retired.get(name)
        setattr(self, f"_{name}", None)
        try:
            if component is not None:
                self._auxiliary_retired[name] = component
                component.shutdown()
                status = component.get_status()
                if (
                    status.get("state") != "stopped" or status.get("cleanup_pending") is True
                    or type(status.get("busy")) is not bool
                ):
                    raise RuntimeError(f"{name} shutdown is not confirmed")
                if status["busy"] is False:
                    self._auxiliary_retired.pop(name, None)
            self._auxiliary_errors.pop(name, None)
            self._auxiliary_load_blocked.discard(name)
            if name == "segmenter":
                self._segmenter_consecutive_failures = 0
            return True
        except Exception as exc:
            self._auxiliary_errors[name] = f"cleanup failed: {exc}"
            return False

    def reconcile_auxiliary_desired(self, desired: dict[str, bool]) -> bool:
        """Reconcile startup against newer disables without loading any model."""
        with self._scene_processing_lock:
            if self._auxiliary_shutdown:
                return False
            applied = True
            for key in ("classifier.enabled", "segmenter.enabled", "physics.speed_enabled"):
                if desired.get(key) is False:
                    applied = self.reload_module(key, False) and applied
            return applied

    def update_classifier_vocabulary(self, vocabulary: list[str]) -> bool:
        """Keep the active model vocabulary and alert policy in one frame boundary."""
        with self._scene_processing_lock:
            if self._classifier is None:
                return False
            self._classifier.update_vocabulary(vocabulary)
            self._classifier_vocabulary = set(self._classifier.vocabulary)
            self._apply_classifier_policy()
            return True

    def update_segmenter_params(
        self, *, max_points: int | None = None, min_anomaly_score: float | None = None,
        min_mask_area_px: int | None = None, timeout_seconds: float | None = None,
    ) -> bool:
        with self._scene_processing_lock:
            if self._segmenter is None or self._auxiliary_shutdown:
                return False
            self._segmenter.update_runtime_params(
                min_mask_area_px=min_mask_area_px, timeout_seconds=timeout_seconds,
            )
            if max_points is not None:
                self._segmenter_max_points = max_points
            if min_anomaly_score is not None:
                self._segmenter_min_score = min_anomaly_score
            return True

    def _revalidate_scene_guard_reference(self) -> None:
        """Re-resolve profile authority against the loaded runtime artifact."""

        if self._scene_guard is None:
            return
        reference, source, profile_id, metadata, error = (
            self._resolve_scene_guard_reference()
        )
        if error is not None or reference is None or source is None:
            self._reject_scene_guard_reference(
                source=source,
                profile_id=profile_id,
                metadata=metadata,
                error=error
                or "runtime model state invalidated the active scene profile",
            )
            return
        try:
            self.set_scene_reference(
                reference,
                source=source,
                profile_id=profile_id,
                metadata=metadata,
            )
        except Exception as exc:
            # Stored image checksums can be valid while the content is
            # featureless or otherwise unsuitable for the configured guard.
            # A failed replacement must erase any previously trusted pose.
            self._reject_scene_guard_reference(
                source=source,
                profile_id=profile_id,
                metadata=metadata,
                error=f"approved scene reference is unusable: {exc}",
            )

    def _reject_scene_guard_reference(
        self,
        *,
        source: str | None,
        profile_id: str | None,
        metadata: dict,
        error: str,
    ) -> None:
        """Clear every cached trust token for a rejected scene profile."""

        if self._scene_guard is None:
            return
        with self._scene_guard_lock:
            self._scene_guard.clear_reference()
            self._scene_guard_reference_frame = None
            self._scene_guard_reference_sha256 = None
            self._scene_guard_reference_edges = None
            self._scene_guard_quick_hanning = None
            self._scene_guard_reference_source = source
            self._scene_guard_profile_id = profile_id
            self._scene_guard_reference_error = error
            self._scene_guard_runtime_error = None
            self._scene_guard_last_result = None
            self._scene_guard_last_check_frame = 0
            self._scene_guard_last_check_monotonic = 0.0
            self._scene_guard_last_usable = False
            self._scene_guard_last_stream_key = None
            self._scene_guard_last_reconnect_count = None
            self._scene_guard_policy_recovery_required = False
            self._scene_guard_policy_recovery_count = 0
            self._scene_guard_last_quick_score = None
            self._scene_guard_last_quick_shift_px = None
            self._scene_guard_bound_model_version = (
                str(metadata.get("model_version_id"))
                if metadata.get("model_version_id") is not None
                else None
            )
            self._scene_guard_bound_model_artifact_sha256 = (
                str(metadata.get("model_artifact_sha256"))
                if metadata.get("model_artifact_sha256") is not None
                else None
            )
            self._scene_sensitive_rois = [
                tuple(int(value) for value in roi)
                for roi in self.camera_config.static_confirmation.sensitive_rois
            ]
        self._reset_scene_sensitive_state()
        logger.error(
            "pipeline.scene_profile_rejected",
            camera_id=self.camera_config.camera_id,
            model_version=self._model_version_id,
            model_artifact_sha256=self._model_artifact_sha256,
            error=self._scene_guard_reference_error,
        )

    def set_model_version_id(self, model_version_id: str | None) -> None:
        """Update the model version and revalidate any scene-profile authority."""
        with self._scene_processing_lock:
            version_changed = self._model_version_id != model_version_id
            self._model_version_id = model_version_id
            if version_changed:
                self._reset_candidate_model_evidence_state()
            self._revalidate_scene_guard_reference()

    def get_shadow_runner_version(self) -> str | None:
        """Return the attached shadow runner version, if any."""
        if not hasattr(self, "_shadow_runner_lock"):
            self._shadow_runner_lock = threading.RLock()
        with self._shadow_runner_lock:
            runner = self._shadow_runner
        if runner is None:
            return None
        return getattr(runner, "shadow_version_id", None)

    def get_shadow_runner(self) -> object | None:
        """Return the currently attached runner for transactional release CAS."""

        if not hasattr(self, "_shadow_runner_lock"):
            self._shadow_runner_lock = threading.RLock()
        with self._shadow_runner_lock:
            return self._shadow_runner

    def set_shadow_runner(self, shadow_runner: object | None) -> dict:
        """Atomically replace the shadow runner and flush the old one."""
        if not hasattr(self, "_shadow_runner_lock"):
            self._shadow_runner_lock = threading.RLock()
        new_version = (
            getattr(shadow_runner, "shadow_version_id", None)
            if shadow_runner is not None
            else None
        )
        with self._shadow_runner_lock:
            old_runner = self._shadow_runner
            old_version = (
                getattr(old_runner, "shadow_version_id", None)
                if old_runner is not None
                else None
            )
            if old_version == new_version:
                return {
                    "shadow_previous_version": old_version,
                    "shadow_model_version": new_version,
                    "shadow_attached": False,
                    "shadow_detached": False,
                }
            self._shadow_runner = shadow_runner

        detached = old_runner is not None and old_version != new_version
        if old_runner is not None:
            try:
                old_runner.flush()
            except Exception as e:
                logger.warning(
                    "pipeline.shadow_flush_failed",
                    camera_id=self.camera_config.camera_id,
                    error=str(e),
                )

        return {
            "shadow_previous_version": old_version,
            "shadow_model_version": new_version,
            "shadow_attached": shadow_runner is not None and old_version != new_version,
            "shadow_detached": detached and shadow_runner is None,
        }

    def shutdown(self) -> None:
        """Clean up all pipeline resources."""
        with self._shutdown_lock:
            self._shutdown_inner()

    def _shutdown_inner(self) -> None:
        errors = []
        with self._scene_processing_lock:
            self._auxiliary_shutdown = True
            for name in ("classifier", "segmenter"):
                if not self._stop_auxiliary_module(name):
                    errors.append(self._auxiliary_errors[name])
        try:
            self._camera.stop()
        except Exception as exc:
            errors.append(f"camera stop failed: {exc}")
        if not hasattr(self, "_shadow_runner_lock"):
            self._shadow_runner_lock = threading.RLock()
        with self._shadow_runner_lock:
            shadow_runner = self._shadow_runner
        if shadow_runner is not None:
            try:
                shadow_runner.flush()
                with self._shadow_runner_lock:
                    if self._shadow_runner is shadow_runner:
                        self._shadow_runner = None
            except Exception as e:
                errors.append(f"shadow flush failed: {e}")
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
        if errors:
            raise RuntimeError("; ".join(errors))

    def _decay_all_unobserved_alert_incidents(
        self,
        camera_id: str,
        *,
        decay_detection_types: set[str] | None = None,
    ) -> None:
        """Decay active incidents for channels observed on a normal frame."""
        if not isinstance(self._alert_grader, AlertGrader):
            return
        include_zones = self._zone_mask.get_include_zones()
        zone_ids = (
            [zone.zone_id for zone in include_zones]
            if include_zones
            else ["default"]
        )
        for zone_id in zone_ids:
            self._alert_grader.decay_unobserved(
                camera_id=camera_id,
                zone_id=zone_id,
                observed_incidents=set(),
                decay_detection_types=decay_detection_types,
            )

    def _evaluate_zones(
        self, frame_data: FrameData, anomaly_result: AnomalyResult, frame: np.ndarray,
        detection_type: str = "anomaly", detected_objects: list[dict] | None = None,
        classification_label: str | None = None,
        classification_confidence: float | None = None,
        appearance_result: AnomalyResult | None = None,
        incident_support_maps: dict[str, np.ndarray] | None = None,
        decay_detection_types: set[str] | None = None,
    ) -> Alert | None:
        """Evaluate appearance and localized incident scores by include zone.

        H4 fix: When anomaly_map is available and zones have polygons,
        compute per-zone score (max within zone region) instead of using
        the global score for all zones. This prevents zone A's anomaly
        from causing false CUSUM accumulation in zone B.

        An explicitly allowlisted non-person YOLO class may independently
        contribute its confidence as the score.  This keeps dangerous-object
        alerts live when the appearance model is intentionally skipped on a
        person frame, without treating every configured detector class as a
        safety incident.
        """
        include_zones = self._zone_mask.get_include_zones()
        appearance = appearance_result or anomaly_result
        self._last_zone_alerts = []
        raw_detection_type = getattr(
            detection_type,
            "value",
            str(detection_type),
        )

        def scoped_channel_map(
            channel_map: np.ndarray | None,
            zone,
        ) -> np.ndarray | None:
            if channel_map is None:
                return None
            if zone is None or not zone.polygon:
                return channel_map
            return self._mask_anomaly_map_to_zone(
                channel_map,
                zone.zone_id,
                zone.polygon,
                frame.shape,
            )

        def grade_zone(
            zone_id: str,
            priority: ZonePriority,
            score: float,
            zone_objects: list[dict],
            zone_detection_type: str,
            channel_heatmap: np.ndarray | None,
            continuity_mask: np.ndarray | None,
            incident_key: str | None,
            *,
            include_classification: bool = False,
        ):
            kwargs = {
                "camera_id": frame_data.camera_id,
                "zone_id": zone_id,
                "zone_priority": priority,
                "anomaly_score": score,
                "frame_number": frame_data.frame_number,
                "frame": frame,
                "anomaly_map": channel_heatmap,
                "detection_type": zone_detection_type,
                "detected_objects": zone_objects,
                "classification_label": (
                    classification_label if include_classification else None
                ),
                "classification_confidence": (
                    classification_confidence
                    if include_classification
                    else None
                ),
            }
            if isinstance(self._alert_grader, AlertGrader):
                kwargs["continuity_map"] = continuity_mask
                kwargs["incident_key"] = incident_key
                return self._alert_grader.evaluate_with_evidence(**kwargs)

            alert = self._alert_grader.evaluate(**kwargs)
            return SimpleNamespace(
                alert=alert,
                score=score,
                suppression_reason=None if alert is not None else "grader_no_alert",
            )

        scopes = (
            [(None, "default", ZonePriority.STANDARD)]
            if not include_zones
            else [
                (zone, zone.zone_id, zone.priority)
                for zone in include_zones
            ]
        )

        # Grade every (zone, channel) independently.  The frame-level merged
        # map is an operator display only; it must never raise a lower-scoring
        # localized incident or provide continuity evidence for another type.
        best_alert = None
        best_grade = None
        best_rank = -1
        emitted_alerts: list[Alert] = []
        for zone, zone_id, priority in scopes:
            zone_polygon = getattr(zone, "polygon", None)
            if zone is not None and zone_polygon:
                zone_mask = self._get_zone_polygon_mask(
                    zone_id,
                    zone_polygon,
                    frame.shape[:2],
                    source_shape=frame.shape[:2],
                )
                zone_objects = self._detected_objects_for_zone(
                    detected_objects,
                    zone_mask=zone_mask,
                )
            else:
                zone_objects = self._detected_objects_for_zone(
                    detected_objects,
                )

            channel_specs: list[
                tuple[
                    float,
                    DetectionType,
                    list[dict],
                    np.ndarray | None,
                    bool,
                    str | None,
                ]
            ] = []

            if raw_detection_type == DetectionType.PROJECTILE.value:
                projectile_objects = [
                    item
                    for item in zone_objects
                    if str(item.get("class_name") or "").lower()
                    == "fast_projectile"
                ]
                for projectile in projectile_objects:
                    try:
                        projectile_score = float(
                            projectile["projectile_alert_score"]
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not math.isfinite(projectile_score):
                        continue
                    projectile_map = scoped_channel_map(
                        self._localized_score_map_from_objects(
                            [projectile],
                            frame.shape,
                        ),
                        zone,
                    )
                    if not self._map_has_positive_support(projectile_map):
                        continue
                    incident_key = self._incident_identity(
                        DetectionType.PROJECTILE,
                        projectile,
                        zone_id=zone_id,
                        frame_number=frame_data.frame_number,
                    )
                    channel_specs.append((
                        min(projectile_score, 1.0),
                        DetectionType.PROJECTILE,
                        [projectile],
                        projectile_map,
                        False,
                        incident_key,
                    ))
            else:
                appearance_zone_score = 0.0
                if appearance.is_anomalous:
                    if appearance.anomaly_map is not None and zone_polygon:
                        appearance_zone_score = self._compute_zone_score(
                            appearance.anomaly_map,
                            zone_id,
                            zone_polygon,
                            frame.shape,
                        )
                    elif not zone_polygon or len(include_zones) == 1:
                        appearance_zone_score = appearance.anomaly_score
                if appearance_zone_score > 0.0:
                    appearance_map = scoped_channel_map(
                        appearance.anomaly_map,
                        zone,
                    )
                    appearance_components = self._appearance_map_components(
                        appearance_map,
                        frame.shape,
                    )
                    if appearance_components:
                        for component_index, (
                            component_score,
                            component_map,
                            component_bbox,
                        ) in enumerate(appearance_components):
                            # Preserve the calibrated frame scalar on the
                            # strongest component when no spatial zone score
                            # was previously required. Other components keep
                            # their exact local peak.
                            if component_index == 0 and (
                                not zone_polygon or len(include_zones) == 1
                            ):
                                component_score = max(
                                    component_score,
                                    appearance_zone_score,
                                )
                            component_identity_object = {
                                "class_name": "appearance_anomaly",
                                "bbox": component_bbox,
                            }
                            # Derive corroboration from exact component overlap,
                            # not the mutable frame-global detection type.  A
                            # water/plume/Simplex channel may be appended later
                            # and change that global summary without changing
                            # the fact that YOLO corroborated this component.
                            corroborators = [
                                item
                                for item in zone_objects
                                if "class_id" in item
                                and str(
                                    item.get("class_name") or ""
                                ).lower() != "person"
                                and item.get(
                                    "object_safety_alert_contributed"
                                ) is not True
                                and self._object_overlaps_score_support(
                                    item,
                                    component_map,
                                    frame.shape,
                                )
                            ]
                            component_type = (
                                DetectionType.HYBRID
                                if corroborators
                                else DetectionType.ANOMALY
                            )
                            incident_key = self._incident_identity(
                                # ANOMALY/HYBRID are two observations of the
                                # same appearance family.  Keep association and
                                # temporal identity stable while preserving the
                                # emitted HYBRID type for corroborated frames.
                                DetectionType.ANOMALY,
                                component_identity_object,
                                zone_id=zone_id,
                                frame_number=frame_data.frame_number,
                            )
                            channel_specs.append((
                                component_score,
                                component_type,
                                corroborators,
                                component_map,
                                len(appearance_components) == 1,
                                incident_key,
                            ))
                    else:
                        # A detector without a usable spatial map retains the
                        # legacy zone-level fallback; it cannot be safely split
                        # into incident identities.
                        channel_specs.append((
                            appearance_zone_score,
                            DetectionType.ANOMALY,
                            [],
                            appearance_map,
                            True,
                            None,
                        ))

                object_incidents = [
                    item
                    for item in zone_objects
                    if item.get("object_safety_alert_contributed") is True
                ]
                for object_incident in object_incidents:
                    object_score = self._object_safety_score(
                        [object_incident],
                        require_contributed=True,
                    )
                    if object_score <= 0.0:
                        continue
                    object_map = scoped_channel_map(
                        self._localized_score_map_from_objects(
                            [object_incident],
                            frame.shape,
                        ),
                        zone,
                    )
                    if not self._map_has_positive_support(object_map):
                        continue
                    incident_key = self._incident_identity(
                        DetectionType.OBJECT,
                        object_incident,
                        zone_id=zone_id,
                        frame_number=frame_data.frame_number,
                    )
                    channel_specs.append((
                        object_score,
                        DetectionType.OBJECT,
                        [object_incident],
                        object_map,
                        False,
                        incident_key,
                    ))

                for item in zone_objects:
                    if (
                        item.get("independent_visual_alert_eligible")
                        is not True
                    ):
                        continue
                    visual_type = _INDEPENDENT_VISUAL_DETECTION_TYPES.get(
                        str(item.get("class_name") or "").lower()
                    )
                    if visual_type is None:
                        continue
                    visual_score, _ = self._independent_visual_evidence(
                        [item]
                    )
                    if visual_score <= 0.0:
                        continue
                    incident_key = self._incident_identity(
                        visual_type,
                        item,
                        zone_id=zone_id,
                        frame_number=frame_data.frame_number,
                    )
                    source_map = (incident_support_maps or {}).get(
                        incident_key
                    )
                    if source_map is None:
                        source_map = self._localized_score_map_from_objects(
                            [item],
                            frame.shape,
                        )
                    visual_map = scoped_channel_map(
                        source_map,
                        zone,
                    )
                    if not self._map_has_positive_support(visual_map):
                        continue
                    channel_specs.append((
                        visual_score,
                        visual_type,
                        [item],
                        visual_map,
                        False,
                        incident_key,
                    ))

            # Preserve one zero-score evaluation for compatibility and to
            # decay the legacy anomaly tracker when a frame contains only
            # non-eligible detector output.
            if not channel_specs and (
                decay_detection_types is None
                or DetectionType.ANOMALY.value in decay_detection_types
            ):
                channel_specs.append((
                    0.0,
                    DetectionType.ANOMALY,
                    [],
                    None,
                    False,
                    None,
                ))

            handled_incidents: set[tuple[str, str | None]] = set()
            for (
                channel_score,
                channel_type,
                channel_objects,
                channel_map,
                include_classification,
                incident_key,
            ) in channel_specs:
                handled_incidents.add((
                    getattr(channel_type, "value", str(channel_type)),
                    incident_key,
                ))
                grade = grade_zone(
                    zone_id,
                    priority,
                    channel_score,
                    channel_objects,
                    channel_type,
                    channel_map,
                    channel_map,
                    incident_key,
                    include_classification=include_classification,
                )
                alert = grade.alert
                if alert is not None:
                    emitted_alerts.append(alert)
                    rank = _severity_rank(alert.severity)
                    if best_alert is None or rank > best_rank or (
                        rank == best_rank
                        and alert.anomaly_score > best_alert.anomaly_score
                    ):
                        best_alert = alert
                        best_grade = grade
                        best_rank = rank
                elif best_grade is None or grade.score > best_grade.score:
                    best_grade = grade

            if isinstance(self._alert_grader, AlertGrader):
                self._alert_grader.decay_unobserved(
                    camera_id=frame_data.camera_id,
                    zone_id=zone_id,
                    observed_incidents=handled_incidents,
                    decay_detection_types=decay_detection_types,
                )

        self._last_grade_result = best_grade
        self._last_zone_alerts = emitted_alerts
        if best_alert is not None:
            best_alert._secondary_zone_alerts = [
                candidate
                for candidate in emitted_alerts
                if candidate is not best_alert
            ]
        return best_alert

    def _object_detection_payload(self, obj: ObjectDetection) -> dict:
        """Serialize YOLO evidence while retaining exact safety confidence."""
        confidence = float(obj.confidence)
        config = self.camera_config.person_filter
        eligible = (
            obj.class_id in config.alert_class_ids
            and math.isfinite(confidence)
            and confidence >= config.alert_min_confidence
        )
        return {
            "class_id": obj.class_id,
            "class_name": obj.class_name,
            "confidence": round(confidence, 3),
            "track_id": obj.track_id,
            "bbox": [obj.x1, obj.y1, obj.x2, obj.y2],
            "object_safety_alert_eligible": eligible,
            # Keep scoring exact: the rounded display confidence can cross a
            # configured threshold in either direction.
            "object_safety_score": confidence if eligible else None,
        }

    @staticmethod
    def _independent_visual_payload(candidate) -> dict:
        """Serialize a confirmed visual hazard with its exact channel score."""
        payload = candidate.to_detected_object()
        class_name = str(payload.get("class_name") or "").lower()
        try:
            score = float(candidate.score)
        except (TypeError, ValueError):
            score = float("nan")
        eligible = bool(
            class_name in _INDEPENDENT_VISUAL_DETECTION_TYPES
            and math.isfinite(score)
            and score > 0.0
        )
        payload["independent_visual_alert_eligible"] = eligible
        payload["independent_visual_score"] = score if eligible else None
        return payload

    @staticmethod
    def _simplex_visual_payload(region, score: float, index: int) -> dict:
        """Serialize one Simplex static region as localized safety evidence."""
        x, y, width, height = (int(value) for value in region.bbox)
        finite_score = float(score)
        stable_track_id = getattr(region, "track_id", None)
        payload_track_id = (
            stable_track_id if stable_track_id is not None else index
        )
        eligible = bool(
            width > 0
            and height > 0
            and math.isfinite(finite_score)
            and finite_score > 0.0
        )
        return {
            "class_name": "simplex_static",
            "confidence": round(finite_score, 3) if eligible else 0.0,
            "track_id": f"simplex-{payload_track_id}",
            "bbox": [x, y, x + width, y + height],
            "area_px": int(region.area_px),
            "persistence_seconds": round(
                max(0.0, float(region.last_seen) - float(region.first_seen)),
                3,
            ),
            "independent_visual_alert_eligible": eligible,
            "independent_visual_score": finite_score if eligible else None,
        }

    @staticmethod
    def _localized_score_map_from_objects(
        detected_objects: list[dict] | None,
        frame_shape: tuple[int, ...],
    ) -> np.ndarray | None:
        """Build a display map whose localized boxes retain their exact scores."""
        if len(frame_shape) < 2:
            return None
        height, width = int(frame_shape[0]), int(frame_shape[1])
        if height <= 0 or width <= 0:
            return None
        score_map = np.zeros((height, width), dtype=np.float32)
        has_support = False
        for detected_object in detected_objects or []:
            if (
                detected_object.get("independent_visual_alert_eligible")
                is True
            ):
                raw_score = detected_object.get("independent_visual_score")
            elif detected_object.get("object_safety_alert_contributed") is True:
                raw_score = detected_object.get("object_safety_score")
            elif (
                str(detected_object.get("class_name") or "").lower()
                == "fast_projectile"
            ):
                raw_score = detected_object.get(
                    "projectile_alert_score",
                    detected_object.get("confidence"),
                )
            else:
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score) or score <= 0.0:
                continue
            bbox = detected_object.get("bbox")
            if not isinstance(bbox, list | tuple) or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = (
                    int(round(float(value))) for value in bbox
                )
            except (OverflowError, TypeError, ValueError):
                continue
            left = max(0, min(x1, x2))
            right = min(width, max(x1, x2))
            top = max(0, min(y1, y2))
            bottom = min(height, max(y1, y2))
            if right <= left or bottom <= top:
                continue
            np.maximum(
                score_map[top:bottom, left:right],
                min(score, 1.0),
                out=score_map[top:bottom, left:right],
            )
            has_support = True
        return score_map if has_support else None

    @staticmethod
    def _confirmed_persistent_score_map(
        candidates: list,
        frame_shape: tuple[int, ...],
    ) -> np.ndarray | None:
        """Build exact component support for confirmed persistent candidates."""
        if len(frame_shape) < 2:
            return None
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        if frame_h <= 0 or frame_w <= 0:
            return None
        score_map = np.zeros((frame_h, frame_w), dtype=np.float32)
        has_support = False
        for candidate in candidates:
            bbox = getattr(candidate, "bbox", None)
            support_mask = getattr(candidate, "support_mask", None)
            try:
                score = float(candidate.score)
            except (AttributeError, TypeError, ValueError):
                continue
            if (
                not math.isfinite(score)
                or score <= 0.0
                or not isinstance(bbox, list | tuple)
                or len(bbox) != 4
            ):
                continue
            x, y, width, height = (int(value) for value in bbox)
            if (
                width <= 0
                or height <= 0
                or x < 0
                or y < 0
                or x + width > frame_w
                or y + height > frame_h
            ):
                continue
            if (
                isinstance(support_mask, np.ndarray)
                and support_mask.shape == (height, width)
            ):
                region = score_map[y : y + height, x : x + width]
                support_values = np.where(
                    support_mask > 0,
                    min(score, 1.0),
                    0.0,
                ).astype(np.float32, copy=False)
                np.maximum(
                    region,
                    support_values,
                    out=region,
                )
            else:
                np.maximum(
                    score_map[y : y + height, x : x + width],
                    min(score, 1.0),
                    out=score_map[y : y + height, x : x + width],
                )
            has_support = True
        return score_map if has_support else None

    @staticmethod
    def _independent_visual_evidence(
        detected_objects: list[dict] | None,
    ) -> tuple[float, DetectionType | None]:
        """Return the strongest explicitly tagged visual hazard in a scope."""
        best_score = 0.0
        best_type = None
        for detected_object in detected_objects or []:
            if (
                detected_object.get("independent_visual_alert_eligible")
                is not True
            ):
                continue
            class_name = str(
                detected_object.get("class_name") or ""
            ).lower()
            candidate_type = _INDEPENDENT_VISUAL_DETECTION_TYPES.get(class_name)
            if candidate_type is None:
                continue
            try:
                score = float(detected_object["independent_visual_score"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(score) or score <= 0.0:
                continue
            score = min(score, 1.0)
            if score > best_score:
                best_score = score
                best_type = candidate_type
        return best_score, best_type

    def _incident_identity(
        self,
        detection_type: DetectionType | str,
        detected_object: dict,
        *,
        zone_id: str | None = None,
        frame_number: int | None = None,
    ) -> str:
        """Return a stable, serializable identity for one localized incident."""
        type_value = getattr(detection_type, "value", str(detection_type))
        class_name = str(detected_object.get("class_name") or "unknown").lower()
        track_id = detected_object.get("track_id")
        if track_id is not None:
            return f"{type_value}:{class_name}:track:{track_id}"
        bbox = self._finite_xyxy_bbox(detected_object.get("bbox"))
        frame_token = (
            int(frame_number)
            if frame_number is not None
            else int(self._frame_counter)
        )
        if bbox is not None and zone_id is not None:
            runtime_id = self._associate_runtime_incident(
                zone_id=zone_id,
                detection_type=type_value,
                class_name=class_name,
                bbox=bbox,
                frame_token=frame_token,
            )
            return f"{type_value}:{class_name}:runtime:{runtime_id}"

        # Unlocalized evidence cannot safely share a temporal identity. Give
        # it a one-observation key so it cannot borrow CUSUM from another
        # object; localized channels normally fail closed before this point.
        self._incident_association_counter += 1
        return (
            f"{type_value}:{class_name}:unlocalized:"
            f"{frame_token}:{self._incident_association_counter}"
        )

    @staticmethod
    def _finite_xyxy_bbox(
        raw_bbox: object,
    ) -> tuple[float, float, float, float] | None:
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = (float(value) for value in raw_bbox)
        except (OverflowError, TypeError, ValueError):
            return None
        if (
            not all(math.isfinite(value) for value in (x1, y1, x2, y2))
            or x2 <= x1
            or y2 <= y1
        ):
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _incident_bbox_iou(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        second_area = (second[2] - second[0]) * (second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0.0 else 0.0

    def _associate_runtime_incident(
        self,
        *,
        zone_id: str,
        detection_type: str,
        class_name: str,
        bbox: tuple[float, float, float, float],
        frame_token: int,
    ) -> int:
        """Associate a jittering bbox with one short-lived runtime identity.

        Upstream track IDs remain authoritative. This fallback is deliberately
        local to one zone/type/class and one temporal CUSUM gap; matching is
        one-to-one per processed frame so two nearby hazards cannot inherit
        the same identity.
        """
        now = time.monotonic()
        key = (str(zone_id), str(detection_type), str(class_name))
        records = [
            record
            for record in self._incident_associations.get(key, [])
            if now - float(record["last_seen"]) <= (
                self._incident_association_max_gap_seconds
            )
        ]
        self._incident_associations[key] = records

        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
        diagonal = math.hypot(x2 - x1, y2 - y1)
        best_record: dict[str, object] | None = None
        best_score = -1.0
        for record in records:
            if int(record.get("matched_frame", -1)) == frame_token:
                continue
            previous_bbox = record.get("bbox")
            if not isinstance(previous_bbox, tuple) or len(previous_bbox) != 4:
                continue
            px1, py1, px2, py2 = previous_bbox
            previous_area = max(0.0, (px2 - px1) * (py2 - py1))
            if previous_area <= 0.0 or area <= 0.0:
                continue
            area_ratio = min(area, previous_area) / max(area, previous_area)
            iou = self._incident_bbox_iou(bbox, previous_bbox)
            previous_center = ((px1 + px2) * 0.5, (py1 + py2) * 0.5)
            center_distance = math.hypot(
                center[0] - previous_center[0],
                center[1] - previous_center[1],
            )
            previous_diagonal = math.hypot(px2 - px1, py2 - py1)
            distance_gate = max(24.0, 0.6 * max(diagonal, previous_diagonal))
            if area_ratio < 0.25 or (iou < 0.20 and center_distance > distance_gate):
                continue
            match_score = iou + max(0.0, 1.0 - center_distance / distance_gate)
            if match_score > best_score:
                best_record = record
                best_score = match_score

        if best_record is None:
            self._incident_association_counter += 1
            best_record = {
                "runtime_id": self._incident_association_counter,
                "bbox": bbox,
                "last_seen": now,
                "matched_frame": frame_token,
            }
            records.append(best_record)
        else:
            best_record["bbox"] = bbox
            best_record["last_seen"] = now
            best_record["matched_frame"] = frame_token
        return int(best_record["runtime_id"])

    @staticmethod
    def _map_has_positive_support(score_map: np.ndarray | None) -> bool:
        if not isinstance(score_map, np.ndarray) or score_map.size == 0:
            return False
        values = np.asarray(score_map).squeeze()
        return bool(
            values.ndim == 2
            and np.any(np.isfinite(values) & (values > 0.0))
        )

    @staticmethod
    def _object_overlaps_score_support(
        detected_object: dict,
        score_map: np.ndarray,
        frame_shape: tuple[int, ...],
    ) -> bool:
        """Return whether an object's bbox intersects this exact map support."""
        bbox = DetectionPipeline._finite_xyxy_bbox(
            detected_object.get("bbox")
        )
        values = np.asarray(score_map).squeeze()
        if bbox is None or values.ndim != 2 or values.size == 0:
            return False
        frame_h, frame_w = frame_shape[:2]
        if frame_h <= 0 or frame_w <= 0:
            return False
        map_h, map_w = values.shape
        x1, y1, x2, y2 = bbox
        left = max(0, min(map_w, int(math.floor(x1 * map_w / frame_w))))
        right = max(0, min(map_w, int(math.ceil(x2 * map_w / frame_w))))
        top = max(0, min(map_h, int(math.floor(y1 * map_h / frame_h))))
        bottom = max(0, min(map_h, int(math.ceil(y2 * map_h / frame_h))))
        if right <= left or bottom <= top:
            return False
        region = values[top:bottom, left:right]
        return bool(np.any(np.isfinite(region) & (region > 0.0)))

    @staticmethod
    def _appearance_map_components(
        score_map: np.ndarray | None,
        frame_shape: tuple[int, ...],
        *,
        max_components: int = 32,
    ) -> list[tuple[float, np.ndarray, list[int]]]:
        """Partition a decision heatmap into localized connected incidents.

        Support is defined at 50% of the finite map range. A constant positive
        map remains one conservative full-map incident. Returned bboxes use
        original-frame XYXY coordinates.
        """
        if not isinstance(score_map, np.ndarray) or score_map.size == 0:
            return []
        values = np.asarray(score_map, dtype=np.float32).squeeze()
        if values.ndim != 2 or len(frame_shape) < 2:
            return []
        finite = np.isfinite(values)
        if not np.any(finite):
            return []
        finite_values = values[finite]
        peak = float(np.max(finite_values))
        floor = float(np.min(finite_values))
        if peak <= 0.0:
            return []
        if peak - floor <= 1e-6:
            support = finite & (values > 0.0)
        else:
            support = finite & (values >= floor + 0.5 * (peak - floor))
        if not np.any(support):
            return []

        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            support.astype(np.uint8),
            connectivity=8,
        )
        map_h, map_w = values.shape
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        components: list[tuple[float, np.ndarray, list[int], int]] = []
        for label in range(1, count):
            x, y, width, height, area = (
                int(value) for value in stats[label]
            )
            if area <= 0 or width <= 0 or height <= 0:
                continue
            component_support = labels == label
            component_values = np.zeros_like(values, dtype=np.float32)
            component_values[component_support] = values[component_support]
            component_score = float(np.max(values[component_support]))
            left = max(0, min(frame_w, int(math.floor(x * frame_w / map_w))))
            top = max(0, min(frame_h, int(math.floor(y * frame_h / map_h))))
            right = max(
                0,
                min(frame_w, int(math.ceil((x + width) * frame_w / map_w))),
            )
            bottom = max(
                0,
                min(frame_h, int(math.ceil((y + height) * frame_h / map_h))),
            )
            if right <= left or bottom <= top:
                continue
            components.append((
                component_score,
                component_values,
                [left, top, right, bottom],
                area,
            ))
        components.sort(key=lambda item: (-item[0], -item[3], item[2]))
        return [
            (score, component_map, bbox)
            for score, component_map, bbox, _area in components[:max_components]
        ]

    @staticmethod
    def _merge_visual_maps_max(
        channel_maps,
    ) -> np.ndarray | None:
        """Pixelwise-max all finite visual maps in one spatial canvas.

        Classifier and stationary-track suppression remove the appearance and
        Simplex channels.  Keeping only the highest-scoring independent map at
        that point would erase a lower-scoring hazard in another include zone.
        """
        valid_maps: list[np.ndarray] = []
        for channel_map in channel_maps:
            if not isinstance(channel_map, np.ndarray):
                continue
            values = np.asarray(channel_map, dtype=np.float32).squeeze()
            if values.ndim != 2 or values.size == 0:
                continue
            if not np.all(np.isfinite(values)):
                continue
            valid_maps.append(values)
        if not valid_maps:
            return None

        target = max(valid_maps, key=lambda values: values.size)
        target_h, target_w = target.shape
        merged = np.zeros((target_h, target_w), dtype=np.float32)
        for values in valid_maps:
            if values.shape != (target_h, target_w):
                values = cv2.resize(
                    values,
                    (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            np.maximum(merged, values, out=merged)
        return merged

    def _object_safety_score(
        self,
        detected_objects: list[dict] | None,
        *,
        zone_mask: np.ndarray | None = None,
        require_contributed: bool = False,
    ) -> float:
        """Return confidence for explicitly allowlisted non-person detections.

        With a zone mask, malformed or out-of-zone boxes fail closed and cannot
        contribute a score to that zone.
        """
        if not detected_objects:
            return 0.0

        config = self.camera_config.person_filter
        alert_ids = set(config.alert_class_ids)
        if not alert_ids:
            return 0.0

        best = 0.0
        for detected_object in detected_objects:
            if detected_object.get("object_safety_alert_eligible") is not True:
                continue
            if (
                require_contributed
                and detected_object.get("object_safety_alert_contributed") is not True
            ):
                continue
            try:
                class_id = int(detected_object["class_id"])
                confidence = float(detected_object["object_safety_score"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                class_id not in alert_ids
                or not math.isfinite(confidence)
                or confidence < config.alert_min_confidence
            ):
                continue

            if zone_mask is not None and not self._object_overlaps_zone_mask(
                detected_object,
                zone_mask,
            ):
                continue

            best = max(best, min(confidence, 1.0))

        return best

    @staticmethod
    def _object_overlaps_zone_mask(
        detected_object: dict,
        zone_mask: np.ndarray,
    ) -> bool:
        """Return whether a finite, non-empty bbox overlaps the zone mask."""
        bbox = detected_object.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        try:
            x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
        except (OverflowError, TypeError, ValueError):
            return False
        height, width = zone_mask.shape[:2]
        left = max(0, min(x1, x2))
        right = min(width, max(x1, x2))
        top = max(0, min(y1, y2))
        bottom = min(height, max(y1, y2))
        return bool(
            right > left
            and bottom > top
            and np.any(zone_mask[top:bottom, left:right] > 0)
        )

    def _detected_objects_for_zone(
        self,
        detected_objects: list[dict] | None,
        *,
        zone_mask: np.ndarray | None = None,
    ) -> list[dict]:
        """Scope localized evidence and object-safety contribution to one zone."""
        scoped: list[dict] = []
        for detected_object in detected_objects or []:
            if zone_mask is not None and not self._object_overlaps_zone_mask(
                detected_object,
                zone_mask,
            ):
                continue
            item = dict(detected_object)
            item["object_safety_alert_contributed"] = (
                item.get("object_safety_alert_eligible") is True
            )
            scoped.append(item)
        return scoped

    def _get_zone_polygon_mask(
        self,
        zone_id: str,
        polygon: list[tuple[int, int]],
        shape: tuple[int, int],
        *,
        source_shape: tuple[int, int],
    ) -> np.ndarray:
        """Return a cached binary zone mask at the requested resolution."""
        mask_h, mask_w = shape
        source_h, source_w = source_shape
        cache_key = (zone_id, mask_h, mask_w, source_h, source_w)
        zone_mask = self._zone_mask_cache.get(cache_key)
        if zone_mask is not None:
            self._zone_mask_cache.move_to_end(cache_key)
            return zone_mask

        scale_x = mask_w / max(source_w, 1)
        scale_y = mask_h / max(source_h, 1)
        scaled_pts = np.array(
            [(int(x * scale_x), int(y * scale_y)) for x, y in polygon],
            dtype=np.int32,
        )
        zone_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        cv2.fillPoly(zone_mask, [scaled_pts], 255)
        if len(self._zone_mask_cache) >= self._zone_mask_cache_limit:
            self._zone_mask_cache.popitem(last=False)
        self._zone_mask_cache[cache_key] = zone_mask
        return zone_mask

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
        zone_mask = self._get_zone_polygon_mask(
            zone_id,
            polygon,
            (map_h, map_w),
            source_shape=frame_shape[:2],
        )

        zone_values = anomaly_map[zone_mask > 0]
        if zone_values.size == 0:
            return 0.0
        return float(zone_values.max())

    def _mask_anomaly_map_to_zone(
        self,
        anomaly_map: np.ndarray,
        zone_id: str,
        polygon: list[tuple[int, int]],
        frame_shape: tuple,
    ) -> np.ndarray:
        """Return display evidence with every pixel outside one zone cleared."""
        map_h, map_w = anomaly_map.shape[:2]
        zone_mask = self._get_zone_polygon_mask(
            zone_id,
            polygon,
            (map_h, map_w),
            source_shape=frame_shape[:2],
        )
        return np.where(
            zone_mask > 0,
            anomaly_map,
            np.zeros((), dtype=anomaly_map.dtype),
        )

    def _update_lock_state(
        self,
        anomaly_result: AnomalyResult,
        *,
        allow_engage: bool = True,
    ) -> None:
        """Update the anomaly region lock based on detection results.

        HIGH-05: Uses hysteresis — lock engages at lock_score_threshold,
        clears only when score drops below lock_clear_threshold (80% of lock threshold).
        HIGH-03: Uses time-based clearing instead of frame count.
        """
        now = time.monotonic()
        with self._lock_state_lock:
            if not allow_engage:
                if self._locked:
                    self._locked = False
                    self._lock_last_below_time = None
                    logger.info(
                        "pipeline.lock_cleared",
                        camera_id=self.camera_config.camera_id,
                        reason="lock_disabled_for_detector",
                    )
                return
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
        from argus.alerts.grader import HANDLING_POLICIES
        from argus.config.schema import AlertSeverity

        levels = [AlertSeverity.INFO, AlertSeverity.LOW, AlertSeverity.MEDIUM, AlertSeverity.HIGH]
        current_idx = levels.index(alert.severity) if alert.severity in levels else 0
        if escalate:
            new_idx = min(current_idx + 1, len(levels) - 1)
        else:
            new_idx = max(current_idx - 1, 0)
        original = alert.severity
        alert.severity = levels[new_idx]
        alert.handling_policy = HANDLING_POLICIES[alert.severity]
        if isinstance(alert.evidence_json, dict):
            previous_summary = alert.evidence_json.get("summary") or alert.evidence_summary
            alert.evidence_json["severity"] = alert.severity.value
            alert.evidence_json["classifier_adjustment"] = {
                "original_severity": original.value,
                "applied_severity": alert.severity.value,
                "label": alert.classification_label,
                "confidence": alert.classification_confidence,
            }
            if previous_summary:
                alert.evidence_json.setdefault("original_summary", previous_summary)
            from argus.alerts.evidence import DetectionEvidence

            evidence = DetectionEvidence.from_dict({
                **alert.evidence_json, "summary": None, "evidence_summary": None,
            })
            alert.evidence_summary = evidence.evidence_summary()
            alert.evidence_json["summary"] = alert.evidence_summary
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
