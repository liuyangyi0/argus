"""Alert severity grading with temporal confirmation and deduplication.

This module decides whether an anomaly detection result should become an alert,
what severity level it should have, and whether it's a duplicate of a recent alert.
The temporal confirmation filter requires anomalies to persist across multiple
consecutive frames before triggering, which dramatically reduces false positives.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import structlog

from pathlib import Path

if TYPE_CHECKING:
    from argus.core.alert_ring_buffer import SolidifiedRecording
    from argus.sensors.fusion import SensorFusion

from argus.alerts.evidence import DetectionEvidence
from argus.config.schema import AlertConfig, AlertSeverity, SeverityThresholds, ZonePriority
from argus.contracts.validation import validate_alert
from argus.core.metrics import METRICS

logger = structlog.get_logger()

# UX v2 §5.1: Canonical severity → handling policy mapping
HANDLING_POLICIES: dict[AlertSeverity, str] = {
    AlertSeverity.INFO: "quick",
    AlertSeverity.LOW: "quick",
    AlertSeverity.MEDIUM: "confirm",
    AlertSeverity.HIGH: "detail_required",
}

# F1 early-warning: detection classes that do NOT on their own justify a
# fast-path fire. "Person" intrusions still require CUSUM corroboration because
# a transient person walk-through is expected behaviour in many zones.
_EARLY_WARNING_IGNORED_DETECTION_CLASSES: frozenset[str] = frozenset({"person"})


class AlertCategory(str, Enum):
    PROJECTILE = "projectile"
    STATIC_FOREIGN = "static_foreign"
    WATER_LEAK = "water_leak"
    VISIBLE_PLUME = "visible_plume"
    SCENE_CHANGE = "scene_change"
    ENVIRONMENTAL = "environmental"
    PERSON_INTRUSION = "person_intrusion"
    EQUIPMENT_DISPLACEMENT = "equipment_displacement"
    UNKNOWN = "unknown"


@dataclass
class Alert:
    """A graded alert ready for dispatch."""

    alert_id: str
    camera_id: str
    zone_id: str
    severity: AlertSeverity
    anomaly_score: float
    timestamp: float
    frame_number: int
    snapshot: np.ndarray | None = None
    heatmap: np.ndarray | None = None
    # YOLO-005: Semantic detection context
    detection_type: str = "anomaly"  # DetectionType value: "anomaly", "object", "hybrid"
    detected_objects: list[dict] = field(default_factory=list)
    evidence_summary: str = ""
    evidence_json: dict = field(default_factory=dict)
    suppression_reason: str | None = None
    # C3-4: Cross-camera correlation
    corroborated: bool = True
    correlation_partner: str | None = None
    # C4-3: Model version tracking
    model_version_id: str | None = None
    # D1: Open vocabulary classification
    classification_label: str | None = None
    classification_confidence: float | None = None
    severity_adjusted_by_classifier: bool = False
    # Alert category classification
    category: str = "unknown"
    # F1: Single-frame early-warning fast path (bypasses CUSUM evidence wait)
    triggered_by_early_warning: bool = False
    # D2: Instance segmentation
    segmentation_count: int = 0
    segmentation_total_area_px: int = 0
    segmentation_objects: list[dict] = field(default_factory=list)
    # UX v2 §5.1: Severity-based handling policy
    # "quick" (LOW), "confirm" (MEDIUM), "detail_required" (HIGH)
    handling_policy: str = "quick"
    # Alert aggregation: alerts in same camera+zone within a window share this ID
    event_group_id: str | None = None
    event_group_count: int = 1
    # Physics enrichment (phase 1: speed, phase 2: trajectory/localization)
    speed_ms: float | None = None
    speed_px_per_sec: float | None = None
    trajectory_model: str | None = None  # "free_fall", "projectile", or "projectile_drag"
    origin_x_mm: float | None = None
    origin_y_mm: float | None = None
    origin_z_mm: float | None = None
    landing_x_mm: float | None = None
    landing_y_mm: float | None = None
    landing_z_mm: float | None = None
    # Trajectory centroid history for visualization (max 300 points)
    trajectory_points: list[tuple[float, float, float]] | None = None
    # Per-track trajectory fits (primary + others) serialised as JSON for persistence
    trajectories_json: str | None = None
    # Set by pipeline after alert recording is solidified to disk
    _solidified_recording: SolidifiedRecording | None = None


class DetectionType(str, Enum):
    """Type of detection that triggered an alert (YOLO-005)."""

    ANOMALY = "anomaly"  # Anomalib only
    OBJECT = "object"  # YOLO object detection only
    HYBRID = "hybrid"  # Both YOLO and Anomalib agree
    PROJECTILE = "projectile"  # Fast-motion fly-through detector
    PERSISTENT_CHANGE = "persistent_change"  # Fixed-scene added/left-behind object
    WATER_LEAK = "water_leak"  # Visible drip/leak/pooling signs
    PLUME = "plume"  # Visible smoke/steam/dust/mist signs


@dataclass(frozen=True)
class CusumSnapshot:
    """Read-only snapshot of CUSUM evidence state for a zone.

    Exposed via AlertGrader.get_cusum_state() so that CameraInferenceRunner
    can include per-zone evidence in RunnerSnapshot without breaking
    AlertGrader encapsulation.
    """

    evidence: float
    first_seen: float
    last_seen: float
    max_score: float


@dataclass(frozen=True)
class GradeResult:
    """Structured grader outcome for alert and non-alert explanations."""

    should_alert: bool
    severity: str | None
    score: float
    detection_type: str
    evidence: DetectionEvidence
    suppression_reason: str | None = None
    alert: Alert | None = None

    def to_dict(self) -> dict:
        return {
            "should_alert": self.should_alert,
            "severity": self.severity,
            "score": self.score,
            "detection_type": self.detection_type,
            "evidence": self.evidence.to_dict(),
            "suppression_reason": self.suppression_reason,
            "alert_id": self.alert.alert_id if self.alert is not None else None,
        }


@dataclass
class _AnomalyTracker:
    """Tracks anomaly evidence for a specific zone using exponential accumulation."""

    evidence: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    max_score: float = 0.0
    prev_anomaly_mask: np.ndarray | None = None


@dataclass(frozen=True)
class _SpatialAlert:
    """A recent localized alert used for incident-aware deduplication."""

    camera_id: str
    zone_key: str
    detection_type: str
    bbox: tuple[float, float, float, float] | None
    incident_key: str | None
    timestamp: float


_SPATIAL_DEDUP_DETECTION_TYPES: frozenset[str] = frozenset({
    DetectionType.OBJECT.value,
    DetectionType.HYBRID.value,
    DetectionType.PROJECTILE.value,
    DetectionType.PERSISTENT_CHANGE.value,
    DetectionType.WATER_LEAK.value,
    DetectionType.PLUME.value,
})
_SPATIAL_DEDUP_IOU_THRESHOLD = 0.30


class AlertGrader:
    """Grades anomaly detections into alerts with temporal filtering.

    The grading pipeline:
    1. Apply zone priority multiplier to the raw anomaly score
    2. Map the adjusted score to a severity level via thresholds
    3. Track consecutive detections per zone (temporal confirmation)
    4. Only emit an alert when the anomaly persists for N consecutive frames
    5. Suppress duplicate alerts within configurable time windows
    """

    def __init__(
        self,
        config: AlertConfig,
        node_id: str = "edge",
        calibration_path: Path | None = None,
        sensor_fusion: SensorFusion | None = None,
    ):
        self._config = config
        self._node_id = node_id
        self._sensor_fusion = sensor_fusion
        self._trackers: dict[str, _AnomalyTracker] = defaultdict(_AnomalyTracker)
        self._last_alerts: dict[str, float] = {}  # zone_key -> last alert timestamp
        self._last_camera_alerts: dict[str, float] = {}  # camera_id -> last alert timestamp
        self._recent_spatial_alerts: list[_SpatialAlert] = []
        # Alert aggregation: zone_key -> (event_group_id, first_alert_time, count)
        self._event_groups: dict[str, tuple[str, float, int]] = {}
        self._aggregation_window = getattr(config, "aggregation_window_seconds", 300.0)  # 5 min
        # Random offset avoids ID collisions across process restarts
        self._alert_counter = random.randint(1000, 9999)
        self._tracker_lock = threading.Lock()
        self._eval_count = 0
        self._prune_interval = 1000  # auto-prune every N evaluations

        # Load calibrated thresholds if available and enabled
        if calibration_path and self._config.calibration.enabled:
            from argus.alerts.calibration import ConformalCalibrator

            calibrator = ConformalCalibrator()
            cal = calibrator.load(calibration_path)
            if cal:
                self._config.severity_thresholds = SeverityThresholds(
                    info=cal.info_threshold,
                    low=cal.low_threshold,
                    medium=cal.medium_threshold,
                    high=cal.high_threshold,
                )
                logger.info("grader.using_calibrated_thresholds", thresholds=str(cal))
            else:
                logger.warning("grader.calibration_not_found", path=str(calibration_path))

    @staticmethod
    def _normalize_incident_key(incident_key: str | None) -> str | None:
        """Normalize an optional upstream incident/track identity."""
        if incident_key is None:
            return None
        normalized = incident_key.strip()
        return normalized or None

    @classmethod
    def _tracking_detection_type(
        cls,
        detection_type: str,
        incident_key: str | None,
    ) -> str:
        """Return the stable temporal family for an emitted detection type.

        Appearance incidents may be emitted as ``hybrid`` on frames where an
        overlapping detector object corroborates the heatmap and ``anomaly``
        on adjacent frames where it does not.  That display/evidence detail
        must not split one physical incident into two CUSUM/cooldown streams.
        Other channel types remain strictly isolated.
        """
        normalized_incident = cls._normalize_incident_key(incident_key)
        if (
            detection_type in {
                DetectionType.ANOMALY.value,
                DetectionType.HYBRID.value,
            }
            and normalized_incident is not None
            and ":appearance_anomaly:" in f":{normalized_incident}:"
        ):
            return DetectionType.ANOMALY.value
        return detection_type

    @classmethod
    def _tracker_key(
        cls,
        zone_key: str,
        detection_type: str,
        incident_key: str | None = None,
    ) -> str:
        """Keep temporal evidence isolated by zone, channel, and incident.

        The legacy anomaly key is preserved for stored state and callers that
        inspect ``camera:zone`` directly. Localized/semantic channels receive
        an explicit suffix, and an upstream incident identity receives a final
        suffix so separate tracks cannot contribute evidence to one another.
        """
        normalized_incident = cls._normalize_incident_key(incident_key)
        tracking_type = cls._tracking_detection_type(
            detection_type,
            normalized_incident,
        )
        if normalized_incident is not None:
            return (
                f"{zone_key}:{tracking_type}:incident:"
                f"{normalized_incident}"
            )
        if tracking_type == DetectionType.ANOMALY.value:
            return zone_key
        return f"{zone_key}:{tracking_type}"

    @classmethod
    def _camera_suppression_key(
        cls,
        camera_id: str,
        zone_key: str,
        detection_type: str,
        incident_key: str | None = None,
    ) -> str:
        """Scope cooldowns so explicit zones cannot suppress one another.

        A camera without configured include zones keeps the historical camera
        key through its ``default`` zone.  Explicit zones are independent
        safety scopes; their same-zone window still deduplicates repeats.
        """
        tracking_type = cls._tracking_detection_type(
            detection_type,
            incident_key,
        )
        type_suffix = (
            "" if tracking_type == DetectionType.ANOMALY.value
            else f":{tracking_type}"
        )
        if zone_key == f"{camera_id}:default":
            suppression_key = f"{camera_id}{type_suffix}"
        else:
            suppression_key = f"{zone_key}{type_suffix}"
        normalized_incident = cls._normalize_incident_key(incident_key)
        if normalized_incident is not None:
            return f"{suppression_key}:incident:{normalized_incident}"
        return suppression_key

    def evaluate(
        self,
        camera_id: str,
        zone_id: str,
        zone_priority: ZonePriority,
        anomaly_score: float,
        frame_number: int,
        frame: np.ndarray | None = None,
        anomaly_map: np.ndarray | None = None,
        detection_type: str = "anomaly",
        detected_objects: list[dict] | None = None,
        classification_label: str | None = None,
        classification_confidence: float | None = None,
        continuity_map: np.ndarray | None = None,
        *,
        incident_key: str | None = None,
    ) -> Alert | None:
        """Evaluate an anomaly detection and potentially produce an alert.

        Returns an Alert if the anomaly passes all filters, None otherwise.
        """
        detection_type = getattr(detection_type, "value", str(detection_type))
        incident_key = self._normalize_incident_key(incident_key)
        now = time.monotonic()
        zone_key = f"{camera_id}:{zone_id}"
        tracker_key = self._tracker_key(zone_key, detection_type, incident_key)

        # Auto-prune stale trackers periodically to prevent memory growth
        self._eval_count += 1
        if self._eval_count % self._prune_interval == 0:
            self.prune_stale_trackers()

        # Defense-in-depth: reject NaN/Inf scores
        if not math.isfinite(anomaly_score):
            logger.error(
                "grader.invalid_score",
                zone=zone_key,
                raw_score=str(anomaly_score),
                msg="NaN/Inf score rejected",
            )
            return None

        # Step 1: Apply zone priority multiplier
        multiplier = self._config.zone_multipliers.get(zone_priority.value, 1.0)
        adjusted_score = anomaly_score * multiplier

        # Step 1b: External sensor fusion — bias severity via (camera, zone) signal
        if self._sensor_fusion is not None:
            fusion_multiplier = self._sensor_fusion.get_multiplier(camera_id, zone_id)
            if fusion_multiplier != 1.0:
                adjusted_score *= fusion_multiplier
                logger.debug(
                    "grader.fusion_applied",
                    zone=zone_key,
                    fusion_multiplier=round(fusion_multiplier, 3),
                    adjusted_score=round(adjusted_score, 3),
                )

        adjusted_score = max(0.0, min(adjusted_score, 1.0))

        # Step 2: Determine severity (None if below info threshold)
        severity = self._score_to_severity(adjusted_score)

        # F1: Single-frame early-warning fast path.  When anomaly + detection (or
        # classifier) both fire strongly on one frame, bypass CUSUM evidence
        # accumulation but still respect suppression windows so we don't storm.
        ew_cfg = getattr(self._config, "early_warning", None)
        if (
            ew_cfg is not None
            and ew_cfg.enabled
            and severity is not None
            and adjusted_score >= ew_cfg.score_threshold
            and self._early_warning_corroborated(
                ew_cfg, detection_type, detected_objects,
                classification_label, classification_confidence,
            )
        ):
            ew_alert = self._fire_early_warning(
                camera_id=camera_id,
                zone_id=zone_id,
                zone_key=zone_key,
                adjusted_score=adjusted_score,
                severity=severity,
                frame_number=frame_number,
                frame=frame,
                anomaly_map=anomaly_map,
                detection_type=detection_type,
                detected_objects=detected_objects,
                classification_label=classification_label,
                classification_confidence=classification_confidence,
                now=now,
                incident_key=incident_key,
            )
            if ew_alert is not None:
                return ew_alert

        # Step 3: CUSUM temporal evidence accumulation (thread-safe)
        with self._tracker_lock:
            tracker = self._trackers[tracker_key]
            gap = now - tracker.last_seen if tracker.last_seen > 0 else 0
            lam = self._config.temporal.evidence_lambda

            # Sub-threshold frame: pure decay, clean up if negligible
            if severity is None:
                tracker.evidence *= lam
                if tracker.evidence < 0.01:
                    if incident_key is None:
                        self._trackers[tracker_key] = _AnomalyTracker()
                    else:
                        # Localized identities are high-cardinality. A
                        # sub-threshold observation must not leave a permanent
                        # empty entry whose ``last_seen`` can never age out.
                        self._trackers.pop(tracker_key, None)
                logger.debug(
                    "grader.below_threshold",
                    zone=zone_key,
                    score=round(adjusted_score, 3),
                    evidence=round(tracker.evidence, 3),
                )
                return None

            # Gap timeout: reset evidence entirely
            if gap > self._config.temporal.max_gap_seconds:
                tracker.evidence = 0.0
                tracker.first_seen = now
                tracker.last_seen = now
                tracker.max_score = adjusted_score

            tracker.last_seen = now

            # Step 3b: Spatial continuity check — anomaly region must overlap with
            # the previous frame's anomaly region to filter transient noise.
            #
            # F3: `spatial_continuity_enabled` + `spatial_continuity_iou_threshold`
            # give operators explicit control over the IoU gate (default 0.7, was
            # effectively 0.3 via min_spatial_overlap). `spatial_continuity_mode`
            # chooses between fully resetting evidence on mismatch ("reset", new
            # stricter default) and the legacy 50% decay ("decay"). We fall back
            # on the legacy `min_spatial_overlap` when the new-style gate is
            # explicitly disabled, so existing configs keep behaving the same.
            temporal_cfg = self._config.temporal
            legacy_overlap = temporal_cfg.min_spatial_overlap
            if getattr(temporal_cfg, "spatial_continuity_enabled", True):
                sc_threshold = float(
                    getattr(
                        temporal_cfg,
                        "spatial_continuity_iou_threshold",
                        legacy_overlap,
                    )
                )
                sc_mode = getattr(temporal_cfg, "spatial_continuity_mode", "reset")
            else:
                sc_threshold = legacy_overlap
                sc_mode = "decay"  # preserve historical 50% decay when new gate is off

            explicit_continuity = continuity_map is not None
            spatial_map = continuity_map if explicit_continuity else anomaly_map
            if sc_threshold > 0 and spatial_map is not None:
                curr_mask = (
                    spatial_map > (0.0 if explicit_continuity else 0.5)
                ).astype(np.uint8)
                if tracker.prev_anomaly_mask is not None:
                    prev_mask = tracker.prev_anomaly_mask
                    # Resize to match if anomaly map dimensions changed (e.g. tiling mode switch)
                    if curr_mask.shape != prev_mask.shape:
                        import cv2
                        prev_mask = cv2.resize(
                            prev_mask, (curr_mask.shape[1], curr_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        # Update tracker to avoid repeated resize on subsequent frames
                        tracker.prev_anomaly_mask = prev_mask
                        logger.debug(
                            "grader.spatial_resize",
                            zone=zone_key,
                            prev_shape=tracker.prev_anomaly_mask.shape,
                            curr_shape=curr_mask.shape,
                        )
                    intersection = np.count_nonzero(curr_mask & prev_mask)
                    union = np.count_nonzero(curr_mask | prev_mask)
                    iou = intersection / union if union > 0 else 0.0
                    if iou < sc_threshold:
                        logger.debug(
                            "grader.spatial_mismatch",
                            zone=zone_key,
                            iou=round(iou, 3),
                            threshold=sc_threshold,
                            mode=sc_mode,
                        )
                        if sc_mode == "reset":
                            # F3: hard reset — local-motion false positives that
                            # keep scoring high but on shifting pixels no longer
                            # accumulate evidence.
                            tracker.evidence = 0.0
                            tracker.first_seen = now
                            tracker.max_score = adjusted_score
                            try:
                                METRICS.spatial_continuity_resets.labels(
                                    camera_id=camera_id,
                                ).inc()
                            except Exception:
                                # Metrics must never break the grader.
                                logger.debug(
                                    "grader.spatial_reset_metric_failed",
                                    exc_info=True,
                                )
                        else:
                            # Legacy 50% decay to tolerate camera vibration.
                            tracker.evidence *= 0.5
                            tracker.max_score = adjusted_score
                # Keep only the most recent mask — bounded memory per tracker.
                tracker.prev_anomaly_mask = curr_mask

            # Exponential evidence accumulation
            tracker.evidence = lam * tracker.evidence + adjusted_score
            tracker.max_score = max(tracker.max_score, adjusted_score)

            if tracker.evidence < self._config.temporal.evidence_threshold:
                logger.debug(
                    "grader.accumulating_evidence",
                    zone=zone_key,
                    evidence=round(tracker.evidence, 3),
                    threshold=self._config.temporal.evidence_threshold,
                )
                return None

            # Step 4: Check suppression windows. Localized hazards are
            # deduplicated by zone, detection type, and incident position; an
            # independent hazard in another zone or channel must still alert.
            suppression_reason = self._suppression_reason(
                camera_id=camera_id,
                zone_key=zone_key,
                detection_type=detection_type,
                detected_objects=detected_objects,
                now=now,
                incident_key=incident_key,
            )
            if suppression_reason is not None:
                logger.debug(
                    "grader.suppressed",
                    zone=zone_key,
                    camera=camera_id,
                    score=round(adjusted_score, 3),
                    reason=suppression_reason,
                )
                return None

            # Step 5: Emit alert
            self._alert_counter += 1
            seq = self._alert_counter
            self._last_alerts[tracker_key] = now
            self._last_camera_alerts[
                self._camera_suppression_key(
                    camera_id,
                    zone_key,
                    detection_type,
                    incident_key,
                )
            ] = now
            self._record_spatial_alert(
                camera_id=camera_id,
                zone_key=zone_key,
                detection_type=detection_type,
                detected_objects=detected_objects,
                now=now,
                incident_key=incident_key,
            )

            # Use the max score seen during the confirmation window for severity
            final_severity = self._score_to_severity(tracker.max_score) or severity

            # Timestamp-based ID with random suffix for collision resistance
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S%f")
            rnd = random.randint(0, 0xFFFF)
            alert_id = f"ALT-{self._node_id}-{ts}-{seq:04d}-{rnd:04x}"

            # Alert aggregation: assign event_group_id
            event_group_id, event_group_count = self._assign_event_group(
                zone_key, alert_id, now,
            )

            alert = Alert(
                alert_id=alert_id,
                camera_id=camera_id,
                zone_id=zone_id,
                severity=final_severity,
                anomaly_score=tracker.max_score,
                timestamp=time.time(),
                frame_number=frame_number,
                snapshot=frame,
                heatmap=anomaly_map,
                detection_type=detection_type,
                detected_objects=detected_objects or [],
                handling_policy=HANDLING_POLICIES.get(final_severity, "quick"),
                event_group_id=event_group_id,
                event_group_count=event_group_count,
            )

            # Reset tracker after emitting alert
            if incident_key is None:
                self._trackers[tracker_key] = _AnomalyTracker()
            else:
                # Incident identities can be high-cardinality (for example,
                # projectile track IDs), so do not retain an empty tracker for
                # every completed incident.
                self._trackers.pop(tracker_key, None)

        # Data contract validation at grader→dispatcher boundary (advisory only)
        if alert is not None:
            try:
                validate_alert(alert)
            except Exception:
                logger.debug("grader.contract_validation_failed", exc_info=True)

        return alert

    def evaluate_with_evidence(
        self,
        camera_id: str,
        zone_id: str,
        zone_priority: ZonePriority,
        anomaly_score: float,
        frame_number: int,
        frame: np.ndarray | None = None,
        anomaly_map: np.ndarray | None = None,
        detection_type: str = "anomaly",
        detected_objects: list[dict] | None = None,
        classification_label: str | None = None,
        classification_confidence: float | None = None,
        continuity_map: np.ndarray | None = None,
        *,
        incident_key: str | None = None,
    ) -> GradeResult:
        """Evaluate and return a structured explanation result.

        ``evaluate()`` remains the compatibility API. This method calls it
        exactly once, then attaches evidence to any emitted Alert and returns
        a suppression reason for non-alert candidates.
        """
        detection_type_value = getattr(detection_type, "value", str(detection_type))
        incident_key = self._normalize_incident_key(incident_key)
        zone_key = f"{camera_id}:{zone_id}"
        now_before = time.monotonic()

        if not math.isfinite(anomaly_score):
            evidence = DetectionEvidence(
                detection_type=detection_type_value,
                score=None,
                camera_id=camera_id,
                zone_id=zone_id,
                frame_number=frame_number,
                features={"raw_score": str(anomaly_score)},
                suppression_reason="invalid_score",
            )
            return GradeResult(
                should_alert=False,
                severity=None,
                score=0.0,
                detection_type=detection_type_value,
                evidence=evidence,
                suppression_reason="invalid_score",
            )

        multiplier = self._config.zone_multipliers.get(zone_priority.value, 1.0)
        adjusted_score = anomaly_score * multiplier
        fusion_multiplier = 1.0
        if self._sensor_fusion is not None:
            fusion_multiplier = self._sensor_fusion.get_multiplier(camera_id, zone_id)
            adjusted_score *= fusion_multiplier
        adjusted_score = max(0.0, min(adjusted_score, 1.0))
        severity = self._score_to_severity(adjusted_score)
        suppression_before = self._suppression_reason(
            camera_id=camera_id,
            zone_key=zone_key,
            detection_type=detection_type_value,
            detected_objects=detected_objects,
            now=now_before,
            incident_key=incident_key,
        )

        alert = self.evaluate(
            camera_id=camera_id,
            zone_id=zone_id,
            zone_priority=zone_priority,
            anomaly_score=anomaly_score,
            frame_number=frame_number,
            frame=frame,
            anomaly_map=anomaly_map,
            continuity_map=continuity_map,
            detection_type=detection_type_value,
            detected_objects=detected_objects,
            classification_label=classification_label,
            classification_confidence=classification_confidence,
            incident_key=incident_key,
        )

        evidence_features: dict = {
            "zone_priority": zone_priority.value,
            "zone_multiplier": multiplier,
        }
        if fusion_multiplier != 1.0:
            evidence_features["sensor_fusion_multiplier"] = fusion_multiplier
        if detected_objects:
            evidence_features["detected_object_count"] = len(detected_objects)
        if classification_label is not None:
            evidence_features["classification"] = {
                "label": classification_label,
                "confidence": classification_confidence,
            }
        if incident_key is not None:
            evidence_features["incident_key"] = incident_key

        if alert is not None:
            severity_value = alert.severity.value
            evidence = DetectionEvidence(
                detection_type=detection_type_value,
                score=float(alert.anomaly_score),
                camera_id=camera_id,
                zone_id=zone_id,
                frame_number=frame_number,
                severity=severity_value,
                features=evidence_features,
                candidates=detected_objects or [],
                model_version_id=getattr(alert, "model_version_id", None),
            )
            alert.evidence_summary = evidence.evidence_summary()
            alert.evidence_json = evidence.to_dict()
            alert.suppression_reason = None
            return GradeResult(
                should_alert=True,
                severity=severity_value,
                score=float(alert.anomaly_score),
                detection_type=detection_type_value,
                evidence=evidence,
                suppression_reason=None,
                alert=alert,
            )

        if severity is None:
            reason = "score_below_threshold"
        elif suppression_before is not None:
            reason = suppression_before
        else:
            reason = "duration_too_short"

        evidence = DetectionEvidence(
            detection_type=detection_type_value,
            score=float(adjusted_score),
            camera_id=camera_id,
            zone_id=zone_id,
            frame_number=frame_number,
            severity=severity.value if severity is not None else None,
            features=evidence_features,
            candidates=detected_objects or [],
            suppression_reason=reason,
        )
        return GradeResult(
            should_alert=False,
            severity=severity.value if severity is not None else None,
            score=float(adjusted_score),
            detection_type=detection_type_value,
            evidence=evidence,
            suppression_reason=reason,
        )

    @staticmethod
    def _early_warning_corroborated(
        ew_cfg,
        detection_type: str,
        detected_objects: list[dict] | None,
        classification_label: str | None,
        classification_confidence: float | None,
    ) -> bool:
        """Return True if the frame carries enough corroboration to take the F1 fast path.

        With ``require_detection_or_classifier=True`` (default), one of the following
        must hold:
          - ``detected_objects`` contains at least one non-person class
          - ``detection_type == "hybrid"`` (YOLO and anomaly both agree)
          - A classifier label fired with confidence ≥ ``classifier_min_confidence``

        With the guard disabled, pure anomaly-only frames also qualify.
        """
        if not ew_cfg.require_detection_or_classifier:
            return True

        # Detection corroboration: at least one non-person detected object.
        if detected_objects:
            for obj in detected_objects:
                class_name = (obj.get("class_name") or "").strip().lower()
                if class_name and class_name not in _EARLY_WARNING_IGNORED_DETECTION_CLASSES:
                    return True

        # Hybrid mode: both YOLO and anomaly agree, so even if the class filter
        # above excluded "person" we trust the hybrid flag itself.
        if detection_type == "hybrid":
            return True

        # Classifier corroboration: high-confidence open-vocabulary label.
        if (
            isinstance(classification_label, str)
            and bool(classification_label.strip())
            and isinstance(classification_confidence, Real)
            and not isinstance(classification_confidence, bool)
            and math.isfinite(classification_confidence)
            and 0.0 <= classification_confidence <= 1.0
            and classification_confidence >= ew_cfg.classifier_min_confidence
        ):
            return True

        return False

    def _fire_early_warning(
        self,
        *,
        camera_id: str,
        zone_id: str,
        zone_key: str,
        adjusted_score: float,
        severity: AlertSeverity,
        frame_number: int,
        frame: np.ndarray | None,
        anomaly_map: np.ndarray | None,
        detection_type: str,
        detected_objects: list[dict] | None,
        classification_label: str | None,
        classification_confidence: float | None,
        now: float,
        incident_key: str | None,
    ) -> Alert | None:
        """Build and emit an early-warning alert, honoring suppression windows.

        Returns None when a suppression window rejects the fire, letting the
        caller fall through to the standard CUSUM path (which will also no-op
        on the same window).
        """
        detection_type = getattr(detection_type, "value", str(detection_type))
        with self._tracker_lock:
            suppression_reason = self._suppression_reason(
                camera_id=camera_id,
                zone_key=zone_key,
                detection_type=detection_type,
                detected_objects=detected_objects,
                now=now,
                incident_key=incident_key,
            )
            if suppression_reason is not None:
                logger.debug(
                    "grader.early_warning_suppressed",
                    zone=zone_key,
                    camera=camera_id,
                    score=round(adjusted_score, 3),
                    reason=suppression_reason,
                )
                return None

            self._alert_counter += 1
            seq = self._alert_counter
            self._last_alerts[
                self._tracker_key(zone_key, detection_type, incident_key)
            ] = now
            self._last_camera_alerts[
                self._camera_suppression_key(
                    camera_id,
                    zone_key,
                    detection_type,
                    incident_key,
                )
            ] = now
            self._record_spatial_alert(
                camera_id=camera_id,
                zone_key=zone_key,
                detection_type=detection_type,
                detected_objects=detected_objects,
                now=now,
                incident_key=incident_key,
            )

            # Reset CUSUM tracker — the early-warning fire consumes the evidence
            # the same way a normal fire would, so we don't double-alert.
            tracker_key = self._tracker_key(
                zone_key, detection_type, incident_key,
            )
            if incident_key is None:
                self._trackers[tracker_key] = _AnomalyTracker()
            else:
                self._trackers.pop(tracker_key, None)

            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S%f")
            rnd = random.randint(0, 0xFFFF)
            alert_id = f"ALT-{self._node_id}-{ts}-{seq:04d}-{rnd:04x}"

            event_group_id, event_group_count = self._assign_event_group(
                zone_key, alert_id, now,
            )

            alert = Alert(
                alert_id=alert_id,
                camera_id=camera_id,
                zone_id=zone_id,
                severity=severity,
                anomaly_score=adjusted_score,
                timestamp=time.time(),
                frame_number=frame_number,
                snapshot=frame,
                heatmap=anomaly_map,
                detection_type=detection_type,
                detected_objects=detected_objects or [],
                classification_label=classification_label,
                classification_confidence=classification_confidence,
                handling_policy=HANDLING_POLICIES.get(severity, "quick"),
                event_group_id=event_group_id,
                event_group_count=event_group_count,
                triggered_by_early_warning=True,
            )

        try:
            validate_alert(alert)
        except Exception:
            logger.debug("grader.contract_validation_failed", exc_info=True)

        logger.info(
            "grader.early_warning_fired",
            zone=zone_key,
            camera=camera_id,
            score=round(adjusted_score, 3),
            severity=severity.value,
            detection_type=detection_type,
            classification_label=classification_label,
        )
        return alert

    @staticmethod
    def _incident_bbox(
        detection_type: str,
        detected_objects: list[dict] | None,
    ) -> tuple[float, float, float, float] | None:
        """Return the primary localized incident bbox, if spatial dedup is safe."""
        if detection_type not in _SPATIAL_DEDUP_DETECTION_TYPES or not detected_objects:
            return None
        for detected_object in detected_objects:
            bbox = detected_object.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(value) for value in bbox)
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            return (x1, y1, x2, y2)
        return None

    @staticmethod
    def _bbox_iou(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        second_area = (second[2] - second[0]) * (second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    def _suppression_reason(
        self,
        *,
        camera_id: str,
        zone_key: str,
        detection_type: str,
        detected_objects: list[dict] | None,
        now: float,
        incident_key: str | None,
    ) -> str | None:
        """Return the active suppression reason for this candidate, if any."""
        incident_key = self._normalize_incident_key(incident_key)
        incident_bbox = self._incident_bbox(detection_type, detected_objects)
        tracking_type = self._tracking_detection_type(
            detection_type,
            incident_key,
        )
        localized_candidate = detection_type in _SPATIAL_DEDUP_DETECTION_TYPES
        if localized_candidate and (
            incident_key is not None or incident_bbox is not None
        ):
            zone_window = self._config.suppression.same_zone_window_seconds
            camera_window = self._config.suppression.same_camera_window_seconds
            for recent in reversed(self._recent_spatial_alerts):
                if (
                    recent.camera_id != camera_id
                    or recent.zone_key != zone_key
                    or self._tracking_detection_type(
                        recent.detection_type,
                        recent.incident_key,
                    ) != tracking_type
                ):
                    continue
                age = now - recent.timestamp

                # Explicit upstream identities are authoritative. A repeated
                # identity keeps the long cooldown even when its bbox moves;
                # two explicitly different identities never suppress one
                # another merely because they share a zone or overlap.
                if incident_key is not None and recent.incident_key is not None:
                    if incident_key != recent.incident_key:
                        continue
                    if age < zone_window:
                        return "same_zone_suppression"
                    if age < camera_window:
                        return "same_camera_suppression"
                    continue

                # Legacy callers without an incident identity retain spatial
                # IoU deduplication for the same location. Distinct bboxes are
                # independent incidents; there is deliberately no blanket
                # per-zone/per-type minimum interval.
                if incident_bbox is None or recent.bbox is None:
                    continue
                if self._bbox_iou(incident_bbox, recent.bbox) < (
                    _SPATIAL_DEDUP_IOU_THRESHOLD
                ):
                    continue
                if age < zone_window:
                    return "same_zone_suppression"
                if age < camera_window:
                    return "same_camera_suppression"
            # An explicit identity has a dedicated tracker/suppression key.
            # Fall through so an appearance incident emitted as ANOMALY on one
            # frame and HYBRID on the next still observes the same cooldown.
            # Bbox-only legacy callers must retain the spatial-only behavior,
            # otherwise one localized box could suppress another in the zone.
            if incident_key is None:
                return None

        last_alert_time = self._last_alerts.get(
            self._tracker_key(zone_key, detection_type, incident_key),
        )
        # Missing history is not an alert at monotonic=0. On recently booted
        # hosts that sentinel suppressed the first real incident for an entire
        # window. A recorded zero is valid and must still start a cooldown.
        if (
            last_alert_time is not None
            and now - last_alert_time < self._config.suppression.same_zone_window_seconds
        ):
            return "same_zone_suppression"
        last_camera_alert = self._last_camera_alerts.get(
            self._camera_suppression_key(
                camera_id,
                zone_key,
                detection_type,
                incident_key,
            ),
        )
        if (
            last_camera_alert is not None
            and now - last_camera_alert < self._config.suppression.same_camera_window_seconds
        ):
            return "same_camera_suppression"
        return None

    def _record_spatial_alert(
        self,
        *,
        camera_id: str,
        zone_key: str,
        detection_type: str,
        detected_objects: list[dict] | None,
        now: float,
        incident_key: str | None,
    ) -> None:
        incident_key = self._normalize_incident_key(incident_key)
        if detection_type not in _SPATIAL_DEDUP_DETECTION_TYPES:
            return
        bbox = self._incident_bbox(detection_type, detected_objects)
        if bbox is None and incident_key is None:
            return
        self._recent_spatial_alerts.append(_SpatialAlert(
            camera_id=camera_id,
            zone_key=zone_key,
            detection_type=detection_type,
            bbox=bbox,
            incident_key=incident_key,
            timestamp=now,
        ))

    def _assign_event_group(
        self, zone_key: str, alert_id: str, alert_time: float,
    ) -> tuple[str, int]:
        """Assign an event_group_id for alert aggregation.

        Alerts from the same zone within the aggregation window share the
        same event_group_id. The first alert in a window creates a new group.

        Returns (event_group_id, event_group_count).
        """
        existing = self._event_groups.get(zone_key)
        if existing is not None:
            group_id, first_time, count = existing
            if alert_time - first_time < self._aggregation_window:
                new_count = count + 1
                self._event_groups[zone_key] = (group_id, first_time, new_count)
                return group_id, new_count

        # New group
        group_id = f"EVG-{alert_id}"
        self._event_groups[zone_key] = (group_id, alert_time, 1)
        return group_id, 1

    def _score_to_severity(self, score: float) -> AlertSeverity | None:
        """Map an anomaly score to a severity level."""
        thresholds = self._config.severity_thresholds
        if score >= thresholds.high:
            return AlertSeverity.HIGH
        if score >= thresholds.medium:
            return AlertSeverity.MEDIUM
        if score >= thresholds.low:
            return AlertSeverity.LOW
        if score >= thresholds.info:
            return AlertSeverity.INFO
        return None

    def decay_unobserved(
        self,
        camera_id: str,
        zone_id: str,
        observed_incidents: set[tuple[str, str | None]],
        *,
        decay_detection_types: set[str] | None = None,
    ) -> None:
        """Decay active evidence not observed in the current zone frame.

        Each tuple is ``(detection_type, incident_key)``. The same tracker-key
        contract used by :meth:`evaluate` is used here, so a water-only frame
        cannot accidentally preserve stale plume/anomaly evidence, and one
        active track cannot preserve a different track of the same type.

        ``decay_detection_types`` limits decay to channels that were actually
        observed on this frame.  This is required when, for example, YOLO is
        unavailable but independent water/plume heads completed normally: an
        unknown appearance/object incident must be preserved while a known-
        absent independent incident continues to decay.
        """
        zone_key = f"{camera_id}:{zone_id}"
        observed_tracker_keys = {
            self._tracker_key(
                zone_key,
                getattr(detection_type, "value", str(detection_type)),
                incident_key,
            )
            for detection_type, incident_key in observed_incidents
        }
        eligible_types = (
            {
                getattr(detection_type, "value", str(detection_type))
                for detection_type in decay_detection_types
            }
            if decay_detection_types is not None
            else None
        )
        evidence_lambda = self._config.temporal.evidence_lambda

        with self._tracker_lock:
            for tracker_key, tracker in list(self._trackers.items()):
                if not (
                    tracker_key == zone_key
                    or tracker_key.startswith(f"{zone_key}:")
                ):
                    continue
                if tracker_key in observed_tracker_keys:
                    continue
                if eligible_types is not None:
                    if tracker_key == zone_key:
                        tracker_detection_type = DetectionType.ANOMALY.value
                    else:
                        tracker_detection_type = tracker_key[
                            len(zone_key) + 1:
                        ].split(":", 1)[0]
                    if tracker_detection_type not in eligible_types:
                        continue
                if tracker.evidence == 0.0 and tracker.last_seen == 0.0:
                    if ":incident:" in tracker_key:
                        self._trackers.pop(tracker_key, None)
                    continue

                tracker.evidence *= evidence_lambda
                if tracker.evidence < 0.01:
                    if ":incident:" in tracker_key:
                        self._trackers.pop(tracker_key, None)
                    else:
                        self._trackers[tracker_key] = _AnomalyTracker()

    def reset(self, zone_key: str | None = None) -> None:
        """Reset trackers. If zone_key is None, reset all."""
        with self._tracker_lock:
            if zone_key:
                for tracker_key in list(self._trackers):
                    if (
                        tracker_key == zone_key
                        or tracker_key.startswith(f"{zone_key}:")
                    ):
                        self._trackers.pop(tracker_key, None)
                for suppression_key in list(self._last_alerts):
                    if (
                        suppression_key == zone_key
                        or suppression_key.startswith(f"{zone_key}:")
                    ):
                        self._last_alerts.pop(suppression_key, None)
                self._recent_spatial_alerts = [
                    alert for alert in self._recent_spatial_alerts
                    if alert.zone_key != zone_key
                ]
            else:
                self._trackers.clear()
                self._last_alerts.clear()
                self._last_camera_alerts.clear()
                self._recent_spatial_alerts.clear()

    def reset_camera_suppression(self, camera_id: str) -> None:
        """Reset camera-level suppression timestamp without clearing zone state."""
        with self._tracker_lock:
            for suppression_key in list(self._last_camera_alerts):
                if (
                    suppression_key == camera_id
                    or suppression_key.startswith(f"{camera_id}:")
                ):
                    self._last_camera_alerts.pop(suppression_key, None)
            self._recent_spatial_alerts = [
                alert for alert in self._recent_spatial_alerts
                if alert.camera_id != camera_id
            ]

    @staticmethod
    def _tracker_to_snapshot(tracker: _AnomalyTracker) -> CusumSnapshot | None:
        """Convert an active tracker to a frozen snapshot, or None if inactive."""
        if tracker.evidence == 0.0 and tracker.last_seen == 0.0:
            return None
        return CusumSnapshot(
            evidence=tracker.evidence,
            first_seen=tracker.first_seen,
            last_seen=tracker.last_seen,
            max_score=tracker.max_score,
        )

    def get_cusum_state(self, zone_key: str) -> CusumSnapshot | None:
        """Return a frozen snapshot of the CUSUM evidence for a zone.

        Args:
            zone_key: Zone key in the format "camera_id:zone_id".

        Returns:
            The strongest active channel snapshot for the zone, or None.
        """
        with self._tracker_lock:
            snapshots = [
                snapshot
                for tracker_key, tracker in self._trackers.items()
                if (
                    tracker_key == zone_key
                    or tracker_key.startswith(f"{zone_key}:")
                )
                if (snapshot := self._tracker_to_snapshot(tracker)) is not None
            ]
            if not snapshots:
                return None
            return max(snapshots, key=lambda snapshot: snapshot.evidence)

    def get_all_cusum_states(self) -> dict[str, CusumSnapshot]:
        """Return snapshots for all active zone trackers."""
        with self._tracker_lock:
            result = {}
            for zone_key, tracker in self._trackers.items():
                snap = self._tracker_to_snapshot(tracker)
                if snap is not None:
                    result[zone_key] = snap
            return result

    def prune_stale_trackers(self, max_age_seconds: float = 3600.0) -> int:
        """Remove zone trackers that haven't been updated in max_age_seconds.

        Prevents memory growth from removed/renamed zones.
        Returns the number of pruned entries.
        """
        now = time.monotonic()
        with self._tracker_lock:
            stale_keys = [
                key for key, tracker in self._trackers.items()
                if tracker.last_seen != 0 and (now - tracker.last_seen) > max_age_seconds
            ]
            for key in stale_keys:
                del self._trackers[key]
                self._last_alerts.pop(key, None)
            # Also prune expired event groups (beyond aggregation window)
            stale_groups = [
                key for key, (_, first_time, _) in self._event_groups.items()
                if (now - first_time) > self._aggregation_window * 2
            ]
            for key in stale_groups:
                del self._event_groups[key]
            spatial_max_age = max(
                self._config.suppression.same_zone_window_seconds,
                self._config.suppression.same_camera_window_seconds,
            )
            # Incident-scoped suppression keys are high-cardinality. Expire
            # their timestamps independently of tracker lifetime so completed
            # tracks cannot accumulate indefinitely.
            for suppression_state in (
                self._last_alerts,
                self._last_camera_alerts,
            ):
                for key, timestamp in list(suppression_state.items()):
                    if (now - timestamp) > spatial_max_age:
                        suppression_state.pop(key, None)
            spatial_count_before = len(self._recent_spatial_alerts)
            self._recent_spatial_alerts = [
                alert for alert in self._recent_spatial_alerts
                if (now - alert.timestamp) <= spatial_max_age
            ]
            stale_spatial_count = spatial_count_before - len(self._recent_spatial_alerts)
        pruned = len(stale_keys) + len(stale_groups) + stale_spatial_count
        if pruned:
            logger.debug(
                "grader.pruned_trackers",
                trackers=len(stale_keys),
                event_groups=len(stale_groups),
                spatial_alerts=stale_spatial_count,
            )
        return pruned

    def save_evidence_state(self, path: Path) -> None:
        """Persist CUSUM evidence to disk for crash recovery.

        Only saves zones with meaningful evidence (> 0.01) to avoid
        noise. Restoring stale evidence after restart lets the grader
        resume accumulation instead of starting cold.
        """
        import json

        with self._tracker_lock:
            state = {}
            for zone_key, tracker in self._trackers.items():
                if tracker.evidence > 0.01:
                    state[zone_key] = {
                        "evidence": round(tracker.evidence, 6),
                        "first_seen": tracker.first_seen,
                        "last_seen": tracker.last_seen,
                        "max_score": round(tracker.max_score, 6),
                    }

        if not state:
            return

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state, indent=2))
            logger.debug("grader.evidence_saved", zones=len(state))
        except Exception as e:
            logger.warning("grader.evidence_save_failed", error=str(e))

    def load_evidence_state(self, path: Path) -> int:
        """Restore persisted CUSUM evidence from disk.

        Returns the number of zones restored. Skips entries older than
        max_gap_seconds to avoid triggering on stale evidence.
        """
        import json

        if not path.exists():
            return 0

        try:
            state = json.loads(path.read_text())
        except Exception as e:
            logger.warning("grader.evidence_load_failed", error=str(e))
            return 0

        now = time.time()
        max_gap = self._config.temporal.max_gap_seconds
        restored = 0

        with self._tracker_lock:
            for zone_key, data in state.items():
                # Skip stale evidence that exceeds the temporal gap threshold
                if now - data["last_seen"] > max_gap:
                    continue
                tracker = _AnomalyTracker(
                    evidence=data["evidence"],
                    first_seen=data["first_seen"],
                    last_seen=data["last_seen"],
                    max_score=data["max_score"],
                )
                self._trackers[zone_key] = tracker
                restored += 1

        if restored:
            logger.info("grader.evidence_restored", zones=restored)
        return restored
