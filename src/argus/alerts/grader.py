"""Alert severity grading with temporal confirmation and deduplication.

This module decides whether an anomaly detection result should become an alert,
what severity level it should have, and whether it's a duplicate of a recent alert.
The temporal confirmation filter requires anomalies to persist across multiple
consecutive frames before triggering, which dramatically reduces false positives.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import numpy as np
import structlog

from pathlib import Path

from argus.config.schema import AlertConfig, AlertSeverity, SeverityThresholds, ZonePriority

logger = structlog.get_logger()


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
    # C3-4: Cross-camera correlation
    corroborated: bool = True
    correlation_partner: str | None = None
    # C4-3: Model version tracking
    model_version_id: str | None = None


class DetectionType(str, Enum):
    """Type of detection that triggered an alert (YOLO-005)."""

    ANOMALY = "anomaly"  # Anomalib only
    OBJECT = "object"  # YOLO object detection only
    HYBRID = "hybrid"  # Both YOLO and Anomalib agree


@dataclass
class _AnomalyTracker:
    """Tracks anomaly evidence for a specific zone using exponential accumulation."""

    evidence: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    max_score: float = 0.0
    prev_anomaly_mask: np.ndarray | None = None


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
    ):
        self._config = config
        self._node_id = node_id
        self._trackers: dict[str, _AnomalyTracker] = defaultdict(_AnomalyTracker)
        self._last_alerts: dict[str, float] = {}  # zone_key -> last alert timestamp
        self._alert_counter = 0
        self._counter_lock = threading.Lock()

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
    ) -> Alert | None:
        """Evaluate an anomaly detection and potentially produce an alert.

        Returns an Alert if the anomaly passes all filters, None otherwise.
        """
        import math

        now = time.monotonic()
        zone_key = f"{camera_id}:{zone_id}"

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
        adjusted_score = max(0.0, min(anomaly_score * multiplier, 1.0))

        # Step 2: Determine severity (None if below info threshold)
        severity = self._score_to_severity(adjusted_score)

        # Step 3: CUSUM temporal evidence accumulation
        tracker = self._trackers[zone_key]
        gap = now - tracker.last_seen if tracker.last_seen > 0 else 0
        lam = self._config.temporal.evidence_lambda

        # Sub-threshold frame: pure decay, clean up if negligible
        if severity is None:
            tracker.evidence *= lam
            if tracker.evidence < 0.01:
                self._trackers[zone_key] = _AnomalyTracker()
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
            tracker.max_score = adjusted_score

        tracker.last_seen = now

        # Step 3b: Spatial continuity check — anomaly region must overlap with
        # the previous frame's anomaly region to filter transient noise
        min_overlap = self._config.temporal.min_spatial_overlap
        if min_overlap > 0 and anomaly_map is not None:
            curr_mask = (anomaly_map > 0.5).astype(np.uint8)
            if tracker.prev_anomaly_mask is not None and curr_mask.shape == tracker.prev_anomaly_mask.shape:
                intersection = np.count_nonzero(curr_mask & tracker.prev_anomaly_mask)
                union = np.count_nonzero(curr_mask | tracker.prev_anomaly_mask)
                iou = intersection / union if union > 0 else 0.0
                if iou < min_overlap:
                    logger.debug(
                        "grader.spatial_mismatch",
                        zone=zone_key,
                        iou=round(iou, 3),
                        threshold=min_overlap,
                    )
                    tracker.evidence = 0.0
                    tracker.max_score = adjusted_score
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

        # Step 4: Check suppression window
        last_alert_time = self._last_alerts.get(zone_key, 0)
        if now - last_alert_time < self._config.suppression.same_zone_window_seconds:
            logger.debug("grader.suppressed", zone=zone_key, score=round(adjusted_score, 3))
            return None

        # Step 5: Emit alert
        with self._counter_lock:
            self._alert_counter += 1
            seq = self._alert_counter
        self._last_alerts[zone_key] = now

        # Use the max score seen during the confirmation window for severity
        final_severity = self._score_to_severity(tracker.max_score) or severity

        # Timestamp-based ID: survives restarts, unique across nodes
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S%f")[:19]
        alert = Alert(
            alert_id=f"ALT-{self._node_id}-{ts}-{seq:04d}",
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
        )

        # Reset tracker after emitting alert
        self._trackers[zone_key] = _AnomalyTracker()

        return alert

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

    def reset(self, zone_key: str | None = None) -> None:
        """Reset trackers. If zone_key is None, reset all."""
        if zone_key:
            self._trackers.pop(zone_key, None)
            self._last_alerts.pop(zone_key, None)
        else:
            self._trackers.clear()
            self._last_alerts.clear()

    def prune_stale_trackers(self, max_age_seconds: float = 3600.0) -> int:
        """Remove zone trackers that haven't been updated in max_age_seconds.

        Prevents memory growth from removed/renamed zones.
        Returns the number of pruned entries.
        """
        now = time.monotonic()
        stale_keys = [
            key for key, tracker in self._trackers.items()
            if tracker.last_seen > 0 and (now - tracker.last_seen) > max_age_seconds
        ]
        for key in stale_keys:
            del self._trackers[key]
            self._last_alerts.pop(key, None)
        if stale_keys:
            logger.debug("grader.pruned_trackers", count=len(stale_keys), keys=stale_keys)
        return len(stale_keys)
