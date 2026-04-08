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

import numpy as np
import structlog

from pathlib import Path

from argus.config.schema import AlertConfig, AlertSeverity, SeverityThresholds, ZonePriority

logger = structlog.get_logger()

# UX v2 §5.1: Canonical severity → handling policy mapping
HANDLING_POLICIES: dict[AlertSeverity, str] = {
    AlertSeverity.INFO: "quick",
    AlertSeverity.LOW: "quick",
    AlertSeverity.MEDIUM: "confirm",
    AlertSeverity.HIGH: "detail_required",
}


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
    # D1: Open vocabulary classification
    classification_label: str | None = None
    classification_confidence: float | None = None
    severity_adjusted_by_classifier: bool = False
    # D2: Instance segmentation
    segmentation_count: int = 0
    segmentation_total_area_px: int = 0
    segmentation_objects: list[dict] = field(default_factory=list)
    # UX v2 §5.1: Severity-based handling policy
    # "quick" (LOW), "confirm" (MEDIUM), "detail_required" (HIGH)
    handling_policy: str = "quick"


class DetectionType(str, Enum):
    """Type of detection that triggered an alert (YOLO-005)."""

    ANOMALY = "anomaly"  # Anomalib only
    OBJECT = "object"  # YOLO object detection only
    HYBRID = "hybrid"  # Both YOLO and Anomalib agree


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
        self._last_camera_alerts: dict[str, float] = {}  # camera_id -> last alert timestamp
        # Random offset avoids ID collisions across process restarts
        self._alert_counter = random.randint(1000, 9999)
        self._tracker_lock = threading.Lock()

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

        # Step 3: CUSUM temporal evidence accumulation (thread-safe)
        with self._tracker_lock:
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
                tracker.last_seen = now
                tracker.max_score = adjusted_score

            tracker.last_seen = now

            # Step 3b: Spatial continuity check — anomaly region must overlap with
            # the previous frame's anomaly region to filter transient noise
            min_overlap = self._config.temporal.min_spatial_overlap
            if min_overlap > 0 and anomaly_map is not None:
                curr_mask = (anomaly_map > 0.5).astype(np.uint8)
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
                    if iou < min_overlap:
                        logger.debug(
                            "grader.spatial_mismatch",
                            zone=zone_key,
                            iou=round(iou, 3),
                            threshold=min_overlap,
                        )
                        # Decay instead of reset to tolerate camera vibration
                        tracker.evidence *= 0.5
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

            # Step 4: Check suppression windows (zone-level then camera-level)
            last_alert_time = self._last_alerts.get(zone_key, 0)
            if now - last_alert_time < self._config.suppression.same_zone_window_seconds:
                logger.debug("grader.suppressed", zone=zone_key, score=round(adjusted_score, 3))
                return None

            # Camera-level throttle: after the first alert from each zone
            # passes through, enforce a minimum interval between subsequent
            # alerts on the same camera (across all zones) to prevent storms.
            last_camera_alert = self._last_camera_alerts.get(camera_id, 0)
            cam_window = self._config.suppression.same_camera_window_seconds
            if now - last_camera_alert < cam_window:
                # But always allow the very first alert for a zone that has
                # never fired — ensures every zone gets initial coverage.
                if zone_key in self._last_alerts:
                    logger.debug(
                        "grader.camera_suppressed",
                        zone=zone_key,
                        camera=camera_id,
                        score=round(adjusted_score, 3),
                    )
                    return None

            # Step 5: Emit alert
            self._alert_counter += 1
            seq = self._alert_counter
            self._last_alerts[zone_key] = now
            self._last_camera_alerts[camera_id] = now

            # Use the max score seen during the confirmation window for severity
            final_severity = self._score_to_severity(tracker.max_score) or severity

            # Timestamp-based ID with random suffix for collision resistance
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S%f")
            rnd = random.randint(0, 0xFFFF)
            alert = Alert(
                alert_id=f"ALT-{self._node_id}-{ts}-{seq:04d}-{rnd:04x}",
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
        with self._tracker_lock:
            if zone_key:
                self._trackers.pop(zone_key, None)
                self._last_alerts.pop(zone_key, None)
            else:
                self._trackers.clear()
                self._last_alerts.clear()
                self._last_camera_alerts.clear()

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
            CusumSnapshot if the zone has an active tracker, None otherwise.
        """
        with self._tracker_lock:
            tracker = self._trackers.get(zone_key)
            if tracker is None:
                return None
            return self._tracker_to_snapshot(tracker)

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
                if tracker.last_seen > 0 and (now - tracker.last_seen) > max_age_seconds
            ]
            for key in stale_keys:
                del self._trackers[key]
                self._last_alerts.pop(key, None)
        if stale_keys:
            logger.debug("grader.pruned_trackers", count=len(stale_keys), keys=stale_keys)
        return len(stale_keys)

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
