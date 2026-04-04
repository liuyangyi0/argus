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

import numpy as np
import structlog

from argus.config.schema import AlertConfig, AlertSeverity, ZonePriority

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


@dataclass
class _AnomalyTracker:
    """Tracks consecutive anomaly detections for a specific zone."""

    consecutive_count: int = 0
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

    def __init__(self, config: AlertConfig, node_id: str = "edge"):
        self._config = config
        self._node_id = node_id
        self._trackers: dict[str, _AnomalyTracker] = defaultdict(_AnomalyTracker)
        self._last_alerts: dict[str, float] = {}  # zone_key -> last alert timestamp
        self._alert_counter = 0
        self._counter_lock = threading.Lock()

    def evaluate(
        self,
        camera_id: str,
        zone_id: str,
        zone_priority: ZonePriority,
        anomaly_score: float,
        frame_number: int,
        frame: np.ndarray | None = None,
        anomaly_map: np.ndarray | None = None,
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

        # Step 2: Determine severity
        severity = self._score_to_severity(adjusted_score)
        if severity is None:
            # Score below minimum threshold, reset tracker
            self._trackers[zone_key] = _AnomalyTracker()
            logger.debug(
                "grader.below_threshold",
                zone=zone_key,
                score=round(adjusted_score, 3),
            )
            return None

        # Step 3: Temporal confirmation
        tracker = self._trackers[zone_key]
        gap = now - tracker.last_seen if tracker.last_seen > 0 else 0

        if gap > self._config.temporal.max_gap_seconds:
            # Too much time between detections, reset
            tracker.consecutive_count = 1
            tracker.first_seen = now
            tracker.max_score = adjusted_score
        else:
            tracker.consecutive_count += 1
            tracker.max_score = max(tracker.max_score, adjusted_score)

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
                    tracker.consecutive_count = 1
                    tracker.max_score = adjusted_score
            tracker.prev_anomaly_mask = curr_mask

        if tracker.consecutive_count < self._config.temporal.min_consecutive_frames:
            logger.debug(
                "grader.awaiting_confirmation",
                zone=zone_key,
                count=tracker.consecutive_count,
                needed=self._config.temporal.min_consecutive_frames,
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
