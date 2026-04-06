"""Structured per-frame inference record for audit and compliance (5.2).

Normal frames keep only statistical summaries via FrameDiagnostics.
Frames exceeding the INFO threshold get a full InferenceRecord persisted
to disk, including the raw frame for audit (FR-023).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PrefilterDecision(str, Enum):
    """Outcome of the MOG2 pre-filter stage."""

    PASSED = "passed"
    SKIPPED_NO_CHANGE = "skipped_no_change"
    SKIPPED_HEARTBEAT = "skipped_heartbeat"
    SKIPPED_LOCK = "skipped_lock"


class ConformalLevel(str, Enum):
    """Conformal prediction confidence level for the anomaly score."""

    NONE = "none"
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FinalDecision(str, Enum):
    """Final per-frame decision after all fusion stages."""

    NORMAL = "normal"
    INFO = "info"
    ALERT = "alert"
    SUPPRESSED = "suppressed"


@dataclass
class InferenceRecord:
    """Complete inference record for a single frame.

    This is the audit-grade record carrying all 14 spec-required fields.
    Built at the end of each pipeline frame; only persisted when
    final_decision is INFO or ALERT.
    """

    frame_id: str  # uuid hex
    camera_id: str
    timestamp_ns: int  # time.time_ns()
    model_version: str
    health_metrics: dict = field(default_factory=dict)
    prefilter_result: PrefilterDecision = PrefilterDecision.PASSED
    anomaly_score: float = 0.0
    heatmap_ref: str | None = None  # disk path, filled by InferenceRecordStore
    cusum_evidence: dict = field(default_factory=dict)  # {zone_key: evidence}
    sam2_objects: list[dict] = field(default_factory=list)
    cross_cam_corroboration: dict = field(default_factory=dict)
    conformal_level: ConformalLevel = ConformalLevel.NONE
    safety_channel_result: bool | None = None
    final_decision: FinalDecision = FinalDecision.NORMAL
    stage_durations_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "timestamp_ns": self.timestamp_ns,
            "model_version": self.model_version,
            "health_metrics": self.health_metrics,
            "prefilter_result": self.prefilter_result.value,
            "anomaly_score": self.anomaly_score,
            "heatmap_ref": self.heatmap_ref,
            "cusum_evidence": self.cusum_evidence,
            "sam2_objects": self.sam2_objects,
            "cross_cam_corroboration": self.cross_cam_corroboration,
            "conformal_level": self.conformal_level.value,
            "safety_channel_result": self.safety_channel_result,
            "final_decision": self.final_decision.value,
            "stage_durations_ms": self.stage_durations_ms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> InferenceRecord:
        """Deserialize from a JSON dict."""
        return cls(
            frame_id=data["frame_id"],
            camera_id=data["camera_id"],
            timestamp_ns=data["timestamp_ns"],
            model_version=data["model_version"],
            health_metrics=data.get("health_metrics", {}),
            prefilter_result=PrefilterDecision(data.get("prefilter_result", "passed")),
            anomaly_score=data.get("anomaly_score", 0.0),
            heatmap_ref=data.get("heatmap_ref"),
            cusum_evidence=data.get("cusum_evidence", {}),
            sam2_objects=data.get("sam2_objects", []),
            cross_cam_corroboration=data.get("cross_cam_corroboration", {}),
            conformal_level=ConformalLevel(data.get("conformal_level", "none")),
            safety_channel_result=data.get("safety_channel_result"),
            final_decision=FinalDecision(data.get("final_decision", "normal")),
            stage_durations_ms=data.get("stage_durations_ms", {}),
        )
