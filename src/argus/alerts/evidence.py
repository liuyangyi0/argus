"""Structured evidence payloads for alert explanations.

The alert pipeline has several detection heads, but the dashboard and audit
store need one stable explanation shape.  DetectionEvidence is intentionally
small and JSON-first so it can be persisted on alert rows and copied into
candidate diagnostics without carrying image data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_DETECTION_TYPES: frozenset[str] = frozenset({
    "scene_change",
    "new_object",
    "projectile",
    "water_leak",
    "plume",
    "anomaly",
    # Existing internal detector names kept for backward compatibility.
    "object",
    "hybrid",
    "persistent_change",
})


@dataclass
class DetectionEvidence:
    """JSON-safe explanation for an alert or non-alert candidate."""

    detection_type: str
    score: float | None = None
    camera_id: str | None = None
    zone_id: str | None = None
    zone_type: str | None = None
    bbox: tuple[int, int, int, int] | None = None
    frame_number: int | None = None
    severity: str | None = None
    summary: str | None = None
    evidence_json: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    suppression_reason: str | None = None
    model_version_id: str | None = None
    debug_trace_id: str | None = None
    first_seen_ts: float | None = None
    last_seen_ts: float | None = None
    stable_for_sec: float | None = None

    @property
    def supported(self) -> bool:
        """Whether ``detection_type`` is one of the known canonical values."""
        return self.detection_type in SUPPORTED_DETECTION_TYPES

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "detection_type": self.detection_type,
            "score": self.score,
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "zone_type": self.zone_type,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "frame_number": self.frame_number,
            "severity": self.severity,
            "summary": self.evidence_summary(),
            "evidence_json": self.evidence_json,
            "features": self.features,
            "candidates": self.candidates,
            "suppression_reason": self.suppression_reason,
            "model_version_id": self.model_version_id,
            "debug_trace_id": self.debug_trace_id,
            "first_seen_ts": self.first_seen_ts,
            "last_seen_ts": self.last_seen_ts,
            "stable_for_sec": self.stable_for_sec,
            "supported": self.supported,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectionEvidence":
        """Build DetectionEvidence from a JSON-like dictionary."""
        bbox_raw = data.get("bbox")
        bbox = None
        if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
            bbox = tuple(int(value) for value in bbox_raw)
        return cls(
            detection_type=str(data.get("detection_type") or "anomaly"),
            score=data.get("score"),
            camera_id=data.get("camera_id"),
            zone_id=data.get("zone_id"),
            zone_type=data.get("zone_type"),
            bbox=bbox,
            frame_number=data.get("frame_number"),
            severity=data.get("severity"),
            summary=data.get("summary") or data.get("evidence_summary"),
            evidence_json=dict(data.get("evidence_json") or {}),
            features=dict(data.get("features") or {}),
            candidates=list(data.get("candidates") or []),
            suppression_reason=data.get("suppression_reason"),
            model_version_id=data.get("model_version_id"),
            debug_trace_id=data.get("debug_trace_id"),
            first_seen_ts=data.get("first_seen_ts"),
            last_seen_ts=data.get("last_seen_ts"),
            stable_for_sec=data.get("stable_for_sec"),
        )

    def evidence_summary(self) -> str:
        """Return a compact one-line explanation for operators and exports."""
        if self.summary:
            return self.summary

        parts = [self.detection_type or "anomaly"]
        if self.severity:
            parts.append(f"severity={self.severity}")
        if self.score is not None:
            parts.append(f"score={self.score:.4f}")
        if self.zone_id:
            parts.append(f"zone={self.zone_id}")
        if self.frame_number is not None:
            parts.append(f"frame={self.frame_number}")
        if self.suppression_reason:
            parts.append(f"suppressed={self.suppression_reason}")
        return ", ".join(parts)
