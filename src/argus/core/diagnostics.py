"""Per-frame diagnostics buffer for pipeline debug view (DET-008).

Provides a thread-safe ring buffer that records pipeline processing results
for each frame, enabling the dashboard debug view (DET-003) and sensitivity
threshold preview (DET-005).
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field


@dataclass
class StageResult:
    """Result of a single pipeline stage for diagnostics."""

    stage_name: str  # "zone_mask", "mog2", "person", "anomaly"
    duration_ms: float
    skipped: bool = False
    skip_reason: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-safe representation."""
        return {
            "stage_name": self.stage_name,
            "duration_ms": self.duration_ms,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "metadata": self.metadata,
        }


@dataclass
class CandidateRecord:
    """Candidate-level evidence and decision record for one frame."""

    candidate_id: str
    candidate_type: str
    decision: str
    stage: str = ""
    evidence: dict = field(default_factory=dict)
    suppression_reason: str | None = None

    def to_dict(self) -> dict:
        """Return a JSON-safe representation."""
        return {
            "candidate_id": self.candidate_id,
            "candidate_type": self.candidate_type,
            "decision": self.decision,
            "stage": self.stage,
            "evidence": self.evidence,
            "suppression_reason": self.suppression_reason,
        }


@dataclass
class FrameDiagnostics:
    """Complete diagnostic record for one frame through the pipeline."""

    frame_number: int
    timestamp: float
    camera_id: str
    total_duration_ms: float = 0.0
    stages: list[StageResult] = field(default_factory=list)
    anomaly_score: float = 0.0
    is_anomalous: bool = False
    alert_emitted: bool = False
    pipeline_mode: str = "active"
    low_light: bool = False
    frame_id: str = ""  # uuid hex, populated by pipeline for audit linkage
    alignment_shift_px: tuple[float, float] = (0.0, 0.0)
    scene_guard_state: str = "disabled"
    final_decision: str = "normal"
    suppression_reason: str | None = None
    candidate_records: list[CandidateRecord] = field(default_factory=list)

    def add_candidate(
        self,
        *,
        candidate_type: str,
        decision: str,
        stage: str = "",
        evidence: dict | None = None,
        suppression_reason: str | None = None,
        candidate_id: str | None = None,
    ) -> CandidateRecord:
        """Append and return a candidate diagnostic record."""
        record = CandidateRecord(
            candidate_id=(
                candidate_id
                or f"{self.camera_id}:{self.frame_number}:{len(self.candidate_records) + 1}"
            ),
            candidate_type=candidate_type,
            decision=decision,
            stage=stage,
            evidence=evidence or {},
            suppression_reason=suppression_reason,
        )
        self.candidate_records.append(record)
        if suppression_reason:
            self.suppression_reason = suppression_reason
        return record

    def to_dict(self) -> dict:
        """Return a JSON-safe representation."""
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "camera_id": self.camera_id,
            "total_duration_ms": self.total_duration_ms,
            "stages": [stage.to_dict() for stage in self.stages],
            "anomaly_score": self.anomaly_score,
            "is_anomalous": self.is_anomalous,
            "alert_emitted": self.alert_emitted,
            "pipeline_mode": self.pipeline_mode,
            "low_light": self.low_light,
            "frame_id": self.frame_id,
            "alignment_shift_px": list(self.alignment_shift_px),
            "scene_guard_state": self.scene_guard_state,
            "final_decision": self.final_decision,
            "suppression_reason": self.suppression_reason,
            "candidate_records": [
                record.to_dict() for record in self.candidate_records
            ],
        }


@dataclass
class FrameScoreRecord:
    """Lightweight record for sensitivity preview score history (DET-005)."""

    frame_number: int
    timestamp: float
    anomaly_score: float
    was_alert: bool

    def to_dict(self) -> dict:
        """Return a JSON-safe representation."""
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "anomaly_score": self.anomaly_score,
            "was_alert": self.was_alert,
        }


class DiagnosticsBuffer:
    """Thread-safe ring buffer for per-frame diagnostics.

    Maintains two separate buffers:
    - Full diagnostics (1500 frames) for the debug view (DET-003/008)
    - Score-only records (300 frames) for sensitivity preview (DET-005)
    """

    def __init__(self, maxlen: int = 1500, score_maxlen: int = 300):
        self._buffer: deque[FrameDiagnostics] = deque(maxlen=maxlen)
        self._score_buffer: deque[FrameScoreRecord] = deque(maxlen=score_maxlen)
        self._lock = threading.Lock()

    def append(self, diag: FrameDiagnostics) -> None:
        """Add a frame diagnostic record."""
        with self._lock:
            self._buffer.append(diag)

    def append_score(self, record: FrameScoreRecord) -> None:
        """Add a score record for sensitivity preview."""
        with self._lock:
            self._score_buffer.append(record)

    def get_recent(self, n: int = 50) -> list[FrameDiagnostics]:
        """Get the most recent N diagnostic records."""
        with self._lock:
            items = list(self._buffer)
        return items[-n:] if len(items) > n else items

    def get_scores(self) -> list[FrameScoreRecord]:
        """Get all cached score records."""
        with self._lock:
            return list(self._score_buffer)

    def evaluate_threshold(self, new_threshold: float) -> dict:
        """Re-evaluate cached scores against a new threshold (DET-005).

        Returns a summary of how many frames would trigger alerts at the
        new threshold vs the current alert count.
        """
        with self._lock:
            scores = list(self._score_buffer)

        total = len(scores)
        if total == 0:
            return {
                "total_frames": 0,
                "would_alert_count": 0,
                "current_alert_count": 0,
                "score_distribution": [0] * 10,
                "new_threshold": new_threshold,
            }

        would_alert = sum(1 for s in scores if s.anomaly_score >= new_threshold)
        current_alerts = sum(1 for s in scores if s.was_alert)

        # Histogram: 10 buckets from 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
        buckets = [0] * 10
        for s in scores:
            idx = min(int(s.anomaly_score * 10), 9)
            buckets[idx] += 1

        return {
            "total_frames": total,
            "would_alert_count": would_alert,
            "current_alert_count": current_alerts,
            "score_distribution": buckets,
            "new_threshold": new_threshold,
        }

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
