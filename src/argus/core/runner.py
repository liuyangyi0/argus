"""CameraInferenceRunner — unified stateful inference object per camera (5.1).

Wraps DetectionPipeline and aggregates state from AlertGrader, HealthMonitor,
and DegradationStateMachine into a single RunnerSnapshot.  Acts as the "single
source of truth" for any external query about a camera's inference state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from argus.alerts.grader import Alert
from argus.core.degradation import DegradationState, DegradationStateMachine, LockState
from argus.core.inference_record import FinalDecision, InferenceRecord
from argus.core.pipeline import DetectionPipeline, PipelineStats
from argus.storage.models import InferenceRecord as DBInferenceRecord

if TYPE_CHECKING:
    from argus.core.health import HealthMonitor
    from argus.storage.audit import AuditLogger
    from argus.storage.inference_buffer import InferenceBuffer
    from argus.storage.inference_store import InferenceRecordStore

logger = structlog.get_logger()

_SSIM_FALLBACK_MODE = "ssim_fallback"


@dataclass(frozen=True)
class RunnerSnapshot:
    """Immutable point-in-time view of a camera's inference state."""

    camera_id: str
    model_ref: str
    health_status: str
    cusum_state: dict
    lock_state: LockState
    last_heartbeat: float
    stats: PipelineStats
    version_tag: str
    degradation_state: DegradationState
    consecutive_failures: int


class CameraInferenceRunner:
    """Per-camera inference runner — single source of truth.

    Composes a DetectionPipeline plus degradation tracking, version tagging,
    and health aggregation.  CameraManager creates one Runner per camera
    instead of directly holding a DetectionPipeline.
    """

    def __init__(
        self,
        pipeline: DetectionPipeline,
        health_monitor: HealthMonitor | None = None,
        version_tag: str = "",
        audit_logger: AuditLogger | None = None,
        max_consecutive_failures: int = 5,
        refuse_start_on_backbone_failure: bool = False,
        inference_buffer: InferenceBuffer | None = None,
    ):
        self._pipeline = pipeline
        self._health_monitor = health_monitor
        self._version_tag = version_tag
        self._refuse_start_on_backbone_failure = refuse_start_on_backbone_failure
        self._max_consecutive_failures = max_consecutive_failures

        camera_id = pipeline.camera_config.camera_id
        self._camera_id = camera_id
        self._degradation = DegradationStateMachine(camera_id, audit_logger)
        self._consecutive_failures = 0
        self._record_store: InferenceRecordStore | None = None
        self._inference_buffer: InferenceBuffer | None = inference_buffer

    @property
    def pipeline(self) -> DetectionPipeline:
        """Backward-compatible access to the underlying pipeline."""
        return self._pipeline

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def degradation_state(self) -> DegradationState:
        return self._degradation.state

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def set_version_tag(self, version_tag: str) -> None:
        """Update the runner's externally visible model version tag."""
        self._version_tag = version_tag

    def initialize(self) -> bool:
        """Initialize the pipeline. Returns False if backbone load fails and strict mode is on."""
        result = self._pipeline.initialize()

        if not result:
            self._degradation.transition(
                DegradationState.BACKBONE_FAILED,
                "pipeline.initialize() returned False",
            )
            return False

        # Check if we're in SSIM fallback and strict mode is enabled
        status = self._pipeline.get_detector_status()
        if status.mode == _SSIM_FALLBACK_MODE and self._refuse_start_on_backbone_failure:
            self._degradation.transition(
                DegradationState.BACKBONE_FAILED,
                "backbone model not loaded, strict mode refuses SSIM fallback",
            )
            logger.error(
                "runner.backbone_refused",
                camera_id=self._camera_id,
                msg="refuse_start_on_backbone_failure=True but no trained model found",
            )
            return False

        return True

    def run_once(self) -> Alert | None:
        """Read one frame and process it, tracking consecutive failures."""
        alert = self._pipeline.run_once()

        # Track consecutive detection failures
        if self._pipeline.last_detection_failed:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._max_consecutive_failures:
                self._handle_failure_threshold()
        else:
            self._consecutive_failures = 0

        # Track segmenter failures → transition to DEGRADED_SEGMENTER
        seg_failures = self._pipeline.segmenter_consecutive_failures
        if seg_failures >= self._max_consecutive_failures:
            if self._degradation.state not in (
                DegradationState.DEGRADED_SEGMENTER,
                DegradationState.BACKBONE_FAILED,
                DegradationState.RESTARTING,
            ):
                self._degradation.transition(
                    DegradationState.DEGRADED_SEGMENTER,
                    f"{seg_failures} consecutive segmentation failures",
                )
        elif seg_failures == 0 and self._degradation.state == DegradationState.DEGRADED_SEGMENTER:
            self._degradation.transition(
                DegradationState.NOMINAL,
                "segmenter recovered",
            )

        # Submit audit record for non-NORMAL frames
        last_record = self._pipeline.last_inference_record
        if last_record is not None and self._record_store is not None:
            if last_record.final_decision in (FinalDecision.INFO, FinalDecision.ALERT):
                raw_frame = self._pipeline.get_raw_frame()
                self._record_store.submit(last_record, raw_frame)

        # DB write-behind buffer: persist non-NORMAL frames to the
        # inference_records table for batched queries / evaluation.
        #
        # NORMAL frames are intentionally skipped to mirror the disk
        # InferenceRecordStore behaviour and avoid ~1200-row/min inserts
        # on quiet cameras.  If downstream evaluation later needs NORMAL
        # samples as negatives, drop the final_decision filter below and
        # size InferenceBuffer.max_size accordingly.
        if last_record is not None and self._inference_buffer is not None:
            if last_record.final_decision in (FinalDecision.INFO, FinalDecision.ALERT):
                db_record = self._build_db_record(last_record, alert)
                if db_record is not None:
                    self._inference_buffer.append(db_record)

        return alert

    def _build_db_record(
        self,
        record: InferenceRecord,
        alert: Alert | None,
    ) -> DBInferenceRecord | None:
        """Convert the pipeline's audit dataclass to the ORM row for DB persistence.

        ``InferenceRecord`` in ``argus.core.inference_record`` is the
        audit dataclass built by the pipeline; ``InferenceRecord`` in
        ``argus.storage.models`` is the ORM row expected by
        ``InferenceBuffer``.  They share many fields but not all — this
        helper does the mapping and tolerates missing bits.
        """
        try:
            # Zone: pipeline may encode it in cusum_evidence keys as
            # "camera_id:zone_id"; fall back to "default".
            zone_id = "default"
            if record.cusum_evidence:
                first_key = next(iter(record.cusum_evidence), "")
                if ":" in first_key:
                    zone_id = first_key.split(":", 1)[1] or "default"

            # Frame number: the dataclass doesn't carry it directly, so
            # use the pipeline's running counter (captured just before
            # the record was built).
            frame_number = int(self._pipeline.stats.frames_captured)

            # Inference latency: sum the per-stage durations when available.
            latency_ms = None
            if record.stage_durations_ms:
                try:
                    latency_ms = float(sum(record.stage_durations_ms.values()))
                except (TypeError, ValueError):
                    latency_ms = None

            return DBInferenceRecord(
                camera_id=record.camera_id,
                zone_id=zone_id,
                frame_number=frame_number,
                timestamp=record.timestamp_ns / 1e9,
                model_version_id=record.model_version or None,
                anomaly_score=float(record.anomaly_score),
                inference_latency_ms=latency_ms,
                was_alert=alert is not None,
                alert_id=alert.alert_id if alert is not None else None,
            )
        except Exception:
            logger.debug(
                "runner.db_record_build_failed",
                camera_id=self._camera_id,
                exc_info=True,
            )
            return None

    def _handle_failure_threshold(self) -> None:
        """Handle N consecutive detection failures by attempting model reload."""
        logger.warning(
            "runner.failure_threshold",
            camera_id=self._camera_id,
            consecutive_failures=self._consecutive_failures,
        )
        self._degradation.transition(
            DegradationState.RESTARTING,
            f"{self._consecutive_failures} consecutive detection failures",
        )

        model_path = DetectionPipeline._find_model(self._camera_id)
        if model_path and self._pipeline.reload_anomaly_model(model_path):
            self._degradation.transition(
                DegradationState.NOMINAL,
                "model reload succeeded after consecutive failures",
            )
        else:
            self._degradation.transition(
                DegradationState.DEGRADED_DETECTOR,
                "model reload failed, continuing with current detector",
            )

        self._consecutive_failures = 0

    def get_snapshot(self) -> RunnerSnapshot:
        """Build a point-in-time snapshot of the runner's state."""
        pipeline = self._pipeline

        # Model ref
        status = pipeline.get_detector_status()
        if status.model_path:
            model_ref = status.model_path
        elif status.mode == _SSIM_FALLBACK_MODE:
            model_ref = _SSIM_FALLBACK_MODE
        else:
            model_ref = "unknown"

        # Health status
        health_status = "healthy"
        if self._health_monitor:
            try:
                sys_health = self._health_monitor.get_health()
                health_status = sys_health.status.value
            except Exception:
                health_status = "unknown"

        # CUSUM state via public API
        cusum_state = {}
        all_states = pipeline.get_cusum_states()
        for k, s in all_states.items():
            cusum_state[k] = {
                "evidence": s.evidence,
                "first_seen": s.first_seen,
                "last_seen": s.last_seen,
                "max_score": s.max_score,
            }

        return RunnerSnapshot(
            camera_id=self._camera_id,
            model_ref=model_ref,
            health_status=health_status,
            cusum_state=cusum_state,
            lock_state=pipeline.lock_state,
            last_heartbeat=pipeline.last_heartbeat_time,
            stats=pipeline.stats,
            version_tag=self._version_tag,
            degradation_state=self._degradation.state,
            consecutive_failures=self._consecutive_failures,
        )

    def set_record_store(self, store: InferenceRecordStore | None) -> None:
        """Attach an InferenceRecordStore for audit persistence."""
        self._record_store = store
