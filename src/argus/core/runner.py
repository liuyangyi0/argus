"""CameraInferenceRunner — unified stateful inference object per camera (5.1).

Wraps DetectionPipeline and aggregates state from AlertGrader, HealthMonitor,
and DegradationStateMachine into a single RunnerSnapshot.  Acts as the "single
source of truth" for any external query about a camera's inference state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from argus.alerts.grader import Alert
from argus.core.degradation import DegradationState, DegradationStateMachine, LockState
from argus.core.inference_record import (
    PERSISTED_FINAL_DECISIONS,
    InferenceRecord,
)
from argus.core.pipeline import DetectionPipeline, PipelineStats
from argus.core.runtime import (
    FrameOutcome,
    FrameOutcomeKind,
    ModelIdentity,
    ModelReloadResult,
    RuntimeModelSlot,
)
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
    latest_frame_identity: dict | None = None
    model_identity: ModelIdentity | None = None


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
        authorized_model_resolver: Callable[[str], ModelIdentity | None] | None = None,
        initial_model_path: str | Path | None = None,
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
        self._authorized_model_resolver = authorized_model_resolver
        self._model_slot = RuntimeModelSlot(
            version=version_tag or None,
            path=str(initial_model_path) if initial_model_path is not None else None,
        )

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

    @property
    def model_identity(self) -> ModelIdentity:
        """Return the applied runtime model identity."""
        return self._model_slot.identity

    def set_version_tag(self, version_tag: str) -> None:
        """Update the runner's externally visible model version tag."""
        self._version_tag = version_tag
        identity = self._model_slot.identity
        self._model_slot.adopt(
            version=version_tag,
            path=identity.path,
            artifact_sha256=identity.artifact_sha256,
            scene_profile_id=identity.scene_profile_id,
            release_revision=identity.release_revision,
        )

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

        self._sync_model_identity_from_pipeline()
        return True

    def run_once(self) -> Alert | None:
        """Backward-compatible alert-only runner result."""
        return self.run_once_outcome().alert

    def run_once_outcome(self) -> FrameOutcome:
        """Read one frame and return an explicit, immutable outcome."""
        frames_before = int(self._pipeline.stats.frames_captured)
        analyzed_before = int(getattr(self._pipeline.stats, "frames_analyzed", 0))
        record_before = self._pipeline.last_inference_record
        try:
            alert = self._pipeline.run_once()
        except Exception as exc:
            frames_after = int(self._pipeline.stats.frames_captured)
            self._consecutive_failures += 1
            logger.error(
                "runner.frame_processing_failed",
                camera_id=self._camera_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            if self._consecutive_failures >= self._max_consecutive_failures:
                self._handle_failure_threshold()
            return FrameOutcome(
                kind=FrameOutcomeKind.FAILED,
                alert=None,
                frames_captured_before=frames_before,
                frames_captured_after=frames_after,
                reason=f"{type(exc).__name__}: {exc}",
            )
        frames_after = int(self._pipeline.stats.frames_captured)
        analyzed_after = int(getattr(self._pipeline.stats, "frames_analyzed", 0))
        frame_captured = frames_after > frames_before
        analysis_performed = analyzed_after > analyzed_before
        scene_guard_status = self._scene_guard_status() if frame_captured else None
        explicit_kind = self._pipeline_frame_outcome_kind() if frame_captured else None
        explicit_reason = (
            getattr(self._pipeline, "last_frame_outcome_reason", None)
            if frame_captured
            else None
        )

        # Track consecutive detection failures
        if explicit_kind is FrameOutcomeKind.FAILED:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._max_consecutive_failures:
                self._handle_failure_threshold()
        elif analysis_performed and self._pipeline.last_detection_failed:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._max_consecutive_failures:
                self._handle_failure_threshold()
        elif analysis_performed or explicit_kind in {
            FrameOutcomeKind.NORMAL,
            FrameOutcomeKind.ALERT,
        }:
            self._consecutive_failures = 0

        # Track segmenter failures → transition to DEGRADED_SEGMENTER
        seg_failures = self._pipeline.segmenter_consecutive_failures
        if analysis_performed and seg_failures >= self._max_consecutive_failures:
            if self._degradation.state not in (
                DegradationState.DEGRADED_SEGMENTER,
                DegradationState.BACKBONE_FAILED,
                DegradationState.RESTARTING,
            ):
                self._degradation.transition(
                    DegradationState.DEGRADED_SEGMENTER,
                    f"{seg_failures} consecutive segmentation failures",
                )
        elif (
            analysis_performed
            and seg_failures == 0
            and self._degradation.state == DegradationState.DEGRADED_SEGMENTER
        ):
            self._degradation.transition(
                DegradationState.NOMINAL,
                "segmenter recovered",
            )

        # Submit audit record for non-NORMAL frames
        last_record = self._pipeline.last_inference_record
        new_record = last_record is not None and last_record is not record_before
        if new_record and self._record_store is not None:
            if last_record.final_decision in PERSISTED_FINAL_DECISIONS:
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
        if new_record and self._inference_buffer is not None:
            if last_record.final_decision in PERSISTED_FINAL_DECISIONS:
                db_record = self._build_db_record(last_record, alert)
                if db_record is not None:
                    self._inference_buffer.append(db_record)

        return self._classify_outcome(
            frames_before=frames_before,
            alert=alert,
            scene_guard_status=scene_guard_status,
            analysis_performed=analysis_performed,
            explicit_kind=explicit_kind,
            explicit_reason=(str(explicit_reason) if explicit_reason else None),
        )

    def _classify_outcome(
        self,
        *,
        frames_before: int,
        alert: Alert | None,
        scene_guard_status: dict | None,
        analysis_performed: bool,
        explicit_kind: FrameOutcomeKind | None,
        explicit_reason: str | None,
    ) -> FrameOutcome:
        frames_after = int(self._pipeline.stats.frames_captured)
        if frames_after <= frames_before:
            return FrameOutcome(
                kind=FrameOutcomeKind.NO_FRAME,
                alert=None,
                frames_captured_before=frames_before,
                frames_captured_after=frames_after,
                reason="camera produced no new frame",
            )
        if alert is not None:
            return FrameOutcome(
                kind=FrameOutcomeKind.ALERT,
                alert=alert,
                frames_captured_before=frames_before,
                frames_captured_after=frames_after,
            )

        if explicit_kind is FrameOutcomeKind.BLOCKED:
            scene_state = None
            if (
                scene_guard_status is not None
                and scene_guard_status.get("usable") is False
            ):
                scene_state = str(scene_guard_status.get("state") or "blocked")
            return FrameOutcome(
                kind=FrameOutcomeKind.BLOCKED,
                alert=None,
                frames_captured_before=frames_before,
                frames_captured_after=frames_after,
                reason=explicit_reason or "pipeline blocked inference",
                scene_guard_state=scene_state,
            )
        if explicit_kind is FrameOutcomeKind.FAILED:
            return FrameOutcome(
                kind=FrameOutcomeKind.FAILED,
                alert=None,
                frames_captured_before=frames_before,
                frames_captured_after=frames_after,
                reason=explicit_reason or "pipeline frame processing failed",
            )

        if (
            scene_guard_status is not None
            and scene_guard_status.get("enabled") is True
            and scene_guard_status.get("usable") is False
        ):
            return FrameOutcome(
                kind=FrameOutcomeKind.BLOCKED,
                alert=None,
                frames_captured_before=frames_before,
                frames_captured_after=frames_after,
                reason=str(scene_guard_status.get("reason") or "scene guard blocked inference"),
                scene_guard_state=str(scene_guard_status.get("state") or "blocked"),
            )
        if analysis_performed and self._pipeline.last_detection_failed:
            return FrameOutcome(
                kind=FrameOutcomeKind.FAILED,
                alert=None,
                frames_captured_before=frames_before,
                frames_captured_after=frames_after,
                reason="anomaly detection failed",
            )
        return FrameOutcome(
            kind=FrameOutcomeKind.NORMAL,
            alert=None,
            frames_captured_before=frames_before,
            frames_captured_after=frames_after,
        )

    def _pipeline_frame_outcome_kind(self) -> FrameOutcomeKind | None:
        value = getattr(self._pipeline, "last_frame_outcome_kind", None)
        if isinstance(value, FrameOutcomeKind):
            return value
        if isinstance(value, str):
            try:
                return FrameOutcomeKind(value)
            except ValueError:
                logger.warning(
                    "runner.invalid_pipeline_frame_outcome",
                    camera_id=self._camera_id,
                    value=value,
                )
        return None

    def _scene_guard_status(self) -> dict | None:
        get_status = getattr(self._pipeline, "get_scene_guard_status", None)
        if not callable(get_status):
            return None
        try:
            status = get_status()
        except Exception:
            logger.debug(
                "runner.scene_guard_status_failed",
                camera_id=self._camera_id,
                exc_info=True,
            )
            return None
        return status if isinstance(status, dict) else None

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

        authorized = self._resolve_authorized_model()
        if authorized.path:
            result = self.reload_model(
                authorized.path,
                version_tag=authorized.version,
                artifact_sha256=authorized.artifact_sha256,
                scene_profile_id=authorized.scene_profile_id,
                release_revision=authorized.release_revision,
            )
        else:
            logger.error(
                "runner.recovery_model_not_authorized",
                camera_id=self._camera_id,
                current_version=self._model_slot.identity.version,
            )
            result = ModelReloadResult(
                success=False,
                previous=self._model_slot.identity,
                current=self._model_slot.identity,
                attempted=authorized,
                error="no registry-authorized or currently applied model path",
            )
        if result.success:
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

    def reload_model(
        self,
        model_path: str | Path,
        *,
        version_tag: str | None = None,
        artifact_sha256: str | None = None,
        scene_profile_id: str | None = None,
        release_revision: int | None = None,
    ) -> ModelReloadResult:
        """Atomically reload a model and advance identity only after success."""
        current = self._model_slot.identity
        target_version = version_tag if version_tag is not None else current.version
        result = self._model_slot.reload(
            version=target_version,
            path=str(model_path),
            artifact_sha256=artifact_sha256,
            scene_profile_id=scene_profile_id,
            release_revision=release_revision,
            reloader=lambda path, version: self._pipeline.reload_anomaly_model(
                path,
                version_tag=version,
            ),
        )
        if result.success:
            self._version_tag = result.current.version or ""
        return result

    def adopt_prepared_model_identity(
        self,
        *,
        version: str | None,
        path: str | None,
        artifact_sha256: str | None = None,
        scene_profile_id: str | None = None,
        release_revision: int | None = None,
    ) -> ModelIdentity:
        """Advance slot identity after a pipeline standby swap is confirmed.

        This method never loads or discovers an artifact; it only mirrors an
        already committed two-phase pipeline swap.  The caller retains the
        previous identity and may call this again during rollback.
        """

        identity = self._model_slot.adopt(
            version=version,
            path=path,
            artifact_sha256=artifact_sha256,
            scene_profile_id=scene_profile_id,
            release_revision=release_revision,
        )
        self._version_tag = identity.version or ""
        return identity

    def _resolve_authorized_model(self) -> ModelIdentity:
        """Resolve recovery identity without scanning the filesystem."""
        if self._authorized_model_resolver is not None:
            try:
                resolved = self._authorized_model_resolver(self._camera_id)
                if resolved is not None and resolved.path:
                    return ModelIdentity(
                        version=resolved.version or self._model_slot.identity.version,
                        path=resolved.path,
                        artifact_sha256=resolved.artifact_sha256,
                        scene_profile_id=resolved.scene_profile_id,
                        release_revision=resolved.release_revision,
                    )
            except Exception:
                logger.warning(
                    "runner.authorized_model_resolve_failed",
                    camera_id=self._camera_id,
                    exc_info=True,
                )
            # A configured registry resolver is authoritative. Missing or
            # unreadable authorization must not silently fall back to an old
            # path that the registry may have retired.
            return ModelIdentity(
                version=self._model_slot.identity.version,
                path=None,
                artifact_sha256=self._model_slot.identity.artifact_sha256,
                scene_profile_id=self._model_slot.identity.scene_profile_id,
                release_revision=self._model_slot.identity.release_revision,
            )
        # Legacy/offline runtimes without a registry may retry the exact model
        # identity that CameraManager selected at startup. They still never
        # search model directories from the failure path.
        return self._model_slot.identity

    def _sync_model_identity_from_pipeline(self) -> None:
        """Adopt the detector identity after successful pipeline initialization."""
        current = self._model_slot.identity
        status = self._pipeline.get_detector_status()
        status_path = getattr(status, "model_path", None)
        path = str(status_path) if isinstance(status_path, str | Path) else current.path
        pipeline_version = getattr(self._pipeline, "_model_version_id", None)
        version = (
            pipeline_version
            if isinstance(pipeline_version, str) and pipeline_version
            else self._version_tag or current.version
        )
        self._version_tag = version or ""
        scene_status = self._pipeline.get_scene_guard_status()
        self._model_slot.adopt(
            version=version,
            path=path,
            artifact_sha256=(
                getattr(self._pipeline, "_model_artifact_sha256", None)
                or current.artifact_sha256
            ),
            scene_profile_id=(
                scene_status.get("profile_id")
                if isinstance(scene_status, dict)
                else current.scene_profile_id
            ),
            release_revision=current.release_revision,
        )

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
        latest_frame_identity = None
        get_identity_status = getattr(pipeline, "get_frame_identity_status", None)
        if callable(get_identity_status):
            try:
                identity_status = get_identity_status()
                if isinstance(identity_status, dict):
                    latest = identity_status.get("latest")
                    if isinstance(latest, dict):
                        latest_frame_identity = latest
            except Exception:
                logger.debug(
                    "runner.frame_identity_snapshot_failed",
                    camera_id=self._camera_id,
                    exc_info=True,
                )

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
            latest_frame_identity=latest_frame_identity,
            model_identity=self._model_slot.identity,
        )

    def set_record_store(self, store: InferenceRecordStore | None) -> None:
        """Attach an InferenceRecordStore for audit persistence."""
        self._record_store = store
