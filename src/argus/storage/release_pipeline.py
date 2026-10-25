"""Four-stage model release pipeline: Candidate → Shadow → Canary → Production.

Every stage transition requires explicit engineer action. No auto-promotion.
Each transition is recorded as a ModelVersionEvent and AuditLog entry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

import structlog
from sqlalchemy.orm import Session

from argus.core.error_channel import (
    SEVERITY_CRITICAL,
    SEVERITY_ERROR,
    get_error_channel,
)
from argus.preprocessing.scene_profile import runtime_artifact_sha256
from argus.storage.models import (
    AuditLog,
    ModelRecord,
    ModelReleaseOperation,
    ModelStage,
    ModelVersionEvent,
    ReleaseOperationStatus,
    ShadowInferenceLog,
)
from argus.storage.release_evidence import (
    ReleaseEvidenceValidation,
    validate_release_evidence,
)

logger = structlog.get_logger()

# Valid stage transitions (from_stage -> set of allowed to_stages).
# Canonical definition — ModelRegistry imports this to avoid duplication.
VALID_TRANSITIONS: dict[str, set[str]] = {
    ModelStage.CANDIDATE.value: {ModelStage.SHADOW.value, ModelStage.RETIRED.value},
    ModelStage.SHADOW.value: {
        ModelStage.CANARY.value,
        ModelStage.CANDIDATE.value,
        ModelStage.RETIRED.value,
    },
    ModelStage.CANARY.value: {
        ModelStage.PRODUCTION.value,
        ModelStage.SHADOW.value,
        ModelStage.RETIRED.value,
    },
    ModelStage.PRODUCTION.value: {ModelStage.RETIRED.value},
}


def _ensure_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC-aware. Treats naive datetimes as UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class BackboneIncompatibleError(Exception):
    """Raised when a head's backbone_ref doesn't match the loaded backbone."""


class StageTransitionError(Exception):
    """Raised when a stage transition is invalid."""


@dataclass(frozen=True, slots=True)
class ReleaseTransitionPlan:
    """Immutable result of policy validation before runtime preparation.

    The plan contains only registry-authorized identity.  It is deliberately
    not a mutated ``ModelRecord``: callers may load a standby runtime from this
    exact path/hash while the applied registry row remains untouched.
    ``commit_transition`` rechecks every compare-and-swap and artifact binding
    before it writes the applied stage.
    """

    model_version_id: str
    camera_id: str
    from_stage: str
    target_stage: str
    expected_revision: int
    canary_camera_id: str | None
    model_path: str | None
    runtime_artifact_sha256: str | None
    release_evidence_json: str | None
    release_evidence_sha256: str | None
    validated_at: datetime
    rollback_context: dict | None = None


class ReleasePipeline:
    """Manages model release lifecycle with four-stage promotion."""

    def __init__(
        self,
        session_factory,
        *,
        min_shadow_days: int = 3,
        min_canary_days: int = 7,
        min_shadow_samples: int = 0,
        max_shadow_alert_rate_delta: float = 1.0,
        max_shadow_score_divergence: float = 1.0,
        max_shadow_latency_ms: float = 0.0,
        event_publisher: Callable[[str, dict], None] | None = None,
    ):
        self._session_factory = session_factory
        self._min_shadow_days = min_shadow_days
        self._min_canary_days = min_canary_days
        self._min_shadow_samples = int(min_shadow_samples)
        self._max_shadow_alert_rate_delta = float(max_shadow_alert_rate_delta)
        self._max_shadow_score_divergence = float(max_shadow_score_divergence)
        self._max_shadow_latency_ms = float(max_shadow_latency_ms)
        # Optional broadcaster (typically ws_manager.broadcast) so the
        # frontend can show stage-transition progress in real time. Kept
        # optional so unit tests / non-dashboard callers stay unaffected.
        self._event_publisher = event_publisher

    def transition(
        self,
        model_version_id: str,
        target_stage: str,
        triggered_by: str,
        reason: str | None = None,
        canary_camera_id: str | None = None,
        *,
        now: datetime | None = None,
        release_evidence: object | None = None,
        expected_revision: int | None = None,
    ) -> ModelRecord:
        """Validate and atomically persist a legacy synchronous transition.

        New application code must use :class:`ModelReleaseService`, which
        inserts runtime prepare/commit between these two pipeline phases.
        Keeping this wrapper preserves CLI and focused pipeline callers without
        reintroducing an intermediate committed stage.
        """

        plan = self.prepare_transition(
            model_version_id=model_version_id,
            target_stage=target_stage,
            canary_camera_id=canary_camera_id,
            now=now,
            release_evidence=release_evidence,
            expected_revision=expected_revision,
        )
        return self.commit_transition(
            plan,
            triggered_by=triggered_by,
            reason=reason,
        )

    def prepare_transition(
        self,
        *,
        model_version_id: str,
        target_stage: str,
        canary_camera_id: str | None = None,
        now: datetime | None = None,
        release_evidence: object | None = None,
        expected_revision: int | None = None,
    ) -> ReleaseTransitionPlan:
        """Validate policy and bind the exact registry artifact without writes."""

        now = now or datetime.now(timezone.utc)
        with self._session_factory() as session:
            record = (
                session.query(ModelRecord)
                .filter_by(model_version_id=model_version_id)
                .first()
            )
            if record is None:
                raise ValueError(f"Model not found: {model_version_id}")

            current_revision = int(record.release_revision or 0)
            if expected_revision is not None and int(expected_revision) != current_revision:
                raise StageTransitionError(
                    "Stale release revision: "
                    f"expected {expected_revision}, current {current_revision}"
                )
            if canary_camera_id is not None and canary_camera_id != record.camera_id:
                raise ValueError(
                    "canary_camera_id must match model camera_id "
                    f"({record.camera_id})"
                )

            self._validate_transition(record.stage, target_stage)
            self._validate_runtime_slot(
                session,
                camera_id=record.camera_id,
                model_version_id=record.model_version_id,
                target_stage=target_stage,
            )
            if target_stage == ModelStage.CANARY.value:
                if not canary_camera_id:
                    raise ValueError("canary_camera_id required for canary stage")
                self._validate_min_stage_days(
                    session,
                    record,
                    now,
                    ModelStage.SHADOW.value,
                    self._min_shadow_days,
                    "Shadow",
                )
                self._validate_shadow_evidence(session, record, now)
            if target_stage == ModelStage.PRODUCTION.value:
                self._validate_min_stage_days(
                    session,
                    record,
                    now,
                    ModelStage.CANARY.value,
                    self._min_canary_days,
                    "Canary",
                )

            inactive_candidate_retirement = (
                record.stage == ModelStage.CANDIDATE.value
                and target_stage == ModelStage.RETIRED.value
            )
            if inactive_candidate_retirement and record.is_active:
                raise StageTransitionError(
                    "An active candidate cannot use the no-runtime retirement path"
                )
            artifact_sha256: str | None = None
            if record.model_path and not inactive_candidate_retirement:
                try:
                    artifact_sha256 = runtime_artifact_sha256(record.model_path)
                except (FileNotFoundError, OSError, ValueError) as exc:
                    raise StageTransitionError(
                        f"Runtime artifact fingerprint unavailable: {exc}"
                    ) from exc

            release_validation: ReleaseEvidenceValidation | None = None
            if target_stage == ModelStage.PRODUCTION.value:
                release_validation = self._validate_production_evidence(
                    record,
                    release_evidence,
                    artifact_sha256=artifact_sha256,
                )
                if not release_validation.valid:
                    raise StageTransitionError(
                        "Production release evidence rejected: "
                        + "; ".join(release_validation.reasons)
                    )

            return ReleaseTransitionPlan(
                model_version_id=record.model_version_id,
                camera_id=record.camera_id,
                from_stage=record.stage,
                target_stage=target_stage,
                expected_revision=current_revision,
                canary_camera_id=canary_camera_id,
                model_path=record.model_path,
                runtime_artifact_sha256=artifact_sha256,
                release_evidence_json=(
                    release_validation.canonical_json if release_validation else None
                ),
                release_evidence_sha256=(
                    release_validation.sha256 if release_validation else None
                ),
                validated_at=now,
            )

    def prepare_rollback(
        self,
        *,
        model_version_id: str,
        expected_revision: int,
        rollback_context: dict,
        release_evidence: object,
    ) -> ReleaseTransitionPlan:
        """Restore only proven production history; never alter normal stage rules."""
        from argus.storage.model_registry import validate_persisted_release_evidence
        from argus.storage.rollback_policy import validate_rollback_context

        with self._session_factory() as session:
            record = validate_rollback_context(session, model_version_id, rollback_context)
            if int(record.release_revision or 0) != expected_revision:
                raise StageTransitionError("Stale rollback target revision")
            old_validation = validate_persisted_release_evidence(record)
            if not old_validation.valid:
                raise StageTransitionError(
                    "Original production evidence rejected: " + "; ".join(old_validation.reasons)
                )
            artifact = runtime_artifact_sha256(record.model_path)
            validation = self._validate_production_evidence(
                record, release_evidence, artifact_sha256=artifact,
            )
            if not validation.valid:
                raise StageTransitionError(
                    "Rollback production evidence rejected: " + "; ".join(validation.reasons)
                )
            return ReleaseTransitionPlan(
                model_version_id=record.model_version_id, camera_id=record.camera_id,
                from_stage=record.stage, target_stage=ModelStage.PRODUCTION.value,
                expected_revision=expected_revision, canary_camera_id=None,
                model_path=record.model_path, runtime_artifact_sha256=artifact,
                release_evidence_json=validation.canonical_json,
                release_evidence_sha256=validation.sha256,
                validated_at=datetime.now(timezone.utc),
                rollback_context=json.loads(json.dumps(rollback_context)),
            )

    def commit_transition(
        self,
        plan: ReleaseTransitionPlan,
        *,
        triggered_by: str,
        reason: str | None = None,
        operation_id: str | None = None,
        runtime_state: object = None,
    ) -> ModelRecord:
        """CAS-commit an already prepared runtime identity and desired stage.

        When ``operation_id`` is supplied, the applied model row and operation
        status are committed in the same database transaction.  Any stale row,
        artifact mutation, unique-production conflict, or commit exception
        rolls the entire unit back.
        """

        if not isinstance(triggered_by, str) or not triggered_by.strip():
            raise ValueError("triggered_by is required for audit compliance")
        triggered_by = triggered_by.strip()
        with self._session_factory() as session:
            try:
                operation = None
                if operation_id is not None:
                    operation = (
                        session.query(ModelReleaseOperation)
                        .filter_by(operation_id=operation_id)
                        .first()
                    )
                    if operation is None:
                        raise StageTransitionError(
                            f"Release operation not found: {operation_id}"
                        )
                    if operation.status != ReleaseOperationStatus.PENDING.value:
                        raise StageTransitionError(
                            f"Release operation is not pending: {operation.status}"
                        )
                    if (
                        operation.model_version_id != plan.model_version_id
                        or operation.camera_id != plan.camera_id
                        or operation.from_stage != plan.from_stage
                        or operation.target_stage != plan.target_stage
                        or int(operation.expected_revision) != plan.expected_revision
                    ):
                        raise StageTransitionError(
                            "Release operation no longer matches validated plan"
                        )

                if plan.rollback_context is not None:
                    from argus.storage.model_registry import validate_persisted_release_evidence
                    from argus.storage.rollback_policy import validate_rollback_context

                    if operation is None or json.loads(operation.desired_state_json).get(
                        "rollback"
                    ) != plan.rollback_context:
                        raise StageTransitionError("Rollback requires its exact durable command")
                    source = plan.rollback_context
                    reserved = session.query(ModelRecord).filter_by(
                        model_version_id=source["source_model_version_id"],
                        camera_id=plan.camera_id, stage="production", is_active=True,
                        release_revision=source["source_release_revision"],
                    ).update(
                        {"release_revision": source["source_release_revision"]},
                        synchronize_session=False,
                    )
                    if reserved != 1:
                        raise StageTransitionError("Rollback source production changed")
                    target = validate_rollback_context(
                        session, plan.model_version_id, plan.rollback_context,
                    )
                    validation = validate_persisted_release_evidence(target)
                    if not validation.valid:
                        raise StageTransitionError("Rollback original evidence changed before commit")

                # The conditional UPDATE is the cross-process CAS.  A normal
                # query/check pair would permit two workers to validate the
                # same revision and both overwrite it later.
                inactive_candidate_retirement = (
                    plan.from_stage == ModelStage.CANDIDATE.value
                    and plan.target_stage == ModelStage.RETIRED.value
                )
                update_query = (
                    session.query(ModelRecord)
                    .filter(
                        ModelRecord.model_version_id == plan.model_version_id,
                        ModelRecord.camera_id == plan.camera_id,
                        ModelRecord.stage == plan.from_stage,
                        ModelRecord.release_revision == plan.expected_revision,
                        ModelRecord.model_path == plan.model_path,
                    )
                )
                if inactive_candidate_retirement:
                    update_query = update_query.filter(ModelRecord.is_active.is_(False))
                updated = update_query.update(
                    {
                        ModelRecord.stage: plan.target_stage,
                        ModelRecord.release_revision: plan.expected_revision + 1,
                    },
                    synchronize_session=False,
                )
                if updated != 1:
                    raise StageTransitionError(
                        "Stale release plan: registry stage, revision, or artifact path changed"
                    )
                record = (
                    session.query(ModelRecord)
                    .filter_by(model_version_id=plan.model_version_id)
                    .populate_existing()
                    .one()
                )

                # The preceding UPDATE owns SQLite's write transaction. A
                # second prepared request cannot occupy the same one-slot
                # runtime between this check and the applied-state commit.
                self._validate_runtime_slot(
                    session,
                    camera_id=record.camera_id,
                    model_version_id=record.model_version_id,
                    target_stage=plan.target_stage,
                )

                if plan.model_path and not inactive_candidate_retirement:
                    current_artifact_sha256 = runtime_artifact_sha256(plan.model_path)
                    if current_artifact_sha256 != plan.runtime_artifact_sha256:
                        raise StageTransitionError(
                            "Runtime artifact changed after release preparation"
                        )
                elif plan.runtime_artifact_sha256 is not None:
                    raise StageTransitionError(
                        "Validated runtime artifact identity is inconsistent"
                    )

                if plan.target_stage == ModelStage.CANARY.value:
                    record.canary_camera_id = plan.canary_camera_id
                elif plan.target_stage == ModelStage.PRODUCTION.value:
                    self._retire_current_production(
                        session,
                        record.camera_id,
                        plan.model_version_id,
                        triggered_by,
                    )
                    self._retire_current_canaries(
                        session,
                        record.camera_id,
                        plan.model_version_id,
                        triggered_by,
                    )
                    session.flush()
                    record.is_active = True
                    record.canary_camera_id = None
                    record.release_evidence_json = plan.release_evidence_json
                    record.release_evidence_sha256 = plan.release_evidence_sha256
                    if not record.release_evidence_json or not record.release_evidence_sha256:
                        raise StageTransitionError(
                            "Validated production evidence is missing from release plan"
                        )
                elif plan.target_stage == ModelStage.RETIRED.value:
                    record.is_active = False
                    record.canary_camera_id = None

                session.add(ModelVersionEvent(
                    timestamp=plan.validated_at,
                    camera_id=record.camera_id,
                    from_version=plan.model_version_id,
                    to_version=plan.model_version_id,
                    from_stage=plan.from_stage,
                    to_stage=plan.target_stage,
                    triggered_by=triggered_by,
                    reason=reason,
                ))
                session.add(AuditLog(
                    timestamp=plan.validated_at,
                    user=triggered_by,
                    action="model_rollback" if plan.rollback_context else "model_stage_transition",
                    target_type="model",
                    target_id=plan.model_version_id,
                    detail=(
                        f"{plan.from_stage} → {plan.target_stage}"
                        + (f": {reason}" if reason else "")
                    ),
                ))
                if operation is not None:
                    operation.status = ReleaseOperationStatus.APPLIED.value
                    operation.applied_revision = plan.expected_revision + 1
                    operation.runtime_state_json = json.dumps(
                        runtime_state,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                    operation.error = None
                    operation.completed_at = datetime.now(timezone.utc)

                session.commit()
                session.refresh(record)
            except Exception as exc:
                try:
                    session.rollback()
                except Exception as rollback_exc:  # pragma: no cover - defensive
                    logger.error(
                        "release_pipeline.transition_rollback_failed",
                        model_version_id=plan.model_version_id,
                        from_stage=plan.from_stage,
                        to_stage=plan.target_stage,
                        triggered_by=triggered_by,
                        error_type=type(rollback_exc).__name__,
                        error=str(rollback_exc),
                    )
                    get_error_channel().emit(
                        severity=SEVERITY_CRITICAL,
                        source="release_pipeline",
                        code="rollback_failed",
                        message=(
                            "Stage transition rollback failed "
                            f"({plan.from_stage} → {plan.target_stage})"
                        ),
                        context={
                            "model_version_id": plan.model_version_id,
                            "from_stage": plan.from_stage,
                            "to_stage": plan.target_stage,
                            "triggered_by": triggered_by,
                            "error_type": type(rollback_exc).__name__,
                            "error": str(rollback_exc),
                        },
                    )
                logger.error(
                    "release_pipeline.transition_failed",
                    model_version_id=plan.model_version_id,
                    from_stage=plan.from_stage,
                    to_stage=plan.target_stage,
                    triggered_by=triggered_by,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                get_error_channel().emit(
                    severity=SEVERITY_ERROR,
                    source="release_pipeline",
                    code="transition_failed",
                    message=(
                        "Stage transition failed "
                        f"({plan.from_stage} → {plan.target_stage})"
                    ),
                    context={
                        "model_version_id": plan.model_version_id,
                        "from_stage": plan.from_stage,
                        "to_stage": plan.target_stage,
                        "triggered_by": triggered_by,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                raise

        logger.info(
            "release_pipeline.transition",
            model_version_id=plan.model_version_id,
            from_stage=plan.from_stage,
            to_stage=plan.target_stage,
            triggered_by=triggered_by,
        )
        self._publish_transition(plan, triggered_by=triggered_by, reason=reason)
        return record

    def _publish_transition(
        self,
        plan: ReleaseTransitionPlan,
        *,
        triggered_by: str,
        reason: str | None,
    ) -> None:
        if self._event_publisher is None:
            return
        try:
            self._event_publisher("model_release", {
                "type": "stage_transition",
                "model_version_id": plan.model_version_id,
                "camera_id": plan.camera_id,
                "from_stage": plan.from_stage,
                "to_stage": plan.target_stage,
                "triggered_by": triggered_by,
                "timestamp": plan.validated_at.isoformat(),
                "reason": reason,
            })
        except Exception as publish_exc:
            logger.warning(
                "release_pipeline.broadcast_failed",
                model_version_id=plan.model_version_id,
                from_stage=plan.from_stage,
                to_stage=plan.target_stage,
                error_type=type(publish_exc).__name__,
                error=str(publish_exc),
            )

    @staticmethod
    def _validate_production_evidence(
        record: ModelRecord,
        evidence: object,
        *,
        artifact_sha256: str | None = None,
    ) -> ReleaseEvidenceValidation:
        """Bind release evidence to this row and its on-disk runtime artifact."""

        if not record.model_path:
            return ReleaseEvidenceValidation(
                valid=False,
                reasons=("registry model_path is missing",),
                canonical_json=None,
                sha256=None,
            )
        if artifact_sha256 is None:
            try:
                artifact_sha256 = runtime_artifact_sha256(record.model_path)
            except (FileNotFoundError, OSError, ValueError) as exc:
                return ReleaseEvidenceValidation(
                    valid=False,
                    reasons=(f"runtime artifact fingerprint unavailable: {exc}",),
                    canonical_json=None,
                    sha256=None,
                )
        return validate_release_evidence(
            evidence,
            model_version_id=record.model_version_id,
            camera_id=record.camera_id,
            registry_model_hash=record.model_hash,
            runtime_artifact_sha256=artifact_sha256,
        )

    def get_shadow_stats(
        self,
        shadow_version_id: str,
        camera_id: str | None = None,
        days: int = 7,
    ) -> dict:
        """Get shadow inference comparison statistics."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        with self._session_factory() as session:
            query = (
                session.query(ShadowInferenceLog)
                .filter(ShadowInferenceLog.shadow_version_id == shadow_version_id)
                .filter(ShadowInferenceLog.timestamp >= cutoff)
            )
            if camera_id:
                query = query.filter(ShadowInferenceLog.camera_id == camera_id)

            logs = list(query.all())

            if not logs:
                return {
                    "total_samples": 0,
                    "shadow_alert_rate": 0.0,
                    "production_alert_rate": 0.0,
                    "false_positive_delta": 0,
                    "alert_rate_delta": 0.0,
                    "abs_alert_rate_delta": 0.0,
                    "avg_score_divergence": 0.0,
                    "avg_shadow_latency_ms": 0.0,
                }

            return self._shadow_stats_from_logs(logs)

    def get_shadow_gate_report(
        self,
        shadow_version_id: str,
        camera_id: str | None = None,
        days: int = 7,
    ) -> dict:
        """Return configured shadow-to-canary readiness checks."""
        stats = self.get_shadow_stats(
            shadow_version_id=shadow_version_id,
            camera_id=camera_id,
            days=days,
        )
        checks = [
            {
                "name": "min_shadow_samples",
                "label": "Minimum shadow samples",
                "value": stats["total_samples"],
                "limit": self._min_shadow_samples,
                "passed": stats["total_samples"] >= self._min_shadow_samples,
                "enabled": self._min_shadow_samples > 0,
            },
            {
                "name": "max_shadow_alert_rate_delta",
                "label": "Shadow/production alert-rate delta",
                "value": stats["abs_alert_rate_delta"],
                "limit": self._max_shadow_alert_rate_delta,
                "passed": (
                    stats["abs_alert_rate_delta"]
                    <= self._max_shadow_alert_rate_delta
                ),
                "enabled": self._max_shadow_alert_rate_delta < 1.0,
            },
            {
                "name": "max_shadow_score_divergence",
                "label": "Average score divergence",
                "value": stats["avg_score_divergence"],
                "limit": self._max_shadow_score_divergence,
                "passed": (
                    stats["avg_score_divergence"]
                    <= self._max_shadow_score_divergence
                ),
                "enabled": self._max_shadow_score_divergence < 1.0,
            },
            {
                "name": "max_shadow_latency_ms",
                "label": "Average shadow latency",
                "value": stats["avg_shadow_latency_ms"],
                "limit": self._max_shadow_latency_ms,
                "passed": (
                    self._max_shadow_latency_ms <= 0
                    or stats["avg_shadow_latency_ms"] <= self._max_shadow_latency_ms
                ),
                "enabled": self._max_shadow_latency_ms > 0,
            },
        ]
        enabled_checks = [check for check in checks if check["enabled"]]
        return {
            "ready_for_canary": all(check["passed"] for check in enabled_checks),
            "checks": checks,
        }

    @staticmethod
    def validate_backbone_compatibility(
        head_record: ModelRecord,
        loaded_backbone_version: str,
    ) -> None:
        """Validate head is compatible with the loaded backbone.

        Raises BackboneIncompatibleError if backbone_ref doesn't match.
        """
        if head_record.component_type != "head":
            return

        if head_record.backbone_version_id is None:
            return

        if head_record.backbone_version_id != loaded_backbone_version:
            raise BackboneIncompatibleError(
                f"Head {head_record.model_version_id} requires backbone "
                f"{head_record.backbone_version_id}, but loaded backbone is "
                f"{loaded_backbone_version}"
            )

    @staticmethod
    def _validate_runtime_slot(
        session: Session,
        *,
        camera_id: str,
        model_version_id: str,
        target_stage: str,
    ) -> None:
        """Reject implicit replacement of the camera's shadow/canary slot.

        A stage is an applied assignment, not a list of possible models for a
        router to rank by training time. Operators must explicitly withdraw
        the previous assignment before publishing a different version.
        """
        if target_stage not in {ModelStage.SHADOW.value, ModelStage.CANARY.value}:
            return
        occupied = session.query(ModelRecord.model_version_id).filter(
            ModelRecord.camera_id == camera_id,
            ModelRecord.stage == target_stage,
            ModelRecord.model_version_id != model_version_id,
        ).first()
        if occupied is not None:
            raise StageTransitionError(
                f"Camera {camera_id} {target_stage} runtime slot is occupied by "
                f"{occupied[0]}; explicitly withdraw that version first"
            )

    def _validate_transition(self, current_stage: str, target_stage: str) -> None:
        """Check if the transition is in the allowed set."""
        allowed = VALID_TRANSITIONS.get(current_stage, set())
        if target_stage not in allowed:
            raise StageTransitionError(
                f"Invalid transition: {current_stage} → {target_stage}. "
                f"Allowed: {sorted(allowed)}"
            )

    @staticmethod
    def _validate_min_stage_days(
        session: Session,
        record: ModelRecord,
        now: datetime,
        required_stage: str,
        min_days: int,
        stage_label: str,
    ) -> None:
        """Check that a model has spent enough time in a required stage."""
        event = (
            session.query(ModelVersionEvent)
            .filter_by(to_version=record.model_version_id, to_stage=required_stage)
            .order_by(ModelVersionEvent.timestamp.desc())
            .first()
        )
        if event is None:
            raise StageTransitionError(
                f"No {stage_label.lower()} stage entry found for this model"
            )

        days_elapsed = (
            _ensure_utc(now) - _ensure_utc(event.timestamp)
        ).total_seconds() / 86400

        if days_elapsed < min_days:
            raise StageTransitionError(
                f"{stage_label} period too short: {days_elapsed:.1f} days "
                f"(minimum {min_days} days)"
            )

    def _validate_shadow_evidence(
        self,
        session: Session,
        record: ModelRecord,
        now: datetime,
    ) -> None:
        """Require enough clean shadow evidence before canary promotion."""
        if (
            self._min_shadow_samples <= 0
            and self._max_shadow_alert_rate_delta >= 1.0
            and self._max_shadow_score_divergence >= 1.0
            and self._max_shadow_latency_ms <= 0
        ):
            return

        shadow_entry = (
            session.query(ModelVersionEvent)
            .filter_by(
                to_version=record.model_version_id,
                to_stage=ModelStage.SHADOW.value,
            )
            .order_by(ModelVersionEvent.timestamp.desc())
            .first()
        )
        if shadow_entry is None:
            raise StageTransitionError(
                "No shadow stage entry found for this model"
            )

        logs = (
            session.query(ShadowInferenceLog)
            .filter(ShadowInferenceLog.shadow_version_id == record.model_version_id)
            .filter(ShadowInferenceLog.camera_id == record.camera_id)
            .filter(ShadowInferenceLog.timestamp >= shadow_entry.timestamp)
            .filter(ShadowInferenceLog.timestamp <= now)
            .all()
        )
        stats = self._shadow_stats_from_logs(logs)
        if stats["total_samples"] < self._min_shadow_samples:
            raise StageTransitionError(
                "Shadow evidence too sparse: "
                f"{stats['total_samples']} samples "
                f"(minimum {self._min_shadow_samples})"
            )
        if stats["abs_alert_rate_delta"] > self._max_shadow_alert_rate_delta:
            raise StageTransitionError(
                "Shadow alert-rate delta too high: "
                f"{stats['abs_alert_rate_delta']:.3f} "
                f"(maximum {self._max_shadow_alert_rate_delta:.3f})"
            )
        if stats["avg_score_divergence"] > self._max_shadow_score_divergence:
            raise StageTransitionError(
                "Shadow score divergence too high: "
                f"{stats['avg_score_divergence']:.3f} "
                f"(maximum {self._max_shadow_score_divergence:.3f})"
            )
        if (
            self._max_shadow_latency_ms > 0
            and stats["avg_shadow_latency_ms"] > self._max_shadow_latency_ms
        ):
            raise StageTransitionError(
                "Shadow latency too high: "
                f"{stats['avg_shadow_latency_ms']:.1f} ms "
                f"(maximum {self._max_shadow_latency_ms:.1f} ms)"
            )

    @staticmethod
    def _shadow_stats_from_logs(logs: list[ShadowInferenceLog]) -> dict:
        """Compute aggregate shadow-vs-production statistics."""
        if not logs:
            return {
                "total_samples": 0,
                "shadow_alert_rate": 0.0,
                "production_alert_rate": 0.0,
                "false_positive_delta": 0,
                "alert_rate_delta": 0.0,
                "abs_alert_rate_delta": 0.0,
                "avg_score_divergence": 0.0,
                "avg_shadow_latency_ms": 0.0,
            }

        total = len(logs)
        shadow_alerts = sum(1 for log in logs if log.shadow_would_alert)
        prod_alerts = sum(1 for log in logs if log.production_alerted)
        score_diffs = [
            abs(log.shadow_score - log.production_score)
            for log in logs
            if log.production_score is not None
        ]
        latencies = [log.latency_ms for log in logs if log.latency_ms is not None]
        shadow_rate = shadow_alerts / total
        production_rate = prod_alerts / total
        alert_rate_delta = shadow_rate - production_rate

        return {
            "total_samples": total,
            "shadow_alert_rate": shadow_rate,
            "production_alert_rate": production_rate,
            "false_positive_delta": shadow_alerts - prod_alerts,
            "alert_rate_delta": alert_rate_delta,
            "abs_alert_rate_delta": abs(alert_rate_delta),
            "avg_score_divergence": (
                sum(score_diffs) / len(score_diffs) if score_diffs else 0.0
            ),
            "avg_shadow_latency_ms": (
                sum(latencies) / len(latencies) if latencies else 0.0
            ),
        }

    @staticmethod
    def _retire_current_production(
        session: Session,
        camera_id: str,
        exclude_version_id: str,
        triggered_by: str = "system",
    ) -> None:
        """Retire existing production models for this camera (except the one being promoted)."""
        production_models = (
            session.query(ModelRecord)
            .filter_by(camera_id=camera_id, stage=ModelStage.PRODUCTION.value)
            .filter(ModelRecord.model_version_id != exclude_version_id)
            .all()
        )
        for model in production_models:
            model.stage = ModelStage.RETIRED.value
            model.is_active = False
            model.release_revision = int(model.release_revision or 0) + 1
            session.add(ModelVersionEvent(
                camera_id=camera_id,
                from_version=model.model_version_id,
                to_version=model.model_version_id,
                from_stage=ModelStage.PRODUCTION.value,
                to_stage=ModelStage.RETIRED.value,
                triggered_by=triggered_by,
                reason=f"Replaced by {exclude_version_id}",
            ))

    @staticmethod
    def _retire_current_canaries(
        session: Session,
        camera_id: str,
        exclude_version_id: str,
        triggered_by: str = "system",
    ) -> None:
        """Retire stale canaries for this camera when a model reaches production."""
        canary_models = (
            session.query(ModelRecord)
            .filter_by(camera_id=camera_id, stage=ModelStage.CANARY.value)
            .filter(ModelRecord.model_version_id != exclude_version_id)
            .all()
        )
        for model in canary_models:
            model.stage = ModelStage.RETIRED.value
            model.is_active = False
            model.canary_camera_id = None
            model.release_revision = int(model.release_revision or 0) + 1
            session.add(ModelVersionEvent(
                camera_id=camera_id,
                from_version=model.model_version_id,
                to_version=model.model_version_id,
                from_stage=ModelStage.CANARY.value,
                to_stage=ModelStage.RETIRED.value,
                triggered_by=triggered_by,
                reason=f"Replaced by production {exclude_version_id}",
            ))
