"""Four-stage model release pipeline: Candidate → Shadow → Canary → Production.

Every stage transition requires explicit engineer action. No auto-promotion.
Each transition is recorded as a ModelVersionEvent and AuditLog entry.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable

import structlog
from sqlalchemy.orm import Session

from argus.core.error_channel import (
    SEVERITY_CRITICAL,
    SEVERITY_ERROR,
    get_error_channel,
)
from argus.storage.models import (
    AuditLog,
    ModelRecord,
    ModelStage,
    ModelVersionEvent,
    ShadowInferenceLog,
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


class ReleasePipeline:
    """Manages model release lifecycle with four-stage promotion."""

    def __init__(
        self,
        session_factory,
        *,
        min_shadow_days: int = 3,
        min_canary_days: int = 7,
        event_publisher: Callable[[str, dict], None] | None = None,
    ):
        self._session_factory = session_factory
        self._min_shadow_days = min_shadow_days
        self._min_canary_days = min_canary_days
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
    ) -> ModelRecord:
        """Transition a model to a new stage.

        Args:
            model_version_id: The model to transition.
            target_stage: Target stage (shadow, canary, production, retired, candidate).
            triggered_by: Who initiated the transition (username).
            reason: Optional reason for the transition.
            canary_camera_id: Required when target_stage is "canary".
            now: Override current time (for testing).

        Returns:
            Updated ModelRecord.

        Raises:
            ValueError: Model not found or missing required parameters.
            StageTransitionError: Invalid transition or validation failed.
        """
        now = now or datetime.now(timezone.utc)

        with self._session_factory() as session:
            record = (
                session.query(ModelRecord)
                .filter_by(model_version_id=model_version_id)
                .first()
            )
            if record is None:
                raise ValueError(f"Model not found: {model_version_id}")

            current_stage = record.stage
            self._validate_transition(current_stage, target_stage)

            if target_stage == ModelStage.CANARY.value:
                if not canary_camera_id:
                    raise ValueError("canary_camera_id required for canary stage")
                self._validate_min_stage_days(
                    session, record, now,
                    ModelStage.SHADOW.value, self._min_shadow_days, "Shadow",
                )

            if target_stage == ModelStage.PRODUCTION.value:
                self._validate_min_stage_days(
                    session, record, now,
                    ModelStage.CANARY.value, self._min_canary_days, "Canary",
                )

            # 把 transition 的所有 DB 写入(retire 旧 production / 新增 event /
            # audit / commit)整体包在 try 中。SQLAlchemy 的 with-session 上下文
            # 管理器只负责关闭 session,并不会在 commit 抛错时自动回滚 ——
            # 否则可能出现"已经把旧 production retire 了,但新模型 commit 失败"
            # 的孤儿状态。这里在任何异常路径上都先 rollback 再上抛。
            try:
                record.stage = target_stage

                if target_stage == ModelStage.CANARY.value:
                    record.canary_camera_id = canary_camera_id
                elif target_stage == ModelStage.PRODUCTION.value:
                    self._retire_current_production(
                        session, record.camera_id, model_version_id, triggered_by,
                    )
                    record.is_active = True
                    record.canary_camera_id = None
                elif target_stage == ModelStage.RETIRED.value:
                    record.is_active = False
                    record.canary_camera_id = None

                event = ModelVersionEvent(
                    timestamp=now,
                    camera_id=record.camera_id,
                    from_version=model_version_id,
                    to_version=model_version_id,
                    from_stage=current_stage,
                    to_stage=target_stage,
                    triggered_by=triggered_by,
                    reason=reason,
                )
                session.add(event)

                audit = AuditLog(
                    timestamp=now,
                    user=triggered_by,
                    action="model_stage_transition",
                    target_type="model",
                    target_id=model_version_id,
                    detail=f"{current_stage} → {target_stage}" + (f": {reason}" if reason else ""),
                )
                session.add(audit)

                session.commit()
            except Exception as exc:
                # 任何写入路径上的失败都必须把会话回滚,避免把"半成品"
                # transition(例如已 retire 旧 production 但新模型未落库)
                # 留在内存 / 数据库里,然后 re-raise 让上层调用方知情。
                try:
                    session.rollback()
                except Exception as rollback_exc:  # pragma: no cover - defensive
                    logger.error(
                        "release_pipeline.transition_rollback_failed",
                        model_version_id=model_version_id,
                        from_stage=current_stage,
                        to_stage=target_stage,
                        triggered_by=triggered_by,
                        error_type=type(rollback_exc).__name__,
                        error=str(rollback_exc),
                    )
                    get_error_channel().emit(
                        severity=SEVERITY_CRITICAL,
                        source="release_pipeline",
                        code="rollback_failed",
                        message=(
                            f"Stage transition 回滚失败 "
                            f"({current_stage} → {target_stage})"
                        ),
                        context={
                            "model_version_id": model_version_id,
                            "from_stage": current_stage,
                            "to_stage": target_stage,
                            "triggered_by": triggered_by,
                            "error_type": type(rollback_exc).__name__,
                            "error": str(rollback_exc),
                        },
                    )
                logger.error(
                    "release_pipeline.transition_failed",
                    model_version_id=model_version_id,
                    from_stage=current_stage,
                    to_stage=target_stage,
                    triggered_by=triggered_by,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                get_error_channel().emit(
                    severity=SEVERITY_ERROR,
                    source="release_pipeline",
                    code="transition_failed",
                    message=(
                        f"Stage transition 失败 "
                        f"({current_stage} → {target_stage})"
                    ),
                    context={
                        "model_version_id": model_version_id,
                        "from_stage": current_stage,
                        "to_stage": target_stage,
                        "triggered_by": triggered_by,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                raise

            session.refresh(record)

            logger.info(
                "release_pipeline.transition",
                model_version_id=model_version_id,
                from_stage=current_stage,
                to_stage=target_stage,
                triggered_by=triggered_by,
            )

            # Best-effort broadcast for the frontend release-pipeline UI.
            # Runs only after the DB write has fully committed so the
            # frontend never sees a stage transition that ends up rolled
            # back. A broadcast failure must never bubble up — it would
            # turn a successful transition into a 500 for the operator.
            if self._event_publisher is not None:
                try:
                    self._event_publisher(
                        "model_release",
                        {
                            "type": "stage_transition",
                            "model_version_id": model_version_id,
                            "camera_id": record.camera_id,
                            "from_stage": current_stage,
                            "to_stage": target_stage,
                            "triggered_by": triggered_by,
                            "timestamp": now.isoformat(),
                            "reason": reason,
                        },
                    )
                except Exception as publish_exc:
                    logger.warning(
                        "release_pipeline.broadcast_failed",
                        model_version_id=model_version_id,
                        from_stage=current_stage,
                        to_stage=target_stage,
                        error_type=type(publish_exc).__name__,
                        error=str(publish_exc),
                    )

            return record

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
                    "avg_score_divergence": 0.0,
                    "avg_shadow_latency_ms": 0.0,
                }

            total = len(logs)
            shadow_alerts = sum(1 for l in logs if l.shadow_would_alert)
            prod_alerts = sum(1 for l in logs if l.production_alerted)
            score_diffs = [
                abs(l.shadow_score - l.production_score)
                for l in logs
                if l.production_score is not None
            ]
            latencies = [l.latency_ms for l in logs if l.latency_ms is not None]

            return {
                "total_samples": total,
                "shadow_alert_rate": shadow_alerts / total,
                "production_alert_rate": prod_alerts / total,
                "false_positive_delta": shadow_alerts - prod_alerts,
                "avg_score_divergence": sum(score_diffs) / len(score_diffs) if score_diffs else 0.0,
                "avg_shadow_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
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
            session.add(ModelVersionEvent(
                camera_id=camera_id,
                from_version=model.model_version_id,
                to_version=model.model_version_id,
                from_stage=ModelStage.PRODUCTION.value,
                to_stage=ModelStage.RETIRED.value,
                triggered_by=triggered_by,
                reason=f"Replaced by {exclude_version_id}",
            ))
