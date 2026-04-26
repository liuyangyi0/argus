"""SQLAlchemy ORM models for alert and training persistence."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class ModelStage(str, Enum):
    """Lifecycle stage for model release pipeline."""

    CANDIDATE = "candidate"
    SHADOW = "shadow"
    CANARY = "canary"
    PRODUCTION = "production"
    RETIRED = "retired"


class AlertWorkflowStatus(str, Enum):
    """Workflow status for alert lifecycle management."""

    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"
    FALSE_POSITIVE = "false_positive"
    UNCERTAIN = "uncertain"


class FeedbackType(str, Enum):
    """Operator feedback classification for alerts."""

    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    UNCERTAIN = "uncertain"


class FeedbackStatus(str, Enum):
    """Processing status for feedback entries in the retraining queue."""

    PENDING = "pending"
    PROCESSED = "processed"
    SKIPPED = "skipped"


class FeedbackSource(str, Enum):
    """Origin of a feedback entry."""

    MANUAL = "manual"
    DRIFT = "drift"
    HEALTH = "health"


class BaselineState(str, Enum):
    """Lifecycle state for baseline versions (nuclear audit requirement)."""

    DRAFT = "draft"
    VERIFIED = "verified"
    ACTIVE = "active"
    RETIRED = "retired"


class Base(DeclarativeBase):
    pass


class AlertRecord(Base):
    """Persisted alert record."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    zone_id: Mapped[str] = mapped_column(String(50), nullable=False)
    severity: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    snapshot_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    heatmap_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    false_positive: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    workflow_status: Mapped[str] = mapped_column(
        String(20), default="new", server_default="new", nullable=False
    )
    assigned_to: Mapped[str | None] = mapped_column(String(100), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Alert aggregation: alerts in same camera+zone within a window share this ID
    event_group_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    event_group_count: Mapped[int] = mapped_column(Integer, default=1, server_default="1")

    # Physics enrichment (phase 1: speed, phase 2: trajectory/localization)
    speed_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    speed_px_per_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    trajectory_model: Mapped[str | None] = mapped_column(String(20), nullable=True)
    origin_x_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    origin_y_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    origin_z_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    landing_x_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    landing_y_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    landing_z_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Multi-track trajectory fits (JSON list of per-track origin/landing/speed/model
    # with pixel-space projections for video overlay). Primary track mirrors the
    # origin_*_mm / landing_*_mm / speed_* scalar fields above.
    trajectories_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Classification enrichment
    classification_label: Mapped[str | None] = mapped_column(String(100), nullable=True)
    classification_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Cross-camera corroboration (C3-4 — see src/argus/core/correlation.py).
    # ``corroborated`` is False when the corroborator ran but the partner
    # camera could not confirm the anomaly at the transformed location, and
    # None when cross-camera correlation is disabled for this pipeline.
    corroborated: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    correlation_partner: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # Segmentation enrichment (D2 — SAM2 instance segmentation).
    # ``segmentation_objects`` is a JSON array of per-object dicts
    # (bbox/area_px/centroid/confidence); the raw masks are dropped
    # because they are far too large to persist per-alert.
    segmentation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    segmentation_total_area_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    segmentation_objects: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Alert category auto-classification
    category: Mapped[str | None] = mapped_column(String(30), nullable=True)
    severity_adjusted_by_classifier: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    # Trajectory centroid history (JSON array of {t, x, y})
    trajectory_points: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Model version lineage — points at the active model that produced this alert,
    # so audits can answer "which model fired this?" even after a model switch.
    model_version_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True, index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def to_dict(self) -> dict:
        import json as _json
        segmentation_objects_parsed: list | None = None
        if self.segmentation_objects:
            try:
                segmentation_objects_parsed = _json.loads(self.segmentation_objects)
            except Exception:
                segmentation_objects_parsed = None
        trajectories_parsed: list | None = None
        if self.trajectories_json:
            try:
                trajectories_parsed = _json.loads(self.trajectories_json)
            except Exception:
                trajectories_parsed = None
        return {
            "id": self.id,
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "severity": self.severity,
            "anomaly_score": self.anomaly_score,
            "snapshot_path": self.snapshot_path,
            "heatmap_path": self.heatmap_path,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "false_positive": self.false_positive,
            "notes": self.notes,
            "workflow_status": self.workflow_status,
            "assigned_to": self.assigned_to,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "event_group_id": self.event_group_id,
            "event_group_count": self.event_group_count,
            "speed_ms": self.speed_ms,
            "speed_px_per_sec": self.speed_px_per_sec,
            "trajectory_model": self.trajectory_model,
            "origin_x_mm": self.origin_x_mm,
            "origin_y_mm": self.origin_y_mm,
            "origin_z_mm": self.origin_z_mm,
            "landing_x_mm": self.landing_x_mm,
            "landing_y_mm": self.landing_y_mm,
            "landing_z_mm": self.landing_z_mm,
            "trajectories": trajectories_parsed,
            "classification_label": self.classification_label,
            "classification_confidence": self.classification_confidence,
            "corroborated": self.corroborated,
            "correlation_partner": self.correlation_partner,
            "segmentation_count": self.segmentation_count,
            "segmentation_total_area_px": self.segmentation_total_area_px,
            "segmentation_objects": segmentation_objects_parsed,
            "category": self.category,
            "severity_adjusted_by_classifier": self.severity_adjusted_by_classifier,
            "trajectory_points": _json.loads(self.trajectory_points) if self.trajectory_points else None,
            "model_version_id": self.model_version_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class BaselineRecord(Base):
    """Tracks baseline images used for training."""

    __tablename__ = "baselines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    zone_id: Mapped[str] = mapped_column(String(50), nullable=False)
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    model_trained_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )


class BaselineVersionRecord(Base):
    """Tracks lifecycle state of baseline versions (nuclear audit compliance).

    Each row represents one baseline version (e.g. v001) for a camera/zone.
    State transitions: Draft -> Verified -> Active -> Retired (strict, unidirectional).
    """

    __tablename__ = "baseline_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    zone_id: Mapped[str] = mapped_column(String(50), nullable=False)
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    state: Mapped[str] = mapped_column(String(20), nullable=False, default="draft")
    image_count: Mapped[int] = mapped_column(Integer, default=0)

    # Verification (requires at least one approver)
    verified_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    verified_by_secondary: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Activation / retirement
    activated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    retirement_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Camera group support (Phase 2)
    group_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "version": self.version,
            "state": self.state,
            "image_count": self.image_count,
            "verified_by": self.verified_by,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "verified_by_secondary": self.verified_by_secondary,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "retired_at": self.retired_at.isoformat() if self.retired_at else None,
            "retirement_reason": self.retirement_reason,
            "group_id": self.group_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TrainingRecord(Base):
    """Records each model training run with parameters, metrics, and results."""

    __tablename__ = "training_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    zone_id: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[str] = mapped_column(String(30), nullable=False)
    export_format: Mapped[str | None] = mapped_column(String(30), nullable=True)
    baseline_version: Mapped[str] = mapped_column(String(20), nullable=False)
    baseline_count: Mapped[int] = mapped_column(Integer, nullable=False)
    train_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    val_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Pre-validation (TRN-001)
    pre_validation_passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    corruption_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    near_duplicate_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    brightness_std: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Quality metrics (TRN-003/004/005)
    val_score_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_score_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_score_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_score_p95: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_grade: Mapped[str | None] = mapped_column(String(1), nullable=True)
    threshold_recommended: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Real-labeled P/R/F1/AUROC/PR-AUC (Phase 1)
    # Populated when data/validation/{camera_id}/confirmed (positives) and
    # data/baselines/{camera_id}/false_positives (negatives) have ≥10 samples each.
    # All NULL when real labels are unavailable; do not conflate with val_score_*
    # which are computed on the training-time normal holdout only.
    val_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_auroc: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_pr_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_confusion_matrix: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON {tp,fp,fn,tn}
    val_real_sample_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Phase 2: raw per-sample scores/labels so the frontend can re-compute
    # P/R at arbitrary thresholds (slider) without re-running the model.
    # JSON arrays of the same length; None when real-labeled eval was skipped.
    val_scores_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    val_labels_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Output validation (TRN-006)
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    export_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    checkpoint_valid: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    export_valid: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    smoke_test_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    inference_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    status: Mapped[str] = mapped_column(String(20), nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    trained_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "model_type": self.model_type,
            "export_format": self.export_format,
            "baseline_version": self.baseline_version,
            "baseline_count": self.baseline_count,
            "train_count": self.train_count,
            "val_count": self.val_count,
            "pre_validation_passed": self.pre_validation_passed,
            "corruption_rate": self.corruption_rate,
            "near_duplicate_rate": self.near_duplicate_rate,
            "brightness_std": self.brightness_std,
            "val_score_mean": self.val_score_mean,
            "val_score_std": self.val_score_std,
            "val_score_max": self.val_score_max,
            "val_score_p95": self.val_score_p95,
            "quality_grade": self.quality_grade,
            "threshold_recommended": self.threshold_recommended,
            "val_precision": self.val_precision,
            "val_recall": self.val_recall,
            "val_f1": self.val_f1,
            "val_auroc": self.val_auroc,
            "val_pr_auc": self.val_pr_auc,
            "val_confusion_matrix": self.val_confusion_matrix,
            "val_real_sample_count": self.val_real_sample_count,
            # val_scores_json / val_labels_json are intentionally NOT included here —
            # they can be large and are fetched on demand via the metrics endpoint.
            "model_path": self.model_path,
            "export_path": self.export_path,
            "checkpoint_valid": self.checkpoint_valid,
            "export_valid": self.export_valid,
            "smoke_test_passed": self.smoke_test_passed,
            "inference_latency_ms": self.inference_latency_ms,
            "status": self.status,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AuditLog(Base):
    """Audit trail for operator actions."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False, index=True)
    user: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    target_type: Mapped[str] = mapped_column(String(50), nullable=False)
    target_id: Mapped[str] = mapped_column(String(200), nullable=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)


class FeedbackRecord(Base):
    """Operator/system feedback entry for the retraining queue (Section 6).

    Each feedback entry tracks an operator's assessment of an alert
    (confirmed / false_positive / uncertain) or a system-generated
    passive feedback event (drift / health).  Entries live in a
    pending → processed/skipped lifecycle so that engineers can select
    batches for retraining and audit which feedback was incorporated
    into which model version.
    """

    __tablename__ = "feedback_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feedback_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True,
    )
    alert_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True,
        comment="Nullable — passive feedback from drift/health has no alert",
    )
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    zone_id: Mapped[str] = mapped_column(String(50), nullable=False, default="default")

    # Classification
    feedback_type: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True,
        comment="confirmed / false_positive / uncertain",
    )
    category: Mapped[str | None] = mapped_column(
        String(50), nullable=True,
        comment="FP sub-category (lens_glare, insect, shadow, etc.)",
    )

    # Context at feedback time
    model_version_at_time: Mapped[str | None] = mapped_column(String(128), nullable=True)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    snapshot_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Operator info
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_by: Mapped[str] = mapped_column(
        String(100), nullable=False, default="operator",
    )
    source: Mapped[str] = mapped_column(
        String(20), nullable=False, default="manual",
        comment="manual / drift / health",
    )

    # Queue lifecycle
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending", server_default="pending",
        index=True,
    )
    trained_into: Mapped[str | None] = mapped_column(
        String(128), nullable=True,
        comment="model_version_id of the training run that consumed this feedback",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False,
    )
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    def to_dict(self) -> dict:
        return {
            "feedback_id": self.feedback_id,
            "alert_id": self.alert_id,
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "feedback_type": self.feedback_type,
            "category": self.category,
            "model_version_at_time": self.model_version_at_time,
            "anomaly_score": self.anomaly_score,
            "snapshot_path": self.snapshot_path,
            "notes": self.notes,
            "submitted_by": self.submitted_by,
            "source": self.source,
            "status": self.status,
            "trained_into": self.trained_into,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }


class InferenceRecord(Base):
    """Per-frame inference result for audit trail and model comparison (Section 7).

    High-volume table — use InferenceBuffer for batched writes.
    Indexed on (camera_id, timestamp) and (model_version_id) for
    efficient range queries and A/B comparison.
    """

    __tablename__ = "inference_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    zone_id: Mapped[str] = mapped_column(String(50), nullable=False, default="default")
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    # Float epoch (not DateTime) for bulk-insert performance at ~1200 rows/min
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    model_version_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True, index=True,
    )
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    inference_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    was_alert: Mapped[bool] = mapped_column(Boolean, default=False)
    alert_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    deployment_stage: Mapped[str | None] = mapped_column(
        String(20), nullable=True,
        comment="production / shadow — for A/B comparison",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False,
    )


class ModelRecord(Base):
    """Registered model version for MLOps tracking (C4-1)."""

    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(30), nullable=False)
    model_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    data_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    code_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    training_params: Mapped[str | None] = mapped_column(Text, nullable=True)
    calibration_thresholds: Mapped[str | None] = mapped_column(Text, nullable=True)
    backbone_version_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)

    # Release pipeline fields
    stage: Mapped[str] = mapped_column(
        String(20), default="candidate", server_default="candidate", nullable=False,
    )
    component_type: Mapped[str] = mapped_column(
        String(20), default="full", server_default="full", nullable=False,
    )
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    canary_camera_id: Mapped[str | None] = mapped_column(String(50), nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "model_version_id": self.model_version_id,
            "camera_id": self.camera_id,
            "model_type": self.model_type,
            "model_hash": self.model_hash,
            "data_hash": self.data_hash,
            "code_version": self.code_version,
            "training_params": self.training_params,
            "calibration_thresholds": self.calibration_thresholds,
            "backbone_version_id": self.backbone_version_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_active": self.is_active,
            "stage": self.stage,
            "component_type": self.component_type,
            "model_path": self.model_path,
            "canary_camera_id": self.canary_camera_id,
        }


class TrainingJobStatus(str, Enum):
    """Lifecycle status for training jobs (nuclear environment: human confirmation required)."""

    PENDING_CONFIRMATION = "pending_confirmation"
    QUEUED = "queued"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETE = "complete"
    FAILED = "failed"
    REJECTED = "rejected"


class TrainingJobType(str, Enum):
    """Two-level training architecture job types."""

    SSL_BACKBONE = "ssl_backbone"
    ANOMALY_HEAD = "anomaly_head"


class TrainingTriggerType(str, Enum):
    """How a training job was triggered."""

    MANUAL = "manual"
    DRIFT_SUGGESTED = "drift_suggested"
    SCHEDULED = "scheduled"


class TrainingJobRecord(Base):
    """Training job lifecycle with human confirmation gate (nuclear audit requirement).

    Two-level architecture:
    - ssl_backbone: DINOv2 continue-pretraining, shared across cameras
    - anomaly_head: Per-camera Anomalib head training using backbone checkpoint
    """

    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    job_type: Mapped[str] = mapped_column(String(20), nullable=False)  # ssl_backbone / anomaly_head
    camera_id: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    zone_id: Mapped[str] = mapped_column(String(50), nullable=False, default="default")
    model_type: Mapped[str | None] = mapped_column(String(30), nullable=True)

    # Trigger & confirmation
    trigger_type: Mapped[str] = mapped_column(String(20), nullable=False)  # manual / drift_suggested / scheduled
    triggered_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    confirmation_required: Mapped[bool] = mapped_column(Boolean, default=True)
    confirmed_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(
        String(25), nullable=False, default="pending_confirmation", index=True
    )

    # Model lineage
    base_model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    dataset_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    hyperparameters: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON

    # Results
    metrics: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    artifacts_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    validation_report: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    model_version_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Timing
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "job_type": self.job_type,
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "model_type": self.model_type,
            "trigger_type": self.trigger_type,
            "triggered_by": self.triggered_by,
            "confirmation_required": self.confirmation_required,
            "confirmed_by": self.confirmed_by,
            "confirmed_at": self.confirmed_at.isoformat() if self.confirmed_at else None,
            "status": self.status,
            "base_model_version": self.base_model_version,
            "dataset_version": self.dataset_version,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "artifacts_path": self.artifacts_path,
            "validation_report": self.validation_report,
            "model_version_id": self.model_version_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


class BackboneRecord(Base):
    """Shared SSL backbone version (Level 1 of two-level training).

    DINOv2 continue-pretraining checkpoint shared across all cameras.
    Version evolves independently from per-camera anomaly heads.
    """

    __tablename__ = "backbones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    backbone_version_id: Mapped[str] = mapped_column(
        String(128), unique=True, nullable=False, index=True
    )
    backbone_type: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g. dinov2_vitb14
    checkpoint_path: Mapped[str] = mapped_column(String(500), nullable=False)
    checkpoint_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    camera_ids_used: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list
    training_job_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "backbone_version_id": self.backbone_version_id,
            "backbone_type": self.backbone_type,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_hash": self.checkpoint_hash,
            "dataset_hash": self.dataset_hash,
            "camera_ids_used": self.camera_ids_used,
            "training_job_id": self.training_job_id,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class User(Base):
    """System user with role-based access."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="viewer")  # admin, operator, viewer
    display_name: Mapped[str] = mapped_column(String(100), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    last_login: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Region(Base):
    """Managed region/contact entry for alert notifications."""

    __tablename__ = "regions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    owner: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    email: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True, default=None)
    notification_methods: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    notification_template_ids: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class NotificationTemplate(Base):
    """Reusable alert notification template stored as operational data."""

    __tablename__ = "notification_templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    method: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    subject: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class ModelVersionEvent(Base):
    """Structured version transition event for audit trail."""

    __tablename__ = "model_version_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False, index=True,
    )
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    from_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    to_version: Mapped[str] = mapped_column(String(128), nullable=False)
    from_stage: Mapped[str | None] = mapped_column(String(20), nullable=True)
    to_stage: Mapped[str] = mapped_column(String(20), nullable=False)
    triggered_by: Mapped[str] = mapped_column(String(100), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    warmup_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    sha256_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "camera_id": self.camera_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "triggered_by": self.triggered_by,
            "reason": self.reason,
            "warmup_latency_ms": self.warmup_latency_ms,
            "sha256_verified": self.sha256_verified,
        }


class AlertRecordingRecord(Base):
    """Tracks solidified alert recordings on disk (FR-033).

    Each alert has an associated H.264 MP4 recording with signal
    timeseries for replay in the multi-track timeline.
    """

    __tablename__ = "alert_recordings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True,
    )
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(10), nullable=False)
    recording_path: Mapped[str] = mapped_column(String(500), nullable=False)
    start_timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    end_timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    trigger_timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    frame_count: Mapped[int] = mapped_column(Integer, nullable=False)
    fps: Mapped[int] = mapped_column(Integer, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    linked_alert_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="complete",
    )
    video_codec: Mapped[str] = mapped_column(
        String(10), nullable=False, default="h264",
    )
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False,
    )

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "camera_id": self.camera_id,
            "severity": self.severity,
            "recording_path": self.recording_path,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "trigger_timestamp": self.trigger_timestamp,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "file_size_bytes": self.file_size_bytes,
            "linked_alert_id": self.linked_alert_id,
            "status": self.status,
            "video_codec": self.video_codec,
            "width": self.width,
            "height": self.height,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class LabelingQueueStatus(str, Enum):
    """Status of a labeling queue entry."""

    PENDING = "pending"
    LABELED = "labeled"
    SKIPPED = "skipped"


class LabelingQueueRecord(Base):
    """Uncertain frame queued for operator labeling (active learning).

    Frames with high prediction uncertainty are pushed here by the
    ActiveLearningSampler. Operators label them as normal/anomaly,
    and labeled frames feed back into incremental retraining.
    """

    __tablename__ = "labeling_queue"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    zone_id: Mapped[str] = mapped_column(String(50), nullable=False, default="default")
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    frame_path: Mapped[str] = mapped_column(String(500), nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    entropy: Mapped[float] = mapped_column(Float, nullable=False)
    model_version_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Labeling
    label: Mapped[str | None] = mapped_column(
        String(20), nullable=True,
        comment="normal / anomaly — set by operator",
    )
    labeled_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    labeled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Queue lifecycle
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending", server_default="pending",
        index=True,
    )
    trained_into: Mapped[str | None] = mapped_column(
        String(128), nullable=True,
        comment="model_version_id of the training run that consumed this label",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False,
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "frame_number": self.frame_number,
            "frame_path": self.frame_path,
            "anomaly_score": self.anomaly_score,
            "entropy": self.entropy,
            "model_version_id": self.model_version_id,
            "label": self.label,
            "labeled_by": self.labeled_by,
            "labeled_at": self.labeled_at.isoformat() if self.labeled_at else None,
            "status": self.status,
            "trained_into": self.trained_into,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ShadowInferenceLog(Base):
    """Shadow model parallel inference results for comparison."""

    __tablename__ = "shadow_inference_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False,
    )
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    shadow_version_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    production_version_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    shadow_score: Mapped[float] = mapped_column(Float, nullable=False)
    production_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    shadow_would_alert: Mapped[bool] = mapped_column(Boolean, nullable=False)
    production_alerted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)


class ContinuousRecordingSegment(Base):
    """A single continuous recording video segment."""

    __tablename__ = "continuous_recording_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    segment_path: Mapped[str] = mapped_column(String(512), nullable=False)
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    end_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frame_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    archived: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    archive_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False,
    )
