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
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def to_dict(self) -> dict:
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
