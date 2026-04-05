"""SQLAlchemy ORM models for alert and training persistence."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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
