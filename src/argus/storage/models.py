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
    """Records each model training run for history and comparison."""

    __tablename__ = "training_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    training_id: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    zone_id: Mapped[str] = mapped_column(String(50), default="default")
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    image_count: Mapped[int] = mapped_column(Integer, default=0)
    train_count: Mapped[int] = mapped_column(Integer, default=0)
    val_count: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    quality_grade: Mapped[str | None] = mapped_column(String(1), nullable=True)
    score_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_p95: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    recommended_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    export_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "training_id": self.training_id,
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "model_type": self.model_type,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "image_count": self.image_count,
            "train_count": self.train_count,
            "val_count": self.val_count,
            "duration_seconds": self.duration_seconds,
            "quality_grade": self.quality_grade,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "score_max": self.score_max,
            "score_p95": self.score_p95,
            "threshold": self.threshold,
            "recommended_threshold": self.recommended_threshold,
            "model_path": self.model_path,
            "export_path": self.export_path,
            "error_message": self.error_message,
            "notes": self.notes,
        }
