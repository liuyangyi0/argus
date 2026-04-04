"""SQLAlchemy ORM models for alert persistence."""

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
