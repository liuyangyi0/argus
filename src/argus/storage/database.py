"""Database initialization and session management."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import structlog
from sqlalchemy import create_engine, func as sa_func, select, text
from sqlalchemy.orm import Session, sessionmaker

from argus.storage.models import AlertRecord, AlertWorkflowStatus, Base, BaselineRecord, TrainingRecord

logger = structlog.get_logger()


class Database:
    """SQLite database manager for alert persistence.

    Handles schema creation, session management, and provides
    convenience methods for alert CRUD operations.
    """

    def __init__(self, database_url: str = "sqlite:///data/db/argus.db"):
        self._database_url = database_url
        self._engine = None
        self._session_factory = None

    def initialize(self) -> None:
        """Create the database and tables if they don't exist."""
        # Ensure the directory exists for SQLite file databases
        if self._database_url.startswith("sqlite:///"):
            db_path = self._database_url.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._engine = create_engine(self._database_url, echo=False)
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)

        # Enable WAL mode for better concurrent read/write performance
        if self._database_url.startswith("sqlite"):
            with self._engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.commit()

        logger.info("database.initialized", url=self._database_url)

    def get_session(self) -> Session:
        """Get a new database session."""
        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._session_factory()

    def save_alert(
        self,
        alert_id: str,
        timestamp: datetime,
        camera_id: str,
        zone_id: str,
        severity: str,
        anomaly_score: float,
        snapshot_path: str | None = None,
        heatmap_path: str | None = None,
        _max_retries: int = 3,
    ) -> AlertRecord:
        """Save an alert to the database with retry on transient failures.

        Uses a threading.Event for non-blocking retry delays so the calling
        thread can be interrupted during shutdown.
        """
        import threading

        last_error = None
        retry_event = threading.Event()
        for attempt in range(_max_retries):
            try:
                with self.get_session() as session:
                    record = AlertRecord(
                        alert_id=alert_id,
                        timestamp=timestamp,
                        camera_id=camera_id,
                        zone_id=zone_id,
                        severity=severity,
                        anomaly_score=anomaly_score,
                        snapshot_path=snapshot_path,
                        heatmap_path=heatmap_path,
                    )
                    session.add(record)
                    session.commit()
                    session.refresh(record)
                    logger.debug("database.alert_saved", alert_id=alert_id)
                    return record
            except Exception as e:
                last_error = e
                if attempt < _max_retries - 1:
                    logger.warning(
                        "database.save_retry",
                        alert_id=alert_id,
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    retry_event.wait(timeout=0.5)
        raise last_error  # type: ignore[misc]

    def get_alerts(
        self,
        camera_id: str | None = None,
        severity: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AlertRecord]:
        """Query alerts with optional filters."""
        with self.get_session() as session:
            stmt = select(AlertRecord).order_by(AlertRecord.timestamp.desc())

            if camera_id:
                stmt = stmt.where(AlertRecord.camera_id == camera_id)
            if severity:
                stmt = stmt.where(AlertRecord.severity == severity)

            stmt = stmt.offset(offset).limit(limit)
            return list(session.scalars(stmt).all())

    def get_alert(self, alert_id: str) -> AlertRecord | None:
        """Get a single alert by alert_id."""
        with self.get_session() as session:
            return session.scalar(
                select(AlertRecord).where(AlertRecord.alert_id == alert_id)
            )

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Mark an alert as acknowledged."""
        with self.get_session() as session:
            record = session.scalar(
                select(AlertRecord).where(AlertRecord.alert_id == alert_id)
            )
            if record is None:
                return False
            record.acknowledged = True
            record.acknowledged_by = acknowledged_by
            session.commit()
            return True

    def mark_false_positive(self, alert_id: str, notes: str | None = None) -> bool:
        """Mark an alert as a false positive (used for retraining feedback)."""
        with self.get_session() as session:
            record = session.scalar(
                select(AlertRecord).where(AlertRecord.alert_id == alert_id)
            )
            if record is None:
                return False
            record.false_positive = True
            if notes:
                record.notes = notes
            session.commit()
            return True

    def get_alert_count(
        self,
        camera_id: str | None = None,
        severity: str | None = None,
    ) -> int:
        """Get total alert count with optional filters."""
        with self.get_session() as session:
            stmt = select(sa_func.count()).select_from(AlertRecord)
            if camera_id:
                stmt = stmt.where(AlertRecord.camera_id == camera_id)
            if severity:
                stmt = stmt.where(AlertRecord.severity == severity)
            return session.scalar(stmt) or 0

    def delete_old_alerts(self, days: int = 90) -> tuple[int, list[str]]:
        """Delete alerts older than N days. Returns (count_deleted, image_paths).

        Image paths are returned so the caller can delete files from disk.
        """
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=days)
        with self.get_session() as session:
            old_alerts = list(
                session.scalars(
                    select(AlertRecord).where(AlertRecord.timestamp < cutoff)
                ).all()
            )

            if not old_alerts:
                return 0, []

            # Collect image paths before deleting records
            image_paths = []
            for alert in old_alerts:
                if alert.snapshot_path:
                    image_paths.append(alert.snapshot_path)
                if alert.heatmap_path:
                    image_paths.append(alert.heatmap_path)

            for alert in old_alerts:
                session.delete(alert)
            session.commit()

            logger.info(
                "database.alerts_cleaned",
                deleted=len(old_alerts),
                cutoff_days=days,
                images=len(image_paths),
            )
            return len(old_alerts), image_paths

    # ── Training history (TRN-007) ──

    def save_training_record(self, **kwargs) -> None:
        """Save a training history record."""
        with self.get_session() as session:
            record = TrainingRecord(**kwargs)
            session.add(record)
            session.commit()
            logger.debug("database.training_record_saved", training_id=kwargs.get("training_id"))

    def get_training_history(
        self, camera_id: str | None = None, limit: int = 20
    ) -> list[dict]:
        """Get training history, optionally filtered by camera."""
        with self.get_session() as session:
            stmt = select(TrainingRecord).order_by(TrainingRecord.started_at.desc())
            if camera_id:
                stmt = stmt.where(TrainingRecord.camera_id == camera_id)
            stmt = stmt.limit(limit)
            records = list(session.scalars(stmt).all())
            return [r.to_dict() for r in records]

    def get_training_record(self, training_id: str) -> dict | None:
        """Get a single training record by ID."""
        with self.get_session() as session:
            record = session.scalar(
                select(TrainingRecord).where(TrainingRecord.training_id == training_id)
            )
            return record.to_dict() if record else None

    # ── Alert workflow (YOLO-005) ──

    def update_alert_workflow(
        self, alert_id: str, status: str,
        assigned_to: str | None = None, notes: str | None = None,
    ) -> bool:
        """Update alert workflow status."""
        try:
            AlertWorkflowStatus(status)
        except ValueError:
            return False

        with self.get_session() as session:
            record = session.scalar(
                select(AlertRecord).where(AlertRecord.alert_id == alert_id)
            )
            if record is None:
                return False

            record.workflow_status = status
            if assigned_to is not None:
                record.assigned_to = assigned_to
            if notes is not None:
                record.notes = notes
            if status == AlertWorkflowStatus.RESOLVED.value:
                record.resolved_at = datetime.utcnow()
            if status == AlertWorkflowStatus.ACKNOWLEDGED.value:
                record.acknowledged = True
            if status == AlertWorkflowStatus.FALSE_POSITIVE.value:
                record.false_positive = True

            session.commit()
            return True

    def get_alert_workflow_stats(self) -> dict[str, int]:
        """Get counts per workflow status for dashboard display."""
        with self.get_session() as session:
            alerts = list(session.scalars(select(AlertRecord)).all())
            stats: dict[str, int] = {ws.value: 0 for ws in AlertWorkflowStatus}
            for alert in alerts:
                status = alert.workflow_status or "new"
                stats[status] = stats.get(status, 0) + 1
            return stats

    def close(self) -> None:
        """Close the database engine."""
        if self._engine:
            self._engine.dispose()
