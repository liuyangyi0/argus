"""Database initialization and session management."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import structlog
from sqlalchemy import create_engine, func as sa_func, select, text
from sqlalchemy.orm import Session, sessionmaker

from argus.storage.models import (
    AlertRecord,
    AlertWorkflowStatus,
    Base,
    BaselineRecord,
    BaselineVersionRecord,
    TrainingRecord,
    User,
)

logger = structlog.get_logger()

_USER_UPDATABLE_FIELDS = {"display_name", "role", "password_hash", "is_active"}


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

        # Auto-migrate: add missing columns to existing tables
        self._auto_migrate()

        logger.info("database.initialized", url=self._database_url)

    def _auto_migrate(self) -> None:
        """Add missing columns to existing SQLite tables (lightweight migration)."""
        migrations = [
            ("alerts", "workflow_status", "VARCHAR(20) DEFAULT 'new'"),
            ("alerts", "assigned_to", "VARCHAR(100)"),
            ("alerts", "resolved_at", "DATETIME"),
            ("training_records", "group_id", "VARCHAR(100)"),
        ]
        with self._engine.connect() as conn:
            for table, column, col_type in migrations:
                try:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
                    conn.commit()
                    logger.info("database.migration", table=table, column=column)
                except Exception:
                    pass  # Column already exists

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

    def update_alert_workflow(
        self,
        alert_id: str,
        workflow_status: str,
        notes: str | None = None,
        assigned_to: str | None = None,
    ) -> bool:
        """Transition alert workflow status with optional notes/assignment."""
        valid_statuses = {s.value for s in AlertWorkflowStatus}
        if workflow_status not in valid_statuses:
            return False
        with self.get_session() as session:
            record = session.scalar(
                select(AlertRecord).where(AlertRecord.alert_id == alert_id)
            )
            if record is None:
                return False
            record.workflow_status = workflow_status
            if notes:
                record.notes = notes
            if assigned_to:
                record.assigned_to = assigned_to
            if workflow_status in ("resolved", "closed"):
                record.resolved_at = datetime.now(timezone.utc)
            if workflow_status == "false_positive":
                record.false_positive = True
            if workflow_status == "acknowledged":
                record.acknowledged = True
            session.commit()
            return True

    def get_alert_workflow_stats(self) -> dict[str, int]:
        """Return count of alerts grouped by workflow_status."""
        stats = {s.value: 0 for s in AlertWorkflowStatus}
        with self.get_session() as session:
            rows = session.execute(
                select(AlertRecord.workflow_status, sa_func.count())
                .group_by(AlertRecord.workflow_status)
            ).all()
            for status, count in rows:
                stats[status] = count
        return stats

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
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
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

    # ── Training records ──

    def save_training_record(self, record: TrainingRecord) -> TrainingRecord:
        """Save a training record to the database."""
        with self.get_session() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            logger.debug("database.training_record_saved", camera_id=record.camera_id)
            return record

    def get_training_history(
        self,
        camera_id: str | None = None,
        zone_id: str | None = None,
        limit: int = 20,
    ) -> list[TrainingRecord]:
        """Query training records with optional filters."""
        with self.get_session() as session:
            stmt = select(TrainingRecord).order_by(TrainingRecord.trained_at.desc())
            if camera_id:
                stmt = stmt.where(TrainingRecord.camera_id == camera_id)
            if zone_id:
                stmt = stmt.where(TrainingRecord.zone_id == zone_id)
            stmt = stmt.limit(limit)
            return list(session.scalars(stmt).all())

    def get_latest_training(
        self, camera_id: str, zone_id: str = "default"
    ) -> TrainingRecord | None:
        """Get the most recent successful training record for a camera/zone."""
        with self.get_session() as session:
            return session.scalar(
                select(TrainingRecord)
                .where(TrainingRecord.camera_id == camera_id)
                .where(TrainingRecord.zone_id == zone_id)
                .where(TrainingRecord.status == "complete")
                .order_by(TrainingRecord.trained_at.desc())
                .limit(1)
            )

    def get_training_record(self, record_id: int) -> TrainingRecord | None:
        """Get a single training record by ID."""
        with self.get_session() as session:
            return session.scalar(
                select(TrainingRecord).where(TrainingRecord.id == record_id)
            )

    # ── User management ──

    def create_user(
        self,
        username: str,
        password_hash: str,
        role: str = "viewer",
        display_name: str | None = None,
    ) -> User:
        """Create a new user record."""
        with self.get_session() as session:
            user = User(
                username=username,
                password_hash=password_hash,
                role=role,
                display_name=display_name,
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            logger.info("database.user_created", username=username, role=role)
            return user

    def get_user(self, username: str) -> User | None:
        """Get a user by username."""
        with self.get_session() as session:
            return session.scalar(select(User).where(User.username == username))

    def get_all_users(self) -> list[User]:
        """Return all users ordered by username."""
        with self.get_session() as session:
            return list(session.scalars(select(User).order_by(User.username)).all())

    def update_user(self, username: str, **kwargs) -> bool:
        """Update user fields. Returns True if the user was found and updated."""
        with self.get_session() as session:
            user = session.scalar(select(User).where(User.username == username))
            if user is None:
                return False
            for key, value in kwargs.items():
                if key in _USER_UPDATABLE_FIELDS and hasattr(user, key):
                    setattr(user, key, value)
            session.commit()
            return True

    def delete_user(self, username: str) -> bool:
        """Delete a user. Returns True if deleted."""
        with self.get_session() as session:
            user = session.scalar(select(User).where(User.username == username))
            if user is None:
                return False
            session.delete(user)
            session.commit()
            logger.info("database.user_deleted", username=username)
            return True

    def user_count(self) -> int:
        """Return total number of users."""
        with self.get_session() as session:
            return len(list(session.scalars(select(User)).all()))

    def close(self) -> None:
        """Close the database engine."""
        if self._engine:
            self._engine.dispose()
