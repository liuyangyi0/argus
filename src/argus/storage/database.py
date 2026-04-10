"""Database initialization and session management."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import structlog
from sqlalchemy import create_engine, func as sa_func, select, text
from sqlalchemy.orm import Session, sessionmaker

from argus.storage.models import (
    AlertRecord,
    AlertRecordingRecord,
    AlertWorkflowStatus,
    BackboneRecord,
    Base,
    BaselineRecord,
    BaselineVersionRecord,
    FeedbackRecord,
    FeedbackStatus,
    InferenceRecord,
    LabelingQueueRecord,
    LabelingQueueStatus,
    TrainingJobRecord,
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
            ("models", "backbone_version_id", "VARCHAR(128)"),
            # Model release pipeline
            ("models", "stage", "VARCHAR(20) DEFAULT 'production'"),
            ("models", "component_type", "VARCHAR(20) DEFAULT 'full'"),
            ("models", "model_path", "VARCHAR(500)"),
            ("models", "canary_camera_id", "VARCHAR(50)"),
            # Alert recording MP4 migration (d47d70f)
            ("alert_recordings", "video_codec", "VARCHAR(10) DEFAULT 'h264'"),
            ("alert_recordings", "width", "INTEGER"),
            ("alert_recordings", "height", "INTEGER"),
            # Alert aggregation
            ("alerts", "event_group_id", "VARCHAR(64)"),
            ("alerts", "event_group_count", "INTEGER DEFAULT 1"),
        ]
        with self._engine.connect() as conn:
            for table, column, col_type in migrations:
                try:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
                    conn.commit()
                    logger.info("database.migration", table=table, column=column)
                except Exception:
                    logger.debug("database.migration_column_exists", table=table, column=column)

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
        event_group_id: str | None = None,
        event_group_count: int = 1,
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
                        event_group_id=event_group_id,
                        event_group_count=event_group_count,
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

    def delete_alert(self, alert_id: str) -> tuple[bool, list[str]]:
        """Delete a single alert by ID. Returns (success, image_paths)."""
        with self.get_session() as session:
            record = session.scalar(
                select(AlertRecord).where(AlertRecord.alert_id == alert_id)
            )
            if record is None:
                return False, []
            image_paths = []
            if record.snapshot_path:
                image_paths.append(record.snapshot_path)
            if record.heatmap_path:
                image_paths.append(record.heatmap_path)
            session.delete(record)
            session.commit()
            logger.info("database.alert_deleted", alert_id=alert_id)
            return True, image_paths

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

    # ── Training jobs ──

    def save_training_job(self, record: TrainingJobRecord) -> TrainingJobRecord:
        """Save a training job record."""
        with self.get_session() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            logger.debug("database.training_job_saved", job_id=record.job_id)
            return record

    def get_training_job(self, job_id: str) -> TrainingJobRecord | None:
        """Get a training job by job_id."""
        with self.get_session() as session:
            return session.scalar(
                select(TrainingJobRecord).where(TrainingJobRecord.job_id == job_id)
            )

    def list_training_jobs(
        self,
        status: str | None = None,
        job_type: str | None = None,
        camera_id: str | None = None,
        limit: int = 50,
    ) -> list[TrainingJobRecord]:
        """List training jobs with optional filters."""
        with self.get_session() as session:
            stmt = select(TrainingJobRecord).order_by(TrainingJobRecord.created_at.desc())
            if status:
                stmt = stmt.where(TrainingJobRecord.status == status)
            if job_type:
                stmt = stmt.where(TrainingJobRecord.job_type == job_type)
            if camera_id:
                stmt = stmt.where(TrainingJobRecord.camera_id == camera_id)
            stmt = stmt.limit(limit)
            return list(session.scalars(stmt).all())

    def update_training_job(self, job_id: str, **kwargs) -> bool:
        """Update training job fields. Returns True if found and updated."""
        with self.get_session() as session:
            record = session.scalar(
                select(TrainingJobRecord).where(TrainingJobRecord.job_id == job_id)
            )
            if record is None:
                return False
            for key, value in kwargs.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            session.commit()
            return True

    def count_pending_jobs(self) -> int:
        """Count training jobs pending confirmation."""
        with self.get_session() as session:
            return session.scalar(
                select(sa_func.count()).select_from(TrainingJobRecord)
                .where(TrainingJobRecord.status == "pending_confirmation")
            ) or 0

    # ── Backbones ──

    def save_backbone(self, record: BackboneRecord) -> BackboneRecord:
        """Save a backbone record."""
        with self.get_session() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            logger.debug("database.backbone_saved", version_id=record.backbone_version_id)
            return record

    def get_active_backbone(self) -> BackboneRecord | None:
        """Get the currently active backbone."""
        with self.get_session() as session:
            return session.scalar(
                select(BackboneRecord).where(BackboneRecord.is_active == True)
            )

    def list_backbones(self, limit: int = 20) -> list[BackboneRecord]:
        """List all backbone records."""
        with self.get_session() as session:
            return list(session.scalars(
                select(BackboneRecord).order_by(BackboneRecord.created_at.desc()).limit(limit)
            ).all())

    def activate_backbone(self, backbone_version_id: str) -> bool:
        """Set a backbone as active (deactivates others)."""
        with self.get_session() as session:
            record = session.scalar(
                select(BackboneRecord).where(
                    BackboneRecord.backbone_version_id == backbone_version_id
                )
            )
            if record is None:
                return False
            session.query(BackboneRecord).filter(
                BackboneRecord.is_active == True
            ).update({"is_active": False})
            record.is_active = True
            session.commit()
            return True

    # ── Inference records ──

    def save_inference_batch(self, records: list[InferenceRecord]) -> int:
        """Bulk-insert inference records. Returns count inserted."""
        if not records:
            return 0
        with self.get_session() as session:
            session.add_all(records)
            session.commit()
        return len(records)

    def get_inference_stats(
        self,
        camera_id: str,
        model_version_id: str | None = None,
    ) -> dict:
        """Return score distribution stats for a camera/model."""
        with self.get_session() as session:
            stmt = select(
                sa_func.count(),
                sa_func.avg(InferenceRecord.anomaly_score),
                sa_func.max(InferenceRecord.anomaly_score),
                sa_func.avg(InferenceRecord.inference_latency_ms),
            ).where(InferenceRecord.camera_id == camera_id)
            if model_version_id:
                stmt = stmt.where(InferenceRecord.model_version_id == model_version_id)
            row = session.execute(stmt).one()
            return {
                "total_frames": row[0] or 0,
                "avg_score": round(row[1], 4) if row[1] else 0.0,
                "max_score": round(row[2], 4) if row[2] else 0.0,
                "avg_latency_ms": round(row[3], 2) if row[3] else 0.0,
            }

    def delete_old_inference_records(self, days: int = 30) -> int:
        """Delete inference records older than N days. Returns count deleted."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self.get_session() as session:
            # InferenceRecord.timestamp is a float (epoch seconds)
            cutoff_ts = cutoff.timestamp()
            count = (
                session.query(InferenceRecord)
                .filter(InferenceRecord.timestamp < cutoff_ts)
                .delete()
            )
            session.commit()
        if count:
            logger.info("database.inference_cleaned", deleted=count, cutoff_days=days)
        return count

    # ── Feedback entries ──

    def save_feedback(self, record: FeedbackRecord) -> FeedbackRecord:
        """Save a feedback entry to the database."""
        with self.get_session() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            logger.debug(
                "database.feedback_saved",
                feedback_id=record.feedback_id,
                feedback_type=record.feedback_type,
            )
            return record

    def get_feedback(self, feedback_id: str) -> FeedbackRecord | None:
        """Get a single feedback entry by feedback_id."""
        with self.get_session() as session:
            return session.scalar(
                select(FeedbackRecord).where(FeedbackRecord.feedback_id == feedback_id)
            )

    def get_pending_feedback(
        self,
        camera_id: str | None = None,
        feedback_type: str | None = None,
        limit: int = 500,
    ) -> list[FeedbackRecord]:
        """Get feedback entries with status='pending', optionally filtered."""
        with self.get_session() as session:
            stmt = (
                select(FeedbackRecord)
                .where(FeedbackRecord.status == FeedbackStatus.PENDING)
                .order_by(FeedbackRecord.created_at.asc())
            )
            if camera_id:
                stmt = stmt.where(FeedbackRecord.camera_id == camera_id)
            if feedback_type:
                stmt = stmt.where(FeedbackRecord.feedback_type == feedback_type)
            stmt = stmt.limit(limit)
            return list(session.scalars(stmt).all())

    def mark_feedback_processed(
        self,
        feedback_ids: list[str],
        trained_into: str,
    ) -> int:
        """Bulk-mark feedback entries as processed with the model version they fed into.

        Returns the number of records updated.
        """
        if not feedback_ids:
            return 0
        now = datetime.now(timezone.utc)
        updated = 0
        with self.get_session() as session:
            for fid in feedback_ids:
                record = session.scalar(
                    select(FeedbackRecord).where(FeedbackRecord.feedback_id == fid)
                )
                if record and record.status == FeedbackStatus.PENDING:
                    record.status = FeedbackStatus.PROCESSED
                    record.trained_into = trained_into
                    record.processed_at = now
                    updated += 1
            session.commit()
        logger.info(
            "database.feedback_batch_processed",
            count=updated,
            trained_into=trained_into,
        )
        return updated

    def skip_feedback(self, feedback_ids: list[str], reason: str | None = None) -> int:
        """Mark feedback entries as skipped (will not be used for training).

        Returns the number of records updated.
        """
        if not feedback_ids:
            return 0
        updated = 0
        with self.get_session() as session:
            for fid in feedback_ids:
                record = session.scalar(
                    select(FeedbackRecord).where(FeedbackRecord.feedback_id == fid)
                )
                if record and record.status == FeedbackStatus.PENDING:
                    record.status = FeedbackStatus.SKIPPED
                    if reason:
                        record.notes = (record.notes or "") + f" [skip: {reason}]"
                    updated += 1
            session.commit()
        return updated

    def get_feedback_summary(self, camera_id: str | None = None) -> dict:
        """Return feedback counts grouped by type and status."""
        with self.get_session() as session:
            stmt = select(
                FeedbackRecord.feedback_type,
                FeedbackRecord.status,
                sa_func.count(),
            ).group_by(FeedbackRecord.feedback_type, FeedbackRecord.status)
            if camera_id:
                stmt = stmt.where(FeedbackRecord.camera_id == camera_id)
            rows = session.execute(stmt).all()

        summary: dict = {"total": 0, "by_type": {}, "by_status": {}}
        for fb_type, status, count in rows:
            summary["total"] += count
            summary["by_type"].setdefault(fb_type, 0)
            summary["by_type"][fb_type] += count
            summary["by_status"].setdefault(status, 0)
            summary["by_status"][status] += count
        return summary

    def list_feedback(
        self,
        camera_id: str | None = None,
        feedback_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FeedbackRecord]:
        """List feedback entries with optional filters."""
        with self.get_session() as session:
            stmt = select(FeedbackRecord).order_by(FeedbackRecord.created_at.desc())
            if camera_id:
                stmt = stmt.where(FeedbackRecord.camera_id == camera_id)
            if feedback_type:
                stmt = stmt.where(FeedbackRecord.feedback_type == feedback_type)
            if status:
                stmt = stmt.where(FeedbackRecord.status == status)
            stmt = stmt.offset(offset).limit(limit)
            return list(session.scalars(stmt).all())

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

    # ── Alert recordings ──

    def save_alert_recording(self, record: AlertRecordingRecord) -> AlertRecordingRecord:
        """Save an alert recording metadata record."""
        with self.get_session() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            logger.debug(
                "database.alert_recording_saved",
                alert_id=record.alert_id,
            )
            return record

    def get_alert_recording(self, alert_id: str) -> AlertRecordingRecord | None:
        """Get recording metadata for an alert."""
        with self.get_session() as session:
            return session.scalar(
                select(AlertRecordingRecord).where(
                    AlertRecordingRecord.alert_id == alert_id
                )
            )

    def get_alert_recordings_batch(self, alert_ids: list[str]) -> dict[str, AlertRecordingRecord]:
        """Get recording records for multiple alerts in one query.

        Returns dict mapping alert_id -> AlertRecordingRecord.
        """
        if not alert_ids:
            return {}
        with self.get_session() as session:
            records = session.scalars(
                select(AlertRecordingRecord).where(
                    AlertRecordingRecord.alert_id.in_(alert_ids)
                )
            ).all()
            return {r.alert_id: r for r in records}

    def update_alert_recording_status(
        self,
        alert_id: str,
        status: str,
        frame_count: int | None = None,
        end_timestamp: float | None = None,
        file_size_bytes: int | None = None,
    ) -> bool:
        """Update the status (and optional metadata) of an alert recording."""
        with self.get_session() as session:
            record = session.scalar(
                select(AlertRecordingRecord).where(
                    AlertRecordingRecord.alert_id == alert_id
                )
            )
            if record is None:
                return False
            record.status = status
            if frame_count is not None:
                record.frame_count = frame_count
            if end_timestamp is not None:
                record.end_timestamp = end_timestamp
            if file_size_bytes is not None:
                record.file_size_bytes = file_size_bytes
            session.commit()
            return True

    def cleanup_old_recordings(self, max_age_days: int = 30) -> int:
        """Delete recording records older than max_age_days. Returns count."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        with self.get_session() as session:
            count = (
                session.query(AlertRecordingRecord)
                .filter(AlertRecordingRecord.created_at < cutoff)
                .delete()
            )
            session.commit()
        if count:
            logger.info("database.recordings_cleaned", deleted=count, cutoff_days=max_age_days)
        return count

    # ── Labeling queue (active learning) ──

    def save_labeling_entry(self, record: LabelingQueueRecord) -> LabelingQueueRecord:
        """Save an uncertain frame to the labeling queue."""
        with self.get_session() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            logger.debug(
                "database.labeling_entry_saved",
                camera_id=record.camera_id,
                frame_number=record.frame_number,
            )
            return record

    def get_labeling_queue(
        self,
        camera_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LabelingQueueRecord]:
        """Get labeling queue entries with optional filters."""
        with self.get_session() as session:
            stmt = select(LabelingQueueRecord).order_by(
                LabelingQueueRecord.entropy.desc(),
                LabelingQueueRecord.created_at.desc(),
            )
            if camera_id:
                stmt = stmt.where(LabelingQueueRecord.camera_id == camera_id)
            if status:
                stmt = stmt.where(LabelingQueueRecord.status == status)
            else:
                stmt = stmt.where(
                    LabelingQueueRecord.status == LabelingQueueStatus.PENDING
                )
            stmt = stmt.offset(offset).limit(limit)
            return list(session.scalars(stmt).all())

    def label_entry(
        self,
        entry_id: int,
        label: str,
        labeled_by: str,
    ) -> LabelingQueueRecord | None:
        """Label an entry in the labeling queue. Returns updated record or None."""
        with self.get_session() as session:
            record = session.scalar(
                select(LabelingQueueRecord).where(LabelingQueueRecord.id == entry_id)
            )
            if record is None:
                return None
            record.label = label
            record.labeled_by = labeled_by
            record.labeled_at = datetime.now(timezone.utc)
            record.status = LabelingQueueStatus.LABELED
            session.commit()
            session.refresh(record)
            return record

    def get_labeling_stats(self, camera_id: str | None = None) -> dict:
        """Return labeling queue statistics."""
        with self.get_session() as session:
            base = select(
                LabelingQueueRecord.status,
                sa_func.count(),
            ).group_by(LabelingQueueRecord.status)
            if camera_id:
                base = base.where(LabelingQueueRecord.camera_id == camera_id)
            rows = session.execute(base).all()

        stats: dict = {"total": 0, "pending": 0, "labeled": 0, "skipped": 0}
        for status_val, count in rows:
            stats["total"] += count
            stats[status_val] = count

        # Count labels by type
        with self.get_session() as session:
            label_stmt = (
                select(LabelingQueueRecord.label, sa_func.count())
                .where(LabelingQueueRecord.status == LabelingQueueStatus.LABELED)
                .group_by(LabelingQueueRecord.label)
            )
            if camera_id:
                label_stmt = label_stmt.where(LabelingQueueRecord.camera_id == camera_id)
            label_rows = session.execute(label_stmt).all()

        stats["by_label"] = {label: count for label, count in label_rows if label}
        return stats

    def get_labeled_entries(
        self,
        camera_id: str | None = None,
        label: str | None = None,
        trained_into: str | None = None,
        limit: int = 500,
    ) -> list[LabelingQueueRecord]:
        """Get labeled entries, optionally filtered. Used for incremental training."""
        with self.get_session() as session:
            stmt = (
                select(LabelingQueueRecord)
                .where(LabelingQueueRecord.status == LabelingQueueStatus.LABELED)
                .order_by(LabelingQueueRecord.labeled_at.asc())
            )
            if camera_id:
                stmt = stmt.where(LabelingQueueRecord.camera_id == camera_id)
            if label:
                stmt = stmt.where(LabelingQueueRecord.label == label)
            if trained_into is not None:
                if trained_into == "":
                    stmt = stmt.where(LabelingQueueRecord.trained_into.is_(None))
                else:
                    stmt = stmt.where(LabelingQueueRecord.trained_into == trained_into)
            stmt = stmt.limit(limit)
            return list(session.scalars(stmt).all())

    def mark_labeling_trained(
        self,
        entry_ids: list[int],
        trained_into: str,
    ) -> int:
        """Mark labeled entries as consumed by a training run."""
        if not entry_ids:
            return 0
        updated = 0
        with self.get_session() as session:
            for eid in entry_ids:
                record = session.scalar(
                    select(LabelingQueueRecord).where(LabelingQueueRecord.id == eid)
                )
                if record and record.status == LabelingQueueStatus.LABELED:
                    record.trained_into = trained_into
                    updated += 1
            session.commit()
        logger.info(
            "database.labeling_batch_trained",
            count=updated,
            trained_into=trained_into,
        )
        return updated

    def close(self) -> None:
        """Close the database engine."""
        if self._engine:
            self._engine.dispose()
