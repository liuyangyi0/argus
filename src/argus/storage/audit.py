"""Audit trail for operator actions.

Records who did what, when, and from where. Required for nuclear plant
compliance and traceability.
"""
from __future__ import annotations
from datetime import datetime, timezone
import structlog
from sqlalchemy import func as sa_func, select
from argus.storage.database import Database
from argus.storage.models import AuditLog

logger = structlog.get_logger()


class AuditLogger:
    """Records operator actions to the audit log."""

    def __init__(self, database: Database):
        self._db = database

    def log(
        self,
        user: str,
        action: str,
        target_type: str,
        target_id: str = "",
        detail: str = "",
        ip_address: str = "",
    ) -> None:
        """Record an audit event."""
        try:
            with self._db.get_session() as session:
                entry = AuditLog(
                    timestamp=datetime.now(timezone.utc),
                    user=user,
                    action=action,
                    target_type=target_type,
                    target_id=target_id,
                    detail=detail,
                    ip_address=ip_address,
                )
                session.add(entry)
                session.commit()
        except Exception as e:
            logger.error("audit.log_failed", action=action, error=str(e))

    def get_logs(
        self,
        user: str | None = None,
        action: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditLog]:
        """Query audit logs with optional filters."""
        with self._db.get_session() as session:
            stmt = select(AuditLog).order_by(AuditLog.timestamp.desc())
            if user:
                stmt = stmt.where(AuditLog.user == user)
            if action:
                stmt = stmt.where(AuditLog.action == action)
            stmt = stmt.offset(offset).limit(limit)
            return list(session.scalars(stmt).all())

    def count_logs(self, user: str | None = None, action: str | None = None) -> int:
        """Count audit log entries."""
        with self._db.get_session() as session:
            stmt = select(sa_func.count()).select_from(AuditLog)
            if user:
                stmt = stmt.where(AuditLog.user == user)
            if action:
                stmt = stmt.where(AuditLog.action == action)
            return session.scalar(stmt) or 0
