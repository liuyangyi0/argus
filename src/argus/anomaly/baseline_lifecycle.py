"""Baseline version lifecycle state machine.

Enforces strict unidirectional state transitions for nuclear audit compliance:
    Draft -> Verified -> Active -> Retired

Every transition is recorded in the audit log with who/when/why.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from sqlalchemy import select

from argus.storage.audit import AuditLogger
from argus.storage.database import Database
from argus.storage.models import BaselineState, BaselineVersionRecord

logger = structlog.get_logger()

VALID_TRANSITIONS: dict[str, set[str]] = {
    BaselineState.DRAFT: {BaselineState.VERIFIED},
    BaselineState.VERIFIED: {BaselineState.ACTIVE},
    BaselineState.ACTIVE: {BaselineState.RETIRED},
    BaselineState.RETIRED: set(),
}


class BaselineLifecycleError(Exception):
    """Raised when a lifecycle operation is invalid."""


class BaselineLifecycle:
    """Manages baseline version state transitions with audit trail."""

    def __init__(self, database: Database, audit: AuditLogger | None = None):
        self._db = database
        self._audit = audit

    def register_version(
        self,
        camera_id: str,
        zone_id: str,
        version: str,
        image_count: int = 0,
        group_id: str | None = None,
    ) -> BaselineVersionRecord:
        """Create a new baseline version record in Draft state."""
        with self._db.get_session() as session:
            existing = session.scalar(
                select(BaselineVersionRecord)
                .where(BaselineVersionRecord.camera_id == camera_id)
                .where(BaselineVersionRecord.zone_id == zone_id)
                .where(BaselineVersionRecord.version == version)
            )
            if existing:
                existing.image_count = image_count
                session.commit()
                session.refresh(existing)
                return existing

            record = BaselineVersionRecord(
                camera_id=camera_id,
                zone_id=zone_id,
                version=version,
                state=BaselineState.DRAFT,
                image_count=image_count,
                group_id=group_id,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            logger.info(
                "baseline.version_registered",
                camera_id=camera_id,
                zone_id=zone_id,
                version=version,
                image_count=image_count,
            )
            return record

    def get_version(
        self, camera_id: str, zone_id: str, version: str
    ) -> BaselineVersionRecord | None:
        """Look up a baseline version record."""
        with self._db.get_session() as session:
            return session.scalar(
                select(BaselineVersionRecord)
                .where(BaselineVersionRecord.camera_id == camera_id)
                .where(BaselineVersionRecord.zone_id == zone_id)
                .where(BaselineVersionRecord.version == version)
            )

    def get_versions(
        self, camera_id: str, zone_id: str = "default"
    ) -> list[BaselineVersionRecord]:
        """List all baseline versions for a camera/zone."""
        with self._db.get_session() as session:
            return list(
                session.scalars(
                    select(BaselineVersionRecord)
                    .where(BaselineVersionRecord.camera_id == camera_id)
                    .where(BaselineVersionRecord.zone_id == zone_id)
                    .order_by(BaselineVersionRecord.version.asc())
                ).all()
            )

    def get_active_version(
        self, camera_id: str, zone_id: str = "default"
    ) -> BaselineVersionRecord | None:
        """Get the currently Active version (at most one per camera/zone)."""
        with self._db.get_session() as session:
            return session.scalar(
                select(BaselineVersionRecord)
                .where(BaselineVersionRecord.camera_id == camera_id)
                .where(BaselineVersionRecord.zone_id == zone_id)
                .where(BaselineVersionRecord.state == BaselineState.ACTIVE)
            )

    def get_eligible_versions(
        self,
        camera_id: str,
        zone_id: str = "default",
        *,
        since: datetime | None = None,
    ) -> list[BaselineVersionRecord]:
        """Return baseline versions eligible for auto-retraining.

        Eligible = state in (VERIFIED, ACTIVE). DRAFT versions have not been
        approved yet; RETIRED ones were explicitly removed from the active pool.

        When ``since`` is provided, only versions created strictly after that
        timestamp are returned — used by the ``since_last_train`` retraining
        strategy to pick up newly-captured data without re-feeding history.
        """
        with self._db.get_session() as session:
            stmt = (
                select(BaselineVersionRecord)
                .where(BaselineVersionRecord.camera_id == camera_id)
                .where(BaselineVersionRecord.zone_id == zone_id)
                .where(
                    BaselineVersionRecord.state.in_(
                        [BaselineState.VERIFIED, BaselineState.ACTIVE]
                    )
                )
                .order_by(BaselineVersionRecord.version.asc())
            )
            if since is not None:
                stmt = stmt.where(BaselineVersionRecord.created_at > since)
            return list(session.scalars(stmt).all())

    def get_trainable_versions(
        self, camera_id: str, zone_id: str = "default"
    ) -> list[BaselineVersionRecord]:
        """Return versions eligible for training (Verified or Active)."""
        with self._db.get_session() as session:
            return list(
                session.scalars(
                    select(BaselineVersionRecord)
                    .where(BaselineVersionRecord.camera_id == camera_id)
                    .where(BaselineVersionRecord.zone_id == zone_id)
                    .where(
                        BaselineVersionRecord.state.in_([
                            BaselineState.VERIFIED,
                            BaselineState.ACTIVE,
                        ])
                    )
                    .order_by(BaselineVersionRecord.version.desc())
                ).all()
            )

    def delete_version(
        self,
        camera_id: str,
        zone_id: str,
        version: str,
        user: str = "operator",
        ip_address: str = "",
    ) -> bool:
        """Delete a non-active baseline version record."""
        with self._db.get_session() as session:
            record = session.scalar(
                select(BaselineVersionRecord)
                .where(BaselineVersionRecord.camera_id == camera_id)
                .where(BaselineVersionRecord.zone_id == zone_id)
                .where(BaselineVersionRecord.version == version)
            )
            if record is None:
                return False
            if record.state == BaselineState.ACTIVE:
                raise BaselineLifecycleError(
                    f"Cannot delete active baseline {camera_id}/{zone_id}/{version}"
                )

            session.delete(record)
            session.commit()

        logger.info(
            "baseline.version_deleted",
            camera_id=camera_id,
            zone_id=zone_id,
            version=version,
            user=user,
        )

        if self._audit:
            self._audit.log(
                user=user,
                action="baseline_deleted",
                target_type="baseline_version",
                target_id=f"{camera_id}/{zone_id}/{version}",
                detail="Baseline version deleted",
                ip_address=ip_address,
            )

        return True

    def _transition(
        self,
        camera_id: str,
        zone_id: str,
        version: str,
        target_state: str,
        user: str,
        detail: str = "",
        ip_address: str = "",
        extra_updates: dict | None = None,
    ) -> BaselineVersionRecord:
        """Execute a validated state transition atomically.

        State change and metadata updates happen in a single transaction.
        """
        with self._db.get_session() as session:
            record = session.scalar(
                select(BaselineVersionRecord)
                .where(BaselineVersionRecord.camera_id == camera_id)
                .where(BaselineVersionRecord.zone_id == zone_id)
                .where(BaselineVersionRecord.version == version)
            )
            if record is None:
                raise BaselineLifecycleError(
                    f"Baseline version not found: {camera_id}/{zone_id}/{version}"
                )

            current = record.state
            valid_targets = VALID_TRANSITIONS.get(current, set())
            if target_state not in valid_targets:
                raise BaselineLifecycleError(
                    f"Invalid transition: {current} -> {target_state} "
                    f"(allowed: {valid_targets or 'none'})"
                )

            record.state = target_state
            if extra_updates:
                for key, value in extra_updates.items():
                    setattr(record, key, value)
            session.commit()
            session.refresh(record)

            logger.info(
                "baseline.state_transition",
                camera_id=camera_id,
                zone_id=zone_id,
                version=version,
                from_state=current,
                to_state=target_state,
                user=user,
            )

            if self._audit:
                self._audit.log(
                    user=user,
                    action=f"baseline_{BaselineState(target_state).value}",
                    target_type="baseline_version",
                    target_id=f"{camera_id}/{zone_id}/{version}",
                    detail=detail or f"{current} -> {target_state}",
                    ip_address=ip_address,
                )

            return record

    def verify(
        self,
        camera_id: str,
        zone_id: str,
        version: str,
        verified_by: str,
        verified_by_secondary: str | None = None,
        ip_address: str = "",
    ) -> BaselineVersionRecord:
        """Draft -> Verified. Records who approved."""
        if not verified_by:
            raise BaselineLifecycleError("verified_by is required for verification")

        updates = {
            "verified_by": verified_by,
            "verified_at": datetime.now(timezone.utc),
        }
        if verified_by_secondary:
            updates["verified_by_secondary"] = verified_by_secondary

        return self._transition(
            camera_id, zone_id, version,
            BaselineState.VERIFIED, user=verified_by, ip_address=ip_address,
            detail=f"Verified by {verified_by}"
            + (f", {verified_by_secondary}" if verified_by_secondary else ""),
            extra_updates=updates,
        )

    def activate(
        self,
        camera_id: str,
        zone_id: str,
        version: str,
        user: str,
        ip_address: str = "",
    ) -> BaselineVersionRecord:
        """Verified -> Active. Auto-retires previous Active version."""
        current_active = self.get_active_version(camera_id, zone_id)
        if current_active and current_active.version != version:
            now = datetime.now(timezone.utc)
            self._transition(
                camera_id, zone_id, current_active.version,
                BaselineState.RETIRED, user=user, ip_address=ip_address,
                detail=f"Auto-retired: replaced by {version}",
                extra_updates={
                    "retired_at": now,
                    "retirement_reason": f"Replaced by {version}",
                },
            )

        return self._transition(
            camera_id, zone_id, version,
            BaselineState.ACTIVE, user=user, ip_address=ip_address,
            extra_updates={"activated_at": datetime.now(timezone.utc)},
        )

    def retire(
        self,
        camera_id: str,
        zone_id: str,
        version: str,
        user: str,
        reason: str = "",
        ip_address: str = "",
    ) -> BaselineVersionRecord:
        """Active -> Retired."""
        return self._transition(
            camera_id, zone_id, version,
            BaselineState.RETIRED, user=user, ip_address=ip_address,
            detail=reason or "Manual retirement",
            extra_updates={
                "retired_at": datetime.now(timezone.utc),
                "retirement_reason": reason,
            },
        )

    def is_version_tracked(self, camera_id: str, zone_id: str, version: str) -> bool:
        """Check if a version has a lifecycle record (vs legacy untracked)."""
        return self.get_version(camera_id, zone_id, version) is not None
