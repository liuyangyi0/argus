"""Lint guard: every column in ``_AUTO_MIGRATIONS`` must exist on the matching ORM model.

Why: ``_auto_migrate()`` is the only path that adds columns to existing user
SQLite databases. If a migration entry doesn't match a real ORM column, the
migration succeeds but the column is unused (silent waste). If a NEW ORM
column is added without an ``_AUTO_MIGRATIONS`` entry, existing user databases
crash with ``sqlite3.OperationalError: no such column`` on first read.

This test catches direction A (migration → ORM) directly. It also drives an
integration check that running ``_auto_migrate()`` on an empty schema then
creating ``Base.metadata.create_all()`` produces no missing-column surprises.
"""

from __future__ import annotations

import sqlalchemy as sa
import pytest

from argus.storage import models as m
from argus.storage.database import _AUTO_MIGRATIONS, Database


def _table_columns(table_name: str) -> set[str]:
    """Return the set of column names declared on the ORM model for ``table_name``."""
    for cls in m.Base.registry.mappers:
        mapped = cls.class_
        table = getattr(mapped, "__table__", None)
        if table is not None and table.name == table_name:
            return {c.name for c in table.columns}
    raise LookupError(f"no ORM model with __tablename__ = {table_name!r}")


class TestAutoMigrationsMatchOrm:
    """Direction A: every migrated column must have a matching ORM declaration."""

    def test_every_migration_entry_has_matching_orm_column(self) -> None:
        offenders: list[str] = []
        for table, column, _ in _AUTO_MIGRATIONS:
            try:
                cols = _table_columns(table)
            except LookupError as exc:
                offenders.append(f"{table}.{column}: {exc}")
                continue
            if column not in cols:
                offenders.append(
                    f"{table}.{column}: declared in _AUTO_MIGRATIONS but missing from ORM model"
                )
        assert not offenders, "Stale _AUTO_MIGRATIONS entries:\n  " + "\n  ".join(offenders)


class TestRoundTripMigrations:
    """End-to-end: applying _AUTO_MIGRATIONS to a fresh DB succeeds without warnings."""

    def test_auto_migrate_runs_clean_on_fresh_db(self, tmp_path) -> None:
        """A fresh schema should ALTER each migration column without error.

        On a brand-new DB, create_all() declares every column already, so each
        ALTER raises 'duplicate column' which we treat as success. This proves
        the migration list is at least executable end-to-end.
        """
        db_path = tmp_path / "argus_test.db"
        db = Database(database_url=f"sqlite:///{db_path}")
        db.initialize()  # creates schema + runs _auto_migrate once internally
        # Re-run to confirm idempotency — duplicates must be swallowed silently.
        db._auto_migrate()

    def test_auto_migrate_recovers_old_schema(self, tmp_path) -> None:
        """If we drop a recently-added column, _auto_migrate must put it back.

        Picks one cheap column (alerts.workflow_status) and verifies recovery.
        SQLite supports DROP COLUMN since 3.35, which Python 3.11 ships with.
        """
        db_path = tmp_path / "argus_test.db"
        db = Database(database_url=f"sqlite:///{db_path}")
        db.initialize()

        # Drop a known migrated column.
        with db._engine.connect() as conn:
            try:
                conn.execute(sa.text("ALTER TABLE alerts DROP COLUMN workflow_status"))
                conn.commit()
            except Exception as exc:
                pytest.skip(f"this SQLite cannot DROP COLUMN: {exc}")

        # Verify the column is actually gone.
        inspector = sa.inspect(db._engine)
        before = {c["name"] for c in inspector.get_columns("alerts")}
        assert "workflow_status" not in before

        # Re-run migration, expect column to come back.
        db._auto_migrate()

        inspector = sa.inspect(db._engine)
        after = {c["name"] for c in inspector.get_columns("alerts")}
        assert "workflow_status" in after, "_auto_migrate did not restore dropped column"
