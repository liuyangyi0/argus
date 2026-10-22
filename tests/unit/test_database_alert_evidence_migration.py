"""Tests for persisted alert evidence fields and auto migration."""

from datetime import datetime, timezone

import pytest
import sqlalchemy as sa

from argus.storage.database import Database


EVIDENCE_COLUMNS = {
    "detection_type",
    "evidence_summary",
    "evidence_json",
    "suppression_reason",
}


@pytest.fixture
def db(tmp_path):
    database = Database(database_url=f"sqlite:///{tmp_path / 'argus.db'}")
    database.initialize()
    yield database
    database.close()


def test_save_alert_persists_evidence_fields(db: Database) -> None:
    db.save_alert(
        alert_id="ALT-evidence-1",
        timestamp=datetime.now(timezone.utc),
        camera_id="cam-01",
        zone_id="z1",
        severity="medium",
        anomaly_score=0.88,
        detection_type="new_object",
        evidence_summary="new object in pipe rack",
        evidence_json={"detection_type": "new_object", "score": 0.88},
        suppression_reason=None,
    )

    alert = db.get_alert("ALT-evidence-1")
    assert alert is not None
    assert alert.detection_type == "new_object"
    assert alert.evidence_summary == "new object in pipe rack"
    assert alert.to_dict()["evidence_json"] == {
        "detection_type": "new_object",
        "score": 0.88,
    }


def test_auto_migrate_recovers_alert_evidence_columns(db: Database) -> None:
    with db._engine.connect() as conn:
        for column in EVIDENCE_COLUMNS:
            try:
                conn.execute(sa.text(f"ALTER TABLE alerts DROP COLUMN {column}"))
                conn.commit()
            except Exception as exc:
                pytest.skip(f"this SQLite cannot DROP COLUMN {column}: {exc}")

    inspector = sa.inspect(db._engine)
    before = {column["name"] for column in inspector.get_columns("alerts")}
    assert EVIDENCE_COLUMNS.isdisjoint(before)

    db._auto_migrate()

    inspector = sa.inspect(db._engine)
    after = {column["name"] for column in inspector.get_columns("alerts")}
    assert EVIDENCE_COLUMNS.issubset(after)
