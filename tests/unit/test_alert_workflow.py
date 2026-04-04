"""Tests for alert workflow state machine."""

from datetime import datetime, timezone

import pytest

from argus.storage.database import Database
from argus.storage.models import AlertWorkflowStatus


@pytest.fixture
def db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_workflow.db"
    database = Database(database_url=f"sqlite:///{db_path}")
    database.initialize()
    yield database
    database.close()


def _create_alert(db, alert_id="ALT-001", severity="medium"):
    """Helper to create a test alert."""
    db.save_alert(
        alert_id=alert_id,
        timestamp=datetime.now(tz=timezone.utc),
        camera_id="cam_01",
        zone_id="zone_a",
        severity=severity,
        anomaly_score=0.85,
    )


class TestAlertWorkflow:
    def test_new_alert_has_new_status(self, db):
        """New alert should have workflow_status='new'."""
        _create_alert(db, "ALT-001")
        alert = db.get_alert("ALT-001")
        assert alert is not None
        assert alert.workflow_status == "new"

    def test_transition_new_to_acknowledged(self, db):
        """Should transition from new to acknowledged."""
        _create_alert(db, "ALT-001")

        result = db.update_alert_workflow("ALT-001", "acknowledged")
        assert result is True

        alert = db.get_alert("ALT-001")
        assert alert.workflow_status == "acknowledged"
        # Legacy field should also be set
        assert alert.acknowledged is True

    def test_transition_acknowledged_to_investigating_to_resolved_to_closed(self, db):
        """Should support the full lifecycle: acknowledged -> investigating -> resolved -> closed."""
        _create_alert(db, "ALT-001")

        db.update_alert_workflow("ALT-001", "acknowledged")
        alert = db.get_alert("ALT-001")
        assert alert.workflow_status == "acknowledged"

        db.update_alert_workflow("ALT-001", "investigating")
        alert = db.get_alert("ALT-001")
        assert alert.workflow_status == "investigating"

        db.update_alert_workflow("ALT-001", "resolved")
        alert = db.get_alert("ALT-001")
        assert alert.workflow_status == "resolved"
        assert alert.resolved_at is not None

        db.update_alert_workflow("ALT-001", "closed")
        alert = db.get_alert("ALT-001")
        assert alert.workflow_status == "closed"

    def test_assigned_to_stored_and_retrieved(self, db):
        """assigned_to field should be persisted."""
        _create_alert(db, "ALT-001")

        db.update_alert_workflow("ALT-001", "investigating", assigned_to="operator_li")
        alert = db.get_alert("ALT-001")
        assert alert.assigned_to == "operator_li"

    def test_notes_stored_on_workflow_update(self, db):
        """Notes should be updated via workflow transition."""
        _create_alert(db, "ALT-001")

        db.update_alert_workflow(
            "ALT-001", "resolved", notes="Found and removed debris"
        )
        alert = db.get_alert("ALT-001")
        assert alert.notes == "Found and removed debris"

    def test_false_positive_sets_legacy_field(self, db):
        """Transitioning to false_positive should set the legacy false_positive bool."""
        _create_alert(db, "ALT-001")

        db.update_alert_workflow("ALT-001", "false_positive")
        alert = db.get_alert("ALT-001")
        assert alert.workflow_status == "false_positive"
        assert alert.false_positive is True

    def test_workflow_stats_counts(self, db):
        """get_alert_workflow_stats should return correct counts per status."""
        _create_alert(db, "ALT-001")
        _create_alert(db, "ALT-002")
        _create_alert(db, "ALT-003")
        _create_alert(db, "ALT-004")

        db.update_alert_workflow("ALT-002", "acknowledged")
        db.update_alert_workflow("ALT-003", "investigating")
        db.update_alert_workflow("ALT-004", "resolved")

        stats = db.get_alert_workflow_stats()
        assert stats["new"] == 1
        assert stats["acknowledged"] == 1
        assert stats["investigating"] == 1
        assert stats["resolved"] == 1
        assert stats["closed"] == 0
        assert stats["false_positive"] == 0

    def test_update_nonexistent_alert(self, db):
        """Should return False for nonexistent alert."""
        result = db.update_alert_workflow("ALT-NOPE", "acknowledged")
        assert result is False

    def test_invalid_status_rejected(self, db):
        """Should return False for invalid workflow status."""
        _create_alert(db, "ALT-001")
        result = db.update_alert_workflow("ALT-001", "invalid_status")
        assert result is False

        # Status should remain unchanged
        alert = db.get_alert("ALT-001")
        assert alert.workflow_status == "new"

    def test_workflow_status_in_to_dict(self, db):
        """to_dict should include workflow fields."""
        _create_alert(db, "ALT-001")
        db.update_alert_workflow("ALT-001", "investigating", assigned_to="wang")

        alert = db.get_alert("ALT-001")
        d = alert.to_dict()
        assert d["workflow_status"] == "investigating"
        assert d["assigned_to"] == "wang"
        assert "resolved_at" in d


class TestAlertWorkflowStatusEnum:
    def test_all_values_exist(self):
        """All expected workflow statuses should be defined."""
        assert AlertWorkflowStatus.NEW.value == "new"
        assert AlertWorkflowStatus.ACKNOWLEDGED.value == "acknowledged"
        assert AlertWorkflowStatus.INVESTIGATING.value == "investigating"
        assert AlertWorkflowStatus.RESOLVED.value == "resolved"
        assert AlertWorkflowStatus.CLOSED.value == "closed"
        assert AlertWorkflowStatus.FALSE_POSITIVE.value == "false_positive"

    def test_is_str_enum(self):
        """AlertWorkflowStatus should be usable as a string."""
        assert str(AlertWorkflowStatus.NEW) == "AlertWorkflowStatus.NEW"
        assert AlertWorkflowStatus.NEW.value == "new"
