"""Tests for baseline version lifecycle state machine."""

from __future__ import annotations

import pytest

from argus.anomaly.baseline import BaselineManager
from argus.anomaly.baseline_lifecycle import BaselineLifecycle, BaselineLifecycleError
from argus.storage.audit import AuditLogger
from argus.storage.database import Database
from argus.storage.models import BaselineState


@pytest.fixture()
def db(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'test.db'}")
    database.initialize()
    return database


@pytest.fixture()
def audit(db):
    return AuditLogger(db)


@pytest.fixture()
def lifecycle(db, audit):
    return BaselineLifecycle(db, audit)


@pytest.fixture()
def baseline_mgr(tmp_path, lifecycle):
    return BaselineManager(tmp_path / "baselines", lifecycle=lifecycle)


class TestRegisterVersion:
    def test_register_creates_draft(self, lifecycle):
        rec = lifecycle.register_version("cam_01", "default", "v001", image_count=50)
        assert rec.state == BaselineState.DRAFT
        assert rec.camera_id == "cam_01"
        assert rec.version == "v001"
        assert rec.image_count == 50

    def test_register_idempotent(self, lifecycle):
        lifecycle.register_version("cam_01", "default", "v001", image_count=50)
        rec = lifecycle.register_version("cam_01", "default", "v001", image_count=60)
        assert rec.image_count == 60


class TestDraftToVerified:
    def test_verify_success(self, lifecycle):
        lifecycle.register_version("cam_01", "default", "v001")
        rec = lifecycle.verify("cam_01", "default", "v001", verified_by="wang")
        assert rec.state == BaselineState.VERIFIED
        assert rec.verified_by == "wang"
        assert rec.verified_at is not None

    def test_verify_with_secondary(self, lifecycle):
        lifecycle.register_version("cam_01", "default", "v001")
        rec = lifecycle.verify(
            "cam_01", "default", "v001",
            verified_by="wang", verified_by_secondary="zhang",
        )
        assert rec.verified_by_secondary == "zhang"

    def test_verify_requires_user(self, lifecycle):
        lifecycle.register_version("cam_01", "default", "v001")
        with pytest.raises(BaselineLifecycleError, match="verified_by is required"):
            lifecycle.verify("cam_01", "default", "v001", verified_by="")


class TestVerifiedToActive:
    def test_activate_success(self, lifecycle):
        lifecycle.register_version("cam_01", "default", "v001")
        lifecycle.verify("cam_01", "default", "v001", verified_by="wang")
        rec = lifecycle.activate("cam_01", "default", "v001", user="wang")
        assert rec.state == BaselineState.ACTIVE
        assert rec.activated_at is not None

    def test_activate_retires_previous(self, lifecycle):
        # v001: Draft -> Verified -> Active
        lifecycle.register_version("cam_01", "default", "v001")
        lifecycle.verify("cam_01", "default", "v001", verified_by="wang")
        lifecycle.activate("cam_01", "default", "v001", user="wang")

        # v002: Draft -> Verified -> Active (should auto-retire v001)
        lifecycle.register_version("cam_01", "default", "v002")
        lifecycle.verify("cam_01", "default", "v002", verified_by="wang")
        lifecycle.activate("cam_01", "default", "v002", user="wang")

        v001 = lifecycle.get_version("cam_01", "default", "v001")
        assert v001.state == BaselineState.RETIRED
        assert v001.retired_at is not None
        assert "v002" in v001.retirement_reason


class TestActiveToRetired:
    def test_retire_success(self, lifecycle):
        lifecycle.register_version("cam_01", "default", "v001")
        lifecycle.verify("cam_01", "default", "v001", verified_by="wang")
        lifecycle.activate("cam_01", "default", "v001", user="wang")
        rec = lifecycle.retire(
            "cam_01", "default", "v001", user="wang", reason="Replaced by better set"
        )
        assert rec.state == BaselineState.RETIRED
        assert rec.retired_at is not None
        assert rec.retirement_reason == "Replaced by better set"


class TestInvalidTransitions:
    def test_draft_to_active_rejected(self, lifecycle):
        lifecycle.register_version("cam_01", "default", "v001")
        with pytest.raises(BaselineLifecycleError, match="Invalid transition"):
            lifecycle.activate("cam_01", "default", "v001", user="wang")

    def test_retired_to_verified_rejected(self, lifecycle):
        lifecycle.register_version("cam_01", "default", "v001")
        lifecycle.verify("cam_01", "default", "v001", verified_by="wang")
        lifecycle.activate("cam_01", "default", "v001", user="wang")
        lifecycle.retire("cam_01", "default", "v001", user="wang")
        with pytest.raises(BaselineLifecycleError, match="Invalid transition"):
            lifecycle.verify("cam_01", "default", "v001", verified_by="wang")

    def test_draft_to_retired_rejected(self, lifecycle):
        lifecycle.register_version("cam_01", "default", "v001")
        with pytest.raises(BaselineLifecycleError, match="Invalid transition"):
            lifecycle.retire("cam_01", "default", "v001", user="wang")

    def test_nonexistent_version(self, lifecycle):
        with pytest.raises(BaselineLifecycleError, match="not found"):
            lifecycle.verify("cam_99", "default", "v999", verified_by="wang")


class TestTrainableVersions:
    def test_only_verified_and_active(self, lifecycle):
        lifecycle.register_version("cam_01", "default", "v001")
        lifecycle.register_version("cam_01", "default", "v002")
        lifecycle.register_version("cam_01", "default", "v003")

        # v001: Verified, v002: Active, v003: Draft
        lifecycle.verify("cam_01", "default", "v001", verified_by="wang")
        lifecycle.verify("cam_01", "default", "v002", verified_by="wang")
        lifecycle.activate("cam_01", "default", "v002", user="wang")

        trainable = lifecycle.get_trainable_versions("cam_01", "default")
        versions = {v.version for v in trainable}
        assert "v001" in versions  # Verified
        assert "v002" in versions  # Active
        assert "v003" not in versions  # Draft


class TestAuditLogging:
    def test_transitions_create_audit_entries(self, lifecycle, audit):
        lifecycle.register_version("cam_01", "default", "v001")
        lifecycle.verify("cam_01", "default", "v001", verified_by="wang")
        lifecycle.activate("cam_01", "default", "v001", user="zhang")

        logs = audit.get_logs(action="baseline_verified")
        assert len(logs) == 1
        assert logs[0].user == "wang"

        logs = audit.get_logs(action="baseline_active")
        assert len(logs) == 1
        assert logs[0].user == "zhang"


class TestBaselineManagerIntegration:
    def test_create_version_registers_draft(self, baseline_mgr, lifecycle):
        baseline_mgr.create_new_version("cam_01", "default")
        rec = lifecycle.get_version("cam_01", "default", "v001")
        assert rec is not None
        assert rec.state == BaselineState.DRAFT

    def test_cleanup_preserves_retired(self, baseline_mgr, lifecycle, tmp_path):
        # Create 4 versions
        for _ in range(4):
            d = baseline_mgr.create_new_version("cam_01", "default")
            # Put a dummy image so directory isn't empty
            (d / "baseline_00000.png").write_bytes(b"fake")

        # Make v001 go through full lifecycle to Retired
        lifecycle.verify("cam_01", "default", "v001", verified_by="wang")
        lifecycle.activate("cam_01", "default", "v001", user="wang")
        lifecycle.retire("cam_01", "default", "v001", user="wang", reason="old")

        # Cleanup with keep=2 should remove v002 (draft, deletable) but not v001 (retired)
        removed = baseline_mgr.cleanup_old_versions("cam_01", "default", keep=2)
        assert removed == 1

        base = tmp_path / "baselines" / "cam_01" / "default"
        remaining = sorted(d.name for d in base.glob("v*") if d.is_dir())
        assert "v001" in remaining  # Retired, preserved
        assert "v003" in remaining
        assert "v004" in remaining

    def test_get_all_baselines_includes_state(self, baseline_mgr, lifecycle):
        d = baseline_mgr.create_new_version("cam_01", "default")
        (d / "baseline_00000.png").write_bytes(b"fake")
        baseline_mgr.set_current_version("cam_01", "default", "v001")

        baselines = baseline_mgr.get_all_baselines()
        assert len(baselines) == 1
        assert baselines[0]["state"] == BaselineState.DRAFT
