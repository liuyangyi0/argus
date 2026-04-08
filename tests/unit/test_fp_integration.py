"""Tests for false positive feedback integration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from argus.alerts.feedback import FeedbackManager
from argus.anomaly.baseline import BaselineManager
from argus.anomaly.baseline_lifecycle import BaselineLifecycle
from argus.storage.database import Database
from argus.storage.models import BaselineState


@pytest.fixture()
def db(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'test.db'}")
    database.initialize()
    return database


@pytest.fixture()
def lifecycle(db):
    return BaselineLifecycle(db)


@pytest.fixture()
def baseline_mgr(tmp_path, lifecycle):
    return BaselineManager(tmp_path / "baselines", lifecycle=lifecycle)


@pytest.fixture()
def feedback_mgr(db, tmp_path):
    return FeedbackManager(
        database=db,
        baselines_dir=tmp_path / "baselines",
        alerts_dir=tmp_path / "alerts",
    )


def _create_fp_alert(db, camera_id, alert_id, tmp_path):
    """Create a false positive alert with a snapshot."""
    snapshot_dir = tmp_path / "alerts" / camera_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_dir / f"{alert_id}.jpg"
    snapshot.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

    db.save_alert(
        alert_id=alert_id,
        timestamp=datetime.now(timezone.utc),
        camera_id=camera_id,
        zone_id="default",
        severity="medium",
        anomaly_score=0.45,
        snapshot_path=str(snapshot),
    )
    db.mark_false_positive(alert_id, notes="Shadow artifact")
    return snapshot


def _create_fake_baseline(baseline_mgr, camera_id, count=10):
    """Create a baseline version with fake images."""
    version_dir = baseline_mgr.create_new_version(camera_id, "default")
    for i in range(count):
        (version_dir / f"baseline_{i:05d}.png").write_bytes(b"\x89PNG" + b"\x00" * 100)
    baseline_mgr.set_current_version(camera_id, "default", version_dir.name)
    return version_dir


class TestMergeFpIntoBaseline:
    def test_merge_creates_draft_version(self, feedback_mgr, baseline_mgr, db, tmp_path, lifecycle):
        _create_fake_baseline(baseline_mgr, "cam_01", count=5)
        # Manually create FP images in candidate pool
        fp_dir = tmp_path / "baselines" / "cam_01" / "default" / "false_positives"
        fp_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            (fp_dir / f"fp_test_{i}.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 50)

        result = feedback_mgr.merge_fp_into_baseline(
            "cam_01", baseline_manager=baseline_mgr
        )

        assert result["state"] == "draft"
        assert result["baseline_images"] == 5
        assert result["fp_included"] == 3
        assert "v002" in result["version"]

        # Verify lifecycle shows Draft
        rec = lifecycle.get_version("cam_01", "default", result["version"])
        assert rec is not None
        assert rec.state == BaselineState.DRAFT

    def test_merge_preserves_active(self, feedback_mgr, baseline_mgr, db, tmp_path, lifecycle):
        """Active baseline must not be modified."""
        version_dir = _create_fake_baseline(baseline_mgr, "cam_01", count=5)
        original_count = len(list(version_dir.glob("*.png")))

        # Create FP images
        fp_dir = tmp_path / "baselines" / "cam_01" / "default" / "false_positives"
        fp_dir.mkdir(parents=True, exist_ok=True)
        (fp_dir / "fp_test.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 50)

        feedback_mgr.merge_fp_into_baseline("cam_01", baseline_manager=baseline_mgr)

        # Original version unchanged
        assert len(list(version_dir.glob("*.png"))) == original_count

    def test_merge_empty_pool(self, feedback_mgr, baseline_mgr):
        _create_fake_baseline(baseline_mgr, "cam_01", count=5)
        result = feedback_mgr.merge_fp_into_baseline(
            "cam_01", baseline_manager=baseline_mgr
        )
        assert "error" in result
        assert result["fp_count"] == 0

    def test_merge_copies_meta_json(self, feedback_mgr, baseline_mgr, tmp_path):
        _create_fake_baseline(baseline_mgr, "cam_01", count=3)

        fp_dir = tmp_path / "baselines" / "cam_01" / "default" / "false_positives"
        fp_dir.mkdir(parents=True, exist_ok=True)
        (fp_dir / "fp_alert_001.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 50)
        (fp_dir / "fp_alert_001.meta.json").write_text(
            json.dumps({"alert_id": "alert_001", "operator": "wang"})
        )

        result = feedback_mgr.merge_fp_into_baseline(
            "cam_01", baseline_manager=baseline_mgr
        )

        # Check meta.json was copied to new version
        new_dir = baseline_mgr.baselines_dir / "cam_01" / "default" / result["version"]
        meta_files = list(new_dir.glob("*.meta.json"))
        assert len(meta_files) == 1

    def test_merge_with_max_fp_limit(self, feedback_mgr, baseline_mgr, tmp_path):
        _create_fake_baseline(baseline_mgr, "cam_01", count=3)

        fp_dir = tmp_path / "baselines" / "cam_01" / "default" / "false_positives"
        fp_dir.mkdir(parents=True, exist_ok=True)
        for i in range(10):
            (fp_dir / f"fp_{i:03d}.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 50)

        result = feedback_mgr.merge_fp_into_baseline(
            "cam_01", baseline_manager=baseline_mgr, max_fp_images=3
        )

        assert result["fp_included"] == 3


class TestActiveBaselineImmutability:
    def test_save_frame_to_active_rejected(self, baseline_mgr, lifecycle):
        version_dir = baseline_mgr.create_new_version("cam_01", "default")
        lifecycle.verify("cam_01", "default", "v001", verified_by="wang")
        lifecycle.activate("cam_01", "default", "v001", user="wang")

        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Active baselines are immutable"):
            baseline_mgr.save_frame(fake_frame, version_dir, index=999)

    def test_save_frame_to_draft_allowed(self, baseline_mgr, lifecycle):
        version_dir = baseline_mgr.create_new_version("cam_01", "default")
        # v001 is Draft, should be writable
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = baseline_mgr.save_frame(fake_frame, version_dir, index=0)
        assert result.exists()
