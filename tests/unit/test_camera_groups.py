"""Tests for camera group shared baseline functionality."""

from __future__ import annotations

import pytest

from argus.anomaly.baseline import BaselineManager
from argus.anomaly.baseline_lifecycle import BaselineLifecycle
from argus.config.schema import CameraGroupConfig
from argus.storage.database import Database


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


def _create_fake_images(directory, count=10, prefix="baseline"):
    """Create fake image files for testing."""
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (directory / f"{prefix}_{i:05d}.png").write_bytes(b"\x89PNG" + b"\x00" * 100)


class TestCameraGroupConfig:
    def test_valid_group_id(self):
        cfg = CameraGroupConfig(
            group_id="CORRIDOR-A", name="A栋走廊", camera_ids=["cam_03", "cam_04"]
        )
        assert cfg.group_id == "CORRIDOR-A"

    def test_invalid_group_id(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            CameraGroupConfig(
                group_id="BAD GROUP!", name="test", camera_ids=["cam_01"]
            )

    def test_empty_camera_ids(self):
        with pytest.raises(ValueError):
            CameraGroupConfig(
                group_id="EMPTY", name="test", camera_ids=[]
            )


class TestGroupVersionCreation:
    def test_create_group_version(self, baseline_mgr, tmp_path):
        version_dir = baseline_mgr.create_group_version("CORRIDOR-A")
        assert version_dir.exists()
        assert version_dir.name == "v001"
        assert "_groups" in str(version_dir)
        assert "CORRIDOR-A" in str(version_dir)

    def test_version_auto_increments(self, baseline_mgr):
        v1 = baseline_mgr.create_group_version("CORRIDOR-A")
        v2 = baseline_mgr.create_group_version("CORRIDOR-A")
        assert v1.name == "v001"
        assert v2.name == "v002"

    def test_group_version_registered_in_lifecycle(self, baseline_mgr, lifecycle):
        # create_group_version only creates the directory; lifecycle registration
        # happens in merge_camera_baselines to avoid double-registration.
        version_dir = baseline_mgr.create_group_version("CORRIDOR-A")
        assert version_dir.is_dir()
        # No lifecycle record yet — registration deferred to merge
        rec = lifecycle.get_version("group:CORRIDOR-A", "default", "v001")
        assert rec is None


class TestMergeCameraBaselines:
    def test_merge_two_cameras(self, baseline_mgr, tmp_path):
        # Create baselines for two cameras
        cam1_dir = baseline_mgr.create_new_version("cam_03", "default")
        _create_fake_images(cam1_dir, count=5, prefix="baseline")
        baseline_mgr.set_current_version("cam_03", "default", cam1_dir.name)

        cam2_dir = baseline_mgr.create_new_version("cam_04", "default")
        _create_fake_images(cam2_dir, count=7, prefix="baseline")
        baseline_mgr.set_current_version("cam_04", "default", cam2_dir.name)

        # Merge into group
        merged_dir = baseline_mgr.merge_camera_baselines(
            group_id="CORRIDOR-A",
            camera_ids=["cam_03", "cam_04"],
        )

        images = list(merged_dir.glob("*.png"))
        assert len(images) == 12  # 5 + 7
        assert merged_dir.name == "v001"

    def test_merge_with_deduplication(self, baseline_mgr, tmp_path):
        cam1_dir = baseline_mgr.create_new_version("cam_01", "default")
        _create_fake_images(cam1_dir, count=20, prefix="baseline")
        baseline_mgr.set_current_version("cam_01", "default", cam1_dir.name)

        cam2_dir = baseline_mgr.create_new_version("cam_02", "default")
        _create_fake_images(cam2_dir, count=20, prefix="baseline")
        baseline_mgr.set_current_version("cam_02", "default", cam2_dir.name)

        merged_dir = baseline_mgr.merge_camera_baselines(
            group_id="DEDUP-TEST",
            camera_ids=["cam_01", "cam_02"],
            target_count=15,
        )

        images = list(merged_dir.glob("*.png"))
        assert len(images) <= 15

    def test_merge_missing_camera(self, baseline_mgr):
        """Cameras without baselines are silently skipped."""
        cam1_dir = baseline_mgr.create_new_version("cam_01", "default")
        _create_fake_images(cam1_dir, count=5)
        baseline_mgr.set_current_version("cam_01", "default", cam1_dir.name)

        merged_dir = baseline_mgr.merge_camera_baselines(
            group_id="PARTIAL",
            camera_ids=["cam_01", "cam_nonexistent"],
        )

        images = list(merged_dir.glob("*.png"))
        assert len(images) == 5

    def test_merge_updates_lifecycle_count(self, baseline_mgr, lifecycle):
        cam1_dir = baseline_mgr.create_new_version("cam_01", "default")
        _create_fake_images(cam1_dir, count=8)
        baseline_mgr.set_current_version("cam_01", "default", cam1_dir.name)

        baseline_mgr.merge_camera_baselines(
            group_id="COUNT-TEST",
            camera_ids=["cam_01"],
        )

        rec = lifecycle.get_version("group:COUNT-TEST", "default", "v001")
        assert rec is not None
        assert rec.image_count == 8


class TestGroupBaselineDir:
    def test_get_group_dir_empty(self, baseline_mgr):
        d = baseline_mgr.get_group_baseline_dir("NONEXISTENT")
        assert "_groups" in str(d)
        assert "NONEXISTENT" in str(d)

    def test_get_group_dir_with_version(self, baseline_mgr):
        v = baseline_mgr.create_group_version("TEST-G")
        (v / "test.png").write_bytes(b"fake")

        d = baseline_mgr.get_group_baseline_dir("TEST-G")
        assert d == v


class TestIndividualModeUnaffected:
    def test_individual_camera_still_works(self, baseline_mgr):
        """Cameras without groups continue to work normally."""
        v = baseline_mgr.create_new_version("cam_solo", "default")
        assert "cam_solo" in str(v)
        assert "_groups" not in str(v)
        assert v.name == "v001"
