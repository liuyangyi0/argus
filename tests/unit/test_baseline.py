"""Tests for baseline management."""

import numpy as np
import pytest

from argus.anomaly.baseline import BaselineManager


@pytest.fixture
def bm(tmp_path):
    return BaselineManager(baselines_dir=tmp_path / "baselines")


class TestBaselineManager:
    def test_create_first_version(self, bm):
        """First version should be v001."""
        version_dir = bm.create_new_version("cam_01", "zone_a")
        assert version_dir.name == "v001"
        assert version_dir.is_dir()

    def test_create_incremental_versions(self, bm):
        """Versions should increment."""
        v1 = bm.create_new_version("cam_01", "zone_a")
        v2 = bm.create_new_version("cam_01", "zone_a")
        v3 = bm.create_new_version("cam_01", "zone_a")
        assert v1.name == "v001"
        assert v2.name == "v002"
        assert v3.name == "v003"

    def test_save_frame(self, bm):
        """Should save a frame as PNG."""
        version_dir = bm.create_new_version("cam_01")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        path = bm.save_frame(frame, version_dir, 0)
        assert path.exists()
        assert path.suffix == ".png"

    def test_count_images(self, bm):
        """Should count images correctly."""
        version_dir = bm.create_new_version("cam_01")
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        for i in range(5):
            bm.save_frame(frame, version_dir, i)

        bm.set_current_version("cam_01", "default", "v001")
        assert bm.count_images("cam_01") == 5

    def test_set_and_get_current_version(self, bm):
        """Should track current version."""
        v1 = bm.create_new_version("cam_01")
        v2 = bm.create_new_version("cam_01")
        bm.set_current_version("cam_01", "default", "v002")

        current = bm.get_baseline_dir("cam_01")
        assert current.name == "v002"

    def test_cleanup_old_versions(self, bm):
        """Should remove old versions keeping N most recent."""
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        for i in range(5):
            v = bm.create_new_version("cam_01")
            bm.save_frame(frame, v, 0)

        removed = bm.cleanup_old_versions("cam_01", keep=2)
        assert removed == 3

        # Only v004 and v005 should remain
        base = bm.baselines_dir / "cam_01" / "default"
        remaining = sorted(base.glob("v*"))
        assert len(remaining) == 2
        assert remaining[0].name == "v004"
        assert remaining[1].name == "v005"

    def test_get_all_baselines(self, bm):
        """Should list all baselines."""
        frame = np.zeros((50, 50, 3), dtype=np.uint8)

        v1 = bm.create_new_version("cam_01", "zone_a")
        bm.save_frame(frame, v1, 0)
        bm.save_frame(frame, v1, 1)

        v2 = bm.create_new_version("cam_02", "default")
        bm.save_frame(frame, v2, 0)

        all_baselines = bm.get_all_baselines()
        assert len(all_baselines) == 2
        assert all_baselines[0]["camera_id"] == "cam_01"
        assert all_baselines[0]["image_count"] == 2
        assert all_baselines[1]["camera_id"] == "cam_02"

    def test_empty_baseline_count(self, bm):
        """No images should return 0."""
        assert bm.count_images("nonexistent") == 0
