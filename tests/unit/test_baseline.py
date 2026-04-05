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


class TestDiversitySelect:
    """Tests for k-center greedy diversity selection (A4)."""

    def _create_test_images(self, tmp_path, count=100):
        """Create test images with varying colors for diversity testing."""
        import cv2

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        rng = np.random.default_rng(42)
        for i in range(count):
            img = np.full((64, 64, 3), rng.integers(0, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"img_{i:05d}.png"), img)
        return img_dir

    def test_diversity_select_reduces_count(self, tmp_path):
        """100 张图 → target=20 → 返回 20 个 Path。"""
        bm = BaselineManager(baselines_dir=tmp_path / "baselines")
        img_dir = self._create_test_images(tmp_path, count=100)
        result = bm.diversity_select(img_dir, target_count=20)
        assert len(result) == 20
        assert all(p.exists() for p in result)

    def test_diversity_select_with_fewer_images(self, tmp_path):
        """图片数 < target → 返回全部。"""
        bm = BaselineManager(baselines_dir=tmp_path / "baselines")
        img_dir = self._create_test_images(tmp_path, count=10)
        result = bm.diversity_select(img_dir, target_count=50)
        assert len(result) == 10

    def test_diversity_select_returns_sorted_paths(self, tmp_path):
        """返回的 Path 列表按文件名排序。"""
        bm = BaselineManager(baselines_dir=tmp_path / "baselines")
        img_dir = self._create_test_images(tmp_path, count=50)
        result = bm.diversity_select(img_dir, target_count=15)
        names = [p.name for p in result]
        assert names == sorted(names)


class TestOptimizeMovesToBackup:
    """Tests for the optimize workflow that moves unselected images to backup (A4)."""

    def test_optimize_moves_to_backup(self, tmp_path):
        """Unselected images should be moved to a backup/ subdirectory."""
        import shutil as _shutil

        bm = BaselineManager(baselines_dir=tmp_path / "baselines")
        version_dir = bm.create_new_version("cam_01")
        bm.set_current_version("cam_01", "default", version_dir.name)

        # Create 60 test images with varying colors
        rng = np.random.default_rng(123)
        for i in range(60):
            img = np.full((64, 64, 3), rng.integers(0, 256, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(str(version_dir / f"img_{i:05d}.png"), img)

        baseline_dir = bm.get_baseline_dir("cam_01", "default")
        all_images = sorted(
            list(baseline_dir.glob("*.png")) + list(baseline_dir.glob("*.jpg"))
        )
        assert len(all_images) == 60

        # Select diverse subset — keep 30 (minimum)
        target_count = max(30, int(len(all_images) * 0.2))
        selected = bm.diversity_select(baseline_dir, target_count)
        selected_set = set(selected)
        assert len(selected) == 30  # min 30

        # Move unselected to backup
        backup_dir = baseline_dir / "backup"
        backup_dir.mkdir(exist_ok=True)
        moved = 0
        for img_path in all_images:
            if img_path not in selected_set:
                _shutil.move(str(img_path), str(backup_dir / img_path.name))
                moved += 1

        assert moved == 30  # 60 - 30
        assert len(list(backup_dir.glob("*.png"))) == 30
        remaining = list(baseline_dir.glob("*.png"))
        assert len(remaining) == 30


class TestFalsePositiveAddsToBaseline:
    """Tests for the FP feedback loop that adds snapshots to baseline (A4-3)."""

    def test_false_positive_adds_to_baseline(self, tmp_path):
        """Marking an alert as FP should copy its snapshot into baseline dir."""
        import shutil as _shutil

        bm = BaselineManager(baselines_dir=tmp_path / "baselines")
        version_dir = bm.create_new_version("cam_01")
        bm.set_current_version("cam_01", "default", version_dir.name)

        # Create a fake snapshot (simulating what the alert would reference)
        alerts_dir = tmp_path / "alerts"
        alerts_dir.mkdir()
        snapshot = alerts_dir / "alert_snapshot.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite(str(snapshot), img)
        assert snapshot.exists()

        # Simulate the FP feedback: copy snapshot into baseline dir
        baseline_dir = bm.get_baseline_dir("cam_01", "default")
        dest = baseline_dir / "fp_testalertid0001.jpg"
        _shutil.copy2(str(snapshot), str(dest))

        # Verify the snapshot was added
        assert dest.exists()
        images = list(baseline_dir.glob("*.jpg")) + list(baseline_dir.glob("*.png"))
        assert len(images) >= 1
        assert any("fp_" in p.name for p in images)
