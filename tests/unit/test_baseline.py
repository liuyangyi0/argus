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


class TestImageCrud:
    """Tests for per-image list/delete/upload on baseline versions."""

    def _encode_png(self, color=(0, 0, 0), size=(32, 32)):
        """Build a valid PNG byte stream for upload tests."""
        import cv2

        img = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        ok, buf = cv2.imencode(".png", img)
        assert ok
        return buf.tobytes()

    def _encode_jpg(self, color=(128, 128, 128), size=(32, 32)):
        import cv2

        img = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        return buf.tobytes()

    # ── list_images ──

    def test_list_images_empty_for_missing_version(self, bm):
        """Unknown camera/version returns empty list (not an error)."""
        result = bm.list_images("cam_missing", "v999")
        assert result == []

    def test_list_images_returns_metadata_sorted(self, bm):
        """Should list png + jpg files with size/created_at, sorted by filename."""
        version_dir = bm.create_new_version("cam_01", "default")
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        bm.save_frame(frame, version_dir, 0)  # baseline_00000.png
        bm.save_frame(frame, version_dir, 2)  # baseline_00002.png
        bm.save_frame(frame, version_dir, 1)  # baseline_00001.png

        # Also drop in a jpg and an ignored file
        (version_dir / "extra.jpg").write_bytes(self._encode_jpg())
        (version_dir / "README.txt").write_text("skip me")

        result = bm.list_images("cam_01", "v001")
        names = [item["filename"] for item in result]
        assert names == [
            "baseline_00000.png",
            "baseline_00001.png",
            "baseline_00002.png",
            "extra.jpg",
        ]
        # Metadata shape
        assert all({"filename", "size_bytes", "created_at"} <= set(r) for r in result)
        assert all(isinstance(r["size_bytes"], int) and r["size_bytes"] > 0 for r in result)
        # created_at parses as ISO
        from datetime import datetime as _dt

        _dt.fromisoformat(result[0]["created_at"])

    # ── delete_image safety ──

    @pytest.mark.parametrize(
        "bad_name",
        [
            "../etc/passwd.png",                       # traversal
            "..\\windows\\system32\\bad.png",          # backslash traversal
            "legit/subdir/file.png",                   # nested path
            "legit\\subdir\\file.png",                 # windows nested
            "has\x00null.png",                         # null byte
            "no_ext",                                  # no extension
            "bad.exe",                                 # wrong extension
            "bad.gif",                                 # wrong extension
            "",                                        # empty
            "a" * 200 + ".png",                        # oversized
        ],
    )
    def test_delete_image_rejects_unsafe_filename(self, bm, bad_name):
        """Path traversal / null bytes / bad extensions must be rejected."""
        version_dir = bm.create_new_version("cam_01", "default")
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        bm.save_frame(frame, version_dir, 0)

        assert bm.delete_image("cam_01", "v001", bad_name) is False
        # Existing good file must still be present
        assert (version_dir / "baseline_00000.png").exists()

    def test_delete_image_success(self, bm):
        """Canonical filename, non-active version deletes."""
        version_dir = bm.create_new_version("cam_01", "default")
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        bm.save_frame(frame, version_dir, 0)
        bm.save_frame(frame, version_dir, 1)

        assert bm.delete_image("cam_01", "v001", "baseline_00000.png") is True
        assert not (version_dir / "baseline_00000.png").exists()
        assert (version_dir / "baseline_00001.png").exists()

    def test_delete_image_missing_file_returns_false(self, bm):
        bm.create_new_version("cam_01", "default")
        assert bm.delete_image("cam_01", "v001", "baseline_00099.png") is False

    def test_delete_image_refuses_active_version(self, tmp_path):
        """ACTIVE versions are immutable — delete must be rejected."""
        from argus.anomaly.baseline_lifecycle import BaselineLifecycle
        from argus.storage.audit import AuditLogger
        from argus.storage.database import Database

        database = Database(f"sqlite:///{tmp_path / 'test.db'}")
        database.initialize()
        lifecycle = BaselineLifecycle(database, AuditLogger(database))

        bm_lc = BaselineManager(tmp_path / "baselines", lifecycle=lifecycle)
        version_dir = bm_lc.create_new_version("cam_01", "default")
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        # save_frame refuses to write to ACTIVE, so populate first, then activate
        bm_lc.save_frame(frame, version_dir, 0)

        lifecycle.verify("cam_01", "default", "v001", verified_by="wang")
        lifecycle.activate("cam_01", "default", "v001", user="wang")

        assert bm_lc.delete_image("cam_01", "v001", "baseline_00000.png") is False
        assert (version_dir / "baseline_00000.png").exists()

    # ── add_image_from_bytes ──

    def test_add_image_picks_next_index(self, bm):
        """Next filename should be max existing index + 1 (5-digit)."""
        version_dir = bm.create_new_version("cam_01", "default")
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        bm.save_frame(frame, version_dir, 0)
        bm.save_frame(frame, version_dir, 5)

        new_name = bm.add_image_from_bytes(
            "cam_01", "v001", data=self._encode_png(), ext="png",
        )
        assert new_name == "baseline_00006.png"
        assert (version_dir / new_name).exists()

    def test_add_image_rejects_invalid_ext(self, bm):
        bm.create_new_version("cam_01", "default")
        with pytest.raises(ValueError, match="不支持的图片格式"):
            bm.add_image_from_bytes("cam_01", "v001", data=b"xxx", ext="gif")

    def test_add_image_rejects_corrupt_bytes(self, bm):
        bm.create_new_version("cam_01", "default")
        with pytest.raises(ValueError, match="图片内容损坏"):
            bm.add_image_from_bytes(
                "cam_01", "v001", data=b"not an image at all", ext="png",
            )

    def test_add_image_refuses_active_version(self, tmp_path):
        from argus.anomaly.baseline_lifecycle import BaselineLifecycle
        from argus.storage.audit import AuditLogger
        from argus.storage.database import Database

        database = Database(f"sqlite:///{tmp_path / 'test.db'}")
        database.initialize()
        lifecycle = BaselineLifecycle(database, AuditLogger(database))

        bm_lc = BaselineManager(tmp_path / "baselines", lifecycle=lifecycle)
        version_dir = bm_lc.create_new_version("cam_01", "default")
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        bm_lc.save_frame(frame, version_dir, 0)

        lifecycle.verify("cam_01", "default", "v001", verified_by="wang")
        lifecycle.activate("cam_01", "default", "v001", user="wang")

        import cv2
        ok, buf = cv2.imencode(".png", np.zeros((20, 20, 3), dtype=np.uint8))
        assert ok
        with pytest.raises(ValueError, match="生产中的基线"):
            bm_lc.add_image_from_bytes(
                "cam_01", "v001", data=buf.tobytes(), ext="png",
            )


class TestImageUploadRoute:
    """End-to-end tests for the upload / list / delete HTTP routes."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        """FastAPI TestClient with the baseline router mounted and a
        BaselineManager pointing at tmp_path."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from argus.anomaly.baseline import BaselineManager
        from argus.dashboard.routes.baseline import router as baseline_router

        app = FastAPI()

        class _StubStorage:
            baselines_dir = str(tmp_path / "baselines")
            models_dir = str(tmp_path / "models")

        class _StubConfig:
            storage = _StubStorage()

        class _StubState:
            pass

        bm = BaselineManager(baselines_dir=tmp_path / "baselines")
        app.state.config = _StubConfig()
        app.state.baseline_manager = bm
        app.state.baseline_lifecycle = None
        app.state.audit_logger = None

        app.include_router(baseline_router, prefix="/api/baseline", tags=["baseline"])

        # Seed one version
        bm.create_new_version("cam_01", "default")

        return TestClient(app), bm

    def test_list_empty(self, client):
        tc, _bm = client
        r = tc.get("/api/baseline/cam_01/v001/images")
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == 0
        assert body["data"]["images"] == []
        assert body["data"]["total"] == 0

    def test_upload_png_and_list(self, client):
        import cv2
        import numpy as np

        tc, _bm = client
        img = np.full((40, 40, 3), 10, dtype=np.uint8)
        _ok, buf = cv2.imencode(".png", img)
        payload = buf.tobytes()

        r = tc.post(
            "/api/baseline/cam_01/v001/images",
            files={"file": ("fresh.png", payload, "image/png")},
        )
        assert r.status_code == 200
        assert r.json()["data"]["filename"] == "baseline_00000.png"

        r = tc.get("/api/baseline/cam_01/v001/images")
        assert r.json()["data"]["total"] == 1

    def test_upload_rejects_oversized(self, client):
        tc, _bm = client
        # 11 MB of zeros — over the 10 MB cap
        payload = b"\x89PNG\r\n\x1a\n" + b"0" * (11 * 1024 * 1024)
        r = tc.post(
            "/api/baseline/cam_01/v001/images",
            files={"file": ("big.png", payload, "image/png")},
        )
        assert r.status_code == 400
        body = r.json()
        assert "上限 10 MB" in body["msg"]

    def test_upload_rejects_bad_content_type(self, client):
        tc, _bm = client
        r = tc.post(
            "/api/baseline/cam_01/v001/images",
            files={"file": ("evil.sh", b"#!/bin/sh", "application/x-shellscript")},
        )
        assert r.status_code == 400

    def test_delete_traversal_rejected_by_route(self, client):
        """Path parameter with `..` or `/` must not reach the handler."""
        tc, _bm = client
        # URL-encoded .. still gets filtered by FastAPI normalization; try dot-dot via
        # an explicit filename with a literal `..` string
        r = tc.delete("/api/baseline/cam_01/v001/images/..%2Fpasswd.png")
        # Either 404 (path routing never matches) or 400 (validation rejected).
        assert r.status_code in (400, 404, 405)

    def test_content_route_serves_image(self, client):
        import cv2
        import numpy as np

        tc, bm = client
        img = np.full((20, 20, 3), 50, dtype=np.uint8)
        _ok, buf = cv2.imencode(".png", img)
        bm.add_image_from_bytes("cam_01", "v001", data=buf.tobytes(), ext="png")

        r = tc.get("/api/baseline/cam_01/v001/images/baseline_00000.png/content")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/png")
        assert len(r.content) > 0
