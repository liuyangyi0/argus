"""Tests for _DatasetMerger and BaselineManager.resolve_dataset_dirs (PR5)."""

import os
from pathlib import Path

import pytest

from argus.anomaly.baseline import BaselineManager
from argus.anomaly.dataset_selection import (
    DatasetSelection,
    DatasetSelectionItem,
)
from argus.anomaly.trainer import _DatasetMerger, _build_merger_items


def _populate_version(root: Path, cam: str, zone: str, ver: str, n: int) -> Path:
    """Create a fake baseline version with n PNG files."""
    d = root / cam / zone / ver
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"baseline_{i:05d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return d


def test_resolve_dataset_dirs_returns_existing_paths(tmp_path):
    bm = BaselineManager(baselines_dir=str(tmp_path))
    _populate_version(tmp_path, "cam01", "default", "v001", 3)
    _populate_version(tmp_path, "cam01", "default", "v003", 5)

    sel = DatasetSelection(items=[
        DatasetSelectionItem("cam01", "default", "v001"),
        DatasetSelectionItem("cam01", "default", "v003"),
    ])
    dirs = bm.resolve_dataset_dirs(sel)

    assert len(dirs) == 2
    assert dirs[0].name == "v001"
    assert dirs[1].name == "v003"
    assert all(d.is_dir() for d in dirs)


def test_resolve_dataset_dirs_missing_version(tmp_path):
    bm = BaselineManager(baselines_dir=str(tmp_path))
    _populate_version(tmp_path, "cam01", "default", "v001", 3)
    sel = DatasetSelection(items=[
        DatasetSelectionItem("cam01", "default", "v999"),
    ])
    with pytest.raises(FileNotFoundError, match="not found"):
        bm.resolve_dataset_dirs(sel)


def test_resolve_dataset_dirs_empty_version(tmp_path):
    bm = BaselineManager(baselines_dir=str(tmp_path))
    (tmp_path / "cam01" / "default" / "v001").mkdir(parents=True)
    sel = DatasetSelection(items=[
        DatasetSelectionItem("cam01", "default", "v001"),
    ])
    with pytest.raises(FileNotFoundError, match="no images"):
        bm.resolve_dataset_dirs(sel)


def test_resolve_dataset_dirs_rejects_none():
    bm = BaselineManager(baselines_dir="/nonexistent")
    with pytest.raises(ValueError, match="required"):
        bm.resolve_dataset_dirs(None)


def test_count_images_multi(tmp_path):
    d1 = _populate_version(tmp_path, "cam01", "default", "v001", 3)
    d2 = _populate_version(tmp_path, "cam01", "default", "v003", 7)
    assert BaselineManager.count_images_multi([d1, d2]) == 10


def test_count_images_multi_skips_missing(tmp_path):
    d1 = _populate_version(tmp_path, "cam01", "default", "v001", 4)
    missing = tmp_path / "ghost"
    assert BaselineManager.count_images_multi([d1, missing]) == 4


def test_dataset_merger_merges_into_tmp(tmp_path):
    d1 = _populate_version(tmp_path, "cam01", "default", "v001", 2)
    d2 = _populate_version(tmp_path, "cam01", "default", "v003", 3)

    items = [
        ("cam01_default_v001", d1),
        ("cam01_default_v003", d2),
    ]

    with _DatasetMerger(items) as merged_root:
        files = sorted(p.name for p in merged_root.glob("*.png"))
        assert len(files) == 5
        # Filename prefix prevents collision between same baseline_00000.png in v001 and v003
        assert any(f.startswith("cam01_default_v001_") for f in files)
        assert any(f.startswith("cam01_default_v003_") for f in files)
        # The tmp dir actually exists
        assert merged_root.is_dir()
        captured_root = merged_root

    # After exit the tmp dir is gone
    assert not captured_root.exists()


def test_dataset_merger_falls_back_to_copy_when_symlink_fails(tmp_path, monkeypatch):
    """Win11 普通用户态 os.symlink raises OSError(WinError 1314); merger
    must fall back to shutil.copy2 transparently."""
    d1 = _populate_version(tmp_path, "cam01", "default", "v001", 2)
    items = [("cam01_default_v001", d1)]

    def _fake_symlink(src, dst):
        raise OSError(1314, "symbolic link privilege not held")

    monkeypatch.setattr(os, "symlink", _fake_symlink)

    with _DatasetMerger(items) as merged_root:
        files = list(merged_root.glob("*.png"))
        assert len(files) == 2
        # Files must be REAL files (not symlinks) when copy fallback engaged
        for f in files:
            assert not f.is_symlink()
            assert f.stat().st_size > 0  # real bytes copied


def test_dataset_merger_cleans_up_on_exception(tmp_path, monkeypatch):
    d1 = _populate_version(tmp_path, "cam01", "default", "v001", 1)
    items = [("cam01_default_v001", d1)]

    captured_root: list[Path] = []
    with pytest.raises(RuntimeError, match="boom"):
        with _DatasetMerger(items) as merged_root:
            captured_root.append(merged_root)
            raise RuntimeError("boom")

    assert captured_root and not captured_root[0].exists()


def test_build_merger_items_pairs_labels_and_dirs(tmp_path):
    sel = DatasetSelection(items=[
        DatasetSelectionItem("cam01", "default", "v001"),
        DatasetSelectionItem("cam01", "zone_b", "v003"),
    ])
    dirs = [tmp_path / "a", tmp_path / "b"]
    items = _build_merger_items(sel, dirs)
    assert items == [
        ("cam01_default_v001", tmp_path / "a"),
        ("cam01_zone_b_v003", tmp_path / "b"),
    ]
