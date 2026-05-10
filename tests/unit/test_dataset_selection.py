"""Tests for argus.anomaly.dataset_selection (PR4)."""

from pathlib import Path

import pytest

from argus.anomaly.dataset_selection import (
    DatasetSelection,
    DatasetSelectionItem,
)


def test_item_round_trip():
    item = DatasetSelectionItem(
        camera_id="cam01", zone_id="default", version="v003", session_label="daytime"
    )
    raw = item.to_dict()
    assert raw == {
        "camera_id": "cam01",
        "zone_id": "default",
        "version": "v003",
        "session_label": "daytime",
    }
    assert DatasetSelectionItem.from_dict(raw) == item


def test_item_omits_empty_session_label():
    item = DatasetSelectionItem(
        camera_id="cam01", zone_id="default", version="v003", session_label=None
    )
    assert "session_label" not in item.to_dict()


def test_item_requires_camera_and_version():
    with pytest.raises(ValueError, match="camera_id"):
        DatasetSelectionItem.from_dict({"version": "v001"})
    with pytest.raises(ValueError, match="version"):
        DatasetSelectionItem.from_dict({"camera_id": "cam01"})
    with pytest.raises(ValueError, match="non-empty"):
        DatasetSelectionItem.from_dict({"camera_id": "", "version": "v001"})


def test_selection_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        DatasetSelection(items=[])


def test_selection_rejects_mixed_cameras():
    with pytest.raises(ValueError, match="cannot mix"):
        DatasetSelection(items=[
            DatasetSelectionItem("cam_a", "default", "v001"),
            DatasetSelectionItem("cam_b", "default", "v001"),
        ])


def test_selection_iter_dirs_resolves_under_root():
    sel = DatasetSelection(items=[
        DatasetSelectionItem("cam01", "default", "v001"),
        DatasetSelectionItem("cam01", "zone_b", "v003", session_label="night"),
    ])
    root = Path("/baselines")
    dirs = list(sel.iter_dirs(root))
    assert dirs == [
        root / "cam01" / "default" / "v001",
        root / "cam01" / "zone_b" / "v003",
    ]


def test_selection_camera_id_property():
    sel = DatasetSelection(items=[
        DatasetSelectionItem("cam01", "default", "v001"),
        DatasetSelectionItem("cam01", "default", "v003"),
    ])
    assert sel.camera_id == "cam01"


def test_selection_version_summary():
    sel = DatasetSelection(
        items=[
            DatasetSelectionItem("cam01", "default", "v001"),
            DatasetSelectionItem("cam01", "default", "v003"),
        ],
        total_frames=1234,
    )
    assert sel.version_summary() == "v001+v003 / 1234 frames"


def test_selection_json_round_trip():
    sel = DatasetSelection(
        items=[
            DatasetSelectionItem("cam01", "default", "v001", session_label="day"),
            DatasetSelectionItem("cam01", "zone_b", "v003"),
        ],
        total_frames=789,
    )
    restored = DatasetSelection.from_json(sel.to_json())
    assert restored.items == sel.items
    assert restored.total_frames == 789


def test_selection_from_json_none_returns_none():
    assert DatasetSelection.from_json(None) is None
    assert DatasetSelection.from_json("") is None


def test_selection_from_json_invalid_payload():
    with pytest.raises(ValueError, match="JSON parse"):
        DatasetSelection.from_json("not json")
    with pytest.raises(ValueError, match="must be a dict"):
        DatasetSelection.from_payload([1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a list"):
        DatasetSelection.from_payload({"items": "v001"})


def test_selection_from_payload_total_frames_typecheck():
    with pytest.raises(ValueError, match="must be an int"):
        DatasetSelection.from_payload({
            "items": [{"camera_id": "cam01", "version": "v001"}],
            "total_frames": "abc",
        })
    with pytest.raises(ValueError, match=">= 0"):
        DatasetSelection.from_payload({
            "items": [{"camera_id": "cam01", "version": "v001"}],
            "total_frames": -1,
        })


def test_selection_default_zone_when_missing():
    item = DatasetSelectionItem.from_dict({"camera_id": "cam01", "version": "v001"})
    assert item.zone_id == "default"
