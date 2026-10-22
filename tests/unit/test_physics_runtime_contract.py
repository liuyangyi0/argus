"""Physics Dashboard routes use the public camera runtime boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from argus.dashboard.routes.physics import get_active_tracks


def _request(manager):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(camera_manager=manager)),
    )


@pytest.mark.asyncio
async def test_active_tracks_uses_manager_snapshot_and_preserves_payload():
    tracks = [{
        "track_id": 4,
        "centroid_x": 12.3,
        "centroid_y": 45.6,
        "consecutive_frames": 8,
        "max_score": 0.923,
        "area_px": 120,
        "trajectory_length": 8,
    }]
    manager = MagicMock(spec=["get_active_tracks_snapshot"])
    manager.get_active_tracks_snapshot.return_value = tracks

    response = await get_active_tracks(_request(manager), "cam_01")

    assert response.status_code == 200
    assert json.loads(response.body)["data"] == {
        "camera_id": "cam_01",
        "tracks": tracks,
    }
    manager.get_active_tracks_snapshot.assert_called_once_with("cam_01")


@pytest.mark.asyncio
async def test_active_tracks_distinguishes_absent_runtime_from_empty_tracker():
    manager = MagicMock(spec=["get_active_tracks_snapshot"])
    manager.get_active_tracks_snapshot.side_effect = [None, []]

    missing = await get_active_tracks(_request(manager), "missing")
    empty = await get_active_tracks(_request(manager), "cam_01")

    assert missing.status_code == 404
    assert empty.status_code == 200
    assert json.loads(empty.body)["data"]["tracks"] == []
