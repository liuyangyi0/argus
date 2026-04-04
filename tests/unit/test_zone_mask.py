"""Tests for the zone mask engine."""

import numpy as np
import pytest

from argus.config.schema import ZoneConfig, ZonePriority
from argus.core.zone_mask import ZoneMaskEngine


def make_zone(
    zone_id: str = "z1",
    polygon: list[tuple[int, int]] | None = None,
    zone_type: str = "include",
    priority: ZonePriority = ZonePriority.STANDARD,
) -> ZoneConfig:
    if polygon is None:
        polygon = [(100, 100), (300, 100), (300, 300), (100, 300)]
    return ZoneConfig(
        zone_id=zone_id,
        name=f"Zone {zone_id}",
        polygon=polygon,
        zone_type=zone_type,
        priority=priority,
    )


class TestZoneMaskEngine:
    def test_no_zones_returns_original_frame(self):
        """No zones = no masking."""
        engine = ZoneMaskEngine(zones=[], frame_height=480, frame_width=640)
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        result = engine.apply(frame)
        np.testing.assert_array_equal(result, frame)

    def test_exclude_zone_blacks_out_region(self):
        """Exclude zone should set that region to black."""
        zone = make_zone(zone_type="exclude", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])
        engine = ZoneMaskEngine(zones=[zone], frame_height=480, frame_width=640)

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        result = engine.apply(frame)

        # The excluded region should be black
        assert result[50, 50, 0] == 0
        # Outside the exclude zone should be unchanged
        assert result[200, 200, 0] == 200

    def test_include_zone_only_keeps_region(self):
        """Include zone should only keep that region, black out everything else."""
        zone = make_zone(zone_type="include", polygon=[(100, 100), (200, 100), (200, 200), (100, 200)])
        engine = ZoneMaskEngine(zones=[zone], frame_height=480, frame_width=640)

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        result = engine.apply(frame)

        # Inside include zone should be preserved
        assert result[150, 150, 0] == 200
        # Outside include zone should be black
        assert result[0, 0, 0] == 0
        assert result[300, 300, 0] == 0

    def test_exclude_overrides_include(self):
        """Exclude zone should punch a hole in include zone."""
        include = make_zone(
            zone_id="inc", zone_type="include",
            polygon=[(0, 0), (400, 0), (400, 400), (0, 400)],
        )
        exclude = make_zone(
            zone_id="exc", zone_type="exclude",
            polygon=[(100, 100), (200, 100), (200, 200), (100, 200)],
        )
        engine = ZoneMaskEngine(zones=[include, exclude], frame_height=480, frame_width=640)

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        result = engine.apply(frame)

        # Inside include but outside exclude: preserved
        assert result[50, 50, 0] == 200
        # Inside exclude zone: black (even though inside include)
        assert result[150, 150, 0] == 0

    def test_update_zones_hot_reload(self):
        """update_zones should change the mask without recreating the engine."""
        engine = ZoneMaskEngine(zones=[], frame_height=480, frame_width=640)
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200

        # Initially no mask
        result1 = engine.apply(frame)
        assert result1[50, 50, 0] == 200

        # Add exclude zone
        zone = make_zone(zone_type="exclude", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])
        engine.update_zones([zone])

        result2 = engine.apply(frame)
        assert result2[50, 50, 0] == 0  # Now masked

    def test_get_include_and_exclude_zones(self):
        """Should correctly separate include and exclude zones."""
        zones = [
            make_zone(zone_id="inc1", zone_type="include"),
            make_zone(zone_id="exc1", zone_type="exclude"),
            make_zone(zone_id="inc2", zone_type="include"),
        ]
        engine = ZoneMaskEngine(zones=zones, frame_height=480, frame_width=640)

        assert len(engine.get_include_zones()) == 2
        assert len(engine.get_exclude_zones()) == 1

    def test_has_zones(self):
        """has_zones property."""
        empty = ZoneMaskEngine(zones=[], frame_height=480, frame_width=640)
        assert not empty.has_zones

        with_zones = ZoneMaskEngine(zones=[make_zone()], frame_height=480, frame_width=640)
        assert with_zones.has_zones

    def test_multiple_exclude_zones(self):
        """Multiple exclude zones should all be blacked out."""
        zones = [
            make_zone(zone_id="e1", zone_type="exclude", polygon=[(0, 0), (50, 0), (50, 50), (0, 50)]),
            make_zone(zone_id="e2", zone_type="exclude", polygon=[(400, 400), (450, 400), (450, 450), (400, 450)]),
        ]
        engine = ZoneMaskEngine(zones=zones, frame_height=480, frame_width=640)

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        result = engine.apply(frame)

        assert result[25, 25, 0] == 0    # First exclude
        assert result[425, 425, 0] == 0  # Second exclude
        assert result[200, 200, 0] == 200  # Outside both
