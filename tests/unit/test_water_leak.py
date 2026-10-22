"""Unit tests for visible water leak / drip / pooling head."""

from __future__ import annotations

import numpy as np

from argus.anomaly.water_leak import WaterLeakHead


def _gray_frame(value: int = 128, *, h: int = 180, w: int = 240) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_expanding_dark_pool_triggers_visible_water_leak_candidate():
    head = WaterLeakHead(
        process_width=240,
        reference_frames=2,
        diff_threshold=12,
        darkening_threshold=10,
        min_area_px=40,
        min_growth_area_px=80,
        min_persistence_seconds=0.2,
        min_persistence_frames=3,
        learning_alpha=0.0,
        alert_score=0.84,
    )
    base = _gray_frame(140)
    assert head.process(base, timestamp=1.0).ready is False
    assert head.process(base, timestamp=1.1).ready is True

    small = base.copy()
    small[90:108, 80:98] = 82
    pending = head.process(small, timestamp=2.0)
    assert pending.has_detection is False
    assert pending.suppression_reason in {"growth_not_met", "persistence_not_met"}

    medium = base.copy()
    medium[88:112, 78:104] = 82
    head.process(medium, timestamp=2.1)

    large = base.copy()
    large[84:120, 74:114] = 82
    result = head.process(large, timestamp=2.35)

    assert result.has_detection is True
    assert result.score >= 0.84
    assert result.anomaly_map is not None
    candidate = result.candidates[0]
    assert candidate.leak_type == "pooling"
    assert candidate.growth_area_px >= 80
    assert candidate.persistence_seconds >= 0.2
    detected = candidate.to_detected_object()
    assert detected["class_name"] == "visible_water_leak"
    assert detected["leak_type"] == "pooling"


def test_vertical_bright_drip_triggers_without_pool_growth():
    head = WaterLeakHead(
        process_width=240,
        reference_frames=1,
        diff_threshold=15,
        brightening_threshold=12,
        min_area_px=35,
        min_growth_area_px=5000,
        min_persistence_seconds=0.1,
        min_persistence_frames=3,
        min_drip_length_px=28,
        min_vertical_aspect=3.0,
        learning_alpha=0.0,
    )
    base = _gray_frame(70)
    head.process(base, timestamp=1.0)

    for idx, ts in enumerate((2.0, 2.07, 2.16)):
        frame = base.copy()
        frame[40:92, 120 + idx:124 + idx] = 210
        result = head.process(frame, timestamp=ts)

    assert result.has_detection is True
    candidate = result.candidates[0]
    assert candidate.leak_type == "drip"
    assert candidate.vertical_aspect >= 3.0
    assert candidate.growth_area_px < 5000


def test_global_brightness_change_is_suppressed():
    head = WaterLeakHead(
        process_width=240,
        reference_frames=1,
        diff_threshold=10,
        max_change_fraction=0.05,
    )
    head.process(_gray_frame(100), timestamp=1.0)

    result = head.process(_gray_frame(150), timestamp=2.0)

    assert result.has_detection is False
    assert result.suppression_reason == "global_change"
    assert result.changed_fraction > 0.05
