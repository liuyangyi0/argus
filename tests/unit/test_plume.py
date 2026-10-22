"""Unit tests for visible smoke / steam / dust / mist plume head."""

from __future__ import annotations

import cv2
import numpy as np

from argus.anomaly.plume import PlumeHead


def _gray_frame(value: int = 128, *, h: int = 180, w: int = 240) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _soft_ellipse(
    base: np.ndarray,
    *,
    center: tuple[int, int],
    axes: tuple[int, int],
    delta: int,
) -> np.ndarray:
    mask = np.zeros(base.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, thickness=-1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    return np.clip(base.astype(np.float32) + float(delta) * alpha, 0, 255).astype(
        np.uint8,
    )


def test_expanding_bright_diffuse_plume_triggers_candidate():
    head = PlumeHead(
        process_width=240,
        reference_frames=2,
        diff_threshold=8,
        brightening_threshold=7,
        min_area_px=50,
        min_growth_area_px=60,
        min_persistence_seconds=0.2,
        min_persistence_frames=3,
        max_edge_density=0.10,
        max_bbox_fill_fraction=0.90,
        learning_alpha=0.0,
        alert_score=0.83,
    )
    base = _gray_frame(105)
    assert head.process(base, timestamp=1.0).ready is False
    assert head.process(base, timestamp=1.1).ready is True

    small = _soft_ellipse(base, center=(124, 82), axes=(18, 10), delta=45)
    pending = head.process(small, timestamp=2.0)
    assert pending.has_detection is False
    assert pending.suppression_reason in {"growth_not_met", "persistence_not_met"}

    medium = _soft_ellipse(base, center=(126, 84), axes=(25, 15), delta=48)
    head.process(medium, timestamp=2.1)

    large = _soft_ellipse(base, center=(128, 86), axes=(36, 22), delta=50)
    result = head.process(large, timestamp=2.35)

    assert result.has_detection is True
    assert result.score >= 0.83
    assert result.anomaly_map is not None
    candidate = result.candidates[0]
    assert candidate.plume_type == "steam_or_mist"
    assert candidate.growth_area_px >= 60
    assert candidate.edge_density <= 0.10
    detected = candidate.to_detected_object()
    assert detected["class_name"] == "visible_plume"
    assert detected["plume_type"] == "steam_or_mist"


def test_dark_diffuse_plume_triggers_smoke_or_dust_candidate():
    head = PlumeHead(
        process_width=240,
        reference_frames=1,
        diff_threshold=8,
        darkening_threshold=7,
        min_area_px=50,
        min_growth_area_px=40,
        min_persistence_seconds=0.1,
        min_persistence_frames=3,
        learning_alpha=0.0,
    )
    base = _gray_frame(150)
    head.process(base, timestamp=1.0)

    for ts, axes in ((2.0, (15, 9)), (2.08, (24, 13)), (2.18, (34, 19))):
        frame = _soft_ellipse(base, center=(102, 92), axes=axes, delta=-42)
        result = head.process(frame, timestamp=ts)

    assert result.has_detection is True
    assert result.candidates[0].plume_type == "smoke_or_dust"


def test_sharp_solid_object_is_suppressed_by_diffuse_shape_gate():
    head = PlumeHead(
        process_width=240,
        reference_frames=1,
        diff_threshold=8,
        brightening_threshold=7,
        min_area_px=40,
        min_growth_area_px=0,
        min_persistence_seconds=0.0,
        min_persistence_frames=1,
        max_edge_density=0.04,
        max_bbox_fill_fraction=0.85,
        learning_alpha=0.0,
    )
    base = _gray_frame(100)
    head.process(base, timestamp=1.0)
    solid = base.copy()
    solid[65:110, 82:140] = 170

    result = head.process(solid, timestamp=2.0)

    assert result.has_detection is False
    assert result.suppression_reason == "solid_edge_suppressed"
    assert result.suppressed_region_count >= 1


def test_global_brightness_change_is_suppressed():
    head = PlumeHead(
        process_width=240,
        reference_frames=1,
        diff_threshold=8,
        max_change_fraction=0.05,
    )
    head.process(_gray_frame(100), timestamp=1.0)

    result = head.process(_gray_frame(150), timestamp=2.0)

    assert result.has_detection is False
    assert result.suppression_reason == "global_change"
    assert result.changed_fraction > 0.05
