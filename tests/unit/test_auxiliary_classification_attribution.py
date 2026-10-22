"""Pure classification-evidence geometry checks; all heatmaps/results are fake."""

from __future__ import annotations

import numpy as np
import pytest

from argus.anomaly.classification_policy import localized_classification, trusted_classification


def _map():
    heatmap = np.zeros((20, 20), dtype=np.float32)
    heatmap[6:10, 6:10] = 0.99
    return heatmap


def _evidence(**updates):
    return {"label": "shadow", "confidence": 0.99, "bbox": [60, 60, 100, 100], **updates}


def _check(result=None, heatmap=None, frame=(200, 200, 3), crop=(50, 50, 60, 60)):
    return localized_classification(
        _evidence() if result is None else result,
        {"shadow", "wrench"},
        _map() if heatmap is None else heatmap,
        frame, crop,
    )


def test_accepts_precisely_same_single_anomaly_support():
    assert _check() == ("shadow", 0.99)


@pytest.mark.parametrize("result", [
    ("shadow", 0.99), {},
    _evidence(bbox=None), _evidence(bbox=[]), _evidence(bbox=[60, 60, 100]),
    _evidence(bbox=[60, 60, 60, 100]), _evidence(bbox=[100, 60, 60, 100]),
    _evidence(bbox=[True, 60, 100, 100]), _evidence(bbox=["60", 60, 100, 100]),
    _evidence(bbox=[float("nan"), 60, 100, 100]),
    _evidence(bbox=[60, 60, float("inf"), 100]),
    _evidence(bbox=[60, 60, 10**500, 100]),
    _evidence(bbox=[-1, 60, 100, 100]),
    _evidence(bbox=[60, 60, 201, 100]),
    _evidence(bbox=[40, 60, 100, 100]),
    _evidence(bbox=[60, 60, 120, 100]),
    _evidence(label="not-in-vocabulary"),
    _evidence(confidence=0.89), _evidence(confidence=True),
    _evidence(confidence="0.99"), _evidence(confidence=float("nan")),
    _evidence(confidence=10**500),
])
def test_rejects_untrusted_missing_or_invalid_sdk_location(result):
    assert _check(result=result) is None


@pytest.mark.parametrize("crop", [
    None, (0, 0, 0, 100), (0, 0, -10, 100), (0, 0, 201, 200),
    (-1, 0, 200, 200), (0, 0, True, 100), (0, float("nan"), 100, 100),
    (0, 0, float("inf"), 200), (0, 0, 10**500, 200),
])
def test_rejects_missing_or_invalid_requested_crop(crop):
    assert _check(crop=crop) is None


@pytest.mark.parametrize("frame", [None, (), (200,), (0, 200), (200, -1), (True, 200), ("200", 200), (200.5, 200), (float("inf"), 200)])
def test_rejects_invalid_frame_geometry(frame):
    assert _check(frame=frame) is None


@pytest.mark.parametrize("heatmap", [
    np.empty((0, 20)), np.zeros((20, 20)), np.ones((20, 20)),
    np.zeros((20, 20, 1)), np.zeros(20), np.ones((20, 20), dtype=bool),
    np.full((20, 20), "0.99"), np.ones((20, 20), dtype=np.complex64),
])
def test_rejects_empty_constant_or_malformed_heatmap(heatmap):
    assert _check(heatmap=heatmap) is None


def test_missing_heatmap_cannot_use_full_frame_classification_as_evidence():
    assert localized_classification(_evidence(), {"shadow"}, None, (200, 200), (0, 0, 200, 200)) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_rejects_invalid_heatmap_values(value):
    heatmap = _map()
    heatmap[0, 0] = value
    assert _check(heatmap=heatmap) is None


def test_rejects_tiny_peak_and_tiny_contrast_normalization_artifacts():
    assert _check(heatmap=_map() * 1e-7) is None
    heatmap = np.full((20, 20), 0.95)
    heatmap[6:10, 6:10] = 0.99
    assert _check(heatmap=heatmap) is None


def test_rejects_two_anomaly_components_even_if_detection_matches_one():
    heatmap = _map()
    heatmap[14:17, 14:17] = 0.99
    assert _check(heatmap=heatmap) is None


@pytest.mark.parametrize("box", [
    [0, 0, 200, 200],       # huge box that contains the anomaly and other objects
    [90, 90, 130, 130],    # merely grazes one corner
    [60, 60, 80, 100],     # misses half the anomalous support
    [110, 110, 150, 150],  # unrelated object elsewhere in the frame
])
def test_rejects_partial_large_or_unrelated_object_boxes(box):
    assert _check(result=_evidence(bbox=box), crop=(0, 0, 200, 200)) is None


def test_requires_center_in_support_even_with_high_bidirectional_coverage():
    heatmap = np.zeros((20, 20), dtype=np.float32)
    heatmap[5:15, 5:15] = 0.99
    heatmap[10, 10] = 0
    assert _check(
        result=_evidence(bbox=[50, 50, 150, 150]),
        heatmap=heatmap, crop=(40, 40, 120, 120),
    ) is None


def test_uses_conservative_floor_ceil_mapping_for_fractional_sdk_box():
    # A subpixel expansion consumes an extra heatmap border, so confidence
    # alone must not turn it into full support coverage.
    assert _check(result=_evidence(bbox=[59.9, 59.9, 100.1, 100.1])) is None


def test_trusted_tuple_does_not_accept_string_vocabulary_substring():
    assert trusted_classification(("shadow", 0.99), "shadow-wrench") is None
