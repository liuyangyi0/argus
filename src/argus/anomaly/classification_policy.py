"""Validate optional classification evidence before it can affect an alert.

This is a control-plane evidence check, not a detector or an alternative safety
gate. Uncertain/malformed evidence must leave the primary decision unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from numbers import Real

import cv2
import numpy as np


def trusted_classification(
    result: object,
    vocabulary: Collection[str],
    min_confidence: float = 0.9,
) -> tuple[str, float] | None:
    """Return finite, in-vocabulary, high-confidence evidence or no evidence."""
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        return None
    label, confidence = result
    if not isinstance(label, str) or not label:
        return None
    if not isinstance(vocabulary, Collection) or isinstance(vocabulary, (str, bytes)):
        return None
    if label not in vocabulary:
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        return None
    if isinstance(min_confidence, bool) or not isinstance(min_confidence, Real):
        return None
    try:
        threshold = float(min_confidence)
        value = float(confidence)
    except (OverflowError, ValueError):
        return None
    # A malformed runtime configuration cannot lower the conservative floor.
    if not math.isfinite(threshold) or not 0.9 <= threshold <= 1.0:
        return None
    if not math.isfinite(value) or not threshold <= value <= 1.0:
        return None
    return label, value


def _finite_coordinates(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
        return None
    try:
        numbers = tuple(float(item) for item in value)
    except (OverflowError, ValueError):
        return None
    if not all(math.isfinite(item) for item in numbers):
        return None
    return numbers


def localized_classification(
    result: object,
    vocabulary: Collection[str],
    anomaly_map: object,
    frame_shape: object,
    crop_bbox: object,
    min_confidence: float = 0.9,
) -> tuple[str, float] | None:
    """Trust classification only for the same uniquely localized anomaly.

    ``result.bbox`` is full-frame ``[x1, y1, x2, y2]`` evidence from the SDK;
    ``crop_bbox`` is the requested full-frame ``(x, y, width, height)``. A label
    alone or a full-frame request must not suppress/downgrade an alert.

    Heatmaps are normalized alarm maps, not native model-evidence outputs.
    Require an absolute peak of 0.5 and contrast of 0.1 before the normalized
    half-peak support has localization meaning. The detection must cover at
    least 90% of the one support component, have at least 90% support inside its
    own mapped box, and place its center inside that support. Uncertain
    geometry leaves the original alert decision untouched.
    """
    if not isinstance(result, dict):
        return None
    trusted = trusted_classification(
        (result.get("label"), result.get("confidence")), vocabulary, min_confidence,
    )
    if trusted is None:
        return None
    if not isinstance(frame_shape, (tuple, list)) or len(frame_shape) < 2:
        return None
    dimensions = frame_shape[:2]
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in dimensions):
        return None
    try:
        frame_h, frame_w = (float(item) for item in dimensions)
    except (OverflowError, ValueError):
        return None
    if not all(
        math.isfinite(item) and 1 <= item <= 2**31 - 1 and item.is_integer()
        for item in (frame_h, frame_w)
    ):
        return None
    crop = _finite_coordinates(crop_bbox)
    detection = _finite_coordinates(result.get("bbox"))
    if crop is None or detection is None:
        return None
    crop_x, crop_y, crop_w, crop_h = crop
    if not (0 <= crop_x < frame_w and 0 <= crop_y < frame_h and crop_w > 0 and crop_h > 0):
        return None
    if crop_x + crop_w > frame_w or crop_y + crop_h > frame_h:
        return None
    x1, y1, x2, y2 = detection
    if not (crop_x <= x1 < x2 <= crop_x + crop_w and crop_y <= y1 < y2 <= crop_y + crop_h):
        return None

    if not isinstance(anomaly_map, np.ndarray) or anomaly_map.ndim != 2 or anomaly_map.size == 0:
        return None
    if not np.issubdtype(anomaly_map.dtype, np.number) or np.iscomplexobj(anomaly_map):
        return None
    if not np.isfinite(anomaly_map).all():
        return None
    low, peak = float(anomaly_map.min()), float(anomaly_map.max())
    contrast = peak - low
    if low < 0 or peak > 1 or peak < 0.5 or contrast < 0.1:
        return None
    support = (anomaly_map > low + contrast * 0.5).astype(np.uint8)
    component_count, _ = cv2.connectedComponents(support, connectivity=8)
    if component_count != 2:  # exactly one foreground component plus background
        return None

    map_h, map_w = support.shape
    left = math.floor(x1 * map_w / frame_w)
    right = math.ceil(x2 * map_w / frame_w)
    top = math.floor(y1 * map_h / frame_h)
    bottom = math.ceil(y2 * map_h / frame_h)
    box_support = support[top:bottom, left:right]
    support_count = int(np.count_nonzero(support))
    covered = int(np.count_nonzero(box_support))
    if support_count == 0 or box_support.size == 0:
        return None
    if covered / support_count < 0.9 or covered / box_support.size < 0.9:
        return None
    center_x = math.floor((x1 + x2) * 0.5 * map_w / frame_w)
    center_y = math.floor((y1 + y2) * 0.5 * map_h / frame_h)
    if not support[center_y, center_x]:
        return None
    return trusted
