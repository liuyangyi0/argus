"""Tests for fixed-scene persistent change scoring."""

from __future__ import annotations

import cv2
import numpy as np

from argus.anomaly.persistent_change import PersistentChangeScorer


def _base_frame() -> np.ndarray:
    return np.full((120, 160, 3), 80, dtype=np.uint8)


def test_persistent_change_requires_reference_and_persistence():
    scorer = PersistentChangeScorer(
        process_width=160,
        reference_frames=2,
        diff_threshold=20,
        min_area_px=100,
        min_persistence_seconds=0.25,
        min_persistence_frames=2,
        learning_alpha=0.0,
        alert_score=0.82,
    )
    base = _base_frame()

    warmup = scorer.process(base, timestamp=1.0)
    assert warmup.ready is False
    assert warmup.suppression_reason == "reference_warmup"

    ready = scorer.process(base, timestamp=1.1)
    assert ready.ready is False
    assert ready.warmup_remaining == 0

    changed = base.copy()
    changed[35:65, 50:90] = 230

    pending = scorer.process(changed, timestamp=2.0)
    assert pending.ready is True
    assert pending.has_detection is False
    assert pending.suppression_reason == "persistence_not_met"
    assert pending.active_tracks[0].seen_frames == 1

    detected = scorer.process(changed, timestamp=2.4)
    assert detected.has_detection is True
    assert detected.score >= 0.82
    assert detected.anomaly_map is not None
    assert detected.candidates[0].bbox[0] <= 50
    assert detected.candidates[0].bbox[1] <= 35
    assert detected.candidates[0].to_detected_object()["class_name"] == "static_foreign"


def test_persistent_change_suppresses_global_frame_changes():
    scorer = PersistentChangeScorer(
        process_width=160,
        reference_frames=1,
        diff_threshold=20,
        min_area_px=100,
        max_change_fraction=0.05,
        learning_alpha=0.0,
    )
    base = _base_frame()
    scorer.set_reference(base)

    changed = np.full_like(base, 220)
    result = scorer.process(changed, timestamp=2.0)

    assert result.ready is True
    assert result.has_detection is False
    assert result.suppression_reason == "global_change"
    assert result.changed_fraction > 0.05


def test_candidate_retains_exact_component_support_not_filled_bbox():
    scorer = PersistentChangeScorer(
        process_width=100,
        reference_frames=1,
        diff_threshold=20,
        min_area_px=20,
        max_area_fraction=0.95,
        max_change_fraction=0.95,
        min_persistence_seconds=0.0,
        min_persistence_frames=1,
        learning_alpha=0.0,
        morph_kernel_size=1,
    )
    base = np.zeros((100, 100, 3), dtype=np.uint8)
    changed = base.copy()
    cv2.line(changed, (3, 3), (96, 96), (255, 255, 255), thickness=2)
    scorer.set_reference(base)

    result = scorer.process(changed, timestamp=1.0)

    assert result.candidates
    candidate = result.candidates[0]
    assert candidate.support_mask is not None
    assert candidate.support_mask.shape == (
        candidate.bbox[3],
        candidate.bbox[2],
    )
    # The diagonal component occupies only a small part of its large bbox;
    # attribution must retain that distinction.
    assert float(candidate.support_mask.mean()) < 0.20
