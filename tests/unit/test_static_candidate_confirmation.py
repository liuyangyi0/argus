from __future__ import annotations

import cv2
import numpy as np
import pytest

from argus.anomaly.static_candidate_confirmation import (
    ConfirmationReason,
    StaticCandidateConfirmationConfig,
    StaticCandidateConfirmer,
)


def _textured_reference(size: int = 128) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    image = 65 + ((x // 16 + y // 16) % 2) * 22
    return np.repeat(image[:, :, None].astype(np.uint8), 3, axis=2)


def test_confirms_candidate_with_new_structural_edges() -> None:
    reference = _textured_reference()
    current = reference.copy()
    cv2.rectangle(current, (42, 42), (85, 85), (220, 220, 220), -1)
    cv2.line(current, (43, 43), (84, 84), (20, 20, 20), 3)

    decision = StaticCandidateConfirmer().confirm(
        current, reference, (40, 40, 88, 88)
    )

    assert decision.accepted
    assert decision.reason is ConfirmationReason.CONFIRMED
    assert decision.metrics.structure_score >= decision.required_structure_score
    assert decision.metrics.new_edge_fraction > 0


def test_rejects_uniform_global_illumination_change() -> None:
    reference = _textured_reference()
    current = np.clip(reference.astype(np.int16) + 35, 0, 255).astype(np.uint8)

    decision = StaticCandidateConfirmer().confirm(
        current, reference, (36, 36, 92, 92)
    )

    assert not decision.accepted
    assert decision.reason is ConfirmationReason.GLOBAL_ILLUMINATION_CHANGE
    assert decision.metrics.global_changed_fraction > 0.9
    assert decision.metrics.local_global_brightness_similarity > 0.9


def test_rejects_soft_reflection_like_change_without_edges() -> None:
    reference = np.full((128, 128, 3), 75, dtype=np.uint8)
    current = reference.copy()
    yy, xx = np.mgrid[:128, :128]
    glow = 55.0 * np.exp(-(((xx - 64) / 24) ** 2 + ((yy - 64) / 20) ** 2))
    current = np.clip(current.astype(np.float32) + glow[:, :, None], 0, 255).astype(np.uint8)

    decision = StaticCandidateConfirmer().confirm(
        current, reference, (40, 40, 88, 88)
    )

    assert not decision.accepted
    assert decision.reason is ConfirmationReason.REFLECTION_LIKE_CHANGE
    assert decision.metrics.local_mean_abs_difference > 0.025
    assert decision.metrics.structure_score < decision.required_structure_score


def test_sensitive_roi_requires_stronger_structural_evidence() -> None:
    reference = _textured_reference()
    current = reference.copy()
    # A single new line provides modest evidence: enough for an ordinary ROI but
    # intentionally below the stronger glass/display threshold.
    cv2.line(current, (42, 64), (86, 64), (230, 230, 230), 5)
    config = StaticCandidateConfirmationConfig(min_local_mean_abs_difference=0.005)
    confirmer = StaticCandidateConfirmer(config)

    ordinary = confirmer.confirm(current, reference, (40, 40, 88, 88))
    sensitive = confirmer.confirm(
        current,
        reference,
        (40, 40, 88, 88),
        sensitive_rois=[(32, 32, 96, 96)],
    )

    assert ordinary.accepted
    assert not sensitive.accepted
    assert sensitive.in_sensitive_roi
    assert sensitive.reason is ConfirmationReason.SENSITIVE_ROI_WEAK_STRUCTURE
    assert ordinary.required_structure_score < sensitive.required_structure_score


def test_external_score_is_inert_until_explicitly_configured() -> None:
    reference = np.full((96, 96, 3), 80, dtype=np.uint8)
    current = reference.copy().astype(np.int16)
    yy, xx = np.mgrid[:96, :96]
    low_contrast_texture = 10 + (((xx // 4 + yy // 4) % 2) * 2 - 1) * 8
    current[25:71, 25:71] += low_contrast_texture[25:71, 25:71, None]
    current = np.clip(current, 0, 255).astype(np.uint8)

    default_decision = StaticCandidateConfirmer().confirm(
        current,
        reference,
        (25, 25, 71, 71),
        external_anomaly_score=0.99,
    )
    configured_decision = StaticCandidateConfirmer(
        StaticCandidateConfirmationConfig(external_anomaly_score_threshold=0.8)
    ).confirm(
        current,
        reference,
        (25, 25, 71, 71),
        external_anomaly_score=0.99,
    )

    assert not default_decision.accepted
    assert not default_decision.metrics.external_score_supports_candidate
    assert configured_decision.accepted
    assert configured_decision.reason is ConfirmationReason.CONFIRMED_BY_EXTERNAL_SCORE


def test_local_external_score_can_rescue_sensitive_roi_weak_structure() -> None:
    reference = np.full((96, 96, 3), 80, dtype=np.uint8)
    current = reference.copy()
    current[30:66, 30:66] = 110
    confirmer = StaticCandidateConfirmer(
        StaticCandidateConfirmationConfig(external_anomaly_score_threshold=0.8)
    )

    decision = confirmer.confirm(
        current,
        reference,
        (30, 30, 66, 66),
        sensitive_rois=[(20, 20, 76, 76)],
        external_anomaly_score=0.99,
    )

    assert decision.accepted
    assert decision.reason is ConfirmationReason.CONFIRMED_BY_EXTERNAL_SCORE


@pytest.mark.parametrize(
    ("current_shape", "reference_shape", "bbox", "message"),
    [
        ((64, 64, 3), (63, 64, 3), (10, 10, 30, 30), "identical"),
        ((64, 64, 3), (64, 64, 3), (10, 10, 14, 14), "at least"),
        ((64, 64, 3), (64, 64, 3), (70, 70, 80, 80), "valid area"),
    ],
)
def test_validates_inputs(
    current_shape: tuple[int, ...],
    reference_shape: tuple[int, ...],
    bbox: tuple[int, int, int, int],
    message: str,
) -> None:
    current = np.zeros(current_shape, dtype=np.uint8)
    reference = np.zeros(reference_shape, dtype=np.uint8)

    with pytest.raises(ValueError, match=message):
        StaticCandidateConfirmer().confirm(current, reference, bbox)
