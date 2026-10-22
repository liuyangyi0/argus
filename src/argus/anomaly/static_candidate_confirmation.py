"""Lightweight confirmation for persistent-change candidates.

This module deliberately does not claim to be an anomaly model.  It measures
photometric and edge evidence around a candidate and rejects changes that are
well explained by a global illumination shift or that have no stable structural
evidence.  A caller may pass an external anomaly score (for example, a future
PatchCore ROI score), but that score is only used when an explicit threshold is
configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

BBox = tuple[int, int, int, int]


class ConfirmationReason(str, Enum):
    """Machine-readable reason for a confirmation decision."""

    CONFIRMED = "confirmed"
    CONFIRMED_BY_EXTERNAL_SCORE = "confirmed_by_external_score"
    INSUFFICIENT_LOCAL_CHANGE = "insufficient_local_change"
    GLOBAL_ILLUMINATION_CHANGE = "global_illumination_change"
    REFLECTION_LIKE_CHANGE = "reflection_like_change"
    INSUFFICIENT_STRUCTURE = "insufficient_structure"
    SENSITIVE_ROI_WEAK_STRUCTURE = "sensitive_roi_weak_structure"


@dataclass(frozen=True, slots=True)
class StaticCandidateConfirmationConfig:
    """Thresholds for :class:`StaticCandidateConfirmer`.

    All intensity thresholds are expressed on a normalized ``0..1`` scale.
    The defaults are conservative starting values and should be calibrated with
    the site's negative-video corpus before deployment.
    """

    min_bbox_size: int = 8
    context_padding_ratio: float = 0.15
    min_context_padding: int = 3
    min_local_mean_abs_difference: float = 0.025
    changed_pixel_threshold: float = 0.04
    global_illumination_min_delta: float = 0.025
    global_changed_fraction_threshold: float = 0.35
    max_global_model_residual: float = 0.025
    min_local_global_brightness_similarity: float = 0.70
    canny_low_threshold: int = 40
    canny_high_threshold: int = 100
    edge_tolerance_px: float = 2.0
    edge_change_scale: float = 0.12
    gradient_change_scale: float = 0.08
    min_structure_score: float = 0.30
    sensitive_min_structure_score: float = 0.55
    reflection_min_photometric_correlation: float = 0.92
    reflection_max_structure_score: float = 0.08
    sensitive_roi_overlap_threshold: float = 0.10
    external_anomaly_score_threshold: float | None = None


@dataclass(frozen=True, slots=True)
class StaticCandidateMetrics:
    """Observable evidence used by the confirmer."""

    local_mean_abs_difference: float
    local_changed_fraction: float
    local_brightness_delta: float
    global_brightness_delta: float
    local_global_brightness_difference: float
    local_global_brightness_similarity: float
    global_changed_fraction: float
    global_gain: float
    global_offset: float
    global_model_local_residual: float
    local_photometric_correlation: float
    edge_change_fraction: float
    new_edge_fraction: float
    gradient_change_mean: float
    structure_score: float
    external_anomaly_score: float | None
    external_score_supports_candidate: bool


@dataclass(frozen=True, slots=True)
class StaticCandidateDecision:
    """Structured result returned for one persistent-change candidate."""

    accepted: bool
    reason: ConfirmationReason
    bbox: BBox
    analysis_bbox: BBox
    in_sensitive_roi: bool
    required_structure_score: float
    metrics: StaticCandidateMetrics


class StaticCandidateConfirmer:
    """Confirm persistent-change candidates using explainable image evidence."""

    def __init__(self, config: StaticCandidateConfirmationConfig | None = None) -> None:
        self.config = config or StaticCandidateConfirmationConfig()

    def confirm(
        self,
        current_frame: NDArray[np.generic],
        reference_frame: NDArray[np.generic],
        bbox: BBox,
        *,
        sensitive_rois: Sequence[BBox] | None = None,
        external_anomaly_score: float | None = None,
    ) -> StaticCandidateDecision:
        """Evaluate one candidate.

        Args:
            current_frame: Current grayscale, BGR, or BGRA frame.
            reference_frame: Reference frame with the same spatial dimensions.
            bbox: Candidate ``(x1, y1, x2, y2)`` in image coordinates.
            sensitive_rois: Optional reflective/display/glass regions.  A candidate
                overlapping one of these regions must pass the stronger structural
                threshold.
            external_anomaly_score: Optional score supplied by another component.
                It has no effect unless ``external_anomaly_score_threshold`` is set.

        Raises:
            ValueError: If frames or the candidate box are invalid.
        """

        current = _to_gray_float(current_frame)
        reference = _to_gray_float(reference_frame)
        if current.shape != reference.shape:
            raise ValueError(
                "current_frame and reference_frame must have identical spatial dimensions"
            )

        height, width = current.shape
        candidate_bbox = _clip_bbox(bbox, width, height)
        x1, y1, x2, y2 = candidate_bbox
        if x2 - x1 < self.config.min_bbox_size or y2 - y1 < self.config.min_bbox_size:
            raise ValueError(
                f"candidate bbox must be at least {self.config.min_bbox_size}x"
                f"{self.config.min_bbox_size} pixels after clipping"
            )

        padding = max(
            self.config.min_context_padding,
            int(round(max(x2 - x1, y2 - y1) * self.config.context_padding_ratio)),
        )
        analysis_bbox = (
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(width, x2 + padding),
            min(height, y2 + padding),
        )

        current_local = current[y1:y2, x1:x2]
        reference_local = reference[y1:y2, x1:x2]
        local_difference = current_local - reference_local
        local_mean_abs_difference = float(np.mean(np.abs(local_difference)))
        local_changed_fraction = float(
            np.mean(np.abs(local_difference) >= self.config.changed_pixel_threshold)
        )
        local_brightness_delta = float(np.median(local_difference))

        global_mask = np.ones((height, width), dtype=bool)
        ax1, ay1, ax2, ay2 = analysis_bbox
        global_mask[ay1:ay2, ax1:ax2] = False
        reference_global = reference[global_mask]
        current_global = current[global_mask]
        if reference_global.size < 64:
            # Very large candidates still need a meaningful background estimate.
            reference_global = reference.reshape(-1)
            current_global = current.reshape(-1)

        global_delta_values = current_global - reference_global
        global_brightness_delta = float(np.median(global_delta_values))
        global_changed_fraction = float(
            np.mean(np.abs(global_delta_values) >= self.config.changed_pixel_threshold)
        )
        global_gain, global_offset = _fit_global_photometric_model(
            reference_global, current_global
        )
        predicted_local = np.clip(global_gain * reference_local + global_offset, 0.0, 1.0)
        predicted_local_delta = predicted_local - reference_local
        global_model_local_residual = float(np.mean(np.abs(current_local - predicted_local)))
        predicted_local_brightness_delta = float(np.median(predicted_local_delta))
        local_global_brightness_difference = abs(
            local_brightness_delta - predicted_local_brightness_delta
        )
        similarity_scale = max(
            self.config.max_global_model_residual,
            abs(predicted_local_brightness_delta) * 0.5,
            1e-6,
        )
        local_global_brightness_similarity = float(
            np.exp(-local_global_brightness_difference / similarity_scale)
        )
        local_photometric_correlation = _safe_correlation(
            reference_local.reshape(-1), current_local.reshape(-1)
        )

        current_analysis = current[ay1:ay2, ax1:ax2]
        reference_analysis = reference[ay1:ay2, ax1:ax2]
        edge_change_fraction, new_edge_fraction, gradient_change_mean, structure_score = (
            self._structure_metrics(reference_analysis, current_analysis)
        )

        in_sensitive_roi = _overlaps_sensitive_roi(
            candidate_bbox,
            sensitive_rois or (),
            width,
            height,
            self.config.sensitive_roi_overlap_threshold,
        )
        required_structure_score = (
            self.config.sensitive_min_structure_score
            if in_sensitive_roi
            else self.config.min_structure_score
        )
        external_support = bool(
            external_anomaly_score is not None
            and self.config.external_anomaly_score_threshold is not None
            and external_anomaly_score >= self.config.external_anomaly_score_threshold
        )

        metrics = StaticCandidateMetrics(
            local_mean_abs_difference=local_mean_abs_difference,
            local_changed_fraction=local_changed_fraction,
            local_brightness_delta=local_brightness_delta,
            global_brightness_delta=global_brightness_delta,
            local_global_brightness_difference=local_global_brightness_difference,
            local_global_brightness_similarity=local_global_brightness_similarity,
            global_changed_fraction=global_changed_fraction,
            global_gain=global_gain,
            global_offset=global_offset,
            global_model_local_residual=global_model_local_residual,
            local_photometric_correlation=local_photometric_correlation,
            edge_change_fraction=edge_change_fraction,
            new_edge_fraction=new_edge_fraction,
            gradient_change_mean=gradient_change_mean,
            structure_score=structure_score,
            external_anomaly_score=external_anomaly_score,
            external_score_supports_candidate=external_support,
        )

        if local_mean_abs_difference < self.config.min_local_mean_abs_difference:
            reason = ConfirmationReason.INSUFFICIENT_LOCAL_CHANGE
        elif self._is_global_illumination(metrics, required_structure_score):
            reason = ConfirmationReason.GLOBAL_ILLUMINATION_CHANGE
        elif in_sensitive_roi and structure_score < required_structure_score and external_support:
            reason = ConfirmationReason.CONFIRMED_BY_EXTERNAL_SCORE
        elif in_sensitive_roi and structure_score < required_structure_score:
            reason = ConfirmationReason.SENSITIVE_ROI_WEAK_STRUCTURE
        elif structure_score < required_structure_score and (
            local_photometric_correlation
            >= self.config.reflection_min_photometric_correlation
            or structure_score <= self.config.reflection_max_structure_score
        ):
            reason = ConfirmationReason.REFLECTION_LIKE_CHANGE
        elif structure_score < required_structure_score and not external_support:
            reason = ConfirmationReason.INSUFFICIENT_STRUCTURE
        elif structure_score < required_structure_score:
            reason = ConfirmationReason.CONFIRMED_BY_EXTERNAL_SCORE
        else:
            reason = ConfirmationReason.CONFIRMED

        return StaticCandidateDecision(
            accepted=reason
            in {ConfirmationReason.CONFIRMED, ConfirmationReason.CONFIRMED_BY_EXTERNAL_SCORE},
            reason=reason,
            bbox=candidate_bbox,
            analysis_bbox=analysis_bbox,
            in_sensitive_roi=in_sensitive_roi,
            required_structure_score=required_structure_score,
            metrics=metrics,
        )

    def _structure_metrics(
        self, reference: NDArray[np.float32], current: NDArray[np.float32]
    ) -> tuple[float, float, float, float]:
        reference_u8 = np.rint(reference * 255.0).astype(np.uint8)
        current_u8 = np.rint(current * 255.0).astype(np.uint8)
        reference_blur = cv2.GaussianBlur(reference_u8, (3, 3), 0)
        current_blur = cv2.GaussianBlur(current_u8, (3, 3), 0)

        reference_edges = cv2.Canny(
            reference_blur,
            self.config.canny_low_threshold,
            self.config.canny_high_threshold,
        ).astype(bool)
        current_edges = cv2.Canny(
            current_blur,
            self.config.canny_low_threshold,
            self.config.canny_high_threshold,
        ).astype(bool)
        # Exact XOR turns harmless 1px registration noise into two changed
        # edge bands.  Distance transforms give each edge a small spatial
        # tolerance while preserving genuinely new/removed structure.
        reference_distance = cv2.distanceTransform(
            (~reference_edges).astype(np.uint8),
            cv2.DIST_L2,
            3,
        )
        current_distance = cv2.distanceTransform(
            (~current_edges).astype(np.uint8),
            cv2.DIST_L2,
            3,
        )
        new_edges = current_edges & (
            reference_distance > self.config.edge_tolerance_px
        )
        removed_edges = reference_edges & (
            current_distance > self.config.edge_tolerance_px
        )
        edge_change_fraction = float(np.mean(new_edges | removed_edges))
        new_edge_fraction = float(np.mean(new_edges))

        reference_sobel_x = cv2.Sobel(reference, cv2.CV_32F, 1, 0, ksize=3)
        reference_sobel_y = cv2.Sobel(reference, cv2.CV_32F, 0, 1, ksize=3)
        current_sobel_x = cv2.Sobel(current, cv2.CV_32F, 1, 0, ksize=3)
        current_sobel_y = cv2.Sobel(current, cv2.CV_32F, 0, 1, ksize=3)
        reference_gradient = cv2.magnitude(reference_sobel_x, reference_sobel_y)
        current_gradient = cv2.magnitude(current_sobel_x, current_sobel_y)
        # A 3x3 Sobel response is at most approximately four times the input range.
        gradient_change_mean = float(
            np.mean(np.abs(current_gradient - reference_gradient)) / 4.0
        )

        edge_evidence = min(1.0, edge_change_fraction / self.config.edge_change_scale)
        gradient_evidence = min(
            1.0, gradient_change_mean / self.config.gradient_change_scale
        )
        structure_score = float(0.7 * edge_evidence + 0.3 * gradient_evidence)
        return (
            edge_change_fraction,
            new_edge_fraction,
            gradient_change_mean,
            structure_score,
        )

    def _is_global_illumination(
        self, metrics: StaticCandidateMetrics, required_structure_score: float
    ) -> bool:
        return bool(
            abs(metrics.global_brightness_delta)
            >= self.config.global_illumination_min_delta
            and metrics.global_changed_fraction
            >= self.config.global_changed_fraction_threshold
            and metrics.global_model_local_residual
            <= self.config.max_global_model_residual
            and metrics.local_global_brightness_similarity
            >= self.config.min_local_global_brightness_similarity
            and metrics.structure_score < required_structure_score
        )


def _to_gray_float(frame: NDArray[np.generic]) -> NDArray[np.float32]:
    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
        raise ValueError("frame must be a 2D grayscale or 3D BGR/BGRA numpy array")
    if frame.ndim == 3:
        if frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.shape[2] == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("3D frames must have 3 (BGR) or 4 (BGRA) channels")
    else:
        gray = frame

    if not np.all(np.isfinite(gray)):
        raise ValueError("frame contains non-finite values")
    result = gray.astype(np.float32)
    if np.issubdtype(gray.dtype, np.integer):
        result /= float(np.iinfo(gray.dtype).max)
    elif result.size and (float(result.min()) < 0.0 or float(result.max()) > 1.0):
        if float(result.min()) >= 0.0 and float(result.max()) <= 255.0:
            result /= 255.0
        else:
            raise ValueError("floating-point frames must use the 0..1 or 0..255 range")
    return np.clip(result, 0.0, 1.0)


def _clip_bbox(bbox: BBox, width: int, height: int) -> BBox:
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four coordinates")
    x1, y1, x2, y2 = (int(value) for value in bbox)
    clipped = (
        min(max(x1, 0), width),
        min(max(y1, 0), height),
        min(max(x2, 0), width),
        min(max(y2, 0), height),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("bbox does not contain a valid area inside the frame")
    return clipped


def _intersection_over_candidate(candidate: BBox, roi: BBox) -> float:
    x1 = max(candidate[0], roi[0])
    y1 = max(candidate[1], roi[1])
    x2 = min(candidate[2], roi[2])
    y2 = min(candidate[3], roi[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    candidate_area = (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
    return intersection / candidate_area


def _overlaps_sensitive_roi(
    candidate: BBox,
    rois: Sequence[BBox],
    width: int,
    height: int,
    threshold: float,
) -> bool:
    for roi in rois:
        if len(roi) != 4:
            raise ValueError("each sensitive ROI must contain exactly four coordinates")
        x1, y1, x2, y2 = (int(value) for value in roi)
        clipped = (
            min(max(x1, 0), width),
            min(max(y1, 0), height),
            min(max(x2, 0), width),
            min(max(y2, 0), height),
        )
        # An ROI completely outside the image simply cannot overlap the candidate.
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        if _intersection_over_candidate(candidate, clipped) >= threshold:
            return True
    return False


def _fit_global_photometric_model(
    reference: NDArray[np.float32], current: NDArray[np.float32]
) -> tuple[float, float]:
    """Fit a robust affine illumination model ``current ~= gain*reference+offset``."""

    if reference.size > 100_000:
        step = max(1, reference.size // 100_000)
        reference = reference[::step]
        current = current[::step]
    reference64 = reference.astype(np.float64, copy=False)
    current64 = current.astype(np.float64, copy=False)
    if float(np.std(reference64)) < 1e-4:
        return 1.0, float(np.median(current64 - reference64))

    design = np.column_stack((reference64, np.ones_like(reference64)))
    coefficients, *_ = np.linalg.lstsq(design, current64, rcond=None)
    prediction = design @ coefficients
    residual = current64 - prediction
    residual_median = float(np.median(residual))
    residual_mad = float(np.median(np.abs(residual - residual_median)))
    if residual_mad > 1e-6:
        inliers = np.abs(residual - residual_median) <= 3.5 * 1.4826 * residual_mad
        if int(np.count_nonzero(inliers)) >= 64:
            coefficients, *_ = np.linalg.lstsq(design[inliers], current64[inliers], rcond=None)

    gain = float(np.clip(coefficients[0], 0.5, 1.5))
    offset = float(np.clip(coefficients[1], -0.5, 0.5))
    return gain, offset


def _safe_correlation(first: NDArray[np.float32], second: NDArray[np.float32]) -> float:
    first_std = float(np.std(first))
    second_std = float(np.std(second))
    if first_std < 1e-6 or second_std < 1e-6:
        if first_std < 1e-6 and second_std < 1e-6:
            return 1.0
        return 0.0
    correlation = float(np.corrcoef(first, second)[0, 1])
    return float(np.clip(correlation, -1.0, 1.0))
