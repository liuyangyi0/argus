"""Calibrated model evidence and spatial temporal confirmation for candidates.

The legacy ``AnomalyResult.anomaly_map`` is normalized independently for every
frame and remains useful for display, but it cannot support an absolute ROI
threshold.  This module consumes only the separate, comparable model-output map
and makes every unavailable/uncalibrated state explicit.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

BBox = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class CandidateModelThresholds:
    """Thresholds learned for one exact model/profile/output-space binding."""

    global_score_min: float | None = None
    roi_p95_min: float | None = None
    roi_context_delta_min: float | None = None

    @property
    def complete(self) -> bool:
        return all(
            value is not None
            for value in (
                self.global_score_min,
                self.roi_p95_min,
                self.roi_context_delta_min,
            )
        )


@dataclass(frozen=True, slots=True)
class CandidateModelEvidence:
    """Serializable scalar evidence for one candidate.

    ``would_accept`` is deliberately nullable.  ``False`` means the rule was
    actually evaluated and rejected the candidate; unavailable, skipped and
    uncalibrated observations use ``None`` and must not enter reject/FN counts.
    """

    status: str
    status_reason: str
    would_accept: bool | None
    calibration_id: str | None
    expected_calibration_id: str | None
    output_space: str | None
    output_comparable: bool
    map_shape: tuple[int, int] | None
    model_score: float | None
    model_input_variant: str | None = None
    roi_p95: float | None = None
    context_ring_p95: float | None = None
    roi_context_delta: float | None = None
    roi_hot_fraction: float | None = None
    roi_energy_fraction: float | None = None
    hot_pixel_threshold: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PreparedModelEvidenceMap:
    """Frame-level validation and aggregates shared by all candidate ROIs."""

    status: str
    status_reason: str
    values: np.ndarray | None
    map_shape: tuple[int, int] | None
    model_score: float | None
    output_space: str | None
    output_comparable: bool
    positive_energy: float | None


def prepare_model_evidence_map(
    *,
    model_anomaly_map: np.ndarray | None,
    model_score: float | None,
    output_space: str | None,
    output_comparable: bool,
    detection_failed: bool = False,
) -> PreparedModelEvidenceMap:
    """Validate one model map once before evaluating multiple candidates."""
    finite_score = _finite_float_or_none(model_score)
    common = {
        "values": None,
        "map_shape": None,
        "model_score": finite_score,
        "output_space": output_space,
        "output_comparable": bool(output_comparable),
        "positive_energy": None,
    }
    if detection_failed:
        return PreparedModelEvidenceMap(
            status="inference_error",
            status_reason="detector_result_failed",
            **common,
        )
    if model_anomaly_map is None:
        return PreparedModelEvidenceMap(
            status="unavailable",
            status_reason="model_map_missing",
            **common,
        )
    values = np.asarray(model_anomaly_map).squeeze()
    if values.ndim != 2 or values.size == 0:
        return PreparedModelEvidenceMap(
            status="unavailable",
            status_reason="model_map_invalid_shape",
            **common,
        )
    map_shape = (int(values.shape[0]), int(values.shape[1]))
    if not np.all(np.isfinite(values)):
        return PreparedModelEvidenceMap(
            status="inference_error",
            status_reason="model_map_non_finite",
            values=None,
            map_shape=map_shape,
            model_score=finite_score,
            output_space=output_space,
            output_comparable=bool(output_comparable),
            positive_energy=None,
        )
    if not output_comparable or not output_space:
        return PreparedModelEvidenceMap(
            status="unavailable",
            status_reason="model_output_not_comparable",
            values=None,
            map_shape=map_shape,
            model_score=finite_score,
            output_space=output_space,
            output_comparable=bool(output_comparable),
            positive_energy=None,
        )
    if finite_score is None:
        return PreparedModelEvidenceMap(
            status="inference_error",
            status_reason="model_score_non_finite_or_missing",
            values=None,
            map_shape=map_shape,
            model_score=None,
            output_space=output_space,
            output_comparable=True,
            positive_energy=None,
        )
    positive_energy = float(
        np.maximum(values, 0.0).sum(dtype=np.float64)
    )
    return PreparedModelEvidenceMap(
        status="ready",
        status_reason="model_map_ready",
        values=values,
        map_shape=map_shape,
        model_score=finite_score,
        output_space=output_space,
        output_comparable=True,
        positive_energy=positive_energy,
    )


def build_calibration_id(binding: Mapping[str, object]) -> str:
    """Build an opaque, deterministic ID for a complete runtime score space."""

    canonical = json.dumps(
        dict(binding),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return f"model-evidence-v1:{digest}"


def evaluate_candidate_model_evidence(
    *,
    model_anomaly_map: np.ndarray | None,
    bbox: BBox,
    frame_shape: tuple[int, ...],
    model_score: float | None,
    output_space: str | None,
    output_comparable: bool,
    calibration_id: str | None,
    expected_calibration_id: str | None,
    thresholds: CandidateModelThresholds,
    context_ratio: float = 0.5,
    hot_pixel_threshold: float = 0.5,
    detection_failed: bool = False,
    prepared_map: PreparedModelEvidenceMap | None = None,
    model_input_variant: str | None = None,
) -> CandidateModelEvidence:
    """Extract model-scale candidate evidence without resizing the heatmap.

    Candidate coordinates are mapped directly into model-map coordinates.  The
    context statistic is measured in a padded ring excluding the candidate.
    All values must be finite; silently replacing NaN/Inf with zero would turn
    inference corruption into a model rejection and poison calibration data.
    """

    prepared = prepared_map or prepare_model_evidence_map(
        model_anomaly_map=model_anomaly_map,
        model_score=model_score,
        output_space=output_space,
        output_comparable=output_comparable,
        detection_failed=detection_failed,
    )
    common = {
        "calibration_id": calibration_id,
        "expected_calibration_id": expected_calibration_id,
        "output_space": prepared.output_space,
        "output_comparable": prepared.output_comparable,
        "map_shape": prepared.map_shape,
        "model_score": prepared.model_score,
        "model_input_variant": model_input_variant,
        "hot_pixel_threshold": _finite_float_or_none(hot_pixel_threshold),
    }
    if prepared.status != "ready" or prepared.values is None:
        return CandidateModelEvidence(
            status=prepared.status,
            status_reason=prepared.status_reason,
            would_accept=None,
            **common,
        )
    values = prepared.values

    mapped = _map_bbox_to_array(bbox, frame_shape, values.shape)
    if mapped is None:
        return CandidateModelEvidence(
            status="unavailable",
            status_reason="candidate_bbox_invalid",
            would_accept=None,
            **common,
        )
    x1, y1, x2, y2 = mapped
    roi = values[y1:y2, x1:x2].astype(np.float64, copy=False)
    if roi.size == 0:
        return CandidateModelEvidence(
            status="unavailable",
            status_reason="candidate_roi_empty",
            would_accept=None,
            **common,
        )

    pad_x = max(1, int(math.ceil((x2 - x1) * float(context_ratio))))
    pad_y = max(1, int(math.ceil((y2 - y1) * float(context_ratio))))
    ax1 = max(0, x1 - pad_x)
    ay1 = max(0, y1 - pad_y)
    ax2 = min(values.shape[1], x2 + pad_x)
    ay2 = min(values.shape[0], y2 + pad_y)
    analysis = values[ay1:ay2, ax1:ax2].astype(np.float64, copy=False)
    ring_mask = np.ones(analysis.shape, dtype=bool)
    ring_mask[y1 - ay1 : y2 - ay1, x1 - ax1 : x2 - ax1] = False
    ring = analysis[ring_mask]
    if ring.size == 0:
        return CandidateModelEvidence(
            status="unavailable",
            status_reason="context_ring_empty",
            would_accept=None,
            **common,
        )

    roi_p95 = float(np.percentile(roi, 95))
    ring_p95 = float(np.percentile(ring, 95))
    delta = roi_p95 - ring_p95
    hot_fraction = float(np.mean(roi >= hot_pixel_threshold))
    total_energy = float(prepared.positive_energy or 0.0)
    roi_energy = float(np.sum(np.maximum(roi, 0.0)))
    energy_fraction = roi_energy / total_energy if total_energy > 0 else 0.0

    metrics = {
        "roi_p95": roi_p95,
        "context_ring_p95": ring_p95,
        "roi_context_delta": delta,
        "roi_hot_fraction": hot_fraction,
        "roi_energy_fraction": energy_fraction,
    }
    if calibration_id is None:
        return CandidateModelEvidence(
            status="uncalibrated",
            status_reason="runtime_calibration_binding_incomplete",
            would_accept=None,
            **common,
            **metrics,
        )
    if expected_calibration_id is None or not thresholds.complete:
        return CandidateModelEvidence(
            status="uncalibrated",
            status_reason="thresholds_or_calibration_missing",
            would_accept=None,
            **common,
            **metrics,
        )
    if calibration_id != expected_calibration_id:
        return CandidateModelEvidence(
            status="uncalibrated",
            status_reason="calibration_binding_mismatch",
            would_accept=None,
            **common,
            **metrics,
        )

    passed = bool(
        common["model_score"] >= thresholds.global_score_min  # type: ignore[operator]
        and roi_p95 >= thresholds.roi_p95_min  # type: ignore[operator]
        and delta >= thresholds.roi_context_delta_min  # type: ignore[operator]
    )
    return CandidateModelEvidence(
        status="evaluated",
        status_reason="thresholds_passed" if passed else "thresholds_failed",
        would_accept=passed,
        **common,
        **metrics,
    )


@dataclass(frozen=True, slots=True)
class SpatialTemporalDecision:
    """Outcome of the independent same-location N-of-N confirmation gate."""

    status: str
    status_reason: str
    would_accept: bool | None
    confirmed: bool
    streak: int
    matched_iou: float | None
    gap_seconds: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class _SpatialState:
    source_track_id: int
    bbox: BBox
    streak: int
    last_qualified_at: float
    last_observed_at: float
    last_observation_number: int
    confirmed: bool = False


class SpatialTemporalConfirmer:
    """Require consecutive qualifying observations at the same scene location."""

    def __init__(
        self,
        *,
        required_passes: int = 2,
        min_gap_seconds: float = 0.4,
        max_gap_seconds: float = 2.0,
        iou_threshold: float = 0.5,
    ) -> None:
        if required_passes < 2:
            raise ValueError("required_passes must be at least 2")
        if min_gap_seconds < 0 or max_gap_seconds <= 0:
            raise ValueError("temporal gaps must be positive")
        if min_gap_seconds > max_gap_seconds:
            raise ValueError("min_gap_seconds cannot exceed max_gap_seconds")
        if not 0 < iou_threshold <= 1:
            raise ValueError("iou_threshold must be in (0, 1]")
        self.required_passes = int(required_passes)
        self.min_gap_seconds = float(min_gap_seconds)
        self.max_gap_seconds = float(max_gap_seconds)
        self.iou_threshold = float(iou_threshold)
        self._states: dict[int, _SpatialState] = {}
        self._lock = threading.RLock()

    def reset(self) -> None:
        with self._lock:
            self._states.clear()

    def observe(
        self,
        *,
        source_track_id: int,
        bbox: BBox,
        now: float,
        observation_number: int,
        evaluated: bool,
        passed: bool | None,
        status: str,
        observed_this_frame: bool = True,
    ) -> SpatialTemporalDecision:
        """Consume one observation; missing/skipped observations fail closed."""

        if not math.isfinite(now):
            raise ValueError("now must be finite")
        with self._lock:
            key, state, matched_iou = self._match(
                source_track_id,
                bbox,
                observation_number=observation_number,
            )
            if not observed_this_frame or not evaluated or passed is None:
                if key is not None:
                    self._states.pop(key, None)
                return SpatialTemporalDecision(
                    status=status,
                    status_reason=(
                        "candidate_not_observed_in_current_frame"
                        if not observed_this_frame
                        else "model_evidence_not_evaluated"
                    ),
                    would_accept=None,
                    confirmed=False,
                    streak=0,
                    matched_iou=matched_iou,
                    gap_seconds=None,
                )

            if (
                state is not None
                and observation_number != state.last_observation_number + 1
            ):
                self._states.pop(key, None)
                key, state, matched_iou = None, None, None

            if passed is False:
                if key is not None:
                    self._states.pop(key, None)
                return SpatialTemporalDecision(
                    status="evaluated",
                    status_reason="single_pass_failed",
                    would_accept=False,
                    confirmed=False,
                    streak=0,
                    matched_iou=matched_iou,
                    gap_seconds=None,
                )

            if state is None:
                self._states[source_track_id] = _SpatialState(
                    source_track_id=source_track_id,
                    bbox=bbox,
                    streak=1,
                    last_qualified_at=now,
                    last_observed_at=now,
                    last_observation_number=observation_number,
                )
                return SpatialTemporalDecision(
                    status="evaluated",
                    status_reason="first_qualified_observation",
                    would_accept=False,
                    confirmed=False,
                    streak=1,
                    matched_iou=None,
                    gap_seconds=None,
                )

            gap = now - state.last_qualified_at
            if gap < -1e-9 or gap > self.max_gap_seconds + 1e-9:
                self._states.pop(key, None)
                self._states[source_track_id] = _SpatialState(
                    source_track_id=source_track_id,
                    bbox=bbox,
                    streak=1,
                    last_qualified_at=now,
                    last_observed_at=now,
                    last_observation_number=observation_number,
                )
                return SpatialTemporalDecision(
                    status="evaluated",
                    status_reason="temporal_window_restarted",
                    would_accept=False,
                    confirmed=False,
                    streak=1,
                    matched_iou=matched_iou,
                    gap_seconds=gap,
                )

            if state.confirmed:
                state.bbox = bbox
                state.last_qualified_at = now
                state.last_observed_at = now
                state.last_observation_number = observation_number
                if key != source_track_id:
                    self._states.pop(key, None)
                    state.source_track_id = source_track_id
                    self._states[source_track_id] = state
                return SpatialTemporalDecision(
                    status="evaluated",
                    status_reason="temporal_confirmation_latched",
                    would_accept=True,
                    confirmed=True,
                    streak=state.streak,
                    matched_iou=matched_iou,
                    gap_seconds=gap,
                )

            state.bbox = bbox
            state.last_observed_at = now
            state.last_observation_number = observation_number
            if gap + 1e-9 < self.min_gap_seconds:
                return SpatialTemporalDecision(
                    status="evaluated",
                    status_reason="minimum_gap_pending",
                    would_accept=False,
                    confirmed=False,
                    streak=state.streak,
                    matched_iou=matched_iou,
                    gap_seconds=gap,
                )

            state.streak += 1
            state.last_qualified_at = now
            state.confirmed = state.streak >= self.required_passes
            if key != source_track_id:
                self._states.pop(key, None)
                state.source_track_id = source_track_id
                self._states[source_track_id] = state
            return SpatialTemporalDecision(
                status="evaluated",
                status_reason=(
                    "temporal_confirmed" if state.confirmed else "additional_pass_required"
                ),
                would_accept=state.confirmed,
                confirmed=state.confirmed,
                streak=state.streak,
                matched_iou=matched_iou,
                gap_seconds=gap,
            )

    def finish_frame(
        self,
        *,
        observed_track_ids: set[int],
        observation_number: int,
    ) -> None:
        """Drop states not represented by a current, evaluated candidate."""

        with self._lock:
            for key, state in list(self._states.items()):
                if (
                    state.source_track_id not in observed_track_ids
                    or state.last_observation_number != observation_number
                ):
                    self._states.pop(key, None)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "source_track_id": state.source_track_id,
                    "bbox": list(state.bbox),
                    "streak": state.streak,
                    "last_qualified_at": state.last_qualified_at,
                    "last_observed_at": state.last_observed_at,
                    "last_observation_number": state.last_observation_number,
                    "confirmed": state.confirmed,
                }
                for state in self._states.values()
            ]

    def _match(
        self,
        source_track_id: int,
        bbox: BBox,
        *,
        observation_number: int,
    ) -> tuple[int | None, _SpatialState | None, float | None]:
        exact = self._states.get(source_track_id)
        if (
            exact is not None
            and exact.last_observation_number != observation_number
        ):
            overlap = _bbox_iou(exact.bbox, bbox)
            if overlap >= self.iou_threshold:
                return source_track_id, exact, overlap

        best: tuple[float, int, _SpatialState] | None = None
        for key, state in self._states.items():
            if state.last_observation_number == observation_number:
                continue
            overlap = _bbox_iou(state.bbox, bbox)
            if overlap < self.iou_threshold:
                continue
            candidate = (overlap, -key, state)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None:
            return None, None, None
        overlap, negative_key, state = best
        return -negative_key, state, overlap


def _map_bbox_to_array(
    bbox: BBox,
    frame_shape: tuple[int, ...],
    map_shape: tuple[int, int],
) -> BBox | None:
    if len(frame_shape) < 2:
        return None
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    map_h, map_w = int(map_shape[0]), int(map_shape[1])
    if frame_h <= 0 or frame_w <= 0 or map_h <= 0 or map_w <= 0:
        return None
    x1, y1, x2, y2 = (int(value) for value in bbox)
    x1 = min(max(x1, 0), frame_w)
    x2 = min(max(x2, 0), frame_w)
    y1 = min(max(y1, 0), frame_h)
    y2 = min(max(y2, 0), frame_h)
    if x2 <= x1 or y2 <= y1:
        return None
    mx1 = min(map_w - 1, max(0, int(math.floor(x1 * map_w / frame_w))))
    my1 = min(map_h - 1, max(0, int(math.floor(y1 * map_h / frame_h))))
    mx2 = min(map_w, max(mx1 + 1, int(math.ceil(x2 * map_w / frame_w))))
    my2 = min(map_h, max(my1 + 1, int(math.ceil(y2 * map_h / frame_h))))
    return mx1, my1, mx2, my2


def _bbox_iou(first: BBox, second: BBox) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = float((x2 - x1) * (y2 - y1))
    first_area = float(max(0, first[2] - first[0]) * max(0, first[3] - first[1]))
    second_area = float(max(0, second[2] - second[0]) * max(0, second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _finite_float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


__all__ = [
    "CandidateModelEvidence",
    "CandidateModelThresholds",
    "PreparedModelEvidenceMap",
    "SpatialTemporalConfirmer",
    "SpatialTemporalDecision",
    "build_calibration_id",
    "evaluate_candidate_model_evidence",
    "prepare_model_evidence_map",
]
