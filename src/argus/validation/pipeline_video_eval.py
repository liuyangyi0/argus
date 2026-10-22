"""Offline video replay through the production ``DetectionPipeline``.

Unlike :mod:`argus.validation.video_eval`, this evaluator records final alerts
after pre-filtering, person filtering, persistent-change detection, temporal
grading, and suppression. A virtual clock advances with video timestamps so
time-based pipeline behavior can be tested without replaying in real time.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import cv2
import numpy as np

from argus.capture.camera import CameraState, FrameData
from argus.config.loader import load_config
from argus.core.pipeline import (
    DetectionPipeline,
    PipelineMode,
    _frame_content_sha256,
)
from argus.validation.video_eval import load_video_annotations


PIPELINE_VIDEO_EVAL_SCHEMA_VERSION = "argus.pipeline-video-eval.v1"


def _stream_sha256(path: Path) -> str:
    """Hash an artifact without loading a potentially large video into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_artifact_sha256_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    from argus.storage.release_evidence import evidence_artifact_sha256

    return evidence_artifact_sha256(path)


def _annotation_inventory(
    raw: Any,
    *,
    fps: float,
    parsed: dict[str, Any],
    clipped: dict[str, Any],
    total_frames: int,
) -> dict[str, Any]:
    """Audit raw annotation bounds before the permissive loader normalizes them."""
    raw_counts = {"positive": 0, "ignore": 0}
    invalid_ranges = 0
    out_of_bounds_ranges = 0
    loader_normalized_ranges = 0
    container_valid = isinstance(raw, dict)

    def raw_items(keys: tuple[str, ...]) -> list[Any]:
        nonlocal container_valid
        items: list[Any] = []
        if not isinstance(raw, dict):
            return items
        for key in keys:
            if key not in raw:
                continue
            value = raw[key]
            if not isinstance(value, list):
                container_valid = False
                continue
            items.extend(value)
        return items

    def coordinate(item: dict[str, Any], prefix: str) -> int | None:
        frame_key = f"{prefix}_frame"
        sec_key = f"{prefix}_sec"
        present = [key for key in (frame_key, sec_key) if key in item]
        if len(present) != 1:
            return None
        value = item[present[0]]
        if present[0] == frame_key:
            return value if type(value) is int else None
        if type(value) not in (int, float):
            return None
        seconds = float(value)
        if not math.isfinite(seconds):
            return None
        return int(round(seconds * fps))

    for kind, keys in (
        ("positive", ("positive_ranges", "anomaly_ranges", "events")),
        ("ignore", ("ignore_ranges",)),
    ):
        items = raw_items(keys)
        raw_counts[kind] = len(items)
        for item in items:
            if not isinstance(item, dict):
                invalid_ranges += 1
                continue
            start = coordinate(item, "start")
            end = coordinate(item, "end")
            if start is None or end is None:
                invalid_ranges += 1
                continue
            if start < 0 or end < start:
                invalid_ranges += 1
                loader_normalized_ranges += 1
                continue
            if total_frames <= 0 or start >= total_frames or end >= total_frames:
                out_of_bounds_ranges += 1

    parsed_counts = {
        "positive": len(parsed.get("positive_ranges") or []),
        "ignore": len(parsed.get("ignore_ranges") or []),
    }
    clipped_counts = {
        "positive": len(clipped.get("positive_ranges") or []),
        "ignore": len(clipped.get("ignore_ranges") or []),
    }
    clipped_or_dropped_ranges = sum(
        1
        for key in ("positive_ranges", "ignore_ranges")
        for item in parsed.get(key) or []
        if (
            type(item.get("start_frame")) is not int
            or type(item.get("end_frame")) is not int
            or item["start_frame"] < 0
            or item["end_frame"] < item["start_frame"]
            or item["start_frame"] >= total_frames
            or item["end_frame"] >= total_frames
        )
    )
    all_ranges_within_source = bool(
        container_valid
        and invalid_ranges == 0
        and out_of_bounds_ranges == 0
        and loader_normalized_ranges == 0
        and clipped_or_dropped_ranges == 0
        and raw_counts == parsed_counts == clipped_counts
    )
    return {
        "raw_range_counts": raw_counts,
        "parsed_range_counts": parsed_counts,
        "clipped_range_counts": clipped_counts,
        "invalid_range_count": invalid_ranges,
        "out_of_bounds_range_count": out_of_bounds_ranges,
        "loader_normalized_range_count": loader_normalized_ranges,
        "clipped_or_dropped_range_count": clipped_or_dropped_ranges,
        "all_ranges_within_source": all_ranges_within_source,
    }


@dataclass
class VirtualVideoClock:
    """Clock whose time is controlled by the current video timestamp."""

    epoch: float = 1_700_000_000.0
    monotonic_origin: float = 1_000_000.0
    seconds: float = 0.0

    def set(self, seconds: float) -> None:
        self.seconds = float(seconds)

    def monotonic(self) -> float:
        # Production monotonic clocks are already far from zero. Starting an
        # offline replay at zero makes sentinel timestamps such as
        # ``last_alert = 0`` look recent and incorrectly activates cooldowns.
        return self.monotonic_origin + self.seconds

    def time(self) -> float:
        return self.epoch + self.seconds


class OfflineCapture:
    """Minimal capture adapter used to initialize a pipeline for direct frames."""

    protocol = "file"

    def __init__(self, camera_id: str, source: str) -> None:
        self.camera_id = camera_id
        self.source = source
        self._state = CameraState(connected=False)

    @property
    def state(self) -> CameraState:
        return self._state

    def connect(self) -> bool:
        self._state.connected = True
        return True

    def read(self) -> None:
        return None

    def read_latest(self) -> None:
        return None

    def start_capture_thread(self) -> None:
        return None

    def request_reconnect(self) -> None:
        return None

    def stop(self) -> None:
        self._state.connected = False


def create_pipeline(
    *,
    config_path: Path,
    camera_id: str,
    video_path: Path,
    model_path: Path | None,
    model_version_id: str | None = None,
) -> DetectionPipeline:
    """Build a production pipeline with external side effects disabled."""
    config = load_config(config_path)
    if not config.cameras:
        raise ValueError(f"config has no cameras: {config_path}")
    source_camera = next(
        (camera for camera in config.cameras if camera.camera_id == camera_id),
        None,
    )
    if source_camera is None:
        available = ", ".join(sorted(camera.camera_id for camera in config.cameras))
        raise ValueError(
            f"camera_id {camera_id!r} is not present in {config_path}; "
            f"available cameras: {available}"
        )
    ring_buffer = source_camera.ring_buffer.model_copy(update={"enabled": False})
    camera = source_camera.model_copy(update={
        "camera_id": camera_id,
        "source": str(video_path),
        "protocol": "file",
        "ring_buffer": ring_buffer,
    })
    capture = OfflineCapture(camera_id, str(video_path))
    effective_model_version_id = model_version_id
    if effective_model_version_id is None:
        effective_model_version_id = (
            f"{camera.anomaly.model_type}:{model_path.parent.name}"
            if model_path is not None
            else f"{camera.anomaly.model_type}:ssim_fallback"
        )
    pipeline = DetectionPipeline(
        camera_config=camera,
        alert_config=config.alerts,
        classifier_config=config.classifier,
        segmenter_config=config.segmenter,
        physics_config=config.physics,
        imaging_config=config.imaging,
        model_path=model_path,
        model_version_id=effective_model_version_id,
        capture=capture,
        auto_discover_model=False,
        baselines_dir=config.storage.baselines_dir,
    )
    # Sensor signals are not represented by the video/annotation corpus.  Keep
    # this fact beside the concrete pipeline instance so custom diagnostic
    # factories can continue to work without forcing a second config load.
    pipeline._offline_evaluation_context = {  # type: ignore[attr-defined]
        "sensor_fusion_enabled": bool(config.sensor_fusion.enabled),
        "synchronized_sensor_corpus": False,
        # CameraManager performs cross-camera suppression/downgrade outside
        # DetectionPipeline. A single-video replay cannot reproduce it.
        "cross_camera_enabled": bool(config.cross_camera.enabled),
        "synchronized_cross_camera_corpus": False,
    }
    return pipeline


def _normalized_path(value: str | Path | None) -> str | None:
    """Return a comparison-safe absolute path without requiring it to exist."""
    if value in (None, ""):
        return None
    try:
        resolved = Path(value).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    # Preserve case-sensitive Linux semantics while normalizing Windows case
    # and separators. The concrete path is still reported for audit/debugging.
    return os.path.normcase(str(resolved))


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _runtime_model_identity(
    pipeline: Any,
    *,
    requested_path: Path | None,
    requested_version_id: str | None,
) -> dict[str, Any]:
    """Capture the artifact identity of the engine that actually initialized."""
    get_detector_status = getattr(pipeline, "get_detector_status", None)
    detector_status: Any = None
    detector_status_error: str | None = None
    if callable(get_detector_status):
        try:
            detector_status = get_detector_status()
        except Exception as exc:  # pragma: no cover - defensive audit path
            detector_status_error = f"{type(exc).__name__}: {exc}"

    pipeline_path_value = getattr(pipeline, "_anomaly_model_path", None)
    detector_path_value = _value(detector_status, "model_path")
    actual_version = getattr(pipeline, "_model_version_id", None)
    artifact_sha256 = getattr(pipeline, "_model_artifact_sha256", None)
    requested_normalized = _normalized_path(requested_path)
    pipeline_normalized = _normalized_path(pipeline_path_value)
    detector_normalized = _normalized_path(detector_path_value)
    explicit_path = requested_path is not None
    explicit_version = bool(
        requested_version_id is not None and str(requested_version_id).strip()
    )
    normalized_path_match = bool(
        explicit_path
        and requested_normalized is not None
        and pipeline_normalized == requested_normalized
        and detector_normalized == requested_normalized
    )
    version_match = bool(
        explicit_version
        and actual_version not in (None, "")
        and str(actual_version) == str(requested_version_id)
    )
    detector_loaded = bool(_value(detector_status, "model_loaded", False))
    release_approved = bool(
        explicit_path
        and explicit_version
        and normalized_path_match
        and version_match
        and detector_loaded
        and artifact_sha256 not in (None, "")
    )
    return {
        "requested": {
            "path": str(requested_path) if requested_path is not None else None,
            "normalized_path": requested_normalized,
            "version_id": (
                str(requested_version_id)
                if requested_version_id is not None
                else None
            ),
            "explicit_path": explicit_path,
            "explicit_version": explicit_version,
        },
        "actual": {
            "pipeline_path": (
                str(pipeline_path_value)
                if pipeline_path_value not in (None, "")
                else None
            ),
            "normalized_pipeline_path": pipeline_normalized,
            "detector_path": (
                str(detector_path_value)
                if detector_path_value not in (None, "")
                else None
            ),
            "normalized_detector_path": detector_normalized,
            "version_id": (
                str(actual_version) if actual_version not in (None, "") else None
            ),
            "artifact_sha256": (
                str(artifact_sha256)
                if artifact_sha256 not in (None, "")
                else None
            ),
            "detector_loaded": detector_loaded,
            "detector_mode": _value(detector_status, "mode"),
            "detector_status_error": detector_status_error,
        },
        "normalized_path_match": normalized_path_match,
        "version_match": version_match,
        "release_approved": release_approved,
    }


def _candidate_confirmation_contract(pipeline: Any) -> dict[str, Any]:
    """Serialize the confirmation policy on the instantiated camera pipeline."""
    camera_config = getattr(pipeline, "camera_config", None)
    confirmation = getattr(camera_config, "static_confirmation", None)
    if camera_config is None or confirmation is None:
        return {"available": False, "camera_id": None}
    model_dump = getattr(confirmation, "model_dump", None)
    if callable(model_dump):
        payload = model_dump(mode="json")
    elif isinstance(confirmation, dict):
        payload = dict(confirmation)
    else:
        payload = {
            key: value
            for key, value in vars(confirmation).items()
            if not key.startswith("_")
        }
    return {
        "available": True,
        "camera_id": getattr(camera_config, "camera_id", None),
        **payload,
    }


def _production_parity_contract(pipeline: Any) -> dict[str, Any]:
    context = getattr(pipeline, "_offline_evaluation_context", None)
    sensor_fusion_enabled = bool(_value(context, "sensor_fusion_enabled", False))
    synchronized_sensor_corpus = bool(
        _value(context, "synchronized_sensor_corpus", False)
    )
    cross_camera_enabled = bool(_value(context, "cross_camera_enabled", False))
    synchronized_cross_camera_corpus = bool(
        _value(context, "synchronized_cross_camera_corpus", False)
    )
    reasons: list[str] = []
    if sensor_fusion_enabled and not synchronized_sensor_corpus:
        reasons.append("sensor_fusion_enabled_without_synchronized_sensor_corpus")
    if cross_camera_enabled and not synchronized_cross_camera_corpus:
        reasons.append(
            "cross_camera_enabled_without_synchronized_multi_camera_corpus"
        )
    supported = not reasons
    return {
        "supported": supported,
        "sensor_fusion_enabled": sensor_fusion_enabled,
        "synchronized_sensor_corpus": synchronized_sensor_corpus,
        "cross_camera_enabled": cross_camera_enabled,
        "synchronized_cross_camera_corpus": synchronized_cross_camera_corpus,
        "reason": reasons[0] if reasons else None,
        "reasons": reasons,
    }


def _active_profile_operator_authorization(
    pipeline: Any,
    configured_profile_id: str | None,
) -> dict[str, Any]:
    """Read explicit human approval from the active immutable profile.

    Model/profile/hash agreement proves only technical identity.  Production
    release additionally requires ``metadata.operator_approved`` to be the
    literal JSON boolean ``true`` on the active profile.  Missing stores,
    legacy profiles, non-boolean truthy values, and profile mismatches all fail
    closed while retaining a machine-readable diagnostic reason.
    """
    result: dict[str, Any] = {
        "required": True,
        "configured_profile_id": configured_profile_id,
        "active_profile_id": None,
        "metadata_field_present": False,
        "operator_approved": False,
        "profile_artifact_path": None,
        "source_baseline_name": None,
        "source_baseline_sha256": None,
        "reason": None,
    }
    if configured_profile_id is None:
        result["reason"] = "scene_profile_operator_approval_profile_missing"
        return result

    camera_config = getattr(pipeline, "camera_config", None)
    scene_guard_config = getattr(camera_config, "scene_guard", None)
    profiles_dir = getattr(scene_guard_config, "profiles_dir", None)
    camera_id = getattr(camera_config, "camera_id", None)
    if profiles_dir in (None, "") or camera_id in (None, ""):
        result["reason"] = "scene_profile_operator_approval_store_unavailable"
        return result

    try:
        from argus.preprocessing.scene_profile import SceneProfileStore

        store = SceneProfileStore(profiles_dir)
        active = store.get_active(str(camera_id))
    except Exception as exc:
        result["reason"] = "scene_profile_operator_approval_unavailable"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    if active is None:
        result["reason"] = "scene_profile_operator_approval_profile_missing"
        return result
    result["active_profile_id"] = active.profile_id
    result["profile_artifact_path"] = str(
        store.root / str(camera_id) / active.profile_id
    )
    result["source_baseline_sha256"] = active.source_baseline_sha256
    result["source_baseline_name"] = active.source_baseline_name
    if active.profile_id != configured_profile_id:
        result["reason"] = "scene_profile_operator_approval_profile_mismatch"
        return result

    metadata = dict(active.metadata)
    result["metadata_field_present"] = "operator_approved" in metadata
    if not result["metadata_field_present"]:
        result["reason"] = "scene_profile_operator_approval_missing"
        return result
    if metadata.get("operator_approved") is not True:
        result["reason"] = "scene_profile_operator_approval_not_granted"
        return result

    result["operator_approved"] = True
    return result


def _frame_label(frame_index: int, annotations: dict[str, Any] | None) -> int | None:
    if annotations is None:
        return None
    for item in annotations.get("ignore_ranges") or []:
        if int(item["start_frame"]) <= frame_index <= int(item["end_frame"]):
            return None
    for item in annotations.get("positive_ranges") or []:
        if int(item["start_frame"]) <= frame_index <= int(item["end_frame"]):
            return 1
    return 0


def _clip_annotations(
    annotations: dict[str, Any] | None,
    *,
    evaluated_frames: int,
) -> dict[str, Any] | None:
    """Limit annotation ranges to the portion of video actually replayed."""
    if annotations is None:
        return None
    clipped: dict[str, Any] = {
        key: value
        for key, value in annotations.items()
        if key not in {"positive_ranges", "ignore_ranges"}
    }
    last_frame = evaluated_frames - 1
    for key in ("positive_ranges", "ignore_ranges"):
        ranges: list[dict[str, Any]] = []
        if last_frame >= 0:
            for item in annotations.get(key) or []:
                start = int(item["start_frame"])
                if start > last_frame:
                    continue
                copied = dict(item)
                copied["end_frame"] = min(int(item["end_frame"]), last_frame)
                ranges.append(copied)
        clipped[key] = ranges
    return clipped


def _decision(label: int | None, predicted: int) -> str:
    if label is None:
        return "unlabeled"
    if label == 1:
        return "tp" if predicted else "fn"
    return "fp" if predicted else "tn"


def _serialize_alert(alert: Any, timestamp_sec: float) -> dict[str, Any]:
    severity = getattr(alert.severity, "value", str(alert.severity))
    detection_type = getattr(alert.detection_type, "value", str(alert.detection_type))
    return {
        "alert_id": alert.alert_id,
        "camera_id": alert.camera_id,
        "zone_id": alert.zone_id,
        "frame_number": int(alert.frame_number),
        "timestamp_sec": float(timestamp_sec),
        "severity": severity,
        "anomaly_score": float(alert.anomaly_score),
        "detection_type": detection_type,
        "category": str(getattr(alert, "category", "unknown")),
        "detected_objects": list(getattr(alert, "detected_objects", []) or []),
        "evidence_summary": str(getattr(alert, "evidence_summary", "")),
        "evidence_json": dict(getattr(alert, "evidence_json", {}) or {}),
        "event_group_id": getattr(alert, "event_group_id", None),
    }


def _summarize_scene_guard(
    records: list[dict[str, Any]],
    *,
    sample_period_seconds: float,
) -> dict[str, Any]:
    """Summarize detection availability so isolation cannot hide false positives."""
    statuses = [record.get("scene_guard") or {} for record in records]
    state_counts = Counter(str(status.get("state") or "unknown") for status in statuses)
    enabled = [bool(status.get("enabled", False)) for status in statuses]
    usable = [bool(status.get("usable", False)) for status in statuses]
    geometry_normalized = [
        status.get("geometry_normalization_applied") is True
        for status in statuses
    ]
    warp_border_policy_counts = Counter(
        str(status.get("warp_border_policy") or "not_reported")
        for status in statuses
    )
    longest_blocked_run = 0
    current_blocked_run = 0
    transitions = 0
    previous: bool | None = None
    for is_usable in usable:
        if previous is not None and previous != is_usable:
            transitions += 1
        previous = is_usable
        if is_usable:
            current_blocked_run = 0
        else:
            current_blocked_run += 1
            longest_blocked_run = max(longest_blocked_run, current_blocked_run)

    usable_frames = sum(usable)
    blocked_frames = len(usable) - usable_frames
    enabled_frames = sum(enabled)
    disabled_frames = len(enabled) - enabled_frames
    return {
        "state_counts": dict(sorted(state_counts.items())),
        "observed_frames": len(statuses),
        "enabled_frames": enabled_frames,
        "disabled_frames": disabled_frames,
        "usable_frames": usable_frames,
        "blocked_frames": blocked_frames,
        "geometry_normalized_frames": sum(geometry_normalized),
        "alignable_without_normalization_frames": sum(
            1
            for status, normalized in zip(statuses, geometry_normalized)
            if str(status.get("state") or "unknown") == "alignable"
            and not normalized
        ),
        "warp_border_policy_counts": dict(sorted(warp_border_policy_counts.items())),
        "usable_coverage_ratio": (
            usable_frames / len(usable) if usable else None
        ),
        "blocked_seconds_estimate": blocked_frames * sample_period_seconds,
        "longest_blocked_seconds_estimate": (
            longest_blocked_run * sample_period_seconds
        ),
        "availability_transitions": transitions,
    }


def _summarize_candidate_model_evidence(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Separate map extraction availability from calibrated rule coverage.

    Missing, failed, sampled-out, and uncalibrated observations are never
    converted into model rejects.  Annotation-ignore and warmup frames remain
    in the raw replay JSON, but are excluded from every quality denominator.
    """
    status_counts: Counter[str] = Counter()
    output_spaces: Counter[str] = Counter()
    calibration_ids: Counter[str] = Counter()
    accepted = 0
    rejected = 0
    nullable_contract_violations = 0
    extraction_contract_violations = 0
    observed = 0
    excluded = 0
    not_observed = 0
    extraction_eligible = 0
    extracted = 0
    calibrated_evaluated = 0

    for frame in records:
        score_eligible = bool(frame.get("score_eligible", True))
        for candidate in frame.get("candidate_records") or []:
            if candidate.get("candidate_type") != "persistent_confirmation":
                continue
            evidence = candidate.get("evidence") or {}
            model = evidence.get("model_evidence")
            if not isinstance(model, dict):
                continue
            if not score_eligible:
                excluded += 1
                continue
            if evidence.get("observed_this_frame") is False:
                not_observed += 1
                continue
            observed += 1
            status = str(model.get("status") or "unknown")
            status_counts[status] += 1
            output_space = model.get("output_space")
            if output_space:
                output_spaces[str(output_space)] += 1
            calibration_id = model.get("calibration_id")
            if calibration_id:
                calibration_ids[str(calibration_id)] += 1

            # Upstream policy skips had no opportunity to produce a model map.
            # Every other persistent candidate belongs in the extraction
            # denominator, including unavailable/error/rate-limited states.
            if status != "upstream_skipped":
                extraction_eligible += 1
            core_metrics: list[float] = []
            for key in ("model_score", "roi_p95", "roi_context_delta"):
                try:
                    value = float(model[key])
                except (KeyError, TypeError, ValueError):
                    break
                if not math.isfinite(value):
                    break
                core_metrics.append(value)
            extraction_succeeded = (
                status in {"evaluated", "uncalibrated"}
                and len(core_metrics) == 3
            )
            if extraction_succeeded:
                extracted += 1
                if status == "evaluated":
                    calibrated_evaluated += 1
            elif status in {"evaluated", "uncalibrated"}:
                extraction_contract_violations += 1

            decision = model.get("would_accept")
            if status == "evaluated":
                if decision is True:
                    accepted += 1
                elif decision is False:
                    rejected += 1
                else:
                    nullable_contract_violations += 1
            elif decision is not None:
                nullable_contract_violations += 1

    evaluated = status_counts["evaluated"]
    return {
        "observed_candidates": observed,
        "all_observed_candidates": observed + excluded + not_observed,
        "excluded_candidate_observations": excluded,
        "not_observed_candidate_records": not_observed,
        "status_counts": dict(sorted(status_counts.items())),
        "evaluated_candidates": evaluated,
        # Backward-compatible aliases now use the fail-safe extraction
        # denominator rather than silently omitting unavailable model maps.
        "eligible_candidates": extraction_eligible,
        "telemetry_coverage_ratio": (
            extracted / extraction_eligible if extraction_eligible else None
        ),
        "map_extraction_eligible_candidates": extraction_eligible,
        "map_extracted_candidates": extracted,
        "map_extraction_coverage_ratio": (
            extracted / extraction_eligible if extraction_eligible else None
        ),
        "calibrated_evaluation_eligible_candidates": extracted,
        "calibrated_evaluated_candidates": calibrated_evaluated,
        "calibrated_evaluation_coverage_ratio": (
            calibrated_evaluated / extracted if extracted else None
        ),
        "accepted_candidates": accepted,
        "rejected_candidates": rejected,
        "rule_accept_rate": accepted / evaluated if evaluated else None,
        "nullable_contract_violations": nullable_contract_violations,
        "extraction_contract_violations": extraction_contract_violations,
        "output_space_counts": dict(sorted(output_spaces.items())),
        "calibration_id_counts": dict(sorted(calibration_ids.items())),
    }


def _summarize_prospective_candidate_fusion(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize the shadow structure+model+temporal decision, not alerts."""
    decision_status_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    fusion_policy_counts: Counter[str] = Counter()
    temporal_status_counts: Counter[str] = Counter()
    temporal_reason_counts: Counter[str] = Counter()
    observed = 0
    excluded = 0
    not_observed = 0
    accepted = 0
    pending = 0
    rejected = 0
    unavailable = 0
    confirmed = 0
    attribution_observed = 0
    attribution_excluded = 0
    attribution_would_suppress = 0
    attribution_would_preserve = 0
    attribution_unavailable = 0
    attribution_effective = 0
    attribution_reasons: Counter[str] = Counter()
    attribution_hot_coverages: list[float] = []

    for frame in records:
        score_eligible = bool(frame.get("score_eligible", True))
        for candidate in frame.get("candidate_records") or []:
            candidate_type = candidate.get("candidate_type")
            if candidate_type == "anomaly_attribution":
                evidence = candidate.get("evidence") or {}
                if not isinstance(evidence, dict):
                    continue
                if not score_eligible:
                    attribution_excluded += 1
                    continue
                attribution_observed += 1
                attribution_reasons[
                    str(evidence.get("reason") or "unknown")
                ] += 1
                if evidence.get("would_suppress") is True:
                    attribution_would_suppress += 1
                elif evidence.get("would_suppress") is False:
                    attribution_would_preserve += 1
                else:
                    attribution_unavailable += 1
                if evidence.get("effective") is True:
                    attribution_effective += 1
                try:
                    hot_coverage = float(evidence["hot_coverage"])
                except (KeyError, TypeError, ValueError):
                    hot_coverage = None
                if hot_coverage is not None and math.isfinite(hot_coverage):
                    attribution_hot_coverages.append(hot_coverage)
                continue
            if candidate_type != "persistent_confirmation":
                continue
            evidence = candidate.get("evidence") or {}
            if not isinstance(evidence, dict):
                continue
            if not score_eligible:
                excluded += 1
                continue
            if evidence.get("observed_this_frame") is False:
                not_observed += 1
                continue
            observed += 1
            mode_counts[str(evidence.get("mode") or "unknown")] += 1
            fusion_policy_counts[
                str(evidence.get("fusion_policy") or "unknown")
            ] += 1
            decision_status = str(evidence.get("decision_status") or "unknown")
            decision_status_counts[decision_status] += 1
            temporal = evidence.get("temporal_evidence")
            if isinstance(temporal, dict):
                temporal_status_counts[
                    str(temporal.get("status") or "unknown")
                ] += 1
                temporal_reason_counts[
                    str(temporal.get("status_reason") or "unknown")
                ] += 1
                if bool(temporal.get("confirmed", False)):
                    confirmed += 1

            would_accept = evidence.get("would_accept")
            if would_accept is True:
                accepted += 1
            elif would_accept is None:
                unavailable += 1
            else:
                is_pending = bool(
                    isinstance(temporal, dict)
                    and temporal.get("status") == "evaluated"
                    and int(temporal.get("streak") or 0) > 0
                    and not bool(temporal.get("confirmed", False))
                )
                if is_pending:
                    pending += 1
                else:
                    rejected += 1

    evaluated = accepted + pending + rejected
    attribution_decided = (
        attribution_would_suppress + attribution_would_preserve
    )
    return {
        "observed_candidates": observed,
        "all_observed_candidates": observed + excluded + not_observed,
        "excluded_candidate_observations": excluded,
        "not_observed_candidate_records": not_observed,
        "evaluated_candidates": evaluated,
        "accepted_candidates": accepted,
        "pending_candidates": pending,
        "rejected_candidates": rejected,
        "unavailable_candidates": unavailable,
        "accept_rate": accepted / evaluated if evaluated else None,
        "mode_counts": dict(sorted(mode_counts.items())),
        "fusion_policy_counts": dict(sorted(fusion_policy_counts.items())),
        "decision_status_counts": dict(sorted(decision_status_counts.items())),
        "temporal_status_counts": dict(sorted(temporal_status_counts.items())),
        "temporal_reason_counts": dict(sorted(temporal_reason_counts.items())),
        "confirmed_candidates": confirmed,
        "anomaly_attribution": {
            "observed_frames": attribution_observed,
            "excluded_frame_observations": attribution_excluded,
            "decided_frames": attribution_decided,
            "would_suppress_frames": attribution_would_suppress,
            "would_preserve_frames": attribution_would_preserve,
            "unavailable_frames": attribution_unavailable,
            "effective_suppressed_frames": attribution_effective,
            "suppress_rate": (
                attribution_would_suppress / attribution_decided
                if attribution_decided
                else None
            ),
            "reason_counts": dict(sorted(attribution_reasons.items())),
            "hot_coverage_summary": {
                "count": len(attribution_hot_coverages),
                "min": (
                    min(attribution_hot_coverages)
                    if attribution_hot_coverages
                    else None
                ),
                "max": (
                    max(attribution_hot_coverages)
                    if attribution_hot_coverages
                    else None
                ),
                "mean": (
                    sum(attribution_hot_coverages) / len(attribution_hot_coverages)
                    if attribution_hot_coverages
                    else None
                ),
            },
        },
    }


def _load_scene_reference(
    video_path: Path,
    *,
    reference_path: Path | None,
    frame_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load an explicit scene reference without teaching from replay frames.

    A supplied image is preferred. Otherwise a separate decoder reads the
    requested source-video frame, leaving the main replay decoder positioned at
    frame zero. This is deliberately an offline-evaluator action; production
    capture never calls this helper or adopts a live frame automatically.
    """
    if frame_index < 0:
        raise ValueError("scene_reference_frame_index must be >= 0")

    if reference_path is not None:
        if not reference_path.is_file():
            raise FileNotFoundError(f"scene reference image not found: {reference_path}")
        frame = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"could not decode scene reference image: {reference_path}")
        return frame, {
            "mode": "image",
            "path": str(reference_path),
            "frame_index": None,
            "source": f"offline-eval:image:{reference_path}",
        }

    reference_capture = cv2.VideoCapture(str(video_path))
    try:
        if not reference_capture.isOpened():
            raise FileNotFoundError(f"could not open scene reference video: {video_path}")
        if frame_index:
            reference_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = reference_capture.read()
        if not ok or frame is None:
            raise ValueError(
                f"could not decode scene reference frame {frame_index}: {video_path}"
            )
    finally:
        reference_capture.release()

    return frame, {
        "mode": "video_frame",
        "path": str(video_path),
        "frame_index": frame_index,
        "source": f"offline-eval:video-frame:{frame_index}",
    }


def evaluate_pipeline_video(
    video_path: str | Path,
    *,
    config_path: str | Path,
    camera_id: str,
    annotation_path: str | Path | None = None,
    model_path: str | Path | None = None,
    model_version_id: str | None = None,
    sample_every: int = 1,
    max_frames: int | None = None,
    warmup_seconds: float = 30.0,
    scene_reference_path: str | Path | None = None,
    scene_reference_frame_index: int = 0,
    scene_profile_id: str | None = None,
    pipeline_factory: Callable[..., Any] = create_pipeline,
) -> dict[str, Any]:
    """Replay a video through the complete production decision pipeline.

    The scene guard receives an explicit reference image before replay starts.
    By default this is source-video frame zero; callers can instead provide an
    approved image or another source-video frame. No processed runtime frame is
    ever learned as a reference by this evaluator.  The initialized production
    scene-profile/model binding is preserved unless an explicit profile ID is
    supplied for an equivalent approved reference. Technical binding alone is
    diagnostic: release approval additionally requires the active immutable
    profile to carry the explicit ``operator_approved: true`` metadata flag.
    """
    if sample_every < 1:
        raise ValueError("sample_every must be >= 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1 when provided")
    if warmup_seconds < 0:
        raise ValueError("warmup_seconds must be >= 0")

    video = Path(video_path)
    config_file = Path(config_path)
    config_artifact_sha256 = _evidence_artifact_sha256_if_exists(config_file)
    source_video_sha256 = _stream_sha256(video)
    annotation_file = Path(annotation_path) if annotation_path is not None else None
    annotation_sha256 = (
        _stream_sha256(annotation_file) if annotation_file is not None else None
    )
    raw_annotation = (
        json.loads(annotation_file.read_text(encoding="utf-8"))
        if annotation_file is not None
        else None
    )
    reference_frame, scene_reference = _load_scene_reference(
        video,
        reference_path=(
            Path(scene_reference_path) if scene_reference_path is not None else None
        ),
        frame_index=scene_reference_frame_index,
    )
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {video}")
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    source_fps_available = bool(math.isfinite(reported_fps) and reported_fps > 0.0)
    # Keep diagnostic replay possible for malformed containers, but expose the
    # assumption explicitly so a release gate can never treat it as complete
    # timing evidence.
    fps = reported_fps if source_fps_available else 30.0
    reported_total_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    source_total_frames_known = bool(
        math.isfinite(reported_total_frames) and reported_total_frames > 0.0
    )
    total_frames = int(reported_total_frames) if source_total_frames_known else 0
    annotations = (
        load_video_annotations(annotation_path, fps=fps)
        if annotation_path is not None
        else None
    )
    requested_model_path = Path(model_path) if model_path is not None else None
    pipeline_kwargs = {
        "config_path": Path(config_path),
        "camera_id": camera_id,
        "video_path": video,
        "model_path": requested_model_path,
    }
    if model_version_id is not None:
        pipeline_kwargs["model_version_id"] = model_version_id
    pipeline = pipeline_factory(
        **pipeline_kwargs,
    )
    clock = VirtualVideoClock()
    records: list[dict[str, Any]] = []
    scene_guard_records: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []
    failed_frames: list[dict[str, Any]] = []
    ignored_frames = 0
    warmup_frames = 0
    excluded_frames = 0
    raw_frame_index = 0
    source_exhausted = False
    truncated_by_max_frames = False
    source_read_error = False
    activated = False
    processing_started = time.perf_counter()
    runtime_model_version: str | None = None
    model_identity: dict[str, Any] = {}
    production_parity: dict[str, Any] = {}
    candidate_confirmation_contract: dict[str, Any] = {}
    release_contract: dict[str, Any] = {}

    try:
        with patch("time.monotonic", clock.monotonic), patch("time.time", clock.time):
            if not pipeline.initialize():
                raise RuntimeError("pipeline initialization failed")
            set_scene_reference = getattr(pipeline, "set_scene_reference", None)
            if not callable(set_scene_reference):
                raise TypeError("pipeline does not provide set_scene_reference()")
            get_scene_status = getattr(pipeline, "get_scene_guard_status", None)
            initial_scene_status = (
                get_scene_status()
                if callable(get_scene_status)
                else {}
            )
            if not isinstance(initial_scene_status, dict):
                initial_scene_status = {}
            model_identity = _runtime_model_identity(
                pipeline,
                requested_path=requested_model_path,
                requested_version_id=model_version_id,
            )
            runtime_model_version = model_identity["actual"]["version_id"]
            production_parity = _production_parity_contract(pipeline)
            candidate_confirmation_contract = _candidate_confirmation_contract(
                pipeline
            )

            configured_profile_value = initial_scene_status.get("profile_id")
            configured_profile_id = (
                str(configured_profile_value)
                if configured_profile_value not in (None, "")
                else None
            )
            configured_reference_value = initial_scene_status.get(
                "reference_sha256"
            )
            configured_reference_sha256 = (
                str(configured_reference_value)
                if configured_reference_value not in (None, "")
                else None
            )
            configured_bound_value = initial_scene_status.get(
                "bound_model_version"
            )
            # This value is evidence owned by the initialized profile.  Never
            # fill it from a caller argument or the runtime model version.
            bound_model_version = (
                str(configured_bound_value)
                if configured_bound_value not in (None, "")
                else None
            )
            configured_reference_frame = getattr(
                pipeline,
                "_scene_guard_reference_frame",
                None,
            )
            configured_frame_sha256 = (
                _frame_content_sha256(configured_reference_frame)
                if isinstance(configured_reference_frame, np.ndarray)
                and configured_reference_frame.size > 0
                else None
            )
            configured_profile_initialized = bool(
                initial_scene_status.get("enabled") is True
                and initial_scene_status.get("reference_available") is True
                and configured_profile_id is not None
                and configured_reference_sha256 is not None
                and configured_frame_sha256 == configured_reference_sha256
            )
            if (
                scene_reference_path is None
                and isinstance(configured_reference_frame, np.ndarray)
                and configured_reference_frame.size > 0
            ):
                # Release evaluation must use the exact approved production
                # profile image, never let the video under test nominate its
                # own frame zero as the trusted scene.
                reference_frame = configured_reference_frame.copy()
                scene_reference = {
                    "mode": "active_profile",
                    "path": initial_scene_status.get("reference_source"),
                    "frame_index": None,
                    "source": str(
                        initial_scene_status.get("reference_source")
                        or f"profile:{configured_profile_id}"
                    ),
                }
            effective_profile_id = (
                str(scene_profile_id)
                if scene_profile_id is not None
                else configured_profile_id
            )
            if (
                effective_profile_id is not None
                and not effective_profile_id.strip()
            ):
                raise ValueError("scene_profile_id must not be blank")
            scene_reference["profile_id"] = effective_profile_id
            scene_reference["configured_profile_id"] = configured_profile_id
            scene_reference["profile_id_matches_configured"] = bool(
                effective_profile_id is not None
                and configured_profile_id is not None
                and effective_profile_id == configured_profile_id
            )
            scene_reference["configured_profile_initialized"] = (
                configured_profile_initialized
            )
            scene_reference["bound_model_version"] = bound_model_version
            selected_reference_sha256 = _frame_content_sha256(reference_frame)
            scene_reference["reference_sha256"] = selected_reference_sha256
            scene_reference["configured_reference_sha256"] = configured_reference_sha256
            scene_reference["configured_reference_frame_sha256"] = (
                configured_frame_sha256
            )
            scene_reference["matches_configured_reference"] = bool(
                selected_reference_sha256 is not None
                and configured_reference_sha256 is not None
                and selected_reference_sha256 == configured_reference_sha256
            )
            scene_reference["profile_bound_model_matches_runtime"] = bool(
                bound_model_version is not None
                and runtime_model_version is not None
                and bound_model_version == runtime_model_version
            )
            operator_authorization = _active_profile_operator_authorization(
                pipeline,
                configured_profile_id,
            )
            scene_reference["operator_authorization"] = operator_authorization
            scene_reference["operator_approved"] = (
                operator_authorization.get("operator_approved") is True
            )
            scene_reference["technical_reference_match"] = bool(
                configured_profile_initialized
                and scene_reference["profile_id_matches_configured"]
                and scene_reference["matches_configured_reference"]
                and scene_reference["profile_bound_model_matches_runtime"]
                and scene_reference["mode"] in {"active_profile", "image"}
            )
            scene_reference["reference_approved"] = bool(
                scene_reference["technical_reference_match"]
                and scene_reference["operator_approved"]
            )
            release_reasons: list[str] = []
            if not scene_reference["technical_reference_match"]:
                release_reasons.append("scene_reference_not_approved")
            if not scene_reference["operator_approved"]:
                release_reasons.append(
                    str(
                        operator_authorization.get("reason")
                        or "scene_profile_operator_approval_not_granted"
                    )
                )
            if not model_identity["release_approved"]:
                release_reasons.append("model_identity_not_approved")
            if not production_parity["supported"]:
                parity_reasons = production_parity.get("reasons")
                if isinstance(parity_reasons, list) and parity_reasons:
                    release_reasons.extend(str(reason) for reason in parity_reasons)
                else:
                    release_reasons.append(str(production_parity["reason"]))
            release_contract = {
                "approved": not release_reasons,
                "reference_approved": scene_reference["reference_approved"],
                "operator_approved": scene_reference["operator_approved"],
                "operator_approval_reason": operator_authorization.get("reason"),
                "model_identity_approved": model_identity["release_approved"],
                "production_parity_supported": production_parity["supported"],
                "reasons": release_reasons,
            }
            scene_reference["release_approved"] = release_contract["approved"]
            set_scene_reference(
                reference_frame,
                source=scene_reference["source"],
                profile_id=effective_profile_id,
                metadata={
                    "offline_evaluation": True,
                    "mode": scene_reference["mode"],
                    "path": scene_reference["path"],
                    "frame_index": scene_reference["frame_index"],
                    "model_version_id": scene_reference["bound_model_version"],
                },
            )
            if warmup_seconds == 0:
                pipeline.set_mode(PipelineMode.ACTIVE)
                activated = True

            while True:
                if max_frames is not None and raw_frame_index >= max_frames:
                    # Do not infer EOF from CAP_PROP_FRAME_COUNT here. Container
                    # metadata can be wrong; only a capture read ending the
                    # stream proves exhaustion for release evidence.
                    truncated_by_max_frames = True
                    break
                ok, frame = capture.read()
                if not ok:
                    if source_total_frames_known and raw_frame_index < total_frames:
                        source_read_error = True
                    else:
                        source_exhausted = True
                    break
                frame_index = raw_frame_index
                raw_frame_index += 1
                if frame_index % sample_every != 0:
                    continue

                timestamp_sec = frame_index / fps
                clock.set(timestamp_sec)
                warmup_for_scoring = timestamp_sec < warmup_seconds
                if not activated and not warmup_for_scoring:
                    pipeline.set_mode(PipelineMode.ACTIVE)
                    activated = True

                label = _frame_label(frame_index, annotations)
                ignored_for_scoring = label is None and annotations is not None
                if ignored_for_scoring and warmup_for_scoring:
                    exclusion_reason = "annotation_ignore_and_warmup"
                elif ignored_for_scoring:
                    exclusion_reason = "annotation_ignore"
                elif warmup_for_scoring:
                    exclusion_reason = "warmup"
                else:
                    exclusion_reason = None
                score_eligible = exclusion_reason is None
                try:
                    alert = pipeline.process_frame(FrameData(
                        frame=frame,
                        camera_id=camera_id,
                        timestamp=timestamp_sec,
                        frame_number=frame_index,
                        resolution=(frame.shape[1], frame.shape[0]),
                    ))
                except Exception as exc:
                    failed_frames.append({"frame_index": frame_index, "error": str(exc)})
                    continue

                recent = pipeline.get_diagnostics_buffer().get_recent(1)
                diagnostic = recent[0].to_dict() if recent else None
                try:
                    diagnostic_frame = int(diagnostic["frame_number"])
                except (KeyError, TypeError, ValueError):
                    diagnostic_frame = None
                if diagnostic_frame != frame_index:
                    # DetectionPipeline intentionally catches internal stage
                    # exceptions to keep the live camera loop alive.  Offline
                    # evaluation must not mistake that swallowed failure for a
                    # normal frame or reuse the previous frame's diagnostic.
                    failed_frames.append({
                        "frame_index": frame_index,
                        "error": (
                            "pipeline produced no current-frame diagnostic "
                            f"(latest={diagnostic_frame})"
                        ),
                    })
                    continue
                score = float(diagnostic.get("anomaly_score") or 0.0)
                scene_stage_metadata: dict[str, Any] = {}
                for stage in reversed(diagnostic.get("stages") or []):
                    if not isinstance(stage, dict):
                        continue
                    if stage.get("stage_name") != "scene_guard":
                        continue
                    metadata = stage.get("metadata")
                    if isinstance(metadata, dict):
                        scene_stage_metadata = metadata
                    break
                get_scene_status = getattr(pipeline, "get_scene_guard_status", None)
                if callable(get_scene_status):
                    maybe_scene_status = get_scene_status()
                else:
                    maybe_scene_status = None
                if isinstance(maybe_scene_status, dict):
                    scene_status = {
                        "enabled": bool(maybe_scene_status.get("enabled", False)),
                        "state": str(maybe_scene_status.get("state") or "unknown"),
                        "raw_state": str(
                            maybe_scene_status.get("raw_state")
                            or maybe_scene_status.get("state")
                            or "unknown"
                        ),
                        "usable": bool(maybe_scene_status.get("usable", False)),
                        "reason": str(maybe_scene_status.get("reason") or ""),
                        "profile_id": maybe_scene_status.get("profile_id"),
                        "bound_model_version": maybe_scene_status.get(
                            "bound_model_version"
                        ),
                        "geometry_normalization_applied": (
                            scene_stage_metadata.get(
                                "geometry_normalization_applied"
                            )
                            is True
                        ),
                        "warp_border_policy": str(
                            scene_stage_metadata.get("warp_border_policy")
                            or "not_reported"
                        ),
                    }
                else:
                    scene_state = str(
                        diagnostic.get("scene_guard_state") or "disabled"
                    ) if diagnostic else "disabled"
                    scene_status = {
                        "enabled": scene_state != "disabled",
                        "state": scene_state,
                        "raw_state": scene_state,
                        "usable": scene_state in {"disabled", "stable"},
                        "reason": "",
                        "profile_id": None,
                        "bound_model_version": None,
                        "geometry_normalization_applied": (
                            scene_stage_metadata.get(
                                "geometry_normalization_applied"
                            )
                            is True
                        ),
                        "warp_border_policy": str(
                            scene_stage_metadata.get("warp_border_policy")
                            or "not_reported"
                        ),
                    }
                scene_guard_records.append({
                    "scene_guard": scene_status,
                    "score_eligible": score_eligible,
                    "exclusion_reason": exclusion_reason,
                })
                if ignored_for_scoring:
                    ignored_frames += 1
                if warmup_for_scoring:
                    warmup_frames += 1
                if not score_eligible:
                    excluded_frames += 1
                emitted_alerts = list(
                    getattr(pipeline, "_last_emitted_alerts", None) or []
                )
                if alert is not None and not emitted_alerts:
                    emitted_alerts = [alert]
                predicted = int(bool(emitted_alerts))
                record: dict[str, Any] = {
                    "frame_index": frame_index,
                    "timestamp_sec": timestamp_sec,
                    "score": score,
                    "label": label,
                    "predicted": predicted,
                    "decision": (
                        _decision(label, predicted) if score_eligible else "excluded"
                    ),
                    "score_eligible": score_eligible,
                    "exclusion_reason": exclusion_reason,
                    "pipeline_mode": pipeline.mode.value,
                    "scene_guard": scene_status,
                    "candidate_records": (
                        diagnostic.get("candidate_records", []) if diagnostic else []
                    ),
                    "suppression_reason": (
                        diagnostic.get("suppression_reason") if diagnostic else None
                    ),
                }
                if emitted_alerts:
                    serialized_alerts = []
                    for emitted_alert in emitted_alerts:
                        serialized = _serialize_alert(
                            emitted_alert,
                            timestamp_sec,
                        )
                        serialized["score_eligible"] = score_eligible
                        serialized["exclusion_reason"] = exclusion_reason
                        alerts.append(serialized)
                        serialized_alerts.append(serialized)
                    primary = serialized_alerts[0]
                    record["detection_type"] = primary["detection_type"]
                    record["category"] = primary["category"]
                    record["alert_id"] = primary["alert_id"]
                    record["alert_ids"] = [
                        item["alert_id"] for item in serialized_alerts
                    ]
                    record["detection_types"] = [
                        item["detection_type"] for item in serialized_alerts
                    ]
                    record["alert_count"] = len(serialized_alerts)
                records.append(record)
    finally:
        capture.release()
        try:
            pipeline.shutdown()
        except Exception:
            pass

    final_source_video_sha256 = _stream_sha256(video)
    final_annotation_sha256 = (
        _stream_sha256(annotation_file) if annotation_file is not None else None
    )
    if final_source_video_sha256 != source_video_sha256:
        raise RuntimeError("source video changed during evaluation")
    if final_annotation_sha256 != annotation_sha256:
        raise RuntimeError("annotation changed during evaluation")
    if (
        _evidence_artifact_sha256_if_exists(config_file)
        != config_artifact_sha256
    ):
        raise RuntimeError("config artifact changed during evaluation")

    processing_seconds = max(0.0, time.perf_counter() - processing_started)
    # `total_frames` means frames actually read by this invocation. Preserve
    # the separately reported source count even when container metadata is
    # inaccurate instead of silently clipping the audit trail.
    evaluated_total_frames = raw_frame_index
    expected_processed_frames = (
        math.ceil(evaluated_total_frames / sample_every)
        if evaluated_total_frames > 0
        else 0
    )
    evaluated_annotations = _clip_annotations(
        annotations,
        evaluated_frames=evaluated_total_frames,
    )
    annotation_inventory = (
        _annotation_inventory(
            raw_annotation,
            fps=fps,
            parsed=annotations,
            clipped=evaluated_annotations,
            total_frames=evaluated_total_frames,
        )
        if annotations is not None
        else None
    )
    score_eligible_records = [
        record for record in records if bool(record.get("score_eligible", False))
    ]
    score_eligible_scene_guard_records = [
        record
        for record in scene_guard_records
        if bool(record.get("score_eligible", False))
    ]
    sample_period_seconds = sample_every / fps
    scene_guard_summary_all = _summarize_scene_guard(
        scene_guard_records,
        sample_period_seconds=sample_period_seconds,
    )
    scene_guard_summary_score_eligible = _summarize_scene_guard(
        score_eligible_scene_guard_records,
        sample_period_seconds=sample_period_seconds,
    )
    return {
        "schema_version": PIPELINE_VIDEO_EVAL_SCHEMA_VERSION,
        "evaluation_origin": "fresh",
        "evaluation_mode": "pipeline",
        "video_path": str(video),
        "annotation_path": str(annotation_path) if annotation_path is not None else None,
        "source_video_sha256": source_video_sha256,
        "annotation_sha256": annotation_sha256,
        "source_artifacts_stable": True,
        "config_path": str(config_path),
        "config_artifact_sha256": config_artifact_sha256,
        "model_path": str(model_path) if model_path is not None else None,
        "model_version_id": runtime_model_version,
        "model_identity": model_identity,
        "production_parity": production_parity,
        "release_contract": release_contract,
        "candidate_confirmation_contract": candidate_confirmation_contract,
        "camera_id": camera_id,
        "sample_every": sample_every,
        "effective_eval_fps": fps / sample_every,
        "expected_processed_frames": expected_processed_frames,
        "warmup_seconds": warmup_seconds,
        "scene_reference": scene_reference,
        "fps": fps,
        "reported_fps": reported_fps if source_fps_available else None,
        "source_fps_available": source_fps_available,
        "source_total_frames": total_frames,
        "source_total_frames_known": source_total_frames_known,
        "source_exhausted": source_exhausted,
        "truncated_by_max_frames": truncated_by_max_frames,
        "source_read_error": source_read_error,
        "max_frames": max_frames,
        "total_frames": evaluated_total_frames,
        "processed_frames": len(records),
        "scored_frames": len(score_eligible_records),
        "ignored_frames": ignored_frames,
        "warmup_frames": warmup_frames,
        "excluded_frames": excluded_frames,
        "failed_frames": failed_frames,
        "false_positive_frames": [
            int(record["frame_index"]) for record in records if record["decision"] == "fp"
        ],
        "false_negative_frames": [
            int(record["frame_index"]) for record in records if record["decision"] == "fn"
        ],
        "metrics": None,
        "optimal_threshold": None,
        "annotations": evaluated_annotations,
        "annotation_inventory": annotation_inventory,
        "alerts": alerts,
        "frames": records,
        "processing_seconds": processing_seconds,
        "processing_fps": (
            len(scene_guard_records) / processing_seconds
            if processing_seconds > 0
            else None
        ),
        # Compatibility points at the score-eligible denominator used by the
        # release gate.  The all-frame view remains available for isolation and
        # state-transition audits.
        "scene_guard_summary": scene_guard_summary_score_eligible,
        "scene_guard_summary_all": scene_guard_summary_all,
        "scene_guard_summary_score_eligible": (
            scene_guard_summary_score_eligible
        ),
        "candidate_model_evidence_summary": _summarize_candidate_model_evidence(
            records
        ),
        "prospective_candidate_fusion_summary": (
            _summarize_prospective_candidate_fusion(records)
        ),
        "pipeline_stats": pipeline.stats.snapshot(),
    }


__all__ = [
    "OfflineCapture",
    "VirtualVideoClock",
    "PIPELINE_VIDEO_EVAL_SCHEMA_VERSION",
    "create_pipeline",
    "evaluate_pipeline_video",
    "_summarize_candidate_model_evidence",
    "_summarize_prospective_candidate_fusion",
]
