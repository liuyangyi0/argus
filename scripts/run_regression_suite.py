"""Run the Phase 1 real-video regression suite.

The suite consumes a YAML manifest under ``data/regression`` and wraps
``argus.validation.video_eval.evaluate_video`` with event-level aggregation:
false positives per hour, event recall, delay percentiles, score distribution,
detector-channel attribution, and protected SceneGuard availability for
production-pipeline replays.

Examples:
    python scripts/run_regression_suite.py \
        --manifest data/regression/manifest.yaml \
        --model data/models/latest/model.xml \
        --output-dir data/regression/reports/current

    python scripts/run_regression_suite.py \
        --from-eval-json data/regression/reports/current/clip.video_eval.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DEFAULT_FP_GAP_SECONDS = 60.0
DEFAULT_THRESHOLD = 0.7
DEFAULT_MIN_SCENE_GUARD_COVERAGE = 0.99
DEFAULT_MIN_NORMAL_HOURS = 2.0
DEFAULT_MIN_POSITIVE_EVENTS = 20
DEFAULT_MIN_POSITIVE_ANNOTATION_COVERAGE = 0.99
DEFAULT_MIN_CANDIDATE_COVERAGE = 0.99
DEFAULT_MIN_EFFECTIVE_EVAL_FPS = 5.0
DEFAULT_MIN_PROJECTILE_EVAL_FPS = 59.0
DEFAULT_MIN_PROJECTILE_EVENTS = 20
DEFAULT_MIN_EVENT_ELIGIBLE_FRAME_FRACTION = 0.5
DEFAULT_WARMUP_SECONDS = 30.0
RELEASE_ACCEPTANCE_POLICY_SCHEMA = "argus.regression-release-policy.v1"
PIPELINE_VIDEO_EVAL_SCHEMA_VERSION = "argus.pipeline-video-eval.v1"
RELEASE_EXPECTED_CHANNEL_BY_KIND = {
    "static_foreign": "persistent_change",
    "projectile": "fast_motion",
}
KNOWN_CHANNELS = {
    "fast_motion",
    "simplex",
    "persistent_change",
    "ssim_fallback",
    "anomaly",
    "water_leak",
    "plume",
}


@dataclass(frozen=True)
class RegressionClip:
    """One manifest row resolved relative to the regression root."""

    clip_id: str
    kind: str
    camera_id: str
    source_session_id: str | None = None
    video_path: Path | None = None
    annotation_path: Path | None = None
    scene_reference_path: Path | None = None
    scene_reference_frame_index: int | None = None
    candidate_events_path: Path | None = None
    evaluation_json_path: Path | None = None
    expected_detection_types: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    description: str = ""

    @property
    def is_negative(self) -> bool:
        return self.kind in {"negative", "normal"}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    if type(value) is not str or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _same_path(first: Path, second: Path) -> bool:
    try:
        return first.resolve(strict=True) == second.resolve(strict=True)
    except (OSError, ValueError):
        return False


def _normalized_artifact_path(value: Any) -> str | None:
    if type(value) is not str or not value:
        return None
    try:
        return str(Path(value).resolve(strict=False))
    except (OSError, ValueError):
        return None


def _evidence_artifact_sha256_if_present(path: Path | None) -> str | None:
    """Use the release contract's canonical file/tree hash without guessing."""
    if path is None:
        return None
    try:
        from argus.storage.release_evidence import evidence_artifact_sha256

        return evidence_artifact_sha256(path)
    except (FileNotFoundError, OSError, ValueError):
        return None


def _resolve(root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def load_manifest(manifest_path: str | Path) -> tuple[dict, list[RegressionClip]]:
    """Load and lightly validate a regression manifest."""
    path = Path(manifest_path)
    if path.is_dir():
        path = path / "manifest.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("regression manifest must be a YAML mapping")
    clips_raw = raw.get("clips") or []
    if not isinstance(clips_raw, list):
        raise ValueError("regression manifest 'clips' must be a list")

    root = path.parent
    default_camera_id = str(raw.get("default_camera_id") or "")
    clips: list[RegressionClip] = []
    seen: set[str] = set()
    for idx, item in enumerate(clips_raw):
        if not isinstance(item, dict):
            raise ValueError(f"manifest clip #{idx + 1} must be a mapping")
        clip_id = str(item.get("id") or "").strip()
        if not clip_id:
            raise ValueError(f"manifest clip #{idx + 1} is missing id")
        if clip_id in seen:
            raise ValueError(f"duplicate clip id: {clip_id}")
        seen.add(clip_id)

        kind = str(item.get("kind") or "negative").strip()
        camera_id = str(item.get("camera_id") or default_camera_id or "unknown")
        raw_source_session_id = item.get("source_session_id")
        source_session_id = None
        if raw_source_session_id is not None:
            if (
                type(raw_source_session_id) is not str
                or not raw_source_session_id
                or raw_source_session_id != raw_source_session_id.strip()
            ):
                raise ValueError(
                    f"manifest clip {clip_id} source_session_id must be a "
                    "non-empty string without surrounding whitespace"
                )
            source_session_id = raw_source_session_id
        expected_raw = item.get("expected_detection_types") or []
        if not isinstance(expected_raw, list) or any(
            type(value) is not str or not value.strip()
            for value in expected_raw
        ):
            raise ValueError(
                f"manifest clip {clip_id} expected_detection_types must be "
                "a list of non-empty strings"
            )
        expected = tuple(value.strip() for value in expected_raw)
        tags = tuple(str(x) for x in item.get("tags") or [])
        reference_frame = item.get("scene_reference_frame")
        if reference_frame is not None:
            try:
                reference_frame = int(reference_frame)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"manifest clip {clip_id} has invalid scene_reference_frame"
                ) from exc
            if reference_frame < 0:
                raise ValueError(
                    f"manifest clip {clip_id} scene_reference_frame must be >= 0"
                )
        clips.append(RegressionClip(
            clip_id=clip_id,
            kind=kind,
            camera_id=camera_id,
            source_session_id=source_session_id,
            video_path=_resolve(root, item.get("video")),
            annotation_path=_resolve(root, item.get("annotations")),
            scene_reference_path=_resolve(root, item.get("scene_reference")),
            scene_reference_frame_index=reference_frame,
            candidate_events_path=_resolve(root, item.get("candidate_events")),
            evaluation_json_path=_resolve(root, item.get("evaluation_json")),
            expected_detection_types=expected,
            tags=tags,
            description=str(item.get("description") or ""),
        ))
    return raw, clips


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _is_score_eligible(record: dict) -> bool:
    """Return whether a frame participates in quality metrics.

    Older detector-only evaluation JSON did not emit this field.  Treating a
    missing value as eligible preserves those reports while new pipeline
    replays can retain excluded frames for audit without contaminating scores.
    """
    return record.get("score_eligible", True) is True


def score_distribution(records: list[dict]) -> dict:
    """Summarize frame score distribution without numpy."""
    scores = [
        float(record["score"])
        for record in records
        if (
            _is_score_eligible(record)
            and record.get("score") is not None
            and math.isfinite(float(record["score"]))
        )
    ]
    if not scores:
        return {"count": 0}
    return {
        "count": len(scores),
        "min": min(scores),
        "max": max(scores),
        "mean": sum(scores) / len(scores),
        "p50": _percentile(scores, 0.50),
        "p90": _percentile(scores, 0.90),
        "p95": _percentile(scores, 0.95),
        "p99": _percentile(scores, 0.99),
    }


def _normalize_channel(raw: Any) -> str | None:
    if raw in (None, ""):
        return None
    value = str(raw).lower()
    if value in {"projectile", "fast_projectile"}:
        return "fast_motion"
    if value in {"frame_anomaly", "inference", "object", "hybrid"}:
        return "anomaly"
    if value in {"scene_change", "new_object"}:
        return "persistent_change"
    if value in {"safety_channel", "simple_detector"}:
        return "simplex"
    if value in {"visible_plume"}:
        return "plume"
    return value


def _channels_from_record(record: dict, default_channel: str) -> list[str]:
    channels: list[str] = []
    detection_types = record.get("detection_types")
    if isinstance(detection_types, list):
        for raw in detection_types:
            channel = _normalize_channel(raw)
            if channel:
                channels.append(channel)
    scalar_keys = ["channel", "candidate_type", "source", "mode"]
    if not isinstance(detection_types, list):
        scalar_keys.append("detection_type")
    for key in scalar_keys:
        channel = _normalize_channel(record.get(key))
        if channel:
            channels.append(channel)
    if not channels:
        channels.append(default_channel)
    return list(dict.fromkeys(channels))


def load_candidate_events(path: str | Path | None) -> list[dict]:
    """Load replay-style candidate events from a JSON sidecar."""
    if path is None:
        return []
    payload = _read_json(Path(path))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    for key in ("candidate_events", "events"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    frame_records = payload.get("candidate_records")
    if isinstance(frame_records, list):
        events: list[dict] = []
        for frame_index, records in enumerate(frame_records):
            if not isinstance(records, list):
                continue
            for record in records:
                if isinstance(record, dict):
                    event = dict(record)
                    event.setdefault("frame_index", frame_index)
                    events.append(event)
        return events
    return []


def _events_by_frame(candidate_events: list[dict]) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = {}
    for event in candidate_events:
        frame = event.get("frame_index", event.get("frame_number"))
        if frame is None:
            continue
        try:
            by_frame.setdefault(int(frame), []).append(event)
        except (TypeError, ValueError):
            continue
    return by_frame


def _channels_for_frame(
    record: dict,
    events_by_frame: dict[int, list[dict]],
    default_channel: str,
) -> list[str]:
    frame_index = record.get("frame_index")
    try:
        events = events_by_frame.get(int(frame_index), [])
    except (TypeError, ValueError):
        events = []
    channels: list[str] = []
    for event in events:
        channels.extend(_channels_from_record(event, default_channel))
    if channels:
        return channels
    return _channels_from_record(record, default_channel)


def _emitted_alert_channels(record: dict, default_channel: str) -> set[str]:
    """Return channels from emitted alerts, never from candidate-only sidecars."""
    if int(record.get("predicted") or 0) != 1:
        return set()
    raw_types = record.get("detection_types")
    if not isinstance(raw_types, list):
        raw_types = [record.get("detection_type")]
    channels = {
        channel
        for raw in raw_types
        if (channel := _normalize_channel(raw)) is not None
    }
    return channels or {default_channel}


def false_positive_events(
    result: dict,
    *,
    gap_seconds: float = DEFAULT_FP_GAP_SECONDS,
    candidate_events: list[dict] | None = None,
    default_channel: str = "anomaly",
) -> list[dict]:
    """Group frame-level false positives into event-level false positives."""
    frames = sorted(
        (
            record
            for record in result.get("frames", [])
            if _is_score_eligible(record) and record.get("decision") == "fp"
        ),
        key=lambda record: float(record.get("timestamp_sec") or 0.0),
    )
    if not frames:
        return []

    by_frame = _events_by_frame(candidate_events or [])
    events: list[list[dict]] = []
    current: list[dict] = []
    last_ts: float | None = None
    for record in frames:
        ts = float(record.get("timestamp_sec") or 0.0)
        if current and last_ts is not None and ts - last_ts > gap_seconds:
            events.append(current)
            current = []
        current.append(record)
        last_ts = ts
    if current:
        events.append(current)

    grouped: list[dict] = []
    for records in events:
        channels = Counter()
        for record in records:
            channels.update(_channels_for_frame(record, by_frame, default_channel))
        grouped.append({
            "start_frame": int(records[0].get("frame_index", 0)),
            "end_frame": int(records[-1].get("frame_index", 0)),
            "start_sec": float(records[0].get("timestamp_sec") or 0.0),
            "end_sec": float(records[-1].get("timestamp_sec") or 0.0),
            "frame_count": len(records),
            "max_score": max(float(r.get("score") or 0.0) for r in records),
            "mean_score": sum(float(r.get("score") or 0.0) for r in records) / len(records),
            "channels": dict(sorted(channels.items())),
            "primary_channel": channels.most_common(1)[0][0] if channels else default_channel,
        })
    return grouped


FrameInterval = tuple[int, int]


def _range_interval(
    item: dict,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> FrameInterval | None:
    """Parse and optionally clip one inclusive frame range."""
    try:
        start = int(item.get("start_frame", 0))
        end = int(item.get("end_frame", start))
    except (TypeError, ValueError):
        return None
    if minimum is not None:
        start = max(start, minimum)
    if maximum is not None:
        end = min(end, maximum)
    return (start, end) if end >= start else None


def _merge_intervals(intervals: list[FrameInterval]) -> list[FrameInterval]:
    """Return the union of inclusive integer intervals."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged: list[FrameInterval] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return merged


def _subtract_intervals(
    intervals: list[FrameInterval],
    exclusions: list[FrameInterval],
) -> list[FrameInterval]:
    """Subtract inclusive exclusions from inclusive base intervals."""
    remaining = _merge_intervals(intervals)
    for excluded_start, excluded_end in _merge_intervals(exclusions):
        updated: list[FrameInterval] = []
        for start, end in remaining:
            if excluded_end < start or excluded_start > end:
                updated.append((start, end))
                continue
            if start < excluded_start:
                updated.append((start, excluded_start - 1))
            if excluded_end < end:
                updated.append((excluded_end + 1, end))
        remaining = updated
        if not remaining:
            break
    return remaining


def _interval_frame_count(intervals: list[FrameInterval]) -> int:
    return sum(end - start + 1 for start, end in _merge_intervals(intervals))


def _frame_in_intervals(frame_index: int, intervals: list[FrameInterval]) -> bool:
    return any(start <= frame_index <= end for start, end in intervals)


def _unique_positive_ranges(
    ranges: list[dict],
) -> tuple[list[tuple[dict, FrameInterval, int]], dict[str, int | bool]]:
    """Merge overlapping annotations so they cannot inflate event evidence."""
    parsed: list[tuple[int, int, dict]] = []
    invalid_count = 0
    for item in ranges:
        interval = _range_interval(item)
        if interval is None:
            invalid_count += 1
            continue
        parsed.append((interval[0], interval[1], item))
    parsed.sort(key=lambda value: (value[0], value[1]))

    groups: list[tuple[dict, FrameInterval, int]] = []
    duplicate_count = 0
    overlap_count = 0
    adjacent_merge_count = 0
    for start, end, item in parsed:
        if not groups or start > groups[-1][1][1] + 1:
            groups.append((item, (start, end), 1))
            continue
        previous_item, (previous_start, previous_end), source_count = groups[-1]
        if start == previous_start and end == previous_end:
            duplicate_count += 1
        elif start == previous_end + 1:
            adjacent_merge_count += 1
        else:
            overlap_count += 1
        groups[-1] = (
            previous_item,
            (previous_start, max(previous_end, end)),
            source_count + 1,
        )

    return groups, {
        "complete": not (invalid_count or duplicate_count or overlap_count),
        "invalid_count": invalid_count,
        "duplicate_count": duplicate_count,
        "overlap_count": overlap_count,
        "adjacent_merge_count": adjacent_merge_count,
    }


def _score_excluded_intervals(result: dict) -> list[FrameInterval]:
    """Expand explicitly non-score-eligible samples into source-frame spans."""
    records = [record for record in result.get("frames") or [] if isinstance(record, dict)]
    if not any("score_eligible" in record for record in records):
        return []
    stride = max(1, int(result.get("sample_every") or 1))
    total_frames = int(result.get("total_frames") or 0)
    maximum = total_frames - 1 if total_frames > 0 else None
    intervals: list[FrameInterval] = []
    for record in records:
        if _is_score_eligible(record):
            continue
        try:
            frame_index = int(record.get("frame_index", -1))
        except (TypeError, ValueError):
            continue
        if frame_index < 0 or (maximum is not None and frame_index > maximum):
            continue
        end = frame_index + stride - 1
        if maximum is not None:
            end = min(end, maximum)
        intervals.append((frame_index, end))
    return _merge_intervals(intervals)


def effective_normal_seconds(result: dict) -> float:
    """Return video seconds eligible for false-positive rate."""
    fps = float(result.get("fps") or 30.0)
    total_frames = int(result.get("total_frames") or 0)
    if total_frames <= 0:
        scored = len(result.get("frames") or [])
        sample_every = int(result.get("sample_every") or 1)
        total_frames = scored * max(1, sample_every)
    annotations = result.get("annotations") or {}
    last_frame = total_frames - 1
    excluded = _merge_intervals([
        interval
        for key in ("positive_ranges", "ignore_ranges")
        for item in annotations.get(key) or []
        if isinstance(item, dict)
        if (
            interval := _range_interval(
                item,
                minimum=0,
                maximum=last_frame,
            )
        ) is not None
    ])

    records = [record for record in result.get("frames") or [] if isinstance(record, dict)]
    if any("score_eligible" in record for record in records):
        # A sampled observation represents its source-frame stride.  Building
        # explicit intervals avoids crediting warmup/ignored/failed periods as
        # normal surveillance time and clips the final sample at EOF.
        stride = max(1, int(result.get("sample_every") or 1))
        eligible = _merge_intervals([
            (frame_index, min(last_frame, frame_index + stride - 1))
            for record in records
            if _is_score_eligible(record)
            if (frame_index := int(record.get("frame_index", -1))) >= 0
            if frame_index <= last_frame
        ])
    else:
        # Legacy detector reports contain only metric records and describe
        # coverage using total_frames rather than a per-frame eligibility bit.
        eligible = [(0, last_frame)] if total_frames > 0 else []

    normal = _subtract_intervals(eligible, excluded)
    return _interval_frame_count(normal) / fps


def positive_event_summary(
    result: dict,
    *,
    candidate_events: list[dict] | None = None,
    expected_detection_types: tuple[str, ...] = (),
    default_channel: str = "anomaly",
    min_eligible_frame_fraction: float = (
        DEFAULT_MIN_EVENT_ELIGIBLE_FRAME_FRACTION
    ),
) -> dict:
    """Summarize event-level recall and detection delay for positive ranges."""
    if not 0.0 <= min_eligible_frame_fraction <= 1.0:
        raise ValueError("min_eligible_frame_fraction must be between 0 and 1")
    annotations = result.get("annotations") or {}
    ranges = [
        item for item in annotations.get("positive_ranges") or []
        if isinstance(item, dict)
    ]
    if not ranges:
        return {
            "events_total": 0,
            "events_hit": 0,
            "recall": None,
            "delay_p95_sec": None,
            "fully_ignored_events": 0,
            "fully_excluded_events": 0,
            "fully_score_excluded_events": 0,
            "insufficient_eligible_fraction_events": 0,
            "annotated_events_total": 0,
            "annotation_coverage_ratio": None,
            "positive_event_uniqueness_evidence_complete": True,
            "duplicate_positive_interval_count": 0,
            "overlapping_positive_interval_count": 0,
            "adjacent_positive_interval_merge_count": 0,
            "invalid_positive_interval_count": 0,
            "min_eligible_frame_fraction": min_eligible_frame_fraction,
            "events": [],
        }

    by_frame = _events_by_frame(candidate_events or [])
    records = sorted(
        (
            record
            for record in result.get("frames") or []
            if isinstance(record, dict) and _is_score_eligible(record)
        ),
        key=lambda record: int(record.get("frame_index", -1)),
    )
    events: list[dict] = []
    delays: list[float] = []
    fully_ignored_events = 0
    fully_excluded_events = 0
    fully_score_excluded_events = 0
    insufficient_eligible_fraction_events = 0
    expected_channels = {_normalize_channel(item) for item in expected_detection_types}
    expected_channels.discard(None)
    ignore_intervals = _merge_intervals([
        interval
        for ignored in annotations.get("ignore_ranges") or []
        if isinstance(ignored, dict)
        if (interval := _range_interval(ignored)) is not None
    ])
    score_excluded_intervals = _score_excluded_intervals(result)
    unique_ranges, uniqueness = _unique_positive_ranges(ranges)

    for item, positive_interval, source_annotation_count in unique_ranges:
        start, end = positive_interval
        annotation_eligible_ranges = _subtract_intervals(
            [positive_interval],
            ignore_intervals,
        )
        if not annotation_eligible_ranges:
            # An annotation that explicitly excludes the entire positive span
            # supplies no recall evidence and must not be reported as a miss.
            fully_ignored_events += 1
            fully_excluded_events += 1
            continue
        effective_ranges = _subtract_intervals(
            annotation_eligible_ranges,
            score_excluded_intervals,
        )
        if not effective_ranges:
            # Warmup and any future explicit score exclusions are quality
            # denominator exclusions just like annotation ignore ranges.
            fully_excluded_events += 1
            fully_score_excluded_events += 1
            continue
        eligible_frame_count = _interval_frame_count(effective_ranges)
        annotated_frame_count = end - start + 1
        eligible_frame_fraction = eligible_frame_count / annotated_frame_count
        if eligible_frame_fraction < min_eligible_frame_fraction:
            fully_excluded_events += 1
            insufficient_eligible_fraction_events += 1
            continue
        effective_start = effective_ranges[0][0]
        effective_end = effective_ranges[-1][1]
        in_range = [
            record for record in records
            if _frame_in_intervals(
                int(record.get("frame_index", -1)),
                effective_ranges,
            )
        ]
        emitted_alert_records = [
            record
            for record in in_range
            if int(record.get("predicted") or 0) == 1
        ]
        semantic_match: bool | None = None
        if expected_channels:
            hits = [
                record
                for record in emitted_alert_records
                if _emitted_alert_channels(record, default_channel)
                & expected_channels
            ]
            semantic_match = bool(hits) if emitted_alert_records else None
        else:
            hits = emitted_alert_records

        delay = None
        if hits:
            fps = float(result.get("fps") or 30.0)
            first_ts = float(hits[0].get("timestamp_sec") or 0.0)
            start_ts = effective_start / fps
            delay = max(0.0, first_ts - start_ts)
            delays.append(delay)

        candidate_types: Counter[str] = Counter()
        candidate_outcomes: Counter[str] = Counter()
        suppression_reasons: Counter[str] = Counter()
        alert_types: Counter[str] = Counter()
        for record in in_range:
            frame_reason = record.get("suppression_reason")
            if frame_reason:
                suppression_reasons[str(frame_reason)] += 1
            if int(record.get("predicted") or 0) == 1:
                for detection_type in record.get("detection_types") or []:
                    alert_types[str(detection_type)] += 1
                if not record.get("detection_types") and record.get("detection_type"):
                    alert_types[str(record["detection_type"])] += 1
            try:
                frame_candidates = by_frame.get(
                    int(record.get("frame_index", -1)),
                    [],
                )
            except (TypeError, ValueError):
                frame_candidates = []
            for candidate in [
                *(record.get("candidate_records") or []),
                *frame_candidates,
            ]:
                if not isinstance(candidate, dict):
                    continue
                candidate_type = str(candidate.get("candidate_type") or "unknown")
                decision = str(candidate.get("decision") or "unknown")
                candidate_types[candidate_type] += 1
                candidate_outcomes[f"{candidate_type}:{decision}"] += 1
                candidate_reason = candidate.get("suppression_reason")
                if candidate_reason:
                    suppression_reasons[str(candidate_reason)] += 1

        if hits:
            diagnosis = "alert_emitted"
        elif emitted_alert_records:
            diagnosis = "wrong_detection_channel"
        elif candidate_types:
            diagnosis = "candidate_suppressed"
        else:
            diagnosis = "no_candidate"

        events.append({
            "name": item.get("name") or item.get("label") or "",
            "start_frame": start,
            "end_frame": end,
            "source_annotation_count": source_annotation_count,
            "eligible_frame_count": eligible_frame_count,
            "annotated_frame_count": annotated_frame_count,
            "eligible_frame_fraction": eligible_frame_fraction,
            "effective_start_frame": effective_start,
            "effective_end_frame": effective_end,
            "effective_ranges": [
                {"start_frame": range_start, "end_frame": range_end}
                for range_start, range_end in effective_ranges
            ],
            "hit": bool(hits),
            "semantic_match": semantic_match,
            "first_detection_frame": int(hits[0]["frame_index"]) if hits else None,
            "delay_sec": delay,
            "max_score": max((float(r.get("score") or 0.0) for r in in_range), default=None),
            "diagnosis": diagnosis,
            "candidate_type_counts": dict(candidate_types.most_common()),
            "candidate_outcome_counts": dict(candidate_outcomes.most_common()),
            "suppression_reason_counts": dict(suppression_reasons.most_common()),
            "alert_detection_type_counts": dict(alert_types.most_common()),
            "wrong_channel_alert_count": (
                len(emitted_alert_records) - len(hits)
                if expected_channels
                else 0
            ),
        })

    events_hit = sum(1 for item in events if item["hit"])
    annotated_events_total = len(events) + fully_excluded_events
    return {
        "events_total": len(events),
        "events_hit": events_hit,
        "recall": events_hit / len(events) if events else None,
        "delay_p95_sec": _percentile(delays, 0.95),
        "fully_ignored_events": fully_ignored_events,
        "fully_excluded_events": fully_excluded_events,
        "fully_score_excluded_events": fully_score_excluded_events,
        "insufficient_eligible_fraction_events": (
            insufficient_eligible_fraction_events
        ),
        "annotated_events_total": annotated_events_total,
        "annotation_coverage_ratio": (
            len(events) / annotated_events_total
            if annotated_events_total
            else None
        ),
        "positive_event_uniqueness_evidence_complete": uniqueness["complete"],
        "duplicate_positive_interval_count": uniqueness["duplicate_count"],
        "overlapping_positive_interval_count": uniqueness["overlap_count"],
        "adjacent_positive_interval_merge_count": uniqueness[
            "adjacent_merge_count"
        ],
        "invalid_positive_interval_count": uniqueness["invalid_count"],
        "min_eligible_frame_fraction": min_eligible_frame_fraction,
        "events": events,
    }


def channel_totals(fp_events: list[dict]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for event in fp_events:
        counter.update(event.get("channels") or {})
    return dict(sorted(counter.items()))


def summarize_clip_result(
    result: dict,
    clip: RegressionClip,
    *,
    candidate_events: list[dict] | None = None,
    fp_gap_seconds: float = DEFAULT_FP_GAP_SECONDS,
    default_channel: str = "anomaly",
) -> dict:
    """Build a clip-level event summary from an evaluate_video result."""
    frames = list(result.get("frames") or [])
    fp_events = false_positive_events(
        result,
        gap_seconds=fp_gap_seconds,
        candidate_events=candidate_events,
        default_channel=default_channel,
    )
    normal_seconds = effective_normal_seconds(result)
    positive = positive_event_summary(
        result,
        candidate_events=candidate_events,
        expected_detection_types=clip.expected_detection_types,
        default_channel=default_channel,
    )
    normal_hours = normal_seconds / 3600.0
    fp_per_hour = len(fp_events) / normal_hours if normal_hours > 0 else None
    metrics = result.get("metrics")
    legacy_scene_guard = result.get("scene_guard_summary")
    scene_guard_score_eligible = result.get(
        "scene_guard_summary_score_eligible",
        legacy_scene_guard,
    )
    scene_guard_all = result.get("scene_guard_summary_all", legacy_scene_guard)
    result_camera_id = result.get("camera_id")
    camera_id = (
        str(result_camera_id)
        if clip.camera_id == "unknown" and result_camera_id not in (None, "")
        else clip.camera_id
    )
    return {
        "clip_id": clip.clip_id,
        "kind": clip.kind,
        "camera_id": camera_id,
        "source_session_id": clip.source_session_id,
        "schema_version": result.get("schema_version"),
        "evaluation_origin": result.get("evaluation_origin"),
        "cached_evaluation_validation": result.get(
            "cached_evaluation_validation"
        ),
        "video_path": (
            str(clip.video_path)
            if clip.video_path is not None
            else result.get("video_path")
        ),
        "annotation_path": (
            str(clip.annotation_path)
            if clip.annotation_path is not None
            else result.get("annotation_path")
        ),
        "config_path": result.get("config_path"),
        "config_artifact_sha256": result.get("config_artifact_sha256"),
        "source_video_sha256": result.get("source_video_sha256"),
        "annotation_sha256": result.get("annotation_sha256"),
        "source_artifacts_stable": result.get("source_artifacts_stable"),
        "annotation_inventory": result.get("annotation_inventory"),
        "scene_reference": result.get("scene_reference"),
        "model_path": result.get("model_path"),
        "model_version_id": result.get("model_version_id"),
        "model_identity": result.get("model_identity"),
        "production_parity": result.get("production_parity"),
        "candidate_confirmation_contract": result.get(
            "candidate_confirmation_contract"
        ),
        "release_contract": result.get("release_contract"),
        # ``scene_guard`` remains the compatibility key and intentionally
        # means score-eligible coverage.  The explicit keys preserve both the
        # release-gate view and the all-frame operational audit.
        "scene_guard": scene_guard_score_eligible,
        "scene_guard_summary_score_eligible": scene_guard_score_eligible,
        "scene_guard_summary_all": scene_guard_all,
        "tags": list(clip.tags),
        "description": clip.description,
        "status": "evaluated",
        "evaluation_mode": result.get("evaluation_mode") or "detector",
        "fps": result.get("fps"),
        "reported_fps": result.get("reported_fps"),
        "sample_every": result.get("sample_every"),
        "effective_eval_fps": result.get("effective_eval_fps"),
        "expected_processed_frames": result.get("expected_processed_frames"),
        "source_fps_available": result.get("source_fps_available"),
        "source_total_frames": result.get("source_total_frames"),
        "source_total_frames_known": result.get("source_total_frames_known"),
        "source_exhausted": result.get("source_exhausted"),
        "truncated_by_max_frames": result.get("truncated_by_max_frames"),
        "source_read_error": result.get("source_read_error"),
        "max_frames": result.get("max_frames"),
        "total_frames": result.get("total_frames"),
        "processed_frames": result.get("processed_frames", result.get("scored_frames")),
        "scored_frames": result.get("scored_frames"),
        "excluded_frames": result.get("excluded_frames"),
        "ignored_frames": result.get("ignored_frames"),
        "warmup_frames": result.get("warmup_frames"),
        "failed_frames": len(result.get("failed_frames") or []),
        "normal_seconds": normal_seconds,
        "false_positive_events": fp_events,
        "false_positive_event_count": len(fp_events),
        "false_positive_events_per_hour": fp_per_hour,
        "channel_totals": channel_totals(fp_events),
        "positive_events": positive,
        "expected_detection_types": list(clip.expected_detection_types),
        "score_distribution": score_distribution(frames),
        "frame_metrics": metrics,
        "candidate_model_evidence_summary": result.get(
            "candidate_model_evidence_summary"
        ),
        "prospective_candidate_fusion_summary": result.get(
            "prospective_candidate_fusion_summary"
        ),
    }


def _aggregate_candidate_model_evidence(summaries: list[dict]) -> dict:
    counters = {
        key: Counter()
        for key in ("status_counts", "output_space_counts", "calibration_id_counts")
    }

    def total(key: str, *, fallback: str | None = None) -> int:
        return sum(
            int(summary.get(key, summary.get(fallback, 0)) or 0)
            for summary in summaries
        )

    for summary in summaries:
        for key, counter in counters.items():
            counter.update(summary.get(key) or {})

    extraction_eligible = total(
        "map_extraction_eligible_candidates",
        fallback="eligible_candidates",
    )
    extracted = total(
        "map_extracted_candidates",
        fallback="evaluated_candidates",
    )
    calibrated_eligible = total(
        "calibrated_evaluation_eligible_candidates",
        fallback="evaluated_candidates",
    )
    calibrated_evaluated = total(
        "calibrated_evaluated_candidates",
        fallback="evaluated_candidates",
    )
    evaluated = total("evaluated_candidates")
    accepted = total("accepted_candidates")
    return {
        "clip_count_with_evidence": len(summaries),
        "observed_candidates": total("observed_candidates"),
        "all_observed_candidates": total(
            "all_observed_candidates",
            fallback="observed_candidates",
        ),
        "excluded_candidate_observations": total("excluded_candidate_observations"),
        "not_observed_candidate_records": total("not_observed_candidate_records"),
        "status_counts": dict(sorted(counters["status_counts"].items())),
        "evaluated_candidates": evaluated,
        "eligible_candidates": extraction_eligible,
        "telemetry_coverage_ratio": (
            extracted / extraction_eligible if extraction_eligible else None
        ),
        "map_extraction_eligible_candidates": extraction_eligible,
        "map_extracted_candidates": extracted,
        "map_extraction_coverage_ratio": (
            extracted / extraction_eligible if extraction_eligible else None
        ),
        "calibrated_evaluation_eligible_candidates": calibrated_eligible,
        "calibrated_evaluated_candidates": calibrated_evaluated,
        "calibrated_evaluation_coverage_ratio": (
            calibrated_evaluated / calibrated_eligible
            if calibrated_eligible
            else None
        ),
        "accepted_candidates": accepted,
        "rejected_candidates": total("rejected_candidates"),
        "rule_accept_rate": accepted / evaluated if evaluated else None,
        "nullable_contract_violations": total("nullable_contract_violations"),
        "extraction_contract_violations": total("extraction_contract_violations"),
        "output_space_counts": dict(sorted(counters["output_space_counts"].items())),
        "calibration_id_counts": dict(sorted(counters["calibration_id_counts"].items())),
    }


def _aggregate_prospective_candidate_fusion(summaries: list[dict]) -> dict:
    counters = {
        key: Counter()
        for key in (
            "decision_status_counts",
            "temporal_status_counts",
            "temporal_reason_counts",
            "mode_counts",
            "fusion_policy_counts",
        )
    }

    def total(key: str, *, fallback: str | None = None) -> int:
        return sum(
            int(summary.get(key, summary.get(fallback, 0)) or 0)
            for summary in summaries
        )

    for summary in summaries:
        for key, counter in counters.items():
            counter.update(summary.get(key) or {})

    attribution_summaries = [
        attribution
        for summary in summaries
        if isinstance(attribution := summary.get("anomaly_attribution"), dict)
    ]
    attribution_reasons: Counter[str] = Counter()
    for attribution in attribution_summaries:
        attribution_reasons.update(attribution.get("reason_counts") or {})

    def attribution_total(key: str) -> int:
        return sum(int(item.get(key) or 0) for item in attribution_summaries)

    hot_summaries = [
        hot
        for attribution in attribution_summaries
        if isinstance(hot := attribution.get("hot_coverage_summary"), dict)
    ]
    hot_count = sum(int(item.get("count") or 0) for item in hot_summaries)
    hot_minima = [
        float(item["min"])
        for item in hot_summaries
        if item.get("min") is not None
    ]
    hot_maxima = [
        float(item["max"])
        for item in hot_summaries
        if item.get("max") is not None
    ]
    hot_weighted_sum = sum(
        float(item["mean"]) * int(item.get("count") or 0)
        for item in hot_summaries
        if item.get("mean") is not None
    )

    evaluated = total("evaluated_candidates")
    accepted = total("accepted_candidates")
    attribution_decided = attribution_total("decided_frames")
    attribution_would_suppress = attribution_total("would_suppress_frames")
    return {
        "clip_count_with_evidence": len(summaries),
        "observed_candidates": total("observed_candidates"),
        "all_observed_candidates": total(
            "all_observed_candidates",
            fallback="observed_candidates",
        ),
        "excluded_candidate_observations": total("excluded_candidate_observations"),
        "not_observed_candidate_records": total("not_observed_candidate_records"),
        "evaluated_candidates": evaluated,
        "accepted_candidates": accepted,
        "pending_candidates": total("pending_candidates"),
        "rejected_candidates": total("rejected_candidates"),
        "unavailable_candidates": total("unavailable_candidates"),
        "accept_rate": accepted / evaluated if evaluated else None,
        "decision_status_counts": dict(sorted(counters["decision_status_counts"].items())),
        "temporal_status_counts": dict(sorted(counters["temporal_status_counts"].items())),
        "temporal_reason_counts": dict(sorted(counters["temporal_reason_counts"].items())),
        "mode_counts": dict(sorted(counters["mode_counts"].items())),
        "fusion_policy_counts": dict(
            sorted(counters["fusion_policy_counts"].items())
        ),
        "confirmed_candidates": total("confirmed_candidates"),
        "anomaly_attribution": {
            "clip_count_with_evidence": len(attribution_summaries),
            "observed_frames": attribution_total("observed_frames"),
            "excluded_frame_observations": attribution_total(
                "excluded_frame_observations"
            ),
            "decided_frames": attribution_decided,
            "would_suppress_frames": attribution_would_suppress,
            "would_preserve_frames": attribution_total("would_preserve_frames"),
            "unavailable_frames": attribution_total("unavailable_frames"),
            "effective_suppressed_frames": attribution_total(
                "effective_suppressed_frames"
            ),
            "suppress_rate": (
                attribution_would_suppress / attribution_decided
                if attribution_decided
                else None
            ),
            "reason_counts": dict(sorted(attribution_reasons.items())),
            "hot_coverage_summary": {
                "count": hot_count,
                "min": min(hot_minima) if hot_minima else None,
                "max": max(hot_maxima) if hot_maxima else None,
                "mean": hot_weighted_sum / hot_count if hot_count else None,
            },
        },
    }


def aggregate_suite(
    clip_summaries: list[dict],
    *,
    min_scene_guard_coverage: float = DEFAULT_MIN_SCENE_GUARD_COVERAGE,
    min_effective_eval_fps: float = DEFAULT_MIN_EFFECTIVE_EVAL_FPS,
    min_projectile_eval_fps: float = DEFAULT_MIN_PROJECTILE_EVAL_FPS,
    expected_clip_count: int | None = None,
    evaluated_artifact_hashes: dict[str, str | None] | None = None,
) -> dict:
    """Aggregate clip summaries into suite-level acceptance numbers."""
    if not 0.0 <= min_scene_guard_coverage <= 1.0:
        raise ValueError("min_scene_guard_coverage must be between 0 and 1")
    if not math.isfinite(min_effective_eval_fps) or min_effective_eval_fps <= 0:
        raise ValueError("min_effective_eval_fps must be finite and > 0")
    if not math.isfinite(min_projectile_eval_fps) or min_projectile_eval_fps <= 0:
        raise ValueError("min_projectile_eval_fps must be finite and > 0")
    if expected_clip_count is None:
        expected_clip_count = len(clip_summaries)
    try:
        expected_clip_count = int(expected_clip_count)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected_clip_count must be an integer") from exc
    if expected_clip_count < 0:
        raise ValueError("expected_clip_count must be >= 0")
    evaluated = [item for item in clip_summaries if item.get("status") == "evaluated"]

    def _annotation_inventory_complete(value: Any) -> bool:
        expected_keys = {
            "raw_range_counts",
            "parsed_range_counts",
            "clipped_range_counts",
            "invalid_range_count",
            "out_of_bounds_range_count",
            "loader_normalized_range_count",
            "clipped_or_dropped_range_count",
            "all_ranges_within_source",
        }
        if not isinstance(value, dict) or set(value) != expected_keys:
            return False
        counts: list[dict] = []
        for key in (
            "raw_range_counts",
            "parsed_range_counts",
            "clipped_range_counts",
        ):
            item = value[key]
            if (
                not isinstance(item, dict)
                or set(item) != {"positive", "ignore"}
                or any(type(item[name]) is not int for name in item)
                or any(item[name] < 0 for name in item)
            ):
                return False
            counts.append(item)
        audit_count_keys = (
            "invalid_range_count",
            "out_of_bounds_range_count",
            "loader_normalized_range_count",
            "clipped_or_dropped_range_count",
        )
        return bool(
            all(type(value[key]) is int and value[key] == 0 for key in audit_count_keys)
            and value.get("all_ranges_within_source") is True
            and counts[0] == counts[1] == counts[2]
        )

    def _source_inventory_complete(item: dict) -> bool:
        exact_int_fields = (
            "failed_frames",
            "source_total_frames",
            "total_frames",
            "sample_every",
            "processed_frames",
            "expected_processed_frames",
        )
        if any(type(item.get(key)) is not int for key in exact_int_fields):
            return False
        failed = item["failed_frames"]
        source_total = item["source_total_frames"]
        evaluated_total = item["total_frames"]
        sample_every = item["sample_every"]
        processed = item["processed_frames"]
        expected_processed = item["expected_processed_frames"]
        fps_raw = item.get("fps")
        reported_fps_raw = item.get("reported_fps")
        effective_fps_raw = item.get("effective_eval_fps")
        if any(
            type(value) not in (int, float)
            for value in (fps_raw, reported_fps_raw, effective_fps_raw)
        ):
            return False
        fps = float(fps_raw)
        reported_fps = float(reported_fps_raw)
        effective_fps = float(effective_fps_raw)
        expected_from_inventory = (
            math.ceil(evaluated_total / sample_every)
            if sample_every >= 1 and evaluated_total > 0
            else 0
        )
        cache_contract = item.get("cached_evaluation_validation")
        origin = item.get("evaluation_origin")
        kind = str(item.get("kind") or "").strip().lower()
        required_eval_fps = (
            min_projectile_eval_fps
            if kind == "projectile"
            else min_effective_eval_fps
        )
        return bool(
            failed == 0
            and math.isfinite(fps)
            and fps > 0.0
            and math.isfinite(reported_fps)
            and reported_fps == fps
            and math.isfinite(effective_fps)
            and sample_every >= 1
            and effective_fps == fps / sample_every
            and effective_fps >= required_eval_fps
            and (
                kind != "projectile"
                or reported_fps >= min_projectile_eval_fps
            )
            and expected_processed == expected_from_inventory
            and processed == expected_processed
            and item.get("schema_version")
            == PIPELINE_VIDEO_EVAL_SCHEMA_VERSION
            and origin in {"fresh", "cached"}
            and (
                origin == "fresh"
                or (
                    isinstance(cache_contract, dict)
                    and cache_contract.get("valid") is True
                )
            )
            and _is_sha256(item.get("source_video_sha256"))
            and _is_sha256(item.get("annotation_sha256"))
            and item.get("source_artifacts_stable") is True
            and _annotation_inventory_complete(item.get("annotation_inventory"))
            and item.get("source_fps_available") is True
            and item.get("source_total_frames_known") is True
            and source_total > 0
            # A container-reported count that is either larger or smaller
            # than the frames actually decoded is not verified inventory.
            # Using >= would let an under-reported/corrupt index pass merely
            # because the decoder happened to continue beyond it.
            and evaluated_total == source_total
            and item.get("source_exhausted") is True
            and item.get("truncated_by_max_frames") is False
            and item.get("source_read_error") is False
        )

    camera_ids = [
        item.get("camera_id")
        for item in evaluated
        if type(item.get("camera_id")) is str
        and item["camera_id"].strip()
        and item["camera_id"] != "unknown"
    ]
    evaluated_camera_id = (
        camera_ids[0]
        if len(camera_ids) == len(evaluated) and len(set(camera_ids)) == 1
        else None
    )
    single_camera_evidence_complete = bool(evaluated and evaluated_camera_id)
    artifact_hashes = evaluated_artifact_hashes or {}
    bound_artifact_hashes = {
        key: value if _is_sha256(value) else None
        for key in (
            "config_sha256",
            "scene_profile_sha256",
            "baseline_sha256",
            "corpus_sha256",
        )
        if (value := artifact_hashes.get(key)) is not None
    }
    pipeline_only = bool(
        evaluated
        and all(item.get("evaluation_mode") == "pipeline" for item in evaluated)
    )
    evaluated_artifacts = {
        "camera_id": evaluated_camera_id,
        "config_sha256": (
            bound_artifact_hashes.get("config_sha256") if pipeline_only else None
        ),
        "scene_profile_sha256": (
            bound_artifact_hashes.get("scene_profile_sha256")
            if pipeline_only
            else None
        ),
        "baseline_sha256": (
            bound_artifact_hashes.get("baseline_sha256") if pipeline_only else None
        ),
        "corpus_sha256": (
            bound_artifact_hashes.get("corpus_sha256") if pipeline_only else None
        ),
    }
    source_video_hashes = [
        item.get("source_video_sha256") for item in evaluated
    ]
    valid_source_video_hashes = [
        value for value in source_video_hashes if _is_sha256(value)
    ]
    source_video_paths = [
        path
        for item in evaluated
        if (path := _normalized_artifact_path(item.get("video_path")))
        is not None
    ]
    source_session_ids = [item.get("source_session_id") for item in evaluated]
    valid_source_session_ids = [
        value
        for value in source_session_ids
        if type(value) is str and value and value == value.strip()
    ]
    source_session_identity_evidence_complete = bool(
        evaluated
        and len(valid_source_session_ids) == len(evaluated)
        and len(set(valid_source_session_ids)) == len(evaluated)
    )
    unique_source_artifacts_evidence_complete = bool(
        evaluated
        and len(source_video_hashes) == len(evaluated)
        and len(valid_source_video_hashes) == len(evaluated)
        and len(set(valid_source_video_hashes)) == len(evaluated)
        and len(source_video_paths) == len(evaluated)
        and len(set(source_video_paths)) == len(evaluated)
        and source_session_identity_evidence_complete
    )

    source_inventory_evidence_complete = bool(
        expected_clip_count > 0
        and len(clip_summaries) == expected_clip_count
        and len(evaluated) == expected_clip_count
        and all(_source_inventory_complete(item) for item in evaluated)
        and single_camera_evidence_complete
        and unique_source_artifacts_evidence_complete
    )
    kind_specific_sampling_evidence_complete = bool(
        evaluated
        and all(
            type(item.get("reported_fps")) in (int, float)
            and type(item.get("effective_eval_fps")) in (int, float)
            and math.isfinite(float(item["reported_fps"]))
            and math.isfinite(float(item["effective_eval_fps"]))
            and float(item["effective_eval_fps"])
            >= (
                min_projectile_eval_fps
                if str(item.get("kind") or "").strip().lower()
                == "projectile"
                else min_effective_eval_fps
            )
            and (
                str(item.get("kind") or "").strip().lower()
                != "projectile"
                or float(item["reported_fps"]) >= min_projectile_eval_fps
            )
            for item in evaluated
        )
    )
    normal_hours = sum(float(item.get("normal_seconds") or 0.0) for item in evaluated) / 3600.0
    fp_events = sum(int(item.get("false_positive_event_count") or 0) for item in evaluated)
    positive_total = sum(
        int((item.get("positive_events") or {}).get("events_total") or 0)
        for item in evaluated
    )
    positive_hit = sum(
        int((item.get("positive_events") or {}).get("events_hit") or 0)
        for item in evaluated
    )
    positive_events_by_kind: Counter[str] = Counter()
    positive_hits_by_kind: Counter[str] = Counter()
    for item in evaluated:
        kind = str(item.get("kind") or "unknown")
        positive = item.get("positive_events") or {}
        positive_events_by_kind[kind] += int(positive.get("events_total") or 0)
        positive_hits_by_kind[kind] += int(positive.get("events_hit") or 0)
    positive_recall_by_kind = {
        kind: positive_hits_by_kind[kind] / count
        for kind, count in positive_events_by_kind.items()
        if count > 0
    }
    static_foreign_total = positive_events_by_kind["static_foreign"]
    static_foreign_hit = positive_hits_by_kind["static_foreign"]
    projectile_total = positive_events_by_kind["projectile"]
    projectile_hit = positive_hits_by_kind["projectile"]
    static_foreign_recall = (
        static_foreign_hit / static_foreign_total
        if static_foreign_total > 0
        else None
    )
    fully_excluded_positive_total = sum(
        int((item.get("positive_events") or {}).get("fully_excluded_events") or 0)
        for item in evaluated
    )
    annotated_positive_total = positive_total + fully_excluded_positive_total
    positive_annotation_coverage = (
        positive_total / annotated_positive_total
        if annotated_positive_total > 0
        else None
    )
    positive_clips = [
        item
        for item in evaluated
        if str(item.get("kind") or "").strip().lower()
        not in {"negative", "normal", "background"}
    ]

    def _expected_channel_policy_matches(item: dict) -> bool:
        kind = str(item.get("kind") or "").strip().lower()
        expected = item.get("expected_detection_types")
        if not isinstance(expected, list) or any(
            type(channel) is not str or not channel.strip()
            for channel in expected
        ):
            return False
        if kind in {"negative", "normal", "background"}:
            return not expected
        canonical = RELEASE_EXPECTED_CHANNEL_BY_KIND.get(kind)
        if canonical is None or len(expected) != 1:
            return False
        return _normalize_channel(expected[0]) == canonical

    expected_detection_type_policy_evidence_complete = bool(
        evaluated and all(_expected_channel_policy_matches(item) for item in evaluated)
    )
    positive_expected_detection_types_evidence_complete = bool(
        positive_clips
        and expected_detection_type_policy_evidence_complete
    )
    positive_event_uniqueness_evidence_complete = bool(
        positive_clips
        and all(
            (item.get("positive_events") or {}).get(
                "positive_event_uniqueness_evidence_complete"
            )
            is True
            for item in positive_clips
        )
    )
    positive_annotation_coverage_evidence_complete = bool(
        positive_clips
        and positive_expected_detection_types_evidence_complete
        and positive_event_uniqueness_evidence_complete
        and all(
            item.get("annotation_path") not in (None, "")
            and (
                int((item.get("positive_events") or {}).get("events_total") or 0)
                + int(
                    (item.get("positive_events") or {}).get(
                        "fully_excluded_events"
                    )
                    or 0
                )
            )
            > 0
            for item in positive_clips
        )
    )
    corpus_complete = bool(
        source_inventory_evidence_complete
        and expected_detection_type_policy_evidence_complete
        and (
            not positive_clips
            or positive_annotation_coverage_evidence_complete
        )
    )
    delays = [
        float(event["delay_sec"])
        for item in evaluated
        for event in (item.get("positive_events") or {}).get("events") or []
        if event.get("delay_sec") is not None
    ]
    channels: Counter[str] = Counter()
    for item in evaluated:
        channels.update(item.get("channel_totals") or {})
    candidate_model_evidence = _aggregate_candidate_model_evidence([
        summary
        for item in evaluated
        if isinstance(
            summary := item.get("candidate_model_evidence_summary"),
            dict,
        )
    ])
    prospective_candidate_fusion = _aggregate_prospective_candidate_fusion([
        summary
        for item in evaluated
        if isinstance(
            summary := item.get("prospective_candidate_fusion_summary"),
            dict,
        )
    ])

    guard_keys = (
        "observed_frames",
        "enabled_frames",
        "disabled_frames",
        "usable_frames",
        "blocked_frames",
    )

    def _guard_summary(item: dict, *, score_eligible: bool) -> dict | None:
        key = (
            "scene_guard_summary_score_eligible"
            if score_eligible
            else "scene_guard_summary_all"
        )
        summary = item.get(key)
        if not isinstance(summary, dict):
            summary = item.get("scene_guard")
        return summary if isinstance(summary, dict) else None

    def _valid_guard_summary(
        summary: dict | None,
        *,
        expected_frames: int,
        require_enabled: bool,
    ) -> bool:
        if not isinstance(summary, dict):
            return False
        try:
            counts = {key: int(summary[key]) for key in guard_keys}
            state_total = sum(int(value) for value in summary["state_counts"].values())
        except (KeyError, TypeError, ValueError, AttributeError):
            return False
        if (
            any(value < 0 for value in counts.values())
            or state_total < 0
            or expected_frames < 0
        ):
            return False
        observed = counts["observed_frames"]
        return bool(
            observed > 0
            and observed == expected_frames
            and observed == counts["enabled_frames"] + counts["disabled_frames"]
            and observed == counts["usable_frames"] + counts["blocked_frames"]
            and observed == state_total
            and (
                not require_enabled
                or (
                    counts["enabled_frames"] == observed
                    and counts["disabled_frames"] == 0
                )
            )
        )

    def _has_complete_scene_guard_all_evidence(item: dict) -> bool:
        if item.get("evaluation_mode") != "pipeline":
            return False
        try:
            failed = int(item.get("failed_frames") or 0)
            processed = int(item.get("processed_frames") or 0)
        except (TypeError, ValueError):
            return False
        return bool(
            failed == 0
            and _valid_guard_summary(
                _guard_summary(item, score_eligible=False),
                expected_frames=processed,
                require_enabled=False,
            )
        )

    def _has_complete_scene_guard_evidence(item: dict) -> bool:
        if not _has_complete_scene_guard_all_evidence(item):
            return False
        try:
            processed = int(item.get("processed_frames") or 0)
            scored = int(item.get("scored_frames") or 0)
            excluded_raw = item.get("excluded_frames")
            if excluded_raw is None:
                # Legacy pipeline reports omitted excluded_frames and used
                # ignored_frames as the only non-scored category.
                excluded = int(item.get("ignored_frames") or 0)
            else:
                excluded = int(excluded_raw)
        except (TypeError, ValueError):
            return False
        return bool(
            min(processed, scored, excluded) >= 0
            and processed == scored + excluded
            and _valid_guard_summary(
                _guard_summary(item, score_eligible=True),
                expected_frames=scored,
                require_enabled=True,
            )
        )

    scene_guard_score_eligible_summaries = [
        summary
        for item in evaluated
        if (summary := _guard_summary(item, score_eligible=True)) is not None
    ]
    scene_guard_all_summaries = [
        summary
        for item in evaluated
        if (summary := _guard_summary(item, score_eligible=False)) is not None
    ]
    has_scene_guard_evidence = bool(
        scene_guard_score_eligible_summaries or scene_guard_all_summaries
    )
    scene_guard_all_evidence_complete = bool(
        scene_guard_all_summaries
        and len(scene_guard_all_summaries) == len(evaluated)
        and all(_has_complete_scene_guard_all_evidence(item) for item in evaluated)
    )
    scene_guard_score_eligible_evidence_complete = bool(
        has_scene_guard_evidence
        and len(scene_guard_score_eligible_summaries) == len(evaluated)
        and all(_has_complete_scene_guard_evidence(item) for item in evaluated)
    )
    scene_guard_evidence_complete = bool(
        scene_guard_score_eligible_evidence_complete
        and scene_guard_all_evidence_complete
    )
    scene_guard_usable_frames = sum(
        int(item.get("usable_frames") or 0)
        for item in scene_guard_score_eligible_summaries
    )
    scene_guard_blocked_frames = sum(
        int(item.get("blocked_frames") or 0)
        for item in scene_guard_score_eligible_summaries
    )
    scene_guard_total_frames = scene_guard_usable_frames + scene_guard_blocked_frames
    raw_scene_guard_coverage = (
        scene_guard_usable_frames / scene_guard_total_frames
        if scene_guard_total_frames > 0
        else None
    )
    scene_guard_coverage = (
        raw_scene_guard_coverage
        if scene_guard_score_eligible_evidence_complete
        else None
    )
    scene_guard_all_usable_frames = sum(
        int(item.get("usable_frames") or 0) for item in scene_guard_all_summaries
    )
    scene_guard_all_blocked_frames = sum(
        int(item.get("blocked_frames") or 0) for item in scene_guard_all_summaries
    )
    scene_guard_all_total_frames = (
        scene_guard_all_usable_frames + scene_guard_all_blocked_frames
    )
    raw_scene_guard_all_coverage = (
        scene_guard_all_usable_frames / scene_guard_all_total_frames
        if scene_guard_all_total_frames > 0
        else None
    )
    scene_guard_all_coverage = (
        raw_scene_guard_all_coverage if scene_guard_all_evidence_complete else None
    )

    scene_references = [
        reference
        for item in evaluated
        if isinstance(reference := item.get("scene_reference"), dict)
    ]

    def _operator_approved(reference: dict) -> bool:
        # Explicit evidence is mandatory.  Legacy reports that predate the
        # operator-authorization contract must be regenerated, not grandfathered.
        authorization = reference.get("operator_authorization")
        if not isinstance(authorization, dict):
            return False
        configured_profile_id = authorization.get("configured_profile_id")
        return bool(
            reference.get("operator_approved") is True
            and authorization.get("operator_approved") is True
            and authorization.get("metadata_field_present") is True
            and authorization.get("reason") is None
            and configured_profile_id not in (None, "")
            and authorization.get("active_profile_id") == configured_profile_id
        )

    def _reference_approved(reference: dict) -> bool:
        return (
            reference.get("reference_approved") is True
            and _operator_approved(reference)
        )

    operator_approved_reference_clip_count = sum(
        1 for reference in scene_references if _operator_approved(reference)
    )
    operator_approval_complete = bool(
        evaluated
        and len(scene_references) == len(evaluated)
        and operator_approved_reference_clip_count == len(evaluated)
    )
    operator_approval_reasons: Counter[str] = Counter()
    missing_reference_evidence = max(0, len(evaluated) - len(scene_references))
    if missing_reference_evidence:
        operator_approval_reasons[
            "scene_profile_operator_approval_evidence_missing"
        ] += missing_reference_evidence
    for reference in scene_references:
        if _operator_approved(reference):
            continue
        authorization = reference.get("operator_authorization")
        reason = (
            authorization.get("reason")
            if isinstance(authorization, dict)
            else None
        )
        operator_approval_reasons[
            str(reason or "scene_profile_operator_approval_evidence_missing")
        ] += 1

    trusted_reference_clip_count = sum(
        1 for reference in scene_references if _reference_approved(reference)
    )
    trusted_scene_reference_complete = bool(
        evaluated
        and len(scene_references) == len(evaluated)
        and trusted_reference_clip_count == len(evaluated)
    )

    model_identities = [
        identity
        for item in evaluated
        if isinstance(identity := item.get("model_identity"), dict)
    ]

    def _model_identity_key(identity: dict) -> tuple[str, str] | None:
        actual = identity.get("actual")
        if not isinstance(actual, dict):
            return None
        version_id = actual.get("version_id")
        artifact_sha256 = actual.get("artifact_sha256")
        if version_id in (None, "") or artifact_sha256 in (None, ""):
            return None
        return str(version_id), str(artifact_sha256)

    model_identity_keys = [
        key
        for identity in model_identities
        if (key := _model_identity_key(identity)) is not None
    ]
    approved_model_identity_clip_count = sum(
        1 for identity in model_identities if identity.get("release_approved") is True
    )
    runtime_model_identity_complete = bool(
        evaluated
        and len(model_identities) == len(evaluated)
        and approved_model_identity_clip_count == len(evaluated)
        and len(model_identity_keys) == len(evaluated)
        and len(set(model_identity_keys)) == 1
    )

    parity_records = [
        parity
        for item in evaluated
        if isinstance(parity := item.get("production_parity"), dict)
    ]
    production_parity_supported_clip_count = sum(
        1 for parity in parity_records if parity.get("supported") is True
    )
    production_parity_complete = bool(
        evaluated
        and len(parity_records) == len(evaluated)
        and production_parity_supported_clip_count == len(evaluated)
    )
    independent_holdout_clip_count = sum(
        1
        for item in evaluated
        if "independent_test" in {str(tag) for tag in item.get("tags") or []}
    )
    independent_holdout_complete = bool(
        evaluated
        and independent_holdout_clip_count == len(evaluated)
        and source_session_identity_evidence_complete
    )

    candidate_contracts = [
        contract
        for item in evaluated
        if isinstance(
            contract := item.get("candidate_confirmation_contract"),
            dict,
        )
    ]
    candidate_contract_evidence_complete = bool(
        evaluated
        and len(candidate_contracts) == len(evaluated)
        and all(contract.get("available") is True for contract in candidate_contracts)
    )

    def _candidate_disabled(contract: dict) -> bool:
        return (
            contract.get("available") is True
            and contract.get("enabled") is False
        )

    def _candidate_enforce_model_required(contract: dict) -> bool:
        model_evidence = contract.get("model_evidence")
        return bool(
            contract.get("available") is True
            and contract.get("enabled") is True
            and contract.get("mode") == "enforce"
            and isinstance(model_evidence, dict)
            and model_evidence.get("enabled") is True
            and model_evidence.get("fusion_policy") == "model_required"
        )

    candidate_explicitly_not_configured = bool(
        candidate_contract_evidence_complete
        and all(_candidate_disabled(contract) for contract in candidate_contracts)
    )
    candidate_enforce_contract_complete = bool(
        candidate_contract_evidence_complete
        and all(
            _candidate_enforce_model_required(contract)
            for contract in candidate_contracts
        )
    )
    contract_calibration_ids = [
        calibration_id
        for contract in candidate_contracts
        if isinstance(model_evidence := contract.get("model_evidence"), dict)
        if (calibration_id := model_evidence.get("calibration_id"))
        not in (None, "")
    ]
    candidate_contract_calibration_id = (
        str(contract_calibration_ids[0])
        if len(contract_calibration_ids) == len(candidate_contracts)
        and len({str(item) for item in contract_calibration_ids}) == 1
        else None
    )

    def _finite_ratio_at_least(summary: dict, key: str, minimum: float) -> bool:
        try:
            value = float(summary[key])
        except (KeyError, TypeError, ValueError):
            return False
        return math.isfinite(value) and value >= minimum

    candidate_observed = int(
        prospective_candidate_fusion.get("observed_candidates") or 0
    )
    model_evidence_observed = int(
        candidate_model_evidence.get("observed_candidates") or 0
    )
    mode_counts = prospective_candidate_fusion.get("mode_counts") or {}
    fusion_policy_counts = (
        prospective_candidate_fusion.get("fusion_policy_counts") or {}
    )
    candidate_runtime_contract_matches = bool(
        candidate_observed > 0
        and mode_counts == {"enforce": candidate_observed}
        and fusion_policy_counts == {"model_required": candidate_observed}
    )
    calibration_id_counts = (
        candidate_model_evidence.get("calibration_id_counts") or {}
    )
    candidate_calibration_contract_matches = bool(
        candidate_contract_calibration_id is not None
        and model_evidence_observed == candidate_observed
        and calibration_id_counts
        == {candidate_contract_calibration_id: candidate_observed}
    )
    candidate_telemetry_complete = bool(
        candidate_model_evidence.get("clip_count_with_evidence") == len(evaluated)
        and prospective_candidate_fusion.get("clip_count_with_evidence")
        == len(evaluated)
        and int(
            candidate_model_evidence.get("map_extraction_eligible_candidates")
            or 0
        )
        > 0
        and int(
            candidate_model_evidence.get(
                "calibrated_evaluation_eligible_candidates"
            )
            or 0
        )
        > 0
        and _finite_ratio_at_least(
            candidate_model_evidence,
            "map_extraction_coverage_ratio",
            DEFAULT_MIN_CANDIDATE_COVERAGE,
        )
        and _finite_ratio_at_least(
            candidate_model_evidence,
            "calibrated_evaluation_coverage_ratio",
            DEFAULT_MIN_CANDIDATE_COVERAGE,
        )
        and int(candidate_model_evidence.get("nullable_contract_violations") or 0)
        == 0
        and int(
            candidate_model_evidence.get("extraction_contract_violations") or 0
        )
        == 0
        and int(prospective_candidate_fusion.get("unavailable_candidates") or 0)
        == 0
        and candidate_runtime_contract_matches
        and candidate_calibration_contract_matches
    )
    candidate_confirmation_required = bool(
        static_foreign_total > 0
        or any(item.get("kind") == "static_foreign" for item in evaluated)
    )
    if candidate_explicitly_not_configured and not candidate_confirmation_required:
        candidate_confirmation_acceptance = "not_applicable"
    elif candidate_enforce_contract_complete and candidate_telemetry_complete:
        candidate_confirmation_acceptance = "pass"
    elif evaluated:
        candidate_confirmation_acceptance = "fail"
    else:
        candidate_confirmation_acceptance = "insufficient_data"

    fp_per_hour = fp_events / normal_hours if normal_hours > 0 else None
    recall = positive_hit / positive_total if positive_total > 0 else None
    acceptance = {
        "corpus_completeness": "pass" if corpus_complete else "fail",
        "corpus_sufficiency": (
            "pass"
            if source_inventory_evidence_complete
            and normal_hours >= DEFAULT_MIN_NORMAL_HOURS
            and static_foreign_total >= DEFAULT_MIN_POSITIVE_EVENTS
            and projectile_total >= DEFAULT_MIN_PROJECTILE_EVENTS
            else "fail"
        ),
        "kind_specific_sampling": (
            "pass"
            if kind_specific_sampling_evidence_complete
            else "fail"
            if evaluated
            else "insufficient_data"
        ),
        "independent_holdout": (
            "pass"
            if independent_holdout_complete
            else "fail"
            if evaluated
            else "insufficient_data"
        ),
        "positive_annotation_coverage": (
            "pass"
            if positive_annotation_coverage_evidence_complete
            and positive_annotation_coverage is not None
            and positive_annotation_coverage
            >= DEFAULT_MIN_POSITIVE_ANNOTATION_COVERAGE
            else "fail"
            if positive_clips
            else "insufficient_data"
        ),
        "false_positive_rate": (
            "pass" if fp_per_hour is not None and fp_per_hour <= 2.0
            else "fail" if fp_per_hour is not None
            else "insufficient_data"
        ),
        "event_recall": (
            "pass" if recall is not None and recall >= 0.95
            else "fail" if recall is not None
            else "insufficient_data"
        ),
        "static_foreign_recall": (
            "pass"
            if static_foreign_recall is not None and static_foreign_recall >= 0.95
            else "fail"
            if static_foreign_recall is not None
            else "insufficient_data"
        ),
        "delay_p95": (
            "pass" if delays and (_percentile(delays, 0.95) or 0.0) <= 10.0
            else "fail" if delays
            else "insufficient_data"
        ),
        # A detector that suppresses every frame after a camera move can report
        # an excellent FP/hour while providing no surveillance.  Availability
        # is therefore an independent release gate for production-pipeline
        # replays.  Raw detector-only reports have no SceneGuard observations
        # and remain explicitly insufficient instead of being treated as pass.
        "scene_guard_coverage": (
            "pass"
            if scene_guard_evidence_complete
            and scene_guard_coverage is not None
            and scene_guard_coverage >= min_scene_guard_coverage
            and scene_guard_all_coverage is not None
            and scene_guard_all_coverage >= min_scene_guard_coverage
            else "fail"
            if has_scene_guard_evidence
            else "insufficient_data"
        ),
        "trusted_scene_reference": (
            "pass"
            if trusted_scene_reference_complete
            else "fail"
            if evaluated
            else "insufficient_data"
        ),
        "scene_profile_operator_approval": (
            "pass"
            if operator_approval_complete
            else "fail"
            if evaluated
            else "insufficient_data"
        ),
        "runtime_model_identity": (
            "pass"
            if runtime_model_identity_complete
            else "fail"
            if evaluated
            else "insufficient_data"
        ),
        "production_parity": (
            "pass"
            if production_parity_complete
            else "fail"
            if evaluated
            else "insufficient_data"
        ),
        "candidate_confirmation": candidate_confirmation_acceptance,
    }
    return {
        "clip_count": len(clip_summaries),
        "expected_clip_count": expected_clip_count,
        "evaluated_clip_count": len(evaluated),
        "corpus_complete": corpus_complete,
        "source_inventory_evidence_complete": (
            source_inventory_evidence_complete
        ),
        "min_effective_eval_fps": min_effective_eval_fps,
        "min_projectile_eval_fps": min_projectile_eval_fps,
        "kind_specific_sampling_evidence_complete": (
            kind_specific_sampling_evidence_complete
        ),
        "effective_eval_fps_min_observed": min(
            (
                float(item["effective_eval_fps"])
                for item in evaluated
                if type(item.get("effective_eval_fps")) in (int, float)
                and math.isfinite(float(item["effective_eval_fps"]))
            ),
            default=None,
        ),
        "single_camera_evidence_complete": single_camera_evidence_complete,
        "unique_source_artifacts_evidence_complete": (
            unique_source_artifacts_evidence_complete
        ),
        "source_session_identity_evidence_complete": (
            source_session_identity_evidence_complete
        ),
        "duplicate_source_session_id_count": (
            len(valid_source_session_ids) - len(set(valid_source_session_ids))
        ),
        "source_session_inventory": sorted(valid_source_session_ids),
        "duplicate_source_video_hash_count": (
            len(valid_source_video_hashes)
            - len(set(valid_source_video_hashes))
        ),
        "duplicate_source_video_path_count": (
            len(source_video_paths) - len(set(source_video_paths))
        ),
        "evaluated_artifacts": evaluated_artifacts,
        "evaluated_source_inventory": [],
        "evaluated_scene_profile": None,
        "evaluated_artifact_binding_evidence_complete": False,
        "source_inventory": {
            "source_video_sha256": sorted(
                value
                for value in valid_source_video_hashes
            ),
        },
        "normal_hours": normal_hours,
        "min_normal_hours": DEFAULT_MIN_NORMAL_HOURS,
        "normal_corpus_sufficient": normal_hours >= DEFAULT_MIN_NORMAL_HOURS,
        "false_positive_event_count": fp_events,
        "false_positive_events_per_hour": fp_per_hour,
        "positive_event_count": positive_total,
        "min_positive_event_count": DEFAULT_MIN_POSITIVE_EVENTS,
        "positive_events_by_kind": dict(sorted(positive_events_by_kind.items())),
        "positive_event_hits_by_kind": dict(
            sorted(positive_hits_by_kind.items())
        ),
        "positive_event_recall_by_kind": dict(
            sorted(positive_recall_by_kind.items())
        ),
        "static_foreign_event_count": static_foreign_total,
        "static_foreign_event_hit_count": static_foreign_hit,
        "static_foreign_event_recall": static_foreign_recall,
        "projectile_event_count": projectile_total,
        "projectile_event_hit_count": projectile_hit,
        "min_projectile_event_count": DEFAULT_MIN_PROJECTILE_EVENTS,
        "positive_corpus_sufficient": (
            static_foreign_total >= DEFAULT_MIN_POSITIVE_EVENTS
            and projectile_total >= DEFAULT_MIN_PROJECTILE_EVENTS
        ),
        "positive_event_hit_count": positive_hit,
        "fully_excluded_positive_event_count": fully_excluded_positive_total,
        "annotated_positive_event_count": annotated_positive_total,
        "positive_annotation_coverage_ratio": positive_annotation_coverage,
        "positive_annotation_coverage_evidence_complete": (
            positive_annotation_coverage_evidence_complete
        ),
        "positive_expected_detection_types_evidence_complete": (
            positive_expected_detection_types_evidence_complete
        ),
        "expected_detection_type_policy_evidence_complete": (
            expected_detection_type_policy_evidence_complete
        ),
        "positive_event_uniqueness_evidence_complete": (
            positive_event_uniqueness_evidence_complete
        ),
        "min_positive_annotation_coverage": (
            DEFAULT_MIN_POSITIVE_ANNOTATION_COVERAGE
        ),
        "event_recall": recall,
        "delay_p95_sec": _percentile(delays, 0.95),
        "scene_guard_usable_frames": scene_guard_usable_frames,
        "scene_guard_blocked_frames": scene_guard_blocked_frames,
        "scene_guard_usable_coverage_ratio": scene_guard_coverage,
        "scene_guard_raw_coverage_ratio": raw_scene_guard_coverage,
        "scene_guard_score_eligible_usable_frames": scene_guard_usable_frames,
        "scene_guard_score_eligible_blocked_frames": scene_guard_blocked_frames,
        "scene_guard_score_eligible_usable_coverage_ratio": scene_guard_coverage,
        "scene_guard_score_eligible_raw_coverage_ratio": raw_scene_guard_coverage,
        "scene_guard_all_usable_frames": scene_guard_all_usable_frames,
        "scene_guard_all_blocked_frames": scene_guard_all_blocked_frames,
        "scene_guard_all_usable_coverage_ratio": scene_guard_all_coverage,
        "scene_guard_all_raw_coverage_ratio": raw_scene_guard_all_coverage,
        "scene_guard_all_evidence_complete": scene_guard_all_evidence_complete,
        "scene_guard_all_frame_usable_coverage_ratio": (
            scene_guard_all_coverage
        ),
        "scene_guard_all_frame_raw_coverage_ratio": (
            raw_scene_guard_all_coverage
        ),
        "scene_guard_all_frame_evidence_complete": (
            scene_guard_all_evidence_complete
        ),
        "scene_guard_score_eligible_evidence_complete": (
            scene_guard_score_eligible_evidence_complete
        ),
        "scene_guard_evidence_complete": scene_guard_evidence_complete,
        "min_scene_guard_coverage": min_scene_guard_coverage,
        "trusted_scene_reference_clip_count": trusted_reference_clip_count,
        "trusted_scene_reference_evidence_complete": (
            trusted_scene_reference_complete
        ),
        "operator_approved_scene_profile_clip_count": (
            operator_approved_reference_clip_count
        ),
        "scene_profile_operator_approval_evidence_complete": (
            operator_approval_complete
        ),
        "scene_profile_operator_approval_reason_counts": dict(
            sorted(operator_approval_reasons.items())
        ),
        "approved_model_identity_clip_count": approved_model_identity_clip_count,
        "runtime_model_identity_evidence_complete": (
            runtime_model_identity_complete
        ),
        "runtime_model_identity_keys": [
            {"version_id": version_id, "artifact_sha256": artifact_sha256}
            for version_id, artifact_sha256 in sorted(set(model_identity_keys))
        ],
        "production_parity_supported_clip_count": (
            production_parity_supported_clip_count
        ),
        "production_parity_evidence_complete": production_parity_complete,
        "independent_holdout_clip_count": independent_holdout_clip_count,
        "independent_holdout_evidence_complete": independent_holdout_complete,
        "candidate_contract_evidence_complete": (
            candidate_contract_evidence_complete
        ),
        "candidate_explicitly_not_configured": (
            candidate_explicitly_not_configured
        ),
        "candidate_confirmation_required": candidate_confirmation_required,
        "candidate_enforce_contract_complete": (
            candidate_enforce_contract_complete
        ),
        "candidate_runtime_contract_matches": candidate_runtime_contract_matches,
        "candidate_contract_calibration_id": candidate_contract_calibration_id,
        "candidate_calibration_contract_matches": (
            candidate_calibration_contract_matches
        ),
        "candidate_telemetry_complete": candidate_telemetry_complete,
        "min_candidate_coverage": DEFAULT_MIN_CANDIDATE_COVERAGE,
        "candidate_model_evidence_summary": candidate_model_evidence,
        "prospective_candidate_fusion_summary": prospective_candidate_fusion,
        "channel_totals": dict(sorted(channels.items())),
        "acceptance": acceptance,
    }


def build_markdown_report(payload: dict) -> str:
    """Render a compact markdown report for the suite."""
    suite = payload["suite"]
    clips = payload["clips"]
    release_policy = suite.get("release_acceptance_policy")
    policy_lines: list[str] = []
    if isinstance(release_policy, dict):
        canonical = release_policy.get("canonical") or {}
        actual = release_policy.get("actual") or {}
        policy_lines = [
            "## Exact Release Acceptance Policy",
            "",
            "| Parameter | Canonical | Actual | Exact |",
            "|---|---:|---:|---|",
            *[
                (
                    f"| `{name}` | {_fmt(expected)} | {_fmt(actual.get(name))} | "
                    f"{'yes' if actual.get(name) == expected else 'no'} |"
                )
                for name, expected in canonical.items()
            ],
            "",
            (
                "Policy result: `pass`"
                if release_policy.get("exact_match") is True
                else "Policy result: `fail` (diagnostic-only report)"
            ),
            "",
        ]
    lines = [
        "# Argus Regression Suite Report",
        "",
        f"- **Generated at**: {payload['generated_at']}",
        f"- **Suite ID**: `{payload.get('suite_id') or '-'}`",
        f"- **Commit**: `{payload.get('git_commit') or '-'}`",
        f"- **Status**: `{payload['status']}`",
        (
            f"- **Clips**: {suite['evaluated_clip_count']} evaluated / "
            f"{suite['expected_clip_count']} expected "
            f"({suite['clip_count']} reported)"
        ),
        "",
        *policy_lines,
        "## Acceptance Metrics",
        "",
        "| Metric | Target | Actual | Status |",
        "|---|---:|---:|---|",
        (
            "| Corpus completeness | all inventory clips | "
            f"{suite['evaluated_clip_count']} / {suite['expected_clip_count']} | "
            f"{suite['acceptance']['corpus_completeness']} |"
        ),
        (
            "| Independent corpus sufficiency | >= "
            f"{_fmt(suite['min_normal_hours'])} normal h and >= "
            f"{suite['min_positive_event_count']} static_foreign + >= "
            f"{suite['min_projectile_event_count']} projectile events | "
            f"{_fmt(suite['normal_hours'])} h / "
            f"{suite['static_foreign_event_count']} static_foreign / "
            f"{suite['projectile_event_count']} projectile events | "
            f"{suite['acceptance']['corpus_sufficiency']} |"
        ),
        (
            "| Kind-specific sampling | normal/static >= "
            f"{_fmt(suite['min_effective_eval_fps'])} FPS; projectile >= "
            f"{_fmt(suite['min_projectile_eval_fps'])} FPS | "
            f"min observed {_fmt(suite['effective_eval_fps_min_observed'])} | "
            f"{suite['acceptance']['kind_specific_sampling']} |"
        ),
        (
            "| Independent holdout | tag + unique source session per clip | "
            f"{suite['independent_holdout_clip_count']} / "
            f"{suite['evaluated_clip_count']} | "
            f"{suite['acceptance']['independent_holdout']} |"
        ),
        (
            "| Positive annotation coverage | >= "
            f"{_fmt(suite['min_positive_annotation_coverage'])} | "
            f"{_fmt(suite['positive_annotation_coverage_ratio'])} "
            f"({suite['positive_event_count']} / "
            f"{suite['annotated_positive_event_count']}) | "
            f"{suite['acceptance']['positive_annotation_coverage']} |"
        ),
        (
            "| False positives/hour | <= 2.00 | "
            f"{_fmt(suite['false_positive_events_per_hour'])} | "
            f"{suite['acceptance']['false_positive_rate']} |"
        ),
        (
            "| Overall event recall (diagnostic) | >= 0.95 | "
            f"{_fmt(suite['event_recall'])} | {suite['acceptance']['event_recall']} |"
        ),
        (
            "| Static foreign-object recall | >= 0.95 | "
            f"{_fmt(suite['static_foreign_event_recall'])} | "
            f"{suite['acceptance']['static_foreign_recall']} |"
        ),
        (
            "| P95 detection delay (s) | <= 10.00 | "
            f"{_fmt(suite['delay_p95_sec'])} | {suite['acceptance']['delay_p95']} |"
        ),
        (
            "| SceneGuard score-eligible coverage | "
            f">= {_fmt(suite['min_scene_guard_coverage'])} | "
            f"{_fmt(suite['scene_guard_usable_coverage_ratio'])} | "
            f"{suite['acceptance']['scene_guard_coverage']} |"
        ),
        (
            "| Trusted scene reference | all evaluated clips | "
            f"{suite['trusted_scene_reference_clip_count']} / "
            f"{suite['evaluated_clip_count']} | "
            f"{suite['acceptance']['trusted_scene_reference']} |"
        ),
        (
            "| Scene-profile operator approval | explicit approval on every clip | "
            f"{suite['operator_approved_scene_profile_clip_count']} / "
            f"{suite['evaluated_clip_count']} | "
            f"{suite['acceptance']['scene_profile_operator_approval']} |"
        ),
        (
            "| Runtime model identity | one approved artifact | "
            f"{suite['approved_model_identity_clip_count']} / "
            f"{suite['evaluated_clip_count']} approved | "
            f"{suite['acceptance']['runtime_model_identity']} |"
        ),
        (
            "| Production parity | all evaluated clips | "
            f"{suite['production_parity_supported_clip_count']} / "
            f"{suite['evaluated_clip_count']} | "
            f"{suite['acceptance']['production_parity']} |"
        ),
        (
            "| Candidate model fusion | enforce + model_required, >= "
            f"{_fmt(suite['min_candidate_coverage'])} coverage | "
            f"map={_fmt((suite.get('candidate_model_evidence_summary') or {}).get('map_extraction_coverage_ratio'))}, "
            f"calibrated={_fmt((suite.get('candidate_model_evidence_summary') or {}).get('calibrated_evaluation_coverage_ratio'))} | "
            f"{suite['acceptance']['candidate_confirmation']} |"
        ),
        "",
        (
            "- Scene-profile operator-approval blockers: "
            f"{suite.get('scene_profile_operator_approval_reason_counts') or {}}"
        ),
        "",
        "## Channel Attribution",
        "",
        "| Channel | FP frame evidence count |",
        "|---|---:|",
    ]
    if suite["channel_totals"]:
        for channel, count in suite["channel_totals"].items():
            lines.append(f"| {channel} | {count} |")
    else:
        lines.append("| - | 0 |")

    model_evidence = suite.get("candidate_model_evidence_summary") or {}
    calibration_ids = ", ".join(
        f"{name}={count}"
        for name, count in (model_evidence.get("calibration_id_counts") or {}).items()
    ) or "-"
    output_spaces = ", ".join(
        f"{name}={count}"
        for name, count in (model_evidence.get("output_space_counts") or {}).items()
    ) or "-"
    fusion = suite.get("prospective_candidate_fusion_summary") or {}
    attribution = fusion.get("anomaly_attribution") or {}
    lines += [
        "",
        "## SceneGuard Coverage Audit",
        "",
        "| Denominator | Usable | Blocked | Coverage | Evidence complete |",
        "|---|---:|---:|---:|---|",
        (
            "| Score eligible | "
            f"{suite.get('scene_guard_usable_frames', 0)} | "
            f"{suite.get('scene_guard_blocked_frames', 0)} | "
            f"{_fmt(suite.get('scene_guard_usable_coverage_ratio'))} | "
            f"{suite.get('scene_guard_evidence_complete')} |"
        ),
        (
            "| All processed | "
            f"{suite.get('scene_guard_all_usable_frames', 0)} | "
            f"{suite.get('scene_guard_all_blocked_frames', 0)} | "
            f"{_fmt(suite.get('scene_guard_all_usable_coverage_ratio'))} | "
            f"{suite.get('scene_guard_all_evidence_complete')} |"
        ),
        "",
        "## Candidate Model Evidence",
        "",
        "| Metric | Numerator | Denominator | Ratio |",
        "|---|---:|---:|---:|",
        (
            "| Model-map extraction | "
            f"{model_evidence.get('map_extracted_candidates', 0)} | "
            f"{model_evidence.get('map_extraction_eligible_candidates', 0)} | "
            f"{_fmt(model_evidence.get('map_extraction_coverage_ratio'))} |"
        ),
        (
            "| Calibrated evaluation | "
            f"{model_evidence.get('calibrated_evaluated_candidates', 0)} | "
            f"{model_evidence.get('calibrated_evaluation_eligible_candidates', 0)} | "
            f"{_fmt(model_evidence.get('calibrated_evaluation_coverage_ratio'))} |"
        ),
        (
            "| Rule acceptance | "
            f"{model_evidence.get('accepted_candidates', 0)} | "
            f"{model_evidence.get('evaluated_candidates', 0)} | "
            f"{_fmt(model_evidence.get('rule_accept_rate'))} |"
        ),
        "",
        f"- Calibration bindings: {calibration_ids}",
        f"- Output spaces: {output_spaces}",
        (
            "- Contract violations: nullable="
            f"{model_evidence.get('nullable_contract_violations', 0)}, extraction="
            f"{model_evidence.get('extraction_contract_violations', 0)}"
        ),
        "",
        "## Prospective Candidate Fusion",
        "",
        "| Observed | Evaluated | Accepted | Pending | Rejected | Unavailable | Confirmed | Accept rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {fusion.get('observed_candidates', 0)} | "
            f"{fusion.get('evaluated_candidates', 0)} | "
            f"{fusion.get('accepted_candidates', 0)} | "
            f"{fusion.get('pending_candidates', 0)} | "
            f"{fusion.get('rejected_candidates', 0)} | "
            f"{fusion.get('unavailable_candidates', 0)} | "
            f"{fusion.get('confirmed_candidates', 0)} | "
            f"{_fmt(fusion.get('accept_rate'))} |"
        ),
        "",
        f"- Runtime mode counts: {fusion.get('mode_counts') or {}}",
        f"- Fusion policy counts: {fusion.get('fusion_policy_counts') or {}}",
        "",
        "### Prospective Anomaly Attribution",
        "",
        "| Observed | Decided | Would suppress | Would preserve | Unavailable | Effective suppressions | Suppress rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {attribution.get('observed_frames', 0)} | "
            f"{attribution.get('decided_frames', 0)} | "
            f"{attribution.get('would_suppress_frames', 0)} | "
            f"{attribution.get('would_preserve_frames', 0)} | "
            f"{attribution.get('unavailable_frames', 0)} | "
            f"{attribution.get('effective_suppressed_frames', 0)} | "
            f"{_fmt(attribution.get('suppress_rate'))} |"
        ),
        "",
        "## Clips",
        "",
        "| Clip | Kind | Camera | FP events | FP/hour | Recall | P95 delay | Guard coverage | Operator approved | Release blockers | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for item in clips:
        pos = item.get("positive_events") or {}
        scene_reference = item.get("scene_reference") or {}
        release_contract = item.get("release_contract") or {}
        release_blockers = ", ".join(
            str(reason) for reason in release_contract.get("reasons") or []
        ) or "-"
        lines.append(
            f"| {item.get('clip_id')} | {item.get('kind')} | {item.get('camera_id')} | "
            f"{item.get('false_positive_event_count', '-')} | "
            f"{_fmt(item.get('false_positive_events_per_hour'))} | "
            f"{_fmt(pos.get('recall'))} | {_fmt(pos.get('delay_p95_sec'))} | "
            f"{_fmt((item.get('scene_guard') or {}).get('usable_coverage_ratio'))} | "
            f"{scene_reference.get('operator_approved', False)} | "
            f"{release_blockers} | "
            f"{item.get('status')} |"
        )
    event_rows = [
        (item.get("clip_id"), event)
        for item in clips
        for event in (item.get("positive_events") or {}).get("events") or []
    ]
    if event_rows:
        lines += [
            "",
            "## Positive Event Diagnostics",
            "",
            "| Clip | Event | Hit | Max score | Diagnosis | Top candidates | Top suppressions |",
            "|---|---|---|---:|---|---|---|",
        ]
        for clip_id, event in event_rows:
            candidate_text = ", ".join(
                f"{name}={count}"
                for name, count in list((event.get("candidate_type_counts") or {}).items())[:3]
            ) or "-"
            suppression_text = ", ".join(
                f"{name}={count}"
                for name, count in list((event.get("suppression_reason_counts") or {}).items())[:3]
            ) or "-"
            lines.append(
                f"| {clip_id} | {event.get('name') or '-'} | {event.get('hit')} | "
                f"{_fmt(event.get('max_score'))} | {event.get('diagnosis')} | "
                f"{candidate_text} | {suppression_text} |"
            )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _git_commit() -> str | None:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _generated_negative_annotation(clip: RegressionClip, output_dir: Path) -> Path:
    path = output_dir / "generated_annotations" / f"{clip.clip_id}.labels.json"
    _write_json(path, {"positive_ranges": [], "ignore_ranges": []})
    return path


def _validate_cached_evaluation(
    result: dict,
    clip: RegressionClip,
    *,
    config_path: Path,
) -> dict:
    """Rebind a cached report to the exact source artifacts on disk."""
    reasons: list[str] = []
    if result.get("schema_version") != PIPELINE_VIDEO_EVAL_SCHEMA_VERSION:
        reasons.append("schema_version_mismatch")
    if result.get("evaluation_mode") != "pipeline":
        reasons.append("evaluation_mode_not_pipeline")
    if result.get("source_artifacts_stable") is not True:
        reasons.append("source_artifacts_stability_evidence_missing")
    annotation_inventory = result.get("annotation_inventory")
    if (
        not isinstance(annotation_inventory, dict)
        or annotation_inventory.get("all_ranges_within_source") is not True
    ):
        reasons.append("annotation_inventory_invalid")

    recorded_video_raw = result.get("video_path")
    recorded_annotation_raw = result.get("annotation_path")
    recorded_video = (
        Path(recorded_video_raw)
        if type(recorded_video_raw) is str and recorded_video_raw
        else None
    )
    recorded_annotation = (
        Path(recorded_annotation_raw)
        if type(recorded_annotation_raw) is str and recorded_annotation_raw
        else None
    )
    expected_video = clip.video_path or recorded_video
    expected_annotation = clip.annotation_path or recorded_annotation

    def verify_artifact(
        *,
        name: str,
        recorded_path: Path | None,
        expected_path: Path | None,
        recorded_sha256: Any,
    ) -> str | None:
        if recorded_path is None or expected_path is None:
            reasons.append(f"{name}_path_missing")
            return None
        if not recorded_path.is_file() or not expected_path.is_file():
            reasons.append(f"{name}_not_found")
            return None
        if not _same_path(recorded_path, expected_path):
            reasons.append(f"{name}_path_mismatch")
            return None
        if not _is_sha256(recorded_sha256):
            reasons.append(f"{name}_sha256_missing_or_invalid")
            return None
        actual_sha256 = _stream_sha256(expected_path)
        if actual_sha256 != recorded_sha256:
            reasons.append(f"{name}_sha256_mismatch")
        return actual_sha256

    actual_video_sha256 = verify_artifact(
        name="source_video",
        recorded_path=recorded_video,
        expected_path=expected_video,
        recorded_sha256=result.get("source_video_sha256"),
    )
    actual_annotation_sha256 = verify_artifact(
        name="annotation",
        recorded_path=recorded_annotation,
        expected_path=expected_annotation,
        recorded_sha256=result.get("annotation_sha256"),
    )
    recorded_config_raw = result.get("config_path")
    recorded_config = (
        Path(recorded_config_raw)
        if type(recorded_config_raw) is str and recorded_config_raw
        else None
    )
    actual_config_sha256 = None
    if recorded_config is None or not config_path.is_file():
        reasons.append("config_path_missing_or_invalid")
    elif not _same_path(recorded_config, config_path):
        reasons.append("config_path_mismatch")
    else:
        actual_config_sha256 = _evidence_artifact_sha256_if_present(config_path)
        if not _is_sha256(result.get("config_artifact_sha256")):
            reasons.append("config_artifact_sha256_missing_or_invalid")
        elif actual_config_sha256 != result.get("config_artifact_sha256"):
            reasons.append("config_artifact_sha256_mismatch")
    recorded_camera_id = result.get("camera_id")
    if type(recorded_camera_id) is not str or not recorded_camera_id.strip():
        reasons.append("camera_id_missing_or_invalid")
    elif clip.camera_id != "unknown" and recorded_camera_id != clip.camera_id:
        reasons.append("camera_id_mismatch")

    return {
        "valid": not reasons,
        "schema_version_valid": (
            result.get("schema_version") == PIPELINE_VIDEO_EVAL_SCHEMA_VERSION
        ),
        "source_video_sha256_recomputed": actual_video_sha256,
        "annotation_sha256_recomputed": actual_annotation_sha256,
        "config_artifact_sha256_recomputed": actual_config_sha256,
        "reasons": reasons,
    }


def _evaluate_clip(
    clip: RegressionClip,
    *,
    output_dir: Path,
    model_path: Path | None,
    threshold: float,
    image_size: int,
    sample_every: int,
    max_frames: int | None,
    sweep: bool,
    evaluation_mode: str = "detector",
    config_path: Path = Path("configs/default.yaml"),
    warmup_seconds: float = 30.0,
    scene_reference_path: Path | None = None,
    scene_reference_frame_index: int | None = None,
    model_version_id: str | None = None,
) -> dict:
    if clip.evaluation_json_path is not None:
        result = _read_json(clip.evaluation_json_path)
        if not isinstance(result, dict):
            raise ValueError(
                f"clip {clip.clip_id} evaluation JSON must contain an object"
            )
        result["cached_evaluation_validation"] = _validate_cached_evaluation(
            result,
            clip,
            config_path=config_path,
        )
        result["evaluation_origin"] = "cached"
        return result
    if clip.video_path is None:
        raise ValueError(f"clip {clip.clip_id} is missing video")
    annotation_path = clip.annotation_path
    if annotation_path is None and clip.is_negative:
        annotation_path = _generated_negative_annotation(clip, output_dir)

    if evaluation_mode == "pipeline":
        from argus.validation.pipeline_video_eval import evaluate_pipeline_video

        effective_reference_path = scene_reference_path or clip.scene_reference_path
        effective_reference_frame = (
            scene_reference_frame_index
            if scene_reference_frame_index is not None
            else clip.scene_reference_frame_index
        )

        return evaluate_pipeline_video(
            clip.video_path,
            config_path=config_path,
            camera_id=clip.camera_id,
            annotation_path=annotation_path,
            model_path=model_path,
            model_version_id=model_version_id,
            sample_every=sample_every,
            max_frames=max_frames,
            warmup_seconds=warmup_seconds,
            scene_reference_path=effective_reference_path,
            scene_reference_frame_index=(
                effective_reference_frame
                if effective_reference_frame is not None
                else 0
            ),
        )

    from argus.anomaly.detector import AnomalibDetector
    from argus.validation.video_eval import evaluate_video

    detector = AnomalibDetector(
        model_path=model_path,
        threshold=threshold,
        image_size=(image_size, image_size),
    )
    if not detector.load() and model_path is not None:
        raise RuntimeError(f"failed to load detector for model: {model_path}")

    return evaluate_video(
        detector,
        clip.video_path,
        annotation_path=annotation_path,
        threshold=threshold,
        sample_every=sample_every,
        max_frames=max_frames,
        export_dir=output_dir / "review" / clip.clip_id,
        sweep=sweep,
    )


def _missing_clip_summary(clip: RegressionClip, reason: str) -> dict:
    return {
        "clip_id": clip.clip_id,
        "kind": clip.kind,
        "camera_id": clip.camera_id,
        "source_session_id": clip.source_session_id,
        "video_path": str(clip.video_path) if clip.video_path is not None else None,
        "annotation_path": str(clip.annotation_path) if clip.annotation_path is not None else None,
        "scene_reference_path": (
            str(clip.scene_reference_path)
            if clip.scene_reference_path is not None
            else None
        ),
        "tags": list(clip.tags),
        "description": clip.description,
        "status": "missing",
        "reason": reason,
    }


def _is_contained_file(root: Path, path: Path) -> bool:
    try:
        root_resolved = root.resolve(strict=True)
        path_resolved = path.resolve(strict=True)
        return root_resolved.is_dir() and path_resolved.is_file() and (
            path_resolved == root_resolved
            or root_resolved in path_resolved.parents
        )
    except (OSError, ValueError):
        return False


def _release_acceptance_policy(
    *,
    fp_gap_seconds: Any,
    warmup_seconds: Any,
    min_scene_guard_coverage: Any,
    min_effective_eval_fps: Any,
    min_projectile_eval_fps: Any,
) -> dict[str, Any]:
    """Describe whether diagnostic knobs exactly match production policy."""

    canonical = {
        "fp_gap_seconds": DEFAULT_FP_GAP_SECONDS,
        "warmup_seconds": DEFAULT_WARMUP_SECONDS,
        "min_scene_guard_coverage": DEFAULT_MIN_SCENE_GUARD_COVERAGE,
        "min_effective_eval_fps": DEFAULT_MIN_EFFECTIVE_EVAL_FPS,
        "min_projectile_eval_fps": DEFAULT_MIN_PROJECTILE_EVAL_FPS,
    }
    actual = {
        "fp_gap_seconds": fp_gap_seconds,
        "warmup_seconds": warmup_seconds,
        "min_scene_guard_coverage": min_scene_guard_coverage,
        "min_effective_eval_fps": min_effective_eval_fps,
        "min_projectile_eval_fps": min_projectile_eval_fps,
    }
    mismatches: list[str] = []
    for name, expected in canonical.items():
        value = actual[name]
        if (
            type(value) not in (int, float)
            or not math.isfinite(float(value))
            or float(value) != expected
        ):
            mismatches.append(name)
    return {
        "schema_version": RELEASE_ACCEPTANCE_POLICY_SCHEMA,
        "canonical": canonical,
        "actual": actual,
        "exact_match": not mismatches,
        "mismatches": mismatches,
    }


def _validate_baseline_artifact_v2(
    root: Path,
    *,
    camera_id: str,
    scene_profile_id: str,
) -> tuple[dict[str, Any], set[str], set[str]]:
    """Validate a V2/V3 baseline through the canonical release verifier.

    The legacy function name remains for callers that import it directly.
    Structurally valid V2 artifacts are accepted for historical/offline
    regression, while release eligibility is computed separately from the V3
    strict decoder evidence.
    """
    from argus.storage.release_evidence import (
        validate_baseline_manifest_source,
    )

    manifest, source_hashes, source_session_ids = (
        validate_baseline_manifest_source(root)
    )
    if (
        manifest.get("camera_id") != camera_id
        or manifest.get("scene_profile_id") != scene_profile_id
    ):
        raise ValueError("baseline camera/profile identity mismatch")
    return manifest, source_hashes, source_session_ids


def _evaluated_artifact_bindings(
    *,
    evaluation_mode: str,
    config_path: Path,
    requested_scene_profile: Path | None,
    requested_baseline: Path | None,
    requested_corpus: Path | None,
    manifest_path: Path | None,
    manifest_clips: list[RegressionClip],
    summaries: list[dict],
    output_dir: Path,
    fp_gap_seconds: float = DEFAULT_FP_GAP_SECONDS,
    warmup_seconds: float = DEFAULT_WARMUP_SECONDS,
    min_scene_guard_coverage: float = DEFAULT_MIN_SCENE_GUARD_COVERAGE,
    min_effective_eval_fps: float = DEFAULT_MIN_EFFECTIVE_EVAL_FPS,
    min_projectile_eval_fps: float = DEFAULT_MIN_PROJECTILE_EVAL_FPS,
) -> tuple[dict[str, str | None], dict[str, Any]]:
    """Bind release artifacts to what this exact replay actually consumed."""
    hashes: dict[str, str | None] = {
        "config_sha256": None,
        "scene_profile_sha256": None,
        "baseline_sha256": None,
        "corpus_sha256": None,
    }
    reasons: list[str] = []
    release_policy = _release_acceptance_policy(
        fp_gap_seconds=fp_gap_seconds,
        warmup_seconds=warmup_seconds,
        min_scene_guard_coverage=min_scene_guard_coverage,
        min_effective_eval_fps=min_effective_eval_fps,
        min_projectile_eval_fps=min_projectile_eval_fps,
    )
    if release_policy["exact_match"] is not True:
        reasons.append("non_canonical_release_acceptance_policy")
    evaluated = [item for item in summaries if item.get("status") == "evaluated"]
    if evaluation_mode != "pipeline" or not evaluated:
        reasons.append("pipeline_evaluation_required")
        return hashes, {
            "complete": False,
            "reasons": reasons,
            "release_acceptance_policy": release_policy,
            "evaluated_source_inventory": [],
            "evaluated_scene_profile": None,
        }

    config_hash = _evidence_artifact_sha256_if_present(config_path)
    config_matches = bool(
        config_hash
        and all(
            _same_path(Path(item["config_path"]), config_path)
            and item.get("config_artifact_sha256") == config_hash
            for item in evaluated
            if type(item.get("config_path")) is str
        )
        and all(type(item.get("config_path")) is str for item in evaluated)
    )
    if config_matches:
        hashes["config_sha256"] = config_hash
    else:
        reasons.append("config_not_bound_to_evaluated_pipeline")

    authorizations = [
        authorization
        for item in evaluated
        if isinstance(reference := item.get("scene_reference"), dict)
        if isinstance(
            authorization := reference.get("operator_authorization"),
            dict,
        )
    ]
    actual_profile_paths = [
        Path(value)
        for authorization in authorizations
        if type(value := authorization.get("profile_artifact_path")) is str
        and value
    ]
    actual_profile_path = (
        actual_profile_paths[0]
        if len(authorizations) == len(evaluated)
        and len(actual_profile_paths) == len(evaluated)
        and all(
            _same_path(path, actual_profile_paths[0])
            for path in actual_profile_paths
        )
        and all(
            path.name == (item.get("scene_reference") or {}).get("profile_id")
            and path.parent.name == item.get("camera_id")
            for item, path in zip(evaluated, actual_profile_paths, strict=True)
        )
        else None
    )
    if (
        actual_profile_path is not None
        and (
            requested_scene_profile is None
            or _same_path(requested_scene_profile, actual_profile_path)
        )
    ):
        hashes["scene_profile_sha256"] = (
            _evidence_artifact_sha256_if_present(actual_profile_path)
        )
    if hashes["scene_profile_sha256"] is None:
        reasons.append("scene_profile_not_bound_to_active_profile")

    baseline_lineage = [
        authorization.get("source_baseline_sha256")
        for authorization in authorizations
    ]
    baseline_names = [
        authorization.get("source_baseline_name")
        for authorization in authorizations
    ]
    valid_baseline_names = [
        value
        for value in baseline_names
        if type(value) is str and bool(value)
    ]
    valid_baseline_lineage = [
        value for value in baseline_lineage if _is_sha256(value)
    ]
    baseline_decoder_sufficient_for_model_release = False
    try:
        if requested_baseline is None or not requested_baseline.is_dir():
            raise ValueError("baseline artifact must be a directory")
        evaluated_camera_ids = {
            value
            for item in evaluated
            if type(value := item.get("camera_id")) is str and value
        }
        evaluated_profile_ids = {
            value
            for item in evaluated
            if isinstance(reference := item.get("scene_reference"), dict)
            if type(value := reference.get("profile_id")) is str and value
        }
        if len(evaluated_camera_ids) != 1 or len(evaluated_profile_ids) != 1:
            raise ValueError("evaluated camera/profile identity is inconsistent")
        (
            baseline_manifest,
            training_source_hashes,
            training_source_session_ids,
        ) = _validate_baseline_artifact_v2(
            requested_baseline,
            camera_id=next(iter(evaluated_camera_ids)),
            scene_profile_id=next(iter(evaluated_profile_ids)),
        )
        baseline_decoder_sufficient_for_model_release = (
            isinstance(baseline_manifest.get("decoder_evidence"), dict)
            and baseline_manifest["decoder_evidence"].get(
                "sufficient_for_model_release"
            )
            is True
        )
        baseline_reference = baseline_manifest.get("reference")
        if not isinstance(baseline_reference, dict):
            raise ValueError("baseline reference is missing")
        if (
            len(valid_baseline_lineage) != len(evaluated)
            or len(set(valid_baseline_lineage)) != 1
            or baseline_reference.get("sha256") != valid_baseline_lineage[0]
            or len(valid_baseline_names) != len(evaluated)
            or len(set(valid_baseline_names)) != 1
            or Path(str(baseline_reference.get("copied_path"))).name
            != valid_baseline_names[0]
        ):
            raise ValueError("baseline manifest does not match active profile lineage")
        evaluated_video_hashes = {
            value
            for item in evaluated
            if _is_sha256(value := item.get("source_video_sha256"))
        }
        if training_source_hashes.intersection(evaluated_video_hashes):
            raise ValueError("training and regression source hashes overlap")
        evaluated_source_session_ids = {
            value
            for item in evaluated
            if type(value := item.get("source_session_id")) is str and value
        }
        if len(evaluated_source_session_ids) != len(evaluated):
            raise ValueError("regression source session identity is incomplete or duplicated")
        if training_source_session_ids.intersection(evaluated_source_session_ids):
            raise ValueError("training and regression source sessions overlap")
        hashes["baseline_sha256"] = _evidence_artifact_sha256_if_present(
            requested_baseline
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        hashes["baseline_sha256"] = None
        reasons.append(f"baseline_validation_failed:{exc}")
    if hashes["baseline_sha256"] is None:
        reasons.append("baseline_not_bound_to_scene_profile_lineage")
    if not baseline_decoder_sufficient_for_model_release:
        reasons.append("baseline_decoder_insufficient_for_model_release")

    corpus_members = [
        path
        for clip in manifest_clips
        for path in (clip.video_path, clip.annotation_path)
        if path is not None
    ]
    corpus_session_ids = [clip.source_session_id for clip in manifest_clips]
    valid_corpus_session_ids = [
        value
        for value in corpus_session_ids
        if type(value) is str and value and value == value.strip()
    ]
    corpus_valid = bool(
        requested_corpus is not None
        and requested_corpus.is_dir()
        and manifest_path is not None
        and _is_contained_file(requested_corpus, manifest_path)
        and manifest_clips
        and len(corpus_members) == 2 * len(manifest_clips)
        and len(valid_corpus_session_ids) == len(manifest_clips)
        and len(set(valid_corpus_session_ids)) == len(manifest_clips)
        and all(
            _is_contained_file(requested_corpus, member)
            for member in corpus_members
        )
        and not (
            requested_corpus.resolve(strict=True)
            in output_dir.resolve(strict=False).parents
            or requested_corpus.resolve(strict=True)
            == output_dir.resolve(strict=False)
        )
    )
    if corpus_valid:
        hashes["corpus_sha256"] = _evidence_artifact_sha256_if_present(
            requested_corpus
        )
    else:
        reasons.append("corpus_root_not_bound_to_manifest_inventory")

    evaluated_source_inventory: list[dict[str, Any]] = []
    if corpus_valid and requested_corpus is not None:
        summary_by_id = {
            str(item.get("clip_id")): item for item in evaluated
        }
        if len(summary_by_id) == len(manifest_clips) == len(evaluated):
            for clip in manifest_clips:
                item = summary_by_id.get(clip.clip_id)
                if (
                    item is None
                    or clip.video_path is None
                    or clip.annotation_path is None
                    or not _is_sha256(item.get("source_video_sha256"))
                    or not _is_sha256(item.get("annotation_sha256"))
                    or item.get("camera_id") != clip.camera_id
                    or item.get("source_session_id") != clip.source_session_id
                    or not _same_path(Path(str(item.get("video_path"))), clip.video_path)
                    or not _same_path(
                        Path(str(item.get("annotation_path"))),
                        clip.annotation_path,
                    )
                ):
                    evaluated_source_inventory = []
                    break
                root = requested_corpus.resolve(strict=True)
                evaluated_source_inventory.append({
                    "clip_id": clip.clip_id,
                    "camera_id": clip.camera_id,
                    "source_session_id": clip.source_session_id,
                    "video_relative_path": (
                        clip.video_path.resolve(strict=True)
                        .relative_to(root)
                        .as_posix()
                    ),
                    "source_video_sha256": item["source_video_sha256"],
                    "annotation_relative_path": (
                        clip.annotation_path.resolve(strict=True)
                        .relative_to(root)
                        .as_posix()
                    ),
                    "annotation_sha256": item["annotation_sha256"],
                })
    if len(evaluated_source_inventory) != len(evaluated):
        evaluated_source_inventory = []
        hashes["corpus_sha256"] = None
        if "corpus_root_not_bound_to_manifest_inventory" not in reasons:
            reasons.append("evaluated_source_inventory_not_bound")

    scene_profile_rows: list[tuple[str, str, str, str, str, str]] = []
    for item in evaluated:
        reference = item.get("scene_reference")
        identity = item.get("model_identity")
        if not isinstance(reference, dict) or not isinstance(identity, dict):
            continue
        authorization = reference.get("operator_authorization")
        actual_model = identity.get("actual")
        if not isinstance(authorization, dict) or not isinstance(actual_model, dict):
            continue
        values = (
            item.get("camera_id"),
            reference.get("profile_id"),
            reference.get("reference_sha256"),
            authorization.get("source_baseline_sha256"),
            actual_model.get("version_id"),
            actual_model.get("artifact_sha256"),
        )
        if all(type(value) is str and value for value in values):
            scene_profile_rows.append(values)
    evaluated_scene_profile = None
    if (
        len(scene_profile_rows) == len(evaluated)
        and len(set(scene_profile_rows)) == 1
        and hashes["scene_profile_sha256"] is not None
    ):
        row = scene_profile_rows[0]
        evaluated_scene_profile = {
            "camera_id": row[0],
            "profile_id": row[1],
            "reference_sha256": row[2],
            "source_baseline_sha256": row[3],
            "model_version_id": row[4],
            "model_artifact_sha256": row[5],
        }
    else:
        hashes["scene_profile_sha256"] = None
        reasons.append("evaluated_scene_profile_not_consistent")

    complete = bool(
        all(_is_sha256(value) for value in hashes.values())
        and evaluated_source_inventory
        and evaluated_scene_profile is not None
        and baseline_decoder_sufficient_for_model_release
        and release_policy["exact_match"] is True
    )
    return hashes, {
        "complete": complete,
        "reasons": reasons,
        "release_acceptance_policy": release_policy,
        "scene_profile_path": (
            str(actual_profile_path) if actual_profile_path is not None else None
        ),
        "baseline_lineage_sha256": (
            valid_baseline_lineage[0]
            if len(set(valid_baseline_lineage)) == 1
            and valid_baseline_lineage
            else None
        ),
        "baseline_decoder_sufficient_for_model_release": (
            baseline_decoder_sufficient_for_model_release
        ),
        "evaluated_source_inventory": evaluated_source_inventory,
        "evaluated_scene_profile": evaluated_scene_profile,
    }


def run_suite(args: argparse.Namespace) -> dict:
    if (
        getattr(args, "scene_reference_frame", None) is not None
        and args.scene_reference_frame < 0
    ):
        raise ValueError("--scene-reference-frame must be >= 0")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_raw: dict = {}
    clips: list[RegressionClip] = []
    manifest_clips: list[RegressionClip] = []
    expected_clip_count = 0
    if args.from_eval_json:
        for path in args.from_eval_json:
            eval_path = Path(path)
            clip_id = eval_path.stem.replace(".video_eval", "")
            clips.append(RegressionClip(
                clip_id=clip_id,
                kind="from_eval_json",
                camera_id="unknown",
                evaluation_json_path=eval_path,
            ))
        expected_clip_count = len(clips)
    else:
        manifest_raw, clips = load_manifest(args.manifest)
        manifest_clips = list(clips)
        # Keep the complete manifest inventory as the release denominator.
        # ``--clip-id`` is useful for diagnosis, but a selected subset must
        # never be able to satisfy a full-corpus acceptance run.
        expected_clip_count = len(clips)
        selected_clip_ids = set(getattr(args, "clip_id", []) or [])
        if selected_clip_ids:
            available = {clip.clip_id for clip in clips}
            unknown = sorted(selected_clip_ids - available)
            if unknown:
                raise ValueError(f"unknown clip id(s): {', '.join(unknown)}")
            clips = [clip for clip in clips if clip.clip_id in selected_clip_ids]

    summaries: list[dict] = []
    if not clips:
        status = "no_clips"
    else:
        status = "ok"

    for clip in clips:
        missing_reason = None
        if clip.evaluation_json_path is not None and not clip.evaluation_json_path.is_file():
            missing_reason = f"evaluation_json not found: {clip.evaluation_json_path}"
        elif clip.evaluation_json_path is None:
            if clip.video_path is None or not clip.video_path.is_file():
                missing_reason = f"video not found: {clip.video_path}"
            elif clip.annotation_path is not None and not clip.annotation_path.is_file():
                missing_reason = f"annotations not found: {clip.annotation_path}"
            else:
                scene_reference = (
                    getattr(args, "scene_reference", None)
                    or clip.scene_reference_path
                )
                if (
                    getattr(args, "evaluation_mode", "detector") == "pipeline"
                    and scene_reference is not None
                    and not Path(scene_reference).is_file()
                ):
                    missing_reason = f"scene reference not found: {scene_reference}"

        if missing_reason is not None:
            status = "missing_inputs"
            summaries.append(_missing_clip_summary(clip, missing_reason))
            continue

        if args.dry_run:
            summaries.append({
                "clip_id": clip.clip_id,
                "kind": clip.kind,
                "camera_id": clip.camera_id,
                "source_session_id": clip.source_session_id,
                "video_path": str(clip.video_path) if clip.video_path is not None else None,
                "annotation_path": (
                    str(clip.annotation_path) if clip.annotation_path is not None else None
                ),
                "tags": list(clip.tags),
                "description": clip.description,
                "scene_reference_path": (
                    str(getattr(args, "scene_reference", None) or clip.scene_reference_path)
                    if getattr(args, "scene_reference", None) or clip.scene_reference_path
                    else None
                ),
                "status": "listed",
            })
            continue

        try:
            result = _evaluate_clip(
                clip,
                output_dir=output_dir,
                model_path=args.model,
                threshold=args.threshold,
                image_size=args.image_size,
                sample_every=args.sample_every,
                max_frames=args.max_frames,
                sweep=args.sweep,
                evaluation_mode=getattr(args, "evaluation_mode", "detector"),
                config_path=getattr(args, "config", Path("configs/default.yaml")),
                warmup_seconds=getattr(
                    args,
                    "warmup_seconds",
                    DEFAULT_WARMUP_SECONDS,
                ),
                scene_reference_path=getattr(args, "scene_reference", None),
                scene_reference_frame_index=getattr(
                    args,
                    "scene_reference_frame",
                    None,
                ),
                model_version_id=getattr(args, "model_version_id", None),
            )
            eval_json_path = output_dir / "video_eval" / f"{clip.clip_id}.json"
            _write_json(eval_json_path, result)
            candidate_events = load_candidate_events(clip.candidate_events_path)
            summaries.append(summarize_clip_result(
                result,
                clip,
                candidate_events=candidate_events,
                fp_gap_seconds=args.fp_gap_seconds,
                default_channel=args.default_channel,
            ) | {"evaluation_json": str(eval_json_path)})
        except Exception as exc:
            status = "error"
            summaries.append(_missing_clip_summary(clip, f"{type(exc).__name__}: {exc}"))
            if not args.allow_missing:
                break

    min_scene_guard_coverage = float(
        getattr(
            args,
            "min_scene_guard_coverage",
            DEFAULT_MIN_SCENE_GUARD_COVERAGE,
        )
    )
    min_effective_eval_fps = float(
        getattr(
            args,
            "min_effective_eval_fps",
            DEFAULT_MIN_EFFECTIVE_EVAL_FPS,
        )
    )
    min_projectile_eval_fps = float(
        getattr(
            args,
            "min_projectile_eval_fps",
            DEFAULT_MIN_PROJECTILE_EVAL_FPS,
        )
    )
    evaluation_mode = getattr(args, "evaluation_mode", "detector")
    artifact_inputs = {
        "config": Path(args.config) if getattr(args, "config", None) else None,
        "scene_profile": getattr(args, "scene_profile_artifact", None),
        "baseline": getattr(args, "baseline_artifact", None),
        "corpus": getattr(args, "corpus_artifact", None),
    }
    evaluated_artifact_hashes, artifact_binding = _evaluated_artifact_bindings(
        evaluation_mode=evaluation_mode,
        config_path=artifact_inputs["config"] or Path(""),
        requested_scene_profile=artifact_inputs["scene_profile"],
        requested_baseline=artifact_inputs["baseline"],
        requested_corpus=artifact_inputs["corpus"],
        manifest_path=(
            Path(args.manifest) if not args.from_eval_json else None
        ),
        manifest_clips=manifest_clips,
        summaries=summaries,
        output_dir=output_dir,
        fp_gap_seconds=getattr(args, "fp_gap_seconds", DEFAULT_FP_GAP_SECONDS),
        warmup_seconds=getattr(args, "warmup_seconds", DEFAULT_WARMUP_SECONDS),
        min_scene_guard_coverage=min_scene_guard_coverage,
        min_effective_eval_fps=min_effective_eval_fps,
        min_projectile_eval_fps=min_projectile_eval_fps,
    )
    suite = aggregate_suite(
        summaries,
        min_scene_guard_coverage=min_scene_guard_coverage,
        min_effective_eval_fps=min_effective_eval_fps,
        min_projectile_eval_fps=min_projectile_eval_fps,
        expected_clip_count=expected_clip_count,
        evaluated_artifact_hashes=evaluated_artifact_hashes,
    )
    suite["evaluated_source_inventory"] = artifact_binding[
        "evaluated_source_inventory"
    ]
    suite["evaluated_scene_profile"] = artifact_binding[
        "evaluated_scene_profile"
    ]
    suite["evaluated_artifact_binding_evidence_complete"] = artifact_binding[
        "complete"
    ]
    suite["release_acceptance_policy"] = artifact_binding[
        "release_acceptance_policy"
    ]
    suite["acceptance"]["release_acceptance_policy"] = (
        "pass"
        if artifact_binding["release_acceptance_policy"]["exact_match"] is True
        else "fail"
    )
    suite["acceptance"]["evaluated_artifact_binding"] = (
        "pass" if artifact_binding["complete"] is True else "fail"
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "suite_id": manifest_raw.get("suite_id") if manifest_raw else None,
        "status": status,
        "manifest": str(args.manifest) if not args.from_eval_json else None,
        "threshold": args.threshold,
        "evaluation_mode": getattr(args, "evaluation_mode", "detector"),
        "evaluated_artifact_inputs": {
            key: str(path) if path is not None else None
            for key, path in artifact_inputs.items()
        },
        "evaluated_artifact_binding": artifact_binding,
        "scene_reference": (
            str(getattr(args, "scene_reference", None))
            if getattr(args, "scene_reference", None) is not None
            else None
        ),
        "scene_reference_frame": getattr(args, "scene_reference_frame", None),
        "min_scene_guard_coverage": min_scene_guard_coverage,
        "min_effective_eval_fps": min_effective_eval_fps,
        "min_projectile_eval_fps": min_projectile_eval_fps,
        "warmup_seconds": getattr(args, "warmup_seconds", DEFAULT_WARMUP_SECONDS),
        "expected_clip_count": expected_clip_count,
        "fp_gap_seconds": args.fp_gap_seconds,
        "suite": suite,
        "clips": summaries,
    }
    _write_json(output_dir / "regression_report.json", payload)
    (output_dir / "regression_report.md").write_text(
        build_markdown_report(payload),
        encoding="utf-8",
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/regression/manifest.yaml"))
    parser.add_argument(
        "--clip-id",
        action="append",
        help="Evaluate only the selected manifest clip; repeat for multiple clips",
    )
    parser.add_argument("--from-eval-json", type=Path, action="append")
    parser.add_argument("--model", type=Path, help="Model file or directory; omitted uses fallback detector")
    parser.add_argument(
        "--model-version-id",
        help=(
            "Exact production model version used in candidate-evidence calibration binding; "
            "otherwise derived from the model directory"
        ),
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument(
        "--evaluation-mode",
        choices=["detector", "pipeline"],
        default="detector",
        help="Evaluate raw detector scores or final production-pipeline alerts",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--scene-profile-artifact",
        type=Path,
        help="Immutable scene-profile file or directory evaluated by this run",
    )
    parser.add_argument(
        "--baseline-artifact",
        type=Path,
        help="Immutable baseline file or directory evaluated by this run",
    )
    parser.add_argument(
        "--corpus-artifact",
        type=Path,
        help="Immutable regression corpus manifest or directory evaluated by this run",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=DEFAULT_WARMUP_SECONDS,
        help="Suppress pipeline alerts during the initial video warmup",
    )
    parser.add_argument(
        "--scene-reference",
        type=Path,
        help=(
            "Approved scene reference image for pipeline replay; otherwise use "
            "the manifest value or an explicit source-video frame"
        ),
    )
    parser.add_argument(
        "--scene-reference-frame",
        type=int,
        help="Source-video frame to use as the explicit scene reference (default: 0)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/regression/reports/current"))
    parser.add_argument("--fp-gap-seconds", type=float, default=DEFAULT_FP_GAP_SECONDS)
    parser.add_argument(
        "--min-scene-guard-coverage",
        type=float,
        default=DEFAULT_MIN_SCENE_GUARD_COVERAGE,
        help=(
            "Minimum usable-frame ratio for production-pipeline replays "
            "(default: 0.99)"
        ),
    )
    parser.add_argument(
        "--min-effective-eval-fps",
        type=float,
        default=DEFAULT_MIN_EFFECTIVE_EVAL_FPS,
        help=(
            "Minimum effective replay sampling rate accepted as release "
            "evidence (default: 5 FPS)"
        ),
    )
    parser.add_argument(
        "--min-projectile-eval-fps",
        type=float,
        default=DEFAULT_MIN_PROJECTILE_EVAL_FPS,
        help=(
            "Minimum source and effective FPS for projectile release "
            "evidence (default: 59 FPS)"
        ),
    )
    parser.add_argument("--default-channel", default="anomaly")
    parser.add_argument("--dry-run", action="store_true", help="Validate inventory without inference")
    parser.add_argument("--allow-missing", action="store_true", help="Report missing inputs instead of failing")
    parser.add_argument("--fail-on-acceptance", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run_suite(args)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    report_path = Path(args.output_dir) / "regression_report.md"
    print(f"Regression report written to {report_path}")
    status = payload["status"]
    if status in {"no_clips", "missing_inputs", "error"}:
        print(f"STATUS: {status}", file=sys.stderr)
        if not args.allow_missing and not args.dry_run:
            return 2
    if args.fail_on_acceptance:
        suite = payload["suite"]
        acceptance = suite["acceptance"]
        not_passed = [
            name
            for name, value in acceptance.items()
            if value not in {"pass", "not_applicable"}
        ]
        if (
            suite.get("evaluated_artifact_binding_evidence_complete") is not True
            and "evaluated_artifact_binding" not in not_passed
        ):
            not_passed.append("evaluated_artifact_binding")
        if not_passed:
            print(
                "FAIL: acceptance checks did not pass: "
                f"{', '.join(not_passed)}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
