"""Optional, offline FiftyOne analysis of verified Argus review bundles.

Importing this module never imports FiftyOne, starts MongoDB, decodes media or
opens a browser. Call :func:`import_bundle_to_fiftyone` in the analysis venv.
FiftyOne frame numbers are one-based; Argus indices and exact PTS remain stored
as separate fields. Saved FP/FN views are frame-level triage, never release or
event-metric evidence. New annotations require a fresh Argus pipeline evaluation.

An Argus-owned dataset represents one suite. Its ``argus_*`` observation fields
and frame labels below are managed by this importer; human scene-review fields
and unrelated user fields survive refreshes. A dataset name is never adopted
from an unrelated user dataset. Scene comparisons are diagnostic review aids,
never inputs to the production SceneGuard gate or anomaly ground truth.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import socket
import tempfile
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

from argus.integrations.contracts import ReviewBox, ReviewBundle, ReviewClip, require_verified_media
from argus.integrations.scene_comparison import (
    SceneComparisonReview,
    require_verified_scene_comparisons,
)

_OWNER = "argus.video-review.fiftyone.v1"
_INFO_KEY = "argus_review"
_FRAME_WRITE_BATCH_SIZE = 256
_FRAME_SCHEMA_REVISION = 2
_SCENE_HUMAN_STATES = ("same_pose", "moved", "uncertain", "unreviewable")
_SCENE_HUMAN_FIELDS = (
    "argus_scene_human_state", "argus_scene_human_note", "argus_scene_reviewed_report_sha256",
)
_SCENE_FRAME_STRINGS = (
    "argus_scene_report_sha256", "argus_scene_frame_sha256", "argus_scene_review_category",
    "argus_scene_error_algorithms_json", "argus_scene_baseline_status",
    "argus_scene_baseline_state", "argus_scene_baseline_raw_state", "argus_scene_baseline_reason",
    "argus_scene_baseline_metrics_json", "argus_scene_lightglue_status",
    "argus_scene_lightglue_state", "argus_scene_lightglue_raw_state",
    "argus_scene_lightglue_reason",
    "argus_scene_lightglue_metrics_json", "argus_scene_lightglue_matcher_json",
)
_SCENE_FRAME_BOOLEANS = (
    "argus_scene_has_comparison", "argus_scene_comparable", "argus_scene_disagreement",
    "argus_scene_raw_disagreement", "argus_scene_comparison_error",
)
_SCENE_FRAME_FLOATS = ("argus_scene_baseline_elapsed_ms", "argus_scene_lightglue_elapsed_ms")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _clip_fingerprint(clip: ReviewClip, comparison: SceneComparisonReview | None) -> str:
    # Schema upgrades must rebuild managed fields even when media/report bytes
    # and the dataset generation are unchanged. Human fields remain untouched.
    payload = {
        "frame_schema_revision": _FRAME_SCHEMA_REVISION,
        "clip": clip.model_dump(mode="json"),
    }
    if comparison is not None:
        payload.update({
            "scene_comparison_sha256": comparison.report_sha256,
            "scene_comparison_path": str(comparison.report_path),
        })
    return _digest(payload)


def _scene_frame_fields(
    comparison: SceneComparisonReview | None, row: dict[str, Any] | None,
) -> dict[str, Any]:
    """Diagnostic observations only; never overwrite evaluation/GT fields.

    Explicit nulls clear stale managed observations when a report is removed or
    a frame is no longer sampled. Human notes and their report-hash binding are
    deliberately excluded, so refreshes cannot erase or auto-approve reviews.
    """
    fields: dict[str, Any] = {
        **dict.fromkeys(_SCENE_FRAME_STRINGS),
        **dict.fromkeys(_SCENE_FRAME_FLOATS),
        **dict.fromkeys(_SCENE_FRAME_BOOLEANS, False),
    }
    if comparison is None or row is None:
        return fields
    fields.update({
        "argus_scene_has_comparison": True,
        "argus_scene_report_sha256": comparison.report_sha256,
        "argus_scene_frame_sha256": row["frame_sha256"],
        "argus_scene_review_category": row["review_category"],
        "argus_scene_comparable": row["comparable"],
        # Null, rather than false, means an error prevented this comparison.
        "argus_scene_disagreement": row["state_disagreement"],
        "argus_scene_raw_disagreement": row["raw_state_disagreement"],
        "argus_scene_comparison_error": row["comparison_error"],
        "argus_scene_error_algorithms_json": _json(row["error_algorithms"]),
    })
    for algorithm in ("baseline", "lightglue"):
        decision = row[algorithm]
        prefix = f"argus_scene_{algorithm}_"
        for key in ("status", "state", "raw_state", "reason", "elapsed_ms"):
            fields[prefix + key] = decision.get(key)
        metrics = decision.get("metrics")
        fields[prefix + "metrics_json"] = _json(metrics) if metrics is not None else None
    matcher = row["lightglue"].get("matcher")
    fields["argus_scene_lightglue_matcher_json"] = _json(matcher) if matcher is not None else None
    return fields


@contextmanager
def _dataset_import_lock(dataset_name: str) -> Iterator[None]:
    """Serialize local CLI importers, including after a failed import/restart.

    A foreign-host in-progress marker is separately refused below. FiftyOne's
    public dataset API does not offer a cross-host import transaction, so a
    dataset must have one importing host. The lock is released by process exit.
    """
    name = hashlib.sha256(dataset_name.encode("utf-8")).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / f"argus-fiftyone-{name}.lock"
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        acquired = False
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
            yield
        except OSError as exc:
            if not acquired:
                raise RuntimeError("this FiftyOne dataset already has a local importer") from exc
            raise
        finally:
            if acquired:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _media_identity(clip: ReviewClip) -> str:
    """Relocation is allowed; different bytes/frame mapping need a new dataset."""
    return _digest({
        "video_sha256": clip.video_sha256,
        "frame_count": clip.frame_count,
        "width": clip.width,
        "height": clip.height,
        "fps": clip.fps,
        "timestamps": [stamp.model_dump(mode="json") for stamp in clip.timestamps],
    })


def _require_scene_player_grid(clip: ReviewClip) -> None:
    """Reject unsupported video seek grids before starting FiftyOne.

    The FiftyOne video player seeks/labels frames using nominal FPS rather than
    our diagnostic PTS fields. Exact report-to-bundle equality alone cannot make
    a VFR source safe for visual frame review. This conservative metadata check
    is not a substitute for browser/codec seek acceptance on the analysis host.
    """
    period = 1 / Fraction(str(clip.fps))
    first = clip.timestamps[0]
    origin = first.pts * Fraction(first.time_base_num, first.time_base_den)
    # Upstream getTime() seeks 0.01 nominal frame periods after the boundary.
    # Keep the maximum phase error below half that margin, even for a coarse
    # source time base. Such coarse CFR sources may be conservatively rejected.
    max_phase_error = period / 200
    for index, stamp in enumerate(clip.timestamps):
        tick = Fraction(stamp.time_base_num, stamp.time_base_den)
        relative = stamp.pts * tick - origin
        tolerance = min(max(tick / 2, Fraction(1, 1_000_000_000)), max_phase_error)
        if abs(relative - index * period) > tolerance:
            raise ValueError(
                "FiftyOne scene video review requires a nominal-FPS seek grid; "
                f"clip {clip.clip_id!r} frame {index} has unsupported PTS drift/VFR. "
                "The report may still be inspected through the read-only scene "
                "comparison bridge; do not approve frames from this video player."
            )


def _evaluation_rows(clip: ReviewClip) -> tuple[dict[int, dict], dict[int, str]]:
    report = clip.evaluation or {}
    records = {row["frame_index"]: row for row in report.get("frames", [])}
    # A matching media hash alone does not validate a decoder's frame map.
    # Reject contradictory source metadata rather than plotting shifted rows
    # against the exact original-frame timeline preserved in this bundle.
    reported_fps = report.get("fps")
    if reported_fps is not None and (
        type(reported_fps) not in (int, float) or not math.isfinite(reported_fps)
        or reported_fps <= 0
        or not math.isclose(reported_fps, clip.fps, rel_tol=1e-6, abs_tol=1e-6)
    ):
        raise ValueError("evaluation fps differs from the original video frame mapping")
    source_count = report.get("source_total_frames")
    if report.get("source_total_frames_known") is True and (
        type(source_count) is not int or source_count != clip.frame_count
    ):
        raise ValueError("evaluation source frame count differs from the original video")
    evaluated_count = report.get("total_frames")
    if evaluated_count is not None and (
        type(evaluated_count) is not int or not 0 <= evaluated_count <= clip.frame_count
    ):
        raise ValueError("evaluation total_frames is outside the original video")
    if (
        evaluated_count is not None and report.get("source_exhausted") is True
        and evaluated_count != clip.frame_count
    ):
        raise ValueError("exhausted evaluation did not preserve every original source frame")
    stride = report.get("sample_every")
    if stride is not None and (type(stride) is not int or stride <= 0):
        raise ValueError("evaluation sample_every must be a positive integer")
    processed_count = report.get("processed_frames")
    if processed_count is not None and (
        type(processed_count) is not int or processed_count != len(records)
    ):
        raise ValueError("evaluation processed_frames differs from its frame records")
    failures: dict[int, str] = {}
    rows = report.get("failed_frames", [])
    if not isinstance(rows, list):
        raise ValueError("failed_frames must be a list")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("failed frame records must be objects")
        index = row.get("frame_index")
        if type(index) is not int or not 0 <= index < clip.frame_count:
            raise ValueError("failed frame index is outside the original video")
        if evaluated_count is not None and index >= evaluated_count:
            raise ValueError("failed frame index exceeds the evaluated source frame range")
        if stride is not None and index % stride:
            raise ValueError("failed frame is not on the original sampled-frame grid")
        if index in failures or index in records:
            raise ValueError("duplicate or contradictory evaluation frame record")
        failures[index] = str(row.get("error") or "pipeline evaluation failed")
    for row in records.values():
        index = row["frame_index"]
        if evaluated_count is not None and index >= evaluated_count:
            raise ValueError("evaluation row exceeds the evaluated source frame range")
        if stride is not None and index % stride:
            raise ValueError("evaluation row is not on the original sampled-frame grid")
        predicted = row.get("predicted")
        if predicted is not None and not (
            type(predicted) is bool or type(predicted) is int and predicted in (0, 1)
        ):
            raise ValueError("pipeline predictions must be boolean or 0/1")
        score = row.get("score")
        if score is not None and (
            type(score) not in (int, float) or not math.isfinite(score)
        ):
            raise ValueError("pipeline score must be a finite number or null")
        if not isinstance(row.get("scene_guard", {}), dict):
            raise ValueError("scene_guard must be an object")
        label = row.get("label")
        if label is not None and not (
            type(label) is bool or type(label) is int and label in (0, 1)
        ):
            raise ValueError("pipeline labels must be boolean or 0/1")
        if row.get("score_eligible") is not None and type(row["score_eligible"]) is not bool:
            raise ValueError("score_eligible must be boolean or null")
        for key in ("decision", "exclusion_reason", "pipeline_mode"):
            if row.get(key) is not None and not isinstance(row[key], str):
                raise ValueError(f"{key} must be a string or null")
        timestamp = row.get("timestamp_sec")
        if timestamp is not None and (
            type(timestamp) not in (int, float) or not math.isfinite(timestamp)
        ):
            raise ValueError("evaluation timestamp_sec must be finite or null")
        if timestamp is not None and not math.isclose(
            timestamp, index / clip.fps, rel_tol=1e-7, abs_tol=1e-6,
        ):
            raise ValueError("evaluation timestamp_sec does not match its original frame index")
        scene = row.get("scene_guard") or {}
        if scene.get("usable") is not None and type(scene["usable"]) is not bool:
            raise ValueError("scene_guard.usable must be boolean or null")
        for key in ("state", "reason", "profile_id"):
            if scene.get(key) is not None and not isinstance(scene[key], str):
                raise ValueError(f"scene_guard.{key} must be a string or null")
    identity = report.get("model_identity") or {}
    if not isinstance(identity, dict) or not isinstance(identity.get("actual", {}), dict):
        raise ValueError("model_identity and actual identity must be objects")
    if not isinstance(report.get("scene_reference", {}), dict):
        raise ValueError("scene_reference must be an object")
    for name, value in (
        ("model_version_id", report.get("model_version_id")),
        ("actual.version_id", identity.get("actual", {}).get("version_id")),
        ("actual.artifact_sha256", identity.get("actual", {}).get("artifact_sha256")),
        ("scene_reference.profile_id", report.get("scene_reference", {}).get("profile_id")),
    ):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} must be a string or null")
    return records, failures


def _frame_fields(
    clip: ReviewClip,
    index: int,
    record: dict[str, Any] | None,
    error: str | None,
    frame_boxes: list[ReviewBox] | None = None,
) -> dict[str, Any]:
    """Derive triage from the current review, not the report's stale labels."""
    annotations = clip.annotations
    positive = [
        item for item in annotations.positive_ranges
        if item.start_frame <= index <= item.end_frame
    ]
    ignored = [
        item for item in annotations.ignore_ranges
        if item.start_frame <= index <= item.end_frame
    ]
    boxes = frame_boxes if frame_boxes is not None else [
        box for box in annotations.boxes if box.frame_index == index
    ]
    # A real reviewed box is positive evidence even without a temporal range.
    truth = None if ignored or not annotations.reviewed else bool(positive or boxes)
    review_state = (
        "ignored" if ignored else "unreviewed" if not annotations.reviewed
        else "positive" if truth else "normal"
    )
    row = record or {}
    scene = row.get("scene_guard") or {}
    usable = scene.get("usable") if type(scene.get("usable")) is bool else None
    sampled = record is not None or error is not None
    report_decision = row.get("decision")
    blocked = usable is False or report_decision == "blocked"
    mode = row.get("pipeline_mode")
    operational = mode in {"active", "maintenance"}
    report_failed = report_decision in {"error", "failed", "no_frame"}
    source_stable = (clip.evaluation or {}).get("source_artifacts_stable") is True
    eligible = (
        row.get("score_eligible") is True and operational and source_stable
        and not blocked and not report_failed
        and report_decision not in {"unavailable", "excluded"}
    )
    predicted = row.get("predicted")
    predicted = bool(predicted) if predicted is not None else None
    if error or report_failed:
        decision = "error"
    elif blocked:
        decision = "blocked"
    elif review_state in {"ignored", "unreviewed"}:
        decision = review_state
    elif not sampled:
        decision = "unsampled"
    elif not source_stable or mode is None or report_decision == "unavailable":
        decision = "unavailable"
    elif not eligible or report_decision == "excluded":
        decision = "excluded"
    elif usable is not True or predicted is None:
        decision = "unavailable"
    else:
        decision = ("tp" if truth else "fp") if predicted else ("fn" if truth else "tn")
    stamp = clip.timestamps[index]
    first = clip.timestamps[0]
    exact_timestamp = stamp.pts * Fraction(stamp.time_base_num, stamp.time_base_den)
    relative_timestamp = (
        exact_timestamp - first.pts * Fraction(first.time_base_num, first.time_base_den)
    )
    return {
        "argus_frame_index": index,
        "argus_pts": stamp.pts,
        # BSON integer values can exceed JavaScript's exact integer range.
        # Keep legacy numeric fields, but expose lossless copyable identities
        # for the App and human-review notes as strings as well.
        "argus_pts_exact": str(stamp.pts),
        "argus_time_base_exact": f"{stamp.time_base_num}/{stamp.time_base_den}",
        "argus_time_base_num": stamp.time_base_num,
        "argus_time_base_den": stamp.time_base_den,
        "argus_timestamp_sec": float(exact_timestamp),
        "argus_relative_timestamp_sec": float(relative_timestamp),
        "argus_eval_timestamp_sec": row.get("timestamp_sec"),
        "argus_review_state": review_state,
        "argus_positive_labels": sorted({item.label for item in positive} | {box.label for box in boxes}),
        "argus_event_ids": sorted({item.event_id for item in positive if item.event_id is not None}),
        "argus_ignore_labels": sorted({item.label for item in ignored}),
        "argus_ignored": bool(ignored),
        "argus_sampled": sampled,
        "argus_score": row.get("score"),
        "argus_predicted": predicted,
        "argus_score_eligible": eligible if record is not None else None,
        "argus_scene_guard_usable": usable,
        "argus_scene_guard_state": scene.get("state"),
        "argus_scene_guard_reason": scene.get("reason"),
        "argus_scene_profile_id": scene.get("profile_id"),
        "argus_blocked": blocked,
        "argus_error": error or (f"evaluation decision: {report_decision}" if report_failed else None),
        "argus_decision": decision,
        "argus_eval_decision": row.get("decision"),
        "argus_eval_label": int(row["label"]) if row.get("label") is not None else None,
        "argus_exclusion_reason": row.get("exclusion_reason") or (
            f"pipeline_mode_{mode}" if not operational else None
        ),
        "argus_pipeline_mode": mode,
        "argus_truth": truth,
    }


def _ensure_schema(dataset: Any, fo: Any) -> None:
    sample_strings = (
        "argus_owner", "argus_clip_id", "argus_suite_id", "argus_bundle_id",
        "argus_manifest_sha256", "argus_video_sha256", "argus_media_identity",
        "argus_import_state", "argus_import_generation", "argus_clip_fingerprint",
        "argus_camera_id", "argus_source_session_id", "argus_split", "argus_kind",
        "argus_scene_profile_id", "argus_expected_scene_profile_id",
        "argus_model_version_id", "argus_expected_model_version_id", "argus_model_sha256",
        "argus_report_model_version_id", "argus_model_identity_status",
        "argus_annotation_sha256", "argus_review_sha256", "argus_reviewed_by",
        "argus_evaluation_summary_json", "argus_event_metrics_status",
        "argus_scene_comparison_sha256", "argus_scene_comparison_path",
        "argus_scene_comparison_status", "argus_scene_comparison_profile_id",
        "argus_scene_comparison_summary_json",
    )
    for field in sample_strings:
        dataset.add_sample_field(field, fo.StringField)
    dataset.add_sample_field("argus_reviewed", fo.BooleanField)
    dataset.add_sample_field("argus_has_evaluation", fo.BooleanField)
    dataset.add_sample_field("argus_has_scene_comparison", fo.BooleanField)
    for field in ("argus_frame_index", "argus_pts", "argus_time_base_num", "argus_time_base_den", "argus_eval_label"):
        dataset.add_frame_field(field, fo.IntField)
    for field in (
        "argus_timestamp_sec", "argus_relative_timestamp_sec",
        "argus_eval_timestamp_sec", "argus_score",
    ):
        dataset.add_frame_field(field, fo.FloatField)
    for field in (
        "argus_ignored", "argus_sampled", "argus_predicted", "argus_score_eligible",
        "argus_scene_guard_usable", "argus_blocked", "argus_truth",
    ):
        dataset.add_frame_field(field, fo.BooleanField)
    for field in (
        "argus_review_state", "argus_scene_guard_state", "argus_scene_guard_reason",
        "argus_scene_profile_id", "argus_error", "argus_decision", "argus_eval_decision",
        "argus_exclusion_reason", "argus_pipeline_mode", "argus_pts_exact", "argus_time_base_exact",
    ):
        dataset.add_frame_field(field, fo.StringField)
    for field in ("argus_positive_labels", "argus_event_ids", "argus_ignore_labels"):
        dataset.add_frame_field(field, fo.ListField, subfield=fo.StringField)
    for field in (*_SCENE_FRAME_STRINGS, *_SCENE_HUMAN_FIELDS):
        dataset.add_frame_field(field, fo.StringField)
    for field in _SCENE_FRAME_BOOLEANS:
        dataset.add_frame_field(field, fo.BooleanField)
    for field in _SCENE_FRAME_FLOATS:
        dataset.add_frame_field(field, fo.FloatField)
    dataset.add_frame_field("ground_truth", fo.EmbeddedDocumentField, embedded_doc_type=fo.Classification)
    for field in ("ground_truth_boxes", "argus_proposed_boxes"):
        dataset.add_frame_field(field, fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections)
    dataset.create_index("argus_clip_id", unique=True)


def _sample_fields(
    bundle: ReviewBundle, clip: ReviewClip, generation: str,
    comparison: SceneComparisonReview | None = None,
) -> dict[str, Any]:
    report = clip.evaluation or {}
    identity = report.get("model_identity") or {}
    actual = identity.get("actual") or {}
    reference = report.get("scene_reference") or {}
    # Keep only report summaries, not the unbounded candidate/frame trace in a
    # single MongoDB sample document. Frame observations are imported separately.
    summary = {key: report.get(key) for key in (
        "schema_version", "evaluation_mode", "source_video_sha256", "annotation_sha256",
        "model_identity", "scene_reference", "release_contract", "metrics", "sample_every",
        "warmup_seconds", "processed_frames", "scored_frames", "source_read_error",
        "source_exhausted", "truncated_by_max_frames",
    )} if clip.evaluation is not None else None
    return {
        "argus_owner": _OWNER,
        "argus_clip_id": clip.clip_id,
        "argus_suite_id": bundle.suite_id,
        "argus_bundle_id": bundle.bundle_id,
        "argus_manifest_sha256": bundle.manifest_sha256,
        "argus_video_sha256": clip.video_sha256,
        "argus_media_identity": _media_identity(clip),
        "argus_import_state": "pending",
        "argus_import_generation": generation,
        "argus_clip_fingerprint": _clip_fingerprint(clip, comparison),
        "argus_camera_id": clip.camera_id,
        "argus_source_session_id": clip.source_session_id,
        "argus_split": clip.split,
        "argus_kind": clip.kind,
        "argus_scene_profile_id": reference.get("profile_id"),
        "argus_expected_scene_profile_id": clip.scene_profile_id,
        # A requested/declared ID is not evidence of what the runtime loaded.
        # Preserve it separately when older reports lack the actual identity.
        "argus_model_version_id": actual.get("version_id"),
        "argus_report_model_version_id": report.get("model_version_id"),
        "argus_expected_model_version_id": clip.model_version_id,
        "argus_model_sha256": actual.get("artifact_sha256"),
        "argus_model_identity_status": (
            "not_evaluated" if clip.evaluation is None else
            "reported" if actual.get("version_id") and actual.get("artifact_sha256") else
            "unavailable"
        ),
        "argus_annotation_sha256": clip.annotation_sha256,
        "argus_review_sha256": _digest(clip.annotations.model_dump(mode="json")),
        "argus_reviewed": clip.annotations.reviewed,
        "argus_reviewed_by": clip.annotations.reviewed_by,
        "argus_has_evaluation": clip.evaluation is not None,
        "argus_evaluation_summary_json": _json(summary) if summary is not None else None,
        "argus_event_metrics_status": "source_report_only_rerun_required" if summary else "not_evaluated",
        "argus_has_scene_comparison": comparison is not None,
        "argus_scene_comparison_sha256": comparison.report_sha256 if comparison else None,
        "argus_scene_comparison_path": str(comparison.report_path) if comparison else None,
        "argus_scene_comparison_status": comparison.report_status if comparison else None,
        "argus_scene_comparison_profile_id": comparison.profile_id if comparison else None,
        "argus_scene_comparison_summary_json": _json(comparison.summary) if comparison else None,
    }


def _write_clip(
    dataset: Any, fo: Any, bundle: ReviewBundle, clip: ReviewClip,
    generation: str, existing: Any | None, comparison: SceneComparisonReview | None = None,
) -> Any:
    fields = _sample_fields(bundle, clip, generation, comparison)
    sample = existing
    if sample is not None:
        # Hide this sample from committed views before changing any frame. A
        # failed/partial save remains visibly pending; a retry rebuilds it fully.
        sample["argus_import_state"] = "pending"
        sample.save()
        sample.filepath = str(Path(clip.video_path).resolve())
    else:
        sample = fo.Sample(filepath=str(Path(clip.video_path).resolve()))
    for key, value in fields.items():
        sample[key] = value
    sample.metadata = fo.VideoMetadata(
        frame_width=clip.width, frame_height=clip.height, frame_rate=clip.fps,
        total_frame_count=clip.frame_count,
    )
    # FiftyOne retains new Frames on an unattached Sample until add_sample().
    # Add the pending sample first, then flush bounded batches through the
    # public Frames API rather than accumulating an entire long video in RAM.
    if existing is None:
        dataset.add_sample(sample)
    else:
        sample.save()
    records, failures = _evaluation_rows(clip)
    scene_rows = {row["frame_index"]: row for row in comparison.frame_rows} if comparison else {}
    boxes: dict[int, list] = {}
    for box in clip.annotations.boxes:
        boxes.setdefault(box.frame_index, []).append(box)
    for index in range(clip.frame_count):
        frame = sample.frames[index + 1]
        values = _frame_fields(
            clip, index, records.get(index), failures.get(index), boxes.get(index, []),
        )
        for key, value in values.items():
            frame[key] = value
        for key, value in _scene_frame_fields(comparison, scene_rows.get(index)).items():
            frame[key] = value
        # Never assign human fields, including a seemingly harmless None.
        # Frames are cached singletons in FiftyOne. A cached missing value can
        # predate a review saved by another client; marking that field dirty
        # would erase the new human value when this batch is flushed.
        truth = values["argus_truth"]
        frame["ground_truth"] = (
            fo.Classification(label="anomaly" if truth else "normal") if truth is not None else None
        )
        detections = [fo.Detection(
            label=box.label,
            bounding_box=[
                box.xtl / clip.width, box.ytl / clip.height,
                (box.xbr - box.xtl) / clip.width, (box.ybr - box.ytl) / clip.height,
            ],
            index=box.track_id,
        ) for box in boxes.get(index, [])]
        reviewed_boxes = clip.annotations.reviewed and not values["argus_ignored"]
        frame["ground_truth_boxes"] = fo.Detections(detections=detections) if reviewed_boxes else None
        frame["argus_proposed_boxes"] = fo.Detections(detections=detections) if not reviewed_boxes else None
        if (index + 1) % _FRAME_WRITE_BATCH_SIZE == 0 or index + 1 == clip.frame_count:
            sample.frames.save()
            # Drop the last strong Frame reference before clearing the frame
            # cache; persisted unrelated user fields survive subsequent reads.
            del frame
            sample.frames.reload()
    # Completion is published only after all media hashes have been rechecked.
    return sample


def _save_views(dataset: Any, fo: Any, generation: str) -> None:
    current = dataset.match({
        "argus_import_state": "complete", "argus_import_generation": generation,
    })
    dataset.save_view("argus-current", current, overwrite=True)
    for name, field, value in (
        ("argus-fp", "argus_decision", "fp"),
        ("argus-fn", "argus_decision", "fn"),
        ("argus-blocked", "argus_blocked", True),
        ("argus-errors", "argus_decision", "error"),
        ("argus-unreviewed", "argus_review_state", "unreviewed"),
        ("argus-unsampled", "argus_sampled", False),
        ("argus-scene-disagreements", "argus_scene_disagreement", True),
        ("argus-scene-raw-disagreements", "argus_scene_raw_disagreement", True),
        ("argus-scene-errors", "argus_scene_comparison_error", True),
    ):
        # MatchFrames places raw Mongo expressions inside $filter. A bare
        # "$field" would refer to the video sample, not the current frame.
        # ViewField applies FiftyOne's correct frame-variable prefix.
        dataset.save_view(name, current.match_frames(fo.ViewField(field) == value), overwrite=True)

    def text_or_empty(name: str) -> Any:
        field = fo.ViewField(name)
        # Mongo distinguishes a missing field from an explicit null. Normalize
        # both via public ViewExpression methods, not Python truth coercion.
        return field.exists().if_else(field, "")

    reviewed_hash = text_or_empty("argus_scene_reviewed_report_sha256")
    report_hash = text_or_empty("argus_scene_report_sha256")
    same_report = reviewed_hash == report_hash
    reviewed_current_report = (
        ~(report_hash == "") & same_report
        & text_or_empty("argus_scene_human_state").is_in(list(_SCENE_HUMAN_STATES))
    )
    # Keep this expression live: a reviewer can edit both fields in FiftyOne
    # without reimporting. Old notes remain visible after a report change but
    # cannot silently approve the new observations.
    has_comparison = fo.ViewField("argus_scene_has_comparison") == True  # noqa: E712
    unreviewed = has_comparison & ~reviewed_current_report
    dataset.save_view("argus-scene-unreviewed", current.match_frames(unreviewed), overwrite=True)
    dataset.save_view(
        "argus-scene-reviewed", current.match_frames(has_comparison & reviewed_current_report),
        overwrite=True,
    )
    stale = has_comparison & ~(reviewed_hash == "") & ~same_report
    dataset.save_view("argus-scene-stale-review", current.match_frames(stale), overwrite=True)


def _validate_split_history(existing: dict[str, Any], bundle: ReviewBundle) -> None:
    """Keep capture sessions isolated even across incremental bundle imports."""
    sources: dict[str, tuple[str, str | None, str]] = {}
    for clip_id, sample in existing.items():
        split = sample.get_field("argus_split")
        if split not in {"unassigned", "train", "validation", "test"}:
            raise ValueError("existing Argus sample has an invalid dataset split")
        session = sample.get_field("argus_source_session_id")
        if split != "unassigned" and not session:
            raise ValueError("existing assigned sample has no capture-session identity")
        sources[clip_id] = (sample.get_field("argus_video_sha256"), session, split)
    for clip in bundle.clips:
        sample = existing.get(clip.clip_id)
        if sample is not None:
            old_split = sample.get_field("argus_split")
            if old_split != "unassigned" and old_split != clip.split:
                raise ValueError("an assigned clip split is immutable; use a new dataset")
            old_session = sample.get_field("argus_source_session_id")
            if old_session and old_session != clip.source_session_id:
                raise ValueError("a clip capture-session identity is immutable")
            if sample.get_field("argus_camera_id") != clip.camera_id:
                raise ValueError("a clip camera identity is immutable")
        sources[clip.clip_id] = (clip.video_sha256, clip.source_session_id, clip.split)
    assignments: dict[tuple[str, str], set[str]] = {}
    for video_hash, session, split in sources.values():
        keys = [("video", video_hash)]
        if session:
            keys.append(("session", session))
        for key in keys:
            assignments.setdefault(key, set()).add(split)
    if any(len(splits) > 1 for splits in assignments.values()):
        raise ValueError(
            "historical video/capture session crosses dataset splits; update all "
            "unassigned members together or use a new dataset"
        )


def import_bundle_to_fiftyone(
    bundle: ReviewBundle, *, dataset_name: str, persistent: bool = True,
    scene_comparisons: dict[str, SceneComparisonReview] | None = None,
) -> Any:
    """Upsert a verified bundle without starting the App or model inference.

    Existing datasets require this importer's ownership marker and the same
    suite. A clip's bytes and exact frame mapping are immutable in a dataset.
    Retrying a partial import rebuilds pending clips; complete identical clips
    are reused. ``argus-current`` excludes removed/stale generation samples.
    Historical samples are retained rather than deleting users' analysis work.
    Optional scene comparisons enrich only independent diagnostic fields. Human
    review requires an explicit state and hash binding; it never changes GT,
    evaluation decisions or production scene-profile approval.
    """
    if not dataset_name or dataset_name != dataset_name.strip():
        raise ValueError("dataset_name must be nonempty without surrounding whitespace")
    if scene_comparisons is not None and not isinstance(scene_comparisons, dict):
        raise ValueError("scene_comparisons must be a clip-ID mapping")
    # Revalidate nested mutable lists/dicts as well as every source file before
    # importing FiftyOne (which may start its database) or mutating a dataset.
    bundle = ReviewBundle.model_validate(bundle.model_dump(mode="json"))
    comparisons = dict(scene_comparisons) if scene_comparisons is not None else {}
    require_verified_media(bundle)
    require_verified_scene_comparisons(comparisons, bundle=bundle)
    _validate_split_history({}, bundle)
    for clip in bundle.clips:
        if clip.clip_id in comparisons:
            _require_scene_player_grid(clip)
        _evaluation_rows(clip)
        _sample_fields(bundle, clip, "preflight", comparisons.get(clip.clip_id))
    fo = importlib.import_module("fiftyone")
    generation_payload = bundle.model_dump(mode="json")
    if comparisons:
        generation_payload = {
            "bundle": generation_payload,
            "scene_comparisons": {key: value.report_sha256 for key, value in comparisons.items()},
        }
    generation = _digest(generation_payload)
    host = socket.gethostname()
    with _dataset_import_lock(dataset_name):
        if fo.dataset_exists(dataset_name):
            dataset = fo.load_dataset(dataset_name)
            info = dataset.info.get(_INFO_KEY)
            if not isinstance(info, dict) or info.get("owner") != _OWNER:
                raise ValueError("refusing to overwrite a dataset not owned by Argus review")
            if info.get("suite_id") != bundle.suite_id:
                raise ValueError("dataset belongs to a different Argus review suite")
            importing_host = info.get("importing_host") or info.get("host")
            if importing_host != host:
                raise RuntimeError("dataset is bound to an importer on another host")
        else:
            dataset = fo.Dataset(name=dataset_name, persistent=persistent)
            info = {
                "owner": _OWNER, "suite_id": bundle.suite_id,
                "state": "failed", "host": host, "importing_host": host,
                "error": "initial import not completed",
            }
            # Bind ownership immediately so any later preflight/schema failure
            # remains retryable, rather than leaving an unowned empty dataset.
            dataset.info[_INFO_KEY] = info
            dataset.save()
        # Preflight every existing sample before writing any of this bundle.
        existing: dict[str, Any] = {}
        for sample in dataset.iter_samples():
            if sample.get_field("argus_owner") != _OWNER:
                raise ValueError("dataset contains a sample not owned by Argus review")
            clip_id = sample.get_field("argus_clip_id")
            if clip_id in existing:
                raise ValueError("dataset contains duplicate Argus clip identities")
            existing[clip_id] = sample
        for clip in bundle.clips:
            sample = existing.get(clip.clip_id)
            if sample is not None and sample.get_field("argus_media_identity") != _media_identity(clip):
                raise ValueError(f"clip media changed; use a new dataset: {clip.clip_id}")
        _validate_split_history(existing, bundle)
        dataset.info[_INFO_KEY] = {
            **info, "state": "importing", "host": host, "bundle_id": bundle.bundle_id,
            "importing_host": host,
            "pending_generation": generation, "error": None,
        }
        dataset.save()
        imported = []
        try:
            dataset.media_type = "video"
            _ensure_schema(dataset, fo)
            for clip in bundle.clips:
                sample = existing.get(clip.clip_id)
                comparison = comparisons.get(clip.clip_id)
                if sample is not None and (
                    sample.get_field("argus_import_state") == "complete"
                    and sample.get_field("argus_import_generation") == generation
                    and sample.get_field("argus_clip_fingerprint")
                    == _clip_fingerprint(clip, comparison)
                ):
                    imported.append(sample)
                    continue
                imported.append(_write_clip(
                    dataset, fo, bundle, clip, generation, sample, comparison,
                ))
            # Long imports must not bless bytes changed while MongoDB frames
            # were written. Pending clips remain excluded if this check fails.
            require_verified_media(bundle)
            require_verified_scene_comparisons(comparisons, bundle=bundle)
            for sample in imported:
                if sample.get_field("argus_import_state") != "complete":
                    sample["argus_import_state"] = "complete"
                    sample.save()
            _save_views(dataset, fo, generation)
            dataset.info[_INFO_KEY] = {
                **dataset.info[_INFO_KEY], "state": "complete", "generation": generation,
                "pending_generation": None, "clip_count": len(bundle.clips),
                "event_metrics_authority": "argus_pipeline_evaluator_only",
                "scene_comparison_authority": "diagnostic_human_review_only",
            }
            dataset.save()
        except Exception as exc:
            # A saved view may already point at this generation when a later
            # view or dataset metadata write fails. Quarantine every touched
            # sample so no committed view can present a partial refresh as
            # complete. This also covers reused samples if media changed.
            for sample in imported:
                try:
                    sample["argus_import_state"] = "pending"
                    sample.save()
                except Exception:
                    pass  # Database failure may prevent compensation too.
            dataset.info[_INFO_KEY] = {
                **dataset.info[_INFO_KEY], "state": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            try:
                dataset.save()
            except Exception:
                pass  # Preserve the original failure; persisted pending clips remain recoverable.
            raise
    return dataset


__all__ = ["import_bundle_to_fiftyone"]
