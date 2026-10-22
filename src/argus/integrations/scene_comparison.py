"""Read-only bridge from offline scene comparison reports to human review.

Only report bytes and the existing review contract are read here. Importing
this module does not import a decoder, SceneGuard, Torch, LightGlue or FiftyOne.
Comparison states are diagnostic proposals, never annotations or gate authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping

from argus.integrations.contracts import ReviewBundle, ReviewClip, file_sha256


_SCHEMA = "argus.scene-guard-comparison.v1"
_DECODER_SCHEMA = "argus.strict-video-decoder-evidence.v1"
_STATES = {"calibrating", "stable", "alignable", "moved", "unknown", "blocked"}
_MAX_REPORT_BYTES = 512 * 1024 * 1024
_MAX_FRAME_JSON_BYTES = 128 * 1024
_MAX_SUMMARY_JSON_BYTES = 64 * 1024


@dataclass(frozen=True)
class SceneComparisonReview:
    """Validated report snapshot, bound to one original clip and timestamp map.

    Nested dictionaries intentionally remain ordinary JSON values for the
    importer. Verification rebuilds them from hash-bound report bytes, so edits
    to these mutable containers cannot bypass the final commit check.
    """

    clip_id: str
    report_path: Path
    report_sha256: str
    report_status: str
    profile_id: str
    frame_rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    _clip_snapshot_json: str = field(repr=False)


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("scene comparison must contain finite UTF-8 JSON values") from exc


def _review_snapshot(review: SceneComparisonReview) -> bytes:
    """Compare JSON types as well as values, not Python's numeric equality.

    The public nested dictionaries remain mutable for the importer. Python
    considers ``False == 0`` and ``1 == 1.0``, which makes dataclass/dict
    equality insufficient to prove that these hash-bound diagnostics still
    contain the validated JSON snapshot.
    """
    return _json_bytes({
        "clip_id": review.clip_id,
        "report_path": str(review.report_path),
        "report_sha256": review.report_sha256,
        "report_status": review.report_status,
        "profile_id": review.profile_id,
        "frame_rows": review.frame_rows,
        "summary": review.summary,
        "clip_snapshot_json": review._clip_snapshot_json,
    })


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 512:
        raise ValueError(f"{label} must be a clean nonempty identity")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_number(value: Any, label: str) -> float:
    if type(value) not in (float, int):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _time_base(value: Any, label: str) -> Fraction:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an exact positive time-base string")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{label} must be an exact positive time-base string") from exc
    if result <= 0 or value != f"{result.numerator}/{result.denominator}":
        raise ValueError(f"{label} must be a canonical positive time-base fraction")
    return result


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in scene comparison: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant in scene comparison: {value}")


def _parse_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite JSON number in scene comparison")
    return result


def _read_report(path: str | Path) -> tuple[Path, str, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size > _MAX_REPORT_BYTES:
        raise ValueError("scene comparison must be a regular report file within the size limit")
    before = file_sha256(resolved)
    raw = resolved.read_bytes()
    if len(raw) > _MAX_REPORT_BYTES:
        raise ValueError("scene comparison report exceeds the size limit")
    loaded_hash = hashlib.sha256(raw).hexdigest()
    if before != loaded_hash or file_sha256(resolved) != loaded_hash:
        raise ValueError("scene comparison report changed while loading")
    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant, parse_float=_parse_float,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("scene comparison report is not valid UTF-8 JSON") from exc
    return resolved, loaded_hash, _object(payload, "scene comparison")


def _source_identity(payload: dict[str, Any]) -> tuple[str, str]:
    reference = _object(payload.get("reference"), "comparison reference")
    inputs = _object(payload.get("inputs"), "comparison inputs")
    video = _object(inputs.get("video"), "comparison video identity")
    return (
        _identity(reference.get("camera_id"), "comparison camera_id"),
        _sha256(video.get("sha256"), "comparison video SHA256"),
    )


def _decoder_identity(value: Any, clip: ReviewClip, label: str) -> None:
    evidence = _object(value, f"{label} strict decoder evidence")
    if (
        evidence.get("schema_version") != _DECODER_SCHEMA
        or evidence.get("backend") != "ffprobe+ffmpeg-rawvideo"
    ):
        raise ValueError(f"{label} must provide the strict FFmpeg decoder schema")
    for key in (
        "strict_release_decoder", "source_sha256_matches", "pts_strictly_increasing",
        "container_declared_frame_count_matches", "decode_error_distinguishable_from_clean_eof",
        "variable_frame_rate_timestamps_preserved", "complete_decode_verified", "source_exhausted",
    ):
        if evidence.get(key) is not True:
            raise ValueError(f"{label} decoder did not prove {key}")
    if evidence.get("truncated") is not False:
        raise ValueError(f"{label} strict source decoding must reach clean EOF, not a prefix")
    for key in ("source_sha256_before", "source_sha256_after"):
        if evidence.get(key) != clip.video_sha256:
            raise ValueError(f"{label} decoder source identity does not match review video")
    for key, expected in (
        ("width", clip.width), ("height", clip.height),
        ("timestamp_frame_count", clip.frame_count),
    ):
        if _integer(evidence.get(key), f"{label}.{key}", minimum=1) != expected:
            raise ValueError(f"{label} decoder dimensions/frame count disagree with review video")
    _integer(evidence.get("stream_index"), f"{label}.stream_index")
    if evidence.get("pixel_format") != "bgr24":
        raise ValueError(f"{label} decoder pixel format must be bgr24")
    declared_count = evidence.get("container_declared_frame_count")
    if declared_count is not None and _integer(
        declared_count, f"{label}.container_declared_frame_count",
    ) != clip.frame_count:
        raise ValueError(f"{label} declared source frame count disagrees with review video")
    time_base = _time_base(evidence.get("time_base"), f"{label}.time_base")
    if any(
        Fraction(stamp.time_base_num, stamp.time_base_den) != time_base
        for stamp in clip.timestamps
    ):
        raise ValueError(f"{label} decoder time base disagrees with review video")
    rate = _finite_number(evidence.get("nominal_frame_rate"), f"{label}.nominal_frame_rate")
    if rate <= 0 or not math.isclose(rate, clip.fps, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{label} decoder frame rate disagrees with review video")
    probe = _object(evidence.get("ffprobe"), f"{label}.ffprobe")
    decoder = _object(evidence.get("ffmpeg"), f"{label}.ffmpeg")
    for name, process, count_field in (
        ("ffprobe", probe, "decoded_frame_inventory_count"),
        ("ffmpeg", decoder, "decoded_frame_count"),
    ):
        if (
            type(process.get("returncode")) is not int or process["returncode"] != 0
            or process.get("stderr_empty") is not True
        ):
            raise ValueError(f"{label}.{name} did not exit without decoder errors")
        if _integer(
            process.get(count_field), f"{label}.{name}.{count_field}", minimum=1,
        ) != clip.frame_count:
            raise ValueError(f"{label}.{name} decoded frame count disagrees with review video")
    for key in ("decoded_frame_count_matches_ffprobe", "clean_eof", "xerror_enabled"):
        if decoder.get(key) is not True:
            raise ValueError(f"{label}.ffmpeg did not prove {key}")
    if decoder.get("partial_raw_frame_detected") is not False:
        raise ValueError(f"{label}.ffmpeg observed an incomplete raw frame")


def _decision(value: Any, label: str) -> dict[str, Any]:
    decision = _object(value, label)
    status = decision.get("status")
    if not isinstance(status, str) or status not in {"evaluated", "error"}:
        raise ValueError(f"{label} status must be evaluated or error")
    for key in ("state", "raw_state"):
        if not isinstance(decision.get(key), str) or decision[key] not in _STATES:
            raise ValueError(f"{label}.{key} is not a recognized SceneGuard diagnostic state")
    reason = decision.get("reason")
    if not isinstance(reason, str) or len(reason) > 8192:
        raise ValueError(f"{label}.reason must be bounded text")
    elapsed = decision.get("elapsed_ms")
    if elapsed is not None and _finite_number(elapsed, f"{label}.elapsed_ms") < 0:
        raise ValueError(f"{label}.elapsed_ms cannot be negative")
    if status == "error":
        if decision["state"] != "unknown" or decision["raw_state"] != "unknown":
            raise ValueError(f"{label} errors cannot claim a successful scene state")
        if decision.get("metrics") is not None:
            raise ValueError(f"{label} errors cannot claim measured geometry")
    else:
        _object(decision.get("metrics"), f"{label}.metrics")
        if elapsed is None:
            raise ValueError(f"{label} evaluated geometry requires elapsed_ms")
    if "matcher" in decision:
        matcher = _object(decision["matcher"], f"{label}.matcher")
        if "error_code" in matcher and status != "error":
            raise ValueError(f"{label} matcher failures cannot claim evaluated geometry")
    expected_stable = status == "evaluated" and decision["state"] == "stable"
    if decision.get("matches_stable_pose") is not expected_stable:
        raise ValueError(f"{label} stable-pose diagnostic contradicts its state")
    return decision


def _bounded_errors(value: Any) -> tuple[list[dict[str, str]], int, bool]:
    if not isinstance(value, list):
        raise ValueError("comparison errors must be a list")
    errors: list[dict[str, str]] = []
    truncated = len(value) > 16
    for item in value:
        entry = _object(item, "comparison error")
        if not isinstance(entry.get("stage"), str) or not isinstance(entry.get("error"), str):
            raise ValueError("comparison errors require stage and error text")
        if len(errors) < 16:
            stage, reason = entry["stage"], entry["error"]
            truncated |= len(stage) > 128 or len(reason) > 1024
            errors.append({"stage": stage[:128], "error": reason[:1024]})
    return errors, len(value), truncated


def _build_review(
    clip: ReviewClip, path: Path, digest: str, payload: dict[str, Any],
) -> SceneComparisonReview:
    if (
        payload.get("schema_version") != _SCHEMA
        or payload.get("diagnostic_only") is not True
        or payload.get("gating_effect") != "none"
        or payload.get("release_authority") is not False
        or payload.get("source_inputs_stable") is not True
        or payload.get("production_pipeline_parity_claimed") is not False
        or payload.get("baseline_scope") != "standalone_scene_guard_full_check"
    ):
        raise ValueError("expected a source-stable, diagnostic-only standalone scene comparison")
    if _source_identity(payload) != (clip.camera_id, clip.video_sha256):
        raise ValueError("scene comparison is not bound to this review clip")
    status = payload.get("status")
    if not isinstance(status, str) or status not in {"complete", "partial", "error"}:
        raise ValueError("comparison status must be complete, partial or error")
    reference = _object(payload["reference"], "comparison reference")
    profile_id = _identity(reference.get("profile_id"), "comparison profile_id")
    if clip.scene_profile_id is not None and clip.scene_profile_id != profile_id:
        raise ValueError(
            "comparison profile differs from the review clip's explicit profile binding"
        )
    if (
        clip.model_version_id is not None
        and clip.model_version_id != reference.get("model_version_id_declared")
    ):
        raise ValueError(
            "comparison model declaration differs from the review clip's explicit model binding"
        )
    if (
        reference.get("operator_approved") is not True
        or reference.get("runtime_model_binding_verified") is not False
    ):
        raise ValueError(
            "comparison reference must be approved but must not claim runtime authorization"
        )
    _sha256(reference.get("reference_sha256"), "comparison reference SHA256")
    inputs = _object(payload["inputs"], "comparison inputs")
    compact_inputs: dict[str, dict[str, Any]] = {}
    for name in ("video", "config", "extractor_weights", "matcher_weights"):
        identity = _object(inputs.get(name), f"comparison {name} identity")
        source_path = identity.get("path")
        if not isinstance(source_path, str) or not source_path or len(source_path) > 4096:
            raise ValueError(f"comparison {name} declared path must be bounded text")
        compact_inputs[name] = {
            "path": source_path,
            "sha256": _sha256(identity.get("sha256"), f"comparison {name} SHA256"),
        }
    effective_hash = _sha256(
        payload.get("effective_config_sha256"), "comparison effective configuration SHA256",
    )
    evidence = _object(payload.get("decoder_evidence"), "comparison decoder_evidence")
    _decoder_identity(evidence.get("baseline"), clip, "baseline")
    if "lightglue" in evidence:
        _decoder_identity(evidence["lightglue"], clip, "lightglue")
    elif status == "complete":
        raise ValueError("a complete comparison requires both source passes to reach verified EOF")
    sample_every = _integer(payload.get("sample_every"), "comparison sample_every", minimum=1)
    max_frames = payload.get("max_frames")
    if max_frames is not None:
        _integer(max_frames, "comparison max_frames", minimum=1)
    compared_limit = (
        min(clip.frame_count, max_frames) if max_frames is not None else clip.frame_count
    )
    truncated = compared_limit < clip.frame_count
    if payload.get("comparison_truncated") is not truncated:
        raise ValueError(
            "comparison truncation flag disagrees with max_frames and original frame count"
        )
    if truncated and status == "complete":
        raise ValueError("a prefix comparison cannot claim complete comparison status")
    source_rows = payload.get("frames")
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("comparison requires at least one original sampled frame")
    expected_indices = range(0, compared_limit, sample_every)
    if len(source_rows) != len(expected_indices):
        raise ValueError("comparison frames must cover the exact original sampling grid")
    first = clip.timestamps[0]
    first_time = first.pts * Fraction(first.time_base_num, first.time_base_den)
    rows: list[dict[str, Any]] = []
    for expected_index, value in zip(expected_indices, source_rows):
        row = _object(value, "comparison frame")
        index = _integer(row.get("frame_index"), "comparison frame_index")
        if index != expected_index:
            raise ValueError(
                "comparison frame indices must be unique, ordered and on the original sampling grid"
            )
        stamp = clip.timestamps[index]
        if type(row.get("pts")) is not int or row["pts"] != stamp.pts:
            raise ValueError("comparison frame PTS disagrees with original review timestamp")
        time_base = _time_base(row.get("time_base"), "comparison frame time_base")
        if time_base != Fraction(stamp.time_base_num, stamp.time_base_den):
            raise ValueError("comparison frame time base disagrees with original review timestamp")
        exact_time = stamp.pts * time_base
        if float(stamp.timestamp_sec) != float(exact_time):
            raise ValueError("review timestamp must exactly match the comparison PTS/time base")
        for name, expected in (
            ("timestamp_sec", float(exact_time)),
            ("relative_timestamp_sec", float(exact_time - first_time)),
        ):
            actual = _finite_number(row.get(name), f"comparison frame {name}")
            if actual != expected:
                raise ValueError(f"comparison frame {name} disagrees with exact PTS/time base")
        _sha256(row.get("frame_sha256"), "comparison decoded frame SHA256")
        baseline = _decision(row.get("baseline"), "baseline frame")
        learned = _decision(row.get("lightglue"), "lightglue frame")
        if len(_json_bytes(row)) > _MAX_FRAME_JSON_BYTES:
            raise ValueError("comparison frame diagnostics exceed the bounded review payload size")
        errors = [
            name for name, decision in (("baseline", baseline), ("lightglue", learned))
            if decision["status"] == "error"
        ]
        comparable = not errors
        state_disagreement = baseline["state"] != learned["state"] if comparable else None
        raw_disagreement = baseline["raw_state"] != learned["raw_state"] if comparable else None
        # Copy all original diagnostics but replace every derived review flag;
        # neither the report's summary nor user-supplied flags are authoritative.
        rows.append({
            **row, "comparable": comparable,
            "state_disagreement": state_disagreement,
            "raw_state_disagreement": raw_disagreement,
            "comparison_error": bool(errors), "error_algorithms": errors,
            "review_category": (
                "error" if errors else "confirmed_disagreement" if state_disagreement
                else "raw_disagreement" if raw_disagreement else "agreement"
            ),
        })
    report_errors, error_count, errors_truncated = _bounded_errors(payload.get("errors"))
    comparable_rows = [row for row in rows if row["comparable"]]
    error_rows = sum(row["comparison_error"] for row in rows)
    if status == "complete" and (error_count or error_rows):
        raise ValueError("a complete comparison cannot contain failed frame observations")
    pair_counts = Counter(
        f"{row['baseline']['state']}|{row['lightglue']['state']}" for row in comparable_rows
    )
    # Keep sample documents bounded: per-frame diagnostics live on Frames;
    # untrusted original summary/frame-index arrays are deliberately omitted.
    summary: dict[str, Any] = {
        "schema_version": _SCHEMA, "report_status": status,
        "diagnostic_only": True, "gating_effect": "none", "release_authority": False,
        "baseline_scope": "standalone_scene_guard_full_check",
        "production_pipeline_parity_claimed": False,
        "pose_ground_truth_available": False, "accuracy_improvement_established": False,
        "source_inputs_stable": True, "source_verified_against_bundle": True,
        "report_summary_ignored": True,
        "sampled_frames": len(rows), "comparable_pairs": len(comparable_rows),
        "state_disagreement_count": sum(row["state_disagreement"] is True for row in rows),
        "raw_state_disagreement_count": sum(row["raw_state_disagreement"] is True for row in rows),
        "error_frames": error_rows,
        "baseline_errors": sum("baseline" in row["error_algorithms"] for row in rows),
        "lightglue_errors": sum("lightglue" in row["error_algorithms"] for row in rows),
        "state_pair_counts": dict(sorted(pair_counts.items())),
        "sample_every": sample_every, "max_frames": max_frames,
        "comparison_truncated": truncated,
        "full_lightglue_decode_verified": "lightglue" in evidence,
        "report_errors": report_errors, "report_error_count": error_count,
        "report_errors_truncated": errors_truncated,
        "reference": {
            key: reference.get(key) for key in (
                "camera_id", "profile_id", "active_profile_id", "selected_profile_is_active",
                "reference_sha256", "operator_approved", "model_version_id_declared",
                "model_artifact_sha256_declared", "runtime_model_binding_verified",
            )
        },
        "inputs": compact_inputs, "effective_config_sha256": effective_hash,
        "algorithm_thresholds": payload.get("algorithm_thresholds"),
        "lightglue_options": payload.get("lightglue_options"),
        "metric_semantics": payload.get("metric_semantics"),
    }
    if len(_json_bytes(summary)) > _MAX_SUMMARY_JSON_BYTES:
        raise ValueError("comparison summary diagnostics exceed the bounded review payload size")
    return SceneComparisonReview(
        clip_id=clip.clip_id, report_path=path, report_sha256=digest,
        report_status=status, profile_id=profile_id, frame_rows=tuple(rows), summary=summary,
        _clip_snapshot_json=clip.model_dump_json(),
    )


def load_scene_comparisons(
    bundle: ReviewBundle, report_paths: Iterable[str | Path],
) -> dict[str, SceneComparisonReview]:
    """Bind reports by unique camera and content SHA, never guessed file names.

    Declared original video paths may be relocated; the bundle's verified media
    hash is the identity. Weights, profiles and the original config are not
    needed on a review workstation. Their declarations remain diagnostics.
    """
    validated = ReviewBundle.model_validate(bundle.model_dump(mode="json"))
    clips_by_identity: dict[tuple[str, str], list[ReviewClip]] = {}
    for clip in validated.clips:
        clips_by_identity.setdefault((clip.camera_id, clip.video_sha256), []).append(clip)
    reviews: dict[str, SceneComparisonReview] = {}
    for report_path in report_paths:
        path, digest, payload = _read_report(report_path)
        clips = clips_by_identity.get(_source_identity(payload), [])
        if len(clips) != 1:
            raise ValueError(
                "scene comparison must match exactly one review clip by camera and video SHA256"
            )
        clip = clips[0]
        if clip.clip_id in reviews:
            raise ValueError("multiple scene comparison reports cannot target the same review clip")
        reviews[clip.clip_id] = _build_review(clip, path, digest, payload)
    for review in reviews.values():
        if file_sha256(review.report_path) != review.report_sha256:
            raise ValueError("scene comparison report changed while loading review bindings")
    return reviews


def require_verified_scene_comparisons(
    reviews: Mapping[str, SceneComparisonReview], *, bundle: ReviewBundle | None = None,
) -> None:
    """Recheck immutable report bytes, derived fields and optional bundle binding.

    Call immediately before importing FiftyOne and again before publishing a
    completed dataset generation. This is verification only: no report rewrite,
    video decoding, inference, profile approval or production state mutation.
    """
    clips: dict[str, ReviewClip] | None = None
    if bundle is not None:
        validated = ReviewBundle.model_validate(bundle.model_dump(mode="json"))
        clips = {clip.clip_id: clip for clip in validated.clips}
    for clip_id, review in reviews.items():
        if not isinstance(review, SceneComparisonReview) or clip_id != review.clip_id:
            raise ValueError("scene comparison mapping key does not match its bound review clip")
        clip = ReviewClip.model_validate_json(review._clip_snapshot_json)
        if clip.clip_id != clip_id:
            raise ValueError("scene comparison clip snapshot identity was changed")
        if clips is not None:
            current = clips.get(clip_id)
            if current is None or current.model_dump_json() != review._clip_snapshot_json:
                raise ValueError("scene comparison belongs to a different review bundle clip")
            candidates = [
                item for item in clips.values()
                if (item.camera_id, item.video_sha256) == (clip.camera_id, clip.video_sha256)
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "scene comparison source identity is ambiguous in this review bundle"
                )
        path, digest, payload = _read_report(review.report_path)
        if digest != review.report_sha256:
            raise ValueError("scene comparison report changed after preflight")
        rebuilt = _build_review(clip, path, digest, payload)
        if _review_snapshot(review) != _review_snapshot(rebuilt):
            raise ValueError(
                "scene comparison review snapshot or derived diagnostics were modified"
            )


__all__ = [
    "SceneComparisonReview", "load_scene_comparisons", "require_verified_scene_comparisons",
]
