"""Shared builder for a validation-only normal corpus for one camera scene.

CLI and Dashboard workflows call the same fail-closed implementation.  It
never chooses or learns a scene reference from the input videos, never grants
operator approval, and never updates an active scene profile.  Every candidate
frame is checked by ``SceneGuard`` against the explicitly supplied reference.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from argus.preprocessing.scene_guard import (
    SceneGuard,
    SceneGuardConfig,
    SceneGuardResult,
    SceneState,
)
from argus.validation.strict_video_decoder import (
    StrictVideoDecoder,
    write_frame_timestamp_jsonl,
)


_SCHEMA_VERSION = "argus.scene-profile-baseline.v2"
_STRICT_SCHEMA_VERSION = "argus.scene-profile-baseline.v3"
_ANNOTATION_SCHEMA_VERSION = "argus.normal-video-review.v1"
_SAMPLE_RULE = "zero_based_frame_index_mod_sample_every_equals_zero"
_DEFAULT_MIN_TRAINING_IMAGES = 100
_DEFAULT_MIN_CALIBRATION_IMAGES = 20
_DEFAULT_MIN_ACCEPTED_TIME_SPAN_SECONDS = 300.0
_NORMALIZATION_CONTRACT_ID = "argus.detection-pipeline.scene-guard-warp.v1"
_ANNOTATION_ROOT_FIELDS = {
    "schema_version",
    "source_video_sha256",
    "source_session_id",
    "camera_id",
    "review_complete",
    "reviewed_by",
    "reviewed_at",
    "positive_ranges",
    "ignore_ranges",
}
_AMBIGUOUS_ANNOTATION_ROOT_FIELDS = {"anomaly_ranges", "events"}
_AMBIGUOUS_RANGE_FIELDS = {
    "start_sec",
    "end_sec",
    "time_sec",
    "frame",
}
_ANNOTATION_RANGE_FIELDS = {"start_frame", "end_frame", "label", "name"}
_SPLIT_ROLES = ("train", "calibration", "diagnostic_only")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_value(state: object) -> str:
    if isinstance(state, SceneState):
        return state.value
    return str(state)


def _range_contains(frame_index: int, item: dict[str, Any]) -> bool:
    return int(item["start_frame"]) <= frame_index <= int(item["end_frame"])


def _annotation_exclusion(
    frame_index: int,
    annotations: dict[str, Any] | None,
) -> str | None:
    if annotations is None:
        return None
    if any(
        _range_contains(frame_index, item)
        for item in annotations.get("ignore_ranges") or []
    ):
        return "annotation_ignore"
    if any(
        _range_contains(frame_index, item)
        for item in annotations.get("positive_ranges") or []
    ):
        return "annotation_positive"
    return None


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _has_timezone(value: object) -> bool:
    if type(value) is not str or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _strict_frame_ranges(
    raw: dict[str, Any],
    *,
    key: str,
    label: int,
    path: Path,
) -> tuple[list[dict[str, Any]], bool]:
    """Parse inclusive frame ranges without permissive seconds conversion.

    A missing root list is retained as incomplete review evidence so an empty
    annotation can still produce explicitly diagnostic samples.  Once a list
    is present, however, malformed or ambiguous entries fail the build rather
    than being silently clipped or normalized.
    """

    if key not in raw:
        return [], False
    items = raw[key]
    if not isinstance(items, list):
        raise ValueError(f"annotation.{key} must be a list: {path}")
    parsed: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        prefix = f"annotation.{key}[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{prefix} must be an object: {path}")
        ambiguous = sorted(_AMBIGUOUS_RANGE_FIELDS.intersection(item))
        if ambiguous:
            raise ValueError(
                f"{prefix} uses ambiguous seconds/legacy coordinates "
                f"{ambiguous}; only start_frame/end_frame are allowed: {path}"
            )
        unknown = sorted(set(item).difference(_ANNOTATION_RANGE_FIELDS))
        if unknown:
            raise ValueError(f"{prefix} contains unknown fields {unknown}: {path}")
        if "start_frame" not in item or "end_frame" not in item:
            raise ValueError(
                f"{prefix} must include start_frame and end_frame: {path}"
            )
        start = item["start_frame"]
        end = item["end_frame"]
        for field, value in (("start_frame", start), ("end_frame", end)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{prefix}.{field} must be finite: {path}")
            if type(value) is not int:
                raise ValueError(f"{prefix}.{field} must be an integer: {path}")
            if value < 0:
                raise ValueError(f"{prefix}.{field} cannot be negative: {path}")
        if end < start:
            raise ValueError(f"{prefix} end_frame precedes start_frame: {path}")
        parsed.append({
            "start_frame": start,
            "end_frame": end,
            "label": label,
            "name": str(item.get("label") or item.get("name") or ""),
        })
    return parsed, True


def _load_annotations_strict(
    path: Path,
    *,
    source_video_sha256: str,
    camera_id: str,
) -> dict[str, Any]:
    """Load ranges and separately attest whether review evidence is complete.

    Missing review metadata or an identity mismatch cannot authorize training,
    but does not turn diagnostic frame extraction into trusted normal data.
    Structural range defects are rejected because continuing could accidentally
    admit anomalous frames into even a diagnostic corpus.
    """

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"annotation root must be an object: {path}")
    aliases = sorted(_AMBIGUOUS_ANNOTATION_ROOT_FIELDS.intersection(raw))
    if aliases:
        raise ValueError(
            "annotation uses positive-range aliases that are forbidden for "
            f"training evidence {aliases}: {path}"
        )
    unknown = sorted(set(raw).difference(_ANNOTATION_ROOT_FIELDS))
    if unknown:
        raise ValueError(f"annotation contains unknown fields {unknown}: {path}")

    positive, positive_present = _strict_frame_ranges(
        raw,
        key="positive_ranges",
        label=1,
        path=path,
    )
    ignore, ignore_present = _strict_frame_ranges(
        raw,
        key="ignore_ranges",
        label=-1,
        path=path,
    )

    reasons: list[str] = []
    recorded_source_hash = raw.get("source_video_sha256")
    source_session_id = raw.get("source_session_id")
    recorded_camera_id = raw.get("camera_id")
    reviewed_by = raw.get("reviewed_by")
    reviewed_at = raw.get("reviewed_at")
    if raw.get("schema_version") != _ANNOTATION_SCHEMA_VERSION:
        reasons.append("schema_version_missing_or_unsupported")
    if not positive_present:
        reasons.append("positive_ranges_missing")
    if not ignore_present:
        reasons.append("ignore_ranges_missing")
    if not _is_sha256(recorded_source_hash):
        reasons.append("source_video_sha256_missing_or_invalid")
    elif recorded_source_hash != source_video_sha256:
        reasons.append("source_video_sha256_mismatch")
    if type(source_session_id) is not str or not source_session_id.strip():
        reasons.append("source_session_id_missing_or_invalid")
    elif source_session_id != source_session_id.strip():
        reasons.append("source_session_id_has_surrounding_whitespace")
    if type(recorded_camera_id) is not str or not recorded_camera_id.strip():
        reasons.append("camera_id_missing_or_invalid")
    elif recorded_camera_id != camera_id:
        reasons.append("camera_id_mismatch")
    if raw.get("review_complete") is not True:
        reasons.append("review_complete_not_literal_true")
    if type(reviewed_by) is not str or not reviewed_by.strip():
        reasons.append("reviewed_by_missing_or_invalid")
    elif reviewed_by != reviewed_by.strip():
        reasons.append("reviewed_by_has_surrounding_whitespace")
    if not _has_timezone(reviewed_at):
        reasons.append("reviewed_at_missing_invalid_or_without_timezone")

    return {
        "path": str(path),
        "positive_ranges": positive,
        "ignore_ranges": ignore,
        "review_evidence": {
            "complete": not reasons,
            "reasons": reasons,
            "schema_version": raw.get("schema_version"),
            "source_video_sha256": recorded_source_hash,
            "source_video_sha256_matches": (
                _is_sha256(recorded_source_hash)
                and recorded_source_hash == source_video_sha256
            ),
            "source_session_id": source_session_id,
            "camera_id": recorded_camera_id,
            "camera_id_matches": recorded_camera_id == camera_id,
            "review_complete": raw.get("review_complete") is True,
            "reviewed_by": reviewed_by,
            "reviewed_at": reviewed_at,
            "reviewed_at_has_timezone": _has_timezone(reviewed_at),
            "coordinate_system": "zero_based_inclusive_frame_ranges_only",
        },
    }


def _validate_annotation_bounds(
    annotations: dict[str, Any] | None,
    *,
    decoded_frame_count: int,
    path: Path | None,
) -> None:
    if annotations is None:
        return
    for key in ("positive_ranges", "ignore_ranges"):
        for item in annotations[key]:
            if int(item["end_frame"]) >= decoded_frame_count:
                raise ValueError(
                    f"annotation.{key} extends beyond decoded video frames: {path}"
                )


def _normalize_alignable_frame(
    frame: np.ndarray,
    reference: np.ndarray,
    homography: np.ndarray | None,
) -> np.ndarray:
    """Apply the exact production SceneGuard normalization contract.

    ``DetectionPipeline`` first uses linear interpolation with replicated
    source borders, then computes a nearest-neighbour source-footprint mask and
    hard-fills pixels outside that footprint from the approved reference.  The
    offline training corpus must be byte-equivalent to that path; alpha
    blending or a constant warp border would introduce a train/runtime domain
    shift at fractional-pixel boundaries.
    """

    if homography is None:
        raise ValueError("ALIGNABLE result did not include a homography")
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("ALIGNABLE result included an invalid homography")

    height, width = reference.shape[:2]
    warped = cv2.warpPerspective(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    validity = cv2.warpPerspective(
        np.full(frame.shape[:2], 255, dtype=np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    invalid = validity == 0
    if np.any(invalid):
        warped = warped.copy()
        warped[invalid] = reference[invalid]
    return warped


def _validated_inputs(
    videos: Sequence[str | Path],
    reference: str | Path,
    annotations: Sequence[str | Path | None] | None,
) -> tuple[list[Path], Path, list[Path | None]]:
    video_paths = [Path(item).resolve() for item in videos]
    if not video_paths:
        raise ValueError("at least one --video is required")
    if len(set(video_paths)) != len(video_paths):
        raise ValueError("duplicate video inputs are not allowed")
    for path in video_paths:
        if not path.is_file():
            raise FileNotFoundError(f"video not found: {path}")

    reference_path = Path(reference).resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f"reference image not found: {reference_path}")

    if annotations is None:
        annotation_paths: list[Path | None] = [None] * len(video_paths)
    else:
        if len(annotations) != len(video_paths):
            raise ValueError(
                "--annotations must be omitted or supplied once per --video; "
                "use '-' for an unannotated source"
            )
        annotation_paths = [
            None if item is None or str(item) == "-" else Path(item).resolve()
            for item in annotations
        ]
    for path in annotation_paths:
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"annotation file not found: {path}")
    return video_paths, reference_path, annotation_paths


def _prepare_output_target(output_dir: Path) -> None:
    if output_dir.is_symlink():
        raise ValueError("output directory cannot be a symbolic link")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        # Publishing uses a same-volume atomic directory rename.  An existing
        # empty target is removed only after the explicit non-empty check.
        output_dir.rmdir()
    output_dir.parent.mkdir(parents=True, exist_ok=True)


def _write_sample(path: Path, frame: np.ndarray) -> str:
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"could not write sample image: {path}")
    return _sha256(path)


def _scene_decision(
    result: SceneGuardResult,
    *,
    allow_alignable: bool,
) -> tuple[bool, str]:
    if result.state is SceneState.STABLE:
        return True, "accepted_stable"
    if result.state is SceneState.ALIGNABLE:
        if not allow_alignable:
            return False, "alignable_not_allowed"
        if result.homography is None:
            return False, "alignable_missing_homography"
        matrix = np.asarray(result.homography)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            return False, "alignable_invalid_homography"
        return True, "accepted_alignable"
    return False, f"scene_guard_{_state_value(result.state)}"


def _validate_scene_guard_config(config: SceneGuardConfig) -> None:
    for name, value in asdict(config).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"scene_guard.{name} must be finite")


def _validate_identifier(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{name} cannot have leading or trailing whitespace")
    return value


def build_scene_profile_baseline(
    *,
    camera_id: str,
    scene_profile_id: str,
    videos: Sequence[str | Path],
    reference: str | Path,
    output_dir: str | Path,
    sample_every: int,
    max_images: int,
    min_training_images: int = _DEFAULT_MIN_TRAINING_IMAGES,
    min_calibration_images: int = _DEFAULT_MIN_CALIBRATION_IMAGES,
    min_accepted_time_span_seconds: float = (
        _DEFAULT_MIN_ACCEPTED_TIME_SPAN_SECONDS
    ),
    annotations: Sequence[str | Path | None] | None = None,
    allow_alignable: bool = False,
    strict_release_decoder: bool = False,
    scene_guard_config: SceneGuardConfig | None = None,
) -> dict[str, Any]:
    """Build immutable inputs that may be ready for offline training only.

    ``training_ready`` never means that a model is approved for production;
    the emitted manifest always has ``release_eligible=false``.
    """

    camera_id = _validate_identifier(camera_id, name="camera_id")
    scene_profile_id = _validate_identifier(
        scene_profile_id,
        name="scene_profile_id",
    )
    if not isinstance(sample_every, int) or isinstance(sample_every, bool) or sample_every < 1:
        raise ValueError("sample_every must be a positive integer")
    if not isinstance(max_images, int) or isinstance(max_images, bool) or max_images < 1:
        raise ValueError("max_images must be a positive integer")
    if (
        not isinstance(min_training_images, int)
        or isinstance(min_training_images, bool)
        or min_training_images < 2
    ):
        raise ValueError("min_training_images must be an integer of at least 2")
    if (
        not isinstance(min_calibration_images, int)
        or isinstance(min_calibration_images, bool)
        or min_calibration_images < 2
    ):
        raise ValueError("min_calibration_images must be an integer of at least 2")
    if (
        isinstance(min_accepted_time_span_seconds, bool)
        or not isinstance(min_accepted_time_span_seconds, (int, float))
        or not math.isfinite(float(min_accepted_time_span_seconds))
        or float(min_accepted_time_span_seconds) <= 0.0
    ):
        raise ValueError(
            "min_accepted_time_span_seconds must be finite and greater than zero"
        )
    min_accepted_time_span_seconds = float(
        min_accepted_time_span_seconds
    )
    if type(allow_alignable) is not bool:
        raise TypeError("allow_alignable must be a literal boolean")
    if type(strict_release_decoder) is not bool:
        raise TypeError("strict_release_decoder must be a literal boolean")
    if scene_guard_config is not None and not isinstance(
        scene_guard_config,
        SceneGuardConfig,
    ):
        raise TypeError("scene_guard_config must be a SceneGuardConfig")

    video_paths, reference_path, annotation_paths = _validated_inputs(
        videos,
        reference,
        annotations,
    )
    target = Path(output_dir).resolve()
    _prepare_output_target(target)

    # Decode and publish the exact same immutable byte sequence.  Reading the
    # path separately for hashing/copying could otherwise bind a concurrently
    # replaced file rather than the pixels SceneGuard actually used.
    reference_bytes = reference_path.read_bytes()
    reference_hash = hashlib.sha256(reference_bytes).hexdigest()
    reference_frame = cv2.imdecode(
        np.frombuffer(reference_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if reference_frame is None or reference_frame.size == 0:
        raise ValueError(f"could not decode reference image: {reference_path}")
    config = scene_guard_config or SceneGuardConfig()
    _validate_scene_guard_config(config)

    source_hashes = [_sha256(path) for path in video_paths]
    if len(set(source_hashes)) != len(source_hashes):
        raise ValueError(
            "duplicate video content is not allowed, even through different paths"
        )
    staging = target.parent / f".{target.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir()
    images_dir = staging / "images"
    images_dir.mkdir()

    samples: list[dict[str, Any]] = []
    source_inventory: list[dict[str, Any]] = []
    confirmed_states: Counter[str] = Counter()
    raw_states: Counter[str] = Counter()
    accepted_states: Counter[str] = Counter()
    rejected_states: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    scene_guard_evaluated = 0
    geometrically_accepted = 0
    sample_opportunities = 0
    total_frames_read = 0
    truncated_by_max_images = False

    try:
        reference_copy = staging / f"reference{reference_path.suffix.lower()}"
        reference_copy.write_bytes(reference_bytes)

        for source_index, (video_path, annotation_path, source_hash) in enumerate(
            zip(video_paths, annotation_paths, source_hashes, strict=True)
        ):
            annotation_hash = (
                _sha256(annotation_path) if annotation_path is not None else None
            )
            strict_decoder = (
                StrictVideoDecoder(video_path, expected_sha256=source_hash)
                if strict_release_decoder
                else None
            )
            capture = None
            if strict_decoder is None:
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    capture.release()
                    raise OSError(f"could not open video: {video_path}")

            source_confirmed_states: Counter[str] = Counter()
            source_raw_states: Counter[str] = Counter()
            source_accepted_states: Counter[str] = Counter()
            source_rejected_states: Counter[str] = Counter()
            source_rejections: Counter[str] = Counter()
            source_sample_opportunities = 0
            source_scene_guard_evaluated = 0
            source_geometrically_accepted = 0
            source_accepted_images = 0
            source_accepted_frame_indices: list[int] = []
            source_accepted_time_seconds: list[float] = []
            source_truncated_by_max_images = False
            frames_read = 0
            strict_decoder_evidence: dict[str, Any] | None = None
            timestamp_evidence: dict[str, Any] | None = None
            guard = SceneGuard(reference_frame=reference_frame, config=config)
            try:
                if strict_decoder is not None:
                    fps = strict_decoder.nominal_frame_rate
                    reported_frame_count = strict_decoder.probed_frame_count
                    container_reported_frame_count = (
                        strict_decoder.declared_frame_count
                    )
                else:
                    assert capture is not None
                    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                    if not math.isfinite(fps) or fps <= 0.0:
                        raise ValueError(
                            f"video has no trustworthy FPS: {video_path}"
                        )
                    reported_raw = float(
                        capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
                    )
                    reported_frame_count = (
                        int(round(reported_raw))
                        if math.isfinite(reported_raw) and reported_raw > 0.0
                        else None
                    )
                    container_reported_frame_count = reported_frame_count
                loaded_annotations = (
                    _load_annotations_strict(
                        annotation_path,
                        source_video_sha256=source_hash,
                        camera_id=camera_id,
                    )
                    if annotation_path is not None
                    else None
                )

                decode_context = (
                    strict_decoder
                    if strict_decoder is not None
                    else nullcontext()
                )
                with decode_context:
                    while True:
                        if strict_decoder is not None:
                            try:
                                decoded = next(strict_decoder)
                            except StopIteration:
                                break
                            frame = decoded.frame
                            frame_index = decoded.frame_index
                            frame_time_seconds = (
                                decoded.timestamp.pts_time_seconds
                            )
                            frame_pts = decoded.timestamp.pts
                            frame_time_base = decoded.timestamp.time_base
                        else:
                            assert capture is not None
                            ok, frame = capture.read()
                            if not ok:
                                break
                            frame_index = frames_read
                            frame_time_seconds = frame_index / fps
                            frame_pts = None
                            frame_time_base = None
                        if frame_index != frames_read:
                            raise OSError(
                                "decoder returned a non-contiguous frame index "
                                f"{frame_index} after {frames_read - 1}: {video_path}"
                            )
                        if (
                            frame is None
                            or not isinstance(frame, np.ndarray)
                            or frame.size == 0
                        ):
                            raise OSError(
                                "video returned an invalid frame at index "
                                f"{frames_read}: {video_path}"
                            )
                        frames_read += 1
                        total_frames_read += 1

                        exclusion = _annotation_exclusion(
                            frame_index,
                            loaded_annotations,
                        )
                        if exclusion is not None:
                            rejections[exclusion] += 1
                            source_rejections[exclusion] += 1
                            continue
                        if frame_index % sample_every != 0:
                            rejections["sampling_interval"] += 1
                            source_rejections["sampling_interval"] += 1
                            continue

                        sample_opportunities += 1
                        source_sample_opportunities += 1
                        scene_guard_evaluated += 1
                        source_scene_guard_evaluated += 1
                        result = guard.evaluate(frame, return_aligned=False)
                        state = _state_value(result.state)
                        raw_state = _state_value(result.raw_state)
                        confirmed_states[state] += 1
                        raw_states[raw_state] += 1
                        source_confirmed_states[state] += 1
                        source_raw_states[raw_state] += 1

                        accepted, decision = _scene_decision(
                            result,
                            allow_alignable=allow_alignable,
                        )
                        if not accepted:
                            rejected_states[state] += 1
                            source_rejected_states[state] += 1
                            rejections[decision] += 1
                            source_rejections[decision] += 1
                            continue

                        geometrically_accepted += 1
                        source_geometrically_accepted += 1
                        if len(samples) >= max_images:
                            truncated_by_max_images = True
                            source_truncated_by_max_images = True
                            rejected_states[state] += 1
                            source_rejected_states[state] += 1
                            rejections["max_images_limit"] += 1
                            source_rejections["max_images_limit"] += 1
                            continue

                        normalized = result.state is SceneState.ALIGNABLE
                        output_frame = (
                            _normalize_alignable_frame(
                                frame,
                                reference_frame,
                                result.homography,
                            )
                            if normalized
                            else frame.copy()
                        )
                        filename = f"sample_{len(samples):06d}.png"
                        relative_path = Path("images") / filename
                        output_hash = _write_sample(
                            staging / relative_path,
                            output_frame,
                        )
                        accepted_states[state] += 1
                        source_accepted_states[state] += 1
                        source_accepted_images += 1
                        source_accepted_frame_indices.append(frame_index)
                        source_accepted_time_seconds.append(frame_time_seconds)
                        sample = {
                            "sample_index": len(samples),
                            "path": relative_path.as_posix(),
                            "sha256": output_hash,
                            "source_index": source_index,
                            "source_path": str(video_path),
                            "source_sha256": source_hash,
                            "frame_index": frame_index,
                            "time_seconds": frame_time_seconds,
                            "scene_state": state,
                            "raw_scene_state": raw_state,
                            "scene_reason": result.reason,
                            "scene_metrics": result.metrics.to_dict(),
                            "normalized": normalized,
                            "normalization": (
                                "homography_to_reference_with_reference_fill"
                                if normalized
                                else "none"
                            ),
                        }
                        if strict_decoder is not None:
                            sample["pts"] = frame_pts
                            sample["time_base"] = frame_time_base
                        samples.append(sample)

                if strict_decoder is not None:
                    strict_decoder_evidence = strict_decoder.evidence()
                    timestamp_path = (
                        staging
                        / "decoder"
                        / f"source_{source_index:06d}.frames.jsonl"
                    )
                    timestamp_evidence = write_frame_timestamp_jsonl(
                        timestamp_path,
                        strict_decoder.timestamps,
                    )
                    timestamp_evidence["path"] = timestamp_path.relative_to(
                        staging
                    ).as_posix()
            finally:
                if capture is not None:
                    capture.release()

            if reported_frame_count is not None and frames_read < reported_frame_count:
                raise OSError(
                    "video decoding ended before the reported frame count "
                    f"({frames_read} < {reported_frame_count}): {video_path}"
                )
            _validate_annotation_bounds(
                loaded_annotations,
                decoded_frame_count=frames_read,
                path=annotation_path,
            )
            if _sha256(video_path) != source_hash:
                raise OSError(f"video changed while it was being decoded: {video_path}")
            if (
                annotation_path is not None
                and _sha256(annotation_path) != annotation_hash
            ):
                raise OSError(
                    f"annotation changed while the video was decoded: {annotation_path}"
                )
            frame_count_matches = (
                reported_frame_count is not None
                and frames_read == reported_frame_count
            )
            accepted_time_span_seconds = (
                source_accepted_time_seconds[-1]
                - source_accepted_time_seconds[0]
                if len(source_accepted_frame_indices) >= 2
                else 0.0
            )
            annotation_review = (
                loaded_annotations["review_evidence"]
                if loaded_annotations is not None
                else {
                    "complete": False,
                    "reasons": ["annotation_missing"],
                    "schema_version": None,
                    "source_video_sha256": None,
                    "source_video_sha256_matches": False,
                    "source_session_id": None,
                    "camera_id": None,
                    "camera_id_matches": False,
                    "review_complete": False,
                    "reviewed_by": None,
                    "reviewed_at": None,
                    "reviewed_at_has_timezone": False,
                    "coordinate_system": (
                        "zero_based_inclusive_frame_ranges_only"
                    ),
                }
            )
            source_entry: dict[str, Any] = {
                "source_index": source_index,
                "path": str(video_path),
                "sha256": source_hash,
                "fps": fps,
                "reported_frame_count": reported_frame_count,
                "decoded_frame_count": frames_read,
                "reported_frame_count_matches": frame_count_matches,
                "source_exhausted": frame_count_matches,
                "source_read_error": None,
                "annotation_path": (
                    str(annotation_path) if annotation_path is not None else None
                ),
                "annotation_sha256": (
                    annotation_hash
                ),
                "annotation_exclusions_auditable": annotation_review["complete"],
                "annotation_review_evidence": annotation_review,
                "sample_opportunities": source_sample_opportunities,
                "scene_guard_evaluated": source_scene_guard_evaluated,
                "geometrically_accepted": source_geometrically_accepted,
                "accepted_images": source_accepted_images,
                "first_accepted_frame_index": (
                    source_accepted_frame_indices[0]
                    if source_accepted_frame_indices
                    else None
                ),
                "last_accepted_frame_index": (
                    source_accepted_frame_indices[-1]
                    if source_accepted_frame_indices
                    else None
                ),
                "accepted_time_span_seconds": accepted_time_span_seconds,
                "rejected_after_sampling": (
                    source_scene_guard_evaluated - source_accepted_images
                ),
                "truncated_by_max_images": source_truncated_by_max_images,
                "confirmed_state_counts": dict(sorted(source_confirmed_states.items())),
                "raw_state_counts": dict(sorted(source_raw_states.items())),
                "accepted_state_counts": dict(sorted(source_accepted_states.items())),
                "rejected_state_counts": dict(sorted(source_rejected_states.items())),
                "rejection_counts": dict(sorted(source_rejections.items())),
            }
            if strict_decoder_evidence is not None:
                source_entry.update({
                    "decode_backend": "ffprobe+ffmpeg-rawvideo",
                    "container_reported_frame_count": (
                        container_reported_frame_count
                    ),
                    "opencv_read_terminated": False,
                    "source_read_error_detectable": True,
                    "vfr_timestamps_preserved": True,
                    "frame_time_basis": "pts_times_stream_time_base",
                    "strict_decoder_evidence": strict_decoder_evidence,
                    "frame_timestamp_evidence": timestamp_evidence,
                })
            else:
                source_entry.update({
                    "opencv_read_terminated": True,
                    "source_read_error_detectable": False,
                    "vfr_timestamps_preserved": False,
                    "frame_time_basis": (
                        "frame_index_divided_by_reported_constant_fps"
                    ),
                })
            source_inventory.append(source_entry)

        declared_session_ids = sorted({
            str(item["annotation_review_evidence"]["source_session_id"])
            for item in source_inventory
            if item["annotation_review_evidence"].get("complete") is True
            and type(
                item["annotation_review_evidence"].get("source_session_id")
            ) is str
            and item["annotation_review_evidence"]["source_session_id"].strip()
            == item["annotation_review_evidence"]["source_session_id"]
        })
        calibration_session_id = (
            declared_session_ids[-1]
            if len(declared_session_ids) >= 2
            else None
        )
        source_roles: dict[int, str] = {}
        for item in source_inventory:
            session_id = item["annotation_review_evidence"].get(
                "source_session_id"
            )
            if calibration_session_id is None or session_id not in declared_session_ids:
                role = "diagnostic_only"
            elif session_id == calibration_session_id:
                role = "calibration"
            else:
                role = "train"
            item["split_role"] = role
            source_roles[int(item["source_index"])] = role
        for sample in samples:
            sample["split_role"] = source_roles[int(sample["source_index"])]

        split_source_counts = Counter(
            item["split_role"] for item in source_inventory
        )
        split_sample_counts = Counter(sample["split_role"] for sample in samples)
        split_time_spans: Counter[str] = Counter()
        for item in source_inventory:
            split_time_spans[item["split_role"]] += float(
                item["accepted_time_span_seconds"]
            )

        if strict_release_decoder:
            source_inventory_complete = all(
                item["source_exhausted"] is True
                and item["reported_frame_count_matches"] is True
                and isinstance(item.get("strict_decoder_evidence"), dict)
                and item["strict_decoder_evidence"].get(
                    "sufficient_for_model_release"
                ) is True
                and isinstance(item.get("frame_timestamp_evidence"), dict)
                and item["frame_timestamp_evidence"].get("frame_count")
                == item["decoded_frame_count"]
                for item in source_inventory
            )
        else:
            source_inventory_complete = all(
                item["source_exhausted"] is True
                and item["opencv_read_terminated"] is True
                and item["reported_frame_count_matches"] is True
                for item in source_inventory
            )
        annotation_evidence_complete = all(
            item["annotation_exclusions_auditable"] is True
            for item in source_inventory
        )
        selection_evidence_complete = not truncated_by_max_images
        corpus_evidence_complete = (
            source_inventory_complete
            and annotation_evidence_complete
            and selection_evidence_complete
        )
        accepted_time_span_seconds = sum(
            float(item["accepted_time_span_seconds"])
            for item in source_inventory
        )
        training_readiness_checks = {
            "corpus_evidence_complete": corpus_evidence_complete,
            "at_least_two_declared_independent_sessions": (
                len(declared_session_ids) >= 2
            ),
            "minimum_training_images_met": (
                split_sample_counts["train"] >= min_training_images
            ),
            "minimum_calibration_images_met": (
                split_sample_counts["calibration"] >= min_calibration_images
            ),
            "minimum_accepted_time_span_met": (
                accepted_time_span_seconds
                >= min_accepted_time_span_seconds
            ),
            "full_source_grouping_has_no_diagnostic_samples": (
                split_sample_counts["diagnostic_only"] == 0
            ),
        }
        training_ready = all(training_readiness_checks.values())
        if training_ready:
            dataset_status = "offline_training_input_ready"
        elif samples:
            dataset_status = "diagnostic_only_not_training_ready"
        else:
            dataset_status = "rejected_no_samples"
        if strict_release_decoder:
            strict_source_evidence = [
                item.get("strict_decoder_evidence") for item in source_inventory
            ]
            all_sources_strictly_audited = bool(strict_source_evidence) and all(
                isinstance(item, dict)
                and item.get("sufficient_for_model_release") is True
                for item in strict_source_evidence
            )
            decoder_evidence: dict[str, Any] = {
                "backend": "ffprobe+ffmpeg-rawvideo",
                "strict_release_decoder_requested": True,
                "all_sources_strictly_audited": all_sources_strictly_audited,
                "source_count": len(source_inventory),
                "read_termination_signal": (
                    "ffmpeg_clean_zero_exit_after_complete_raw_frame_stream"
                ),
                "decoded_frame_count_cross_check": (
                    "ffmpeg_raw_frames_equal_ffprobe_show_frames_inventory"
                ),
                "source_sha256_checked_before_and_after_decode": True,
                "decode_error_distinguishable_from_clean_eof": True,
                "variable_frame_rate_timestamps_preserved": True,
                "time_basis": "per_frame_pts_times_stream_time_base",
                "timestamp_evidence_format": (
                    "sha256_bound_jsonl_one_record_per_decoded_frame"
                ),
                "sufficient_for_model_release": (
                    all_sources_strictly_audited
                    and source_inventory_complete
                ),
            }
        else:
            decoder_evidence = {
                "backend": "opencv.VideoCapture",
                "read_termination_signal": "capture.read_returned_false",
                "reported_frame_count_required_to_match": True,
                "decode_error_distinguishable_from_clean_eof": False,
                "variable_frame_rate_timestamps_preserved": False,
                "time_basis": "frame_index_divided_by_reported_constant_fps",
                "limitation": (
                    "OpenCV does not expose packet timestamps here and cannot "
                    "reliably distinguish a decode error from clean EOF; this "
                    "evidence can qualify offline training inputs only"
                ),
                "sufficient_for_model_release": False,
            }
        manifest: dict[str, Any] = {
            "schema_version": (
                _STRICT_SCHEMA_VERSION
                if strict_release_decoder
                else _SCHEMA_VERSION
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "camera_id": camera_id,
            "scene_profile_id": scene_profile_id,
            "validation_only": True,
            "release_eligible": False,
            "release_eligibility_reason": (
                "offline corpus construction never grants model release approval"
            ),
            "operator_approved": False,
            "active_profile_modified": False,
            "reference_auto_selected": False,
            "dataset_status": dataset_status,
            "training_ready": training_ready,
            "training_readiness_scope": "offline_training_input_only",
            "training_readiness_checks": training_readiness_checks,
            "diagnostic_samples_available": bool(samples),
            "corpus_evidence_complete": corpus_evidence_complete,
            "source_inventory_evidence_complete": source_inventory_complete,
            "annotation_exclusion_evidence_complete": annotation_evidence_complete,
            "selection_evidence_complete": selection_evidence_complete,
            "decoder_evidence": decoder_evidence,
            "split_contract": {
                "policy": "complete_operator_declared_source_session_grouping",
                "random_frame_split_allowed": False,
                "assignment": (
                    "lexicographically_greatest_declared_session_id_is_calibration; "
                    "all_other_declared_sessions_are_train"
                ),
                "declared_session_ids": declared_session_ids,
                "declared_independent_session_count": len(declared_session_ids),
                "independence_evidence_basis": (
                    "operator-reviewed source_session_id; capture independence is "
                    "declared rather than cryptographically proven"
                ),
                "minimum_distinct_sessions_for_training_ready": 2,
                "single_session_policy": "diagnostic_only",
                "single_session_temporal_block_suggestion": (
                    "temporal blocks may be used for exploratory diagnostics only; "
                    "they are not independent train/calibration evidence"
                ),
                "source_counts": {
                    role: split_source_counts[role] for role in _SPLIT_ROLES
                },
                "sample_counts": {
                    role: split_sample_counts[role] for role in _SPLIT_ROLES
                },
                "accepted_time_span_seconds": {
                    role: split_time_spans[role] for role in _SPLIT_ROLES
                },
            },
            "normalization_contract": {
                "contract_id": _NORMALIZATION_CONTRACT_ID,
                "production_consumer": "argus.core.pipeline.DetectionPipeline",
                "reference_coordinates": "explicit_scene_profile_reference",
                "stable_frame_policy": "no_warp_copy_source_frame",
                "alignable_frame_policy": (
                    "warp_to_reference_then_fill_outside_source_footprint_from_reference"
                ),
                "image_warp": {
                    "operation": "cv2.warpPerspective",
                    "interpolation": "INTER_LINEAR",
                    "border_mode": "BORDER_REPLICATE",
                },
                "source_footprint_mask": {
                    "input": "uint8_255_source_sized_mask",
                    "operation": "cv2.warpPerspective",
                    "interpolation": "INTER_NEAREST",
                    "border_mode": "BORDER_CONSTANT",
                    "border_value": 0,
                    "invalid_pixel_policy": "copy_exact_reference_pixel",
                },
                "other_scene_states": "reject",
            },
            "reference": {
                "source_path": str(reference_path),
                "copied_path": reference_copy.relative_to(staging).as_posix(),
                "sha256": reference_hash,
                "width": int(reference_frame.shape[1]),
                "height": int(reference_frame.shape[0]),
            },
            "parameters": {
                "sample_every": sample_every,
                "max_images": max_images,
                "min_training_images": min_training_images,
                "min_calibration_images": min_calibration_images,
                "min_accepted_time_span_seconds": (
                    min_accepted_time_span_seconds
                ),
                "accepted_time_span_definition": (
                    "sum_per_source_last_minus_first_accepted_pts_time"
                    if strict_release_decoder
                    else (
                        "sum_per_source_last_minus_first_accepted_frame_divided_by_fps"
                    )
                ),
                "max_images_scope": "global_across_sources",
                "deterministic_sample_rule": _SAMPLE_RULE,
                "allow_alignable": allow_alignable is True,
                "strict_release_decoder": strict_release_decoder is True,
                "alignable_normalization": (
                    "warp_to_reference_and_fill_outside_source_footprint_with_reference"
                ),
                "scene_guard": asdict(config),
            },
            "selection": {
                "total_frames_read": total_frames_read,
                "sample_opportunities": sample_opportunities,
                "scene_guard_evaluated": scene_guard_evaluated,
                "geometrically_accepted": geometrically_accepted,
                "accepted_images": len(samples),
                "accepted_time_span_seconds": accepted_time_span_seconds,
                "rejected_after_sampling": scene_guard_evaluated - len(samples),
                "truncated_by_max_images": truncated_by_max_images,
                "confirmed_state_counts": dict(sorted(confirmed_states.items())),
                "raw_state_counts": dict(sorted(raw_states.items())),
                "accepted_state_counts": dict(sorted(accepted_states.items())),
                "rejected_state_counts": dict(sorted(rejected_states.items())),
                "rejection_counts": dict(sorted(rejections.items())),
            },
            "sources": source_inventory,
            "samples": samples,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"output target appeared during build: {target}")
        os.replace(staging, target)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build offline-training inputs for one explicit camera scene. "
            "This command never activates a profile or grants release approval."
        ),
    )
    parser.add_argument(
        "--camera-id",
        required=True,
        help="Exact camera identity bound into annotations and output manifest",
    )
    parser.add_argument(
        "--scene-profile-id",
        required=True,
        help="Exact scene-profile identity bound into the output manifest",
    )
    parser.add_argument(
        "--video",
        type=Path,
        action="append",
        required=True,
        help="Recorded video; repeat for multiple sources",
    )
    parser.add_argument(
        "--annotations",
        action="append",
        help=(
            "Annotation JSON corresponding to each --video. Omit for all videos, "
            "or repeat once per video and use '-' where unavailable."
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Explicit trusted reference image; videos can never replace it",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-every", type=int, required=True)
    parser.add_argument("--max-images", type=int, required=True)
    parser.add_argument(
        "--min-training-images",
        type=int,
        default=_DEFAULT_MIN_TRAINING_IMAGES,
        help=(
            "Minimum accepted normal images for offline-training readiness "
            f"(default: {_DEFAULT_MIN_TRAINING_IMAGES}; never below 2)"
        ),
    )
    parser.add_argument(
        "--min-calibration-images",
        type=int,
        default=_DEFAULT_MIN_CALIBRATION_IMAGES,
        help=(
            "Minimum accepted images in the source-grouped calibration split "
            f"(default: {_DEFAULT_MIN_CALIBRATION_IMAGES}; never below 2)"
        ),
    )
    parser.add_argument(
        "--min-accepted-time-span-seconds",
        type=float,
        default=_DEFAULT_MIN_ACCEPTED_TIME_SPAN_SECONDS,
        help=(
            "Minimum summed per-source accepted temporal span for offline-training "
            f"readiness (default: {_DEFAULT_MIN_ACCEPTED_TIME_SPAN_SECONDS:g}s)"
        ),
    )
    parser.add_argument(
        "--allow-alignable",
        action="store_true",
        help=(
            "Accept geometrically ALIGNABLE frames only after warping to the "
            "explicit reference coordinates; disabled by default"
        ),
    )
    parser.add_argument(
        "--strict-release-decoder",
        action="store_true",
        help=(
            "Use fail-closed ffprobe/ffmpeg full-source auditing and emit a v3 "
            "manifest with hash-bound per-frame PTS evidence. Disabled by "
            "default so the existing v2 offline-only behavior remains explicit."
        ),
    )
    parser.add_argument(
        "--scene-guard-config-json",
        default="{}",
        help="Optional JSON object overriding offline SceneGuard thresholds",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config_payload = json.loads(args.scene_guard_config_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--scene-guard-config-json is invalid JSON: {exc}") from exc
    if not isinstance(config_payload, dict):
        raise SystemExit("--scene-guard-config-json must decode to an object")
    try:
        config = SceneGuardConfig(**config_payload)
        manifest = build_scene_profile_baseline(
            camera_id=args.camera_id,
            scene_profile_id=args.scene_profile_id,
            videos=args.video,
            reference=args.reference,
            output_dir=args.output_dir,
            sample_every=args.sample_every,
            max_images=args.max_images,
            min_training_images=args.min_training_images,
            min_calibration_images=args.min_calibration_images,
            min_accepted_time_span_seconds=(
                args.min_accepted_time_span_seconds
            ),
            annotations=args.annotations,
            allow_alignable=args.allow_alignable,
            strict_release_decoder=args.strict_release_decoder,
            scene_guard_config=config,
        )
    except (FileNotFoundError, FileExistsError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"scene-profile baseline build failed: {exc}") from exc

    print(json.dumps({
        "output_dir": str(Path(args.output_dir).resolve()),
        "dataset_status": manifest["dataset_status"],
        "training_ready": manifest["training_ready"],
        "training_readiness_scope": manifest["training_readiness_scope"],
        "validation_only": manifest["validation_only"],
        "release_eligible": manifest["release_eligible"],
        "accepted_images": manifest["selection"]["accepted_images"],
        "truncated_by_max_images": manifest["selection"][
            "truncated_by_max_images"
        ],
        "corpus_evidence_complete": manifest["corpus_evidence_complete"],
        "decoder_sufficient_for_model_release": manifest[
            "decoder_evidence"
        ]["sufficient_for_model_release"],
    }, ensure_ascii=False, indent=2))
    return 0 if manifest["training_ready"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
