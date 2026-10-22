"""Immutable, server-resolved inputs for scene-bound training jobs.

This module performs file/metadata validation only.  It never decodes video,
loads a model, approves a scene, activates a profile, or authorizes release.
Both the workflow service and the queued-job executor use this same contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from argus.storage.release_evidence import (
    evidence_artifact_sha256,
    validate_baseline_manifest_source,
)


SCENE_TRAINING_INPUT_SCHEMA = "argus.scene-training-input.v1"
SCENE_WORKFLOW_SCHEMA = "argus.scene-workflow.v1"
SCENE_MIN_TRAINING_IMAGES = 100
SCENE_MIN_CALIBRATION_IMAGES = 50
SCENE_MIN_ACCEPTED_SPAN_SECONDS = 300.0
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_METADATA_BYTES = 16 * 1024 * 1024
_DESCRIPTOR_FIELDS = frozenset({
    "schema_version", "workflow_id", "scene_profile_id", "baseline_sha256",
    "manifest_sha256", "reference_sha256",
})
_BINDING_FIELDS = tuple(sorted(_DESCRIPTOR_FIELDS - {"schema_version"}))
_SPLIT_POLICY = "complete_operator_declared_source_session_grouping"
_SPLIT_ASSIGNMENT = (
    "lexicographically_greatest_declared_session_id_is_calibration; "
    "all_other_declared_sessions_are_train"
)


@dataclass(frozen=True, slots=True)
class SceneTrainingInput:
    baselines_dir: Path
    camera_id: str
    root: Path
    images_dir: Path
    image_count: int
    training_count: int
    calibration_count: int
    lineage: dict[str, str]


def _identifier(value: object, *, label: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a safe identifier of at most 64 characters")
    return value


def validate_scene_training_descriptor(value: object) -> dict[str, str]:
    """Validate job identity without accepting client-selected filesystem paths."""
    if not isinstance(value, dict) or set(value) != _DESCRIPTOR_FIELDS:
        raise ValueError("scene_workflow must contain the exact scene training identity fields")
    if value["schema_version"] != SCENE_TRAINING_INPUT_SCHEMA:
        raise ValueError("scene_workflow has an unsupported training-input schema")
    workflow_id = _identifier(value["workflow_id"], label="scene_workflow.workflow_id")
    if value["scene_profile_id"] != workflow_id:
        raise ValueError("scene_workflow scene_profile_id must equal workflow_id")
    for field in ("baseline_sha256", "manifest_sha256", "reference_sha256"):
        if type(value[field]) is not str or _SHA256.fullmatch(value[field]) is None:
            raise ValueError(f"scene_workflow.{field} must be a lowercase SHA256")
    return {field: value[field] for field in _DESCRIPTOR_FIELDS}


def _regular_path(path: Path, *, root: Path, directory: bool) -> Path:
    """Reject links/reparse components before following a controlled child path."""
    if not path.is_relative_to(root):
        raise ValueError("scene training path escapes its controlled root")
    candidates = [root]
    for part in path.relative_to(root).parts:
        candidates.append(candidates[-1] / part)
    for candidate in candidates:
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise ValueError("scene training paths must not contain links or reparse points")
    info = path.lstat()
    if directory and not stat.S_ISDIR(info.st_mode):
        raise ValueError("scene training directory is missing or not a regular directory")
    if not directory and not stat.S_ISREG(info.st_mode):
        raise ValueError("scene training metadata must be a regular file")
    return path


def _read_mapping(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = handle.read(_MAX_METADATA_BYTES + 1)
    if len(raw) > _MAX_METADATA_BYTES:
        raise ValueError("scene training metadata exceeds its controlled budget")
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("scene training metadata is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("scene training metadata must be an object")
    return value


def _regular_tree(root: Path) -> None:
    pending = [root]
    while pending:
        directory = pending.pop()
        for child in directory.iterdir():
            info = child.lstat()
            if stat.S_ISLNK(info.st_mode) or (
                getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            ):
                raise ValueError("scene baseline artifact tree cannot contain links or reparse points")
            if stat.S_ISDIR(info.st_mode):
                pending.append(child)
            elif not stat.S_ISREG(info.st_mode):
                raise ValueError("scene baseline artifact tree contains a non-regular file")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_json(value: object) -> str:
    if not isinstance(value, dict) or not value:
        raise ValueError("scene workflow record must contain its frozen scene_guard policy")
    try:
        return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("scene workflow scene_guard policy must be finite JSON") from exc


def _check_record(record: Mapping[str, Any], camera_id: str, lineage: Mapping[str, str]) -> str:
    if record.get("schema_version") != SCENE_WORKFLOW_SCHEMA or record.get("camera_id") != camera_id:
        raise ValueError("scene workflow record schema or camera does not match the training job")
    if any(record.get(field) != lineage[field] for field in _BINDING_FIELDS):
        raise ValueError("scene workflow record identity differs from the queued training job")
    return _policy_json(record.get("scene_guard"))


def validate_scene_training_input(
    *, baselines_dir: str | Path, camera_id: str, scene_workflow: object,
    zone_id: str = "default",
) -> SceneTrainingInput:
    """Resolve and fully verify one strict V3 baseline from its workflow ID.

    Workflow state is intentionally not an authorization flag.  Every queued
    execution rechecks the immutable identity, complete artifact hash, canonical
    V3 decoder evidence, sample bytes and whole-session split.  The workflow
    record and baseline must be located at server-derived paths; arbitrary
    ``baseline_path`` or mixed ``dataset_selection`` inputs are not accepted.
    """
    camera_id = _identifier(camera_id, label="camera_id")
    if zone_id != "default":
        raise ValueError("scene workflow training currently requires the default zone")
    lineage = validate_scene_training_descriptor(scene_workflow)
    workflow_id = lineage["workflow_id"]
    base = Path(baselines_dir).expanduser().absolute()
    _regular_path(base, root=base, directory=True)
    root = base / camera_id / "default" / f"scene-{workflow_id}"
    _regular_path(root, root=base, directory=True)
    images = _regular_path(root / "images", root=base, directory=True)
    manifest_path = _regular_path(root / "manifest.json", root=base, directory=False)
    workflow_root = base.parent / "scene_workflows"
    record_path = workflow_root / camera_id / workflow_id / "workflow.json"
    _regular_path(record_path, root=workflow_root, directory=False)
    scene_policy = _check_record(_read_mapping(record_path), camera_id, lineage)
    if _file_sha256(manifest_path) != lineage["manifest_sha256"]:
        raise ValueError("scene workflow baseline manifest SHA256 changed")
    _regular_tree(root)
    if evidence_artifact_sha256(root) != lineage["baseline_sha256"]:
        raise ValueError("scene workflow complete baseline SHA256 changed")

    # Reuse exactly the publication-side sample/PTS/EOF verification.  This
    # validates stored evidence only; it does not execute FFmpeg or load a model.
    manifest, _source_hashes, sessions = validate_baseline_manifest_source(root)
    if (
        manifest.get("schema_version") != "argus.scene-profile-baseline.v3"
        or manifest.get("camera_id") != camera_id
        or manifest.get("scene_profile_id") != workflow_id
        or manifest.get("training_ready") is not True
        or manifest.get("release_eligible") is not False
    ):
        raise ValueError("scene workflow training requires its exact strict V3 baseline")
    decoder = manifest.get("decoder_evidence")
    if not isinstance(decoder, dict) or decoder.get("sufficient_for_model_release") is not True:
        raise ValueError("scene workflow training requires sufficient strict decoder evidence")
    parameters = manifest.get("parameters")
    if (
        not isinstance(parameters, dict)
        or parameters.get("strict_release_decoder") is not True
        or parameters.get("allow_alignable") is not False
    ):
        raise ValueError("scene workflow baseline must use strict, non-alignable scene checks")
    if _policy_json(parameters.get("scene_guard")) != scene_policy:
        raise ValueError("scene workflow baseline scene_guard policy differs from the frozen workflow")
    for field, minimum in (
        ("min_training_images", SCENE_MIN_TRAINING_IMAGES),
        ("min_calibration_images", SCENE_MIN_CALIBRATION_IMAGES),
    ):
        if type(parameters.get(field)) is not int or parameters[field] < minimum:
            raise ValueError(f"scene workflow baseline {field} is below the controlled minimum")
    span = parameters.get("min_accepted_time_span_seconds")
    try:
        finite_span = type(span) in {int, float} and math.isfinite(span)
    except (OverflowError, TypeError, ValueError):
        finite_span = False
    if (
        not finite_span
        or span < SCENE_MIN_ACCEPTED_SPAN_SECONDS
    ):
        raise ValueError("scene workflow baseline accepted time span minimum is too small")
    reference = manifest.get("reference")
    if not isinstance(reference, dict) or reference.get("sha256") != lineage["reference_sha256"]:
        raise ValueError("scene workflow baseline reference identity differs from its workflow")
    split = manifest.get("split_contract")
    if (
        not isinstance(split, dict)
        or split.get("policy") != _SPLIT_POLICY
        or split.get("assignment") != _SPLIT_ASSIGNMENT
        or split.get("random_frame_split_allowed") is not False
        or split.get("declared_session_ids") != sorted(sessions)
        or len(sessions) < 2
    ):
        raise ValueError("scene workflow baseline has no exact independent session split")
    counts = split.get("sample_counts")
    if (
        not isinstance(counts, dict)
        or set(counts) != {"train", "calibration", "diagnostic_only"}
        or any(type(value) is not int or value < 0 for value in counts.values())
        or counts["train"] < SCENE_MIN_TRAINING_IMAGES
        or counts["calibration"] < SCENE_MIN_CALIBRATION_IMAGES
        or counts["diagnostic_only"] != 0
    ):
        raise ValueError("scene workflow baseline has insufficient train/calibration samples")
    _regular_tree(root)
    if (
        _file_sha256(manifest_path) != lineage["manifest_sha256"]
        or evidence_artifact_sha256(root) != lineage["baseline_sha256"]
    ):
        raise ValueError("scene workflow baseline changed during verification")
    if _check_record(_read_mapping(record_path), camera_id, lineage) != scene_policy:
        raise ValueError("scene workflow scene_guard policy changed during verification")
    return SceneTrainingInput(
        baselines_dir=base, camera_id=camera_id, root=root, images_dir=images,
        image_count=counts["train"] + counts["calibration"],
        training_count=counts["train"], calibration_count=counts["calibration"],
        lineage=dict(lineage),
    )


def revalidate_scene_training_input(value: SceneTrainingInput) -> SceneTrainingInput:
    """Recheck the same immutable inputs before candidate registration."""
    if not isinstance(value, SceneTrainingInput):
        raise TypeError("scene training input must be validated by the shared workflow contract")
    verified = validate_scene_training_input(
        baselines_dir=value.baselines_dir,
        camera_id=value.camera_id,
        scene_workflow=value.lineage,
    )
    if verified != value:
        raise ValueError("scene training input identity or sample counts changed")
    return verified


def validate_scene_training_calibration(
    *, model_path: str | Path, scene_training_input: SceneTrainingInput,
) -> dict[str, Any]:
    """Verify the serialized runtime calibration against the frozen V3 split.

    A trainer's ``ok`` status is not calibration evidence.  Check the exact
    sidecar consumed beside this runtime file without importing a detector or
    NumPy.  Scores are rounded to six decimals by ConformalCalibrator.save(),
    so empirical FPRs cannot be recomputed against its unrounded thresholds.
    """
    if not isinstance(scene_training_input, SceneTrainingInput):
        raise TypeError("scene calibration requires a validated scene training input")
    expected_count = scene_training_input.calibration_count
    if type(expected_count) is not int or expected_count < SCENE_MIN_CALIBRATION_IMAGES:
        raise ValueError("scene calibration requires the complete frozen calibration split")
    model = Path(model_path).expanduser().absolute()
    _regular_path(model, root=model.parent, directory=False)
    calibration_path = _regular_path(
        model.parent / "calibration.json", root=model.parent, directory=False,
    )
    calibration = _read_mapping(calibration_path)
    if (
        type(calibration.get("n_samples")) is not int
        or calibration["n_samples"] != expected_count
    ):
        raise ValueError("scene calibration n_samples must equal the frozen calibration count")

    def finite_number(value: object) -> bool:
        try:
            return type(value) in {int, float} and math.isfinite(value)
        except (OverflowError, TypeError, ValueError):
            return False

    scores = calibration.get("sorted_scores")
    if (
        not isinstance(scores, list) or len(scores) != expected_count
        or not all(finite_number(score) for score in scores)
        or any(left > right for left, right in zip(scores, scores[1:]))
    ):
        raise ValueError("scene calibration sorted_scores must completely cover the frozen split with finite sorted scores")
    levels = ("info", "low", "medium", "high")
    thresholds = [calibration.get(level) for level in levels]
    if (
        not all(finite_number(value) for value in thresholds)
        or any(left >= right for left, right in zip(thresholds, thresholds[1:]))
    ):
        raise ValueError("scene calibration thresholds must be finite and strictly increasing")
    target_fprs = calibration.get("target_fprs")
    if (
        not isinstance(target_fprs, dict) or set(target_fprs) != set(levels)
        or not all(finite_number(value) and 0 < value < 1 for value in target_fprs.values())
        or any(target_fprs[left] <= target_fprs[right] for left, right in zip(levels, levels[1:]))
    ):
        raise ValueError("scene calibration target_fprs must contain four valid decreasing alpha levels")
    actual_fprs = calibration.get("actual_fprs")
    if (
        not isinstance(actual_fprs, dict) or set(actual_fprs) != set(levels)
        or not all(finite_number(value) and 0 <= value <= 1 for value in actual_fprs.values())
        or any(actual_fprs[left] < actual_fprs[right] for left, right in zip(levels, levels[1:]))
    ):
        raise ValueError("scene calibration actual_fprs must contain four finite non-increasing rates")
    return calibration
