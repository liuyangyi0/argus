"""Versioned training and deployment artifact locations.

An artifact path always names one deployable file.  The adjacent manifest
connects it to its training checkpoint and run-specific export directory;
callers must never choose the newest file from a shared camera directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import quote

MANIFEST_NAME = "artifact.json"
_SUFFIXES = {"torch": ".pt", "openvino": ".xml", "onnx": ".onnx"}
_RESERVED_RUN_NAME = re.compile(r"run-\d{8}T\d{12}Z-[0-9a-f]{12}\Z")


def _directory_component(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."} or any(c in value for c in "/\\"):
        raise ValueError(f"Invalid {label} for artifact directory")
    # Logical group IDs use 'group:<id>'; encode ':' and '%' consistently so
    # Linux and Windows reserve the same safe, non-colliding run paths.
    return quote(value, safe="-_.")


def is_independent_artifact_run(directory: Path) -> bool:
    """Identify child runs that must not alter a legacy shared model identity.

    The exact generated name is reserved immediately at mkdir time, before a
    manifest exists. Other historical 'run-*' names are excluded only when a
    validated manifest declares an independent training/export boundary.
    """
    if _RESERVED_RUN_NAME.fullmatch(directory.name):
        return True
    if not directory.name.startswith("run-"):
        return False
    try:
        return bool(artifact_metadata(directory))
    except (OSError, ValueError, TypeError):
        return False


@dataclass(frozen=True)
class ArtifactRun:
    run_id: str
    camera_id: str
    zone_id: str
    training_dir: Path
    export_dir: Path


@dataclass(frozen=True)
class RuntimeCalibrationScope:
    """One shared calibration lookup policy for loading and fingerprinting."""

    candidates: tuple[tuple[str, Path], ...]
    manifest: tuple[str, Path] | None = None
    calibration_required: bool = False


def runtime_calibration_scope(model_path: Path) -> RuntimeCalibrationScope:
    """Bound new-run lookups to export_dir; preserve legacy four-level lookup.

    A run marker is part of runtime identity because removing it can change
    which calibration file is selected. Invalid markers never fall through to
    the legacy ancestor search.
    """
    model = model_path.resolve()
    legacy: list[tuple[str, Path]] = []
    search_dir = model.parent
    for depth in range(4):
        legacy.append((f"{'../' * depth}calibration.json", search_dir / "calibration.json"))
        marker = search_dir / MANIFEST_NAME
        if marker.exists() or marker.is_symlink():
            if marker.is_symlink() or not marker.is_file():
                raise ValueError("Runtime artifact manifest must be a regular file")
            metadata = artifact_metadata(search_dir)
            declared_artifact = Path(metadata["artifact_path"]).resolve()
            selected_artifact = model.with_suffix(".xml") if model.suffix.lower() == ".bin" else model
            if declared_artifact != selected_artifact:
                raise ValueError("Runtime artifact manifest does not describe the selected model")
            export_dir = Path(metadata["export_dir"]).resolve()
            if not model.is_relative_to(export_dir):
                raise ValueError("Runtime model escapes its declared export run")
            bounded: list[tuple[str, Path]] = []
            directory = model.parent
            candidate_depth = 0
            while True:
                calibration = directory / "calibration.json"
                if calibration.exists() and not calibration.resolve().is_relative_to(export_dir):
                    raise ValueError("Runtime calibration escapes its declared export run")
                bounded.append((f"{'../' * candidate_depth}calibration.json", calibration))
                if directory == export_dir:
                    break
                directory = directory.parent
                candidate_depth += 1
            return RuntimeCalibrationScope(
                tuple(bounded), (f"{'../' * depth}{MANIFEST_NAME}", marker),
                metadata.get("calibration_status") == "ok",
            )
        search_dir = search_dir.parent
    return RuntimeCalibrationScope(tuple(legacy))


def new_artifact_run(
    models_dir: Path, exports_dir: Path, camera_id: str, zone_id: str,
) -> ArtifactRun:
    """Reserve fresh directories; never reuse a previous training output."""
    camera_component = _directory_component(camera_id, "camera_id")
    zone_component = _directory_component(zone_id, "zone_id")
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ-") + uuid.uuid4().hex[:12]
    training_dir = (models_dir / camera_component / zone_component / run_id).resolve()
    export_dir = (exports_dir / camera_component / zone_component / run_id).resolve()
    training_dir.mkdir(parents=True, exist_ok=False)
    export_dir.mkdir(parents=True, exist_ok=False)
    return ArtifactRun(run_id, camera_id, zone_id, training_dir, export_dir)


def exported_artifact(root: Path, export_format: str, returned: object = None) -> Path:
    """Resolve the exporter's exact output and reject missing/ambiguous exports."""
    root = root.resolve()
    suffix = _SUFFIXES.get(export_format)
    if suffix is None:
        raise ValueError(f"Unsupported export format: {export_format}")
    if isinstance(returned, (str, os.PathLike)):
        candidate = Path(returned).resolve()
        if candidate.is_relative_to(root) and candidate.is_file() and candidate.suffix == suffix:
            return _validate_file(candidate)
    files = sorted(path.resolve() for path in root.rglob(f"*{suffix}") if path.is_file())
    canonical = [path for path in files if path.name == f"model{suffix}"]
    candidates = canonical or files
    if len(candidates) != 1:
        raise ValueError(f"Expected one {export_format} artifact in {root}, found {len(candidates)}")
    artifact = candidates[0]
    if not artifact.is_relative_to(root):
        raise ValueError("Export artifact escapes its run directory")
    return _validate_file(artifact)


def _validate_file(artifact: Path) -> Path:
    if artifact.stat().st_size == 0:
        raise ValueError(f"Empty deployment artifact: {artifact}")
    if artifact.suffix == ".xml":
        weights = artifact.with_suffix(".bin")
        if not weights.is_file() or weights.stat().st_size == 0:
            raise ValueError(f"OpenVINO weights missing or empty: {weights}")
    return artifact


def artifact_metadata(path: Path) -> dict:
    """Read a run manifest from an exact artifact or training directory."""
    candidate = (path if path.is_dir() else path.parent) / MANIFEST_NAME
    if not candidate.is_file():
        return {}
    value = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int:
        raise ValueError(f"Unsupported artifact manifest: {candidate}")
    if value["schema_version"] == 2:
        return _portable_artifact_metadata(path, candidate, value)
    if value["schema_version"] != 1:
        raise ValueError(f"Unsupported artifact manifest: {candidate}")
    locations = {}
    for key in ("artifact_path", "training_dir", "export_dir"):
        declared = value.get(key)
        if not isinstance(declared, str) or not Path(declared).is_absolute():
            raise ValueError(f"Invalid artifact manifest {key}: {candidate}")
        locations[key] = Path(declared).resolve()
    artifact = locations["artifact_path"]
    if not artifact.is_relative_to(locations["export_dir"]):
        raise ValueError("Artifact manifest file escapes its declared export run")
    run_id = value.get("run_id")
    for key in ("training_dir", "export_dir"):
        directory = locations[key]
        if (
            directory.name != run_id
            or directory.parent.name != _directory_component(value.get("zone_id", ""), "zone_id")
            or directory.parent.parent.name != _directory_component(value.get("camera_id", ""), "camera_id")
        ):
            raise ValueError("Artifact manifest run/camera/zone directory mismatch")
    actual = path.resolve()
    if path.is_dir():
        if actual not in {locations["training_dir"], locations["export_dir"], artifact.parent}:
            raise ValueError("Artifact manifest does not describe the selected directory")
    elif actual != artifact:
        raise ValueError("Artifact manifest does not describe the selected runtime file")
    return value


def _portable_location(root: Path, value: object) -> Path:
    """Resolve a canonical, package-local path without following symlinks."""
    if (
        not isinstance(value, str) or not value or "\\" in value
        or PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
        or ".." in PurePosixPath(value).parts
    ):
        raise ValueError("Portable artifact path must stay inside its bundle")
    location = root / value
    cursor = location
    while True:
        if cursor.is_symlink():
            raise ValueError("Portable artifact paths cannot contain symlinks")
        if cursor == root:
            break
        cursor = cursor.parent
    resolved = location.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Portable artifact path escapes its bundle")
    return resolved


def _portable_artifact_metadata(path: Path, candidate: Path, value: dict) -> dict:
    """Read package-only v2 metadata; v1 training manifests stay unchanged."""
    if value.get("kind") != "argus.portable-runtime.v1" or candidate.is_symlink():
        raise ValueError("Unsupported portable artifact manifest")
    root_steps = value.get("bundle_root")
    if not isinstance(root_steps, str) or not re.fullmatch(r"\.|\.\.(?:/\.\.)*", root_steps):
        raise ValueError("Invalid portable bundle root")
    raw_root = candidate.parent / root_steps
    root = raw_root.resolve()
    # The marker may be at the root, beside the artifact, or beside its source
    # checkpoint. No component from marker to bundle root may be a symlink.
    _portable_location(root, candidate.absolute().relative_to(root).as_posix())
    for key in ("camera_id", "zone_id"):
        _directory_component(value.get(key, ""), key)
    if not isinstance(value.get("run_id"), str) or not value["run_id"]:
        raise ValueError("Portable artifact is missing its run identity")
    if value.get("calibration_status") not in {"ok", "skipped"}:
        raise ValueError("Portable artifact has an invalid calibration state")
    locations = {
        key: _portable_location(root, value.get(key))
        for key in ("artifact_path", "training_dir", "export_dir")
    }
    artifact = locations["artifact_path"]
    if locations["export_dir"] != root or not artifact.is_file():
        raise ValueError("Portable artifact must be an existing bundle-local file")
    if candidate.parent.resolve() not in {root, locations["training_dir"], artifact.parent}:
        raise ValueError("Portable artifact marker is in an unexpected directory")
    actual = path.resolve()
    if actual not in {root, locations["training_dir"], artifact.parent, artifact}:
        raise ValueError("Portable manifest does not describe the selected artifact")
    if not path.is_dir() and actual != artifact:
        raise ValueError("Portable manifest does not describe the selected runtime file")
    checkpoint = value.get("checkpoint_path")
    if checkpoint is not None:
        checkpoint_path = _portable_location(root, checkpoint)
        if not checkpoint_path.is_file() or not checkpoint_path.is_relative_to(locations["training_dir"]):
            raise ValueError("Portable checkpoint is missing or outside its training directory")
    return {**value, **{key: str(location) for key, location in locations.items()}}


def write_portable_artifact_manifest(
    bundle_root: Path, artifact: Path, *, source_metadata: dict | None = None,
    camera_id: str, model_type: str | None = None,
) -> None:
    """Finalize a relocatable package with a new, independently hashed identity.

    Original release/scene evidence is not transferable: the portable marker
    is a different runtime artifact and must be registered and approved anew.
    """
    root = bundle_root.resolve()
    artifact = _validate_file(artifact.resolve())
    _portable_location(root, artifact.relative_to(root).as_posix())
    source = source_metadata or {}
    training_dir = root / "_training"
    training_dir.mkdir(exist_ok=True)
    metadata = {
        "schema_version": 2,
        "kind": "argus.portable-runtime.v1",
        "run_id": source.get("run_id") or f"package-{uuid.uuid4().hex}",
        "camera_id": camera_id,
        "zone_id": source.get("zone_id", "default"),
        "model_type": model_type or source.get("model_type"),
        "training_dir": "_training",
        "export_dir": ".",
        "artifact_path": artifact.relative_to(root).as_posix(),
        "checkpoint_path": "_training/source.ckpt" if (training_dir / "source.ckpt").is_file() else None,
        "calibration_status": source.get("calibration_status", "skipped"),
        "requires_release_review": True,
    }
    for directory in {root, training_dir, artifact.parent}:
        depth = len(directory.relative_to(root).parts)
        marker = {**metadata, "bundle_root": "/".join([".."] * depth) if depth else "."}
        (directory / MANIFEST_NAME).write_text(
            json.dumps(marker, indent=2, ensure_ascii=False), encoding="utf-8",
        )


def training_directory(path: Path) -> Path:
    """Accept new exact-file paths and legacy training-directory records."""
    metadata = artifact_metadata(path)
    if metadata.get("training_dir"):
        return Path(metadata["training_dir"])
    return path if path.is_dir() else path.parent


def inherited_baseline_quality(metadata: dict, *, source_model_path: str) -> dict:
    """Preserve source limitations, never claim a fresh input evaluation."""
    unavailable = {
        "report_status": "not_evaluated", "passed": None, "integrity_passed": None,
        "scope": "unrecorded_source_conditions", "warnings": [], "errors": [],
        "production_readiness": "not_evaluated",
    }
    value = metadata.get("baseline_quality")
    snapshot = dict(unavailable)
    if value is not None:
        try:
            if not isinstance(value, dict):
                raise ValueError("source quality must be a report")
            encoded = json.dumps(value, allow_nan=False)
            if len(encoded.encode("utf-8")) > 512 * 1024:
                raise ValueError("source quality report exceeds its budget")
            candidate = json.loads(encoded)
            if "passed" not in candidate or (type(candidate.get("passed")) is not bool and candidate.get("passed") is not None):
                raise ValueError("source quality has no literal decision")
            for key in ("warnings", "errors"):
                if not isinstance(candidate.get(key, []), list) or any(type(x) is not str for x in candidate.get(key, [])):
                    raise ValueError("source quality messages are invalid")
            if candidate.get("passed") is True and (
                candidate.get("errors") or candidate.get("integrity_passed") is False
                or (candidate.get("scope") == "observed_fixed_scene_only" and candidate.get("integrity_passed") is not True)
            ):
                raise ValueError("source quality evidence contradicts a passing decision")
            snapshot = candidate
        except (TypeError, ValueError, OverflowError):
            snapshot = {**unavailable, "report_status": "invalid", "errors": ["源制品质量报告无效，不能推断输入质量通过"]}
    snapshot["production_readiness"] = "not_evaluated"
    snapshot["evidence_origin"] = "inherited_source_conditions_not_reevaluated"
    snapshot["inherited_from"] = {"model_path": source_model_path, "run_id": metadata.get("run_id")}
    return snapshot


def write_artifact_manifest(
    run: ArtifactRun, artifact: Path, *, model_type: str,
    calibration_status: str, source_model_path: str | None = None,
    baseline_quality: dict | None = None,
) -> dict:
    """Finalize metadata only after calibration and all export sidecars exist."""
    artifact = _validate_file(artifact.resolve())
    if not artifact.is_relative_to(run.export_dir):
        raise ValueError("Deployment artifact must belong to the new export run")
    if calibration_status == "ok":
        calibration = artifact.parent / "calibration.json"
        if not calibration.is_file() or calibration.stat().st_size == 0:
            raise ValueError("Successful calibration must be present beside the deployment artifact")
    hashes: dict[str, str] = {}
    for file in sorted(run.export_dir.rglob("*")):
        if not file.is_file() or file.name == MANIFEST_NAME:
            continue
        if not file.resolve().is_relative_to(run.export_dir):
            raise ValueError("Deployment sidecar escapes its run directory")
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[file.relative_to(run.export_dir).as_posix()] = digest.hexdigest()
    quality_snapshot = None
    if baseline_quality is not None:
        if not isinstance(baseline_quality, dict):
            raise ValueError("baseline_quality must be a structured quality report")
        # A provenance snapshot, never a mutable reference or release flag.
        quality_snapshot = json.loads(json.dumps(baseline_quality, allow_nan=False))
    metadata = {
        "schema_version": 1,
        **({"baseline_quality": quality_snapshot} if quality_snapshot is not None else {}),
        "run_id": run.run_id,
        "camera_id": run.camera_id,
        "zone_id": run.zone_id,
        "model_type": model_type,
        "training_dir": str(run.training_dir),
        "export_dir": str(run.export_dir),
        "artifact_path": str(artifact),
        "calibration_status": calibration_status,
        "source_model_path": source_model_path,
        "files": hashes,
    }
    payload = json.dumps(metadata, indent=2, ensure_ascii=False)
    for directory in {run.training_dir, run.export_dir, artifact.parent}:
        target = directory / MANIFEST_NAME
        with target.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    return metadata
