"""Persistent scene profiles for camera-position-aware detection.

A scene profile is deliberately created from an *existing baseline image*.
This module has no API that accepts a live ``numpy`` frame: deciding that a
new camera view is normal must be an explicit, audited action made elsewhere.

Storage layout::

    <root>/
        <camera_id>/
            active.json
            <profile_id>/
                reference.jpg
                metadata.json

Profile directories are immutable once published.  Creation happens in a
temporary sibling directory and is published with ``os.replace`` only after
both files have been flushed.  The active pointer is also replaced atomically.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from argus.anomaly.artifacts import (
    artifact_metadata,
    is_independent_artifact_run,
    runtime_calibration_scope,
)


_SCHEMA_VERSION = 1
_REFERENCE_FILENAME = "reference.jpg"
_METADATA_FILENAME = "metadata.json"
_ACTIVE_FILENAME = "active.json"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
RUNTIME_ARTIFACT_HASH_SCHEME = "argus-runtime-artifact-v1"
_RUNTIME_ARTIFACT_SUFFIXES = frozenset({".pt", ".pth", ".xml", ".bin", ".onnx"})
_RUNTIME_SIDECAR_FILENAMES = frozenset({"metadata.json", "calibration.json", "artifact.json"})


def runtime_artifact_sha256(model_path: str | Path) -> str:
    """Fingerprint the complete Argus runtime artifact set.

    This is deliberately not a bare file checksum.  The digest binds each
    runtime filename and size as well as its bytes; OpenVINO ``.xml`` files
    also bind the sibling ``.bin``.  Argus runtime sidecars are included too:
    ``metadata.json`` controls input shape and the first ``calibration.json``
    found by the detector changes score calibration.  Artifact directories
    bind every supported runtime file/sidecar in deterministic path order.
    """

    root_path = Path(model_path).expanduser()
    if not root_path.exists():
        raise FileNotFoundError(f"Runtime model artifact not found: {root_path}")
    entries: dict[str, Path] = {}
    artifact_found = False
    if root_path.is_dir():
        # New-run directories identify the same exact runtime artifact as
        # their registry file path; unrelated checkpoints are not runtime.
        metadata = artifact_metadata(root_path)
        if metadata:
            return runtime_artifact_sha256(Path(metadata["artifact_path"]))
        digest_root = root_path
        # Shared legacy roots can remain referenced by old registry rows.
        # Retraining now reserves independent child runs below them; neither
        # an in-progress run nor its finalized bundle belongs to the old model.
        for directory_name, child_dirs, filenames in os.walk(root_path):
            directory = Path(directory_name)
            child_dirs[:] = [
                name for name in child_dirs
                if not is_independent_artifact_run(directory / name)
            ]
            for filename in filenames:
                path = directory / filename
                if not path.is_file():
                    continue
                if (
                    path.suffix.lower() not in _RUNTIME_ARTIFACT_SUFFIXES
                    and path.name not in _RUNTIME_SIDECAR_FILENAMES
                ):
                    continue
                if path.suffix.lower() in _RUNTIME_ARTIFACT_SUFFIXES:
                    artifact_found = True
                entries[path.relative_to(digest_root).as_posix()] = path
        for path in entries.values():
            suffix = path.suffix.lower()
            if suffix == ".xml":
                companion = path.with_suffix(".bin")
            elif suffix == ".bin":
                companion = path.with_suffix(".xml")
            else:
                continue
            if not companion.is_file():
                raise FileNotFoundError(
                    f"OpenVINO runtime companion artifact not found: {companion}"
                )
    else:
        if root_path.suffix.lower() not in _RUNTIME_ARTIFACT_SUFFIXES:
            raise ValueError(
                f"Unsupported runtime model artifact: {root_path}"
            )
        artifact_found = True
        entries[root_path.name] = root_path
        suffix = root_path.suffix.lower()
        if suffix in {".xml", ".bin"}:
            companion = root_path.with_suffix(".bin" if suffix == ".xml" else ".xml")
            if not companion.is_file():
                raise FileNotFoundError(
                    f"OpenVINO runtime companion artifact not found: {companion}"
                )
            entries[companion.name] = companion

        metadata = root_path.parent / "metadata.json"
        if metadata.is_file():
            entries[metadata.name] = metadata

        # Loader and fingerprinting must select the same sidecar, including
        # the new-run export boundary and the unchanged legacy search policy.
        scope = runtime_calibration_scope(root_path)
        if scope.manifest is not None:
            label, marker = scope.manifest
            entries[label] = marker
        for label, calibration in scope.candidates:
            if calibration.is_file():
                entries[label] = calibration
                break
    if not artifact_found:
        raise ValueError(f"No supported runtime model artifacts found: {root_path}")

    digest = hashlib.sha256()
    for label, path in sorted(entries.items()):
        encoded_label = label.encode("utf-8")
        digest.update(len(encoded_label).to_bytes(4, "big"))
        digest.update(encoded_label)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def normalize_runtime_artifact_sha256(value: object) -> str:
    """Return a canonical runtime digest or reject malformed input."""

    if not isinstance(value, str) or not re.fullmatch(r"[0-9A-Fa-f]{64}", value):
        raise ValueError("runtime artifact SHA-256 must contain exactly 64 hex digits")
    return value.lower()


class SceneProfileError(RuntimeError):
    """Base error for invalid or corrupt scene-profile storage."""


class SceneProfileCorruptError(SceneProfileError):
    """Raised when a stored profile or active pointer fails validation."""


@dataclass(frozen=True, slots=True)
class SceneProfile:
    """Validated metadata for one immutable camera scene profile."""

    camera_id: str
    profile_id: str
    created_at: datetime
    reference_path: Path
    reference_sha256: str
    image_width: int
    image_height: int
    source_baseline_name: str
    source_baseline_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = _SCHEMA_VERSION


class SceneProfileStore:
    """Filesystem store for explicit, immutable camera scene profiles.

    Args:
        root: Directory that owns every camera/profile entry.
        jpeg_quality: Quality used when normalising a baseline to
            ``reference.jpg``.  The stored reference is hashed after encoding.

    ``create_from_baseline`` is intentionally the sole creation method.  It
    accepts only a path to an existing, decodable image and never adopts the
    current live camera frame automatically.
    """

    def __init__(self, root: str | Path, *, jpeg_quality: int = 95) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        root_path = Path(root).expanduser()
        root_path.mkdir(parents=True, exist_ok=True)
        self.root = root_path.resolve()
        self.jpeg_quality = jpeg_quality
        self._lock = threading.RLock()

    def create_from_baseline(
        self,
        camera_id: str,
        profile_id: str,
        baseline_image: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
        activate: bool = False,
    ) -> SceneProfile:
        """Create a profile from an explicitly selected baseline image.

        Existing profiles are immutable and cause ``FileExistsError``.  A
        ``numpy.ndarray`` (or any non-path input) is rejected so a live frame
        cannot accidentally become a trusted reference through this API.
        """

        camera_id = self._validate_identifier(camera_id, "camera_id")
        profile_id = self._validate_identifier(profile_id, "profile_id")
        if not isinstance(baseline_image, (str, Path)):
            raise TypeError("baseline_image must be an explicit filesystem path")

        source = Path(baseline_image).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Baseline image not found: {source}")
        try:
            source_bytes = source.read_bytes()
        except OSError as exc:
            raise SceneProfileError(f"Unable to read baseline image: {source}") from exc

        decoded = cv2.imdecode(np.frombuffer(source_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            raise ValueError(f"Baseline is not a decodable image: {source}")
        height, width = decoded.shape[:2]
        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            decoded,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not encoded_ok:
            raise SceneProfileError(f"Unable to encode baseline image: {source}")
        reference_bytes = encoded.tobytes()

        custom_metadata = dict(metadata or {})
        # Validate before touching storage and reject NaN/Infinity as non-JSON.
        try:
            json.dumps(custom_metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must contain only finite JSON values") from exc

        source_hash = hashlib.sha256(source_bytes).hexdigest()
        reference_hash = hashlib.sha256(reference_bytes).hexdigest()
        created_at = datetime.now(timezone.utc)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "camera_id": camera_id,
            "profile_id": profile_id,
            "created_at": created_at.isoformat(),
            "reference": {
                "filename": _REFERENCE_FILENAME,
                "sha256": reference_hash,
                "width": width,
                "height": height,
            },
            "source": {
                "kind": "existing_baseline",
                "filename": source.name,
                "sha256": source_hash,
            },
            "metadata": custom_metadata,
        }

        camera_dir = self._camera_dir(camera_id)
        profile_dir = self._profile_dir(camera_id, profile_id)
        with self._lock:
            camera_dir.mkdir(parents=True, exist_ok=True)
            self._assert_contained(camera_dir)
            if camera_dir.is_symlink():
                raise ValueError("camera_id resolves through a symbolic link")
            if profile_dir.exists() or profile_dir.is_symlink():
                raise FileExistsError(f"Scene profile already exists: {camera_id}/{profile_id}")

            staging = camera_dir / f".{profile_id}.{uuid.uuid4().hex}.tmp"
            self._assert_contained(staging)
            staging.mkdir()
            try:
                self._write_bytes(staging / _REFERENCE_FILENAME, reference_bytes)
                self._write_json(staging / _METADATA_FILENAME, payload)
                # Same-volume directory replacement publishes both files at once.
                os.replace(staging, profile_dir)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise

        profile = self.load(camera_id, profile_id)
        if activate:
            self.set_active(camera_id, profile_id)
        return profile

    def load(self, camera_id: str, profile_id: str) -> SceneProfile:
        """Load and validate a profile, including reference-file integrity."""

        camera_id = self._validate_identifier(camera_id, "camera_id")
        profile_id = self._validate_identifier(profile_id, "profile_id")
        profile_dir = self._profile_dir(camera_id, profile_id)
        self._reject_symlink(profile_dir, "profile directory")
        if not profile_dir.is_dir():
            raise FileNotFoundError(f"Scene profile not found: {camera_id}/{profile_id}")

        metadata_path = profile_dir / _METADATA_FILENAME
        reference_path = profile_dir / _REFERENCE_FILENAME
        payload = self._read_json(metadata_path, "profile metadata")

        try:
            schema_version = payload["schema_version"]
            stored_camera_id = payload["camera_id"]
            stored_profile_id = payload["profile_id"]
            created_at = datetime.fromisoformat(payload["created_at"])
            reference = payload["reference"]
            source = payload["source"]
            custom_metadata = payload.get("metadata", {})
        except (KeyError, TypeError, ValueError) as exc:
            raise SceneProfileCorruptError(
                f"Invalid metadata for scene profile {camera_id}/{profile_id}"
            ) from exc

        if schema_version != _SCHEMA_VERSION:
            raise SceneProfileCorruptError(
                f"Unsupported scene profile schema version: {schema_version!r}"
            )
        if stored_camera_id != camera_id or stored_profile_id != profile_id:
            raise SceneProfileCorruptError("Profile metadata identity does not match its path")
        if created_at.tzinfo is None:
            raise SceneProfileCorruptError("Profile created_at must include a timezone")
        if not isinstance(reference, dict) or reference.get("filename") != _REFERENCE_FILENAME:
            raise SceneProfileCorruptError("Profile reference filename is invalid")
        if not isinstance(source, dict) or source.get("kind") != "existing_baseline":
            raise SceneProfileCorruptError("Profile source is invalid")
        if not isinstance(custom_metadata, dict):
            raise SceneProfileCorruptError("Profile metadata field must be an object")

        self._reject_symlink(reference_path, "profile reference")
        if not reference_path.is_file():
            raise SceneProfileCorruptError("Profile reference image is missing")
        try:
            reference_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise SceneProfileCorruptError("Profile reference image cannot be read") from exc
        expected_hash = reference.get("sha256")
        if not isinstance(expected_hash, str) or reference_hash != expected_hash:
            raise SceneProfileCorruptError("Profile reference checksum does not match metadata")

        width = reference.get("width")
        height = reference.get("height")
        source_name = source.get("filename")
        source_hash = source.get("sha256")
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or width <= 0
            or not isinstance(height, int)
            or isinstance(height, bool)
            or height <= 0
            or not isinstance(source_name, str)
            or not source_name
            or Path(source_name).name != source_name
            or not self._is_sha256(source_hash)
        ):
            raise SceneProfileCorruptError("Profile image metadata is invalid")

        return SceneProfile(
            camera_id=camera_id,
            profile_id=profile_id,
            created_at=created_at,
            reference_path=reference_path,
            reference_sha256=reference_hash,
            image_width=width,
            image_height=height,
            source_baseline_name=source_name,
            source_baseline_sha256=source_hash,
            metadata=dict(custom_metadata),
            schema_version=schema_version,
        )

    def list_profiles(self, camera_id: str) -> list[SceneProfile]:
        """List valid profiles for a camera in profile-id order."""

        camera_id = self._validate_identifier(camera_id, "camera_id")
        camera_dir = self._camera_dir(camera_id)
        self._reject_symlink(camera_dir, "camera directory")
        if not camera_dir.is_dir():
            return []

        results: list[SceneProfile] = []
        for entry in sorted(camera_dir.iterdir(), key=lambda path: path.name):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            try:
                profile_id = self._validate_identifier(entry.name, "stored profile_id")
            except ValueError as exc:
                raise SceneProfileCorruptError(
                    f"Invalid profile directory under camera {camera_id}: {entry.name!r}"
                ) from exc
            results.append(self.load(camera_id, profile_id))
        return results

    def set_active(self, camera_id: str, profile_id: str) -> SceneProfile:
        """Atomically make an existing profile active for its camera."""

        camera_id = self._validate_identifier(camera_id, "camera_id")
        profile_id = self._validate_identifier(profile_id, "profile_id")
        profile = self.load(camera_id, profile_id)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "camera_id": camera_id,
            "profile_id": profile_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._write_json_atomic(self._camera_dir(camera_id) / _ACTIVE_FILENAME, payload)
        return profile

    def get_active_profile_id(self, camera_id: str) -> str | None:
        """Return the active profile ID, or ``None`` when no pointer exists."""

        camera_id = self._validate_identifier(camera_id, "camera_id")
        camera_dir = self._camera_dir(camera_id)
        self._reject_symlink(camera_dir, "camera directory")
        marker = camera_dir / _ACTIVE_FILENAME
        if not marker.exists():
            return None
        self._reject_symlink(marker, "active profile pointer")
        payload = self._read_json(marker, "active profile pointer")
        try:
            schema_version = payload["schema_version"]
            stored_camera_id = payload["camera_id"]
            profile_id = self._validate_identifier(payload["profile_id"], "active profile_id")
        except (KeyError, TypeError, ValueError) as exc:
            raise SceneProfileCorruptError("Active profile pointer is invalid") from exc
        if schema_version != _SCHEMA_VERSION or stored_camera_id != camera_id:
            raise SceneProfileCorruptError("Active profile pointer identity is invalid")
        # Fail closed if the pointer references a missing or corrupt profile.
        try:
            self.load(camera_id, profile_id)
        except (FileNotFoundError, SceneProfileCorruptError) as exc:
            raise SceneProfileCorruptError(
                f"Active scene profile is unavailable: {camera_id}/{profile_id}"
            ) from exc
        return profile_id

    def get_active(self, camera_id: str) -> SceneProfile | None:
        """Load the active profile for a camera, if one is configured."""

        profile_id = self.get_active_profile_id(camera_id)
        if profile_id is None:
            return None
        return self.load(camera_id, profile_id)

    def clear_active(self, camera_id: str) -> bool:
        """Remove a camera's active pointer without deleting any profile."""

        camera_id = self._validate_identifier(camera_id, "camera_id")
        marker = self._camera_dir(camera_id) / _ACTIVE_FILENAME
        self._reject_symlink(marker, "active profile pointer")
        with self._lock:
            try:
                marker.unlink()
            except FileNotFoundError:
                return False
        return True

    @staticmethod
    def _validate_identifier(value: str, label: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{label} must be a string")
        if value in {".", ".."} or ".." in value or not _IDENTIFIER_RE.fullmatch(value):
            raise ValueError(
                f"Invalid {label}: use 1-64 ASCII letters, digits, '.', '_' or '-' without '..'"
            )
        return value

    def _camera_dir(self, camera_id: str) -> Path:
        path = self.root / camera_id
        self._assert_contained(path)
        return path

    def _profile_dir(self, camera_id: str, profile_id: str) -> Path:
        path = self._camera_dir(camera_id) / profile_id
        self._assert_contained(path)
        return path

    def _assert_contained(self, path: Path) -> None:
        try:
            path.resolve(strict=False).relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Path escapes scene-profile root: {path}") from exc

    def _reject_symlink(self, path: Path, label: str) -> None:
        self._assert_contained(path)
        if path.is_symlink():
            raise ValueError(f"{label} must not be a symbolic link")

    @staticmethod
    def _write_bytes(path: Path, content: bytes) -> None:
        with path.open("xb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        SceneProfileStore._write_bytes(path, encoded + b"\n")

    def _write_json_atomic(self, path: Path, payload: Mapping[str, Any]) -> None:
        self._assert_contained(path)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            self._write_json(temporary, payload)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _read_json(path: Path, label: str) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SceneProfileCorruptError(f"Unable to read {label}: {path}") from exc
        if not isinstance(payload, dict):
            raise SceneProfileCorruptError(f"{label.capitalize()} must be a JSON object")
        return payload

    @staticmethod
    def _is_sha256(value: object) -> bool:
        return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


__all__ = [
    "RUNTIME_ARTIFACT_HASH_SCHEME",
    "SceneProfile",
    "SceneProfileCorruptError",
    "SceneProfileError",
    "SceneProfileStore",
    "normalize_runtime_artifact_sha256",
    "runtime_artifact_sha256",
]
