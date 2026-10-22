"""Canonical, fail-closed evidence contract for production model releases.

The regression report is intentionally not itself a release authorization.  A
release evidence document binds one report to one registry row and to the
immutable artifacts/configuration that were evaluated.  Promotion code stores
the canonical JSON and its SHA-256 in the same transaction as the stage change.

This module has no database dependency.  ``validate_release_evidence`` and
``build_release_evidence`` are pure functions so the contract can be exercised
without constructing a dashboard or a release pipeline.

The persisted digest detects mutation and closes workflow bypasses.  Soak
authorization additionally requires a protected-key HMAC attestation loaded
from outside the repository.  A hostile process that can write the database,
artifacts and protected key store remains outside this trust boundary.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


RELEASE_EVIDENCE_SCHEMA = "argus.release-evidence.v1"
SOAK_REPORT_SCHEMA = "argus.soak-report.v2"
SOAK_TRUST_STORE_SCHEMA = "argus.soak-trust-store.v1"
SOAK_ATTESTATION_ALGORITHM = "hmac-sha256"
SOAK_TRUST_STORE_ENV = "ARGUS_SOAK_TRUST_STORE"
EVIDENCE_ARTIFACT_HASH_SCHEME = "argus-evidence-artifact-v1"
SCENE_PROFILE_BASELINE_V2_SCHEMA = "argus.scene-profile-baseline.v2"
SCENE_PROFILE_BASELINE_V3_SCHEMA = "argus.scene-profile-baseline.v3"
SCENE_PROFILE_BASELINE_SCHEMAS = frozenset(
    {
        SCENE_PROFILE_BASELINE_V2_SCHEMA,
        SCENE_PROFILE_BASELINE_V3_SCHEMA,
    }
)
# Backwards-compatible public name used by older callers.  Production release
# authorization below deliberately requires V3 rather than this alias.
SCENE_PROFILE_BASELINE_SCHEMA = SCENE_PROFILE_BASELINE_V2_SCHEMA
SCENE_PROFILE_SCHEMA_VERSION = 1
STRICT_DECODER_EVIDENCE_SCHEMA = "argus.strict-video-decoder-evidence.v1"
STRICT_TIMESTAMP_EVIDENCE_SCHEMA = "argus.strict-video-frame-timestamps.v1"
RUNTIME_ARTIFACT_HASH_SCHEME = "argus-runtime-artifact-v1"
MIN_SCENE_GUARD_COVERAGE = 0.99
MAX_FALSE_POSITIVES_PER_HOUR = 2.0
MIN_STATIC_FOREIGN_RECALL = 0.95
MIN_EVENT_RECALL = 0.95
MIN_POSITIVE_ANNOTATION_COVERAGE = 0.99
MAX_P95_DELAY_SECONDS = 10.0
MIN_NORMAL_HOURS = 2.0
MIN_STATIC_FOREIGN_EVENTS = 20
MIN_SOAK_HOURS = 72.0
MAX_SOAK_SAMPLE_INTERVAL_SECONDS = 60.0

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm"}


@dataclass(frozen=True, slots=True)
class ReleaseEvidenceValidation:
    """Result of validating and canonicalizing one evidence document."""

    valid: bool
    reasons: tuple[str, ...]
    canonical_json: str | None
    sha256: str | None


def canonical_json(value: object) -> str:
    """Serialize JSON with one deterministic representation.

    NaN/Infinity and non-JSON values are rejected; accepting either would make
    a persisted digest ambiguous across JSON implementations.
    """

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: object) -> str:
    """Return the SHA-256 of :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def evidence_artifact_sha256(path: str | Path) -> str:
    """Fingerprint one evidence file or complete directory tree.

    Directory hashes bind every relative filename, size, and byte in sorted
    order.  This is the canonical scheme for config/profile/baseline/corpus
    bindings; callers must select immutable, narrowly scoped roots rather than
    a live directory that also contains logs or generated reports.
    """

    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Release evidence artifact not found: {source}")
    if source.is_symlink():
        raise ValueError(f"Release evidence artifact must not be a symlink: {source}")
    if source.is_file():
        files = [source]
        root = source.parent
    elif source.is_dir():
        entries = sorted(source.rglob("*"))
        symlinks = [item for item in entries if item.is_symlink()]
        if symlinks:
            raise ValueError(
                "Release evidence artifact tree contains a symlink: "
                f"{symlinks[0]}"
            )
        files = [item for item in entries if item.is_file()]
        root = source
    else:
        raise ValueError(f"Unsupported release evidence artifact: {source}")
    if not files:
        raise ValueError(f"Release evidence artifact tree is empty: {source}")

    digest = hashlib.sha256()
    digest.update(EVIDENCE_ARTIFACT_HASH_SCHEME.encode("ascii"))
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def canonical_training_artifact_root(path: str | Path) -> Path:
    """Return the registry hash root for legacy or manifest-backed baselines."""

    source = Path(path).expanduser()
    candidates = [source]
    if source.is_dir() and source.name == "images":
        candidates.insert(0, source.parent)
    for candidate in candidates:
        manifest_path = candidate / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if (
            isinstance(manifest, Mapping)
            and manifest.get("schema_version") in SCENE_PROFILE_BASELINE_SCHEMAS
        ):
            return candidate
    return source


def registry_training_data_hash(path: str | Path) -> str:
    """Reproduce the registry's legacy training-data fingerprint.

    Model registration persists this 16-hex digest.  Release construction uses
    the same algorithm to prove that the named baseline tree is the tree bound
    to the registry row, rather than an unrelated directory with a convenient
    manifest.  The stronger full-tree evidence digest is stored separately in
    the release document.
    """

    source = canonical_training_artifact_root(path)
    files = (
        [source]
        if source.is_file()
        else sorted(source.rglob("*"))
        if source.is_dir()
        else []
    )
    digest = hashlib.sha256()
    for item in files:
        if not item.is_file():
            continue
        with item.open("rb") as handle:
            while chunk := handle.read(65536):
                digest.update(chunk)
    return digest.hexdigest()[:16]


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _has_symlink_component(path: Path) -> bool:
    """Return whether any existing lexical path component is a symlink."""

    absolute = Path(os.path.abspath(path))
    parts = absolute.parts
    if not parts:
        return False
    current = Path(parts[0])
    if current.is_symlink():
        return True
    for part in parts[1:]:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _read_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must name a regular JSON file, not a symlink")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{label} JSON root must be an object")
    return parsed


def _contained_relative_file(
    root: Path,
    raw_relative: object,
    *,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(raw_relative, str) or not raw_relative.strip():
        raise ValueError(f"{label} must be a non-empty relative path")
    if raw_relative != raw_relative.strip():
        raise ValueError(f"{label} must not contain surrounding whitespace")
    if "\\" in raw_relative:
        raise ValueError(f"{label} must use a canonical POSIX relative path")
    pure_relative = PurePosixPath(raw_relative)
    if (
        pure_relative.is_absolute()
        or pure_relative.as_posix() != raw_relative
        or any(part in {"", ".", ".."} for part in pure_relative.parts)
    ):
        raise ValueError(f"{label} must stay inside the declared artifact root")
    candidate = root.joinpath(*pure_relative.parts)
    current = root
    for part in pure_relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        normalized = resolved.relative_to(root).as_posix()
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"{label} escapes or is missing from the declared artifact root"
        ) from exc
    if not resolved.is_file() or normalized != pure_relative.as_posix():
        raise ValueError(f"{label} must resolve to a regular contained file")
    return resolved, normalized


def _strict_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _strict_finite_number(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (minimum is not None and float(value) < minimum)
    ):
        qualifier = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{label} must be a finite number{qualifier}")
    return float(value)


def _canonical_time_base(value: object, *, label: str) -> tuple[str, Fraction]:
    if type(value) is not str or "/" not in value or value != value.strip():
        raise ValueError(f"{label} must be a canonical positive fraction")
    try:
        fraction = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{label} must be a canonical positive fraction") from exc
    canonical = f"{fraction.numerator}/{fraction.denominator}"
    if fraction <= 0 or canonical != value:
        raise ValueError(f"{label} must be a canonical positive fraction")
    return canonical, fraction


def _read_timestamp_jsonl(
    baseline_root: Path,
    evidence: object,
    *,
    source_index: int,
) -> list[Mapping[str, Any]]:
    label = (
        f"release baseline manifest sources[{source_index}]"
        ".frame_timestamp_evidence"
    )
    expected_keys = {
        "schema_version",
        "path",
        "sha256",
        "frame_count",
        "time_base",
        "record_format",
        "coordinates",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != expected_keys:
        raise ValueError(f"{label} has an unsupported shape")
    expected_path = f"decoder/source_{source_index:06d}.frames.jsonl"
    if (
        evidence.get("schema_version") != STRICT_TIMESTAMP_EVIDENCE_SCHEMA
        or evidence.get("path") != expected_path
        or evidence.get("record_format")
        != "one_json_object_per_decoded_frame"
        or evidence.get("coordinates")
        != "zero_based_decoded_presentation_order"
    ):
        raise ValueError(f"{label} contract is invalid")
    expected_sha256 = evidence.get("sha256")
    if type(expected_sha256) is not str or _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError(f"{label}.sha256 is invalid")
    expected_count = _strict_int(
        evidence.get("frame_count"),
        label=f"{label}.frame_count",
        minimum=1,
    )
    expected_time_base, time_base_fraction = _canonical_time_base(
        evidence.get("time_base"),
        label=f"{label}.time_base",
    )
    timestamp_path, normalized = _contained_relative_file(
        baseline_root,
        evidence.get("path"),
        label=f"{label}.path",
    )
    if normalized != expected_path or timestamp_path.suffix != ".jsonl":
        raise ValueError(f"{label}.path is not the canonical decoder inventory")
    try:
        raw = timestamp_path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{label} is not readable UTF-8 JSONL") from exc
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{label} checksum does not match")
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise ValueError(f"{label} must use canonical LF-terminated JSONL")
    lines = text.splitlines()
    if len(lines) != expected_count or any(not line for line in lines):
        raise ValueError(f"{label} frame count does not match JSONL records")

    record_keys = {
        "frame_index",
        "pts",
        "pts_time_seconds",
        "time_base",
        "duration_pts",
        "duration_seconds",
    }
    records: list[Mapping[str, Any]] = []
    previous_pts: int | None = None
    for position, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} record {position} is invalid JSON") from exc
        if not isinstance(record, Mapping) or set(record) != record_keys:
            raise ValueError(f"{label} record {position} has an unsupported shape")
        try:
            canonical_record = canonical_json(record)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} record {position} contains non-canonical values"
            ) from exc
        if line != canonical_record:
            raise ValueError(f"{label} record {position} is not canonical JSON")
        if record.get("frame_index") != position or type(record.get("frame_index")) is not int:
            raise ValueError(f"{label} frame indexes must be contiguous from zero")
        pts = record.get("pts")
        if type(pts) is not int or (previous_pts is not None and pts <= previous_pts):
            raise ValueError(f"{label} PTS values must be strictly increasing integers")
        previous_pts = pts
        if record.get("time_base") != expected_time_base:
            raise ValueError(f"{label} record time_base does not match its evidence")
        pts_time = _strict_finite_number(
            record.get("pts_time_seconds"),
            label=f"{label} record {position}.pts_time_seconds",
        )
        expected_pts_time = float(pts * time_base_fraction)
        tolerance = max(1e-12, abs(expected_pts_time) * 1e-12)
        if not math.isclose(pts_time, expected_pts_time, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(f"{label} record {position} PTS/time_base is inconsistent")
        duration_pts = record.get("duration_pts")
        duration_seconds = record.get("duration_seconds")
        if duration_pts is None:
            if duration_seconds is not None:
                raise ValueError(f"{label} record {position} duration is inconsistent")
        else:
            duration_pts = _strict_int(
                duration_pts,
                label=f"{label} record {position}.duration_pts",
            )
            duration_seconds_value = _strict_finite_number(
                duration_seconds,
                label=f"{label} record {position}.duration_seconds",
                minimum=0.0,
            )
            expected_duration = float(duration_pts * time_base_fraction)
            if not math.isclose(
                duration_seconds_value,
                expected_duration,
                rel_tol=0.0,
                abs_tol=max(1e-12, abs(expected_duration) * 1e-12),
            ):
                raise ValueError(f"{label} record {position} duration is inconsistent")
        records.append(record)
    return records


def _validate_strict_decoder_source(
    baseline_root: Path,
    source: Mapping[str, Any],
    *,
    source_index: int,
    source_sha256: str,
) -> list[Mapping[str, Any]]:
    label = f"release baseline manifest sources[{source_index}]"
    if (
        source.get("decode_backend") != "ffprobe+ffmpeg-rawvideo"
        or source.get("opencv_read_terminated") is not False
        or source.get("source_read_error_detectable") is not True
        or source.get("vfr_timestamps_preserved") is not True
        or source.get("frame_time_basis") != "pts_times_stream_time_base"
        or source.get("source_exhausted") is not True
        or source.get("reported_frame_count_matches") is not True
        or source.get("source_read_error") is not None
        or source.get("truncated_by_max_images") is not False
    ):
        raise ValueError(f"{label} strict decode state is incomplete")

    decoded_count = _strict_int(
        source.get("decoded_frame_count"),
        label=f"{label}.decoded_frame_count",
        minimum=1,
    )
    reported_count = _strict_int(
        source.get("reported_frame_count"),
        label=f"{label}.reported_frame_count",
        minimum=1,
    )
    if reported_count != decoded_count:
        raise ValueError(f"{label} decoded/reported frame counts disagree")
    fps = _strict_finite_number(
        source.get("fps"),
        label=f"{label}.fps",
        minimum=0.0,
    )
    if fps <= 0.0:
        raise ValueError(f"{label}.fps must be greater than zero")
    container_count = source.get("container_reported_frame_count")
    if container_count is not None:
        container_count = _strict_int(
            container_count,
            label=f"{label}.container_reported_frame_count",
        )
        if container_count != decoded_count:
            raise ValueError(f"{label} container frame count disagrees")

    strict = source.get("strict_decoder_evidence")
    expected_strict_keys = {
        "schema_version",
        "backend",
        "strict_release_decoder",
        "source_sha256_before",
        "source_sha256_after",
        "source_sha256_matches",
        "stream_index",
        "width",
        "height",
        "pixel_format",
        "time_base",
        "timestamp_frame_count",
        "pts_strictly_increasing",
        "variable_frame_rate_detected",
        "vfr_detection_basis",
        "distinct_pts_deltas",
        "nominal_frame_rate",
        "container_declared_frame_count",
        "container_declared_frame_count_matches",
        "ffprobe",
        "ffmpeg",
        "decode_error_distinguishable_from_clean_eof",
        "variable_frame_rate_timestamps_preserved",
        "complete_decode_verified",
        "source_exhausted",
        "truncated",
        "sufficient_for_model_release",
    }
    if not isinstance(strict, Mapping) or set(strict) != expected_strict_keys:
        raise ValueError(f"{label}.strict_decoder_evidence has an unsupported shape")
    if (
        strict.get("schema_version") != STRICT_DECODER_EVIDENCE_SCHEMA
        or strict.get("backend") != "ffprobe+ffmpeg-rawvideo"
        or strict.get("strict_release_decoder") is not True
        or strict.get("source_sha256_before") != source_sha256
        or strict.get("source_sha256_after") != source_sha256
        or strict.get("source_sha256_matches") is not True
        or strict.get("pixel_format") != "bgr24"
        or strict.get("pts_strictly_increasing") is not True
        or strict.get("vfr_detection_basis") != "distinct_adjacent_pts_deltas"
        or strict.get("container_declared_frame_count_matches") is not True
        or strict.get("decode_error_distinguishable_from_clean_eof") is not True
        or strict.get("variable_frame_rate_timestamps_preserved") is not True
        or strict.get("complete_decode_verified") is not True
        or strict.get("source_exhausted") is not True
        or strict.get("truncated") is not False
        or strict.get("sufficient_for_model_release") is not True
    ):
        raise ValueError(f"{label}.strict_decoder_evidence is not release-grade")
    _strict_int(strict.get("stream_index"), label=f"{label}.strict_decoder_evidence.stream_index")
    _strict_int(strict.get("width"), label=f"{label}.strict_decoder_evidence.width", minimum=1)
    _strict_int(strict.get("height"), label=f"{label}.strict_decoder_evidence.height", minimum=1)
    strict_count = _strict_int(
        strict.get("timestamp_frame_count"),
        label=f"{label}.strict_decoder_evidence.timestamp_frame_count",
        minimum=1,
    )
    if strict_count != decoded_count:
        raise ValueError(f"{label} strict timestamp frame count disagrees")
    strict_time_base, _ = _canonical_time_base(
        strict.get("time_base"),
        label=f"{label}.strict_decoder_evidence.time_base",
    )
    nominal_rate = _strict_finite_number(
        strict.get("nominal_frame_rate"),
        label=f"{label}.strict_decoder_evidence.nominal_frame_rate",
        minimum=0.0,
    )
    if nominal_rate <= 0.0 or not math.isclose(nominal_rate, fps, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label} nominal frame rate disagrees")
    strict_container_count = strict.get("container_declared_frame_count")
    if (
        strict_container_count != container_count
        or (
            container_count is not None
            and type(strict_container_count) is not int
        )
        or (container_count is None and strict_container_count is not None)
    ):
        raise ValueError(f"{label} container frame count evidence disagrees")

    ffprobe = strict.get("ffprobe")
    if (
        not isinstance(ffprobe, Mapping)
        or set(ffprobe) != {"returncode", "stderr_empty", "decoded_frame_inventory_count"}
        or ffprobe.get("returncode") != 0
        or type(ffprobe.get("returncode")) is not int
        or ffprobe.get("stderr_empty") is not True
        or ffprobe.get("decoded_frame_inventory_count") != decoded_count
        or type(ffprobe.get("decoded_frame_inventory_count")) is not int
    ):
        raise ValueError(f"{label} ffprobe audit is incomplete")
    ffmpeg = strict.get("ffmpeg")
    if (
        not isinstance(ffmpeg, Mapping)
        or set(ffmpeg)
        != {
            "returncode",
            "stderr_empty",
            "decoded_frame_count",
            "decoded_frame_count_matches_ffprobe",
            "partial_raw_frame_detected",
            "clean_eof",
            "xerror_enabled",
        }
        or ffmpeg.get("returncode") != 0
        or type(ffmpeg.get("returncode")) is not int
        or ffmpeg.get("stderr_empty") is not True
        or ffmpeg.get("decoded_frame_count") != decoded_count
        or type(ffmpeg.get("decoded_frame_count")) is not int
        or ffmpeg.get("decoded_frame_count_matches_ffprobe") is not True
        or ffmpeg.get("partial_raw_frame_detected") is not False
        or ffmpeg.get("clean_eof") is not True
        or ffmpeg.get("xerror_enabled") is not True
    ):
        raise ValueError(f"{label} ffmpeg audit is incomplete")

    records = _read_timestamp_jsonl(
        baseline_root,
        source.get("frame_timestamp_evidence"),
        source_index=source_index,
    )
    if len(records) != decoded_count:
        raise ValueError(f"{label} timestamp inventory count disagrees")
    timestamp_evidence = source["frame_timestamp_evidence"]
    if timestamp_evidence.get("time_base") != strict_time_base:
        raise ValueError(f"{label} timestamp/decoder time_base disagrees")
    distinct_deltas = sorted({
        int(records[index]["pts"]) - int(records[index - 1]["pts"])
        for index in range(1, len(records))
    })
    raw_deltas = strict.get("distinct_pts_deltas")
    expected_vfr = len(distinct_deltas) > 1
    if (
        not isinstance(raw_deltas, list)
        or any(type(value) is not int or value <= 0 for value in raw_deltas)
        or raw_deltas != distinct_deltas
        or type(strict.get("variable_frame_rate_detected")) is not bool
        or strict.get("variable_frame_rate_detected") != expected_vfr
    ):
        raise ValueError(f"{label} VFR/PTS delta evidence disagrees")
    return records


def _validate_baseline_manifest(
    baseline_root: Path,
    *,
    registry_data_hash: str,
) -> tuple[Mapping[str, Any], set[str]]:
    if not baseline_root.is_dir():
        raise ValueError("release_evidence_sources.baseline_path must be a directory")
    if registry_training_data_hash(baseline_root) != registry_data_hash:
        raise ValueError(
            "release baseline does not match the registry training data hash"
        )

    manifest = _read_json_mapping(
        baseline_root / "manifest.json",
        label="release baseline manifest.json",
    )
    schema_version = manifest.get("schema_version")
    if schema_version not in SCENE_PROFILE_BASELINE_SCHEMAS:
        raise ValueError(
            "release baseline manifest schema_version must be one of "
            f"{sorted(SCENE_PROFILE_BASELINE_SCHEMAS)!r}"
        )
    for field in (
        "training_ready",
        "corpus_evidence_complete",
        "source_inventory_evidence_complete",
        "annotation_exclusion_evidence_complete",
        "selection_evidence_complete",
    ):
        if manifest.get(field) is not True:
            raise ValueError(
                f"release baseline manifest {field} must be literal true"
            )
    if (
        manifest.get("validation_only") is not True
        or manifest.get("release_eligible") is not False
        or manifest.get("operator_approved") is not False
        or manifest.get("active_profile_modified") is not False
        or manifest.get("reference_auto_selected") is not False
        or manifest.get("dataset_status") != "offline_training_input_ready"
        or manifest.get("training_readiness_scope")
        != "offline_training_input_only"
    ):
        raise ValueError(
            "release baseline manifest offline authority/readiness scope is invalid"
        )
    readiness_checks = manifest.get("training_readiness_checks")
    expected_readiness_checks = {
        "corpus_evidence_complete",
        "at_least_two_declared_independent_sessions",
        "minimum_training_images_met",
        "minimum_calibration_images_met",
        "minimum_accepted_time_span_met",
        "full_source_grouping_has_no_diagnostic_samples",
    }
    if (
        not isinstance(readiness_checks, Mapping)
        or set(readiness_checks) != expected_readiness_checks
        or any(value is not True for value in readiness_checks.values())
    ):
        raise ValueError(
            "release baseline manifest training readiness checks are incomplete"
        )

    split_contract = manifest.get("split_contract")
    if not isinstance(split_contract, Mapping):
        raise ValueError("release baseline manifest split_contract must be an object")
    declared_sessions = split_contract.get("declared_session_ids")
    if (
        split_contract.get("random_frame_split_allowed") is not False
        or not isinstance(declared_sessions, list)
        or len(declared_sessions) < 2
        or len(set(declared_sessions)) != len(declared_sessions)
        or any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            for value in declared_sessions
        )
        or split_contract.get("declared_independent_session_count")
        != len(declared_sessions)
        or split_contract.get("minimum_distinct_sessions_for_training_ready") != 2
        or split_contract.get("single_session_policy") != "diagnostic_only"
    ):
        raise ValueError("release baseline manifest split evidence is incomplete")

    reference = manifest.get("reference")
    if not isinstance(reference, Mapping):
        raise ValueError("release baseline manifest reference must be an object")
    reference_file, _ = _contained_relative_file(
        baseline_root,
        reference.get("copied_path"),
        label="release baseline manifest reference.copied_path",
    )
    reference_sha256 = reference.get("sha256")
    if (
        not isinstance(reference_sha256, str)
        or _SHA256_RE.fullmatch(reference_sha256) is None
        or _stream_sha256(reference_file) != reference_sha256
    ):
        raise ValueError(
            "release baseline manifest reference checksum does not match"
        )

    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("release baseline manifest sources must be a non-empty list")
    training_source_hashes: set[str] = set()
    training_source_paths: set[str] = set()
    source_roles: dict[int, str] = {}
    source_entries: dict[int, Mapping[str, Any]] = {}
    strict_timestamps: dict[int, list[Mapping[str, Any]]] = {}
    calibration_session = max(declared_sessions)
    for index, item in enumerate(sources):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"release baseline manifest sources[{index}] must be an object"
            )
        source_hash = item.get("sha256")
        if not isinstance(source_hash, str) or _SHA256_RE.fullmatch(source_hash) is None:
            raise ValueError(
                f"release baseline manifest sources[{index}].sha256 is invalid"
            )
        if source_hash in training_source_hashes:
            raise ValueError("release baseline manifest has duplicate source hashes")
        training_source_hashes.add(source_hash)
        source_path = item.get("path")
        if (
            type(source_path) is not str
            or not source_path
            or source_path != source_path.strip()
            or "\x00" in source_path
            or source_path in training_source_paths
        ):
            raise ValueError(
                f"release baseline manifest sources[{index}].path is invalid"
            )
        training_source_paths.add(source_path)
        source_index = item.get("source_index")
        if (
            type(source_index) is not int
            or source_index != index
            or source_index in source_roles
        ):
            raise ValueError("release baseline manifest source indexes are invalid")
        review = item.get("annotation_review_evidence")
        if not isinstance(review, Mapping):
            raise ValueError(
                f"release baseline manifest sources[{index}] review is missing"
            )
        session_id = review.get("source_session_id")
        expected_role = (
            "calibration" if session_id == calibration_session else "train"
        )
        if (
            review.get("complete") is not True
            or review.get("reasons") != []
            or review.get("schema_version") != "argus.normal-video-review.v1"
            or review.get("source_video_sha256") != source_hash
            or review.get("source_video_sha256_matches") is not True
            or session_id not in declared_sessions
            or review.get("camera_id") != manifest.get("camera_id")
            or review.get("camera_id_matches") is not True
            or review.get("review_complete") is not True
            or not isinstance(review.get("reviewed_by"), str)
            or not review.get("reviewed_by")
            or review.get("reviewed_at_has_timezone") is not True
            or item.get("annotation_exclusions_auditable") is not True
            or item.get("source_exhausted") is not True
            or item.get("reported_frame_count_matches") is not True
            or item.get("source_read_error") is not None
            or item.get("truncated_by_max_images") is not False
            or item.get("split_role") != expected_role
        ):
            raise ValueError(
                f"release baseline manifest sources[{index}] review/decode "
                "evidence is incomplete"
            )
        annotation_sha256 = item.get("annotation_sha256")
        if (
            not isinstance(annotation_sha256, str)
            or _SHA256_RE.fullmatch(annotation_sha256) is None
        ):
            raise ValueError(
                f"release baseline manifest sources[{index}] annotation hash "
                "is invalid"
            )
        source_roles[source_index] = expected_role
        source_entries[source_index] = item
        if schema_version == SCENE_PROFILE_BASELINE_V2_SCHEMA:
            if (
                item.get("opencv_read_terminated") is not True
                or (
                    item.get("source_read_error_detectable") is not None
                    and item.get("source_read_error_detectable") is not False
                )
                or (
                    item.get("vfr_timestamps_preserved") is not None
                    and item.get("vfr_timestamps_preserved") is not False
                )
            ):
                raise ValueError(
                    f"release baseline manifest sources[{index}] v2 decode "
                    "evidence is invalid"
                )
        else:
            strict_timestamps[source_index] = _validate_strict_decoder_source(
                baseline_root,
                item,
                source_index=source_index,
                source_sha256=source_hash,
            )
    if set(source_roles) != set(range(len(sources))):
        raise ValueError(
            "release baseline manifest source indexes must be contiguous from zero"
        )

    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("release baseline manifest samples must be a non-empty list")
    strict_sample_frames: dict[int, list[int]] = {
        source_index: [] for source_index in source_roles
    }
    sample_files: set[Path] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(
                f"release baseline manifest samples[{index}] must be an object"
            )
        sample_file, _ = _contained_relative_file(
            baseline_root,
            sample.get("path"),
            label=f"release baseline manifest samples[{index}].path",
        )
        sample_relative = PurePosixPath(str(sample.get("path")))
        if (
            len(sample_relative.parts) != 2
            or sample_relative.parts[0] != "images"
            or sample_file.suffix.lower() not in {".png", ".jpg", ".jpeg"}
            or sample_file in sample_files
        ):
            raise ValueError(
                f"release baseline manifest samples[{index}] path is not a "
                "unique image artifact"
            )
        sample_files.add(sample_file)
        sample_hash = sample.get("sha256")
        if (
            not isinstance(sample_hash, str)
            or _SHA256_RE.fullmatch(sample_hash) is None
            or _stream_sha256(sample_file) != sample_hash
        ):
            raise ValueError(
                f"release baseline manifest samples[{index}] checksum does not match"
            )
        if sample.get("source_sha256") not in training_source_hashes:
            raise ValueError(
                f"release baseline manifest samples[{index}] source is not inventoried"
            )
        source_index = sample.get("source_index")
        if (
            type(source_index) is not int
            or source_index not in source_roles
            or sample.get("split_role") != source_roles[source_index]
        ):
            raise ValueError(
                f"release baseline manifest samples[{index}] split lineage is invalid"
            )
        if schema_version == SCENE_PROFILE_BASELINE_V3_SCHEMA:
            if sample.get("sample_index") != index or type(sample.get("sample_index")) is not int:
                raise ValueError(
                    "release baseline manifest sample indexes must be contiguous "
                    "from zero"
                )
            source = source_entries[source_index]
            if (
                sample.get("source_path") != source.get("path")
                or sample.get("source_sha256") != source.get("sha256")
            ):
                raise ValueError(
                    f"release baseline manifest samples[{index}] source identity "
                    "is inconsistent"
                )
            frame_index = sample.get("frame_index")
            records = strict_timestamps[source_index]
            if (
                type(frame_index) is not int
                or frame_index < 0
                or frame_index >= len(records)
                or frame_index in strict_sample_frames[source_index]
            ):
                raise ValueError(
                    f"release baseline manifest samples[{index}] frame index is invalid"
                )
            record = records[frame_index]
            sample_time = _strict_finite_number(
                sample.get("time_seconds"),
                label=f"release baseline manifest samples[{index}].time_seconds",
            )
            if (
                sample.get("pts") != record.get("pts")
                or type(sample.get("pts")) is not int
                or sample.get("time_base") != record.get("time_base")
                or not math.isclose(
                    sample_time,
                    float(record["pts_time_seconds"]),
                    rel_tol=0.0,
                    abs_tol=max(
                        1e-12,
                        abs(float(record["pts_time_seconds"])) * 1e-12,
                    ),
                )
            ):
                raise ValueError(
                    f"release baseline manifest samples[{index}] PTS timestamp "
                    "does not match decoder inventory"
                )
            strict_sample_frames[source_index].append(frame_index)

    images_root = baseline_root / "images"
    actual_image_entries = (
        set(images_root.rglob("*"))
        if images_root.is_dir() and not images_root.is_symlink()
        else set()
    )
    if actual_image_entries != sample_files:
        raise ValueError(
            "release baseline manifest samples do not exactly cover images directory"
        )

    source_counts = split_contract.get("source_counts")
    sample_counts = split_contract.get("sample_counts")
    actual_source_counts = {
        role: sum(value == role for value in source_roles.values())
        for role in ("train", "calibration", "diagnostic_only")
    }
    actual_sample_counts = {
        role: sum(sample.get("split_role") == role for sample in samples)
        for role in ("train", "calibration", "diagnostic_only")
    }
    if (
        source_counts != actual_source_counts
        or sample_counts != actual_sample_counts
        or actual_source_counts["train"] < 1
        or actual_source_counts["calibration"] < 1
        or actual_source_counts["diagnostic_only"] != 0
        or actual_sample_counts["train"] < 1
        or actual_sample_counts["calibration"] < 1
        or actual_sample_counts["diagnostic_only"] != 0
    ):
        raise ValueError("release baseline manifest split counts are inconsistent")

    decoder = manifest.get("decoder_evidence")
    if schema_version == SCENE_PROFILE_BASELINE_V2_SCHEMA:
        if (
            not isinstance(decoder, Mapping)
            or decoder.get("backend") != "opencv.VideoCapture"
            or decoder.get("reported_frame_count_required_to_match") is not True
            or decoder.get("sufficient_for_model_release") is not False
            or decoder.get("decode_error_distinguishable_from_clean_eof") is not False
            or decoder.get("variable_frame_rate_timestamps_preserved") is not False
        ):
            raise ValueError("release baseline manifest decoder evidence is invalid")
    else:
        decoder_root = baseline_root / "decoder"
        expected_decoder_files = {
            baseline_root / f"decoder/source_{source_index:06d}.frames.jsonl"
            for source_index in source_roles
        }
        actual_decoder_entries = (
            set(decoder_root.rglob("*"))
            if decoder_root.is_dir() and not decoder_root.is_symlink()
            else set()
        )
        if actual_decoder_entries != expected_decoder_files:
            raise ValueError(
                "release baseline manifest v3 decoder inventory does not "
                "exactly cover the decoder directory"
            )
        expected_decoder_keys = {
            "backend",
            "strict_release_decoder_requested",
            "all_sources_strictly_audited",
            "source_count",
            "read_termination_signal",
            "decoded_frame_count_cross_check",
            "source_sha256_checked_before_and_after_decode",
            "decode_error_distinguishable_from_clean_eof",
            "variable_frame_rate_timestamps_preserved",
            "time_basis",
            "timestamp_evidence_format",
            "sufficient_for_model_release",
        }
        if (
            not isinstance(decoder, Mapping)
            or set(decoder) != expected_decoder_keys
            or decoder.get("backend") != "ffprobe+ffmpeg-rawvideo"
            or decoder.get("strict_release_decoder_requested") is not True
            or decoder.get("all_sources_strictly_audited") is not True
            or decoder.get("source_count") != len(sources)
            or type(decoder.get("source_count")) is not int
            or decoder.get("read_termination_signal")
            != "ffmpeg_clean_zero_exit_after_complete_raw_frame_stream"
            or decoder.get("decoded_frame_count_cross_check")
            != "ffmpeg_raw_frames_equal_ffprobe_show_frames_inventory"
            or decoder.get("source_sha256_checked_before_and_after_decode") is not True
            or decoder.get("decode_error_distinguishable_from_clean_eof") is not True
            or decoder.get("variable_frame_rate_timestamps_preserved") is not True
            or decoder.get("time_basis") != "per_frame_pts_times_stream_time_base"
            or decoder.get("timestamp_evidence_format")
            != "sha256_bound_jsonl_one_record_per_decoded_frame"
            or decoder.get("sufficient_for_model_release") is not True
        ):
            raise ValueError(
                "release baseline manifest v3 decoder evidence is invalid"
            )
        for source_index, frame_indexes in strict_sample_frames.items():
            source = source_entries[source_index]
            if frame_indexes != sorted(frame_indexes):
                raise ValueError(
                    "release baseline manifest sample frame indexes must be ordered"
                )
            expected_first = frame_indexes[0] if frame_indexes else None
            expected_last = frame_indexes[-1] if frame_indexes else None
            expected_span = (
                float(strict_timestamps[source_index][expected_last]["pts_time_seconds"])
                - float(strict_timestamps[source_index][expected_first]["pts_time_seconds"])
                if expected_first is not None and expected_last is not None
                else 0.0
            )
            actual_span = _strict_finite_number(
                source.get("accepted_time_span_seconds"),
                label=(
                    "release baseline manifest sources"
                    f"[{source_index}].accepted_time_span_seconds"
                ),
                minimum=0.0,
            )
            if (
                source.get("accepted_images") != len(frame_indexes)
                or type(source.get("accepted_images")) is not int
                or source.get("first_accepted_frame_index") != expected_first
                or (
                    expected_first is not None
                    and type(source.get("first_accepted_frame_index")) is not int
                )
                or source.get("last_accepted_frame_index") != expected_last
                or (
                    expected_last is not None
                    and type(source.get("last_accepted_frame_index")) is not int
                )
                or not math.isclose(
                    actual_span,
                    expected_span,
                    rel_tol=0.0,
                    abs_tol=max(1e-12, abs(expected_span) * 1e-12),
                )
            ):
                raise ValueError(
                    f"release baseline manifest sources[{source_index}] accepted "
                    "sample counters are inconsistent"
                )
        selection = manifest.get("selection")
        expected_frames_read = sum(len(value) for value in strict_timestamps.values())
        if (
            not isinstance(selection, Mapping)
            or selection.get("total_frames_read") != expected_frames_read
            or type(selection.get("total_frames_read")) is not int
            or selection.get("accepted_images") != len(samples)
            or type(selection.get("accepted_images")) is not int
            or selection.get("truncated_by_max_images") is not False
        ):
            raise ValueError(
                "release baseline manifest v3 selection counts are inconsistent"
            )
    return manifest, training_source_hashes


def validate_baseline_manifest_source(
    baseline_root: str | Path,
    *,
    registry_data_hash: str | None = None,
) -> tuple[dict[str, Any], set[str], set[str]]:
    """Validate one V2/V3 baseline with the canonical release rules.

    This is the public, database-independent entry point shared by release
    construction and offline regression tooling.  Omitting
    ``registry_data_hash`` is appropriate for inspecting an artifact in
    isolation; production callers that bind a registry row must provide the
    persisted hash.  Passing V2 here only proves that the historical baseline
    is structurally valid.  Production authorization still requires V3 strict
    decoder evidence in :func:`build_release_evidence_from_sources`.

    The returned sets contain the immutable training source hashes and the
    operator-declared source session identities used for holdout checks.
    """

    root = canonical_training_artifact_root(baseline_root)
    expected_registry_hash = (
        registry_training_data_hash(root)
        if registry_data_hash is None
        else registry_data_hash
    )
    manifest, source_hashes = _validate_baseline_manifest(
        root,
        registry_data_hash=expected_registry_hash,
    )
    source_session_ids = {
        session_id
        for source in manifest.get("sources", [])
        if isinstance(source, Mapping)
        if isinstance(
            review := source.get("annotation_review_evidence"),
            Mapping,
        )
        if isinstance(session_id := review.get("source_session_id"), str)
    }
    return dict(manifest), set(source_hashes), source_session_ids


def _validate_scene_profile_source(
    profile_root: Path,
    baseline_root: Path,
    baseline_manifest: Mapping[str, Any],
    *,
    model_version_id: str,
    camera_id: str,
    runtime_artifact_digest: str,
) -> dict[str, str]:
    if not profile_root.is_dir():
        raise ValueError(
            "release_evidence_sources.scene_profile_path must be a profile directory"
        )
    metadata = _read_json_mapping(
        profile_root / "metadata.json",
        label="release scene profile metadata.json",
    )
    expected_metadata_keys = {
        "schema_version",
        "camera_id",
        "profile_id",
        "created_at",
        "reference",
        "source",
        "metadata",
    }
    if set(metadata) != expected_metadata_keys:
        raise ValueError("release scene profile metadata has an unsupported shape")
    profile_id = metadata.get("profile_id")
    if (
        metadata.get("schema_version") != SCENE_PROFILE_SCHEMA_VERSION
        or metadata.get("camera_id") != camera_id
        or not isinstance(profile_id, str)
        or not profile_id
        or profile_root.name != profile_id
    ):
        raise ValueError("release scene profile camera/profile identity is invalid")
    if (
        baseline_manifest.get("camera_id") != camera_id
        or baseline_manifest.get("scene_profile_id") != profile_id
    ):
        raise ValueError(
            "release baseline camera/profile lineage does not match active profile"
        )

    active = _read_json_mapping(
        profile_root.parent / "active.json",
        label="release scene profile active.json",
    )
    if set(active) != {"schema_version", "camera_id", "profile_id", "updated_at"}:
        raise ValueError("release scene profile active pointer has an unsupported shape")
    if (
        active.get("schema_version") != SCENE_PROFILE_SCHEMA_VERSION
        or active.get("camera_id") != camera_id
        or active.get("profile_id") != profile_id
    ):
        raise ValueError("release scene profile is not the camera's active profile")

    reference = metadata.get("reference")
    if not isinstance(reference, Mapping) or set(reference) != {
        "filename",
        "sha256",
        "width",
        "height",
    }:
        raise ValueError("release scene profile reference metadata is invalid")
    reference_file, _ = _contained_relative_file(
        profile_root,
        reference.get("filename"),
        label="release scene profile reference.filename",
    )
    reference_sha256 = reference.get("sha256")
    if (
        not isinstance(reference_sha256, str)
        or _SHA256_RE.fullmatch(reference_sha256) is None
        or _stream_sha256(reference_file) != reference_sha256
    ):
        raise ValueError("release scene profile reference checksum does not match")

    source = metadata.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "kind",
        "filename",
        "sha256",
    }:
        raise ValueError("release scene profile baseline source is invalid")
    if source.get("kind") != "existing_baseline":
        raise ValueError("release scene profile source must be an existing baseline")
    baseline_reference = baseline_manifest.get("reference")
    if not isinstance(baseline_reference, Mapping):
        raise ValueError("release baseline reference lineage is missing")
    baseline_file, baseline_relative = _contained_relative_file(
        baseline_root,
        baseline_reference.get("copied_path"),
        label="release baseline manifest reference.copied_path",
    )
    source_sha256 = source.get("sha256")
    if (
        source.get("filename") != Path(baseline_relative).name
        or baseline_reference.get("sha256") != source_sha256
        or not isinstance(source_sha256, str)
        or _SHA256_RE.fullmatch(source_sha256) is None
        or _stream_sha256(baseline_file) != source_sha256
    ):
        raise ValueError(
            "release scene profile source does not match the evaluated baseline"
        )

    custom = metadata.get("metadata")
    if not isinstance(custom, Mapping):
        raise ValueError("release scene profile custom metadata must be an object")
    if (
        custom.get("operator_approved") is not True
        or custom.get("validation_only") is not False
    ):
        raise ValueError("release scene profile lacks production operator approval")
    if (
        custom.get("model_version_id") != model_version_id
        or custom.get("model_artifact_sha256") != runtime_artifact_digest
        or custom.get("model_artifact_hash_scheme")
        != RUNTIME_ARTIFACT_HASH_SCHEME
    ):
        raise ValueError(
            "release scene profile model identity does not match current runtime"
        )
    return {
        "camera_id": camera_id,
        "profile_id": profile_id,
        "reference_sha256": reference_sha256,
        "source_baseline_sha256": source_sha256,
        "model_version_id": model_version_id,
        "model_artifact_sha256": runtime_artifact_digest,
    }


def _validate_regression_corpus(
    corpus_root: Path,
    suite: Mapping[str, Any],
    *,
    camera_id: str,
) -> tuple[set[str], set[str]]:
    if not corpus_root.is_dir():
        raise ValueError("release_evidence_sources.corpus_path must be a directory")
    manifest_path = corpus_root / "manifest.yaml"
    if not manifest_path.is_file():
        raise ValueError("release regression corpus must contain manifest.yaml")
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"release regression corpus manifest is invalid: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("release regression corpus manifest root must be an object")
    clips = manifest.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("release regression corpus manifest clips must be non-empty")

    inventory = suite.get("evaluated_source_inventory")
    if not isinstance(inventory, list) or len(inventory) != len(clips):
        raise ValueError(
            "regression report evaluated_source_inventory must cover every manifest clip"
        )
    inventory_by_id: dict[str, Mapping[str, Any]] = {}
    inventory_keys = {
        "clip_id",
        "camera_id",
        "source_session_id",
        "video_relative_path",
        "source_video_sha256",
        "annotation_relative_path",
        "annotation_sha256",
    }
    for index, item in enumerate(inventory):
        if not isinstance(item, Mapping) or set(item) != inventory_keys:
            raise ValueError(
                "regression report evaluated_source_inventory entries must "
                "use the canonical seven-field shape"
            )
        clip_id = item.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id or clip_id in inventory_by_id:
            raise ValueError("regression report source inventory clip IDs are invalid")
        inventory_by_id[clip_id] = item

    default_camera_id = manifest.get("default_camera_id")
    evaluated_hashes: set[str] = set()
    evaluated_sessions: set[str] = set()
    manifest_video_paths: set[Path] = set()
    seen_clip_ids: set[str] = set()
    for index, raw_clip in enumerate(clips):
        if not isinstance(raw_clip, Mapping):
            raise ValueError(f"release regression manifest clip {index} must be an object")
        clip_id = raw_clip.get("id")
        if not isinstance(clip_id, str) or not clip_id or clip_id in seen_clip_ids:
            raise ValueError("release regression manifest clip IDs are invalid")
        seen_clip_ids.add(clip_id)
        source_session_id = raw_clip.get("source_session_id")
        if (
            not isinstance(source_session_id, str)
            or not source_session_id
            or source_session_id != source_session_id.strip()
        ):
            raise ValueError(
                f"regression clip {clip_id} lacks immutable source_session_id"
            )
        if source_session_id in evaluated_sessions:
            raise ValueError(
                "release regression corpus contains duplicate source sessions"
            )
        evaluated_sessions.add(source_session_id)
        item = inventory_by_id.get(clip_id)
        if item is None:
            raise ValueError(f"regression source inventory is missing clip {clip_id}")
        manifest_camera = raw_clip.get("camera_id") or default_camera_id
        if manifest_camera != camera_id or item.get("camera_id") != camera_id:
            raise ValueError(f"regression clip {clip_id} camera does not match registry")
        if item.get("source_session_id") != source_session_id:
            raise ValueError(
                f"regression clip {clip_id} source session inventory does not match"
            )

        video_file, video_relative = _contained_relative_file(
            corpus_root,
            raw_clip.get("video"),
            label=f"release regression manifest clip {clip_id}.video",
        )
        manifest_video_paths.add(video_file)
        video_hash = _stream_sha256(video_file)
        if (
            item.get("video_relative_path") != video_relative
            or item.get("source_video_sha256") != video_hash
        ):
            raise ValueError(
                f"regression clip {clip_id} video inventory does not match corpus"
            )
        if video_hash in evaluated_hashes:
            raise ValueError("regression corpus contains duplicate source video content")
        evaluated_hashes.add(video_hash)

        raw_annotation = raw_clip.get("annotations")
        if raw_annotation in (None, ""):
            if (
                item.get("annotation_relative_path") is not None
                or item.get("annotation_sha256") is not None
            ):
                raise ValueError(
                    f"regression clip {clip_id} has unexpected annotation inventory"
                )
        else:
            annotation_file, annotation_relative = _contained_relative_file(
                corpus_root,
                raw_annotation,
                label=f"release regression manifest clip {clip_id}.annotations",
            )
            annotation_hash = _stream_sha256(annotation_file)
            if (
                item.get("annotation_relative_path") != annotation_relative
                or item.get("annotation_sha256") != annotation_hash
            ):
                raise ValueError(
                    f"regression clip {clip_id} annotation inventory does not match corpus"
                )

    discovered_videos = {
        item.resolve()
        for item in corpus_root.rglob("*")
        if item.is_file() and item.suffix.lower() in _VIDEO_SUFFIXES
    }
    if discovered_videos != manifest_video_paths:
        raise ValueError(
            "release regression corpus contains unregistered or missing video files"
        )
    source_inventory = suite.get("source_inventory")
    if source_inventory != {"source_video_sha256": sorted(evaluated_hashes)}:
        raise ValueError(
            "regression report source_inventory does not match evaluated corpus"
        )
    return evaluated_hashes, evaluated_sessions


def _aware_datetime(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _validate_soak_report_source(
    soak_report: Mapping[str, Any],
    *,
    expected_subject: Mapping[str, Any],
    telemetry_path: Path,
    trust_store_path: Path | None = None,
) -> tuple[bool, float]:
    """Validate auditable soak shape and its protected-key attestation."""

    expected_root = {
        "schema_version",
        "producer",
        "generated_at",
        "subject",
        "window",
        "telemetry",
        "result",
        "attestation",
    }
    if set(soak_report) != expected_root:
        raise ValueError("soak report v2 root shape is invalid")
    if soak_report.get("schema_version") != SOAK_REPORT_SCHEMA:
        raise ValueError(f"soak report schema_version must be {SOAK_REPORT_SCHEMA!r}")
    if soak_report.get("subject") != expected_subject:
        raise ValueError(
            "soak report subject does not match the registry, current runtime "
            "artifact, evaluated artifacts, and regression report"
        )

    producer = soak_report.get("producer")
    if not isinstance(producer, Mapping) or set(producer) != {
        "name",
        "version",
        "build_sha256",
    }:
        raise ValueError("soak report producer identity is invalid")
    if (
        not isinstance(producer.get("name"), str)
        or not producer.get("name")
        or not isinstance(producer.get("version"), str)
        or not producer.get("version")
        or not isinstance(producer.get("build_sha256"), str)
        or _SHA256_RE.fullmatch(producer["build_sha256"]) is None
    ):
        raise ValueError("soak report producer identity is incomplete")

    generated_at = _aware_datetime(
        soak_report.get("generated_at"),
        label="soak report generated_at",
    )
    window = soak_report.get("window")
    if not isinstance(window, Mapping) or set(window) != {
        "started_at",
        "ended_at",
        "duration_hours",
    }:
        raise ValueError("soak report window shape is invalid")
    started_at = _aware_datetime(
        window.get("started_at"),
        label="soak report window.started_at",
    )
    ended_at = _aware_datetime(
        window.get("ended_at"),
        label="soak report window.ended_at",
    )
    if ended_at <= started_at or generated_at < ended_at:
        raise ValueError("soak report timestamps are not monotonic")
    duration_hours = window.get("duration_hours")
    if isinstance(duration_hours, bool) or not isinstance(
        duration_hours,
        (int, float),
    ):
        raise ValueError("soak report duration_hours must be numeric")
    duration_hours = float(duration_hours)
    actual_duration = (ended_at - started_at).total_seconds() / 3600.0
    if (
        not math.isfinite(duration_hours)
        or abs(duration_hours - actual_duration) > (1.0 / 3600.0)
        or duration_hours < MIN_SOAK_HOURS
    ):
        raise ValueError("soak report duration does not match a >=72h window")

    telemetry = soak_report.get("telemetry")
    telemetry_keys = {
        "raw_telemetry_sha256",
        "sample_interval_seconds",
        "expected_sample_count",
        "observed_sample_count",
        "gap_count",
        "max_gap_seconds",
        "continuity_complete",
    }
    if not isinstance(telemetry, Mapping) or set(telemetry) != telemetry_keys:
        raise ValueError("soak report telemetry shape is invalid")
    if not telemetry_path.is_file():
        raise ValueError("soak raw telemetry source must be a file")
    if telemetry.get("raw_telemetry_sha256") != _stream_sha256(telemetry_path):
        raise ValueError("soak raw telemetry checksum does not match")
    interval = telemetry.get("sample_interval_seconds")
    expected_count = telemetry.get("expected_sample_count")
    observed_count = telemetry.get("observed_sample_count")
    max_gap = telemetry.get("max_gap_seconds")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not math.isfinite(float(interval))
        or float(interval) <= 0
        or float(interval) > MAX_SOAK_SAMPLE_INTERVAL_SECONDS
        or not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count <= 0
        or observed_count != expected_count
        or telemetry.get("gap_count") != 0
        or isinstance(max_gap, bool)
        or not isinstance(max_gap, (int, float))
        or not math.isfinite(float(max_gap))
        or float(max_gap) > float(interval)
        or telemetry.get("continuity_complete") is not True
        or expected_count < math.floor(actual_duration * 3600.0 / float(interval))
    ):
        raise ValueError("soak report continuity/sample evidence is incomplete")

    result = soak_report.get("result")
    if (
        not isinstance(result, Mapping)
        or set(result) != {"passed"}
        or result.get("passed") is not True
    ):
        raise ValueError("soak report result must be literal passed=true")
    attestation = soak_report.get("attestation")
    if not isinstance(attestation, Mapping) or set(attestation) != {
        "algorithm",
        "key_id",
        "signature",
    }:
        raise ValueError("soak report attestation shape is invalid")
    _verify_soak_attestation(
        soak_report,
        trust_store_path=trust_store_path,
    )
    return True, duration_hours


def soak_attestation_payload(soak_report: Mapping[str, Any]) -> bytes:
    """Canonical bytes signed by the trusted soak producer."""

    unsigned = dict(soak_report)
    unsigned.pop("attestation", None)
    return canonical_json(unsigned).encode("utf-8")


def sign_soak_report_hmac(
    soak_report: Mapping[str, Any],
    *,
    secret: bytes,
) -> str:
    """Return the URL-safe HMAC signature used by the v2 attestation."""

    if len(secret) < 32:
        raise ValueError("soak attestation key must contain at least 32 bytes")
    digest = hmac.new(
        secret,
        soak_attestation_payload(soak_report),
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _verify_soak_attestation(
    soak_report: Mapping[str, Any],
    *,
    trust_store_path: Path | None,
) -> None:
    raw_path = trust_store_path
    if raw_path is None:
        configured = os.environ.get(SOAK_TRUST_STORE_ENV, "").strip()
        if not configured:
            raise ValueError(
                f"trusted soak attestation unavailable: {SOAK_TRUST_STORE_ENV} "
                "is not configured; production release remains NO-GO"
            )
        raw_path = Path(configured)
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise ValueError("trusted soak trust store is unavailable") from exc
    if not path.is_file():
        raise ValueError("trusted soak trust store must be a regular file")

    # A shared-secret verifier is only meaningful when the key material is not
    # readable by other local users.  Windows ACLs do not map to POSIX mode
    # bits, so this fail-closed check is applied on POSIX deployments.
    if os.name == "posix":
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise ValueError(
                "trusted soak trust store permissions must not allow group/other access"
            )
        if hasattr(os, "geteuid") and path.stat().st_uid != os.geteuid():
            raise ValueError("trusted soak trust store must be owned by the service user")

    store = _read_json_mapping(path, label="trusted soak trust store")
    if set(store) != {"schema_version", "keys"}:
        raise ValueError("trusted soak trust store shape is invalid")
    if store.get("schema_version") != SOAK_TRUST_STORE_SCHEMA:
        raise ValueError("trusted soak trust store schema_version is unsupported")
    keys = store.get("keys")
    if not isinstance(keys, list) or not keys:
        raise ValueError("trusted soak trust store has no keys")

    attestation = soak_report.get("attestation")
    assert isinstance(attestation, Mapping)
    if attestation.get("algorithm") != SOAK_ATTESTATION_ALGORITHM:
        raise ValueError("soak report attestation algorithm is unsupported")
    key_id = attestation.get("key_id")
    if not isinstance(key_id, str) or not key_id:
        raise ValueError("soak report attestation key_id is invalid")
    matches = [
        item
        for item in keys
        if isinstance(item, Mapping) and item.get("key_id") == key_id
    ]
    if len(matches) != 1:
        raise ValueError("soak report attestation key is unknown or ambiguous")
    key = matches[0]
    if set(key) != {"key_id", "algorithm", "secret_base64", "enabled"}:
        raise ValueError("trusted soak key shape is invalid")
    if (
        key.get("enabled") is not True
        or key.get("algorithm") != SOAK_ATTESTATION_ALGORITHM
    ):
        raise ValueError("trusted soak key is disabled or uses another algorithm")
    encoded_secret = key.get("secret_base64")
    if not isinstance(encoded_secret, str) or not encoded_secret:
        raise ValueError("trusted soak key material is invalid")
    try:
        secret = base64.b64decode(encoded_secret, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("trusted soak key material is not valid base64") from exc
    expected_signature = sign_soak_report_hmac(soak_report, secret=secret)
    actual_signature = attestation.get("signature")
    if (
        not isinstance(actual_signature, str)
        or not hmac.compare_digest(actual_signature, expected_signature)
    ):
        raise ValueError("soak report attestation signature verification failed")


def _exact_mapping(
    value: object,
    *,
    path: str,
    keys: set[str],
    reasons: list[str],
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        reasons.append(f"{path} must be an object")
        return None
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys, key=lambda item: repr(item))
        if missing:
            reasons.append(f"{path} missing fields: {', '.join(missing)}")
        if extra:
            reasons.append(
                f"{path} has unsupported fields: "
                + ", ".join(
                    item if isinstance(item, str) else repr(item)
                    for item in extra
                )
            )
    return value


def _required_string(
    value: object,
    *,
    path: str,
    reasons: list[str],
) -> str | None:
    if not isinstance(value, str) or not value.strip():
        reasons.append(f"{path} must be a non-empty string")
        return None
    if value != value.strip():
        reasons.append(f"{path} must not contain surrounding whitespace")
        return None
    return value


def _required_sha256(
    value: object,
    *,
    path: str,
    reasons: list[str],
) -> str | None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        reasons.append(f"{path} must be exactly 64 lowercase hexadecimal digits")
        return None
    return value


def _required_true(
    value: object,
    *,
    path: str,
    reasons: list[str],
) -> None:
    # ``== True`` is intentionally forbidden: 1 and truthy strings are not
    # operator authorization or validation evidence.
    if type(value) is not bool or value is not True:
        reasons.append(f"{path} must be the literal boolean true")


def _finite_number(
    value: object,
    *,
    path: str,
    reasons: list[str],
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        reasons.append(f"{path} must be a finite number")
        return None
    number = float(value)
    if not math.isfinite(number):
        reasons.append(f"{path} must be a finite number")
        return None
    return number


def validate_release_evidence(
    evidence: object,
    *,
    model_version_id: str,
    camera_id: str,
    registry_model_hash: str,
    runtime_artifact_sha256: str,
) -> ReleaseEvidenceValidation:
    """Validate one production release-evidence document.

    Besides threshold checks, the document must bind exactly to the registry
    row and to the runtime artifact fingerprint recalculated from
    ``ModelRecord.model_path`` by the caller.  No root-level ``approved``
    shorthand or truthy substitute can satisfy this contract.
    """

    reasons: list[str] = []
    root_keys = {
        "schema_version",
        "subject",
        "artifacts",
        "operator_approval",
        "validation",
        "soak",
    }
    root = _exact_mapping(
        evidence,
        path="release_evidence",
        keys=root_keys,
        reasons=reasons,
    )
    if root is None:
        return ReleaseEvidenceValidation(False, tuple(reasons), None, None)

    if root.get("schema_version") != RELEASE_EVIDENCE_SCHEMA:
        reasons.append(
            "release_evidence.schema_version must be "
            f"{RELEASE_EVIDENCE_SCHEMA!r}"
        )

    subject = _exact_mapping(
        root.get("subject"),
        path="release_evidence.subject",
        keys={"model_version_id", "camera_id", "registry_model_hash"},
        reasons=reasons,
    )
    if subject is not None:
        supplied_version = _required_string(
            subject.get("model_version_id"),
            path="release_evidence.subject.model_version_id",
            reasons=reasons,
        )
        supplied_camera = _required_string(
            subject.get("camera_id"),
            path="release_evidence.subject.camera_id",
            reasons=reasons,
        )
        supplied_model_hash = _required_string(
            subject.get("registry_model_hash"),
            path="release_evidence.subject.registry_model_hash",
            reasons=reasons,
        )
        if supplied_version is not None and supplied_version != model_version_id:
            reasons.append("release evidence model_version_id does not match registry")
        if supplied_camera is not None and supplied_camera != camera_id:
            reasons.append("release evidence camera_id does not match registry")
        if (
            supplied_model_hash is not None
            and supplied_model_hash != registry_model_hash
        ):
            reasons.append("release evidence model hash does not match registry")

    artifacts = _exact_mapping(
        root.get("artifacts"),
        path="release_evidence.artifacts",
        keys={
            "runtime_artifact_sha256",
            "config_sha256",
            "scene_profile_sha256",
            "baseline_sha256",
            "corpus_sha256",
            "regression_report_sha256",
        },
        reasons=reasons,
    )
    if artifacts is not None:
        supplied_runtime_hash = _required_sha256(
            artifacts.get("runtime_artifact_sha256"),
            path="release_evidence.artifacts.runtime_artifact_sha256",
            reasons=reasons,
        )
        for field in (
            "config_sha256",
            "scene_profile_sha256",
            "baseline_sha256",
            "corpus_sha256",
            "regression_report_sha256",
        ):
            _required_sha256(
                artifacts.get(field),
                path=f"release_evidence.artifacts.{field}",
                reasons=reasons,
            )
        if (
            supplied_runtime_hash is not None
            and supplied_runtime_hash != runtime_artifact_sha256
        ):
            reasons.append(
                "release evidence runtime artifact digest does not match model_path"
            )

    approval = _exact_mapping(
        root.get("operator_approval"),
        path="release_evidence.operator_approval",
        keys={"operator_approved", "operator_id", "approved_at"},
        reasons=reasons,
    )
    if approval is not None:
        _required_true(
            approval.get("operator_approved"),
            path="release_evidence.operator_approval.operator_approved",
            reasons=reasons,
        )
        _required_string(
            approval.get("operator_id"),
            path="release_evidence.operator_approval.operator_id",
            reasons=reasons,
        )
        approved_at = _required_string(
            approval.get("approved_at"),
            path="release_evidence.operator_approval.approved_at",
            reasons=reasons,
        )
        if approved_at is not None:
            try:
                parsed = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
            except ValueError:
                reasons.append(
                    "release_evidence.operator_approval.approved_at must be ISO-8601"
                )
            else:
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    reasons.append(
                        "release_evidence.operator_approval.approved_at must include timezone"
                    )

    validation = _exact_mapping(
        root.get("validation"),
        path="release_evidence.validation",
        keys={
            "report_status_ok",
            "pipeline_evaluation",
            "production_parity",
            "trusted_scene_reference",
            "scene_profile_operator_approval",
            "runtime_model_identity",
            "positive_annotation_coverage",
            "candidate_confirmation",
            "corpus",
            "scene_guard",
            "metrics",
        },
        reasons=reasons,
    )
    if validation is not None:
        for field in (
            "report_status_ok",
            "pipeline_evaluation",
            "production_parity",
            "trusted_scene_reference",
            "scene_profile_operator_approval",
            "runtime_model_identity",
            "positive_annotation_coverage",
            "candidate_confirmation",
        ):
            _required_true(
                validation.get(field),
                path=f"release_evidence.validation.{field}",
                reasons=reasons,
            )

        corpus = _exact_mapping(
            validation.get("corpus"),
            path="release_evidence.validation.corpus",
            keys={
                "complete",
                "independent_holdout",
                "source_inventory_verified",
                "normal_hours",
                "static_foreign_event_count",
            },
            reasons=reasons,
        )
        if corpus is not None:
            for field in (
                "complete",
                "independent_holdout",
                "source_inventory_verified",
            ):
                _required_true(
                    corpus.get(field),
                    path=f"release_evidence.validation.corpus.{field}",
                    reasons=reasons,
                )
            normal_hours = _finite_number(
                corpus.get("normal_hours"),
                path="release_evidence.validation.corpus.normal_hours",
                reasons=reasons,
            )
            if normal_hours is not None and normal_hours < MIN_NORMAL_HOURS:
                reasons.append(
                    "release evidence normal corpus is too short: "
                    f"{normal_hours:.3f}h < {MIN_NORMAL_HOURS:.3f}h"
                )
            static_count = _finite_number(
                corpus.get("static_foreign_event_count"),
                path=(
                    "release_evidence.validation.corpus."
                    "static_foreign_event_count"
                ),
                reasons=reasons,
            )
            if static_count is not None:
                if not static_count.is_integer() or static_count < 0:
                    reasons.append(
                        "release_evidence.validation.corpus."
                        "static_foreign_event_count must be a non-negative integer"
                    )
                elif static_count < MIN_STATIC_FOREIGN_EVENTS:
                    reasons.append(
                        "release evidence static corpus is too small: "
                        f"{int(static_count)} < {MIN_STATIC_FOREIGN_EVENTS}"
                    )

        scene_guard = _exact_mapping(
            validation.get("scene_guard"),
            path="release_evidence.validation.scene_guard",
            keys={"evidence_complete", "coverage"},
            reasons=reasons,
        )
        if scene_guard is not None:
            _required_true(
                scene_guard.get("evidence_complete"),
                path="release_evidence.validation.scene_guard.evidence_complete",
                reasons=reasons,
            )
            coverage = _finite_number(
                scene_guard.get("coverage"),
                path="release_evidence.validation.scene_guard.coverage",
                reasons=reasons,
            )
            if coverage is not None and not (
                MIN_SCENE_GUARD_COVERAGE <= coverage <= 1.0
            ):
                reasons.append(
                    "release evidence SceneGuard coverage is below 0.99 or above 1.0"
                )

        metrics = _exact_mapping(
            validation.get("metrics"),
            path="release_evidence.validation.metrics",
            keys={
                "false_positive_events_per_hour",
                "event_recall",
                "static_foreign_recall",
                "positive_annotation_coverage",
                "delay_p95_sec",
            },
            reasons=reasons,
        )
        if metrics is not None:
            fp_per_hour = _finite_number(
                metrics.get("false_positive_events_per_hour"),
                path=(
                    "release_evidence.validation.metrics."
                    "false_positive_events_per_hour"
                ),
                reasons=reasons,
            )
            if fp_per_hour is not None and not (
                0.0 <= fp_per_hour <= MAX_FALSE_POSITIVES_PER_HOUR
            ):
                reasons.append("release evidence FP/hour must be between 0 and 2")
            event_recall = _finite_number(
                metrics.get("event_recall"),
                path="release_evidence.validation.metrics.event_recall",
                reasons=reasons,
            )
            if event_recall is not None and not (
                MIN_EVENT_RECALL <= event_recall <= 1.0
            ):
                reasons.append(
                    "release evidence overall event recall must be between 0.95 and 1"
                )
            static_recall = _finite_number(
                metrics.get("static_foreign_recall"),
                path="release_evidence.validation.metrics.static_foreign_recall",
                reasons=reasons,
            )
            if static_recall is not None and not (
                MIN_STATIC_FOREIGN_RECALL <= static_recall <= 1.0
            ):
                reasons.append(
                    "release evidence static-foreign recall must be between 0.95 and 1"
                )
            annotation_coverage = _finite_number(
                metrics.get("positive_annotation_coverage"),
                path=(
                    "release_evidence.validation.metrics."
                    "positive_annotation_coverage"
                ),
                reasons=reasons,
            )
            if annotation_coverage is not None and not (
                MIN_POSITIVE_ANNOTATION_COVERAGE <= annotation_coverage <= 1.0
            ):
                reasons.append(
                    "release evidence positive annotation coverage must be "
                    "between 0.99 and 1"
                )
            delay = _finite_number(
                metrics.get("delay_p95_sec"),
                path="release_evidence.validation.metrics.delay_p95_sec",
                reasons=reasons,
            )
            if delay is not None and not (0.0 <= delay <= MAX_P95_DELAY_SECONDS):
                reasons.append("release evidence P95 delay must be between 0 and 10s")

    soak = _exact_mapping(
        root.get("soak"),
        path="release_evidence.soak",
        keys={"passed", "duration_hours", "report_sha256"},
        reasons=reasons,
    )
    if soak is not None:
        _required_true(
            soak.get("passed"),
            path="release_evidence.soak.passed",
            reasons=reasons,
        )
        duration = _finite_number(
            soak.get("duration_hours"),
            path="release_evidence.soak.duration_hours",
            reasons=reasons,
        )
        if duration is not None and duration < MIN_SOAK_HOURS:
            reasons.append(
                f"release evidence soak is too short: {duration:.3f}h < 72h"
            )
        _required_sha256(
            soak.get("report_sha256"),
            path="release_evidence.soak.report_sha256",
            reasons=reasons,
        )

    serialized: str | None
    digest: str | None
    try:
        serialized = canonical_json(evidence)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    except (TypeError, ValueError) as exc:
        reasons.append(f"release evidence is not canonical JSON: {exc}")
        serialized = None
        digest = None
    return ReleaseEvidenceValidation(
        valid=not reasons,
        reasons=tuple(reasons),
        canonical_json=serialized,
        sha256=digest,
    )


def build_release_evidence(
    regression_report: Mapping[str, Any],
    *,
    model_version_id: str,
    camera_id: str,
    registry_model_hash: str,
    config_sha256: str,
    scene_profile_sha256: str,
    baseline_sha256: str,
    corpus_sha256: str,
    operator_id: str,
    operator_approved: bool,
    approved_at: str,
    soak_passed: bool,
    soak_duration_hours: float,
    soak_report_sha256: str,
) -> dict[str, Any]:
    """Build the v1 contract from a complete regression-suite report.

    The builder deliberately does not turn a report's root-level
    ``approved: true`` into authorization.  Every boolean is derived from the
    suite's detailed evidence-completeness and acceptance fields, except the
    separate release-operator approval supplied by an authenticated caller.
    Scene-profile approval and release approval are independent hard gates.
    Missing fields therefore produce a document that the validator rejects
    (NO-GO), instead of optimistically filling them.
    """

    suite_raw = regression_report.get("suite")
    suite = suite_raw if isinstance(suite_raw, Mapping) else {}
    acceptance_raw = suite.get("acceptance")
    acceptance = acceptance_raw if isinstance(acceptance_raw, Mapping) else {}

    identity_keys = suite.get("runtime_model_identity_keys")
    runtime_digest = ""
    if isinstance(identity_keys, list) and len(identity_keys) == 1:
        identity = identity_keys[0]
        if (
            isinstance(identity, Mapping)
            and identity.get("version_id") == model_version_id
            and isinstance(identity.get("artifact_sha256"), str)
        ):
            runtime_digest = identity["artifact_sha256"]

    release_operator_approved = (
        type(operator_approved) is bool and operator_approved is True
    )
    scene_profile_operator_approval = bool(
        acceptance.get("scene_profile_operator_approval") == "pass"
        and suite.get("scene_profile_operator_approval_evidence_complete") is True
    )
    production_parity = bool(
        acceptance.get("production_parity") == "pass"
        and suite.get("production_parity_evidence_complete") is True
    )
    trusted_scene_reference = bool(
        acceptance.get("trusted_scene_reference") == "pass"
        and suite.get("trusted_scene_reference_evidence_complete") is True
    )
    runtime_model_identity = bool(
        acceptance.get("runtime_model_identity") == "pass"
        and suite.get("runtime_model_identity_evidence_complete") is True
    )
    positive_annotation_coverage = bool(
        acceptance.get("positive_annotation_coverage") == "pass"
        and suite.get("positive_annotation_coverage_evidence_complete") is True
    )
    candidate_confirmation = bool(
        acceptance.get("candidate_confirmation") == "pass"
        and suite.get("candidate_contract_evidence_complete") is True
    )

    return {
        "schema_version": RELEASE_EVIDENCE_SCHEMA,
        "subject": {
            "model_version_id": model_version_id,
            "camera_id": camera_id,
            "registry_model_hash": registry_model_hash,
        },
        "artifacts": {
            "runtime_artifact_sha256": runtime_digest,
            "config_sha256": config_sha256,
            "scene_profile_sha256": scene_profile_sha256,
            "baseline_sha256": baseline_sha256,
            "corpus_sha256": corpus_sha256,
            "regression_report_sha256": canonical_sha256(regression_report),
        },
        "operator_approval": {
            # This is the authenticated release action, not the independent
            # approval embedded in the active scene profile.
            "operator_approved": release_operator_approved,
            "operator_id": operator_id,
            "approved_at": approved_at,
        },
        "validation": {
            "report_status_ok": regression_report.get("status") == "ok",
            "pipeline_evaluation": (
                regression_report.get("evaluation_mode") == "pipeline"
            ),
            "production_parity": production_parity,
            "trusted_scene_reference": trusted_scene_reference,
            "scene_profile_operator_approval": scene_profile_operator_approval,
            "runtime_model_identity": runtime_model_identity,
            "positive_annotation_coverage": positive_annotation_coverage,
            "candidate_confirmation": candidate_confirmation,
            "corpus": {
                "complete": bool(
                    acceptance.get("corpus_completeness") == "pass"
                    and suite.get("corpus_complete") is True
                ),
                "independent_holdout": bool(
                    acceptance.get("independent_holdout") == "pass"
                    and suite.get("independent_holdout_evidence_complete") is True
                ),
                "source_inventory_verified": (
                    suite.get("source_inventory_evidence_complete") is True
                ),
                "normal_hours": suite.get("normal_hours"),
                "static_foreign_event_count": suite.get(
                    "static_foreign_event_count"
                ),
            },
            "scene_guard": {
                "evidence_complete": (
                    suite.get("scene_guard_all_frame_evidence_complete") is True
                ),
                "coverage": suite.get(
                    "scene_guard_all_frame_usable_coverage_ratio"
                ),
            },
            "metrics": {
                "false_positive_events_per_hour": suite.get(
                    "false_positive_events_per_hour"
                ),
                "event_recall": suite.get("event_recall"),
                "static_foreign_recall": suite.get(
                    "static_foreign_event_recall"
                ),
                "positive_annotation_coverage": suite.get(
                    "positive_annotation_coverage_ratio"
                ),
                "delay_p95_sec": suite.get("delay_p95_sec"),
            },
        },
        "soak": {
            "passed": soak_passed,
            "duration_hours": soak_duration_hours,
            "report_sha256": soak_report_sha256,
        },
    }


def build_release_evidence_from_sources(
    sources: object,
    *,
    model_version_id: str,
    camera_id: str,
    registry_model_hash: str,
    registry_data_hash: str,
    runtime_artifact_digest: str,
    operator_id: str,
    operator_approved: bool,
    approved_at: str,
) -> dict[str, Any]:
    """Build evidence from server-read artifacts instead of client assertions.

    The HTTP boundary may name immutable evidence artifacts, but it must not be
    able to submit a ready-made bundle of booleans and digests.  This helper
    reads the regression and soak reports on the server and recalculates every
    non-runtime artifact digest from the named file/directory.  It additionally
    proves that the baseline is the registry-bound training tree, that the
    named scene profile is the camera's active model-bound profile, and that
    every evaluated video/annotation belongs to the corpus manifest.  Training
    and evaluation source-video hashes must be disjoint.

    This is an integrity improvement, not a cryptographic signature.  A caller
    that can also rewrite the server's evidence files is part of the separate
    signed-ledger trust boundary documented for P3.
    """

    expected = {
        "regression_report_path",
        "config_path",
        "scene_profile_path",
        "baseline_path",
        "corpus_path",
        "soak_report_path",
        "soak_telemetry_path",
    }
    if not isinstance(sources, Mapping):
        raise ValueError("release_evidence_sources must be an object")
    actual = set(sources)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected, key=lambda item: repr(item))
        details: list[str] = []
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        if extra:
            details.append(
                "unsupported fields: "
                + ", ".join(
                    item if isinstance(item, str) else repr(item)
                    for item in extra
                )
            )
        raise ValueError("release_evidence_sources " + "; ".join(details))

    paths: dict[str, Path] = {}
    for field in sorted(expected):
        raw_path = sources.get(field)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(
                f"release_evidence_sources.{field} must be a non-empty path"
            )
        if raw_path != raw_path.strip():
            raise ValueError(
                f"release_evidence_sources.{field} must not contain "
                "surrounding whitespace"
            )
        expanded = Path(raw_path).expanduser()
        if _has_symlink_component(expanded):
            raise ValueError(
                f"release_evidence_sources.{field} must not traverse a symlink"
            )
        paths[field] = expanded.resolve(strict=True)

    initial_source_digests = {
        field: evidence_artifact_sha256(path)
        for field, path in paths.items()
    }

    regression_report = _read_json_mapping(
        paths["regression_report_path"],
        label="release_evidence_sources.regression_report_path",
    )
    soak_report = _read_json_mapping(
        paths["soak_report_path"],
        label="release_evidence_sources.soak_report_path",
    )

    if not paths["config_path"].is_file():
        raise ValueError("release_evidence_sources.config_path must be a file")

    actual_digests = {
        "config_sha256": evidence_artifact_sha256(paths["config_path"]),
        "scene_profile_sha256": evidence_artifact_sha256(
            paths["scene_profile_path"]
        ),
        "baseline_sha256": evidence_artifact_sha256(paths["baseline_path"]),
        "corpus_sha256": evidence_artifact_sha256(paths["corpus_path"]),
    }
    suite = regression_report.get("suite")
    if not isinstance(suite, Mapping):
        raise ValueError("regression report suite must be an object")
    canonical_parameters = {
        "fp_gap_seconds": 60.0,
        "warmup_seconds": 30.0,
        "min_scene_guard_coverage": 0.99,
        "min_effective_eval_fps": 5.0,
        "min_projectile_eval_fps": 59.0,
    }
    for field, expected_value in canonical_parameters.items():
        value = regression_report.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) != expected_value
        ):
            raise ValueError(
                f"regression report canonical {field} must equal "
                f"{expected_value}"
            )
    if suite.get("kind_specific_sampling_evidence_complete") is not True:
        raise ValueError("regression kind-specific sampling evidence is incomplete")
    positive_by_kind = suite.get("positive_events_by_kind")
    recall_by_kind = suite.get("positive_event_recall_by_kind")
    for kind in ("static_foreign", "projectile"):
        if (
            not isinstance(positive_by_kind, Mapping)
            or not isinstance(positive_by_kind.get(kind), int)
            or isinstance(positive_by_kind.get(kind), bool)
            or positive_by_kind[kind] < MIN_STATIC_FOREIGN_EVENTS
            or not isinstance(recall_by_kind, Mapping)
            or isinstance(recall_by_kind.get(kind), bool)
            or not isinstance(recall_by_kind.get(kind), (int, float))
            or float(recall_by_kind[kind]) < MIN_EVENT_RECALL
        ):
            raise ValueError(
                f"regression report lacks canonical {kind} event evidence"
            )
    if suite.get("evaluated_artifact_binding_evidence_complete") is not True:
        raise ValueError(
            "regression report evaluated artifact binding is incomplete"
        )
    evaluated = suite.get("evaluated_artifacts")
    evaluated_keys = {"camera_id", *actual_digests}
    if not isinstance(evaluated, Mapping) or set(evaluated) != evaluated_keys:
        raise ValueError(
            "regression report suite.evaluated_artifacts must contain exactly "
            "camera_id, config_sha256, scene_profile_sha256, baseline_sha256, "
            "and corpus_sha256"
        )
    if evaluated.get("camera_id") != camera_id:
        raise ValueError(
            "regression report evaluated camera_id does not match registry"
        )
    for field, actual_digest in actual_digests.items():
        if evaluated.get(field) != actual_digest:
            raise ValueError(
                f"regression report {field} does not match server-side source"
            )

    baseline_manifest, training_source_hashes = _validate_baseline_manifest(
        paths["baseline_path"],
        registry_data_hash=registry_data_hash,
    )
    scene_profile_identity = _validate_scene_profile_source(
        paths["scene_profile_path"],
        paths["baseline_path"],
        baseline_manifest,
        model_version_id=model_version_id,
        camera_id=camera_id,
        runtime_artifact_digest=runtime_artifact_digest,
    )
    evaluated_scene_profile = suite.get("evaluated_scene_profile")
    if (
        not isinstance(evaluated_scene_profile, Mapping)
        or set(evaluated_scene_profile) != set(scene_profile_identity)
        or dict(evaluated_scene_profile) != scene_profile_identity
    ):
        raise ValueError(
            "regression report evaluated scene profile does not match the "
            "active server profile"
        )

    evaluated_source_hashes, evaluated_sessions = _validate_regression_corpus(
        paths["corpus_path"],
        suite,
        camera_id=camera_id,
    )
    overlap = sorted(training_source_hashes.intersection(evaluated_source_hashes))
    if overlap:
        raise ValueError(
            "regression corpus is not an independent holdout: training and "
            "evaluation source hashes overlap: " + ", ".join(overlap)
        )
    training_sessions = {
        review.get("source_session_id")
        for source in baseline_manifest.get("sources", [])
        if isinstance(source, Mapping)
        if isinstance(
            review := source.get("annotation_review_evidence"),
            Mapping,
        )
        if isinstance(review.get("source_session_id"), str)
    }
    session_overlap = sorted(training_sessions.intersection(evaluated_sessions))
    if session_overlap:
        raise ValueError(
            "regression corpus is not an independent holdout: training and "
            "evaluation source sessions overlap: " + ", ".join(session_overlap)
        )
    decoder_evidence = baseline_manifest.get("decoder_evidence")
    if (
        baseline_manifest.get("schema_version")
        != SCENE_PROFILE_BASELINE_V3_SCHEMA
        or not isinstance(decoder_evidence, Mapping)
        or decoder_evidence.get("sufficient_for_model_release") is not True
    ):
        raise ValueError(
            "release baseline decoder evidence is insufficient for model release; "
            "production requires argus.scene-profile-baseline.v3 strict evidence"
        )

    identities = suite.get("runtime_model_identity_keys")
    expected_identity = {
        "version_id": model_version_id,
        "artifact_sha256": runtime_artifact_digest,
    }
    if identities != [expected_identity]:
        raise ValueError(
            "regression report must contain exactly one runtime model identity "
            "matching the registry version and current artifact"
        )

    regression_digest = canonical_sha256(regression_report)
    expected_soak_subject = {
        "model_version_id": model_version_id,
        "camera_id": camera_id,
        "registry_model_hash": registry_model_hash,
        "runtime_artifact_sha256": runtime_artifact_digest,
        **actual_digests,
        "regression_report_sha256": regression_digest,
    }
    soak_passed, soak_duration = _validate_soak_report_source(
        soak_report,
        expected_subject=expected_soak_subject,
        telemetry_path=paths["soak_telemetry_path"],
    )
    final_source_digests = {
        field: evidence_artifact_sha256(path)
        for field, path in paths.items()
    }
    if final_source_digests != initial_source_digests:
        changed = sorted(
            field
            for field, digest in initial_source_digests.items()
            if final_source_digests.get(field) != digest
        )
        raise ValueError(
            "release evidence sources changed during validation: "
            + ", ".join(changed)
        )

    return build_release_evidence(
        regression_report,
        model_version_id=model_version_id,
        camera_id=camera_id,
        registry_model_hash=registry_model_hash,
        config_sha256=actual_digests["config_sha256"],
        scene_profile_sha256=actual_digests["scene_profile_sha256"],
        baseline_sha256=actual_digests["baseline_sha256"],
        corpus_sha256=actual_digests["corpus_sha256"],
        operator_id=operator_id,
        operator_approved=operator_approved,
        approved_at=approved_at,
        soak_passed=soak_passed,
        soak_duration_hours=soak_duration,
        soak_report_sha256=evidence_artifact_sha256(paths["soak_report_path"]),
    )


__all__ = [
    "EVIDENCE_ARTIFACT_HASH_SCHEME",
    "MAX_FALSE_POSITIVES_PER_HOUR",
    "MAX_P95_DELAY_SECONDS",
    "MAX_SOAK_SAMPLE_INTERVAL_SECONDS",
    "MIN_EVENT_RECALL",
    "MIN_NORMAL_HOURS",
    "MIN_POSITIVE_ANNOTATION_COVERAGE",
    "MIN_SCENE_GUARD_COVERAGE",
    "MIN_SOAK_HOURS",
    "MIN_STATIC_FOREIGN_EVENTS",
    "MIN_STATIC_FOREIGN_RECALL",
    "RELEASE_EVIDENCE_SCHEMA",
    "SCENE_PROFILE_BASELINE_V2_SCHEMA",
    "SCENE_PROFILE_BASELINE_V3_SCHEMA",
    "SOAK_REPORT_SCHEMA",
    "ReleaseEvidenceValidation",
    "build_release_evidence",
    "build_release_evidence_from_sources",
    "canonical_training_artifact_root",
    "canonical_json",
    "canonical_sha256",
    "evidence_artifact_sha256",
    "registry_training_data_hash",
    "validate_baseline_manifest_source",
    "validate_release_evidence",
]
