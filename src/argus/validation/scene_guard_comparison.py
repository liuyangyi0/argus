"""Offline, diagnostic-only comparison of classical and learned scene geometry.

Run from the dedicated comparison CLI, never inside the production process.
The classical pass finishes before importing LightGlue: upstream imports change
global Torch settings. No camera runtime, model registry or release service is
created, and neither comparison result authorizes a model or changes a profile.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict, fields
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from argus.config.loader import load_config
from argus.integrations.contracts import file_sha256, write_json
from argus.preprocessing.scene_guard import SceneGuard, SceneGuardConfig, SceneGuardResult
from argus.preprocessing.scene_profile import (
    RUNTIME_ARTIFACT_HASH_SCHEME,
    SceneProfileStore,
    normalize_runtime_artifact_sha256,
)
from argus.validation.strict_video_decoder import StrictVideoDecoder


COMPARISON_SCHEMA = "argus.scene-guard-comparison.v1"


def _is_comparison_report(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("schema_version") == COMPARISON_SCHEMA
        and payload.get("diagnostic_only") is True
        and payload.get("gating_effect") == "none"
        and payload.get("release_authority") is False
    )


def _frame_sha256(frame: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(frame.dtype).encode("ascii"))
    digest.update(json.dumps(list(frame.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(frame).tobytes())
    return digest.hexdigest()


def _optional_file_sha256(path: Path) -> str | None:
    if not path.exists() and not path.is_symlink():
        return None
    return file_sha256(path)


def _profile_reference(camera: Any, profile_id: str) -> tuple[np.ndarray, dict[str, Any], list[Path]]:
    directory = Path(camera.scene_guard.profiles_dir).expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("scene profile root must already exist")
    # The constructor only mkdir(exist_ok=True)'s this already verified directory.
    store = SceneProfileStore(directory)
    # Use the store's path validation, but never create, approve or activate a
    # profile. Capture metadata BEFORE load() reads its approval/binding fields.
    store._validate_identifier(camera.camera_id, "camera_id")
    store._validate_identifier(profile_id, "profile_id")
    profile_directory = store._profile_dir(camera.camera_id, profile_id)
    profile_path = profile_directory / "metadata.json"
    active_path = profile_directory.parent / "active.json"
    metadata_hash = file_sha256(profile_path)
    active_hash = _optional_file_sha256(active_path)
    profile = store.load(camera.camera_id, profile_id)
    metadata = dict(profile.metadata)
    if metadata.get("operator_approved") is not True:
        raise ValueError("comparison requires an explicitly operator-approved scene profile")
    if metadata.get("validation_only", False) is not False:
        raise ValueError("validation-only profiles are not approved comparison references")
    version = metadata.get("model_version_id")
    if not isinstance(version, str) or not version or version != version.strip():
        raise ValueError("approved profile has no explicit model version binding")
    if metadata.get("model_artifact_hash_scheme") != RUNTIME_ARTIFACT_HASH_SCHEME:
        raise ValueError("profile model artifact hash scheme is missing or invalid")
    model_hash = normalize_runtime_artifact_sha256(metadata.get("model_artifact_sha256"))
    reference_bytes = profile.reference_path.read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != profile.reference_sha256:
        raise ValueError("approved reference changed while loading")
    frame = cv2.imdecode(np.frombuffer(reference_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.shape[:2] != (profile.image_height, profile.image_width):
        raise ValueError("approved reference dimensions do not match its profile")
    active_id = store.get_active_profile_id(camera.camera_id)
    if (
        file_sha256(profile_path) != metadata_hash
        or _optional_file_sha256(active_path) != active_hash
    ):
        raise ValueError("scene profile metadata or active pointer changed while loading")
    profile_hashes = {
        str(profile_path): metadata_hash,
        str(profile.reference_path): profile.reference_sha256,
        str(active_path): active_hash,
    }
    return frame, {
        "camera_id": profile.camera_id,
        "profile_id": profile.profile_id,
        "active_profile_id": active_id,
        "selected_profile_is_active": profile.profile_id == active_id,
        "reference_path": str(profile.reference_path),
        "reference_sha256": profile.reference_sha256,
        "reference_tensor_sha256": _frame_sha256(frame),
        "metadata_path": str(profile_path),
        "source_file_sha256": profile_hashes,
        "source_baseline_sha256": profile.source_baseline_sha256,
        "operator_approved": True,
        "model_version_id_declared": version,
        "model_artifact_sha256_declared": model_hash,
        "runtime_model_binding_verified": False,
    }, [profile_path, profile.reference_path, active_path]


def _make_shadow(reference: np.ndarray, config: SceneGuardConfig, **kwargs: Any) -> Any:
    # Optional deep-learning dependencies are imported only after the entire
    # classical pass. Importing this comparison module alone never imports Torch.
    from argus.preprocessing.lightglue_matcher import LightGlueMatcher, LightGlueMatcherConfig
    from argus.preprocessing.scene_guard_shadow import LightGlueSceneGuard

    matcher = LightGlueMatcher(LightGlueMatcherConfig(
        process_width=config.process_width, **kwargs,
    ))
    return LightGlueSceneGuard(reference.copy(), config=config, matcher=matcher)


def _decision(result: SceneGuardResult, elapsed_ms: float) -> dict[str, Any]:
    return {
        "status": "evaluated", "state": result.state.value,
        "raw_state": result.raw_state.value, "reason": result.reason,
        "metrics": result.metrics.to_dict(), "elapsed_ms": elapsed_ms,
        "matches_stable_pose": result.state.value == "stable",
        # Deliberately no `usable` field: standalone geometry is not the
        # production pipeline's profile/model/quick-sentinel safety decision.
    }


def _failed(error: str) -> dict[str, Any]:
    return {
        "status": "error", "state": "unknown", "raw_state": "unknown",
        "reason": error, "metrics": None, "elapsed_ms": None,
        "matches_stable_pose": False,
    }


def _timing(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "count": len(values), "mean_ms": sum(values) / len(values),
        "p50_ms": percentile(0.50), "p95_ms": percentile(0.95),
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [row for row in records if all(
        row[key]["status"] == "evaluated" for key in ("baseline", "lightglue")
    )]
    state_pairs = Counter(
        f"{row['baseline']['state']}|{row['lightglue']['state']}" for row in comparable
    )
    disagreements = [
        row["frame_index"] for row in comparable
        if row["baseline"]["state"] != row["lightglue"]["state"]
    ]
    raw_disagreements = [
        row["frame_index"] for row in comparable
        if row["baseline"]["raw_state"] != row["lightglue"]["raw_state"]
    ]
    return {
        "sampled_frames": len(records), "comparable_pairs": len(comparable),
        "state_pair_counts": dict(sorted(state_pairs.items())),
        "state_agreement_fraction": (
            (len(comparable) - len(disagreements)) / len(comparable) if comparable else None
        ),
        "disagreement_frame_indices": disagreements,
        "raw_state_disagreement_frame_indices": raw_disagreements,
        "baseline_errors": sum(row["baseline"]["status"] == "error" for row in records),
        "lightglue_errors": sum(row["lightglue"]["status"] == "error" for row in records),
        "baseline_timing": _timing([
            row["baseline"]["elapsed_ms"] for row in records
            if row["baseline"]["elapsed_ms"] is not None
        ]),
        "lightglue_timing": _timing([
            row["lightglue"]["elapsed_ms"] for row in records
            if row["lightglue"]["elapsed_ms"] is not None
        ]),
        "latency_scope": "wall-clock evaluate, including first-pair reference cache cold start",
        "pose_ground_truth_available": False,
        "accuracy_improvement_established": False,
        "release_evidence": False,
    }


def run_scene_guard_comparison(
    *, config_path: Path, camera_id: str, video_path: Path, profile_id: str,
    extractor_weights: Path, extractor_sha256: str, matcher_weights: Path,
    matcher_sha256: str, device: str = "cuda:0", max_keypoints: int = 2048,
    sample_every: int = 1, max_frames: int | None = None,
) -> dict[str, Any]:
    """Replay identical source frames in two diagnostic passes.

    Each decoder is consumed through verified EOF even if only a prefix is
    compared. Sampling changes this offline experiment's confirmation cadence;
    the report must never be interpreted as live pipeline gate equivalence.
    """
    if type(sample_every) is not int or sample_every < 1:
        raise ValueError("sample_every must be a positive integer")
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError("max_frames must be a positive integer")
    if type(max_keypoints) is not int or not 32 <= max_keypoints <= 8192:
        raise ValueError("max_keypoints must be in [32, 8192]")
    for digest in (extractor_sha256, matcher_sha256):
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError("local weight SHA256 must contain exactly 64 hex digits")
    extractor_sha256, matcher_sha256 = extractor_sha256.lower(), matcher_sha256.lower()
    inputs = {
        "config": Path(config_path).resolve(strict=True),
        "video": Path(video_path).resolve(strict=True),
        "extractor_weights": Path(extractor_weights).resolve(strict=True),
        "matcher_weights": Path(matcher_weights).resolve(strict=True),
    }
    if not all(path.is_file() for path in inputs.values()):
        raise ValueError("comparison inputs must be regular local files")
    hashes = {key: file_sha256(path) for key, path in inputs.items()}
    if hashes["extractor_weights"] != extractor_sha256:
        raise ValueError("ALIKED local weight SHA256 mismatch")
    if hashes["matcher_weights"] != matcher_sha256:
        raise ValueError("LightGlue local weight SHA256 mismatch")
    config = load_config(inputs["config"])
    cameras = [camera for camera in config.cameras if camera.camera_id == camera_id]
    if len(cameras) != 1:
        raise ValueError("camera_id must identify exactly one configured camera")
    camera = cameras[0]
    reference, reference_info, profile_paths = _profile_reference(camera, profile_id)
    # Record the active pointer if present, but protect it even when absent.
    protected_paths = [*inputs.values(), *profile_paths]
    protected_directories = sorted({
        str(Path(item.scene_guard.profiles_dir).expanduser().resolve())
        for item in config.cameras
    })
    profile_hashes = reference_info["source_file_sha256"]
    effective_json = config.model_dump(mode="json")
    effective_config_sha256 = hashlib.sha256(json.dumps(
        effective_json, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    if file_sha256(inputs["config"]) != hashes["config"]:
        raise ValueError("configuration changed while loading")
    algorithm_fields = {field.name for field in fields(SceneGuardConfig)}
    algorithm_config = SceneGuardConfig(**{
        key: value for key, value in camera.scene_guard.model_dump().items()
        if key in algorithm_fields
    })
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    baseline = None
    try:
        baseline = SceneGuard(reference.copy(), config=algorithm_config)
    except Exception as exc:
        errors.append({"stage": "baseline_initialization", "error": f"{type(exc).__name__}: {exc}"})
    decoder_evidence: dict[str, Any] = {}
    try:
        decoder = StrictVideoDecoder(inputs["video"], expected_sha256=hashes["video"])
        with decoder:
            first_pts: Fraction | None = None
            for item in decoder:
                stamp = item.timestamp
                exact = stamp.pts * Fraction(stamp.time_base)
                if first_pts is None:
                    first_pts = exact
                if max_frames is not None and item.frame_index >= max_frames:
                    continue
                if item.frame_index % sample_every:
                    continue
                result = _failed(errors[0]["error"]) if baseline is None else None
                if baseline is not None:
                    started = time.perf_counter()
                    try:
                        result = _decision(baseline.evaluate(item.frame.copy()),
                                           (time.perf_counter() - started) * 1000)
                    except Exception as exc:
                        result = _failed(f"{type(exc).__name__}: {exc}")
                        baseline.require_reconfirmation()
                records.append({
                    "frame_index": item.frame_index,
                    "pts": stamp.pts, "time_base": stamp.time_base,
                    "timestamp_sec": stamp.pts_time_seconds,
                    "relative_timestamp_sec": float(exact - first_pts),
                    "frame_sha256": _frame_sha256(item.frame),
                    "baseline": result, "lightglue": _failed("not evaluated"),
                })
        decoder_evidence["baseline"] = decoder.evidence()
    except Exception as exc:
        errors.append({"stage": "baseline_decode", "error": f"{type(exc).__name__}: {exc}"})
    shadow = None
    if "baseline" in decoder_evidence and records:
        try:
            shadow = _make_shadow(
                reference, algorithm_config,
                extractor_weights=inputs["extractor_weights"], extractor_sha256=extractor_sha256,
                matcher_weights=inputs["matcher_weights"], matcher_sha256=matcher_sha256,
                device=device, max_keypoints=max_keypoints,
            )
        except Exception as exc:
            errors.append({"stage": "lightglue_initialization", "error": f"{type(exc).__name__}: {exc}"})
            for row in records:
                row["lightglue"] = _failed(errors[-1]["error"])
    if shadow is not None:
        try:
            second_decoder = StrictVideoDecoder(inputs["video"], expected_sha256=hashes["video"])
            next_record = 0
            with second_decoder:
                for item in second_decoder:
                    if next_record >= len(records):
                        continue
                    row = records[next_record]
                    if item.frame_index != row["frame_index"]:
                        continue
                    next_record += 1
                    if (
                        _frame_sha256(item.frame) != row["frame_sha256"]
                        or item.timestamp.pts != row["pts"]
                        or item.timestamp.time_base != row["time_base"]
                    ):
                        raise ValueError("decoded source frame identity differs between comparison passes")
                    started = time.perf_counter()
                    try:
                        result = shadow.evaluate(item.frame.copy())
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        metadata = shadow.last_match_metadata
                        if "error_code" in metadata:
                            row["lightglue"] = _failed(result.reason)
                            row["lightglue"]["elapsed_ms"] = elapsed_ms
                        else:
                            row["lightglue"] = _decision(result, elapsed_ms)
                        row["lightglue"]["matcher"] = metadata
                    except Exception as exc:
                        row["lightglue"] = _failed(f"{type(exc).__name__}: {exc}")
                        shadow.require_reconfirmation()
            if next_record != len(records):
                raise ValueError("second comparison pass omitted source frames")
            decoder_evidence["lightglue"] = second_decoder.evidence()
        except Exception as exc:
            errors.append({"stage": "lightglue_decode", "error": f"{type(exc).__name__}: {exc}"})
    stable_inputs = True
    expected_files = {str(path): hashes[key] for key, path in inputs.items()}
    expected_files.update(profile_hashes)
    for name, expected in expected_files.items():
        try:
            actual = _optional_file_sha256(Path(name))
        except (OSError, ValueError) as exc:
            actual = None
            stable_inputs = False
            errors.append({"stage": "input_identity", "error": f"{name}: {type(exc).__name__}: {exc}"})
        if actual != expected:
            stable_inputs = False
    if not stable_inputs:
        errors.append({"stage": "input_identity", "error": "source inputs changed during comparison"})
    summary = _summary(records)
    if not records:
        errors.append({"stage": "comparison", "error": "no sampled frames were available"})
    truncated = bool(max_frames is not None and decoder_evidence.get("baseline", {}).get(
        "timestamp_frame_count", 0,
    ) > max_frames)
    status = "complete"
    if errors or summary["baseline_errors"] or summary["lightglue_errors"]:
        status = "partial" if summary["comparable_pairs"] else "error"
    elif truncated:
        status = "partial"
    return {
        "schema_version": COMPARISON_SCHEMA, "status": status,
        "diagnostic_only": True, "gating_effect": "none", "release_authority": False,
        "baseline_scope": "standalone_scene_guard_full_check",
        "production_pipeline_parity_claimed": False,
        "execution": "classical_pass_then_lightglue_pass_in_offline_command_process",
        "inputs": {key: {"path": str(path), "sha256": hashes[key]} for key, path in inputs.items()},
        "protected_paths": [str(path) for path in protected_paths],
        "protected_directories": protected_directories,
        "effective_config_sha256": effective_config_sha256,
        "reference": reference_info,
        "algorithm_thresholds": asdict(algorithm_config),
        "production_policy_snapshot": {
            "enabled": camera.scene_guard.enabled,
            "allow_alignable": camera.scene_guard.allow_alignable,
            "require_profile_model_binding": camera.scene_guard.require_profile_model_binding,
            "require_operator_approved_profile": camera.scene_guard.require_operator_approved_profile,
            "policy_applied_to_runtime": False,
        },
        "lightglue_options": {"device": device, "max_keypoints": max_keypoints, "extractor": "ALIKED"},
        "sample_every": sample_every, "max_frames": max_frames,
        "comparison_truncated": truncated, "source_inputs_stable": stable_inputs,
        "decoder_evidence": decoder_evidence, "errors": errors,
        "summary": summary, "frames": records,
        "metric_semantics": {
            "baseline_raw_matches": "number of classical KNN query groups",
            "lightglue_raw_matches": "number of pairs returned by the matcher before adapter filtering",
            "raw_match_counts_comparable": False,
            "pose_states_are_production_authorization": False,
        },
    }


def write_comparison_report(
    payload: dict[str, Any], output_path: str | Path, *, overwrite: bool = False,
) -> None:
    if not _is_comparison_report(payload):
        raise ValueError("expected a diagnostic-only SceneGuard comparison report")
    protected = payload.get("protected_paths")
    if not isinstance(protected, list) or not protected or any(
        not isinstance(path, str) or not path for path in protected
    ):
        raise ValueError("comparison report must identify all protected source paths")
    protected_directories = payload.get("protected_directories")
    if not isinstance(protected_directories, list) or not protected_directories or any(
        not isinstance(path, str) or not path for path in protected_directories
    ):
        raise ValueError("comparison report must identify all protected profile directories")
    output = Path(output_path)
    resolved = output.resolve()
    if output.is_symlink() or resolved in {Path(path).resolve() for path in protected}:
        raise ValueError("comparison report cannot replace a source input or symbolic link")
    if any(resolved.is_relative_to(Path(path).resolve()) for path in protected_directories):
        raise ValueError("comparison report cannot write inside a protected scene profile directory")
    if output.exists():
        for source in protected:
            if Path(source).exists() and output.samefile(source):
                raise ValueError("comparison report cannot replace a hard link to a source input")
        if not overwrite:
            raise FileExistsError(f"comparison report already exists: {output}")
        if not output.is_file():
            raise ValueError("comparison report output must be a regular file")
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise ValueError("overwrite is restricted to existing SceneGuard comparison reports") from exc
        if not _is_comparison_report(existing):
            raise ValueError("overwrite is restricted to existing SceneGuard comparison reports")
    write_json(output, payload, overwrite=overwrite)
