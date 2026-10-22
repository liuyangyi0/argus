#!/usr/bin/env python3
"""Create and optionally activate an immutable camera scene profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from argus.preprocessing.scene_profile import (
        RUNTIME_ARTIFACT_HASH_SCHEME,
        SceneProfileStore,
        normalize_runtime_artifact_sha256,
        runtime_artifact_sha256,
    )
except ModuleNotFoundError:  # Direct execution from a source checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from argus.preprocessing.scene_profile import (
        RUNTIME_ARTIFACT_HASH_SCHEME,
        SceneProfileStore,
        normalize_runtime_artifact_sha256,
        runtime_artifact_sha256,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a trusted scene profile from an explicitly selected "
            "normal baseline image. Live frames are intentionally unsupported."
        ),
    )
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--profiles-dir", type=Path, default=Path("data/scene_profiles"))
    parser.add_argument(
        "--model-version-id",
        required=True,
        help="Exact runtime model version this approved camera pose is allowed to use",
    )
    artifact = parser.add_mutually_exclusive_group(required=True)
    artifact.add_argument(
        "--model-artifact",
        type=Path,
        help=(
            "Runtime artifact file/directory to fingerprint with the complete "
            "Argus runtime-artifact-v1 algorithm"
        ),
    )
    artifact.add_argument(
        "--model-artifact-sha256",
        help=(
            "Precomputed 64-hex Argus runtime-artifact-v1 digest; this is not "
            "the bare SHA-256 of model.pt/model.xml"
        ),
    )
    parser.add_argument(
        "--metadata-json",
        default="{}",
        help=(
            "Optional JSON object, for example sensitive_rois or audit notes; "
            "operator approval requires --operator-approved"
        ),
    )
    parser.add_argument(
        "--operator-approved",
        action="store_true",
        help=(
            "Record explicit human production approval; use --validation-only "
            "instead for non-production evidence."
        ),
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Explicitly mark this profile as non-production validation evidence",
    )
    parser.add_argument("--activate", action="store_true")
    parser.add_argument(
        "--allow-validation-only-activation",
        action="store_true",
        help=(
            "Allow --activate for an explicitly --validation-only profile. "
            "Production activation never needs this flag."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        metadata = json.loads(args.metadata_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--metadata-json is invalid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise SystemExit("--metadata-json must decode to a JSON object")
    if (
        not args.model_version_id.strip()
        or args.model_version_id != args.model_version_id.strip()
    ):
        raise SystemExit("--model-version-id must be non-empty without outer whitespace")
    if args.operator_approved == args.validation_only:
        raise SystemExit(
            "exactly one of --operator-approved or --validation-only is required"
        )
    if (
        args.validation_only
        and args.activate
        and not args.allow_validation_only_activation
    ):
        raise SystemExit(
            "activating a validation-only profile requires "
            "--allow-validation-only-activation"
        )

    # Authority and runtime identity are owned exclusively by dedicated CLI
    # arguments.  Reject, rather than silently overwrite, generic metadata
    # that tries to claim any reserved field: a typo or conflicting automation
    # must fail closed before a profile directory is created.
    reserved_fields = {
        "operator_approved",
        "validation_only",
        "model_version_id",
        "model_artifact_sha256",
        "model_artifact_hash_scheme",
    }
    conflicts = sorted(reserved_fields.intersection(metadata))
    if conflicts:
        raise SystemExit(
            "--metadata-json contains reserved fields: " + ", ".join(conflicts)
        )
    try:
        artifact_digest = (
            runtime_artifact_sha256(args.model_artifact)
            if args.model_artifact is not None
            else normalize_runtime_artifact_sha256(args.model_artifact_sha256)
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(f"runtime model artifact fingerprint failed: {exc}") from exc

    metadata["operator_approved"] = args.operator_approved is True
    metadata["validation_only"] = args.validation_only is True
    metadata["model_version_id"] = args.model_version_id
    metadata["model_artifact_sha256"] = artifact_digest
    metadata["model_artifact_hash_scheme"] = RUNTIME_ARTIFACT_HASH_SCHEME
    metadata.setdefault("baseline_version", args.reference.parent.name)

    store = SceneProfileStore(args.profiles_dir)
    profile = store.create_from_baseline(
        args.camera_id,
        args.profile_id,
        args.reference,
        metadata=metadata,
        activate=False,
    )
    if args.activate:
        profile = store.set_active(profile.camera_id, profile.profile_id)
    print(json.dumps({
        "camera_id": profile.camera_id,
        "profile_id": profile.profile_id,
        "reference_path": str(profile.reference_path),
        "reference_sha256": profile.reference_sha256,
        "model_version_id": profile.metadata.get("model_version_id"),
        "model_artifact_sha256": profile.metadata.get("model_artifact_sha256"),
        "model_artifact_hash_scheme": profile.metadata.get(
            "model_artifact_hash_scheme"
        ),
        "operator_approved": profile.metadata.get("operator_approved") is True,
        "validation_only": profile.metadata.get("validation_only") is True,
        "active": store.get_active_profile_id(profile.camera_id) == profile.profile_id,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
