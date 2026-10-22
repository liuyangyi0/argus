#!/usr/bin/env python3
"""Activate an existing immutable scene profile after binding verification."""

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
            "Atomically activate an existing scene profile only when its "
            "model binding matches the intended runtime model version."
        ),
    )
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--model-version-id", required=True)
    artifact = parser.add_mutually_exclusive_group(required=True)
    artifact.add_argument(
        "--model-artifact",
        type=Path,
        help="Runtime artifact file/directory to fingerprint before activation",
    )
    artifact.add_argument(
        "--model-artifact-sha256",
        help=(
            "Precomputed Argus runtime-artifact-v1 digest (not a bare model "
            "file checksum)"
        ),
    )
    parser.add_argument("--profiles-dir", type=Path, default=Path("data/scene_profiles"))
    parser.add_argument(
        "--allow-validation-only",
        action="store_true",
        help="Explicitly allow activation of a non-production validation profile",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        not args.model_version_id.strip()
        or args.model_version_id != args.model_version_id.strip()
    ):
        print(
            "ERROR: --model-version-id must be non-empty without outer whitespace",
            file=sys.stderr,
        )
        return 2
    try:
        requested_digest = (
            runtime_artifact_sha256(args.model_artifact)
            if args.model_artifact is not None
            else normalize_runtime_artifact_sha256(args.model_artifact_sha256)
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: runtime model artifact fingerprint failed: {exc}", file=sys.stderr)
        return 2

    store = SceneProfileStore(args.profiles_dir)
    profile = store.load(args.camera_id, args.profile_id)
    bound_version = profile.metadata.get("model_version_id")
    if bound_version != args.model_version_id:
        print(
            "ERROR: scene profile/model mismatch: "
            f"profile={bound_version!r}, requested={args.model_version_id!r}",
            file=sys.stderr,
        )
        return 2
    bound_digest = profile.metadata.get("model_artifact_sha256")
    try:
        bound_digest = normalize_runtime_artifact_sha256(bound_digest)
    except ValueError:
        print(
            "ERROR: scene profile has no valid Argus runtime artifact digest",
            file=sys.stderr,
        )
        return 2
    if (
        profile.metadata.get("model_artifact_hash_scheme")
        != RUNTIME_ARTIFACT_HASH_SCHEME
    ):
        print(
            "ERROR: scene profile runtime artifact hash scheme is missing or invalid",
            file=sys.stderr,
        )
        return 2
    if bound_digest != requested_digest:
        print(
            "ERROR: scene profile/runtime artifact mismatch: "
            f"profile={bound_digest!r}, requested={requested_digest!r}",
            file=sys.stderr,
        )
        return 2

    validation_only = profile.metadata.get("validation_only", False)
    operator_approved = profile.metadata.get("operator_approved")
    if not isinstance(validation_only, bool):
        print("ERROR: scene profile validation_only metadata is invalid", file=sys.stderr)
        return 2
    if validation_only is True:
        if operator_approved is not False:
            print(
                "ERROR: validation-only profile has conflicting or missing "
                "operator approval metadata",
                file=sys.stderr,
            )
            return 2
        if not args.allow_validation_only:
            print(
                "ERROR: validation-only activation requires --allow-validation-only",
                file=sys.stderr,
            )
            return 2
    elif operator_approved is not True:
        print(
            "ERROR: production scene profile is not explicitly operator-approved",
            file=sys.stderr,
        )
        return 2

    activated = store.set_active(args.camera_id, args.profile_id)
    print(json.dumps({
        "camera_id": activated.camera_id,
        "profile_id": activated.profile_id,
        "model_version_id": bound_version,
        "model_artifact_sha256": bound_digest,
        "model_artifact_hash_scheme": RUNTIME_ARTIFACT_HASH_SCHEME,
        "operator_approved": activated.metadata.get("operator_approved") is True,
        "validation_only": validation_only is True,
        "reference_sha256": activated.reference_sha256,
        "active": True,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
