from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.activate_scene_profile import main
from argus.preprocessing.scene_profile import (
    RUNTIME_ARTIFACT_HASH_SCHEME,
    SceneProfileStore,
    runtime_artifact_sha256,
)


def _baseline(path: Path) -> Path:
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[8:56, 12:84] = (70, 180, 230)
    assert cv2.imwrite(str(path), image)
    return path


def test_activation_requires_exact_profile_model_binding(tmp_path, capsys) -> None:
    profiles_dir = tmp_path / "profiles"
    store = SceneProfileStore(profiles_dir)
    baseline = _baseline(tmp_path / "baseline.png")
    model = tmp_path / "model.pt"
    model.write_bytes(b"production runtime artifact")
    model_digest = runtime_artifact_sha256(model)
    store.create_from_baseline(
        "cam",
        "front-v1",
        baseline,
        metadata={
            "model_version_id": "model-v1",
            "model_artifact_sha256": model_digest,
            "model_artifact_hash_scheme": RUNTIME_ARTIFACT_HASH_SCHEME,
            "operator_approved": True,
            "validation_only": False,
        },
    )

    mismatch = main([
        "--camera-id",
        "cam",
        "--profile-id",
        "front-v1",
        "--model-version-id",
        "model-v2",
        "--model-artifact",
        str(model),
        "--profiles-dir",
        str(profiles_dir),
    ])
    assert mismatch == 2
    assert store.get_active_profile_id("cam") is None

    activated = main([
        "--camera-id",
        "cam",
        "--profile-id",
        "front-v1",
        "--model-version-id",
        "model-v1",
        "--model-artifact",
        str(model),
        "--profiles-dir",
        str(profiles_dir),
    ])
    output = json.loads(capsys.readouterr().out)

    assert activated == 0
    assert output["active"] is True
    assert output["model_version_id"] == "model-v1"
    assert output["model_artifact_sha256"] == model_digest
    assert store.get_active_profile_id("cam") == "front-v1"


def test_activation_rejects_digest_mismatch(tmp_path) -> None:
    profiles_dir = tmp_path / "profiles"
    store = SceneProfileStore(profiles_dir)
    baseline = _baseline(tmp_path / "baseline.png")
    approved_model = tmp_path / "approved.pt"
    requested_model = tmp_path / "requested.pt"
    approved_model.write_bytes(b"approved")
    requested_model.write_bytes(b"different")
    store.create_from_baseline(
        "cam",
        "front-v1",
        baseline,
        metadata={
            "model_version_id": "model-v1",
            "model_artifact_sha256": runtime_artifact_sha256(approved_model),
            "model_artifact_hash_scheme": RUNTIME_ARTIFACT_HASH_SCHEME,
            "operator_approved": True,
            "validation_only": False,
        },
    )

    result = main([
        "--camera-id", "cam",
        "--profile-id", "front-v1",
        "--model-version-id", "model-v1",
        "--model-artifact", str(requested_model),
        "--profiles-dir", str(profiles_dir),
    ])

    assert result == 2
    assert store.get_active_profile_id("cam") is None


def test_validation_only_activation_requires_explicit_allow(tmp_path, capsys) -> None:
    profiles_dir = tmp_path / "profiles"
    store = SceneProfileStore(profiles_dir)
    baseline = _baseline(tmp_path / "baseline.png")
    model = tmp_path / "model.pt"
    model.write_bytes(b"validation runtime artifact")
    digest = runtime_artifact_sha256(model)
    store.create_from_baseline(
        "cam",
        "validation-v1",
        baseline,
        metadata={
            "model_version_id": "model-v1",
            "model_artifact_sha256": digest,
            "model_artifact_hash_scheme": RUNTIME_ARTIFACT_HASH_SCHEME,
            "operator_approved": False,
            "validation_only": True,
        },
    )
    common = [
        "--camera-id", "cam",
        "--profile-id", "validation-v1",
        "--model-version-id", "model-v1",
        "--model-artifact-sha256", digest,
        "--profiles-dir", str(profiles_dir),
    ]

    assert main(common) == 2
    assert store.get_active_profile_id("cam") is None
    assert main(common + ["--allow-validation-only"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["active"] is True
    assert output["validation_only"] is True
    assert output["operator_approved"] is False
    assert store.get_active_profile_id("cam") == "validation-v1"


@pytest.mark.parametrize("operator_approved", [True, None, "false"])
def test_validation_only_activation_rejects_conflicting_or_ambiguous_authority(
    tmp_path,
    operator_approved,
) -> None:
    profiles_dir = tmp_path / "profiles"
    store = SceneProfileStore(profiles_dir)
    baseline = _baseline(tmp_path / "baseline.png")
    model = tmp_path / "model.pt"
    model.write_bytes(b"validation runtime artifact")
    digest = runtime_artifact_sha256(model)
    metadata = {
        "model_version_id": "model-v1",
        "model_artifact_sha256": digest,
        "model_artifact_hash_scheme": RUNTIME_ARTIFACT_HASH_SCHEME,
        "validation_only": True,
    }
    if operator_approved is not None:
        metadata["operator_approved"] = operator_approved
    store.create_from_baseline(
        "cam",
        "conflicted-v1",
        baseline,
        metadata=metadata,
    )

    result = main([
        "--camera-id", "cam",
        "--profile-id", "conflicted-v1",
        "--model-version-id", "model-v1",
        "--model-artifact-sha256", digest,
        "--profiles-dir", str(profiles_dir),
        "--allow-validation-only",
    ])

    assert result == 2
    assert store.get_active_profile_id("cam") is None
