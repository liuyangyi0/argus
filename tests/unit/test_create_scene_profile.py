from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from argus.preprocessing.scene_profile import (
    RUNTIME_ARTIFACT_HASH_SCHEME,
    SceneProfileStore,
    runtime_artifact_sha256,
)
from scripts.create_scene_profile import main


def _baseline(path) -> None:
    assert cv2.imwrite(
        str(path),
        np.full((16, 24, 3), 73, dtype=np.uint8),
    )


def _model(path) -> None:
    path.write_bytes(b"argus runtime model")


def test_reserved_authority_metadata_is_rejected_before_profile_creation(
    tmp_path,
) -> None:
    baseline = tmp_path / "baseline.png"
    profiles = tmp_path / "profiles"
    model = tmp_path / "model.pt"
    _baseline(baseline)
    _model(model)

    with pytest.raises(SystemExit, match="reserved fields"):
        main([
            "--camera-id", "cam",
            "--profile-id", "validation-pose",
            "--reference", str(baseline),
            "--profiles-dir", str(profiles),
            "--model-version-id", "model-v1",
            "--model-artifact", str(model),
            "--validation-only",
            # Generic metadata cannot grant authority or inject a fake digest.
            "--metadata-json", (
                '{"operator_approved": true, "validation_only": false, '
                '"model_artifact_sha256": "ffffffffffffffffffffffffffffffff'
                'ffffffffffffffffffffffffffffffff"}'
            ),
        ])

    assert SceneProfileStore(profiles).list_profiles("cam") == []


def test_validation_profile_uses_dedicated_flags_and_explicit_activation_allow(
    tmp_path,
    capsys,
) -> None:
    baseline = tmp_path / "baseline.png"
    profiles = tmp_path / "profiles"
    model = tmp_path / "model.pt"
    _baseline(baseline)
    _model(model)

    code = main([
        "--camera-id", "cam",
        "--profile-id", "validation-pose",
        "--reference", str(baseline),
        "--profiles-dir", str(profiles),
        "--model-version-id", "model-v1",
        "--model-artifact", str(model),
        "--validation-only",
        "--metadata-json", '{"audit_note": "recorded-corpus"}',
        "--activate",
        "--allow-validation-only-activation",
    ])

    assert code == 0
    profile = SceneProfileStore(profiles).get_active("cam")
    assert profile is not None
    assert profile.metadata["operator_approved"] is False
    assert profile.metadata["validation_only"] is True
    assert profile.metadata["model_artifact_sha256"] == runtime_artifact_sha256(model)
    assert (
        profile.metadata["model_artifact_hash_scheme"]
        == RUNTIME_ARTIFACT_HASH_SCHEME
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["operator_approved"] is False
    assert payload["validation_only"] is True
    assert payload["model_artifact_sha256"] == runtime_artifact_sha256(model)
    assert payload["active"] is True


def test_operator_approval_requires_dedicated_flag(tmp_path, capsys) -> None:
    baseline = tmp_path / "baseline.png"
    profiles = tmp_path / "profiles"
    model = tmp_path / "model.pt"
    _baseline(baseline)
    _model(model)

    code = main([
        "--camera-id", "cam",
        "--profile-id", "production-pose",
        "--reference", str(baseline),
        "--profiles-dir", str(profiles),
        "--model-version-id", "model-v1",
        "--model-artifact", str(model),
        "--operator-approved",
        "--activate",
    ])

    assert code == 0
    profile = SceneProfileStore(profiles).get_active("cam")
    assert profile is not None
    assert profile.metadata["operator_approved"] is True
    assert profile.metadata["validation_only"] is False
    assert profile.metadata["model_artifact_sha256"] == runtime_artifact_sha256(model)
    payload = json.loads(capsys.readouterr().out)
    assert payload["operator_approved"] is True
    assert payload["validation_only"] is False
    assert payload["active"] is True


def test_operator_and_validation_authority_conflict_fails_closed(tmp_path) -> None:
    baseline = tmp_path / "baseline.png"
    profiles = tmp_path / "profiles"
    model = tmp_path / "model.pt"
    _baseline(baseline)
    _model(model)

    with pytest.raises(SystemExit, match="exactly one"):
        main([
            "--camera-id", "cam",
            "--profile-id", "conflicted",
            "--reference", str(baseline),
            "--profiles-dir", str(profiles),
            "--model-version-id", "model-v1",
            "--model-artifact", str(model),
            "--operator-approved",
            "--validation-only",
        ])

    assert SceneProfileStore(profiles).list_profiles("cam") == []


def test_profile_authority_must_be_explicit(tmp_path) -> None:
    baseline = tmp_path / "baseline.png"
    profiles = tmp_path / "profiles"
    model = tmp_path / "model.pt"
    _baseline(baseline)
    _model(model)

    with pytest.raises(SystemExit, match="exactly one"):
        main([
            "--camera-id", "cam",
            "--profile-id", "ambiguous",
            "--reference", str(baseline),
            "--profiles-dir", str(profiles),
            "--model-version-id", "model-v1",
            "--model-artifact", str(model),
        ])

    assert SceneProfileStore(profiles).list_profiles("cam") == []


def test_validation_profile_cannot_activate_without_explicit_allow(tmp_path) -> None:
    baseline = tmp_path / "baseline.png"
    profiles = tmp_path / "profiles"
    model = tmp_path / "model.pt"
    _baseline(baseline)
    _model(model)

    with pytest.raises(SystemExit, match="allow-validation-only-activation"):
        main([
            "--camera-id", "cam",
            "--profile-id", "validation-pose",
            "--reference", str(baseline),
            "--profiles-dir", str(profiles),
            "--model-version-id", "model-v1",
            "--model-artifact", str(model),
            "--validation-only",
            "--activate",
        ])

    assert SceneProfileStore(profiles).list_profiles("cam") == []
