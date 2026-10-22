"""Tests for immutable camera scene-profile storage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from argus.preprocessing.scene_profile import (
    SceneProfileCorruptError,
    SceneProfileStore,
    runtime_artifact_sha256,
)


def _baseline(path: Path, *, width: int = 96, height: int = 64) -> Path:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = 35
    image[8:34, 12:57, 1] = 180
    image[20:55, 44:88, 2] = 240
    assert cv2.imwrite(str(path), image)
    return path


def test_runtime_artifact_digest_binds_argus_envelope_not_only_file_bytes(
    tmp_path: Path,
) -> None:
    first = tmp_path / "model.pt"
    second_dir = tmp_path / "renamed"
    second_dir.mkdir()
    second = second_dir / "renamed.pt"
    first.write_bytes(b"same model bytes")
    second.write_bytes(first.read_bytes())

    bare = hashlib.sha256(first.read_bytes()).hexdigest()
    first_digest = runtime_artifact_sha256(first)

    assert first_digest != bare
    assert runtime_artifact_sha256(second) != first_digest


def test_runtime_artifact_digest_binds_openvino_xml_and_bin(tmp_path: Path) -> None:
    model = tmp_path / "model.xml"
    weights = tmp_path / "model.bin"
    model.write_bytes(b"openvino graph")
    weights.write_bytes(b"weights-v1")

    first_digest = runtime_artifact_sha256(model)
    weights.write_bytes(b"weights-v2")

    assert runtime_artifact_sha256(model) != first_digest


def test_openvino_digest_is_complete_from_either_companion_path(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.xml"
    weights = tmp_path / "model.bin"
    model.write_bytes(b"openvino graph")
    weights.write_bytes(b"weights")

    assert runtime_artifact_sha256(weights) == runtime_artifact_sha256(model)


@pytest.mark.parametrize("sidecar_name", ["metadata.json", "calibration.json"])
def test_runtime_artifact_digest_binds_argus_runtime_sidecars(
    tmp_path: Path,
    sidecar_name: str,
) -> None:
    model = tmp_path / "model.pt"
    sidecar = tmp_path / sidecar_name
    model.write_bytes(b"torch runtime")
    sidecar.write_text('{"revision": 1}', encoding="utf-8")

    first_digest = runtime_artifact_sha256(model)
    sidecar.write_text('{"revision": 2}', encoding="utf-8")

    assert runtime_artifact_sha256(model) != first_digest


def test_runtime_artifact_digest_rejects_incomplete_or_unsupported_files(
    tmp_path: Path,
) -> None:
    incomplete_openvino = tmp_path / "model.xml"
    incomplete_openvino.write_bytes(b"graph without weights")
    orphaned_weights = tmp_path / "orphaned.bin"
    orphaned_weights.write_bytes(b"weights without graph")
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("not a runtime model", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="companion"):
        runtime_artifact_sha256(incomplete_openvino)
    with pytest.raises(FileNotFoundError, match="companion"):
        runtime_artifact_sha256(orphaned_weights)
    with pytest.raises(ValueError, match="Unsupported"):
        runtime_artifact_sha256(unsupported)


def test_legacy_shared_model_digest_ignores_independent_runs_before_and_after_manifest(tmp_path):
    from argus.anomaly.artifacts import new_artifact_run, write_artifact_manifest

    models_root = tmp_path / "models"
    legacy_root = models_root / "cam" / "zone"
    legacy_root.mkdir(parents=True)
    (legacy_root / "model.pt").write_bytes(b"legacy runtime")
    original = runtime_artifact_sha256(legacy_root)

    run = new_artifact_run(models_root, tmp_path / "exports", "cam", "zone")
    in_progress_model = run.training_dir / "model.pt"
    in_progress_model.write_bytes(b"new training output")
    (run.training_dir / "calibration.json").write_text('{"revision": 1}', encoding="utf-8")
    assert not (run.training_dir / "artifact.json").exists()
    assert runtime_artifact_sha256(legacy_root) == original

    artifact = run.export_dir / "model.pt"
    artifact.write_bytes(b"new runtime export")
    write_artifact_manifest(run, artifact, model_type="patchcore", calibration_status="skipped")
    assert runtime_artifact_sha256(legacy_root) == original

    in_progress_model.write_bytes(b"a failed or later modified independent run")
    (run.training_dir / "calibration.json").write_text('{"revision": 2}', encoding="utf-8")
    assert runtime_artifact_sha256(legacy_root) == original
    # The independent run's exact-file identity remains fully observable.
    new_digest = runtime_artifact_sha256(artifact)
    artifact.write_bytes(b"changed runtime export")
    assert runtime_artifact_sha256(artifact) != new_digest
    assert runtime_artifact_sha256(legacy_root) == original


def test_legacy_digest_still_binds_historical_run_prefixed_directories(tmp_path):
    legacy_root = tmp_path / "legacy"
    historical = legacy_root / "run-historical"
    historical.mkdir(parents=True)
    artifact = historical / "model.pt"
    artifact.write_bytes(b"historical model v1")
    original = runtime_artifact_sha256(legacy_root)

    artifact.write_bytes(b"historical model v2")

    assert runtime_artifact_sha256(legacy_root) != original


def test_create_load_and_metadata_round_trip(tmp_path: Path) -> None:
    source = _baseline(tmp_path / "baseline_00001.png")
    store = SceneProfileStore(tmp_path / "profiles")

    created = store.create_from_baseline(
        "cam_01",
        "front-view-v1",
        source,
        metadata={"zone_ids": ["cabinet"], "approved_by": "operator"},
    )

    profile_dir = tmp_path / "profiles" / "cam_01" / "front-view-v1"
    assert (profile_dir / "reference.jpg").is_file()
    assert (profile_dir / "metadata.json").is_file()
    assert created.camera_id == "cam_01"
    assert created.profile_id == "front-view-v1"
    assert created.image_width == 96
    assert created.image_height == 64
    assert created.source_baseline_name == source.name
    assert created.metadata == {"zone_ids": ["cabinet"], "approved_by": "operator"}
    assert created.created_at.tzinfo is not None
    assert cv2.imread(str(created.reference_path)).shape == (64, 96, 3)

    loaded = store.load("cam_01", "front-view-v1")
    assert loaded == created


def test_list_activate_switch_and_clear(tmp_path: Path) -> None:
    source = _baseline(tmp_path / "baseline.jpg")
    store = SceneProfileStore(tmp_path / "profiles")
    store.create_from_baseline("cam", "scene-b", source)
    store.create_from_baseline("cam", "scene-a", source, activate=True)

    assert [item.profile_id for item in store.list_profiles("cam")] == ["scene-a", "scene-b"]
    assert store.get_active_profile_id("cam") == "scene-a"
    assert store.get_active("cam").profile_id == "scene-a"

    store.set_active("cam", "scene-b")
    marker = json.loads((tmp_path / "profiles" / "cam" / "active.json").read_text())
    assert marker["profile_id"] == "scene-b"
    assert store.get_active("cam").profile_id == "scene-b"

    assert store.clear_active("cam") is True
    assert store.clear_active("cam") is False
    assert store.get_active("cam") is None
    # Clearing the pointer never deletes immutable profiles.
    assert len(store.list_profiles("cam")) == 2


def test_profile_is_immutable_and_missing_active_is_rejected(tmp_path: Path) -> None:
    source = _baseline(tmp_path / "baseline.png")
    store = SceneProfileStore(tmp_path / "profiles")
    store.create_from_baseline("cam", "scene", source)

    with pytest.raises(FileExistsError):
        store.create_from_baseline("cam", "scene", source)
    with pytest.raises(FileNotFoundError):
        store.set_active("cam", "missing")


@pytest.mark.parametrize(
    ("camera_id", "profile_id"),
    [
        ("../outside", "safe"),
        ("cam/child", "safe"),
        (r"cam\child", "safe"),
        (".", "safe"),
        ("safe", "../outside"),
        ("safe", "profile/child"),
        ("safe", r"profile\child"),
        ("safe", "..hidden"),
        ("safe", ""),
        ("safe", "white space"),
    ],
)
def test_identifiers_cannot_traverse_storage(
    tmp_path: Path,
    camera_id: str,
    profile_id: str,
) -> None:
    source = _baseline(tmp_path / "baseline.png")
    store = SceneProfileStore(tmp_path / "profiles")

    with pytest.raises(ValueError):
        store.create_from_baseline(camera_id, profile_id, source)
    assert not (tmp_path / "outside").exists()


def test_creation_requires_explicit_existing_baseline_path(tmp_path: Path) -> None:
    store = SceneProfileStore(tmp_path / "profiles")

    with pytest.raises(TypeError, match="explicit filesystem path"):
        store.create_from_baseline("cam", "scene", np.zeros((10, 10, 3), dtype=np.uint8))
    with pytest.raises(FileNotFoundError):
        store.create_from_baseline("cam", "scene", tmp_path / "missing.jpg")
    invalid = tmp_path / "not-an-image.jpg"
    invalid.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="not a decodable image"):
        store.create_from_baseline("cam", "scene", invalid)
    assert not hasattr(store, "create_from_frame")


def test_non_json_metadata_is_rejected_before_profile_is_written(tmp_path: Path) -> None:
    source = _baseline(tmp_path / "baseline.png")
    store = SceneProfileStore(tmp_path / "profiles")

    with pytest.raises(ValueError, match="finite JSON values"):
        store.create_from_baseline("cam", "scene", source, metadata={"threshold": float("nan")})
    assert store.list_profiles("cam") == []


def test_reference_tampering_fails_closed(tmp_path: Path) -> None:
    source = _baseline(tmp_path / "baseline.png")
    store = SceneProfileStore(tmp_path / "profiles")
    profile = store.create_from_baseline("cam", "scene", source, activate=True)
    profile.reference_path.write_bytes(b"tampered")

    with pytest.raises(SceneProfileCorruptError, match="checksum"):
        store.load("cam", "scene")
    with pytest.raises(SceneProfileCorruptError, match="unavailable"):
        store.get_active("cam")


def test_malicious_active_pointer_cannot_escape_root(tmp_path: Path) -> None:
    source = _baseline(tmp_path / "baseline.png")
    store = SceneProfileStore(tmp_path / "profiles")
    store.create_from_baseline("cam", "scene", source)
    marker = tmp_path / "profiles" / "cam" / "active.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "camera_id": "cam",
                "profile_id": "../../outside",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SceneProfileCorruptError, match="pointer is invalid"):
        store.get_active("cam")


def test_failed_publish_cleans_staging_directory(tmp_path: Path, monkeypatch) -> None:
    source = _baseline(tmp_path / "baseline.png")
    root = tmp_path / "profiles"
    store = SceneProfileStore(root)

    def fail_replace(_source, _destination) -> None:
        raise OSError("simulated publish failure")

    monkeypatch.setattr("argus.preprocessing.scene_profile.os.replace", fail_replace)
    with pytest.raises(OSError, match="publish failure"):
        store.create_from_baseline("cam", "scene", source)

    camera_dir = root / "cam"
    assert not (camera_dir / "scene").exists()
    assert list(camera_dir.glob("*.tmp")) == []
    assert list(camera_dir.glob(".*.tmp")) == []
