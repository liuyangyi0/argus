"""Tests for immutable model package assembly."""

from __future__ import annotations

import json
import shutil
from unittest.mock import MagicMock

import numpy as np
import pytest

from argus.anomaly.artifacts import (
    artifact_metadata,
    new_artifact_run,
    write_artifact_manifest,
)
from argus.anomaly.detector import AnomalibDetector
from argus.anomaly.model_package import ModelPackager
from argus.preprocessing.scene_profile import runtime_artifact_sha256


def test_package_places_calibration_next_to_openvino_model(tmp_path):
    """Packaged OpenVINO models must not load silently uncalibrated."""
    model_dir = tmp_path / "models" / "cam_01" / "default"
    model_dir.mkdir(parents=True)
    (model_dir / "model.ckpt").write_bytes(b"checkpoint")

    export_dir = tmp_path / "exports" / "cam_01" / "default"
    openvino_dir = export_dir / "weights" / "openvino"
    openvino_dir.mkdir(parents=True)
    (openvino_dir / "model.xml").write_text("<net/>", encoding="utf-8")
    (openvino_dir / "model.bin").write_bytes(b"binary")

    calibration = model_dir / "calibration.json"
    calibration.write_text('{"sorted_scores": [0.1, 0.2, 0.3]}', encoding="utf-8")

    packager = ModelPackager()
    package_dir = packager.package(
        output_dir=tmp_path / "packages",
        model_dir=model_dir,
        export_dir=export_dir,
        calibration_path=calibration,
        camera_id="cam_01",
        model_type="patchcore",
    )

    packaged_calibration = package_dir / "model.openvino" / "calibration.json"
    assert packaged_calibration.read_text(encoding="utf-8") == calibration.read_text(
        encoding="utf-8"
    )
    assert (package_dir / "conformal_quantiles.json").read_text(
        encoding="utf-8"
    ) == calibration.read_text(encoding="utf-8")

    detector = AnomalibDetector(model_path=package_dir / "model.openvino" / "model.xml")
    detector._load_calibration()
    assert detector.is_calibrated
    assert packager.verify(package_dir)


def test_package_preserves_cuda_runtime_and_training_checkpoint_separately(tmp_path):
    model_dir = tmp_path / "models" / "run-one"
    model_dir.mkdir(parents=True)
    (model_dir / "best.ckpt").write_bytes(b"training checkpoint")
    (model_dir / "other.pt").write_bytes(b"not the checkpoint")
    export_dir = tmp_path / "exports" / "run-one"
    runtime_dir = export_dir / "weights" / "torch"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "model.pt").write_bytes(b"CUDA inference artifact")
    (runtime_dir / "calibration.json").write_text('{"sorted_scores": [0.1]}', encoding="utf-8")
    (runtime_dir / "metadata.json").write_text('{"image_size": 256}', encoding="utf-8")
    packager = ModelPackager()

    package_dir = packager.package(
        output_dir=tmp_path / "packages", model_dir=model_dir, export_dir=export_dir,
        camera_id="cam", model_type="patchcore",
    )

    assert (package_dir / "model.pytorch").read_bytes() == b"training checkpoint"
    packaged_runtime = package_dir / "runtime" / "weights" / "torch"
    assert (packaged_runtime / "model.pt").read_bytes() == b"CUDA inference artifact"
    assert (packaged_runtime / "calibration.json").is_file()
    assert (packaged_runtime / "metadata.json").is_file()
    assert (package_dir / "runtime" / "artifact.json").is_file()
    assert packager.verify(package_dir)


def _portable_package(tmp_path, *, signed=False):
    run = new_artifact_run(tmp_path / "models", tmp_path / "exports", "cam", "default")
    (run.training_dir / "best.ckpt").write_bytes(b"checkpoint")
    runtime_dir = run.export_dir / "weights" / "torch"
    runtime_dir.mkdir(parents=True)
    artifact = runtime_dir / "model.pt"
    artifact.write_bytes(b"runtime")
    calibration = runtime_dir / "calibration.json"
    calibration.write_text('{"sorted_scores": [0.1, 0.2, 0.3]}', encoding="utf-8")
    write_artifact_manifest(run, artifact, model_type="patchcore", calibration_status="ok")
    packager = ModelPackager(signing_key="secret" if signed else "")
    package = packager.package(
        tmp_path / "packages", run.training_dir, run.export_dir,
        camera_id="cam", model_type="patchcore", runtime_artifact=artifact,
    )
    return packager, package, run


def test_versioned_package_survives_relocation_with_bounded_calibration_and_stable_hash(tmp_path):
    packager, package, run = _portable_package(tmp_path)
    relative = "runtime/weights/torch/model.pt"
    original_hash = runtime_artifact_sha256(package / relative)
    assert artifact_metadata(package / relative)["schema_version"] == 2
    assert (package / "provenance/runtime/artifact.json").is_file()
    shutil.rmtree(run.export_dir)
    shutil.rmtree(run.training_dir)
    moved = tmp_path / "moved" / "bundle"
    moved.parent.mkdir()
    shutil.move(str(package), str(moved))

    detector = AnomalibDetector(model_path=moved / relative)
    detector._load_calibration()
    assert detector.is_calibrated
    assert runtime_artifact_sha256(moved / relative) == original_hash
    metadata = artifact_metadata(moved / relative)
    assert (moved / "runtime" / metadata["checkpoint_path"]).is_file()
    assert metadata["requires_release_review"] is True
    assert packager.verify(moved)


@pytest.mark.parametrize("contents", [None, "{}", '{"sorted_scores": [0.3, 0.1]}', '{"sorted_scores": [NaN]}', '{"sorted_scores": [[0.1]]}'])
def test_manifest_required_calibration_never_degrades_to_loaded_uncalibrated(tmp_path, contents):
    _packager, package, _run = _portable_package(tmp_path)
    runtime_dir = package / "runtime/weights/torch"
    calibration = runtime_dir / "calibration.json"
    if contents is None:
        calibration.unlink()
    else:
        calibration.write_text(contents, encoding="utf-8")
    (package.parent / "calibration.json").write_text('{"sorted_scores": [0.9]}', encoding="utf-8")
    detector = AnomalibDetector(model_path=runtime_dir / "model.pt")
    detector._loaded = True
    detector._engine = object()
    with pytest.raises(RuntimeError, match="calibration"):
        detector._load_calibration()
    assert not detector._loaded
    assert detector._engine is None
    assert not detector.is_calibrated


@pytest.mark.parametrize("with_warmup", [False, True])
def test_reload_rejects_bad_candidate_calibration_without_touching_old_runtime(tmp_path, monkeypatch, with_warmup):
    _packager, package, _run = _portable_package(tmp_path)
    candidate_path = package / "runtime/weights/torch/model.pt"
    (candidate_path.parent / "calibration.json").write_text("{}", encoding="utf-8")
    detector = AnomalibDetector(model_path=tmp_path / "old-model.pt")
    old_engine = object()
    detector._engine = old_engine
    detector._loaded = True
    detector._calibration_scores = np.array([0.1, 0.2])
    detector._calibration_n = 2
    detector.status.mark_loaded(backend="torch-cpu", model_path=str(detector._model_path), image_size=detector.image_size)
    old_status = detector.status.to_dict()
    old_path = detector._model_path
    old_scores = detector._calibration_scores
    monkeypatch.setattr(detector, "_load_inferencer_for_backend", lambda _path: (MagicMock(), "torch-cpu"))

    outcome = detector.hot_reload_with_warmup(candidate_path) if with_warmup else detector.hot_reload(candidate_path)

    assert (outcome["success"] if with_warmup else outcome) is False
    assert detector._engine is old_engine
    assert detector._model_path == old_path
    assert detector._calibration_scores is old_scores
    assert detector._calibration_n == 2
    assert detector.status.to_dict() == old_status
    assert detector.is_loaded


def test_exact_runtime_warmup_uses_registry_fingerprint_and_rejects_sidecar_tampering(tmp_path, monkeypatch):
    _packager, package, _run = _portable_package(tmp_path)
    artifact = package / "runtime/weights/torch/model.pt"
    expected = runtime_artifact_sha256(artifact)[:16]
    detector = AnomalibDetector(model_path=tmp_path / "old.pt")
    monkeypatch.setattr(detector, "_load_inferencer_for_backend", lambda _path: (MagicMock(), "torch-cpu"))
    first = detector.hot_reload_with_warmup(artifact, expected_hash=expected)
    assert first["success"] is True
    assert first["sha256_verified"] is True
    old_engine = detector._engine
    (artifact.parent / "calibration.json").write_text('{"sorted_scores": [0.8]}', encoding="utf-8")
    second = detector.hot_reload_with_warmup(artifact, expected_hash=expected)
    assert second["success"] is False
    assert "SHA256 mismatch" in second["error"]
    assert detector._engine is old_engine


def test_signature_cannot_be_downgraded_by_deleting_signature_file(tmp_path):
    packager, package, _run = _portable_package(tmp_path, signed=True)
    assert packager.verify(package)
    (package / "signature.sig").unlink()
    assert not packager.verify(package)


def test_unicode_signature_is_rejected_without_raising(tmp_path):
    packager, package, _run = _portable_package(tmp_path, signed=True)
    (package / "signature.sig").write_text("恶意签名", encoding="utf-8")
    assert not packager.verify(package)


@pytest.mark.parametrize("checksum_contents", ["", "garbage", "0" * 64 + "  ../outside.pt", "0" * 64 + "  /outside.pt"])
def test_checksum_manifest_rejects_empty_malformed_and_escaping_entries(tmp_path, checksum_contents):
    packager, package, _run = _portable_package(tmp_path)
    (package / "SHA256SUMS").write_text(checksum_contents, encoding="utf-8")
    assert not packager.verify(package)


def test_checksum_manifest_rejects_duplicate_or_unlisted_files(tmp_path):
    packager, package, _run = _portable_package(tmp_path)
    sums = package / "SHA256SUMS"
    original = sums.read_text(encoding="utf-8")
    sums.write_text(original + "\n" + original.splitlines()[0], encoding="utf-8")
    assert not packager.verify(package)
    sums.write_text(original, encoding="utf-8")
    (package / "runtime/metadata.json").write_text("{}", encoding="utf-8")
    assert not packager.verify(package)


def test_portable_manifest_rejects_relative_and_absolute_artifact_escape(tmp_path):
    _packager, package, _run = _portable_package(tmp_path)
    artifact = package / "runtime/weights/torch/model.pt"
    marker = artifact.parent / "artifact.json"
    original = json.loads(marker.read_text(encoding="utf-8"))
    for invalid in ("../outside.pt", "/outside.pt", "C:/outside.pt", "..\\outside.pt"):
        marker.write_text(json.dumps({**original, "artifact_path": invalid}), encoding="utf-8")
        with pytest.raises(ValueError, match="bundle"):
            artifact_metadata(artifact)


def test_package_verification_rejects_symlink_even_when_checksum_matches(tmp_path):
    packager, package, _run = _portable_package(tmp_path)
    artifact = package / "runtime/weights/torch/model.pt"
    external = tmp_path / "outside.pt"
    external.write_bytes(artifact.read_bytes())
    artifact.unlink()
    try:
        artifact.symlink_to(external)
    except OSError:
        pytest.skip("Symlink creation is unavailable")
    assert not packager.verify(package)
