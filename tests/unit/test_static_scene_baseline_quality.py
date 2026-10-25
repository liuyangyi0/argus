"""Static-scene quality policy guards using synthetic arrays and I/O doubles.

No camera, video decoder, model loader, training engine, or real inference is
used. A fixed texture is legitimate stationary content, not a varied dataset.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from argus.anomaly.baseline_quality import FixedSceneQualityContext, inspect_baseline_quality
from argus.anomaly.scene_training import SceneTrainingInput
from argus.anomaly.trainer import ModelTrainer
from argus.preprocessing.scene_guard import SceneGuardConfig


SHAPE = (96, 128)


def _texture(shape: tuple[int, int] = SHAPE) -> np.ndarray:
    """A sharp, well-lit texture with no temporal or brightness diversity."""
    yy, xx = np.indices(shape)
    gray = np.where((xx // 4 + yy // 4) % 2, 72, 184).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _context(**overrides) -> FixedSceneQualityContext:
    values = dict(
        config=SceneGuardConfig(), expected_shape=SHAPE, sample_count=150,
        training_count=100, calibration_count=50, source_session_count=2,
        accepted_span_seconds=300.0,
    )
    values.update(overrides)
    return FixedSceneQualityContext(**values)


def _images(tmp_path: Path, count: int = 150) -> list[Path]:
    return [tmp_path / f"sample-{index:04d}.png" for index in range(count)]


def _reader(monkeypatch, images: list[Path], *, exceptional_index=None, exceptional_frame=None):
    positions = {str(path): index for index, path in enumerate(images)}
    texture = _texture()

    def read(path, *args, **kwargs):
        if positions[str(path)] == exceptional_index:
            return exceptional_frame
        return texture.copy()

    mock = MagicMock(side_effect=read)
    monkeypatch.setattr(cv2, "imread", mock)
    return mock


def test_legacy_policy_still_rejects_repetitive_constant_brightness_scene(tmp_path, monkeypatch):
    images = _images(tmp_path)
    _reader(monkeypatch, images)

    report = inspect_baseline_quality(images, min_images=30)

    assert report["passed"] is False
    assert report["near_duplicate_rate"] >= 0.8
    assert report["brightness_std"] <= 2.0
    assert report["errors"]
    assert report.get("policy_version") != "argus.baseline-quality.scene-static.v1"


def test_verified_fixed_scene_accepts_stationarity_as_warning_not_diversity_proof(tmp_path, monkeypatch):
    images = _images(tmp_path)
    reader = _reader(monkeypatch, images)

    report = inspect_baseline_quality(images, min_images=30, fixed_scene=_context())

    assert report["passed"] is True and report["integrity_passed"] is True
    assert report["errors"] == []
    assert report["policy_version"] == "argus.baseline-quality.scene-static.v1"
    assert report["scope"] == "observed_fixed_scene_only"
    assert report["production_readiness"] == "not_evaluated"
    assert report["corruption_rate"] == 0.0
    assert report["brightness_std"] <= 2.0 and report["near_duplicate_rate"] >= 0.8
    assert report["warnings"]
    assert {
        "observed_conditions_only", "temporal_dependence_unverified",
        "frozen_capture_not_excluded_by_pixels", "low_temporal_brightness_variation",
        "high_perceptual_redundancy",
    }.issubset(report["warning_codes"])
    assert reader.call_count == len(images)


def test_one_unreadable_frame_is_fatal_even_below_legacy_corruption_limit(tmp_path, monkeypatch):
    images = _images(tmp_path)
    _reader(monkeypatch, images, exceptional_index=77, exceptional_frame=None)

    report = inspect_baseline_quality(images, min_images=30, fixed_scene=_context())

    assert report["passed"] is False and report["integrity_passed"] is False
    assert 0 < report["corruption_rate"] < 0.1
    assert "corrupt_images" in report["error_codes"]
    assert report["errors"]


def _smooth_gradient() -> np.ndarray:
    gray = np.tile(np.linspace(32, 224, SHAPE[1], dtype=np.uint8), (SHAPE[0], 1))
    return np.repeat(gray[:, :, None], 3, axis=2)


@pytest.mark.parametrize("bad_frame", [
    np.zeros((*SHAPE, 3), dtype=np.uint8),
    np.full((*SHAPE, 3), 255, dtype=np.uint8),
    np.full((*SHAPE, 3), 128, dtype=np.uint8),
    _smooth_gradient(),
    _texture((SHAPE[0], SHAPE[1] + 1)),
], ids=["black", "white", "low-contrast", "blurred-gradient", "wrong-size"])
def test_static_mode_never_accepts_one_bad_sample_as_normal_stillness(tmp_path, monkeypatch, bad_frame):
    images = _images(tmp_path)
    _reader(monkeypatch, images, exceptional_index=93, exceptional_frame=bad_frame)

    report = inspect_baseline_quality(images, min_images=30, fixed_scene=_context())

    assert report["passed"] is False and report["integrity_passed"] is False
    assert "scene_frame_quality_failed" in report["error_codes"]
    assert report["errors"]


def test_static_context_does_not_allow_missing_frozen_sample(tmp_path, monkeypatch):
    images = _images(tmp_path, 149)
    _reader(monkeypatch, images)

    report = inspect_baseline_quality(images, min_images=30, fixed_scene=_context())

    assert report["passed"] is False and report["integrity_passed"] is False
    assert "scene_sample_count_mismatch" in report["error_codes"]


def test_static_context_rejects_repeated_paths_not_repeated_pixels(tmp_path, monkeypatch):
    images = _images(tmp_path)
    images[-1] = images[0]
    _reader(monkeypatch, images)

    report = inspect_baseline_quality(images, min_images=30, fixed_scene=_context())

    assert report["passed"] is False and report["integrity_passed"] is False
    assert "scene_sample_count_mismatch" in report["error_codes"]


def test_static_context_does_not_lower_requested_minimum_image_count(tmp_path, monkeypatch):
    images = _images(tmp_path)
    _reader(monkeypatch, images)

    report = inspect_baseline_quality(images, min_images=151, fixed_scene=_context())

    assert report["passed"] is False and report["integrity_passed"] is False
    assert "insufficient_images" in report["error_codes"]


@pytest.mark.parametrize("changes", [
    {"training_count": 99, "sample_count": 149},
    {"calibration_count": 49, "sample_count": 149},
    {"source_session_count": 1},
    {"accepted_span_seconds": 299.999},
    {"accepted_span_seconds": float("nan")},
    {"accepted_span_seconds": float("inf")},
    {"sample_count": 151},
    {"training_count": True},
    {"source_session_count": True},
    {"expected_shape": (0, SHAPE[1])},
    {"expected_shape": (SHAPE[0], True)},
    {"config": None},
])
def test_fixed_scene_context_cannot_claim_incomplete_or_malformed_evidence(changes):
    with pytest.raises((TypeError, ValueError)):
        _context(**changes)


def _untrusted_input(tmp_path: Path) -> SceneTrainingInput:
    base = tmp_path / "baselines"
    root = base / "camera" / "default" / "scene-test"
    images = root / "images"
    images.mkdir(parents=True)
    return SceneTrainingInput(
        baselines_dir=base, camera_id="camera", root=root, images_dir=images,
        image_count=150, training_count=100, calibration_count=50,
        lineage={
            "schema_version": "argus.scene-training-input.v1", "workflow_id": "test",
            "scene_profile_id": "test", "baseline_sha256": "a" * 64,
            "manifest_sha256": "b" * 64, "reference_sha256": "c" * 64,
        },
    )


def test_typed_but_unvalidated_scene_input_cannot_enable_static_adaptation(tmp_path, monkeypatch):
    untrusted = _untrusted_input(tmp_path)
    reader = MagicMock(side_effect=AssertionError("unverified input must not reach image inspection"))
    monkeypatch.setattr(cv2, "imread", reader)
    trainer = ModelTrainer(SimpleNamespace())

    # Constructing the public value object is not evidence. Missing frozen
    # workflow/baseline metadata must be rejected before reading any images.
    with pytest.raises((OSError, ValueError)):
        trainer._validate_baseline_quality(untrusted.images_dir, scene_training_input=untrusted)
    reader.assert_not_called()


def test_quality_wrapper_revalidates_input_before_inspection(tmp_path, monkeypatch):
    untrusted = _untrusted_input(tmp_path)
    revalidate = MagicMock(side_effect=ValueError("baseline identity changed"))
    reader = MagicMock(side_effect=AssertionError("invalid lineage reached image inspection"))
    monkeypatch.setattr("argus.anomaly.trainer.revalidate_scene_training_input", revalidate)
    monkeypatch.setattr(cv2, "imread", reader)

    with pytest.raises(ValueError, match="baseline identity changed"):
        ModelTrainer(SimpleNamespace())._validate_baseline_quality(
            untrusted.images_dir, scene_training_input=untrusted,
        )
    revalidate.assert_called_once_with(untrusted)
    reader.assert_not_called()


def test_quality_wrapper_does_not_adapt_a_different_directory(tmp_path, monkeypatch):
    verified = _untrusted_input(tmp_path)
    revalidate = MagicMock(return_value=verified)
    reader = MagicMock(side_effect=AssertionError("mismatched path reached image inspection"))
    monkeypatch.setattr("argus.anomaly.trainer.revalidate_scene_training_input", revalidate)
    monkeypatch.setattr(cv2, "imread", reader)
    wrong = verified.images_dir.parent / "other-images"
    wrong.mkdir()

    with pytest.raises((OSError, ValueError)):
        ModelTrainer(SimpleNamespace())._validate_baseline_quality(wrong, scene_training_input=verified)
    revalidate.assert_called_once_with(verified)
    reader.assert_not_called()


def test_quality_wrapper_rejects_inconsistent_verified_count(tmp_path, monkeypatch):
    _trainer, verified, _images_list, _revalidate, _reader_mock, _manifest = _wrapped_scene_case(tmp_path, monkeypatch)
    verified = replace(verified, image_count=149)
    reader = MagicMock(side_effect=AssertionError("inconsistent count reached image inspection"))
    monkeypatch.setattr("argus.anomaly.trainer.revalidate_scene_training_input", lambda _: verified)
    monkeypatch.setattr(cv2, "imread", reader)

    with pytest.raises((OSError, ValueError)):
        ModelTrainer(SimpleNamespace())._validate_baseline_quality(
            verified.images_dir, scene_training_input=verified,
        )
    reader.assert_not_called()


def _wrapped_scene_case(tmp_path, monkeypatch):
    """Exercise the real wrapper with mocked authority, not a fake inspector."""
    verified = _untrusted_input(tmp_path)
    images = _images(verified.images_dir)
    for image in images:
        image.write_bytes(b"mocked-image-reader-no-real-image")
    manifest = {
        "schema_version": "argus.scene-profile-baseline.v3",
        "parameters": {"scene_guard": asdict(SceneGuardConfig())},
        "reference": {"height": SHAPE[0], "width": SHAPE[1]},
        "split_contract": {
            "declared_session_ids": ["session-a", "session-b"],
            "accepted_time_span_seconds": {"train": 150.0, "calibration": 150.0},
        },
    }
    raw = json.dumps(manifest, sort_keys=True).encode("utf-8")
    (verified.root / "manifest.json").write_bytes(raw)
    verified = replace(verified, lineage={
        **verified.lineage, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
    })
    revalidate = MagicMock(side_effect=[verified, verified])
    monkeypatch.setattr("argus.anomaly.trainer.revalidate_scene_training_input", revalidate)
    reader = _reader(monkeypatch, images)
    return ModelTrainer(SimpleNamespace()), verified, images, revalidate, reader, manifest


def test_real_quality_wrapper_uses_complete_frozen_static_context(tmp_path, monkeypatch):
    trainer, verified, images, revalidate, reader, _manifest = _wrapped_scene_case(tmp_path, monkeypatch)

    report = trainer._validate_baseline_quality(verified.images_dir, scene_training_input=verified)

    assert report["passed"] is True and report["integrity_passed"] is True
    assert report["policy_version"] == "argus.baseline-quality.scene-static.v1"
    assert report["scope"] == "observed_fixed_scene_only"
    assert report["production_readiness"] == "not_evaluated"
    assert report["source_session_count"] == 2
    assert report["accepted_span_seconds"] == 300.0
    assert report["sample_count"] == len(images) and report["readable_count"] == len(images)
    assert report["scene_workflow"] == verified.lineage
    assert "high_perceptual_redundancy" in report["warning_codes"]
    assert revalidate.call_count == 2
    assert all(call.args == (verified,) for call in revalidate.call_args_list)
    assert reader.call_count == len(images)


def test_wrapper_rejects_identity_changed_during_quality_scan(tmp_path, monkeypatch):
    trainer, verified, images, revalidate, reader, _manifest = _wrapped_scene_case(tmp_path, monkeypatch)
    revalidate.side_effect = [verified, ValueError("frozen baseline changed while scanning")]

    with pytest.raises(ValueError, match="frozen baseline changed while scanning"):
        trainer._validate_baseline_quality(verified.images_dir, scene_training_input=verified)

    assert reader.call_count == len(images)
    assert revalidate.call_count == 2


def test_wrapper_rejects_manifest_bytes_changed_after_initial_identity_check(tmp_path, monkeypatch):
    trainer, verified, _images_list, revalidate, reader, manifest = _wrapped_scene_case(tmp_path, monkeypatch)
    manifest["reference"]["width"] += 1
    (verified.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest changed"):
        trainer._validate_baseline_quality(verified.images_dir, scene_training_input=verified)

    revalidate.assert_called_once_with(verified)
    reader.assert_not_called()


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(schema_version="argus.scene-profile-baseline.v2"),
    lambda value: value["reference"].update(width=True),
    lambda value: value["parameters"]["scene_guard"].update(min_contrast_std=float("nan")),
    lambda value: value["split_contract"].pop("accepted_time_span_seconds"),
    lambda value: value["split_contract"].update(declared_session_ids=["only-session"]),
], ids=["legacy-schema", "boolean-dimension", "nonfinite-policy", "missing-coverage", "single-session"])
def test_wrapper_does_not_trust_incomplete_context_even_with_hash_bound_bytes(tmp_path, monkeypatch, mutation):
    trainer, verified, _images_list, revalidate, reader, manifest = _wrapped_scene_case(tmp_path, monkeypatch)
    mutation(manifest)
    raw = json.dumps(manifest, sort_keys=True).encode("utf-8")
    (verified.root / "manifest.json").write_bytes(raw)
    verified = replace(verified, lineage={**verified.lineage, "manifest_sha256": hashlib.sha256(raw).hexdigest()})
    revalidate.side_effect = None
    revalidate.return_value = verified

    with pytest.raises((TypeError, ValueError)):
        trainer._validate_baseline_quality(verified.images_dir, scene_training_input=verified)

    reader.assert_not_called()


def test_adjacent_v3_metadata_alone_does_not_select_static_policy(tmp_path, monkeypatch):
    trainer, verified, images, revalidate, reader, _manifest = _wrapped_scene_case(tmp_path, monkeypatch)

    report = trainer._validate_baseline_quality(verified.images_dir)

    assert report["passed"] is False
    assert report["policy_version"] != "argus.baseline-quality.scene-static.v1"
    assert report["brightness_std"] <= 2.0 and report["near_duplicate_rate"] >= 0.8
    revalidate.assert_not_called()
    assert reader.call_count == len(images)
