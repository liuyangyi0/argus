"""Shared builder/CLI contracts; no real decoder or model is used by these tests."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import Mock

from argus.validation import scene_profile_baseline as service
from scripts import build_scene_profile_baseline as legacy_cli


def test_legacy_cli_import_is_the_canonical_builder_module():
    # Keep dependency injection and validation fixes on the same implementation,
    # rather than exposing forwarding functions with a separate global scope.
    assert legacy_cli is service
    assert legacy_cli.build_scene_profile_baseline is service.build_scene_profile_baseline


def test_legacy_cli_delegates_strict_build_to_shared_service(monkeypatch, capsys):
    manifest = {
        "dataset_status": "training_ready",
        "training_ready": True,
        "training_readiness_scope": "offline_training_only",
        "validation_only": True,
        "release_eligible": False,
        "selection": {"accepted_images": 120, "truncated_by_max_images": False},
        "corpus_evidence_complete": True,
        "decoder_evidence": {"sufficient_for_model_release": True},
    }
    build = Mock(return_value=manifest)
    monkeypatch.setattr(service, "build_scene_profile_baseline", build)

    assert legacy_cli.main([
        "--camera-id", "camera-01",
        "--scene-profile-id", "scene-01",
        "--video", "recording-train.mp4",
        "--video", "recording-calibration.mp4",
        "--annotations", "review-train.json",
        "--annotations", "review-calibration.json",
        "--reference", "reference.png",
        "--output-dir", "output-baseline",
        "--sample-every", "30",
        "--max-images", "600",
        "--strict-release-decoder",
    ]) == 0

    build.assert_called_once()
    arguments = build.call_args.kwargs
    assert arguments["camera_id"] == "camera-01"
    assert arguments["scene_profile_id"] == "scene-01"
    assert arguments["videos"] == [Path("recording-train.mp4"), Path("recording-calibration.mp4")]
    assert arguments["annotations"] == ["review-train.json", "review-calibration.json"]
    assert arguments["strict_release_decoder"] is True
    assert arguments["allow_alignable"] is False
    output = capsys.readouterr()
    assert '"release_eligible": false' in output.out
    assert output.err == ""


def test_shared_service_preserves_explicit_offline_defaults_and_no_script_dependency():
    signature = inspect.signature(service.build_scene_profile_baseline)
    assert signature.parameters["strict_release_decoder"].default is False
    assert signature.parameters["allow_alignable"].default is False
    assert service._SCHEMA_VERSION == "argus.scene-profile-baseline.v2"
    assert service._STRICT_SCHEMA_VERSION == "argus.scene-profile-baseline.v3"

    tree = ast.parse(Path(service.__file__).read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module is not None
        and (node.module == "scripts" or node.module.startswith("scripts."))
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.Import)
        and any(alias.name == "scripts" or alias.name.startswith("scripts.") for alias in node.names)
        for node in ast.walk(tree)
    )
