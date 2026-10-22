"""Adversarial report destinations must never become production state writes.

These are regression sources for the isolated comparison command. They use no
model, decoder, camera service or production profile mutation API.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from argus.validation import scene_guard_comparison as comparison


def _report(source: Path, *profile_roots: Path) -> dict:
    return {
        "schema_version": comparison.COMPARISON_SCHEMA,
        "diagnostic_only": True,
        "gating_effect": "none",
        "release_authority": False,
        "protected_paths": [str(source)],
        "protected_directories": [str(path) for path in profile_roots],
    }


@pytest.fixture
def report_inputs(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"immutable video fixture")
    selected_root = tmp_path / "selected-profiles"
    other_root = tmp_path / "other-camera-profiles"
    selected_root.mkdir()
    other_root.mkdir()
    return source, selected_root, other_root


def test_runner_protects_every_configured_camera_profile_root(monkeypatch, tmp_path):
    inputs = {}
    for name in ("config", "video", "extractor", "matcher"):
        inputs[name] = tmp_path / name
        inputs[name].write_bytes(name.encode())
    selected_root = tmp_path / "selected-profiles"
    other_root = tmp_path / "other-camera-profiles"

    def camera(camera_id, profile_root):
        return SimpleNamespace(camera_id=camera_id, scene_guard=SimpleNamespace(
            profiles_dir=str(profile_root), enabled=True, allow_alignable=False,
            require_profile_model_binding=True, require_operator_approved_profile=True,
            model_dump=lambda: {},
        ))

    config = SimpleNamespace(
        cameras=[camera("selected", selected_root), camera("other", other_root)],
        model_dump=lambda **_kwargs: {"fixture": "config"},
    )
    monkeypatch.setattr(comparison, "load_config", lambda _path: config)
    monkeypatch.setattr(comparison, "_profile_reference", lambda *_args: (
        SimpleNamespace(copy=lambda: object()), {"source_file_sha256": {}}, [],
    ))
    monkeypatch.setattr(comparison, "SceneGuard", lambda *_args, **_kwargs: object())

    def unavailable_decoder(*_args, **_kwargs):
        raise RuntimeError("decoder deliberately not invoked by this destination-only test")

    monkeypatch.setattr(comparison, "StrictVideoDecoder", unavailable_decoder)
    report = comparison.run_scene_guard_comparison(
        config_path=inputs["config"], camera_id="selected", video_path=inputs["video"],
        profile_id="approved", extractor_weights=inputs["extractor"],
        extractor_sha256=comparison.file_sha256(inputs["extractor"]),
        matcher_weights=inputs["matcher"], matcher_sha256=comparison.file_sha256(inputs["matcher"]),
    )

    assert set(report["protected_directories"]) == {
        str(selected_root.resolve()), str(other_root.resolve()),
    }
    with pytest.raises(ValueError):
        comparison.write_comparison_report(report, other_root / "other" / "active.json")
    assert not other_root.exists()


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("root_index", [1, 2])
def test_cannot_create_or_overwrite_another_camera_active_pointer(
    report_inputs, existing, root_index,
):
    source, selected_root, other_root = report_inputs
    output = report_inputs[root_index] / "not-selected-camera" / "active.json"
    original = b'{"profile_id":"production-approved"}'
    if existing:
        output.parent.mkdir()
        output.write_bytes(original)

    with pytest.raises(ValueError):
        comparison.write_comparison_report(
            _report(source, selected_root, other_root), output, overwrite=True,
        )

    if existing:
        assert output.read_bytes() == original
    else:
        assert not output.exists()
        assert not output.parent.exists()
    assert source.read_bytes() == b"immutable video fixture"


def test_valid_old_diagnostic_report_inside_profile_root_is_still_protected(report_inputs):
    source, selected_root, other_root = report_inputs
    output = other_root / "diagnostic-looking-metadata.json"
    previous = json.dumps(_report(source, selected_root, other_root)).encode()
    output.write_bytes(previous)

    with pytest.raises(ValueError):
        comparison.write_comparison_report(
            _report(source, selected_root, other_root), output, overwrite=True,
        )

    assert output.read_bytes() == previous


def test_parent_symlink_cannot_hide_a_profile_destination(report_inputs, tmp_path):
    source, selected_root, other_root = report_inputs
    alias = tmp_path / "looks-like-reports"
    try:
        alias.symlink_to(other_root, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this filesystem")
    output = alias / "other-camera" / "active.json"

    with pytest.raises(ValueError):
        comparison.write_comparison_report(
            _report(source, selected_root, other_root), output,
        )

    assert not (other_root / "other-camera").exists()


@pytest.mark.parametrize("previous", [
    "database bytes are not a report",
    '{"schema_version":"argus.scene-profile.v1"}',
    '{"schema_version":"argus.scene-guard-comparison.v1","diagnostic_only":false,'
    '"gating_effect":"none","release_authority":false}',
    '{"schema_version":"argus.scene-guard-comparison.v1","diagnostic_only":true,'
    '"gating_effect":"production","release_authority":false}',
    '{"schema_version":"argus.scene-guard-comparison.v1","diagnostic_only":true,'
    '"gating_effect":"none","release_authority":true}',
    '[]',
    'null',
])
def test_overwrite_does_not_authorize_replacing_non_report_files(
    report_inputs, tmp_path, previous,
):
    source, selected_root, other_root = report_inputs
    output = tmp_path / "model-metadata.json"
    output.write_text(previous, encoding="utf-8")

    with pytest.raises(ValueError):
        comparison.write_comparison_report(
            _report(source, selected_root, other_root), output, overwrite=True,
        )

    assert output.read_text(encoding="utf-8") == previous


def test_report_only_overwrite_remains_available_outside_profile_roots(report_inputs, tmp_path):
    source, selected_root, other_root = report_inputs
    output = tmp_path / "reports" / "comparison.json"
    payload = _report(source, selected_root, other_root)
    comparison.write_comparison_report(payload, output)
    payload["review_note"] = "replacement of a diagnostic report only"
    comparison.write_comparison_report(payload, output, overwrite=True)
    assert json.loads(output.read_text(encoding="utf-8"))["review_note"] == payload["review_note"]


@pytest.mark.parametrize("directories", [None, [], [""], [None], "not-a-list"])
def test_report_writer_requires_explicit_nonempty_protected_directories(
    report_inputs, tmp_path, directories,
):
    source, selected_root, _other_root = report_inputs
    payload = _report(source, selected_root)
    payload["protected_directories"] = directories
    output = tmp_path / "comparison.json"

    with pytest.raises(ValueError):
        comparison.write_comparison_report(payload, output)

    assert not output.exists()


def test_report_without_directory_protection_cannot_write(report_inputs, tmp_path):
    source, selected_root, _other_root = report_inputs
    payload = _report(source, selected_root)
    del payload["protected_directories"]
    output = tmp_path / "comparison.json"

    with pytest.raises(ValueError):
        comparison.write_comparison_report(payload, output)

    assert not output.exists()


def test_cli_rejects_arbitrary_overwrite_before_importing_the_runner(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts" / "compare_scene_guard.py"
    spec = importlib.util.spec_from_file_location("adversarial_scene_guard_cli", path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    output = tmp_path / "model-metadata.json"
    output.write_text('{"stage":"production"}', encoding="utf-8")
    original_import = __import__
    runner_imports = []

    def reject_runner_import(name, *args, **kwargs):
        if name == "argus.validation.scene_guard_comparison":
            runner_imports.append(name)
            raise AssertionError("unsafe output reached the model runner import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", reject_runner_import)
    code = cli.main([
        "--config", str(tmp_path / "config.yaml"), "--camera-id", "cam",
        "--video", str(tmp_path / "video.mp4"), "--profile-id", "approved",
        "--extractor-weights", str(tmp_path / "extractor.pth"),
        "--extractor-sha256", "a" * 64,
        "--matcher-weights", str(tmp_path / "matcher.pth"),
        "--matcher-sha256", "b" * 64,
        "--output", str(output), "--overwrite",
    ])

    assert code == 2
    assert not runner_imports
    assert output.read_text(encoding="utf-8") == '{"stage":"production"}'
