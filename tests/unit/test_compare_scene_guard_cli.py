"""CLI contract regressions; model execution is replaced by a fake runner."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[2] / "scripts" / "compare_scene_guard.py"
    spec = importlib.util.spec_from_file_location("argus_scene_guard_comparison_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--config", str(tmp_path / "config.yaml"),
        "--camera-id", "camera-01", "--profile-id", "approved-profile",
        "--video", str(tmp_path / "video.mp4"),
        "--extractor-weights", str(tmp_path / "aliked-n16.pth"),
        "--extractor-sha256", "a" * 64,
        "--matcher-weights", str(tmp_path / "aliked_lightglue.pth"),
        "--matcher-sha256", "b" * 64,
        "--output", str(tmp_path / "report.json"), *extra,
    ]


def _runner(monkeypatch, *, report=None, error=None):
    module = ModuleType("argus.validation.scene_guard_comparison")
    run = Mock(return_value=report or {"status": "complete"}, side_effect=error)
    write = Mock()
    module.run_scene_guard_comparison = run
    module.write_comparison_report = write
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return run, write


def test_explicit_run_passes_exact_inputs_and_safe_defaults(cli, monkeypatch, tmp_path):
    run, write = _runner(monkeypatch)
    assert cli.main(_argv(tmp_path)) == 0
    run.assert_called_once_with(
        config_path=tmp_path / "config.yaml", camera_id="camera-01",
        video_path=tmp_path / "video.mp4", profile_id="approved-profile",
        extractor_weights=tmp_path / "aliked-n16.pth", extractor_sha256="a" * 64,
        matcher_weights=tmp_path / "aliked_lightglue.pth", matcher_sha256="b" * 64,
        device="cuda:0", max_keypoints=2048, sample_every=1, max_frames=None,
    )
    write.assert_called_once_with(
        {"status": "complete"}, tmp_path / "report.json", overwrite=False,
    )


def test_disagreement_is_diagnostic_not_command_failure(cli, monkeypatch, tmp_path):
    _runner(monkeypatch, report={"status": "complete", "disagreements": 42})
    assert cli.main(_argv(tmp_path)) == 0


@pytest.mark.parametrize("status", ["partial", "error", "unknown"])
def test_incomplete_report_is_written_but_never_success(cli, monkeypatch, tmp_path, status):
    _, write = _runner(monkeypatch, report={"status": status})
    assert cli.main(_argv(tmp_path)) == 1
    write.assert_called_once()


def test_missing_runtime_dependency_is_visible_failure(cli, monkeypatch, tmp_path, capsys):
    run, write = _runner(monkeypatch, error=ImportError("install the lightglue extra"))
    assert cli.main(_argv(tmp_path)) == 2
    run.assert_called_once()
    write.assert_not_called()
    assert "ImportError" in capsys.readouterr().err


def test_existing_report_is_rejected_before_model_initialization(cli, monkeypatch, tmp_path):
    run, write = _runner(monkeypatch)
    report = tmp_path / "report.json"
    report.write_text("prior report", encoding="utf-8")
    assert cli.main(_argv(tmp_path)) == 2
    run.assert_not_called()
    write.assert_not_called()
    assert report.read_text(encoding="utf-8") == "prior report"


def test_overwrite_only_applies_to_output_report(cli, monkeypatch, tmp_path):
    run, write = _runner(monkeypatch)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "schema_version": "argus.scene-guard-comparison.v1", "diagnostic_only": True,
        "gating_effect": "none", "release_authority": False,
    }), encoding="utf-8")
    assert cli.main(_argv(tmp_path, "--overwrite")) == 0
    run.assert_called_once()
    assert write.call_args.kwargs["overwrite"] is True


@pytest.mark.parametrize("source", [
    "config.yaml", "video.mp4", "aliked-n16.pth", "aliked_lightglue.pth",
])
def test_overwrite_cannot_replace_declared_input(cli, monkeypatch, tmp_path, source):
    run, write = _runner(monkeypatch)
    path = tmp_path / source
    path.write_bytes(b"source")
    assert cli.main(_argv(tmp_path, "--output", str(path), "--overwrite")) == 2
    run.assert_not_called()
    write.assert_not_called()
    assert path.read_bytes() == b"source"


def test_report_write_failure_is_not_reported_complete(cli, monkeypatch, tmp_path):
    _, write = _runner(monkeypatch)
    write.side_effect = OSError("disk full")
    assert cli.main(_argv(tmp_path)) == 2


@pytest.mark.parametrize("args", [
    ["--sample-every", "0"], ["--max-frames", "-1"], ["--max-keypoints", "0"],
    ["--max-keypoints", "31"], ["--max-keypoints", "8193"],
    ["--device", "mps"], ["--extractor-sha256", "unknown"],
    ["--matcher-sha256", "f" * 63],
])
def test_invalid_model_parameters_fail_before_runner(cli, monkeypatch, tmp_path, args):
    run, _ = _runner(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        cli.main(_argv(tmp_path, *args))
    assert exc.value.code == 2
    run.assert_not_called()


def test_help_does_not_import_runner_or_models(cli, monkeypatch):
    original_import = __import__

    def forbid_model_import(name, *args, **kwargs):
        if name.startswith(("argus.validation", "lightglue", "torch", "cv2", "av")):
            raise AssertionError(f"unexpected optional import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", forbid_model_import)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0


def test_module_scope_has_no_optional_or_production_imports():
    path = Path(__file__).resolve().parents[2] / "scripts" / "compare_scene_guard.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            imports.extend(alias.name for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            imports.append(statement.module or "")
    assert all(
        not name.startswith(("argus", "lightglue", "torch", "cv2", "av"))
        for name in imports
    )
