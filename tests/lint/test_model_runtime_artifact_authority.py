"""Keep runtime model loading behind persisted release authority."""

from __future__ import annotations

import ast
from pathlib import Path


MANAGER_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "argus"
    / "capture"
    / "manager.py"
)


def _manager_method(name: str) -> ast.FunctionDef:
    tree = ast.parse(
        MANAGER_PATH.read_text(encoding="utf-8"),
        filename=str(MANAGER_PATH),
    )
    manager = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CameraManager"
    )
    return next(
        node
        for node in manager.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _called_attributes(method: ast.FunctionDef) -> set[str]:
    return {
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def test_registry_resolution_never_calls_discovery_helpers() -> None:
    method = _manager_method("_resolve_model_record_path")
    calls = _called_attributes(method)

    assert "_find_model_in_dir" not in calls
    assert "_find_model" not in calls
    assert "glob" not in calls
    assert "rglob" not in calls


def test_registry_resolution_requires_persisted_digest_and_root_artifact() -> None:
    source = ast.get_source_segment(
        MANAGER_PATH.read_text(encoding="utf-8"),
        _manager_method("_resolve_model_record_path"),
    )

    assert source is not None
    assert 'registry_path / "model.pt"' in source
    assert 'registry_path / "model.xml"' in source
    assert "_persisted_runtime_artifact_sha256" in source
    assert "registry_sha256 != expected_sha256" in source
    assert "artifact_sha256 != expected_sha256" in source


def test_legacy_runtime_mutators_are_fail_closed() -> None:
    forbidden_calls = {
        "reload_anomaly_model",
        "set_model_version_id",
        "set_version_tag",
        "prepare_anomaly_model_release",
        "commit_anomaly_model_release",
        "set_shadow_runner",
    }
    for method_name in ("reload_model", "apply_model_release_state"):
        method = _manager_method(method_name)
        calls = _called_attributes(method)
        assert calls.isdisjoint(forbidden_calls), (
            f"CameraManager.{method_name} bypasses ModelReleaseService: "
            f"{sorted(calls & forbidden_calls)}"
        )

    source = MANAGER_PATH.read_text(encoding="utf-8")
    assert source.count('"release_service_required"') >= 2


def test_model_release_service_prepared_boundary_remains_available() -> None:
    method = _manager_method("prepare_model_release")
    calls = _called_attributes(method)

    assert "prepare_anomaly_model_release" in calls
    assert "reload_anomaly_model" not in calls
