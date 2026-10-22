"""Configured scene baseline roots reach the runtime without a default fallback."""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from argus.core.pipeline import DetectionPipeline


def _runtime(root: Path | str | None):
    pipeline = DetectionPipeline.__new__(DetectionPipeline)
    pipeline.camera_config = SimpleNamespace(camera_id="cam1")
    if root is not None:
        pipeline._baselines_dir = Path(root)
    return pipeline


def _baseline(root: Path, payload: bytes = b"approved-reference") -> Path:
    zone = root / "cam1" / "default"
    version = zone / "v003"
    version.mkdir(parents=True)
    (zone / "current.txt").write_text("v003", encoding="utf-8")
    (version / "reference.png").write_bytes(payload)
    return version


def _profile(payload: bytes = b"approved-reference"):
    return SimpleNamespace(
        source_baseline_name="reference.png",
        source_baseline_sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_custom_baseline_root_is_used_for_the_existing_profile_hash_check(tmp_path):
    root = tmp_path / "isolated-baselines"
    expected = _baseline(root)
    runtime = _runtime(root)
    assert runtime._configured_baseline_dir("cam1") == expected
    assert runtime._validate_profile_active_baseline(_profile()) is None


def test_missing_custom_root_does_not_adopt_a_matching_default_baseline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _baseline(tmp_path / "data" / "baselines")
    runtime = _runtime(tmp_path / "missing-controlled-root")
    assert runtime._configured_baseline_dir("cam1") is None
    assert runtime._validate_profile_active_baseline(_profile()) == (
        "active scene profile cannot be tied to an active baseline"
    )


def test_custom_root_hash_mismatch_cannot_fall_back_to_matching_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _baseline(tmp_path / "data" / "baselines")
    root = tmp_path / "isolated-baselines"
    _baseline(root, b"different-scene-reference")
    runtime = _runtime(root)
    assert "baseline hash is not present" in runtime._validate_profile_active_baseline(_profile())


def test_default_lookup_retains_the_existing_single_argument_adapter_contract(monkeypatch):
    runtime = _runtime(None)
    lookup = Mock(return_value=Path("legacy-fixture"))
    monkeypatch.setattr(runtime, "_find_baseline_dir", lookup)
    assert runtime._configured_baseline_dir("cam1") == Path("legacy-fixture")
    lookup.assert_called_once_with("cam1")


def test_legacy_static_lookup_remains_available_with_its_default_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected = _baseline(tmp_path / "data" / "baselines")
    assert DetectionPipeline._find_baseline_dir("cam1").resolve() == expected


def _tree(relative: str) -> ast.Module:
    repository = Path(__file__).resolve().parents[2]
    return ast.parse((repository / relative).read_text(encoding="utf-8-sig"))


def _call_keyword(tree: ast.Module, callee: str, keyword: str) -> ast.expr:
    calls = [node for node in ast.walk(tree) if (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == callee
    )]
    assert len(calls) == 1
    values = [argument.value for argument in calls[0].keywords if argument.arg == keyword]
    assert len(values) == 1
    return values[0]


def test_storage_baseline_root_is_forwarded_from_entrypoint_to_manager_and_pipeline():
    main_value = _call_keyword(_tree("src/argus/__main__.py"), "CameraManager", "baselines_dir")
    assert ast.unparse(main_value) == "config.storage.baselines_dir"
    manager_value = _call_keyword(
        _tree("src/argus/capture/manager.py"), "DetectionPipeline", "baselines_dir",
    )
    assert ast.unparse(manager_value) == "self._baselines_dir"


def test_direct_replay_and_simulation_entrypoints_use_the_same_configured_baseline_root():
    for relative in ("src/argus/validation/pipeline_video_eval.py", "scripts/simulate_camera.py"):
        value = _call_keyword(_tree(relative), "DetectionPipeline", "baselines_dir")
        assert ast.unparse(value) == "config.storage.baselines_dir"


def test_only_the_configured_adapter_calls_the_default_compatible_baseline_lookup():
    tree = _tree("src/argus/core/pipeline.py")
    pipeline = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DetectionPipeline"
    )
    callers = []
    configured_reads = 0
    for method in pipeline.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        for node in ast.walk(method):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "_find_baseline_dir":
                    callers.append(method.name)
                if node.func.attr == "_configured_baseline_dir":
                    configured_reads += 1
    assert callers and set(callers) == {"_configured_baseline_dir"}
    assert configured_reads == 3


def test_runtime_uses_the_shared_scene_guard_mapping_without_geometry_default_drift():
    tree = _tree("src/argus/core/pipeline.py")
    pipeline = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DetectionPipeline"
    )
    initialize = next(
        node for node in pipeline.body
        if isinstance(node, ast.FunctionDef) and node.name == "_initialize_scene_guard"
    )
    mapping_calls = [node for node in ast.walk(initialize) if (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "runtime_scene_guard_config"
    )]
    assert len(mapping_calls) == 1
    assert [ast.unparse(argument) for argument in mapping_calls[0].args] == ["cfg"]
    assert not any(
        isinstance(node, ast.Name) and node.id == "SceneGuardConfig"
        for node in ast.walk(initialize)
    )

    mapper = _tree("src/argus/preprocessing/scene_guard_policy.py")
    constructor = next(node for node in ast.walk(mapper) if (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "SceneGuardConfig"
    ))
    expected_fields = {
        "process_width", "feature_type", "min_matches", "min_inliers", "min_inlier_ratio",
        "max_reprojection_error_px", "min_coverage_ratio", "min_overlap_ratio",
        "stable_max_shift_px", "stable_max_corner_displacement_px", "stable_max_rotation_deg",
        "stable_max_scale_delta", "alignable_max_shift_px", "alignable_max_rotation_deg",
        "alignable_max_scale_delta", "consecutive_matches",
    }
    mapped = {keyword.arg: ast.unparse(keyword.value) for keyword in constructor.keywords}
    assert mapped == {
        **{name: f"policy.{name}" for name in expected_fields},
        "return_aligned_frame": "False",
    }
