"""Prevent optional learned matching from silently entering the live safety gate."""

import ast
from pathlib import Path

import pytest


_FORBIDDEN_MODULE_PARTS = frozenset({
    "lightglue", "lightglue_matcher", "scene_guard_shadow", "scene_guard_comparison",
})
_DYNAMIC_LOADERS = frozenset({"importlib.import_module", "builtins.__import__"})


def _shadow_dependency_violations(
    source: bytes | str, *, filename: str,
    forbidden_parts: frozenset[str] = _FORBIDDEN_MODULE_PARTS,
) -> list[int]:
    """Inspect imports without loading application code or optional model packages.

    This is deliberately conservative source analysis, not a Python sandbox.  It
    follows ordinary loader aliases and static import targets, including imports
    inside functions or TYPE_CHECKING blocks.  Truly computed import targets
    still require review; strings and loader aliases must not bypass the guard.
    """
    tree = ast.parse(source, filename=filename)
    aliases: dict[str, set[str]] = {"__import__": {"builtins.__import__"}}
    bindings: dict[str, list[ast.AST]] = {}
    nodes = list(ast.walk(tree))
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                aliases.setdefault(name, set()).add(
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                aliases.setdefault(alias.asname or alias.name, set()).add(
                    (node.module + "." if node.module else "") + alias.name
                )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if node.value is not None:
                for target in targets:
                    if isinstance(target, ast.Name):
                        bindings.setdefault(target.id, []).append(node.value)
        elif isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(node.target, ast.Name):
            bindings.setdefault(node.target.id, []).append(node.iter)

    def strings(node: ast.AST | None, seen: frozenset[str] = frozenset()) -> set[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name) and node.id not in seen:
            return set().union(*(
                strings(value, seen | {node.id}) for value in bindings.get(node.id, [])
            ))
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return set().union(*(strings(item, seen) for item in node.elts))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return {
                left + right for left in strings(node.left, seen) for right in strings(node.right, seen)
            }
        if isinstance(node, ast.JoinedStr):
            result = {""}
            for value in node.values:
                part = strings(
                    value.value if isinstance(value, ast.FormattedValue) else value, seen,
                )
                result = {prefix + suffix for prefix in result for suffix in part}
            return result
        return set()

    def symbols(node: ast.AST, seen: frozenset[str] = frozenset()) -> set[str]:
        if isinstance(node, ast.Name) and node.id not in seen:
            return aliases.get(node.id, set()).union(*(
                symbols(value, seen | {node.id}) for value in bindings.get(node.id, [])
            ))
        if isinstance(node, ast.Attribute):
            return {name + "." + node.attr for name in symbols(node.value, seen)}
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
        ):
            return {
                name + "." + attribute
                for name in symbols(node.args[0], seen)
                for attribute in strings(node.args[1], seen)
            }
        return set()

    # Resolve aliases even inside functions or declared after a function.  Name
    # cycles stop at ``seen`` rather than repeatedly expanding string values.
    violations: list[int] = []
    for node in nodes:
        names: set[str] = set()
        if isinstance(node, ast.Import):
            names = {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names = {node.module or "", *(alias.name for alias in node.names)}
        elif isinstance(node, ast.Call) and symbols(node.func).intersection(_DYNAMIC_LOADERS):
            target = node.args[0] if node.args else next(
                (keyword.value for keyword in node.keywords if keyword.arg == "name"), None,
            )
            names.update(strings(target))
            # __import__('argus.preprocessing', fromlist=('scene_guard_shadow',))
            # imports the child even though the parent name itself is harmless.
            fromlist = node.args[3] if len(node.args) > 3 else next(
                (keyword.value for keyword in node.keywords if keyword.arg == "fromlist"), None,
            )
            names.update(strings(fromlist))
        if any(forbidden_parts.intersection(name.split(".")) for name in names):
            violations.append(node.lineno)
    return sorted(set(violations))


def test_optional_comparison_modules_are_not_imported_by_production_modules():
    repository = Path(__file__).resolve().parents[2]
    root = repository / "src" / "argus"
    offline = {
        root / "preprocessing" / "lightglue_matcher.py",
        root / "preprocessing" / "scene_guard_shadow.py",
        root / "validation" / "scene_guard_comparison.py",
        repository / "scripts" / "compare_scene_guard.py",
    }
    violations = []
    # Startup, regression and trusted-soak scripts are also production paths;
    # only the explicitly invoked comparison command may reach the experiment.
    paths = [*root.rglob("*.py"), *(repository / "scripts").rglob("*.py")]
    for path in paths:
        if path in offline:
            continue
        for line in _shadow_dependency_violations(path.read_bytes(), filename=str(path)):
            violations.append(f"{path.relative_to(repository)}:{line}")
    assert not violations, "LightGlue must remain offline-only: " + ", ".join(violations)


def test_scene_comparison_review_and_fiftyone_stay_out_of_production_imports():
    repository = Path(__file__).resolve().parents[2]
    root = repository / "src" / "argus"
    review_entry_points = {
        root / "integrations" / "scene_comparison.py",
        root / "integrations" / "fiftyone.py",
        root / "integrations" / "scene_review_images.py",
        root / "integrations" / "fiftyone_scene_images.py",
        repository / "scripts" / "video_review.py",
    }
    violations = []
    paths = [*root.rglob("*.py"), *(repository / "scripts").rglob("*.py")]
    for path in paths:
        if path in review_entry_points:
            continue
        for line in _shadow_dependency_violations(
            path.read_bytes(), filename=str(path),
            forbidden_parts=frozenset({
                "scene_comparison", "fiftyone", "scene_review_images", "fiftyone_scene_images",
            }),
        ):
            violations.append(f"{path.relative_to(repository)}:{line}")
    assert not violations, "Scene comparison review must remain offline-only: " + ", ".join(violations)


@pytest.mark.parametrize("source", [
    "from argus.integrations import scene_comparison",
    "from argus.integrations.fiftyone import import_bundle_to_fiftyone",
    "import importlib as loader\nloader.import_module(name='fiftyone')",
    "from importlib import import_module as load\n"
    "load('argus.integrations.' + 'scene_comparison')",
    "from argus.integrations.scene_review_images import prepare_scene_review_images",
    "import importlib\nimportlib.import_module('argus.integrations.fiftyone_scene_images')",
])
def test_review_import_guard_covers_direct_and_dynamic_offline_dependencies(source: str):
    assert _shadow_dependency_violations(
        source, filename="production-fixture.py",
        forbidden_parts=frozenset({
            "scene_comparison", "fiftyone", "scene_review_images", "fiftyone_scene_images",
        }),
    )


def test_scene_review_bridge_cannot_load_inference_or_decode_dependencies():
    path = Path(__file__).resolve().parents[2] / "src" / "argus" / "integrations" / "scene_comparison.py"
    assert not _shadow_dependency_violations(
        path.read_bytes(), filename=str(path),
        forbidden_parts=frozenset({
            "torch", "cv2", "av", "fiftyone", "lightglue", "lightglue_matcher",
            "scene_guard", "scene_guard_shadow", "scene_guard_comparison", "strict_video_decoder",
        }),
    ), "Report review binding must not perform inference, decoding, or production gate loading"


@pytest.mark.parametrize("relative_path", [
    "src/argus/integrations/contracts.py",
    "src/argus/integrations/scene_comparison.py",
    "src/argus/integrations/fiftyone.py",
    "src/argus/integrations/fiftyone_scene_images.py",
])
def test_scene_review_import_graph_cannot_reach_production_mutation_paths(relative_path: str):
    repository = Path(__file__).resolve().parents[2]
    path = repository / relative_path
    # These modules are used by the report-only preflight before importing the
    # optional FiftyOne SDK. Transitive imports through the common contract must
    # not start a camera, load a gate/model, or gain a production write service.
    assert not _shadow_dependency_violations(
        path.read_bytes(), filename=str(path),
        forbidden_parts=frozenset({
            "storage", "dashboard", "capture", "pipeline", "runner", "jobs",
            "anomaly", "alerts", "torch", "cv2", "av", "lightglue",
            "lightglue_matcher", "scene_guard", "scene_guard_shadow",
            "scene_guard_comparison", "strict_video_decoder",
        }),
    ), "Offline report review must not acquire production gate/model/write dependencies"


def test_human_scene_review_fields_are_not_consumed_as_production_or_annotation_inputs():
    repository = Path(__file__).resolve().parents[2]
    root = repository / "src" / "argus"
    review_adapters = {
        root / "integrations" / "fiftyone.py",
        root / "integrations" / "fiftyone_scene_images.py",
    }
    violations = []
    # Only the explicit diagnostic adapters own these fields. In particular CVAT,
    # annotation export, training and release code must not infer approval/GT
    # from a reviewer marking a frame as same_pose, moved or uncertain.
    exact_fields = {"argus_scene_reviewed_report_sha256"}
    for path in root.rglob("*.py"):
        if path in review_adapters:
            continue
        tree = ast.parse(path.read_bytes(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in exact_fields or node.value.startswith("argus_scene_human_"):
                    violations.append(f"{path.relative_to(repository)}:{node.lineno}")
            elif isinstance(node, ast.Attribute) and (
                node.attr in exact_fields or node.attr.startswith("argus_scene_human_")
            ):
                violations.append(f"{path.relative_to(repository)}:{node.lineno}")
    assert not violations, "Human scene review must remain diagnostic-only: " + ", ".join(violations)


def test_scene_image_extraction_has_no_gate_or_production_write_dependencies():
    path = (
        Path(__file__).resolve().parents[2]
        / "src" / "argus" / "integrations" / "scene_review_images.py"
    )
    # Only the explicit image preparation command may decode a source. It must
    # never load SceneGuard/LightGlue or acquire a release/profile write service.
    assert not _shadow_dependency_violations(
        path.read_bytes(), filename=str(path),
        forbidden_parts=frozenset({
            "storage", "dashboard", "capture", "pipeline", "runner", "jobs",
            "anomaly", "alerts", "torch", "cv2", "av", "lightglue",
            "lightglue_matcher", "scene_guard", "scene_profile", "scene_guard_shadow",
            "scene_guard_comparison",
        }),
    ), "Image review extraction must not acquire production/model dependencies"
    tree = ast.parse(path.read_bytes(), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert not _shadow_dependency_violations(
                ast.unparse(node), filename=str(path),
                forbidden_parts=frozenset({"strict_video_decoder", "numpy"}),
            ), "Decoding dependencies must be lazy, not loaded by artifact import/preflight"


@pytest.mark.parametrize("source", [
    "import lightglue as matching",
    "from argus.preprocessing import scene_guard_shadow as comparison",
    "from .scene_guard_shadow import LightGlueSceneGuard",
    "from importlib import import_module as load\nload('argus.preprocessing.scene_guard_shadow')",
    "import importlib as loader\nloader.import_module(name='lightglue')",
    "import importlib\nload = importlib.import_module\nload('lightglue')",
    "import importlib\ngetattr(importlib, 'import_module')('lightglue')",
    "from builtins import __import__ as load\nload('lightglue')",
    "import builtins as loader\nloader.__import__(name='lightglue')",
    "__import__('argus.preprocessing', fromlist=('scene_guard_shadow',))",
    "__import__('argus.preprocessing', {}, {}, ('scene_guard_shadow',))",
    "import importlib\nMODULE = 'argus.preprocessing.' + 'scene_guard_shadow'\n"
    "importlib.import_module(MODULE)",
    "import importlib\nMODULE = 'scene_guard_shadow'\n"
    "importlib.import_module(f'argus.preprocessing.{MODULE}')",
    "import importlib\nfor name in ('torch', 'lightglue'):\n    importlib.import_module(name)",
    "import importlib\nimportlib.import_module('.scene_guard_shadow', package='argus.preprocessing')",
])
def test_import_guard_detects_alias_keyword_and_computed_literal_targets(source: str):
    assert _shadow_dependency_violations(source, filename="fixture.py")


@pytest.mark.parametrize("source", [
    "from argus.preprocessing.scene_guard import SceneGuard",
    "from importlib import import_module as load\nload('torch')",
    "import importlib\nfor name in ('faiss', 'timm', 'torch'):\n    importlib.import_module(name)",
    "# lightglue must stay offline\nmessage = 'argus.preprocessing.scene_guard_shadow'",
])
def test_import_guard_does_not_flag_unrelated_imports_or_documentation(source: str):
    assert not _shadow_dependency_violations(source, filename="fixture.py")
