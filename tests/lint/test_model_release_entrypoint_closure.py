"""Guard the single model-release mutation boundary.

Dashboard helpers and scheduled retraining used to call ModelRegistry and
CameraManager mutators directly.  Those calls commit registry state and runtime
state in separate steps.  All supported model stage changes must now enter via
ModelReleaseService.submit_transition (durable HTTP command) or its synchronous
internal request_transition API; legacy commands that cannot yet be
represented by that state machine must fail closed.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _tree(relative_path: str) -> ast.AST:
    path = ROOT / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _attribute_calls(relative_path: str) -> list[tuple[str | None, str, int]]:
    calls: list[tuple[str | None, str, int]] = []
    for node in ast.walk(_tree(relative_path)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        receiver = node.func.value.id if isinstance(node.func.value, ast.Name) else None
        calls.append((receiver, node.func.attr, node.lineno))
    return calls


def test_dashboard_model_runtime_is_read_only() -> None:
    forbidden = {
        "activate",
        "promote",
        "rollback",
        "compensate_failed_production_sync",
        "reload_model",
        "apply_model_release_state",
    }
    calls = _attribute_calls("src/argus/dashboard/model_runtime.py")
    violations = [(name, line) for _, name, line in calls if name in forbidden]
    assert violations == [], (
        "model_runtime must remain read-only; use ModelReleaseService instead: "
        f"{violations}"
    )


def test_model_routes_have_one_release_mutation_api() -> None:
    path = "src/argus/dashboard/routes/models.py"
    calls = _attribute_calls(path)
    forbidden = {
        "activate",
        "promote",
        "rollback",
        "compensate_failed_production_sync",
        "reload_model",
        "apply_model_release_state",
    }
    violations = [(name, line) for _, name, line in calls if name in forbidden]
    assert violations == [], (
        "model routes may not bypass the two-phase release service: "
        f"{violations}"
    )

    service_calls = [
        line for receiver, name, line in calls
        if receiver == "service" and name == "submit_transition"
    ]
    assert len(service_calls) == 2, (
        "promote and retire must be the only supported route mutations and both "
        "must call ModelReleaseService.submit_transition"
    )
    assert not any(
        receiver == "service" and name in {"request_transition", "execute_operation"}
        for receiver, name, _line in calls
    ), "HTTP handlers must submit desired state, never execute runtime releases"


def test_legacy_deploy_and_rollback_commands_are_explicitly_fail_closed() -> None:
    models_source = (ROOT / "src/argus/dashboard/routes/models.py").read_text(
        encoding="utf-8"
    )
    baseline_source = (ROOT / "src/argus/dashboard/routes/baseline.py").read_text(
        encoding="utf-8"
    )
    config_source = (ROOT / "src/argus/dashboard/routes/config.py").read_text(
        encoding="utf-8"
    )
    assert "activate_model_version" not in models_source
    assert "rollback_camera_model" not in models_source
    assert "activate_model_version" not in baseline_source
    assert ".reload_model(" not in config_source
    assert "直接激活入口已关闭，未修改 registry 或运行时" in models_source
    assert "回滚入口已安全关闭，未修改 registry 或运行时" in models_source
    assert "路径部署入口已关闭，未修改 registry 或运行时" in baseline_source
    assert "按路径直接热加载模型的入口已关闭" in config_source


def test_scheduler_never_activates_or_promotes_registered_candidate() -> None:
    calls = _attribute_calls("src/argus/core/scheduler.py")
    violations = [
        (name, line)
        for receiver, name, line in calls
        if receiver == "model_registry" and name in {"activate", "promote", "rollback"}
    ]
    assert violations == [], (
        "scheduled retraining may register a candidate but may not publish it: "
        f"{violations}"
    )


def test_training_packaging_cleanup_uses_release_service_only() -> None:
    path = "src/argus/anomaly/job_executor.py"
    calls = _attribute_calls(path)
    forbidden = {
        "activate",
        "promote",
        "rollback",
        "retire",
        "reload_model",
        "apply_model_release_state",
    }
    violations = [(name, line) for _, name, line in calls if name in forbidden]
    assert violations == [], (
        "training failure cleanup may not write registry stage or runtime state "
        f"directly; use ModelReleaseService: {violations}"
    )

    service_calls = [
        line
        for _receiver, name, line in calls
        if name == "request_transition"
    ]
    assert len(service_calls) == 1, (
        "packaging failure must retire its inactive candidate through exactly "
        "one ModelReleaseService.request_transition call"
    )
    source = (ROOT / path).read_text(encoding="utf-8")
    assert "on_model_trained" not in source
    assert "_safe_hot_reload" not in source


def test_candidate_retirement_has_an_internal_no_runtime_transaction() -> None:
    source = (
        ROOT / "src/argus/storage/model_release_service.py"
    ).read_text(encoding="utf-8")
    assert "inactive_candidate_retirement" in source
    assert "plan.from_stage == ModelStage.CANDIDATE.value" in source
    assert "plan.target_stage == ModelStage.RETIRED.value" in source


def test_model_registry_stage_mutators_are_fail_closed_stubs() -> None:
    tree = _tree("src/argus/storage/model_registry.py")
    model_registry = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "ModelRegistry"
    )
    names = {
        "promote",
        "activate",
        "retire",
        "rollback",
        "compensate_failed_production_sync",
    }
    methods = {
        node.name: node
        for node in model_registry.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    }
    assert set(methods) == names

    for name, method in methods.items():
        raises = [node for node in ast.walk(method) if isinstance(node, ast.Raise)]
        assert len(raises) == 1, f"ModelRegistry.{name} must only fail closed"
        error = raises[0].exc
        assert isinstance(error, ast.Call)
        assert isinstance(error.func, ast.Name)
        assert error.func.id == "ModelStageMutationDisabledError"
        forbidden_writes = [
            node
            for node in ast.walk(method)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        ]
        assert forbidden_writes == [], (
            f"ModelRegistry.{name} contains a state write outside "
            "ModelReleaseService/ReleasePipeline"
        )


def test_candidate_registration_is_atomically_audited() -> None:
    source = (
        ROOT / "src/argus/storage/model_registry.py"
    ).read_text(encoding="utf-8")
    assert 'action="model_candidate_registered"' in source
    assert "ModelVersionEvent(" in source
