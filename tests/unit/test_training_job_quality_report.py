"""Training quality-report persistence with fake trainers/registries only."""
from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from argus.anomaly import job_executor as execution
from argus.anomaly.trainer import TrainingResult, TrainingStatus


def quality_report(**overrides):
    return {
        "passed": True, "integrity_passed": True,
        "policy_version": "argus.baseline-quality.scene-static.v1",
        "scope": "observed_fixed_scene_only", "coverage_claim": "observed_conditions_only",
        "production_readiness": "not_evaluated", "corruption_rate": 0.0,
        "near_duplicate_rate": 0.99, "brightness_std": 0.2,
        "sample_count": 150, "readable_count": 150,
        "warnings": ["当前数据仅覆盖固定机位、已观察到的正常工况"], "errors": [],
        **overrides,
    }


@pytest.fixture
def fake_execution(tmp_path, monkeypatch):
    artifact = tmp_path / "exports" / "model.pt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"protocol-only-not-a-model")
    trainer, database, registry, release = Mock(), Mock(), Mock(), Mock()
    database.update_training_job.return_value = True
    registry.register.return_value = "candidate-01"
    executor = execution.TrainingJobExecutor(
        database=database, trainer=trainer, model_registry=registry,
        baselines_dir=tmp_path / "baselines", model_packages_dir=tmp_path / "packages",
        model_release_service=release,
    )
    monkeypatch.setattr(execution, "get_error_channel", lambda: Mock())
    job = SimpleNamespace(
        job_id="job-01", camera_id="camera-01", zone_id="default", job_type="anomaly_head",
        model_type="patchcore", dataset_selection=None, dataset_version="v001", hyperparameters=None,
        to_dict=lambda: {"job_id": "job-01", "job_type": "anomaly_head"},
    )
    result = TrainingResult(
        status=TrainingStatus.COMPLETE, model_path=str(artifact), image_count=150, train_count=100,
        val_count=50, pre_validation=quality_report(), actual_export_format="torch",
        actual_quantization="fp32", duration_seconds=1.0,
    )
    trainer.train.return_value = result
    return SimpleNamespace(executor=executor, trainer=trainer, database=database, registry=registry,
        release=release, job=job, result=result)


def final_metrics(value):
    return json.loads(value.database.update_training_job.call_args.kwargs["metrics"])


def test_completed_job_and_fallback_candidate_keep_static_quality_diagnostics(fake_execution):
    value = fake_execution
    value.executor._execute_job(value.job)

    report = final_metrics(value)["baseline_quality"]
    assert report["passed"] is True and report["report_status"] == "available"
    assert report["scope"] == "observed_fixed_scene_only"
    assert report["warnings"] == value.result.pre_validation["warnings"]
    assert report["production_readiness"] == "not_evaluated"
    assert value.registry.register.call_args.kwargs["training_params"]["baseline_quality"] == report
    value.registry.activate.assert_not_called()
    value.release.request_transition.assert_not_called()


def test_missing_quality_report_is_unknown_even_when_training_completed(fake_execution):
    value = fake_execution
    value.result.pre_validation = None
    value.executor._execute_job(value.job)

    report = final_metrics(value)["baseline_quality"]
    assert report["passed"] is None and report["report_status"] == "not_evaluated"
    assert report["production_readiness"] == "not_evaluated"


def test_failed_trainer_result_keeps_quality_errors_and_warnings(fake_execution):
    value = fake_execution
    value.result.status = TrainingStatus.FAILED
    value.result.error = "baseline integrity rejected"
    value.result.pre_validation = quality_report(passed=False, integrity_passed=False, errors=["sample identity changed"])
    value.executor._execute_job(value.job)

    assert value.database.update_training_job.call_args.kwargs["status"] == "failed"
    report = final_metrics(value)["baseline_quality"]
    assert report["passed"] is False
    assert report["errors"] == ["sample identity changed"]
    assert report["warnings"] == value.result.pre_validation["warnings"]
    value.registry.register.assert_not_called()


def test_post_training_registry_failure_keeps_the_exact_input_report(fake_execution):
    value = fake_execution
    value.registry.register.side_effect = RuntimeError("injected registry failure")
    value.executor._execute_job(value.job)

    assert value.database.update_training_job.call_args.kwargs["error"] == "injected registry failure"
    assert final_metrics(value)["baseline_quality"]["warnings"] == value.result.pre_validation["warnings"]
    assert final_metrics(value)["baseline_quality"]["passed"] is True


def test_training_mode_restore_failure_keeps_already_produced_quality_report(fake_execution, monkeypatch):
    value = fake_execution

    @contextmanager
    def restore_failure(**kwargs):
        yield
        raise RuntimeError("injected mode restore failure")

    monkeypatch.setattr(value.executor, "_training_mode_guard", restore_failure)
    value.executor._execute_job(value.job)

    assert value.database.update_training_job.call_args.kwargs["error"] == "injected mode restore failure"
    assert final_metrics(value)["baseline_quality"]["warnings"] == value.result.pre_validation["warnings"]
    value.registry.register.assert_not_called()


def test_previous_jobs_quality_report_cannot_leak_into_preflight_failure(fake_execution):
    value = fake_execution
    value.executor._execute_job(value.job)
    value.database.update_training_job.reset_mock()
    value.job.job_id = "job-02"
    value.job.camera_id = None  # Fail before trainer returns any report.
    value.executor._execute_job(value.job)

    report = final_metrics(value)["baseline_quality"]
    assert report["passed"] is None and report["report_status"] == "not_evaluated"
    assert report["warnings"] == []


def test_late_failure_quality_write_remains_conditional(fake_execution):
    value = fake_execution
    value.result.status = TrainingStatus.FAILED
    value.database.update_training_job.side_effect = [True, False]
    value.executor._execute_job(value.job)

    assert value.database.update_training_job.call_count == 2
    assert value.database.update_training_job.call_args.kwargs["expected_status"] == "running"
    value.registry.register.assert_not_called()


def test_packaging_failure_preserves_compensation_receipts_and_input_quality(fake_execution):
    value = fake_execution
    value.executor._packager = Mock()
    value.executor._packager.package.side_effect = RuntimeError("packaging rejected")
    operation = SimpleNamespace(operation_id="retirement-01", status="applied", error=None,
        to_dict=lambda: {"runtime_state": "unchanged"})
    retired = SimpleNamespace(stage="retired", is_active=False)
    value.release.request_transition.return_value = operation, retired
    value.executor._execute_job(value.job)

    metrics = final_metrics(value)
    assert metrics["packaging_error"] == "packaging rejected"
    assert metrics["candidate_retirement"]["operation_id"] == "retirement-01"
    assert metrics["candidate_retirement_error"] is None
    assert metrics["baseline_quality"]["scope"] == "observed_fixed_scene_only"
    assert value.release.request_transition.call_args.kwargs["target_stage"] == "retired"


@pytest.mark.parametrize("value", [
    "passed", {}, {"passed": "true"}, {"passed": True, "brightness_std": float("nan")},
    {"passed": True, "warnings": [object()]}, {"passed": True, "warnings": "not-a-list"},
    {"passed": True, "errors": ["integrity failed"]}, {"passed": True, "integrity_passed": False},
    {"passed": True, "scope": "observed_fixed_scene_only"},
])
def test_malformed_report_never_becomes_a_passing_result(value):
    snapshot = execution._baseline_quality_evidence(value)
    assert snapshot["passed"] is None
    assert snapshot["report_status"] == "invalid"


def test_report_snapshot_is_detached_and_cannot_grant_production_readiness():
    original = quality_report(production_readiness="approved")
    snapshot = execution._baseline_quality_evidence(original)
    original["warnings"].append("late mutation")
    assert "late mutation" not in snapshot["warnings"]
    assert snapshot["production_readiness"] == "not_evaluated"
